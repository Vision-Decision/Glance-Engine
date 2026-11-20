"""Independent PyTorch box oracle and per-stage controls for the five-layout calibration."""

import math

import torch
import torch.nn.functional as F


def frozen_replay(head, inputs, mode):
    """Hold real-weight hidden states fixed; remove batch peers and actual patch padding."""
    from speed.box_readout import box_forward

    hidden, visual, valid, centers, images = inputs[:5]
    layout = box_forward(head, hidden, visual, valid, centers, images, mode)
    rows = []
    for row, image in enumerate(images.tolist()):
        mask = valid[image]
        single = box_forward(
            head,
            hidden[row : row + 1],
            visual[image, mask][None],
            valid[image, mask][None],
            centers[image, mask][None],
            torch.zeros_like(images[:1]),
            mode,
        )
        errors = {}
        for key in ("center_logits", "pred_idx", "presence_logit", "box_pred", "iou_logit"):
            actual = layout[key][row]
            if key == "center_logits":
                actual = actual[mask]
            errors[key] = float((actual - single[key][0]).abs().max())
        rows.append(errors)
    return rows


def reference_hook(geometry_rows, traces=None, replay_mode=None, calibration_head="oracle"):
    def hook(head, inputs, output):
        hidden, vis, valid, centers, image = inputs[:5]
        hq = head.q_norm(hidden.float())
        hv = head.v_norm(vis.float())
        q, k = head.c_q(hq), head.c_k(hv)
        scale = math.sqrt(q.shape[-1])
        scores = ((q[:, None].double() * k[image].double()).sum(-1) / scale).masked_fill(~valid[image], -1e4)
        index = scores.argmax(-1)
        pc, v = centers[image, index], hv[image, index]
        edges = head.edges_logits(hq, v, pc)
        for row, im in enumerate(image.tolist()):
            geometry_rows.append((head, hq[row : row + 1], hv[im], centers[im], calibration_head))
        if traces is not None:

            def project64(x, norm, projection):
                x = F.layer_norm(
                    x.double(), norm.normalized_shape, norm.weight.double(), norm.bias.double(), norm.eps
                )
                return F.linear(x, projection.weight.double(), projection.bias.double())

            q64 = project64(hidden, head.q_norm, head.c_q)
            k64 = project64(vis, head.v_norm, head.c_k)
            oracle = (q64[:, None] * k64[image]).sum(-1) / scale
            replay = frozen_replay(head, inputs, replay_mode) if replay_mode else None
            for row, im in enumerate(image.tolist()):
                mask = valid[im]
                keep = lambda x: x.detach().cpu()
                traces.append(
                    {
                        "hidden": keep(hidden[row]),
                        "vis": keep(vis[im, mask]),
                        "q": keep(q[row]),
                        "k": keep(k[im, mask]),
                        "hq": keep(hq[row]),
                        "hv": keep(hv[im, mask]),
                        "centers": keep(centers[im, mask]),
                        "scores": keep(scores[row, mask]),
                        "legacy": keep(output["center_logits"][row, mask]),
                        "oracle": keep(oracle[row, mask]),
                        "edges": keep(edges[row]),
                        "replay": replay[row] if replay else None,
                    }
                )
        if calibration_head == "blocks":
            from speed.box_readout import box_forward, mlp

            actual = box_forward(head, hidden, vis, valid, centers, image, calibration_head)
            selected = actual["pred_idx"]
            features = head._edge_feats(hq, hv[image, selected], centers[image, selected])
            return dict(
                output, **actual, edge_logits_pred=mlp(head.edge, features, calibration_head).view(-1, 4, head.K)
            )
        return dict(
            output,
            center_logits=scores,
            pred_idx=index,
            edge_logits_pred=edges,
            box_pred=head.expected_box(edges, pc),
            iou_logit=head.iou(head._edge_feats(hq, v, pc)).squeeze(-1),
        )

    return hook


def summarize_rows(rows):
    values = []
    for t in rows:
        s = t["scores"]
        best = s.topk(2).values
        values.append(
            {
                "margin": float(best[0] - best[1]),
                "center": int(s.argmax()),
                "legacy_center": int(t["legacy"].argmax()),
                "fp64_projection_center": int(t["oracle"].argmax()),
                "legacy_score_error": float((s - t["legacy"]).abs().max()),
                "projection_score_error": float((s - t["oracle"]).abs().max()),
                "hidden_dtype": str(t["hidden"].dtype),
                "projection_dtype": str(t["q"].dtype),
            }
        )
        if t["replay"] is not None:
            values[-1]["frozen_replay_max_abs"] = t["replay"]
    return values


@torch.no_grad()
def trace_pair(head, first, second):
    """Swap one operand at a time; condition both edge regressions on the first layout's winning patch."""
    dev = head.c_q.weight.device
    maxima, counts = {}, {}
    for a, b in zip(first, second, strict=True):

        def record(name, value):
            maxima[name] = max(maxima.get(name, 0.0), float(value))

        for field in ("hidden", "vis", "q", "k", "scores", "edges"):
            record(field, (a[field].float() - b[field].float()).abs().max())
        sa, sb = a["scores"], b["scores"]
        ia, ib = int(sa.argmax()), int(sb.argmax())
        # Identical projected operands, re-evaluated as a singleton independent of original padding.
        scale = math.sqrt(len(a["q"]))
        for name, query, keys in (
            ("q_only", b["q"], a["k"]),
            ("k_only", a["q"], b["k"]),
            ("singleton_replay", b["q"], b["k"]),
        ):
            s = (query.double() * keys.double()).sum(-1) / scale
            record(name + "_score", (s - (sb if name == "singleton_replay" else sa)).abs().max())
            counts[name + "_flips"] = counts.get(name + "_flips", 0) + (
                int(s.argmax()) != (ib if name == "singleton_replay" else ia)
            )
        for name, x, y in (
            ("fixed", sa, sb),
            ("legacy", a["legacy"], b["legacy"]),
            ("fp64_projection", a["oracle"], b["oracle"]),
        ):
            counts[name + "_flips"] = counts.get(name + "_flips", 0) + int(x.argmax() != y.argmax())
        # Control for treating distinct logits inside the proposed 0.06 band as a tie.
        band = lambda s: int((s >= s.max() - 0.06).int().argmax())
        counts["band_flips"] = counts.get("band_flips", 0) + (band(sa) != band(sb))
        counts["band_changed_from_argmax"] = counts.get("band_changed_from_argmax", 0) + (band(sa) != ia)
        pc = a["centers"][ia : ia + 1].to(dev)
        edge = head.edges_logits(b["hq"][None].to(dev), b["hv"][ia : ia + 1].to(dev), pc)
        original = a["edges"][None].to(dev)
        record("fixed_patch_edges", (edge - original).abs().max())
        record(
            "fixed_patch_expected_box", (head.expected_box(edge, pc) - head.expected_box(original, pc)).abs().max()
        )
        # Same edge logits and patch, double-precision softmax/expectation control.
        record(
            "expectation_roundoff",
            (head.expected_box(original.double(), pc.double()) - head.expected_box(original, pc)).abs().max(),
        )
    return {"count": len(first), "counts": counts, "max": maxima}


@torch.no_grad()
def reference_geometry(reference, indices):
    """Use the original reference outputs at matching centers, evaluate only the alternatives."""
    boxes, logits = reference["box_pred"].clone(), reference["iou_logit"].clone()
    for row in (indices != reference["center_idx"]).nonzero().flatten().tolist():
        head, query, visual, centers, mode = reference["geometry"][row]
        i = int(indices[row])
        v, pc = visual[i : i + 1], centers[i : i + 1]
        if mode == "blocks":
            from speed.box_readout import geometry

            box, iou = geometry(head, query, v, pc, mode)
            boxes[row], logits[row] = box[0].cpu(), iou[0].cpu()
        else:
            boxes[row] = head.expected_box(head.edges_logits(query, v, pc), pc)[0].cpu()
            logits[row] = head.iou(head._edge_feats(query, v, pc))[0, 0].cpu()
    return boxes, logits.sigmoid()
