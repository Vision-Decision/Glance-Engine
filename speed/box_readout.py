"""Inference center selection. Keep the trained hard argmax and the original edge and IoU heads."""

import importlib.util
import math

import torch

_HAS_TRITON = importlib.util.find_spec("triton") is not None


def project(layer, inputs, mode, block_rows=16):
    if mode == "blocks":
        # cuBLAS sees the same contiguous M/K/N, transpose and bias on every call.
        # This also fixes the GEMV vs GEMM switch for a single question.
        x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        with torch.autocast(inputs.device.type, enabled=False):
            blocks = []
            for block in x.split(block_rows):
                n = len(block)
                if n < block_rows:
                    block = torch.nn.functional.pad(block, (0, 0, 0, block_rows - n))
                blocks.append(layer(block)[:n])
        out = blocks[0] if len(blocks) == 1 else torch.cat(blocks)
        return out.view(*inputs.shape[:-1], layer.out_features)
    if mode == "tiled" and inputs.is_cuda and _HAS_TRITON:
        from speed.box_scores_kernel import fixed_project

        return fixed_project(layer, inputs)
    return layer(inputs)


def center_scores(query, keys, valid, images=None, slots=None):
    """FP64 dot products of FP32 projections. slots addresses a compact spatial key table."""
    if query.is_cuda and _HAS_TRITON:
        from speed.box_scores_kernel import cuda_scores

        return cuda_scores(query, keys, valid, images, slots)
    if slots is not None:
        keys = keys[slots.long()]
    if images is not None:
        keys, valid = keys[images], valid[images]
    dots = (query.double()[:, None] * keys.double()).sum(-1)
    return (dots / math.sqrt(query.shape[-1])).masked_fill_(~valid, -1e4)


def mlp(layers, inputs, mode):
    if mode not in ("tiled", "blocks"):
        return layers(inputs)
    first, activation, last = layers
    return project(last, activation(project(first, inputs, mode)), mode)


def geometry(head, query, visual, centers, mode):
    features = head._edge_feats(query, visual, centers)
    edges = mlp(head.edge, features, mode).reshape(-1, 4, head.K)
    return head.expected_box(edges, centers), mlp(head.iou, features, mode).squeeze(-1)


def box_forward(head, hidden, visual, valid, centers, images, mode):
    query = head.q_norm(hidden.float())
    visual = head.v_norm(visual.float())
    scores = center_scores(project(head.c_q, query, mode), project(head.c_k, visual, mode, 512), valid, images)
    index = scores.argmax(-1)
    box, iou = geometry(head, query, visual[images, index], centers[images, index], mode)
    return {
        "presence_logit": mlp(head.presence, query, mode).squeeze(-1),
        "box_pred": box,
        "iou_logit": iou,
        "pred_idx": index,
        "center_logits": scores,
    }
