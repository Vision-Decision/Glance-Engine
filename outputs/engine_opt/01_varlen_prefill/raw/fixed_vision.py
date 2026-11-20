"""Compare deployed BF16 LM layouts on identical COCO vision embeddings. Run from the repo with PYTHONPATH=."""
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from scan.bench_scan import Bench, diff
from scan.data import load_images
from scan.varlen import FreshPack, packed_forward


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    bench = Bench(args)
    eng = bench.to_engine()
    results = {}
    qs = [dict(type="box", text="person standing"), dict(type="box", text="the dog on the left"),
          dict(type="noul", instr="Is there a person in the image?")]
    for tv in (640, 256):
        images = [row["img"] for row in load_images("data/prep/eval_refcoco_val.jsonl",
                  "/content/data/coco/train2014", 24, tv, frame=(tv == 256))]
        for i, q in enumerate(qs):
            batch, tensors, host = eng.build(images, [[q]] * len(images))
            feats = eng.feats(batch)
            pack = FreshPack(batch, eng.dev)
            packed = packed_forward(eng, pack, feats)
            keys = ("presence_logit", "box_pred", "iou_logit") if q["type"] == "box" else ("typed_logits",)
            def cpu_rows(out, n):
                return {k: out[k][:n].float().cpu() for k in keys}
            packed = cpu_rows(packed, len(images))
            inp = eng.to_device(tensors, host, feats)
            legacy = cpu_rows(eng.run(inp, host)[0], len(images))
            single = []
            lengths = batch["p_img_mask"].sum(1).tolist()
            for im, part in zip(images, feats.split(lengths)):
                _, tensors, host = eng.build([im], [[q]])
                inp = eng.to_device(tensors, host, part)
                single.append(cpu_rows(eng._core(inp, host)[0], 1))
            single = {k: torch.cat([v[k] for v in single]) for k in keys}
            separate = cpu_rows(eng.separate(images, q), len(images))
            result = dict(packed_vs_old=diff(packed, legacy), packed_vs_single_lm=diff(packed, single),
                          old_vs_single_lm=diff(legacy, single), packed_vs_separate_e2e=diff(packed, separate))
            results[f"Tv{tv}_q{i}"] = result
            args.out.write_text(json.dumps(results, indent=1))
            print(tv, q, result, flush=True)
        eng.free_graphs()


if __name__ == "__main__":
    main()
