"""Profile a warmed cached image query, isolated from the latency sweeps."""
import argparse
from collections import Counter
import json

import torch
from torch.profiler import profile, ProfilerActivity
from scan.bench_scan import Bench
from scan.data import load_images
from speed.profile_glance import cat


def main(args):
    eng = Bench(args).to_engine()
    images = load_images(args.prep, args.image_root, 256, 640)
    bank = eng.encode([r['img'] for r in images], chunk=16)
    result = {}
    for n in (64, 256):
        sub = bank.head(n)
        for q in [dict(type='box', text='a red car'), dict(type='noul', instr='Is there a red car in the image?')]:
            for _ in range(3):
                eng.query(sub, q)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                eng.query(sub, q)
            categories, kernels = Counter(), Counter()
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    ms = e.time_range.elapsed_us() / 1000
                    categories[cat(e.name)] += ms
                    kernels[e.name] += ms
            key = f'{n}_{q["type"]}'
            result[key] = dict(categories_ms=dict(categories.most_common()), top_kernel_ms=kernels.most_common(20))
            print(key, result[key]['categories_ms'], flush=True)
        eng.free_graphs(sub)
    with open(args.out, 'w') as file:
        json.dump(result, file, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--prep', default='data/prep/eval_refcoco_val.jsonl')
    p.add_argument('--image_root', default='/content/data/coco/train2014')
    main(p.parse_args())
