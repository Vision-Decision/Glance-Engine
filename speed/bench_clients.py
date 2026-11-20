"""A burst of independent in-process clients targeting one serial Glance worker."""
import argparse
import json
import time

import numpy as np

from speed import workload as W
from speed.bench_glance import Runner


def main(args):
    runner = Runner(args)
    runner.set_mode('eng_fused')
    images = W.load_coco(args.prep, args.image_root, 100)
    result = {}
    for clients in (1, 4, 16, 64):
        calls = []
        for c in range(args.n_calls):
            indices = [(c*clients+j) % len(images) for j in range(clients)]
            calls.append([(images[i]['img'], W.questions(images, i, 1)) for i in indices])
        # Capture every encountered shape before timing. All clients arrive together.
        for burst in calls:
            for img, qs in burst:
                runner.call([img], [qs], Nv=640)
        times, finishes, stage_totals = [], [], []
        for burst in calls:
            stages = dict(prep=0., vision=0., lm=0.)
            start = time.perf_counter()
            for img, qs in burst:
                _, timing = runner.call([img], [qs], Nv=640)
                for k in stages:
                    stages[k] += timing[k]
                finishes.append(1000 * (time.perf_counter() - start))
            times.append(finishes[-1])
            stage_totals.append(stages)
        result[str(clients)] = dict(burst_ms=times, total_ms_median=float(np.median(times)),
                                    request_ms_median=float(np.median(finishes)), n_calls=args.n_calls)
        result[str(clients)].update({k+'_ms_median': float(np.median([row[k] for row in stage_totals]))
                                     for k in stages})
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(clients, result[str(clients)]['total_ms_median'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--prep', default='data/prep/eval_refcoco_val.jsonl')
    p.add_argument('--image_root', default='/content/data/coco/train2014')
    p.add_argument('--n_calls', default=12, type=int)
    main(p.parse_args())
