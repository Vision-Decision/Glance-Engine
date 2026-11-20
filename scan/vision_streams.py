"""Measure graph-replayed, concurrent per-image SDPA against the HF singleton reduction."""
import argparse
import json
from pathlib import Path

import torch
from speed.engine import image_attention

from scan.profile_vision import elapsed


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    result = {}
    torch.manual_seed(123)
    for length, ns in ((468, (8, 32, 128)), (1008, (8, 32, 64))):
        for n in ns:
            x = torch.randn((n, length, 3, 16, 64), device='cuda', dtype=torch.bfloat16)
            q, k, v = x.unbind(2)
            q, k = q.contiguous(), k.contiguous()
            ref = image_attention(q, k, v, [torch.cuda.current_stream()])
            row = {}
            for workers in (1, 2, 4, 8, 16):
                pool = [torch.cuda.Stream() for _ in range(workers)] if workers > 1 else []
                def run():
                    return image_attention(q, k, v, pool or [torch.cuda.current_stream()])
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                row[str(workers)] = dict(ms=elapsed(graph.replay, reps=15), equal=actual.equal(ref))
                assert row[str(workers)]['equal']
                del graph, actual
            result[f'L{length}_N{n}'] = row
            Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
