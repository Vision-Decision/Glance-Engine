"""Achievable G4 rates. Run alone on the GPU with python -m speed.roofline --out FILE."""
import argparse
import json
import statistics
import time

import torch


def measure(fn, iterations, *, host=False):
    for i in range(iterations):
        fn(i)
    torch.cuda.synchronize()
    samples = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(9):
        start.record()
        tick = time.perf_counter()
        for i in range(iterations):
            fn(i)
        end.record()
        end.synchronize()
        ms = (time.perf_counter() - tick) * 1000 if host else start.elapsed_time(end)
        samples.append(ms / iterations)
    return dict(ms=statistics.median(samples), samples_ms=samples)


def main(out):
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, gemm=[])
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float32):
        for m, k, n in [(x, 2560, 9216) for x in (1, 16, 32, 64, 256, 1024, 4096)] + [(8192, 8192, 8192)]:
            if dtype == torch.float8_e4m3fn and m < 16:
                continue
            if dtype == torch.float32 and (m, k) not in ((1024, 2560), (8192, 8192)):
                continue
            a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16).to(dtype)
            # Rotate >256 MiB of weights so skinny GEMMs stream from device memory.
            count = max(2, (256 << 20) // (k*n*a.element_size()) + 1)
            weights = [torch.randn(n, k, dtype=torch.bfloat16, device='cuda').to(dtype).T for _ in range(count)]
            scale = torch.ones((), device='cuda')
            if dtype == torch.float8_e4m3fn:
                def gemm(i):
                    return torch._scaled_mm(a, weights[i % count], scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            else:
                def gemm(i):
                    return a @ weights[i % count]
            timing = measure(gemm, max(count, 6))
            result['gemm'].append(dict(dtype=str(dtype), m=m, n=n, k=k,
                                       tflops=2*m*n*k/timing['ms']/1e9, **timing))
            del a, weights
    size = 256 << 20
    src = torch.randint(256, (size,), device='cuda', dtype=torch.uint8)
    dst = torch.empty_like(src)
    timing = measure(lambda i: dst.copy_(src), 16)
    result['copy'] = dict(bytes_read_plus_written=2*size, GBs=2*size/timing['ms']/1e6, **timing)
    for pinned in (False, True):
        host = torch.empty(size, dtype=torch.uint8, pin_memory=pinned).fill_(7)
        timing = measure(lambda i: dst.copy_(host, non_blocking=True), 6, host=True)
        result['pinned' if pinned else 'pageable'] = dict(bytes=size, GBs=size/timing['ms']/1e6, **timing)
    scalar = torch.zeros(1, device='cuda')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scalar.add_(1)
    for label, op in [('launch', lambda: scalar.add_(1)), ('replay', graph.replay)]:
        result[label] = measure(lambda i: op(), 2000, host=True)
        result[label+'_sync'] = measure(lambda i: op(), 1, host=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-4B')
    result['text_config'] = cfg.text_config.to_dict()
    result['vision_config'] = cfg.vision_config.to_dict()
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print('copy GB/s', result['copy']['GBs'], 'bf16 TFLOP/s',
          max(r['tflops'] for r in result['gemm'] if r['dtype'] == 'torch.bfloat16'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    main(parser.parse_args().out)
