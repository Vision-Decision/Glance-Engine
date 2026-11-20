"""Hardware roofline numbers for the AR lower bound: achievable memory bandwidth and GEMM throughput.

python -m speed.hwbench --out OUT.json
  bw_gbs        device-to-device copy bandwidth (read+write counted), 2 GiB tensor
  gemm_tflops   bf16 / fp8 (rowwise) GEMM at a 4B-LLM projection shape (tokens x 2560 -> 9216)
  weight_read_ms  time to stream the Qwen3.5-4B language-model weights once (bf16 bytes / bw): the floor of one
                  decode step for any AR decoder (batch of rows that all read the weights once)
"""
import argparse, json
import torch


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm_params", type=float, default=4.2e9)
    a = ap.parse_args()
    x = torch.empty(2 ** 30, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    ms = timeit(lambda: y.copy_(x))
    bw = 2 * x.numel() * 2 / (ms / 1e3) / 1e9
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, bw_gbs=bw)
    del x, y
    for T in (32, 256, 1024, 4096):
        A = torch.randn(T, 2560, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(9216, 2560, device="cuda", dtype=torch.bfloat16)
        ms = timeit(lambda: A @ B.t())
        res[f"gemm_bf16_T{T}_tflops"] = 2 * T * 2560 * 9216 / (ms / 1e3) / 1e12
        try:
            A8, B8 = A.to(torch.float8_e4m3fn), B.to(torch.float8_e4m3fn)
            sa, sb = torch.ones(T, 1, device="cuda"), torch.ones(1, 9216, device="cuda")
            ms = timeit(lambda: torch._scaled_mm(A8, B8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            res[f"gemm_fp8_T{T}_tflops"] = 2 * T * 2560 * 9216 / (ms / 1e3) / 1e12
        except Exception as e:
            res["fp8_error"] = str(e)[:200]
    res["weight_read_ms_bf16"] = a.lm_params * 2 / (bw * 1e9) * 1e3   # one read of all weights at the copy bandwidth
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
