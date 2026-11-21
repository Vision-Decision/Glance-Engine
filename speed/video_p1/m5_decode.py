"""M5: 1080p H.264 NVDEC decode + GPU preprocessing per frame (torchcodec CUDA decoder).
  python -m speed.video_p1.m5_decode --video /content/v1080.mp4 --out outputs/video_p1/raw/m5.json
Streaming access: frames are decoded in order, one at a time (no batching across time), as a live stream would be."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from speed.video_p1.bench import stats
from speed.video_p1.stream import patchify


def prep(fr, hw, prev):
    x = F.interpolate(fr.float(), size=hw, mode="bilinear", antialias=True, align_corners=False)
    x = (x / 255.0 - 0.5) / 0.5
    p = x if prev is None else prev
    return patchify(torch.stack([p, x], 1)).to(torch.bfloat16), x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/content/v1080.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    import torchcodec
    from torchcodec.decoders import VideoDecoder
    res = dict(torchcodec=torchcodec.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    backends = [("cuda", dict(device="cuda"))]
    try:
        from torchcodec.decoders import set_cuda_backend  # noqa: F401
        backends.append(("cuda_beta", dict(device="cuda", _beta=True)))
    except ImportError:
        pass
    backends.append(("cpu", dict(device="cpu")))
    for name, kw in backends:
        try:
            if kw.pop("_beta", False):
                from torchcodec.decoders import set_cuda_backend
                with set_cuda_backend("beta"):
                    dec = VideoDecoder(a.video, **kw)
            else:
                dec = VideoDecoder(a.video, **kw)
            md = dec.metadata
            res.setdefault("video", dict(w=md.width, h=md.height, fps=md.average_fps, n=md.num_frames,
                                         codec=md.codec, bitrate=getattr(md, "bit_rate", None)))
            for i in range(10):
                dec[i]
            torch.cuda.synchronize()
            dt, pt = [], {}
            prev = {}
            for i in range(10, 10 + a.frames):
                t0 = time.perf_counter()
                fr = dec[i]
                if fr.device.type != "cuda":
                    fr = fr.cuda(non_blocking=False)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                dt.append((t1 - t0) * 1e3)
                for hw in ((224, 224), (256, 256), (448, 448), (256, 512)):
                    t2 = time.perf_counter()
                    px, prev[hw] = prep(fr[None], hw, prev.get(hw))
                    torch.cuda.synchronize()
                    pt.setdefault(f"{hw[0]}x{hw[1]}", []).append((time.perf_counter() - t2) * 1e3)
            # batched-stream decode: 16 independent decoders, one frame each (16 streams at one tick)
            row = dict(decode=stats(dt), prep={k: stats(v) for k, v in pt.items()}, device=str(fr.device))
            # throughput: iterate frames back to back
            t0 = time.perf_counter()
            for i in range(10 + a.frames, 10 + a.frames + 300):
                dec[i]
            torch.cuda.synchronize()
            row["decode_throughput_fps"] = 300 / (time.perf_counter() - t0)
            res[name] = row
            print(name, json.dumps(row)[:600], flush=True)
        except Exception as e:
            res[name] = dict(error=repr(e)[:400])
            print(name, "ERROR", repr(e)[:400], flush=True)
    # 16 concurrent streams on the GPU decoder (round robin)
    try:
        decs = [VideoDecoder(a.video, device="cuda") for _ in range(16)]
        for d in decs:
            d[0]
        base = 1
        for B in (1, 4, 8, 16):
            ts = []
            for i in range(base, base + 120):
                t0 = time.perf_counter()
                frs = torch.stack([decs[j][i] for j in range(B)])
                px, _ = prep(frs, (256, 256), None)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            base += 120
            res.setdefault("multi_stream_decode_prep_256", {})[B] = stats(ts)
            print("multi", B, stats(ts), flush=True)
    except Exception as e:
        res["multi_stream_error"] = repr(e)[:400]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
