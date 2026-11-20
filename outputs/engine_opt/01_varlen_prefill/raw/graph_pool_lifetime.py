"""Capture again with a released private-pool handle. Pass fresh to replace it before recapture."""
import gc
import sys
import torch

x = torch.ones(128, device="cuda")
pool = torch.cuda.graph_pool_handle()
def capture(handle):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=handle):
        y = x + 1
    g.replay()
    torch.cuda.synchronize()
    assert y[0].item() == 2
    return g, y
g, y = capture(pool)
del g
if "hold" not in sys.argv:
    del y
gc.collect()
if "purge" in sys.argv:
    torch.cuda.empty_cache()
capture(torch.cuda.graph_pool_handle() if sys.argv[1] == "fresh" else pool)
print("capture passed", sys.argv[1], torch.__version__)
