#!/usr/bin/env bash
# After chain6: image-presence parity (AUROC, ECE, FA/miss, box Acc@0.5) of v1 vs the short-readout model, 150 images.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "chain6.s[h]" >/dev/null; do sleep 20; done
for M in v1:/content/ckpt/v1/ckpt_final.pt short:/content/out/distill_v1/ckpt_final.pt; do
  T=${M%%:*}; C=${M#*:}
  echo "[chain7] parity_$T start $(date +%H:%M:%S)"
  timeout 3600 python -m scan.bench_scan --ckpt $C --out $O/parity150_$T.json --parts acc --Tvs 256 --n_acc 150 > $O/parity150_$T.log 2>&1
done
echo "[chain7] ALL DONE $(date +%H:%M:%S)"
