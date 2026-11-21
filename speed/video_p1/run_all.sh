#!/usr/bin/env bash
# Full P1 sweep on one G4. cd /content/p1 && nohup bash speed/video_p1/run_all.sh > /content/p1_run.log 2>&1 &
set -uo pipefail
cd /content/p1
export PYTHONPATH=/content/p1 TOKENIZERS_PARALLELISM=false
R=outputs/video_p1/raw; mkdir -p $R
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.limit,temperature.gpu --format=csv > $R/gpu.csv
for M in 0.8B 4B; do
  python -m speed.video_p1.bench --model Qwen/Qwen3.5-$M --tests check,m2,m3,profile,m4 --out $R/m1234_$M.json
done
python -m speed.video_p1.bench --model Qwen/Qwen3.5-4B --tests m9 --m9 A,A8,B --out $R/m9_4B.json
python -m speed.video_p1.bench --model Qwen/Qwen3.5-0.8B --tests m9 --m9 A,B --out $R/m9_0.8B.json
python -m speed.video_p1.bench --model Qwen/Qwen3.5-2B --tests m3,m9 --no-fp8 --m9 A --out $R/m39_2B.json
echo RUN_DONE
