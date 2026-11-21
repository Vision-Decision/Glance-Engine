#!/usr/bin/env bash
# Follow-up after run_all.sh: kernel-time profiles (with fixed attention bucket) and the stream counts
# between the measured 1/4/8/16 points, to locate the largest batch that keeps up with 30 fps.
set -uo pipefail
cd /content/p1
export PYTHONPATH=/content/p1 TOKENIZERS_PARALLELISM=false
R=outputs/video_p1/raw
for M in 0.8B 4B; do
  python -m speed.video_p1.bench --model Qwen/Qwen3.5-$M --tests profile --out $R/profile_$M.json
done
python -m speed.video_p1.bench --model Qwen/Qwen3.5-4B --tests m9 --m9 A,A8 --streams 2,3 --out $R/m9_4B_extra.json
python -m speed.video_p1.bench --model Qwen/Qwen3.5-0.8B --tests m9 --m9 A --streams 6,10,12 --out $R/m9_0.8B_extra.json
echo RUN_DONE
