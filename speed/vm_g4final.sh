#!/usr/bin/env bash
# VM side, G4: final engine-mode pass (all regimes), light ablation modes, then the SGLang configurations.
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TO=7200 bash speed/run_dev.sh glance2_g4 --n_calls 50 --modes eng_graph,eng_fused,eng_fp8 --parts lat,mixed,followup,tput,video,gui
TO=3600 bash speed/run_dev.sh glance3_g4 --n_calls 50 --modes ref_merged,eng_eager --parts lat,mixed
echo "[glance2] ALL DONE" >> /content/out/spd/main_g4.log
SERVE_SET=sglang,sglang_mtp bash speed/vm_serve.sh g4
echo "[g4final] ALL DONE" >> /content/out/spd/main_g4.log
