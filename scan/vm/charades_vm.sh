#!/usr/bin/env bash
# On the VM (CPU only): Charades 480p + annotations from AI2, then 2-fps test frames (clip pilot) and train clips
# (short-readout distillation). Outputs: /content/charades_pilot2, /content/distill.
set -uo pipefail
R=/content/charades; mkdir -p $R && cd $R
[ -f Charades_v1_480.zip ] || curl -sL -o Charades_v1_480.zip https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
[ -d Charades ] || (curl -sL -o Charades.zip https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip && unzip -oq Charades.zip)
for s in train test; do
  [ -f charades_sta_$s.txt ] || curl -sL -o charades_sta_$s.txt https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/Charades-STA/charades_sta_$s.txt
done
ls -la $R; which ffmpeg
cd /content/dv
python scan/distill_prep.py --root $R --out /content/distill --workers 40 > /content/out/scan/distill_prep.log 2>&1
python scan/charades_prep.py --root $R --out /content/charades_pilot2 --fps 2 --workers 40 > /content/out/scan/prep2.log 2>&1
echo CHARADES_VM_DONE
