#!/bin/bash
cd /d/Abandoned_Objects_Detection
SP="/c/Users/admin/AppData/Local/Temp/claude/d--Abandoned-Objects-Detection/c2babd2d-1edf-4283-aadb-47377bf2c2cb/scratchpad"
run(){ tag=$1; clip=$2
  py -3.13 demov2/run_rtsbs_aod.py --video ABODA/video11.avi --bg-learn-seconds 20 --max-frames 2250 \
    --async-semantic on --clip-verify $clip --outdir "$SP/umb_$tag" 2>&1 | grep -E "demo2\] done" | sed "s/^/[$tag clip=$clip] /"
}
run off1 0
run on1  1
run off2 0
run on2  1
echo ">>> UMBRELLA STATUS (GT 309,271 tol45):"
py -3.13 "$SP/score.py" video11 "$SP/umb_off1" "$SP/umb_on1" "$SP/umb_off2" "$SP/umb_on2"
echo "UMB-BATCH DONE"
