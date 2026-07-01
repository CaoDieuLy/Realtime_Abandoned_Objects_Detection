#!/bin/bash
cd /d/Abandoned_Objects_Detection
SP="/c/Users/admin/AppData/Local/Temp/claude/d--Abandoned-Objects-Detection/c2babd2d-1edf-4283-aadb-47377bf2c2cb/scratchpad"
bench(){ mode=$1; tag=$2
  py -3.13 demov2/run_rtsbs_aod.py --video ABODA/video11.avi --bg-learn-seconds 8 --max-frames 800 \
    --async-semantic $mode --outdir "$SP/fps_$tag" 2>&1 | grep -E "demo2\] done" | sed "s/^/[$tag $mode] /"
}
# interleave to average out any drift; machine verified idle
bench off sync1
bench on  async1
bench off sync2
bench on  async2
echo "FPS-BENCH DONE"
