#!/bin/bash
cd /d/Abandoned_Objects_Detection
SP="/c/Users/admin/AppData/Local/Temp/claude/d--Abandoned-Objects-Detection/c2babd2d-1edf-4283-aadb-47377bf2c2cb/scratchpad"
run(){ # name video extra...
  name=$1; vid=$2; shift 2
  out="$SP/p2_$name"
  echo ">>>>> $name  ($(date +%H:%M:%S))  ABODA/$vid $@"
  py -3.13 demov2/run_rtsbs_aod.py --video ABODA/$vid --bg-learn-seconds 20 --outdir "$out" "$@" \
    2>&1 | grep -E "demo2\] done" | tail -1
}
run v11_off  video11.avi --async-semantic off --max-frames 3000
run v11_on   video11.avi --async-semantic on  --max-frames 3000
run v1_on    video1.avi  --async-semantic on  --max-frames 1950
echo ">>>>> SCORING ($(date +%H:%M:%S))"
py -3.13 "$SP/score.py" video11 "$SP/p2_v11_off" "$SP/p2_v11_on"
py -3.13 "$SP/score.py" video1  "$SP/p2_v1_on"
echo ">>>>> PHASE2 DONE ($(date +%H:%M:%S))"
