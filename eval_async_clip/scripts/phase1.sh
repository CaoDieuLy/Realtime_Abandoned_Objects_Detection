#!/bin/bash
cd /d/Abandoned_Objects_Detection
SP="/c/Users/admin/AppData/Local/Temp/claude/d--Abandoned-Objects-Detection/c2babd2d-1edf-4283-aadb-47377bf2c2cb/scratchpad"
run(){ name=$1; vid=$2; shift 2
  out="$SP/p1_$name"
  echo ">>>>> $name ($(date +%H:%M:%S)) ABODA/$vid $@"
  py -3.13 demov2/run_rtsbs_aod.py --video ABODA/$vid --bg-learn-seconds 20 --outdir "$out" "$@" \
    2>&1 | grep -E "demo2\] done|\[CLIP\].*(SUPPRESS|KEEP)" | tail -40
}
# async ON throughout (the #2 win); clip-off baselines for v11/v1 are reused from phase2 (p2_v11_on, p2_v1_on)
run v11_clip video11.avi --async-semantic on --clip-verify 1 --debug-owner 1 --max-frames 3000
run v7_off   video7.avi  --async-semantic on
run v7_clip  video7.avi  --async-semantic on --clip-verify 1 --debug-owner 1
run v1_clip  video1.avi  --async-semantic on --clip-verify 1 --debug-owner 1 --max-frames 1950
echo ">>>>> SCORING ($(date +%H:%M:%S))"
echo "[video11] clip OFF vs ON:"; py -3.13 "$SP/score.py" video11 "$SP/p2_v11_on" "$SP/p1_v11_clip"
echo "[video7]  clip OFF vs ON:"; py -3.13 "$SP/score.py" video7  "$SP/p1_v7_off"  "$SP/p1_v7_clip"
echo "[video1]  clip OFF vs ON:"; py -3.13 "$SP/score.py" video1  "$SP/p2_v1_on"   "$SP/p1_v1_clip"
echo ">>>>> PHASE1 DONE ($(date +%H:%M:%S))"
