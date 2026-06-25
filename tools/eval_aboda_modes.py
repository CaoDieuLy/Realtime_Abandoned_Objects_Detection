"""Batch-run demov2 over several ABODA videos in multiple --mode presets and score
each run against ABODA ground truth (HIT / FP / detection latency). Handles videos
with more than one abandoned object (e.g. video6).

Usage (from repo root):
  python demov2/tools/eval_aboda_modes.py --videos 1 2 3 7 8 11 \
      --modes no-feedback instance-feedback dense-feedback
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "demov2" / "run_rtsbs_aod.py"
GT_JSON = ROOT / "ABODA" / "aboda_gt.json"


def load_gt() -> dict[str, dict]:
    data = json.loads(GT_JSON.read_text(encoding="utf-8"))
    gt = {}
    for v in data["videos"]:
        objs = []
        for obj in v["objects"]:
            x, y, w, h = obj["bbox"]  # xywh
            objs.append({
                "center": (x + w / 2.0, y + h / 2.0),
                "abandon_frame": obj["abandon_frame"],
                "owner_leave_frame": obj.get("owner_leave_frame", obj["abandon_frame"]),
                "tol": max(45.0, 0.6 * (w * h) ** 0.5),  # localization slack ~ object scale
            })
        gt[v["video_id"]] = {"objects": objs, "fps": v["fps"]}
    return gt


def run_one(video: str, mode: str, every: int, outdir: Path, extra: list[str] | None = None) -> float:
    cmd = [
        sys.executable, os.fspath(RUNNER),
        "--video", os.fspath(ROOT / "ABODA" / f"video{video}.avi"),
        "--mode", mode,
        "--semantic-mode", "online-yoloseg",  # preset overrides to segformer for dense-feedback
        "--semantic-every", str(every),
        "--save-masks-every", "0",
        "--outdir", os.fspath(outdir),
    ]
    if mode == "dense-feedback":
        cmd += ["--segformer-device", "cpu", "--segformer-local-files-only"]
    if extra:
        cmd += extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=os.fspath(ROOT), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    fps = 0.0
    for line in proc.stdout.splitlines():
        if "done:" in line and "FPS" in line:
            try:
                fps = float(line.split("events,")[1].split("FPS")[0].strip())
            except Exception:
                pass
    if not (outdir / "events.json").exists():
        print(f"  [WARN] {video}/{mode}: no events.json\n{proc.stdout[-1000:]}")
    return fps


def score(events: list[dict], g: dict) -> dict:
    fps = g["fps"]
    used: set[int] = set()
    per_obj = []
    for o in g["objects"]:
        cx, cy = o["center"]
        tol = o["tol"]
        owner_leave = o["owner_leave_frame"]
        near = [k for k, e in enumerate(events)
                if ((e["center"][0] - cx) ** 2 + (e["center"][1] - cy) ** 2) ** 0.5 <= tol
                and e["frame"] >= owner_leave]
        used.update(near)
        first = min((events[k]["frame"] for k in near), default=None)
        lat = round((first - o["abandon_frame"]) / max(1e-3, fps), 1) if first is not None else None
        per_obj.append({"hit": len(near) >= 1, "latency_s": lat})
    return {
        "n_objs": len(g["objects"]),
        "n_hit": sum(p["hit"] for p in per_obj),
        "fp": len(events) - len(used),
        "n_events": len(events),
        "latencies": [p["latency_s"] for p in per_obj],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["1", "2", "3", "7", "8", "11"])
    ap.add_argument("--modes", nargs="+", default=["no-feedback", "instance-feedback"])
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--outroot", default=os.fspath(ROOT / "demov2" / "eval"))
    ap.add_argument("--extra", default="", help="extra flags forwarded to run_rtsbs_aod.py, e.g. \"--inactive-tol-s 3.0\"")
    args = ap.parse_args()
    extra = args.extra.split()

    gt = load_gt()
    outroot = Path(args.outroot)
    rows = []
    for v in args.videos:
        for mode in args.modes:
            outdir = outroot / f"video{v}_{mode}"
            print(f"[run] video{v} mode={mode} ...", flush=True)
            fps = run_one(v, mode, args.every, outdir, extra)
            ev = json.loads((outdir / "events.json").read_text()) if (outdir / "events.json").exists() else []
            s = score(ev, gt[f"video{v}"])
            s.update(video=f"video{v}", mode=mode, fps=fps)
            rows.append(s)
            print(f"  -> events={s['n_events']} HIT={s['n_hit']}/{s['n_objs']} FP={s['fp']} "
                  f"latency={s['latencies']} fps={fps}", flush=True)

    print("\n=== SUMMARY (ABODA, HIT=#vật bắt được/#vật, FP=báo sai, latency vs abandon_frame) ===")
    hdr = f"{'video':8} {'mode':18} {'HIT':6} {'FP':4} {'events':7} {'latency(s)':16} {'FPS':5}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['video']:8} {r['mode']:18} {str(r['n_hit'])+'/'+str(r['n_objs']):6} {r['fp']:<4} "
              f"{r['n_events']:<7} {str(r['latencies']):16} {r['fps']}")
    print("\n=== Tổng theo mode ===")
    for mode in args.modes:
        mr = [r for r in rows if r["mode"] == mode]
        nh = sum(r["n_hit"] for r in mr); no = sum(r["n_objs"] for r in mr)
        fp = sum(r["fp"] for r in mr)
        print(f"  {mode:18}: HIT {nh}/{no} | tổng FP {fp} | FPS TB {sum(r['fps'] or 0 for r in mr)/max(1,len(mr)):.1f}")
    (outroot / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[summary] {outroot / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
