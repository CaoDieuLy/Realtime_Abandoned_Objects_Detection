from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args():
    root = repo_root()
    ap = argparse.ArgumentParser(description="Generate dense semantics and run RT-SBS-style AOD for ABODA videos.")
    ap.add_argument("--videos", nargs="+", default=["3", "7", "11"])
    ap.add_argument("--provider", choices=["segformer", "pspnet", "both"], default="segformer")
    ap.add_argument("--variant", choices=["b0", "b1"], default="b0", help="SegFormer variant")
    ap.add_argument("--pspnet-config", default="", help="mmseg PSPNet config path")
    ap.add_argument("--pspnet-checkpoint", default="", help="mmseg PSPNet checkpoint path")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-masks-every", type=int, default=300)
    ap.add_argument("--semantics-root", default=os.fspath(root / "demov2" / "semantics_aboda"))
    ap.add_argument("--outputs-root", default=os.fspath(root / "demov2" / "outputs_aboda_demo2"))
    ap.add_argument("--preview-every", type=int, default=0)
    ap.add_argument("--local-files-only", action="store_true", help="load SegFormer from local HuggingFace cache")
    ap.add_argument("--skip-existing-semantics", action="store_true", help="reuse existing semantic maps")
    return ap.parse_args()


def frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(os.fspath(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    root = repo_root()
    make_segformer = root / "demov2" / "tools" / "make_dense_semantics_segformer.py"
    make_pspnet = root / "demov2" / "tools" / "make_dense_semantics_pspnet.py"
    run_aod = root / "demov2" / "run_rtsbs_aod.py"
    summaries = []
    t0 = time.time()
    providers = ["segformer", "pspnet"] if args.provider == "both" else [args.provider]

    for vid in args.videos:
        video_name = f"video{vid}.avi"
        video_path = root / "ABODA" / video_name
        total = frame_count(video_path)
        limit = total if args.max_frames <= 0 else min(total, args.max_frames)
        for provider in providers:
            sem_root = Path(args.semantics_root) / f"video{vid}"
            if provider == "segformer":
                sem_dir = sem_root / f"segformer_{args.variant}"
                out_dir = Path(args.outputs_root) / f"video{vid}_segformer_{args.variant}"
                run_cmd([
                    sys.executable,
                    os.fspath(make_segformer),
                    "--video",
                    os.fspath(video_path),
                    "--variant",
                    args.variant,
                    "--every",
                    str(args.every),
                    "--out-root",
                    os.fspath(sem_root),
                    "--device",
                    args.device,
                    "--preview-every",
                    str(args.preview_every),
                    *(["--skip-existing"] if args.skip_existing_semantics else []),
                    *(["--local-files-only"] if args.local_files_only else []),
                    *([] if args.max_frames <= 0 else ["--max-frames", str(args.max_frames)]),
                ])
            else:
                if not args.pspnet_config or not args.pspnet_checkpoint:
                    raise SystemExit("--provider pspnet/both requires --pspnet-config and --pspnet-checkpoint")
                sem_dir = sem_root / "pspnet"
                out_dir = Path(args.outputs_root) / f"video{vid}_pspnet"
                run_cmd([
                    sys.executable,
                    os.fspath(make_pspnet),
                    "--video",
                    os.fspath(video_path),
                    "--config",
                    args.pspnet_config,
                    "--checkpoint",
                    args.pspnet_checkpoint,
                    "--every",
                    str(args.every),
                    "--out-root",
                    os.fspath(sem_root),
                    "--device",
                    args.device if args.device != "auto" else "cuda:0",
                    "--preview-every",
                    str(args.preview_every),
                    *(["--skip-existing"] if args.skip_existing_semantics else []),
                    *([] if args.max_frames <= 0 else ["--max-frames", str(args.max_frames)]),
                ])

            run_cmd([
                sys.executable,
                os.fspath(run_aod),
                "--video",
                os.fspath(video_path),
                "--semantic-mode",
                "dense",
                "--semantic-dir",
                os.fspath(sem_dir),
                "--semantic-every",
                str(args.every),
                "--outdir",
                os.fspath(out_dir),
                "--save-masks-every",
                str(args.save_masks_every),
                *([] if args.max_frames <= 0 else ["--max-frames", str(args.max_frames)]),
            ])

            events_path = out_dir / "events.json"
            events = []
            if events_path.exists():
                events = json.loads(events_path.read_text(encoding="utf-8"))
            summaries.append({
                "video": video_name,
                "provider": provider,
                "variant": args.variant if provider == "segformer" else "pspnet",
                "frames_requested": limit,
                "semantic_dir": os.fspath(sem_dir),
                "out_dir": os.fspath(out_dir),
                "events_path": os.fspath(events_path),
                "events": len(events),
            })
            print(f"[done] {video_name} {provider}: events={len(events)} out={out_dir}", flush=True)

    summary_name = args.provider if args.provider != "segformer" else args.variant
    summary_path = Path(args.outputs_root) / f"summary_{summary_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "elapsed_s": round(time.time() - t0, 2),
        "runs": summaries,
    }, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
