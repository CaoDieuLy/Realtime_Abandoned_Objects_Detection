from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def resolve_default_video() -> str:
    here = Path(__file__).resolve()
    return os.fspath((here.parents[2] / "ABODA" / "video11.avi").resolve())


def parse_args():
    ap = argparse.ArgumentParser(description="Run demo2 once with SegFormer-B0 maps and once with B1 maps.")
    ap.add_argument("--video", default=resolve_default_video())
    ap.add_argument("--sem-root", default=os.fspath((Path(__file__).resolve().parents[1] / "semantics_v11").resolve()))
    ap.add_argument("--out-root", default=os.fspath((Path(__file__).resolve().parents[1]).resolve()))
    ap.add_argument("--semantic-every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-masks-every", type=int, default=300)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_py = Path(__file__).resolve().parents[1] / "run_rtsbs_aod.py"
    for variant in ("b0", "b1"):
        sem_dir = Path(args.sem_root) / f"segformer_{variant}"
        if not sem_dir.exists():
            raise SystemExit(f"Missing semantic directory: {sem_dir}")
        out_dir = Path(args.out_root) / f"outputs_v11_segformer_{variant}"
        cmd = [
            sys.executable,
            os.fspath(run_py),
            "--video",
            args.video,
            "--semantic-mode",
            "dense",
            "--semantic-dir",
            os.fspath(sem_dir),
            "--semantic-every",
            str(args.semantic_every),
            "--outdir",
            os.fspath(out_dir),
            "--save-masks-every",
            str(args.save_masks_every),
        ]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

