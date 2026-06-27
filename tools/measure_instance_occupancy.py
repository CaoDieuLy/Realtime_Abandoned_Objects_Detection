"""Measure semantic instance occupancy on ABODA videos.

This is deliberately an *area* metric, not a people count.  At each sampled
frame it records the union area of animate instances (people/vehicles/animals),
the union area of static-object instances, their combined union, and instance
counts.  Event-level ABODA annotations identify samples where a labelled
abandoned object is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from core.semantic_classes import build_moving_class_set, build_object_class_set


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_ROOT = ROOT / "ABODA"
DEFAULT_GT = DEFAULT_VIDEO_ROOT / "aboda_gt.json"
DEFAULT_OUT = ROOT / "demov2" / "metrics" / "aboda_instance_occupancy_1fps.json"


def mean_min_max(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
    }


def instance_masks(result, height: int, width: int, animate_ids: set[int], object_ids: set[int]):
    animate = np.zeros((height, width), dtype=np.uint8)
    objects = np.zeros((height, width), dtype=np.uint8)
    animate_count = 0
    object_count = 0
    if result.masks is None or result.boxes is None:
        return animate, objects, animate_count, object_count
    for poly, box in zip(result.masks.xy, result.boxes):
        cls = int(box.cls)
        if poly is None or len(poly) < 3:
            continue
        if cls in animate_ids:
            target = animate
            animate_count += 1
        elif cls in object_ids:
            target = objects
            object_count += 1
        else:
            continue
        cv2.fillPoly(target, [np.asarray(poly, np.int32)], 1)
    return animate, objects, animate_count, object_count


def load_gt(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {v["video_id"]: v["objects"] for v in data["videos"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--weights", default="yolo26n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="semantic samples per source-video second; 1 is a CPU-friendly measurement")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    gt = load_gt(args.gt)
    videos = sorted([*args.video_root.glob("video*.avi"), *args.video_root.glob("vid*.mp4")],
                    key=lambda p: (0, int(p.stem[5:])) if p.stem.startswith("video") else (1, p.stem))
    model = YOLO(args.weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    animate_ids = set(build_moving_class_set(names).ids)
    object_ids = set(build_object_class_set(names).ids)
    print(f"[classes] animate={sorted(animate_ids)} object={sorted(object_ids)}", flush=True)

    report = {
        "method": {
            "model": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "sample_fps": args.sample_fps,
            "notes": [
                "Coverage is the union of detected instance-segmentation masks divided by frame area.",
                "Min/max/mean are over sampled frames, not every decoded source frame.",
                "gt_present metrics include samples between ABODA object start_frame and end_frame inclusive.",
            ],
        },
        "videos": {},
    }

    for path in videos:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(fps / args.sample_fps))
        pending: list[tuple[int, np.ndarray]] = []
        samples: list[dict] = []

        def infer_pending() -> None:
            if not pending:
                return
            frames = [frame for _, frame in pending]
            results = model.predict(frames, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)
            for (frame_idx, frame), result in zip(pending, results):
                h, w = frame.shape[:2]
                animate, objects, n_animate, n_object = instance_masks(result, h, w, animate_ids, object_ids)
                combined = (animate | objects) > 0
                samples.append({
                    "frame": frame_idx,
                    "animate_coverage_pct": float(animate.mean() * 100.0),
                    "object_coverage_pct": float(objects.mean() * 100.0),
                    "combined_coverage_pct": float(combined.mean() * 100.0),
                    "animate_instances": n_animate,
                    "object_instances": n_object,
                })
            pending.clear()

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                pending.append((frame_idx, frame))
                if len(pending) >= args.batch:
                    infer_pending()
            frame_idx += 1
        infer_pending()
        cap.release()

        def metric(key: str, subset: list[dict]) -> dict[str, float] | None:
            return mean_min_max([row[key] for row in subset]) if subset else None

        objects_gt = gt.get(path.stem, [])
        present = [row for row in samples if any(o["start_frame"] <= row["frame"] <= o["end_frame"] for o in objects_gt)]
        report["videos"][path.stem] = {
            "source_frames": frame_idx,
            "fps": fps,
            "sample_stride_frames": stride,
            "sample_count": len(samples),
            "all_samples": {
                "animate_coverage_pct": metric("animate_coverage_pct", samples),
                "object_coverage_pct": metric("object_coverage_pct", samples),
                "combined_coverage_pct": metric("combined_coverage_pct", samples),
                "animate_instances": metric("animate_instances", samples),
                "object_instances": metric("object_instances", samples),
            },
            "gt_object_present_samples": {
                "sample_count": len(present),
                "animate_coverage_pct": metric("animate_coverage_pct", present),
                "object_coverage_pct": metric("object_coverage_pct", present),
                "combined_coverage_pct": metric("combined_coverage_pct", present),
                "animate_instances": metric("animate_instances", present),
                "object_instances": metric("object_instances", present),
            } if objects_gt else None,
        }
        print(f"[done] {path.name}: {len(samples)} samples", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
