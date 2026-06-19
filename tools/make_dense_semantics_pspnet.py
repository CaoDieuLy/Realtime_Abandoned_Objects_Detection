from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
from core.semantic_lut import BG_VALUE, FG_VALUE, build_moving_class_set, probability_to_semantic16


def resolve_default_video() -> str:
    here = Path(__file__).resolve()
    return os.fspath((here.parents[2] / "ABODA" / "video11.avi").resolve())


def resolve_default_out_root() -> str:
    here = Path(__file__).resolve()
    return os.fspath((here.parents[1] / "semantics_v11").resolve())


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Generate RT-SBS dense scalar semantic maps with a PSPNet model from "
            "mmsegmentation. This is optional and requires mmseg/mmcv plus a config/checkpoint."
        )
    )
    ap.add_argument("--video", default=resolve_default_video())
    ap.add_argument("--out-root", default=resolve_default_out_root())
    ap.add_argument("--config", required=True, help="mmseg PSPNet config path")
    ap.add_argument("--checkpoint", required=True, help="mmseg PSPNet checkpoint path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="set low-confidence pixels to zero semantic probability when confidence is available")
    ap.add_argument("--preview-every", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def load_model(config: str, checkpoint: str, device: str):
    try:
        from mmseg.apis import inference_model, init_model
    except ImportError as exc:
        raise SystemExit(
            "Missing PSPNet runtime. Install a compatible mmsegmentation/mmcv stack, "
            "then rerun this script with --config and --checkpoint."
        ) from exc
    model = init_model(config, checkpoint, device=device)
    return inference_model, model


def model_classes(model) -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", None) or {}
    classes = dataset_meta.get("classes") or dataset_meta.get("CLASSES")
    if classes:
        return [str(c) for c in classes]

    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for key in ("classes", "CLASSES"):
            value = getattr(cfg, key, None)
            if value:
                return [str(c) for c in value]
    raise RuntimeError(
        "Cannot find class names in the PSPNet model metadata. Use an mmseg config/checkpoint "
        "with dataset_meta.classes so RT-SBS moving-object probabilities can be built."
    )


def extract_pred(result) -> np.ndarray:
    if isinstance(result, (list, tuple)) and result:
        result = result[0]
    pred = getattr(result, "pred_sem_seg", None)
    if pred is not None:
        data = getattr(pred, "data", pred)
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        arr = np.asarray(data)
        return arr.squeeze().astype(np.int32)

    if isinstance(result, tuple):
        result = result[0]
    arr = np.asarray(result)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr.astype(np.int32)


def _as_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-12)


def extract_probability_scalar(
    result,
    moving_class_ids: list[int],
    min_conf: float,
) -> tuple[np.ndarray, str]:
    if isinstance(result, (list, tuple)) and result:
        result = result[0]

    logits_holder = getattr(result, "seg_logits", None)
    if logits_holder is not None and moving_class_ids:
        logits = _as_numpy(getattr(logits_holder, "data", logits_holder))
        if logits.ndim == 4 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim == 3:
            prob = _softmax_np(logits)
            valid_ids = [cid for cid in moving_class_ids if 0 <= cid < prob.shape[0]]
            moving_prob = prob[valid_ids].sum(axis=0) if valid_ids else np.zeros(prob.shape[1:], dtype=np.float32)
            if min_conf > 0:
                conf = prob.max(axis=0)
                moving_prob[conf < min_conf] = 0.0
            return probability_to_semantic16(moving_prob), "seg_logits_probability"

    pred = extract_pred(result)
    moving = np.isin(pred, np.asarray(moving_class_ids, dtype=np.int32))
    return np.where(moving, FG_VALUE, BG_VALUE).astype(np.uint16), "hard_prediction_fallback"


def write_preview(path: Path, scalar: np.ndarray) -> None:
    norm = np.clip(scalar.astype(np.float32) / float(FG_VALUE), 0.0, 1.0)
    preview = cv2.applyColorMap(np.rint(norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.fspath(path), preview)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_root) / "pspnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "_preview"
    if args.preview_every:
        preview_dir.mkdir(parents=True, exist_ok=True)

    inference_model, model = load_model(args.config, args.checkpoint, args.device)
    classes = model_classes(model)
    moving_classes = build_moving_class_set(classes)

    with open(out_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": "pspnet",
                "config": os.path.abspath(args.config),
                "checkpoint": os.path.abspath(args.checkpoint),
                "encoding": "uint16 round(sum(P(moving-object classes)) * 65535) when logits are available",
                "fallback_encoding": "hard class prediction: moving classes become 65535, all others 0",
                "values": {"zero_probability": BG_VALUE, "full_probability": FG_VALUE},
                "moving_class_ids": moving_classes.ids,
                "moving_classes": moving_classes.labels,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = total if args.max_frames <= 0 else min(total, args.max_frames)
    every = max(1, int(args.every))

    t0 = time.time()
    generated = 0
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "min", "mean", "max", "bg_rule_px", "high_prob_px", "encoding_used"])
        for frame_idx in range(0, limit, every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            out_path = out_dir / f"{frame_idx:06d}.png"
            if args.skip_existing and out_path.exists():
                generated += 1
                continue

            result = inference_model(model, frame)
            scalar, encoding_used = extract_probability_scalar(result, moving_classes.ids, args.min_conf)
            if scalar.shape[:2] != frame.shape[:2]:
                scalar = cv2.resize(scalar, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.fspath(out_path), scalar)

            writer.writerow(
                [
                    frame_idx,
                    int(scalar.min()),
                    float(np.mean(scalar)),
                    int(scalar.max()),
                    int((scalar <= 300).sum()),
                    int((scalar >= 175 * 256).sum()),
                    encoding_used,
                ]
            )
            if args.preview_every and generated % args.preview_every == 0:
                write_preview(preview_dir / f"{frame_idx:06d}.jpg", scalar)

            generated += 1
            elapsed = time.time() - t0
            print(
                f"\r[pspnet] frame={frame_idx}/{limit} maps={generated} "
                f"{generated / max(0.001, elapsed):.2f} maps/s",
                end="",
                flush=True,
            )
    cap.release()
    print(f"\n[pspnet] wrote {generated} maps -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
