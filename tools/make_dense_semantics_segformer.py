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
from PIL import Image

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
from core.semantic_lut import BG_VALUE, FG_VALUE, build_moving_class_set, probability_to_semantic16


MODEL_IDS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
}


def resolve_default_video() -> str:
    here = Path(__file__).resolve()
    return os.fspath((here.parents[2] / "ABODA" / "video11.avi").resolve())


def resolve_default_out_root() -> str:
    here = Path(__file__).resolve()
    return os.fspath((here.parents[1] / "semantics_v11").resolve())


def select_variants(variant: str) -> list[str]:
    return ["b0", "b1"] if variant == "both" else [variant]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate RT-SBS dense scalar semantic maps with SegFormer B0/B1."
    )
    ap.add_argument("--video", default=resolve_default_video())
    ap.add_argument("--out-root", default=resolve_default_out_root())
    ap.add_argument("--variant", choices=["b0", "b1", "both"], default="both")
    ap.add_argument("--every", type=int, default=5, help="generate semantic map for every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--min-conf", type=float, default=0.0, help="set low-confidence pixels to zero semantic probability")
    ap.add_argument("--preview-every", type=int, default=0, help="write color previews every N generated maps")
    ap.add_argument("--skip-existing", action="store_true", help="reuse existing frame-indexed semantic maps")
    ap.add_argument("--local-files-only", action="store_true", help="load SegFormer from local HuggingFace cache only")
    return ap.parse_args()


def load_model(model_id: str, device: str, local_files_only: bool = False):
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: "
            "python -m pip install -r demov2/requirements.txt"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as exc:
        print(f"[segformer] processor config unavailable ({exc}); using built-in SegFormer ADE defaults")
        processor = SegformerImageProcessor(
            do_resize=True,
            size={"height": 512, "width": 512},
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_reduce_labels=True,
        )
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    return torch, F, processor, model, device


def _class_prob_sum(torch, prob, class_ids, device):
    valid_ids = [cid for cid in class_ids if 0 <= cid < prob.shape[1]]
    if not valid_ids:
        return torch.zeros_like(prob[:, 0])
    idx = torch.as_tensor(valid_ids, dtype=torch.long, device=device)
    return prob.index_select(1, idx).sum(dim=1)


def infer_map(
    torch,
    F,
    processor,
    model,
    device: str,
    frame_bgr: np.ndarray,
    moving_class_ids: list[int],
    min_conf: float,
    object_class_ids: list[int] | None = None,
    stuff_class_ids: list[int] | None = None,
):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        moving_prob = _class_prob_sum(torch, prob, moving_class_ids or [], device) if moving_class_ids else torch.zeros_like(conf)
        object_prob = _class_prob_sum(torch, prob, object_class_ids or [], device) if object_class_ids else torch.zeros_like(conf)
        stuff_prob = _class_prob_sum(torch, prob, stuff_class_ids or [], device) if stuff_class_ids else torch.zeros_like(conf)

    moving_prob_np = moving_prob[0].cpu().numpy().astype(np.float32)
    object_prob_np = object_prob[0].cpu().numpy().astype(np.float32)
    stuff_prob_np = stuff_prob[0].cpu().numpy().astype(np.float32)
    if min_conf > 0:
        conf_np = conf[0].cpu().numpy()
        moving_prob_np[conf_np < min_conf] = 0.0
        object_prob_np[conf_np < min_conf] = 0.0
        stuff_prob_np[conf_np < min_conf] = 0.0
    out = probability_to_semantic16(moving_prob_np)
    object16 = probability_to_semantic16(object_prob_np)
    stuff16 = probability_to_semantic16(stuff_prob_np)
    pred_np = pred[0].cpu().numpy().astype(np.int32)
    return out, pred_np, moving_prob_np, object16, stuff16


def write_preview(path: Path, scalar: np.ndarray) -> None:
    norm = np.clip(scalar.astype(np.float32) / float(FG_VALUE), 0.0, 1.0)
    preview = cv2.applyColorMap(np.rint(norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.fspath(path), preview)


def run_variant(args, variant: str) -> None:
    model_id = MODEL_IDS[variant]
    out_dir = Path(args.out_root) / f"segformer_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "_preview"
    if args.preview_every:
        preview_dir.mkdir(parents=True, exist_ok=True)

    torch, F, processor, model, device = load_model(model_id, args.device, args.local_files_only)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    moving_classes = build_moving_class_set(id2label)
    with open(out_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "provider": "segformer",
                "variant": variant,
                "encoding": "uint16 round(sum(P(moving-object classes)) * 65535)",
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

    summary_path = out_dir / "summary.csv"
    t0 = time.time()
    generated = 0
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "min", "mean", "max", "bg_rule_px", "high_prob_px"])
        for frame_idx in range(0, limit, every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            out_path = out_dir / f"{frame_idx:06d}.png"
            if args.skip_existing and out_path.exists():
                scalar = cv2.imread(os.fspath(out_path), cv2.IMREAD_UNCHANGED)
                if scalar is None:
                    raise RuntimeError(f"Cannot read existing semantic map: {out_path}")
                if scalar.ndim == 3:
                    scalar = scalar[:, :, 0]
                generated += 1
                continue
            scalar, _pred, _moving_prob, _object16, _stuff16 = infer_map(
                torch,
                F,
                processor,
                model,
                device,
                frame,
                moving_classes.ids,
                args.min_conf,
            )
            cv2.imwrite(os.fspath(out_path), scalar)
            writer.writerow(
                [
                    frame_idx,
                    int(scalar.min()),
                    float(np.mean(scalar)),
                    int(scalar.max()),
                    int((scalar <= 300).sum()),
                    int((scalar >= 175 * 256).sum()),
                ]
            )
            if args.preview_every and generated % args.preview_every == 0:
                write_preview(preview_dir / f"{frame_idx:06d}.jpg", scalar)
            generated += 1
            elapsed = time.time() - t0
            print(
                f"\r[segformer-{variant}] frame={frame_idx}/{limit} maps={generated} "
                f"{generated / max(0.001, elapsed):.2f} maps/s",
                end="",
                flush=True,
            )
    cap.release()
    print(f"\n[segformer-{variant}] wrote {generated} maps -> {out_dir}")


def main() -> int:
    args = parse_args()
    for variant in select_variants(args.variant):
        run_variant(args, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
