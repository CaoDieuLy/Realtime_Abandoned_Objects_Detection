from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.clean_bg_prior import build_camera_warmup_background, build_warmup_background, open_camera_capture, resize_to_width
from core.controlled_vibe import ControlledViBE, ViBEConfig
from core.dense_semantic import DenseSemanticSequence
from core.semantic_feedback import SemanticFeedback, SemanticFeedbackConfig
from core.static_matching import StaticMatcher
from core.static_state import StaticForegroundState, StaticStateConfig


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def resolve_default_video() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "ABODA", "video11.avi")
    return os.path.abspath(path)


def draw_alert(frame: np.ndarray, bbox: list[int], text: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(out, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def semantic_decision_preview(bg_rule: np.ndarray, fg_rule: np.ndarray) -> np.ndarray:
    out = np.full((*bg_rule.shape, 3), 128, dtype=np.uint8)
    out[bg_rule] = (60, 180, 75)
    out[fg_rule & ~bg_rule] = (230, 25, 75)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def read_frame(video_path: str, frame_idx: int, proc_width: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    frame = resize_to_width(frame, proc_width)
    return frame


class OnlineSegFormer:
    def __init__(
        self,
        variant: str,
        device: str,
        local_files_only: bool,
        min_conf: float,
    ):
        from tools.make_dense_semantics_segformer import MODEL_IDS, infer_map, load_model
        from core.semantic_lut import build_moving_class_set, build_object_class_set, build_stuff_class_set

        self.variant = variant
        self.model_id = MODEL_IDS[variant]
        self.min_conf = float(min_conf)
        self.infer_map = infer_map
        self.torch, self.F, self.processor, self.model, self.device = load_model(
            self.model_id,
            device,
            local_files_only=local_files_only,
        )
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.moving_classes = build_moving_class_set(id2label)
        self.object_classes = build_object_class_set(id2label)
        self.stuff_classes = build_stuff_class_set(id2label)
        self.last_object16: np.ndarray | None = None
        self.last_stuff16: np.ndarray | None = None

    def infer(self, frame: np.ndarray) -> np.ndarray:
        semantic, _pred, _moving_prob, object16, stuff16 = self.infer_map(
            self.torch,
            self.F,
            self.processor,
            self.model,
            self.device,
            frame,
            self.moving_classes.ids,
            self.min_conf,
            object_class_ids=self.object_classes.ids,
            stuff_class_ids=self.stuff_classes.ids,
        )
        self.last_object16 = object16
        self.last_stuff16 = stuff16
        return semantic


class OnlinePSPNet:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str,
        min_conf: float,
    ):
        from tools.make_dense_semantics_pspnet import (
            extract_probability_scalar,
            load_model,
            model_classes,
        )
        from core.semantic_lut import build_moving_class_set

        self.config = config
        self.checkpoint = checkpoint
        self.min_conf = float(min_conf)
        self.extract_probability_scalar = extract_probability_scalar
        self.inference_model, self.model = load_model(config, checkpoint, device)
        classes = model_classes(self.model)
        self.moving_classes = build_moving_class_set(classes)
        self.last_encoding = ""

    def infer(self, frame: np.ndarray) -> np.ndarray:
        result = self.inference_model(self.model, frame)
        semantic, self.last_encoding = self.extract_probability_scalar(
            result,
            self.moving_classes.ids,
            self.min_conf,
        )
        if semantic.shape[:2] != frame.shape[:2]:
            semantic = cv2.resize(
                semantic,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return semantic


class OnlineYoloSeg:
    """Fast instance-segmentation semantic source (YOLO-seg, COCO classes).

    Unlike dense SegFormer/PSPNet, this only paints detected instances; pixels with
    no detection stay 0 -> "abstain" when voting (the BG rule is disabled upstream).
    Class roles are fully configurable so you can, e.g., drop ``person`` and keep
    ``car`` from the animate set, or extend the object set.
      - animate map: P(person/vehicle/...) -> FSM reject gate + ViBE FG-protection.
      - object  map: P(bag/umbrella/...)   -> FSM keep boost.
    """

    SEMANTIC_MAX = 65535

    def __init__(
        self,
        weights: str,
        imgsz: int,
        conf: float,
        device: str,
        animate_terms: set[str] | None = None,
        object_terms: set[str] | None = None,
    ):
        from ultralytics import YOLO
        from core.semantic_lut import build_class_set, build_moving_class_set, build_object_class_set

        self.model = YOLO(weights)
        raw_names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        self.names = {int(k): str(v) for k, v in raw_names.items()}
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device

        animate = build_class_set(self.names, animate_terms) if animate_terms else build_moving_class_set(self.names)
        objects = build_class_set(self.names, object_terms) if object_terms else build_object_class_set(self.names)
        self.animate_ids = set(animate.ids)
        self.object_ids = set(objects.ids)
        self.animate_labels = animate.labels
        self.object_labels = objects.labels
        self.last_object16: np.ndarray | None = None

    def _paint(self, h: int, w: int, results) -> tuple[np.ndarray, np.ndarray]:
        animate = np.zeros((h, w), dtype=np.float32)
        obj = np.zeros((h, w), dtype=np.float32)

        def fill_poly(target: np.ndarray, pts: np.ndarray, conf: float) -> None:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            np.maximum(target, m.astype(np.float32) * conf, out=target)

        if results.masks is not None and results.boxes is not None:
            for poly, box in zip(results.masks.xy, results.boxes):
                c = int(box.cls)
                cf = float(box.conf)
                if poly is None or len(poly) < 3:
                    continue
                pts = np.asarray(poly, np.int32)
                if c in self.animate_ids:
                    fill_poly(animate, pts, cf)
                elif c in self.object_ids:
                    fill_poly(obj, pts, cf)
        elif results.boxes is not None:  # no masks -> box fallback
            for box in results.boxes:
                c = int(box.cls)
                cf = float(box.conf)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if c in self.animate_ids:
                    animate[y1:y2, x1:x2] = np.maximum(animate[y1:y2, x1:x2], cf)
                elif c in self.object_ids:
                    obj[y1:y2, x1:x2] = np.maximum(obj[y1:y2, x1:x2], cf)
        return animate, obj

    def infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame, verbose=False, imgsz=self.imgsz, conf=self.conf, device=self.device
        )[0]
        animate, obj = self._paint(h, w, results)
        self.last_object16 = (obj * self.SEMANTIC_MAX).astype(np.float32)
        return (animate * self.SEMANTIC_MAX).astype(np.float32)


class CrowdEstimator:
    """Density from the animate-blob count (smoothed): low / medium / high."""

    def __init__(self, n_crowd: int = 6, medium_frac: float = 0.5, smooth: int = 15):
        from collections import deque

        self.n_crowd = n_crowd
        self.medium_thr = max(1, int(n_crowd * medium_frac))
        self.hist = deque(maxlen=smooth)

    def update(self, count: int) -> str:
        self.hist.append(int(count))
        avg = sum(self.hist) / len(self.hist)
        if avg >= self.n_crowd:
            return "high"
        if avg >= self.medium_thr:
            return "medium"
        return "low"


class PybgsViBe:
    """pybgs ViBe (C++) as a drop-in moving-mask source.

    apply() segments AND updates the model internally (no update mask), so it CANNOT
    receive RT-SBS semantic feedback -> use for the demov1-classic mode (fast, no feedback).
    Mirrors the ControlledViBE interface: segmentation() + a no-op update().
    """

    def __init__(self, fg_threshold: int = 200):
        import pybgs

        os.makedirs("config", exist_ok=True)  # pybgs ViBe reads/writes ./config/ViBe.xml
        self.bgs = pybgs.ViBe()
        self.fg_threshold = int(fg_threshold)

    def segmentation(self, frame: np.ndarray) -> np.ndarray:
        fg = np.asarray(self.bgs.apply(frame))
        if fg.ndim == 3:
            fg = fg[:, :, 0]
        return (fg >= self.fg_threshold).astype(np.uint8) * 255

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        return  # pybgs updates internally during apply()


def parse_args():
    ap = argparse.ArgumentParser(
        description="demo2: original RT-SBS dense semantic feedback + clean-background AOD"
    )
    ap.add_argument("--video", default=resolve_default_video())
    ap.add_argument("--camera-index", type=int, default=-1,
                    help="live camera index; >=0 uses webcam/camera instead of --video")
    ap.add_argument("--camera-width", type=int, default=0, help="optional live camera capture width")
    ap.add_argument("--camera-height", type=int, default=0, help="optional live camera capture height")
    ap.add_argument("--camera-fps", type=float, default=0.0, help="optional live camera FPS hint")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_v11_demo2"))
    ap.add_argument("--bg-learn-seconds", type=float, default=20.0)
    ap.add_argument("--sample-step", type=int, default=5)
    ap.add_argument("--proc-width", type=int, default=640,
                    help="downscale frames to this width (keep aspect) for the BGS/FSM pipeline; 0=native. "
                         "Heavy per-pixel ops run here -> keep it small (640) for speed + meaningful area thresholds.")
    ap.add_argument("--sem-proc-width", type=int, default=960,
                    help="resolution fed to the ONLINE semantic engine (YOLO/SegFormer), DECOUPLED from --proc-width. "
                         "Higher = better person/object detection (its mask is resized back to --proc-width). "
                         "YOLO cost ~ depends on --yolo-imgsz, not on this, so high detail is ~free.")
    ap.add_argument("--warmup-s", type=float, default=22.0)
    ap.add_argument("--max-frames", type=int, default=0)
    # person-aware warm-up: exclude animate (person/vehicle) pixels from the clean_bg
    # median so a stander in warm-up isn't baked into the background (= ghost FP later).
    # Uses the active semantic engine + the animate class set (--yolo-animate-classes).
    # Mode-agnostic (clean_bg is shared). Off by default; needs an online semantic engine.
    ap.add_argument("--warmup-ignore", type=int, default=0, help="1=mask animate classes out of clean_bg warmup median")
    ap.add_argument("--warmup-ignore-dilate", type=int, default=8, help="dilate the animate mask before excluding (px)")
    ap.add_argument("--warmup-ignore-max", type=int, default=40, help="max warmup frames used for the masked median")

    ap.add_argument("--vibe-samples", type=int, default=30)
    ap.add_argument("--vibe-threshold", type=int, default=10, help="RT-SBS ViBE base threshold; color uses 4.5x this value")
    ap.add_argument("--vibe-matches", type=int, default=2)
    ap.add_argument("--vibe-update-factor", type=int, default=8)
    ap.add_argument("--vibe-color-mult", type=float, default=4.5)
    ap.add_argument("--vibe-neighborhood-radius", type=int, default=1)
    ap.add_argument("--vibe-timeout", type=int, default=150, help="force background update after N frames (0=off)")

    ap.add_argument("--th-diff", type=int, default=40)
    ap.add_argument("--clean-update-lr", type=float, default=0.0008)

    ap.add_argument("--semantic-mode", choices=["dense", "online-segformer", "online-pspnet", "online-yoloseg", "none"], default="dense")
    ap.add_argument("--semantic-dir", default="", help="folder of dense 16-bit semantic maps, one file per semantic frame")
    ap.add_argument(
        "--semantic-index-mode",
        choices=["strict", "sequential"],
        default="strict",
        help="strict reads exact frame-indexed files like 000005.png; sequential mimics full per-frame folders",
    )
    ap.add_argument("--semantic-every", type=int, default=5, help="RT-SBS semantic framerate: use one dense map every N frames")
    ap.add_argument("--segformer-variant", choices=["b0", "b1"], default="b0")
    ap.add_argument("--segformer-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--segformer-min-conf", type=float, default=0.0)
    ap.add_argument("--segformer-local-files-only", action="store_true")
    ap.add_argument("--pspnet-config", default="")
    ap.add_argument("--pspnet-checkpoint", default="")
    ap.add_argument("--pspnet-device", default="cuda:0")
    ap.add_argument("--pspnet-min-conf", type=float, default=0.0)
    ap.add_argument("--yolo-weights", default="yolo26n-seg.pt", help="ultralytics instance-seg weights")
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--yolo-conf", type=float, default=0.15)
    ap.add_argument("--yolo-device", default="cpu", help="cpu / 0 / cuda:0")
    ap.add_argument("--yolo-animate-classes", default="",
                    help="comma COCO names to treat as ANIMATE/reject (default person+vehicles). "
                         "e.g. 'car,bus,truck' to DROP person but keep car")
    ap.add_argument("--yolo-object-classes", default="",
                    help="comma COCO names to treat as KEEP-object (default bag/umbrella/suitcase/...)")
    ap.add_argument("--tau-bg", type=int, default=300, help="tau_BG from RT-SBS, 16-bit semantic units")
    ap.add_argument("--tau-fg", type=int, default=175, help="tau_FG from RT-SBS, 8-bit value multiplied by 256 internally")
    ap.add_argument("--tau-bg-star", type=int, default=65, help="tau_BG* color reuse threshold")
    ap.add_argument("--tau-fg-star", type=int, default=115, help="tau_FG* color reuse threshold")
    ap.add_argument("--modulo-update", type=int, default=256, help="semantic model random update period")
    ap.add_argument(
        "--mode",
        choices=["custom", "no-feedback", "instance-feedback", "dense-feedback"],
        default="no-feedback",
        help="pipeline preset (overrides --bgs-backend/--semantic-feedback/--aod-motion-source). "
             "DEFAULT no-feedback so a bare run works out of the box (= demov1 pipeline). "
             "no-feedback: fast C++ ViBE, semantic is NOT written back into the BGS model; semantic "
             "only removes people downstream (the classic BGS+person-filter pipeline). "
             "instance-feedback: instance seg (YOLO) + one-sided RT-SBS feedback (force-FG protect) into ViBE. "
             "dense-feedback: dense pixel seg (SegFormer) + full two-sided RT-SBS feedback (force-BG + force-FG). "
             "custom: use the individual --bgs-backend / --semantic-* / --aod-* flags as given.",
    )
    ap.add_argument("--bgs-backend", choices=["controlled", "pybgs"], default="controlled",
                    help="moving-mask BGS: controlled=ControlledViBE (numba, supports RT-SBS feedback); "
                         "pybgs=C++ ViBe (faster, NO feedback)")
    ap.add_argument("--semantic-feedback", choices=["on", "off"], default="on",
                    help="on = RT-SBS semantic feedback into ViBE (segment_with_semantics + vibe.update(rtsbs)); "
                         "off = vanilla ViBE update (raw_vibe), semantic still used for the AOD gate")
    ap.add_argument("--aod-motion-source", choices=["framediff", "rtsbs", "raw-vibe"], default="framediff",
                    help="motion gate separating moving FG from static FG. framediff (v2c, default) does NOT "
                         "eat a just-deposited still object; rtsbs/raw-vibe use the ViBE mask instead")
    ap.add_argument("--aod-tstatic-s", type=float, default=1.0,
                    help="seconds a pixel must stay static (and inanimate) before it becomes an abandoned candidate")
    ap.add_argument("--tau-animate", type=float, default=0.3,
                    help="P(person/vehicle/animal) >= this -> treat pixel as animate and reject from AOD")
    ap.add_argument("--dilate-animate", type=int, default=2,
                    help="dilate the animate(person/vehicle) reject mask by this many px (demov1 used 12 for person)")
    ap.add_argument("--tau-object", type=float, default=0.30,
                    help="P(bag/box/bottle/...) >= this -> positive keep evidence for AOD (online only)")
    ap.add_argument("--tau-stuff", type=float, default=0.50,
                    help="P(wall/floor/water/...) >= this -> scene background, REJECT candidate (dense/segformer only)")
    ap.add_argument("--stuff-reject", action="store_true",
                    help="enable stuff-class reject (water/wall/floor). WARNING: a coarse model (SegFormer-b0) "
                         "mislabels a small object (umbrella) as 'floor' and would reject it too. Off by default.")
    ap.add_argument("--no-semantic-gate", action="store_true",
                    help="disable the semantic keep gate on the AOD static-object channel (ablation)")

    # v2c StaticDiffBG features
    ap.add_argument("--persist-s", type=float, default=2.0,
                    help="persistent-diff >= this many s -> protect clean_bg from absorbing it (don't eat static object)")
    ap.add_argument("--light-comp", type=int, default=1, help="1=re-baseline clean_bg on high-coverage lighting events")
    ap.add_argument("--heal-cov", type=float, default=0.15)
    ap.add_argument("--heal-alpha", type=float, default=1.0)
    ap.add_argument("--heal-alpha-dark", type=float, default=0.05)
    ap.add_argument("--dark-s-thresh", type=float, default=15.0)
    ap.add_argument("--relight", type=int, default=1, help="1=rebuild clean_bg on a global lighting-mode switch (day/night, lights on/off)")
    ap.add_argument("--relight-dv", type=float, default=30.0)
    ap.add_argument("--relight-ds", type=float, default=12.0)
    ap.add_argument("--relearn-s", type=float, default=2.0)

    # v2c runner features: dedup, owner-gate, crowd, bbox refine
    ap.add_argument("--heal-revealed", type=int, default=0,
                    help="adaptive clean_bg: run the semantic engine ON clean_bg to find agents (car/person) "
                         "BAKED into it; when one leaves (newdiff fires there), absorb the revealed ground "
                         "instead of alerting. Semantic-based -> robust to outdoor shadows. YOLO/animate modes.")
    ap.add_argument("--heal-min-area", type=int, default=2000,
                    help="only heal candidates at least this large (px @ proc resolution)")
    ap.add_argument("--heal-agent-overlap", type=float, default=0.3,
                    help="bbox must overlap the baked-agent mask >= this fraction to be treated as a departure ghost")
    ap.add_argument("--dedup-dist", type=float, default=40.0, help="two alerts closer than this (px) = same location")
    ap.add_argument("--dedup-clear-s", type=float, default=3.0, help="object must leave a spot (newdiff empty) this long before a new alert there")
    ap.add_argument("--owner-gate", type=int, default=1, help="1=sparse scenes only: delay alert until the object's reach is clear of people")
    ap.add_argument("--owner-clear-s", type=float, default=3.0)
    ap.add_argument("--owner-margin", type=int, default=15)
    ap.add_argument("--owner-k", type=float, default=0.8)
    ap.add_argument("--crowd-n", type=int, default=6, help="avg >= this many person/vehicle blobs -> crowded -> disable owner-gate")
    ap.add_argument("--gather-px", type=int, default=5, help="CLOSE tight_mask this many px to merge fragments when refining the alert bbox; 0=off. "
                    "Default 5 = light join (covers a fragmented object) without over-merging a crowd into one giant box (v11).")

    # Prop3 / Prop2 / Prop1
    ap.add_argument("--motion-to-static", action="store_true",
                    help="Prop3: a candidate must have shown motion (deposited) before going static; "
                         "rejects scene-diff clutter (signage/floor/columns) that never moved")
    ap.add_argument("--motion-reset-s", type=float, default=1.0,
                    help="Prop3: sustained back-to-bg seconds before clearing the moved latch (survives occlusion gaps)")
    ap.add_argument("--motion-latch-dilate", type=int, default=4,
                    help="Prop3: dilate the motion latch px so motion near a deposited object counts")
    ap.add_argument("--owner-gate-local", action="store_true",
                    help="Prop2: apply owner-gate by LOCAL person distance even in crowded scenes "
                         "(not disabled by global density), with --owner-timeout-s to avoid infinite defer")
    ap.add_argument("--owner-timeout-s", type=float, default=30.0,
                    help="Prop2: max seconds to defer an alert while a person stays in reach (then alert anyway)")
    ap.add_argument("--person-hull", action="store_true",
                    help="Prop1: in crowds, close/dilate the person mask to cover clusters, punching holes "
                         "at static candidates (stat_new) so a deposited object is not swallowed")
    ap.add_argument("--person-dilate", type=int, default=12, help="Prop1: dilate person mask px")
    ap.add_argument("--hull-close", type=int, default=21, help="Prop1: close kernel px to bridge adjacent people")
    ap.add_argument("--person-overlap-max", type=float, default=0.0,
                    help="ported from v1: drop a candidate whose bbox overlaps the person/animate mask "
                         ">= this fraction (a standing person). 0 = off; v1 used 0.25")
    ap.add_argument("--person-hull-always", action="store_true",
                    help="apply person-hull regardless of crowd density (since YOLO-nano undercounts people)")

    # Scene Feature Memory: optional conservative suppression AFTER the matcher (default OFF).
    #   relocated  = pre-existing scene object moved elsewhere (appearance match + source changed + far)
    #   background = glare/shadow showing the SAME background structure at the candidate's spot
    ap.add_argument("--scene-memory", type=int, default=0, help="1=enable Scene Feature Memory suppression")
    ap.add_argument("--scene-memory-mode", choices=["relocated", "background", "both"], default="relocated")
    ap.add_argument("--scene-memory-patch", type=int, default=32)
    ap.add_argument("--scene-memory-stride", type=int, default=16)
    ap.add_argument("--scene-memory-thresh", type=float, default=0.88, help="relocated: appearance match to suppress")
    ap.add_argument("--scene-memory-source-change", type=float, default=0.15, help="relocated: source patch changed frac")
    ap.add_argument("--scene-memory-min-move-dist", type=float, default=40.0, help="relocated: source must be this far")
    ap.add_argument("--scene-memory-bg-sim", type=float, default=0.90, help="background: structure self-similarity to clean_bg")
    ap.add_argument("--scene-memory-debug", type=int, default=0, help="1=save scene_suppress_*.jpg debug images")

    ap.add_argument("--ts-static", type=float, default=5.0)
    ap.add_argument("--min-stable-s", type=float, default=1.5)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--match-dist-px", type=float, default=40.0)
    ap.add_argument("--area-min", type=int, default=60)
    ap.add_argument("--area-max", type=int, default=30000)
    ap.add_argument("--miss-tol-s", type=float, default=1.0)
    ap.add_argument("--aspect-max", type=float, default=5.0)
    ap.add_argument("--fill-min", type=float, default=0.18)

    ap.add_argument("--save-masks-every", type=int, default=0)
    args = ap.parse_args()

    # --mode presets bundle the BGS / feedback / motion-gate axes into a coherent pipeline.
    if args.mode == "no-feedback":
        # Fast C++ ViBE; semantic is NOT written back into the BGS model. Semantic is used
        # only downstream to remove people (the classic BGS + person-filter pipeline).
        args.bgs_backend = "pybgs"
        args.semantic_feedback = "off"
        args.aod_motion_source = "raw-vibe"
        args.dilate_animate = 12  # subtract a 12px-dilated person mask
        if args.semantic_mode == "dense":
            args.semantic_mode = "online-yoloseg"
    elif args.mode == "instance-feedback":
        # Instance seg (YOLO) + one-sided RT-SBS feedback (force-FG protect) into ViBE.
        args.bgs_backend = "controlled"
        args.semantic_feedback = "on"
        args.aod_motion_source = "raw-vibe"
        if args.semantic_mode == "dense":
            args.semantic_mode = "online-yoloseg"
    elif args.mode == "dense-feedback":
        # Dense pixel seg (SegFormer) + full two-sided RT-SBS feedback (force-BG + force-FG).
        args.bgs_backend = "controlled"
        args.semantic_feedback = "on"
        args.aod_motion_source = "rtsbs"
        # default to online SegFormer unless the user asked for offline dense maps / pspnet
        if args.semantic_mode in ("none", "online-yoloseg"):
            args.semantic_mode = "online-segformer"
        elif args.semantic_mode == "dense" and not args.semantic_dir:
            args.semantic_mode = "online-segformer"
    return args


def main() -> int:
    args = parse_args()
    if args.semantic_mode == "dense" and not args.semantic_dir:
        raise SystemExit(
            "[demo2] --semantic-mode dense follows original RT-SBS and requires "
            "--semantic-dir with one dense semantic map per frame."
        )
    if args.semantic_mode == "online-pspnet" and (
        not args.pspnet_config or not args.pspnet_checkpoint
    ):
        raise SystemExit(
            "[demo2] --semantic-mode online-pspnet requires "
            "--pspnet-config and --pspnet-checkpoint."
        )

    os.makedirs(args.outdir, exist_ok=True)

    use_camera = args.camera_index >= 0
    if use_camera:
        clean_bg, warmup_frames, fps, total = build_camera_warmup_background(
            args.camera_index,
            args.bg_learn_seconds,
            args.sample_step,
            width=args.camera_width,
            height=args.camera_height,
            fps_hint=args.camera_fps,
        )
        source_label = f"camera:{args.camera_index}"
    else:
        clean_bg, warmup_frames, fps, total = build_warmup_background(
            args.video, args.bg_learn_seconds, args.sample_step, proc_width=args.proc_width
        )
        source_label = args.video
    h, w = clean_bg.shape[:2]
    print(
        f"[warmup] {len(warmup_frames)} samples from 0-{args.bg_learn_seconds:.1f}s "
        f"step={args.sample_step} | {w}x{h} @ {fps:.2f}fps total={total or 'live'} "
        f"source={source_label}"
    )
    first_frame_for_semantic = warmup_frames[0].copy()

    if args.bgs_backend == "pybgs":
        vibe = PybgsViBe()
    else:
        vibe = ControlledViBE(
            warmup_frames,
            ViBEConfig(
                samples=args.vibe_samples,
                matching_threshold=args.vibe_threshold,
                matching_number=args.vibe_matches,
                update_factor=args.vibe_update_factor,
                color_threshold_multiplier=args.vibe_color_mult,
                neighborhood_radius=args.vibe_neighborhood_radius,
                foreground_timeout=args.vibe_timeout,
            ),
        )
    semantic_sequence = None
    online_semantic = None
    online_semantic_name = ""
    online_initial_semantic = None
    initial_semantic = None
    if args.semantic_mode == "dense":
        semantic_sequence = DenseSemanticSequence(
            args.semantic_dir,
            h,
            w,
            strict_frame_index=args.semantic_index_mode == "strict",
        )
        initial_semantic = semantic_sequence.read(0)
        print(
            f"[semantic] dense RT-SBS maps={len(semantic_sequence)} dir={args.semantic_dir} "
            f"every={args.semantic_every} tau_bg={args.tau_bg} tau_fg={args.tau_fg} "
            f"tau_bg*={args.tau_bg_star} tau_fg*={args.tau_fg_star}"
        )
    elif args.semantic_mode == "online-segformer":
        online_semantic = OnlineSegFormer(
            args.segformer_variant,
            args.segformer_device,
            args.segformer_local_files_only,
            args.segformer_min_conf,
        )
        first_frame = first_frame_for_semantic
        online_initial_semantic = online_semantic.infer(first_frame)
        initial_semantic = online_initial_semantic
        print(
            f"[semantic] online SegFormer-{args.segformer_variant} model={online_semantic.model_id} "
            f"device={online_semantic.device} every={args.semantic_every} "
            f"moving_classes={len(online_semantic.moving_classes.ids)} "
            f"tau_bg={args.tau_bg} tau_fg={args.tau_fg} "
            f"tau_bg*={args.tau_bg_star} tau_fg*={args.tau_fg_star}"
        )
        online_semantic_name = f"SegFormer-{args.segformer_variant}"
    elif args.semantic_mode == "online-pspnet":
        online_semantic = OnlinePSPNet(
            args.pspnet_config,
            args.pspnet_checkpoint,
            args.pspnet_device,
            args.pspnet_min_conf,
        )
        first_frame = first_frame_for_semantic
        online_initial_semantic = online_semantic.infer(first_frame)
        initial_semantic = online_initial_semantic
        print(
            f"[semantic] online PSPNet config={args.pspnet_config} "
            f"checkpoint={args.pspnet_checkpoint} device={args.pspnet_device} "
            f"every={args.semantic_every} moving_classes={len(online_semantic.moving_classes.ids)} "
            f"encoding={online_semantic.last_encoding} "
            f"tau_bg={args.tau_bg} tau_fg={args.tau_fg} "
            f"tau_bg*={args.tau_bg_star} tau_fg*={args.tau_fg_star}"
        )
        online_semantic_name = "PSPNet"
    elif args.semantic_mode == "online-yoloseg":
        animate_terms = {t.strip().lower() for t in args.yolo_animate_classes.split(",") if t.strip()} or None
        object_terms = {t.strip().lower() for t in args.yolo_object_classes.split(",") if t.strip()} or None
        online_semantic = OnlineYoloSeg(
            args.yolo_weights,
            args.yolo_imgsz,
            args.yolo_conf,
            args.yolo_device,
            animate_terms=animate_terms,
            object_terms=object_terms,
        )
        first_frame = first_frame_for_semantic
        online_initial_semantic = online_semantic.infer(first_frame)
        initial_semantic = online_initial_semantic
        print(
            f"[semantic] online YOLO-seg weights={args.yolo_weights} imgsz={args.yolo_imgsz} "
            f"conf={args.yolo_conf} device={args.yolo_device} every={args.semantic_every}\n"
            f"           animate(reject)={online_semantic.animate_labels}\n"
            f"           object(keep)={online_semantic.object_labels}\n"
            f"           BG-rule DISABLED (empty=abstain); feedback = FG-protect only"
        )
        online_semantic_name = "YOLO-seg"
    else:
        print("[semantic] disabled: running ViBE feedback without RT-SBS semantic rules")

    feedback = SemanticFeedback(
        h,
        w,
        SemanticFeedbackConfig(
            tau_bg=args.tau_bg,
            tau_fg=args.tau_fg,
            tau_bg_star=args.tau_bg_star,
            tau_fg_star=args.tau_fg_star,
            model_update_factor=args.modulo_update,
            enable_bg_rule=(args.semantic_mode != "online-yoloseg"),
        ),
        initial_semantic=initial_semantic,
    )
    clean_bg_color = np.median(np.stack(warmup_frames, axis=0), axis=0).astype(np.float32) if warmup_frames else None
    if args.warmup_ignore and warmup_frames:
        if online_semantic is not None:
            from core.clean_bg_prior import build_person_aware_clean_bg
            from core.semantic_lut import SEMANTIC_MAX
            tau16 = float(args.tau_animate) * float(SEMANTIC_MAX)
            clean_bg, clean_bg_color, cov = build_person_aware_clean_bg(
                warmup_frames, online_semantic.infer, tau16,
                dilate_px=args.warmup_ignore_dilate, max_frames=args.warmup_ignore_max,
            )
            print(f"[warmup] person-aware ON: animate masked from clean_bg "
                  f"(tau={args.tau_animate}, dilate={args.warmup_ignore_dilate}px, coverage~{cov*100:.1f}%)")
        else:
            print("[warmup] --warmup-ignore set but no online semantic engine (dense precomputed) -> skipped")
    sfg = StaticForegroundState(
        clean_bg,
        StaticStateConfig(
            th_diff=args.th_diff,
            update_lr=args.clean_update_lr,
            t_static_s=args.aod_tstatic_s,
            fps=fps,
            tau_animate=args.tau_animate,
            tau_object=args.tau_object,
            tau_stuff=args.tau_stuff,
            dilate_animate=args.dilate_animate,
            motion_source=args.aod_motion_source,
            persist_s=args.persist_s,
            light_comp=bool(args.light_comp),
            heal_cov=args.heal_cov,
            heal_alpha=args.heal_alpha,
            heal_alpha_dark=args.heal_alpha_dark,
            dark_s_thresh=args.dark_s_thresh,
            relight=bool(args.relight),
            relight_dv=args.relight_dv,
            relight_ds=args.relight_ds,
            relearn_s=args.relearn_s,
            motion_to_static=bool(args.motion_to_static),
            motion_reset_s=args.motion_reset_s,
            motion_latch_dilate=args.motion_latch_dilate,
        ),
        clean_bg_color=clean_bg_color,
    )
    semantic_gate_on = (args.semantic_mode != "none") and (not args.no_semantic_gate)
    last_animate16 = initial_semantic if semantic_gate_on else None
    last_object16: np.ndarray | None = None
    last_stuff16: np.ndarray | None = None

    matcher = StaticMatcher(
        match_iou=args.match_iou,
        match_dist_px=args.match_dist_px,
        area_min=args.area_min,
        area_max=args.area_max,
        miss_tol=max(1, int(args.miss_tol_s * fps)),
        aspect_max=args.aspect_max,
        fill_min=args.fill_min,
    )

    scene_mem = None
    if args.scene_memory and clean_bg_color is not None:
        from core.scene_feature_memory import SceneFeatureMemory, SceneMemoryConfig
        scene_mem = SceneFeatureMemory(
            clean_bg_color,
            SceneMemoryConfig(
                patch=args.scene_memory_patch,
                stride=args.scene_memory_stride,
                match_thresh=args.scene_memory_thresh,
                source_change=args.scene_memory_source_change,
                min_move_dist=args.scene_memory_min_move_dist,
                bg_sim_thresh=args.scene_memory_bg_sim,
            ),
        )
        scene_modes = ["relocated", "background"] if args.scene_memory_mode == "both" else [args.scene_memory_mode]
        print(f"[scene-memory] ON mode={args.scene_memory_mode} patches={len(scene_mem.patches)} "
              f"(thresh={args.scene_memory_thresh} bg_sim={args.scene_memory_bg_sim})")
    suppressed: list[dict] = []
    healed: list[dict] = []   # adaptive clean_bg heals (revealed background absorbed)

    # heal-revealed: detect agents (car/person) BAKED into clean_bg -> their regions, when newdiff
    # fires there later (the agent left), are revealed-ground ghosts to heal, not abandoned objects.
    baked_agent_mask = None
    if args.heal_revealed and online_semantic is not None and clean_bg_color is not None:
        bg_u8 = np.clip(clean_bg_color, 0, 255).astype(np.uint8)
        a16 = online_semantic.infer(bg_u8)
        if a16 is not None and a16.shape[:2] != (h, w):
            a16 = cv2.resize(a16, (w, h), interpolation=cv2.INTER_NEAREST)
        bm = (a16 >= args.tau_animate * 65535).astype(np.uint8)
        bm = cv2.dilate(bm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        baked_agent_mask = bm.astype(bool)
        print(f"[heal-revealed] baked-agent mask = {float(baked_agent_mask.mean()) * 100:.1f}% of clean_bg "
              f"(agents detected in the learned background -> heal on departure)")

    if use_camera:
        cap = open_camera_capture(
            args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(source_label)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    last_semantic_vis: np.ndarray | None = None
    events = []
    alerted: set[int] = set()
    static_since: dict[int, float] = {}
    last_center: dict[int, tuple[float, float]] = {}
    hits: dict[int, int] = {}
    min_hits = max(1, int(args.min_stable_s * fps))

    feedback_on = (
        args.semantic_feedback == "on"
        and args.bgs_backend == "controlled"  # pybgs can't accept the corrected update mask
        and (semantic_sequence is not None or online_semantic is not None)
    )
    crowd = CrowdEstimator(n_crowd=args.crowd_n)
    density = "low"
    alerted_locs: list[list[int]] = []          # [cx, cy, last_seen_frame] for location dedup
    person_near_frame: dict[int, int] = {}       # cand_id -> last frame a person was within reach
    gather_k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.gather_px, args.gather_px))
                if args.gather_px > 0 else None)
    person_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, args.person_dilate),) * 2)

    print(
        f"[demo2] mode={args.mode} | bgs-backend={args.bgs_backend} | "
        f"semantic-feedback={'ON' if feedback_on else 'OFF'} | motion-gate={args.aod_motion_source} | "
        f"semantic-gate={'on' if semantic_gate_on else 'off'} | relight={'on' if args.relight else 'off'} | "
        f"owner-gate={'on' if args.owner_gate else 'off'} | dedup={args.dedup_dist}px"
    )
    t0 = time.time()
    total_label = total if total > 0 else "live"
    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[demo2] stop requested; finishing current frame...", flush=True)

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _request_stop)
    i = 0
    while True:
        if stop_requested:
            break
        ok, frame_full = cap.read()
        if not ok:
            break
        frame = resize_to_width(frame_full, args.proc_width)  # BGS/FSM resolution
        if args.max_frames and i >= args.max_frames:
            break

        t_now = i / fps
        raw_vibe = vibe.segmentation(frame)

        # --- semantic inference (for the AOD gate; independent of the feedback toggle) ---
        ran_semantic = False
        has_semantic = semantic_sequence is not None or online_semantic is not None
        if has_semantic and (i % max(1, args.semantic_every)) == 0:
            if semantic_sequence is not None:
                semantic_map = semantic_sequence.read(i)
            elif i == 0 and online_initial_semantic is not None:
                semantic_map = online_initial_semantic
            else:
                # DECOUPLE: feed the online engine a higher-res frame (sem-proc-width) for
                # detail, then resize its mask back to the BGS/FSM resolution (proc-width).
                frame_yolo = resize_to_width(frame_full, args.sem_proc_width)
                semantic_map = online_semantic.infer(frame_yolo)

            def _to_proc(m):
                if m is None or m.shape[:2] == (h, w):
                    return m
                return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)

            semantic_map = _to_proc(semantic_map)
            last_semantic_vis = semantic_map
            ran_semantic = True
            if semantic_gate_on:
                last_animate16 = semantic_map
                last_object16 = _to_proc(getattr(online_semantic, "last_object16", None) if online_semantic is not None else None)
                last_stuff16 = _to_proc(getattr(online_semantic, "last_stuff16", None) if online_semantic is not None else None)

        # --- RT-SBS semantic feedback into ViBE (toggleable) ---
        if feedback_on:
            if ran_semantic:
                rtsbs_mask = feedback.segment_with_semantics(frame, raw_vibe, semantic_map)
            else:
                rtsbs_mask = feedback.segment_without_semantics(frame, raw_vibe)
        else:
            rtsbs_mask = raw_vibe
        vibe.update(frame, rtsbs_mask)

        # --- AOD static-FG state machine ---
        if args.aod_motion_source == "raw-vibe":
            motion_mask = raw_vibe
        elif args.aod_motion_source == "rtsbs":
            motion_mask = rtsbs_mask
        else:
            motion_mask = None  # framediff: FSM computes it internally
        protect = None
        if alerted_locs:
            protect_u8 = np.zeros((h, w), dtype=np.uint8)
            for L in alerted_locs:
                cv2.circle(protect_u8, (int(L[0]), int(L[1])), 18, 1, -1)
            protect = protect_u8 > 0
        aod = sfg.update(
            frame,
            motion_mask=motion_mask,
            animate_prob16=last_animate16,
            object_prob16=last_object16,
            stuff_prob16=(last_stuff16 if args.stuff_reject else None),
            protect_mask=protect,
        )
        stat_new = aod["abandoned"]
        newdiff = aod["newdiff"]
        aod_fg = cv2.morphologyEx(stat_new, cv2.MORPH_CLOSE, kernel_close)

        # --- person presence (animate proxy) -> crowd density + owner-gate distance map ---
        person_b = (last_animate16 >= args.tau_animate * 65535) if last_animate16 is not None else None
        if ran_semantic and person_b is not None:
            n_blobs = cv2.connectedComponents(person_b.astype(np.uint8))[0] - 1
            density = crowd.update(max(0, n_blobs))

        # Prop1 person-hull: in crowds, bridge adjacent people (close) but punch holes at
        # static candidates (stat_new) so a deposited object inside the cluster survives.
        if args.person_hull and person_b is not None and person_b.any() and (args.person_hull_always or density != "low"):
            pmask = cv2.dilate(person_b.astype(np.uint8), person_dil)
            hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.hull_close, args.hull_close))
            hull = cv2.morphologyEx(pmask, cv2.MORPH_CLOSE, hk)
            hull[stat_new > 0] = 0
            pmask = cv2.bitwise_or(pmask, hull)
            aod_fg = cv2.bitwise_and(aod_fg, aod_fg, mask=cv2.bitwise_not(pmask * 255))

        # owner-gate distance map: Prop2 local mode -> any density; else sparse only
        owner_gate_active = bool(args.owner_gate) and (args.owner_gate_local or density == "low")
        dist_map = None
        if owner_gate_active and person_b is not None and person_b.any():
            inv = np.where(person_b, 0, 255).astype(np.uint8)
            dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        # --- location dedup: refresh occupancy; drop entries whose object has left ---
        for L in alerted_locs:
            lx, ly = L[0], L[1]
            if newdiff[max(0, ly - 5):ly + 5, max(0, lx - 5):lx + 5].max() > 0:
                L[2] = i
        if alerted_locs:
            alerted_locs = [L for L in alerted_locs if (i - L[2]) < args.dedup_clear_s * fps]

        cands = matcher.update(aod_fg, t_now, i)
        for cand in cands:
            center = cand.center()
            moved = cand.cand_id in last_center and np.hypot(
                center[0] - last_center[cand.cand_id][0],
                center[1] - last_center[cand.cand_id][1],
            ) > 12.0
            if moved or cand.cand_id not in static_since:
                static_since[cand.cand_id] = t_now
                hits[cand.cand_id] = 0
            last_center[cand.cand_id] = center
            hits[cand.cand_id] = hits.get(cand.cand_id, 0) + 1

            # owner-gate: record last frame a person was within the object's reach (sparse only)
            if dist_map is not None:
                bb = [int(v) for v in cand.bbox]
                reach = max(args.owner_margin, args.owner_k * ((bb[2] - bb[0]) * (bb[3] - bb[1])) ** 0.5)
                cxg, cyg = int(center[0]), int(center[1])
                if 0 <= cyg < h and 0 <= cxg < w and dist_map[cyg, cxg] < reach:
                    person_near_frame[cand.cand_id] = i

            present = t_now - static_since[cand.cand_id]
            if (
                cand.cand_id not in alerted
                and present >= args.ts_static
                and hits[cand.cand_id] >= min_hits
                and t_now >= args.warmup_s
            ):
                b = [int(v) for v in cand.bbox]
                # ported v1 person_overlap_max: drop a candidate whose bbox is mostly on a
                # person/animate mask (a standing person), at the BLOB level.
                if args.person_overlap_max > 0 and person_b is not None:
                    px1, py1, px2, py2 = max(0, b[0]), max(0, b[1]), b[2], b[3]
                    sub = person_b[py1:py2, px1:px2]
                    if sub.size and float(sub.mean()) >= args.person_overlap_max:
                        continue
                # refine bbox to the FULL object via tight_mask (framediff-based) + gather
                tm = (aod["tight"] > 0).astype(np.uint8)
                if gather_k is not None:
                    tm = cv2.morphologyEx(tm, cv2.MORPH_CLOSE, gather_k)
                cxc, cyc = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                nl, lab, st, _ = cv2.connectedComponentsWithStats(tm, 8)
                lid = int(lab[cyc, cxc]) if (0 <= cyc < h and 0 <= cxc < w) else 0
                if lid == 0:
                    sub = lab[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                    pos = sub[sub > 0]
                    if pos.size:
                        vals, cnts = np.unique(pos, return_counts=True)
                        lid = int(vals[cnts.argmax()])
                if lid > 0 and st[lid, cv2.CC_STAT_AREA] <= args.area_max:
                    x, y, ww, hh = (int(st[lid, k]) for k in (0, 1, 2, 3))
                    b = [x, y, x + ww, y + hh]
                rcx, rcy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2

                # location dedup: same spot still occupied -> same object (cand_id churned) -> skip
                if any(np.hypot(rcx - L[0], rcy - L[1]) < args.dedup_dist for L in alerted_locs):
                    alerted.add(cand.cand_id)
                    continue
                # owner-gate: wait until the object's reach is clear of people. Local mode
                # applies even in crowds, but a timeout prevents deferring forever.
                if owner_gate_active:
                    last_near = person_near_frame.get(cand.cand_id, -10 ** 9)
                    waited = t_now - static_since[cand.cand_id]
                    if (i - last_near) < args.owner_clear_s * fps and waited < args.owner_timeout_s:
                        continue

                # Scene Feature Memory: conservative suppression (moved pre-existing object,
                # or background showing through glare/shadow). Only when very sure.
                if scene_mem is not None:
                    tm_mask = (aod["tight"] > 0).astype(np.uint8)
                    dec = None
                    for md in scene_modes:
                        d = scene_mem.classify(frame, aod["newdiff"], b, mask=tm_mask, mode=md)
                        if d.suppress:
                            dec = d
                            break
                    if dec is not None and dec.suppress:
                        alerted.add(cand.cand_id)
                        suppressed.append({"frame": i, "center": [rcx, rcy], "reason": dec.reason,
                                           "score": round(dec.score, 3)})
                        if args.scene_memory_debug:
                            dbg = draw_alert(frame, b, f"{dec.reason} {dec.score:.2f}")
                            if dec.source_bbox is not None:
                                sx1, sy1, sx2, sy2 = (int(v) for v in dec.source_bbox)
                                cv2.rectangle(dbg, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                            cv2.imwrite(os.path.join(args.outdir, f"scene_suppress_f{i}_c{rcx}_{rcy}.jpg"), dbg)
                        continue

                # adaptive clean_bg heal: if this candidate sits where clean_bg has a BAKED AGENT
                # (car/person detected in clean_bg), then newdiff here = the agent LEFT (revealed
                # ground), not a new object -> absorb into clean_bg + don't alert. Semantic gate
                # (not intensity) so outdoor shadows don't matter. A real object (machine) isn't in
                # clean_bg -> not in the mask -> kept.
                if args.heal_revealed and baked_agent_mask is not None:
                    hx1, hy1, hx2, hy2 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
                    area_b = (hx2 - hx1) * (hy2 - hy1)
                    sub = baked_agent_mask[hy1:hy2, hx1:hx2]
                    if (area_b >= args.heal_min_area and sub.size
                            and float(sub.mean()) >= args.heal_agent_overlap):
                        gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        sfg.clean_bg[hy1:hy2, hx1:hx2] = gray_now[hy1:hy2, hx1:hx2].astype(np.float32)
                        if sfg.clean_bg_color is not None:
                            sfg.clean_bg_color[hy1:hy2, hx1:hx2] = frame[hy1:hy2, hx1:hx2].astype(np.float32)
                        alerted.add(cand.cand_id)
                        healed.append({"frame": i, "center": [rcx, rcy], "area": int(area_b)})
                        continue

                alerted.add(cand.cand_id)
                ev = {
                    "frame": i,
                    "t_s": round(t_now, 2),
                    "center": [rcx, rcy],
                    "wh": [b[2] - b[0], b[3] - b[1]],
                    "present_s": round(present, 2),
                    "cand_id": cand.cand_id,
                }
                events.append(ev)
                alerted_locs.append([rcx, rcy, i])
                alert = draw_alert(frame, b, f"obj#{cand.cand_id} {present:.1f}s")
                cv2.imwrite(
                    os.path.join(args.outdir, f"alert_f{i}_c{rcx}_{rcy}.jpg"),
                    alert,
                )

        if args.save_masks_every and i % args.save_masks_every == 0:
            cv2.imwrite(os.path.join(args.outdir, f"raw_vibe_f{i}.jpg"), raw_vibe)
            cv2.imwrite(os.path.join(args.outdir, f"rtsbs_f{i}.jpg"), rtsbs_mask)
            cv2.imwrite(os.path.join(args.outdir, f"statnew_f{i}.jpg"), stat_new)
            cv2.imwrite(os.path.join(args.outdir, f"newdiff_f{i}.jpg"), newdiff)
            cv2.imwrite(os.path.join(args.outdir, f"fg_f{i}.jpg"), aod_fg)
            cv2.imwrite(os.path.join(args.outdir, f"staticfg_f{i}.jpg"), aod["static_fg"])
            cv2.imwrite(os.path.join(args.outdir, f"moving_f{i}.jpg"), aod["moving"])
            cv2.imwrite(os.path.join(args.outdir, f"keep_f{i}.jpg"), aod["keep"])
            cv2.imwrite(os.path.join(args.outdir, f"stuff_f{i}.jpg"), aod["stuff"])
            cv2.imwrite(os.path.join(args.outdir, f"tight_f{i}.jpg"), aod["tight"])
            cv2.imwrite(os.path.join(args.outdir, f"cleanbg_f{i}.jpg"), sfg.clean_bg.astype(np.uint8))
            cv2.imwrite(
                os.path.join(args.outdir, f"age_f{i}.jpg"),
                cv2.applyColorMap(aod["age"], cv2.COLORMAP_TURBO),
            )
            cv2.imwrite(
                os.path.join(args.outdir, f"semantic_bg_rule_f{i}.jpg"),
                feedback.applied_rule_bg.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(args.outdir, f"semantic_fg_rule_f{i}.jpg"),
                (feedback.applied_rule_fg & ~feedback.applied_rule_bg).astype(np.uint8) * 255,
            )
            cv2.imwrite(
                os.path.join(args.outdir, f"semantic_decision_f{i}.jpg"),
                semantic_decision_preview(feedback.applied_rule_bg, feedback.applied_rule_fg),
            )
            if last_semantic_vis is not None:
                sem_raw = np.clip(last_semantic_vis, 0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(args.outdir, f"semantic_f{i}.png"), sem_raw)
                sem8 = cv2.normalize(sem_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                sem_color = cv2.applyColorMap(sem8, cv2.COLORMAP_TURBO)
                cv2.imwrite(os.path.join(args.outdir, f"semantic_vis_f{i}.jpg"), sem_color)

        i += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"\r[demo2] {i}/{total_label} events={len(events)} cands={len(cands)} "
                f"{i / max(0.001, elapsed):.1f} FPS",
                end="",
                flush=True,
            )

    cap.release()
    elapsed = time.time() - t0
    print(f"\n[demo2] done: {i} frames, {len(events)} events, {i / max(0.001, elapsed):.1f} FPS")

    with open(os.path.join(args.outdir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    if scene_mem is not None:
        with open(os.path.join(args.outdir, "suppressed.json"), "w", encoding="utf-8") as f:
            json.dump(suppressed, f, indent=2, ensure_ascii=False)
        print(f"[scene-memory] suppressed {len(suppressed)} candidate(s) -> suppressed.json")
    if args.heal_revealed:
        with open(os.path.join(args.outdir, "healed.json"), "w", encoding="utf-8") as f:
            json.dump(healed, f, indent=2, ensure_ascii=False)
        print(f"[heal-revealed] absorbed {len(healed)} revealed-background region(s) -> healed.json")

    for ev in events:
        print(
            f"  f{ev['frame']} t={ev['t_s']}s center={ev['center']} "
            f"wh={ev['wh']} present={ev['present_s']}s"
        )
    signal.signal(signal.SIGINT, previous_sigint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
