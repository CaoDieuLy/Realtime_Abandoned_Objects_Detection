"""demov2 abandoned-object detection — main pipeline and CLI.

End-to-end runner: build a frozen clean background from the warm-up window, then per frame run a
background-subtraction motion gate (pybgs ViBe, numba ControlledViBE, or framediff) and an optional
semantic source (YOLO-seg instance masks, or SegFormer/PSPNet dense maps), feed both into the
``StaticForegroundState`` FSM (clean-background diff + semantic keep-gate), track static blobs with
``StaticMatcher``, and emit an alert when a blob stays static and people-free long enough.

Three modes (``--mode``): ``no-feedback`` (default; pybgs ViBe, the proven config), ``instance-feedback``
(ControlledViBE + one-sided FG-protect), ``dense-feedback`` (SegFormer two-sided RT-SBS). Most knobs
have tuned defaults; per scene usually only ``--bg-learn-seconds`` (warm-up must end before the object
appears) and, for cameras with vehicles, ``--heal-revealed`` need touching. See demov2/README.md.
"""
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
        from core.semantic_classes import build_moving_class_set, build_object_class_set, build_stuff_class_set

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
        self.last_object_score: np.ndarray | None = None
        self.last_stuff_score: np.ndarray | None = None

    def infer(self, frame: np.ndarray) -> np.ndarray:
        semantic, _pred, _moving_prob, object_score, stuff_score = self.infer_map(
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
        self.last_object_score = object_score
        self.last_stuff_score = stuff_score
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
        from core.semantic_classes import build_moving_class_set

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


def resolve_yolo_weights(weights: str, imgsz: int) -> str:
    """Make an exported-backend default safe on a fresh checkout: if an OpenVINO dir / ONNX file is
    requested but missing, auto-export it from the matching ``.pt`` (ultralytics downloads the .pt if
    needed). If the export backend isn't installed, fall back to the ``.pt`` (PyTorch) so the run still
    works — just without the speedup."""
    if os.path.exists(weights):
        return weights
    w = weights.rstrip("/\\")
    if w.endswith("_openvino_model"):
        stem, fmt = w[: -len("_openvino_model")], "openvino"
    elif w.endswith(".onnx"):
        stem, fmt = w[: -len(".onnx")], "onnx"
    else:
        return weights  # a .pt / hub name -> let ultralytics auto-download
    pt = stem + ".pt"
    try:
        from ultralytics import YOLO
        print(f"[semantic] '{weights}' not found -> exporting {fmt} from {pt} (imgsz={imgsz})...", flush=True)
        return str(YOLO(pt).export(format=fmt, imgsz=int(imgsz)))
    except Exception as e:
        print(f"[semantic] WARN: {fmt} export failed ({type(e).__name__}: {e}) -> fallback to {pt}", flush=True)
        return pt


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
        from core.semantic_classes import build_class_set, build_moving_class_set, build_object_class_set

        self.model = YOLO(weights)
        # .names works for both .pt and exported backends (.onnx/openvino embed it in metadata);
        # for .onnx self.model.model is a str path, so don't reach into .model.model.
        raw_names = getattr(self.model, "names", None)
        if not raw_names:
            raw_names = getattr(getattr(self.model, "model", None), "names", {})
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
        self.last_object_score: np.ndarray | None = None

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
        self.last_object_score = (obj * self.SEMANTIC_MAX).astype(np.float32)
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
        description="demov2 abandoned-object detection — clean-background FSM + semantic keep-gate. "
                    "Most runs only need the 'common' options below; everything in 'advanced' has a tuned "
                    "default and is rarely changed."
    )
    common = ap.add_argument_group("common (most runs only set these)")
    advanced = ap.add_argument_group(
        "advanced (tuned defaults — change only if you know why)")
    common.add_argument("--video", default="", help="path to a video file from a FIXED camera (or use --camera-index)")
    common.add_argument("--camera-index", type=int, default=-1,
                    help="live camera index; >=0 uses webcam/camera instead of --video")
    advanced.add_argument("--camera-width", type=int, default=0, help="optional live camera capture width")
    advanced.add_argument("--camera-height", type=int, default=0, help="optional live camera capture height")
    advanced.add_argument("--camera-fps", type=float, default=0.0, help="optional live camera FPS hint")
    common.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_v11_demo2"))
    common.add_argument("--bg-learn-seconds", type=float, default=20.0)
    advanced.add_argument("--sample-step", type=int, default=5)
    common.add_argument("--proc-width", type=int, default=640,
                    help="downscale frames to this width (keep aspect) for the BGS/FSM pipeline; 0=native. "
                         "Heavy per-pixel ops run here -> keep it small (640) for speed + meaningful area thresholds.")
    common.add_argument("--sem-proc-width", type=int, default=960,
                    help="resolution fed to the ONLINE semantic engine (YOLO/SegFormer), DECOUPLED from --proc-width. "
                         "Higher = better person/object detection (its mask is resized back to --proc-width). "
                         "YOLO cost ~ depends on --yolo-imgsz, not on this, so high detail is ~free.")
    common.add_argument("--warmup-s", type=float, default=22.0)
    common.add_argument("--max-frames", type=int, default=0)

    advanced.add_argument("--vibe-samples", type=int, default=30)
    advanced.add_argument("--vibe-threshold", type=int, default=10, help="RT-SBS ViBE base threshold; color uses 4.5x this value")
    advanced.add_argument("--vibe-matches", type=int, default=2)
    advanced.add_argument("--vibe-update-factor", type=int, default=8)
    advanced.add_argument("--vibe-color-mult", type=float, default=4.5)
    advanced.add_argument("--vibe-neighborhood-radius", type=int, default=1)
    advanced.add_argument("--vibe-timeout", type=int, default=150, help="force background update after N frames (0=off)")

    advanced.add_argument("--th-diff", type=int, default=40)
    advanced.add_argument("--clean-update-lr", type=float, default=0.0008)

    common.add_argument("--semantic-mode", choices=["dense", "online-segformer", "online-pspnet", "online-yoloseg", "none"], default="dense")
    common.add_argument("--semantic-dir", default="", help="folder of dense 16-bit semantic maps, one file per semantic frame")
    advanced.add_argument(
        "--semantic-index-mode",
        choices=["strict", "sequential"],
        default="strict",
        help="strict reads exact frame-indexed files like 000005.png; sequential mimics full per-frame folders",
    )
    common.add_argument("--semantic-every", type=int, default=5, help="RT-SBS semantic framerate: use one dense map every N frames")
    advanced.add_argument("--segformer-variant", choices=["b0", "b1"], default="b0")
    advanced.add_argument("--segformer-device", choices=["auto", "cpu", "cuda"], default="auto")
    advanced.add_argument("--segformer-min-conf", type=float, default=0.0)
    advanced.add_argument("--segformer-local-files-only", action="store_true")
    advanced.add_argument("--pspnet-config", default="")
    advanced.add_argument("--pspnet-checkpoint", default="")
    advanced.add_argument("--pspnet-device", default="cuda:0")
    advanced.add_argument("--pspnet-min-conf", type=float, default=0.0)
    common.add_argument("--yolo-weights", default="yolo26s-seg_openvino_model",
                    help="ultralytics instance-seg weights. Default = OpenVINO export of yolo26s-seg (~1.5x faster "
                         "on CPU than .pt, same accuracy); auto-exported from yolo26s-seg.pt on first run, falls "
                         "back to PyTorch if the openvino package is missing. Use yolo26s-seg.pt for plain PyTorch, "
                         "yolo26n-seg.pt to trade person-recall for speed, or a .onnx export.")
    advanced.add_argument("--yolo-imgsz", type=int, default=960)
    advanced.add_argument("--yolo-conf", type=float, default=0.15)
    advanced.add_argument("--yolo-device", default="cpu", help="cpu / 0 / cuda:0")
    advanced.add_argument("--yolo-animate-classes", default="",
                    help="comma COCO names to treat as ANIMATE/reject (default person+vehicles). "
                         "e.g. 'car,bus,truck' to DROP person but keep car")
    advanced.add_argument("--yolo-object-classes", default="",
                    help="comma COCO names to treat as KEEP-object (default bag/umbrella/suitcase/...)")
    advanced.add_argument("--tau-bg", type=int, default=300, help="tau_BG from RT-SBS, 16-bit semantic units")
    advanced.add_argument("--tau-fg", type=int, default=175, help="tau_FG from RT-SBS, 8-bit value multiplied by 256 internally")
    advanced.add_argument("--tau-bg-star", type=int, default=65, help="tau_BG* color reuse threshold")
    advanced.add_argument("--tau-fg-star", type=int, default=115, help="tau_FG* color reuse threshold")
    advanced.add_argument("--modulo-update", type=int, default=256, help="semantic model random update period")
    common.add_argument(
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
    advanced.add_argument("--bgs-backend", choices=["controlled", "pybgs"], default="controlled",
                    help="moving-mask BGS: controlled=ControlledViBE (numba, supports RT-SBS feedback); "
                         "pybgs=C++ ViBe (faster, NO feedback)")
    advanced.add_argument("--semantic-feedback", choices=["on", "off"], default="on",
                    help="on = RT-SBS semantic feedback into ViBE (segment_with_semantics + vibe.update(rtsbs)); "
                         "off = vanilla ViBE update (raw_vibe), semantic still used for the AOD gate")
    advanced.add_argument("--aod-motion-source", choices=["framediff", "rtsbs", "raw-vibe"], default="framediff",
                    help="motion gate separating moving FG from static FG. framediff (v2c, default) does NOT "
                         "eat a just-deposited still object; rtsbs/raw-vibe use the ViBE mask instead")
    advanced.add_argument("--aod-tstatic-s", type=float, default=1.0,
                    help="seconds a pixel must stay static (and inanimate) before it becomes an abandoned candidate")
    advanced.add_argument("--tau-animate", type=float, default=0.3,
                    help="P(person/vehicle/animal) >= this -> treat pixel as animate and reject from AOD. "
                         "NOTE: lowering this can't recover a person YOLO never emitted (empty map); the real "
                         "lever for person recall is the YOLO model/conf, not this downstream threshold.")
    advanced.add_argument("--dilate-animate", type=int, default=2,
                    help="dilate the animate(person/vehicle) reject mask by this many px (demov1 used 12 for person)")
    advanced.add_argument("--tau-object", type=float, default=0.30,
                    help="P(bag/box/bottle/...) >= this -> positive keep evidence for AOD (online only)")
    advanced.add_argument("--tau-stuff", type=float, default=0.50,
                    help="P(wall/floor/water/...) >= this -> scene background, REJECT candidate (dense/segformer only)")
    advanced.add_argument("--stuff-reject", action="store_true",
                    help="enable stuff-class reject (water/wall/floor). WARNING: a coarse model (SegFormer-b0) "
                         "mislabels a small object (umbrella) as 'floor' and would reject it too. Off by default.")
    advanced.add_argument("--no-semantic-gate", action="store_true",
                    help="disable the semantic keep gate on the AOD static-object channel (ablation)")

    # v2c StaticDiffBG features
    advanced.add_argument("--persist-s", type=float, default=2.0,
                    help="persistent-diff >= this many s -> protect clean_bg from absorbing it (don't eat static object)")
    advanced.add_argument("--light-comp", type=int, default=1, help="1=re-baseline clean_bg on high-coverage lighting events")
    advanced.add_argument("--heal-cov", type=float, default=0.15)
    advanced.add_argument("--heal-alpha", type=float, default=1.0)
    advanced.add_argument("--heal-alpha-dark", type=float, default=0.05)
    advanced.add_argument("--dark-s-thresh", type=float, default=15.0)
    advanced.add_argument("--relight", type=int, default=1, help="1=rebuild clean_bg on a global lighting-mode switch (day/night, lights on/off)")
    advanced.add_argument("--relight-dv", type=float, default=20.0,
                    help="cumulative |meanV-ref| over this = lighting diverged (lower catches GRADUAL ramps)")
    advanced.add_argument("--relight-ds", type=float, default=12.0)
    advanced.add_argument("--relight-stable-dv", type=float, default=2.0,
                    help="only rebuild once frame-to-frame |dV| <= this (lighting plateaued) so the rebuild "
                         "captures the FINAL lit scene, not a mid-ramp value; detection pauses while ramping")
    advanced.add_argument("--relearn-s", type=float, default=2.0)

    # v2c runner features: dedup, owner-gate, crowd, bbox refine
    common.add_argument("--heal-revealed", type=int, default=1,
                    help="adaptive clean_bg: run the semantic engine ON clean_bg to find agents (car/person) "
                         "BAKED into it, then let clean_bg ADAPT (fast EMA) only at those pixels so a baked "
                         "agent leaving heals to the real ground (no ghost) within ~1s. Robust to outdoor "
                         "shadows (uses real frames, not inpaint). YOLO/animate modes. A real object isn't in "
                         "the mask -> untouched. Default ON: no-op where nothing is baked (e.g. ABODA = none), "
                         "fixes car-ghost where it is; the self-terminating release bounds any blind spot to "
                         "~--heal-release-s. Set 0 for max security caution.")
    advanced.add_argument("--heal-lr", type=float, default=0.15,
                    help="EMA rate clean_bg adapts at baked-agent pixels each frame (higher = ghost heals faster, "
                         "must heal well under the static threshold so the departure ghost never alerts)")
    advanced.add_argument("--heal-release-s", type=float, default=5.0,
                    help="release a baked pixel back to frozen after this many seconds of (moved-then-still AND "
                         "no agent AND clean_bg settled) -> bounds the blind spot to ~this long after a departure; "
                         "long enough that clean_bg fully heals first so no residual ghost. fps-independent.")
    advanced.add_argument("--dedup-dist", type=float, default=40.0, help="two alerts closer than this (px) = same location")
    advanced.add_argument("--dedup-clear-s", type=float, default=3.0, help="object must leave a spot (newdiff empty) this long before a new alert there")
    advanced.add_argument("--debug-owner", type=int, default=0,
                    help="1=print an [OWNER-DBG] line at each alert: density / owner-gate active / person-overlap / "
                         "distance to nearest detected person / frames since a person was last near. Explains why "
                         "an alert passed the owner-gate (e.g. owner present but YOLO missed them, or scene crowded).")
    advanced.add_argument("--owner-gate", type=int, default=1, help="1=sparse scenes only: delay alert until the object's reach is clear of people")
    advanced.add_argument("--owner-clear-s", type=float, default=3.0,
                    help="defer the alert until the object's reach has been clear of people for this long "
                         "(bridges YOLO person-recall dropouts; also the 'unattended' threshold)")
    advanced.add_argument("--owner-margin", type=int, default=15)
    advanced.add_argument("--owner-k", type=float, default=0.8)
    advanced.add_argument("--crowd-n", type=int, default=10,
                          help="avg >= this many person/vehicle blobs -> 'crowded' -> disable owner-gate "
                               "(in a dense crowd a per-object owner check is unreliable). Raised 6->10 since "
                               "yolo26s-seg detects more people than nano, so the blob count runs higher.")
    advanced.add_argument("--gather-px", type=int, default=5, help="CLOSE tight_mask this many px to merge fragments when refining the alert bbox; 0=off. "
                    "Default 5 = light join (covers a fragmented object) without over-merging a crowd into one giant box (v11).")

    # Prop3 / Prop2 / Prop1
    advanced.add_argument("--motion-to-static", action="store_true",
                    help="Prop3: a candidate must have shown motion (deposited) before going static; "
                         "rejects scene-diff clutter (signage/floor/columns) that never moved")
    advanced.add_argument("--motion-reset-s", type=float, default=1.0,
                    help="Prop3: sustained back-to-bg seconds before clearing the moved latch (survives occlusion gaps)")
    advanced.add_argument("--motion-latch-dilate", type=int, default=4,
                    help="Prop3: dilate the motion latch px so motion near a deposited object counts")
    advanced.add_argument("--owner-gate-local", action="store_true",
                    help="Prop2: apply owner-gate by LOCAL person distance even in crowded scenes "
                         "(not disabled by global density), with --owner-timeout-s to avoid infinite defer")
    advanced.add_argument("--owner-timeout-s", type=float, default=30.0,
                    help="Prop2: max seconds to defer an alert while a person stays in reach (then alert anyway)")
    advanced.add_argument("--person-overlap-max", type=float, default=0.0,
                    help="drop a candidate whose bbox overlaps the person/animate mask >= this fraction "
                         "(a standing person). 0 = off; v1 used 0.25")

    advanced.add_argument("--ts-static", type=float, default=5.0)
    advanced.add_argument("--min-stable-s", type=float, default=1.5)
    advanced.add_argument("--match-iou", type=float, default=0.3)
    advanced.add_argument("--match-dist-px", type=float, default=40.0)
    advanced.add_argument("--area-min", type=int, default=60)
    advanced.add_argument("--area-max", type=int, default=30000)
    advanced.add_argument("--miss-tol-s", type=float, default=1.0)
    advanced.add_argument("--aspect-max", type=float, default=5.0)
    advanced.add_argument("--fill-min", type=float, default=0.18)

    common.add_argument("--save-masks-every", type=int, default=0)
    args = ap.parse_args()

    if not args.video and args.camera_index < 0:
        ap.error("provide a source: --video <path-to-video> or --camera-index <n>")

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
        args.yolo_weights = resolve_yolo_weights(args.yolo_weights, args.yolo_imgsz)
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

    # heal-revealed (adaptive dual-bg): agents (car/person) BAKED into clean_bg ghost when they
    # later leave. Detect them by running the semantic engine ON clean_bg (a parked car is stable
    # in the median, so YOLO finds it), then let clean_bg ADAPT (fast EMA) ONLY at those pixels:
    # while the agent is present clean_bg stays = agent (newdiff~0, animate-gate rejects it); when
    # it leaves, clean_bg becomes the REAL revealed ground within ~1s so the ghost never reaches
    # the static threshold -> no alert. A real object isn't in that mask -> untouched (still detected).
    baked_agent_mask = None
    if args.heal_revealed and online_semantic is not None and clean_bg_color is not None:
        clean_bg_agent_score = online_semantic.infer(np.clip(clean_bg_color, 0, 255).astype(np.uint8))
        if clean_bg_agent_score is not None and clean_bg_agent_score.shape[:2] != (h, w):
            clean_bg_agent_score = cv2.resize(clean_bg_agent_score, (w, h), interpolation=cv2.INTER_NEAREST)
        baked_agents = (clean_bg_agent_score >= args.tau_animate * 65535).astype(np.uint8)
        baked_agent_mask = cv2.dilate(baked_agents, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
        print(f"[heal-revealed] {float(baked_agent_mask.mean()) * 100:.1f}% of clean_bg = baked agents "
              f"-> heal on DEPARTURE then release (lr={args.heal_lr})")
    gone_count = np.zeros((h, w), dtype=np.int32)  # consecutive frames a baked pixel looks departed+settled
    heal_release_frames = max(1, int(args.heal_release_s * fps))
    saw_motion = np.zeros((h, w), dtype=bool)      # baked pixel has shown motion (agent actually moved/left)
    prev_heal_gray = None                          # for framediff in the heal block
    prev_relearning = False                        # for B: recompute baked mask after a relight rebuild

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
            relight_stable_dv=args.relight_stable_dv,
            relearn_s=args.relearn_s,
            motion_to_static=bool(args.motion_to_static),
            motion_reset_s=args.motion_reset_s,
            motion_latch_dilate=args.motion_latch_dilate,
        ),
        clean_bg_color=clean_bg_color,
    )
    semantic_gate_on = (args.semantic_mode != "none") and (not args.no_semantic_gate)
    last_animate_score = initial_semantic if semantic_gate_on else None
    last_object_score: np.ndarray | None = None
    last_stuff_score: np.ndarray | None = None

    matcher = StaticMatcher(
        match_iou=args.match_iou,
        match_dist_px=args.match_dist_px,
        area_min=args.area_min,
        area_max=args.area_max,
        miss_tol=max(1, int(args.miss_tol_s * fps)),
        aspect_max=args.aspect_max,
        fill_min=args.fill_min,
    )


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
    person_seen = np.full((h, w), -(10 ** 9), dtype=np.int64)  # per-pixel last frame a person was detected
    person_seen_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))     # generous stamp (seg masks are tight)
    gather_k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.gather_px, args.gather_px))
                if args.gather_px > 0 else None)

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
                last_animate_score = semantic_map
                last_object_score = _to_proc(getattr(online_semantic, "last_object_score", None) if online_semantic is not None else None)
                last_stuff_score = _to_proc(getattr(online_semantic, "last_stuff_score", None) if online_semantic is not None else None)

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
        # adaptive dual-bg (C: unconditional heal + self-terminating release). Fast-EMA clean_bg
        # toward the current frame at baked-agent pixels EVERY frame: while the agent is parked
        # clean_bg stays = agent (newdiff~0); when it leaves clean_bg becomes the real ground in
        # ~1/lr frames -> the departure ghost dies before the static threshold (kills car-ghost).
        # Then RELEASE a pixel once the agent is gone (no animate, debounced) AND clean_bg has
        # settled -> back to normal frozen behaviour, so NO permanent blind spot. Real objects
        # aren't in the mask -> untouched. (A misdetected static object never goes "agent-free" so
        # it stays healed = a local blind spot there -- the one case this trades off; see docs.)
        if baked_agent_mask is not None and baked_agent_mask.any():
            heal_alpha = float(args.heal_lr)
            agents = baked_agent_mask
            heal_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sfg.clean_bg[agents] = (1.0 - heal_alpha) * sfg.clean_bg[agents] + heal_alpha * heal_gray[agents]
            if sfg.clean_bg_color is not None:
                heal_frame_f = frame.astype(np.float32)
                sfg.clean_bg_color[agents] = (1.0 - heal_alpha) * sfg.clean_bg_color[agents] + heal_alpha * heal_frame_f[agents]
            # release needs BOTH cues to be safe: (motion) the region actually MOVED at some point
            # (robust to YOLO missing a parked car -> a still car never releases), AND (semantic)
            # YOLO confirms no agent there now (robust to a passer-by's motion while the car stays).
            moved_now = (np.abs(heal_gray - prev_heal_gray) >= args.th_diff) if prev_heal_gray is not None else np.zeros((h, w), dtype=bool)
            saw_motion |= agents & moved_now
            if last_animate_score is not None:
                animate_resized = last_animate_score if last_animate_score.shape[:2] == (h, w) else cv2.resize(last_animate_score, (w, h), interpolation=cv2.INTER_NEAREST)
                no_agent = animate_resized < args.tau_animate * 65535
            else:
                no_agent = np.ones((h, w), dtype=bool)
            settled = np.abs(heal_gray - sfg.clean_bg) < args.th_diff
            ok_now = agents & saw_motion & no_agent & settled & (~moved_now)
            gone_count[ok_now] += 1
            gone_count[~ok_now] = 0
            release = agents & (gone_count >= heal_release_frames)   # debounced (~heal_release_s seconds)
            if release.any():
                baked_agent_mask[release] = False
                saw_motion[release] = False
            prev_heal_gray = heal_gray
        aod = sfg.update(
            frame,
            motion_mask=motion_mask,
            animate_score=last_animate_score,
            object_score=last_object_score,
            stuff_score=(last_stuff_score if args.stuff_reject else None),
            protect_mask=protect,
        )
        stat_new = aod["abandoned"]
        newdiff = aod["newdiff"]
        aod_fg = cv2.morphologyEx(stat_new, cv2.MORPH_CLOSE, kernel_close)

        # B: after a relight rebuild, clean_bg is brand-new -> a car parked during the rebuild is
        # freshly baked. Re-run the agent detector on the rebuilt clean_bg so it heals on departure.
        relearning_now = bool(aod.get("relearning", False))
        if (prev_relearning and not relearning_now and args.heal_revealed
                and online_semantic is not None and sfg.clean_bg_color is not None):
            ra = online_semantic.infer(np.clip(sfg.clean_bg_color, 0, 255).astype(np.uint8))
            if ra is not None and ra.shape[:2] != (h, w):
                ra = cv2.resize(ra, (w, h), interpolation=cv2.INTER_NEAREST)
            rbm = cv2.dilate((ra >= args.tau_animate * 65535).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
            baked_agent_mask = rbm if baked_agent_mask is None else (baked_agent_mask | rbm)
            gone_count[:] = 0
            saw_motion[:] = False
            print(f"\n[heal-revealed] relight rebuild -> recomputed baked-agent mask "
                  f"({float(baked_agent_mask.mean()) * 100:.1f}% of clean_bg)", flush=True)
        prev_relearning = relearning_now

        # --- person presence (animate proxy) -> crowd density + owner-gate distance map ---
        person_b = (last_animate_score >= args.tau_animate * 65535) if last_animate_score is not None else None
        if ran_semantic and person_b is not None:
            n_blobs = cv2.connectedComponents(person_b.astype(np.uint8))[0] - 1
            density = crowd.update(max(0, n_blobs))
            # spatial-temporal person-presence memory: stamp this frame where a person is detected
            # (dilated). The owner-gate queries this so a YOLO flicker/miss at the exact alert frame
            # doesn't undo "a person was standing here seconds ago" -> robust to person-recall dropouts.
            if person_b.any():
                person_seen[cv2.dilate(person_b.astype(np.uint8), person_seen_k) > 0] = i

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
                # owner-gate: wait until the object's reach has been clear of people for
                # owner_clear_s. Query the spatial-temporal person-presence map in a reach-radius
                # window (robust to YOLO flicker + cand_id churn). Local mode applies even in
                # crowds, but a timeout prevents deferring forever.
                if owner_gate_active:
                    reach = int(max(args.owner_margin, args.owner_k * ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5))
                    wy0, wy1 = max(0, rcy - reach), min(h, rcy + reach + 1)
                    wx0, wx1 = max(0, rcx - reach), min(w, rcx + reach + 1)
                    last_near = int(person_seen[wy0:wy1, wx0:wx1].max()) if (wy1 > wy0 and wx1 > wx0) else -(10 ** 9)
                    waited = t_now - static_since[cand.cand_id]
                    if (i - last_near) < args.owner_clear_s * fps and waited < args.owner_timeout_s:
                        continue


                if args.debug_owner:
                    rch = int(max(args.owner_margin, args.owner_k * ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5))
                    qy0, qy1 = max(0, rcy - rch), min(h, rcy + rch + 1)
                    qx0, qx1 = max(0, rcx - rch), min(w, rcx + rch + 1)
                    ln = int(person_seen[qy0:qy1, qx0:qx1].max()) if (qy1 > qy0 and qx1 > qx0) else -(10 ** 9)
                    since = (i - ln) if ln > -10 ** 8 else None
                    print(f"[OWNER-DBG] f{i} c({rcx},{rcy}) density={density} gate_active={owner_gate_active} "
                          f"reach={rch}px since_person_near(in reach)={since}f "
                          f"(owner_clear={args.owner_clear_s}s={args.owner_clear_s*fps:.0f}f)", flush=True)
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
            cv2.imwrite(os.path.join(args.outdir, f"frame_f{i}.jpg"), frame)  # actual scene (compare vs cleanbg)
            if sfg.clean_bg_color is not None:
                cv2.imwrite(os.path.join(args.outdir, f"cleanbg_color_f{i}.jpg"), sfg.clean_bg_color.astype(np.uint8))
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

    for ev in events:
        print(
            f"  f{ev['frame']} t={ev['t_s']}s center={ev['center']} "
            f"wh={ev['wh']} present={ev['present_s']}s"
        )
    signal.signal(signal.SIGINT, previous_sigint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
