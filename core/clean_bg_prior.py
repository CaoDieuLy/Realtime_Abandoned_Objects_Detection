"""Warm-up clean background: the fixed long-term reference for abandoned-object detection.

The clean background is the per-pixel median of the first ``learn_seconds`` of video
(camera warm-up, before any object is dropped). A frozen reference means a newly
deposited object always differs from it, so it is never silently absorbed the way an
adaptive BGS model would absorb a static object.
"""
from __future__ import annotations

import os
import warnings

import cv2
import numpy as np


def resize_to_width(frame: np.ndarray, proc_width: int) -> np.ndarray:
    """Downscale a frame to ``proc_width`` (keep aspect). No-op if proc_width<=0 or already
    narrower. Used so high-res cameras (e.g. 2560x1920) run at a sane processing resolution:
    per-pixel ops (ViBe, diff, morphology) get ~(orig/proc)^2 faster, and pixel-area
    thresholds (area_min/max) become meaningful again."""
    if proc_width and proc_width > 0 and frame.shape[1] > proc_width:
        nh = int(round(frame.shape[0] * proc_width / float(frame.shape[1])))
        return cv2.resize(frame, (proc_width, nh), interpolation=cv2.INTER_AREA)
    return frame


def build_warmup_background(
    video_path: str,
    learn_seconds: float,
    sample_step: int = 5,
    proc_width: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], float, int]:
    """Return (clean_bg_gray, warmup_frames_bgr, fps, total_frames).

    clean_bg_gray = median of sampled grayscale warm-up frames (float32).
    warmup_frames_bgr is reused to seed ViBE and to build the colour clean background.
    Frames are downscaled to ``proc_width`` (if >0) so the whole pipeline runs at that size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    last = int(max(1.0, learn_seconds) * fps)
    last = min(last, total) if total > 0 else last

    frames: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    fi = 0
    while fi < last:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to_width(frame, proc_width)
        frames.append(frame.copy())
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += max(1, int(sample_step))
    cap.release()

    if not grays:
        raise RuntimeError("No warmup frames were read")

    clean_bg = np.median(np.stack(grays, axis=0), axis=0).astype(np.float32)
    return clean_bg, frames, fps, total


def open_camera_capture(
    camera_index: int,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
) -> cv2.VideoCapture:
    """Open a live camera source with optional capture properties."""
    api = cv2.CAP_DSHOW if os.name == "nt" else 0
    cap = cv2.VideoCapture(int(camera_index), api)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open camera index: {camera_index}")
    return cap


def build_camera_warmup_background(
    camera_index: int,
    learn_seconds: float,
    sample_step: int = 5,
    width: int = 0,
    height: int = 0,
    fps_hint: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray], float, int]:
    """Return a clean background from a live camera warm-up window.

    The camera is read sequentially because live sources cannot seek. The first
    few seconds should be as empty/stable as possible.
    """
    cap = open_camera_capture(camera_index, width=width, height=height, fps=fps_hint)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
    total_to_read = max(1, int(max(1.0, learn_seconds) * fps))
    step = max(1, int(sample_step))

    frames: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    idx = 0
    while idx < total_to_read:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame.copy())
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()

    if not grays:
        raise RuntimeError("No camera warmup frames were read")

    clean_bg = np.median(np.stack(grays, axis=0), axis=0).astype(np.float32)
    return clean_bg, frames, fps, 0


def build_person_aware_clean_bg(
    frames_bgr: list[np.ndarray],
    animate_of,
    tau16: float,
    dilate_px: int = 8,
    max_frames: int = 40,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-pixel median of warm-up frames that IGNORES agent (person/vehicle) pixels.

    A person standing through warm-up otherwise gets baked into the clean background;
    when they later leave, the vacated spot differs from clean_bg -> a "ghost" false
    abandoned alert. Here, pixels where ``animate_of(frame) >= tau16`` (dilated) are
    excluded per frame so the median sees the floor behind the person instead.

    ``animate_of(frame) -> HxW float`` is the animate score in [0, SEMANTIC_MAX]
    (e.g. ``OnlineYoloSeg.infer``). Pixels that are animate in EVERY used frame have no
    clean sample -> fall back to the plain (unmasked) median there.

    Returns ``(clean_bg_gray, clean_bg_color, animate_coverage)`` (float32, float32, float).
    """
    if not frames_bgr:
        raise RuntimeError("build_person_aware_clean_bg: no warmup frames")
    step = max(1, len(frames_bgr) // max(1, int(max_frames)))
    sub = frames_bgr[::step]
    kern = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        if dilate_px > 0
        else None
    )
    ign = []
    for f in sub:
        m = (np.asarray(animate_of(f)) >= tau16).astype(np.uint8)
        if kern is not None:
            m = cv2.dilate(m, kern)
        ign.append(m.astype(bool))

    color = np.stack([f.astype(np.float32) for f in sub], axis=0)  # (n,H,W,3)
    igns = np.stack(ign, axis=0)                                   # (n,H,W) bool
    color_plain = np.median(color, axis=0)
    color_nan = np.where(igns[..., None], np.nan, color)
    with warnings.catch_warnings():  # all-NaN columns (agent in every frame) handled below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        color_med = np.nanmedian(color_nan, axis=0)
    # Pixels where the agent (e.g. a parked car) covered EVERY warm-up frame have no clean
    # sample. Don't restore the agent (that bakes it in -> ghost when it leaves) — INPAINT the
    # hole from the surrounding background so clean_bg shows the ground behind it.
    all_agent = igns.all(axis=0)
    if all_agent.any():
        base = np.where(all_agent[..., None], 0, color_med)
        base = np.clip(np.nan_to_num(base), 0, 255).astype(np.uint8)
        inpainted = cv2.inpaint(base, all_agent.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        color_med[all_agent] = inpainted[all_agent].astype(np.float32)
    color_med = color_med.astype(np.float32)
    gray = cv2.cvtColor(np.clip(color_med, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, color_med, float(igns.mean())
