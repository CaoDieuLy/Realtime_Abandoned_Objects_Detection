"""Warm-up clean background: the fixed long-term reference for abandoned-object detection.

The clean background is the per-pixel median of the first ``learn_seconds`` of video
(camera warm-up, before any object is dropped). A frozen reference means a newly
deposited object always differs from it, so it is never silently absorbed the way an
adaptive BGS model would absorb a static object.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_warmup_background(
    video_path: str,
    learn_seconds: float,
    sample_step: int = 5,
) -> tuple[np.ndarray, list[np.ndarray], float, int]:
    """Return (clean_bg_gray, warmup_frames_bgr, fps, total_frames).

    clean_bg_gray = median of sampled grayscale warm-up frames (float32).
    warmup_frames_bgr is reused to seed ViBE and to build the colour clean background.
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
        frames.append(frame.copy())
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += max(1, int(sample_step))
    cap.release()

    if not grays:
        raise RuntimeError("No warmup frames were read")

    clean_bg = np.median(np.stack(grays, axis=0), axis=0).astype(np.float32)
    return clean_bg, frames, fps, total
