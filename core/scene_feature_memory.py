"""Scene Feature Memory — an optional, conservative suppression layer AFTER the matcher.

Two independent decision modes (both default OFF; only suppress when very sure):

  relocated : a candidate whose appearance strongly matches some OTHER patch of the
              initial clean background, where that source patch has since CHANGED and is
              FAR away -> it is a pre-existing scene object that was moved, not abandoned.
  background: a candidate whose STRUCTURE (edge/orientation, lighting-invariant) still
              matches the clean background at the SAME location -> it is the background
              showing through a glare/shadow, not a new object.

Classical CPU features only (HSV histogram + HOG-lite edge histogram). No SAM/DINO/CLIP.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneMemoryConfig:
    patch: int = 32
    stride: int = 16
    # relocated mode
    match_thresh: float = 0.88        # appearance match (HSV+edge) to call it the same object
    source_change: float = 0.15       # source patch must have changed this fraction (newdiff)
    min_move_dist: float = 40.0       # source must be at least this far from the candidate
    color_w: float = 0.5              # weight of HSV vs edge in the relocated match score
    # background/glare mode
    bg_sim_thresh: float = 0.90       # structure self-similarity to clean_bg at SAME location


@dataclass
class SceneDecision:
    suppress: bool
    reason: str
    score: float
    source_bbox: tuple | None = None


def _hsv_hist(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = None if mask is None else (mask > 0).astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1], m, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
    return h.flatten().astype(np.float32)


def _edge_hist(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """HOG-lite: magnitude-weighted gradient-orientation histogram (9 bins, 0-180)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True) % 180.0
    if mask is not None:
        mag = mag * (mask > 0)
    hist, _ = np.histogram(ang, bins=9, range=(0.0, 180.0), weights=mag)
    s = float(hist.sum())
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else 0.0


class SceneFeatureMemory:
    def __init__(self, clean_bg_color: np.ndarray, cfg: SceneMemoryConfig | None = None):
        self.cfg = cfg or SceneMemoryConfig()
        self.bg = np.clip(clean_bg_color, 0, 255).astype(np.uint8)
        self.H, self.W = self.bg.shape[:2]
        p, s = int(self.cfg.patch), int(self.cfg.stride)
        self.patches: list[tuple] = []  # (bbox, hsv_hist, edge_hist, center)
        for y in range(0, max(1, self.H - p + 1), s):
            for x in range(0, max(1, self.W - p + 1), s):
                reg = self.bg[y:y + p, x:x + p]
                self.patches.append((
                    (x, y, x + p, y + p),
                    _hsv_hist(reg),
                    _edge_hist(reg),
                    (x + p / 2.0, y + p / 2.0),
                ))

    def _clip(self, bbox) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        return max(0, x1), max(0, y1), min(self.W, x2), min(self.H, y2)

    def classify(self, frame, newdiff, bbox, mask=None, mode: str = "relocated") -> SceneDecision:
        x1, y1, x2, y2 = self._clip(bbox)
        if x2 <= x1 or y2 <= y1:
            return SceneDecision(False, "empty_bbox", 0.0)
        crop = frame[y1:y2, x1:x2]
        cmask = mask[y1:y2, x1:x2] if mask is not None and mask.shape[:2] == frame.shape[:2] else None
        if cmask is not None and int((cmask > 0).sum()) < 16:  # mask too small -> use whole bbox
            cmask = None

        if mode == "background":
            return self._classify_background(crop, (x1, y1, x2, y2))
        return self._classify_relocated(crop, cmask, newdiff, (x1, y1, x2, y2))

    def _classify_relocated(self, crop, cmask, newdiff, bbox) -> SceneDecision:
        ch, ce = _hsv_hist(crop, cmask), _edge_hist(crop, cmask)
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        w = float(self.cfg.color_w)
        best, best_p = 0.0, None
        for pb, ph, pe, (pcx, pcy) in self.patches:
            if np.hypot(pcx - cx, pcy - cy) < self.cfg.min_move_dist:
                continue  # too close -> would match the candidate's own spot
            score = w * _cos(ch, ph) + (1.0 - w) * _cos(ce, pe)
            if score > best:
                best, best_p = score, pb
        if best >= self.cfg.match_thresh and best_p is not None:
            sx1, sy1, sx2, sy2 = best_p
            src_changed = float((newdiff[sy1:sy2, sx1:sx2] > 0).mean()) if newdiff is not None else 0.0
            if src_changed >= self.cfg.source_change:
                return SceneDecision(True, "moved_existing", best, best_p)
        return SceneDecision(False, "keep", best, best_p)

    def _classify_background(self, crop, bbox) -> SceneDecision:
        x1, y1, x2, y2 = bbox
        bgreg = self.bg[y1:y2, x1:x2]
        struct = _cos(_edge_hist(crop), _edge_hist(bgreg))  # same structure despite lighting?
        if struct >= self.cfg.bg_sim_thresh:
            return SceneDecision(True, "background_glare", struct, bbox)
        return SceneDecision(False, "keep", struct, bbox)
