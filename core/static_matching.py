from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Candidate:
    cand_id: int
    bbox: tuple
    first_seen: float
    last_seen: float
    first_frame: int
    hits: int = 1
    miss: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=240))
    alerted: bool = False

    def center(self):
        return (0.5 * (self.bbox[0] + self.bbox[2]), 0.5 * (self.bbox[1] + self.bbox[3]))


class StaticMatcher:
    def __init__(
        self,
        match_iou: float = 0.3,
        match_dist_px: float = 40.0,
        area_min: int = 60,
        area_max: int = 30000,
        miss_tol: int = 30,
        aspect_max: float = 5.0,
        fill_min: float = 0.18,
        max_cands: int = 100,
        inactive_tol: int = 0,
    ):
        self.match_iou = match_iou
        self.match_dist_px = match_dist_px
        self.area_min = area_min
        self.area_max = area_max
        self.miss_tol = miss_tol
        self.aspect_max = aspect_max
        self.fill_min = fill_min
        self.max_cands = max_cands
        # C.3 (S-TAO re-ID): >0 keeps timed-out candidates in an inactive buffer for this
        # many frames so a blob that reappears (after occlusion) RECOVERS the same track
        # (same cand_id -> static timer + alerted state preserved). 0 = disabled.
        self.inactive_tol = int(inactive_tol)
        self.cands: list[Candidate] = []
        self.inactive: list[Candidate] = []
        self._next = 1

    def _match_blob(self, cand: "Candidate", blobs, used: set) -> tuple[float, int]:
        best, bj = 0.0, -1
        cc = cand.center()
        for j, (box, cen) in enumerate(blobs):
            if j in used:
                continue
            iou = _iou(cand.bbox, box)
            near = np.hypot(cc[0] - cen[0], cc[1] - cen[1]) <= self.match_dist_px
            score = iou if iou > 0 else (0.5 if near else 0.0)
            if score > best:
                best, bj = score, j
        return best, bj

    def _blobs(self, fg_mask: np.ndarray):
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(fg_mask, 8)
        out = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < self.area_min or area > self.area_max:
                continue
            x, y, w, h = stats[k, :4]
            aspect = max(w, h) / max(1.0, min(w, h))
            fill = area / max(1.0, float(w * h))
            if aspect > self.aspect_max or fill < self.fill_min:
                continue
            out.append(((float(x), float(y), float(x + w), float(y + h)), cent[k]))
        return out

    def update(self, fg_mask: np.ndarray, t_now: float, frame_idx: int) -> list[Candidate]:
        blobs = self._blobs(fg_mask)
        used = set()

        # 1. match ACTIVE candidates to current blobs
        for cand in self.cands:
            best, bj = self._match_blob(cand, blobs, used)
            if bj >= 0 and best >= 0.001:
                box = blobs[bj][0]
                cand.bbox = tuple(0.7 * np.array(cand.bbox) + 0.3 * np.array(box))
                cand.last_seen = t_now
                cand.hits += 1
                cand.miss = 0
                cand.history.append((t_now, cand.bbox))
                used.add(bj)
            else:
                cand.miss += 1

        # 2. prune / demote timed-out actives
        if self.inactive_tol > 0:
            # age the pre-existing inactive buffer first; drop the truly stale
            survivors = []
            for c in self.inactive:
                c.miss += 1
                if c.miss <= self.inactive_tol:
                    survivors.append(c)
            self.inactive = survivors
            # demote (don't delete) actives that just timed out -> keep id/history/state
            still_active = []
            for c in self.cands:
                (still_active if c.miss <= self.miss_tol else self.inactive).append(c)
            self.cands = still_active
        else:
            self.cands = [c for c in self.cands if c.miss <= self.miss_tol]

        # 3. unmatched blobs: try to RECOVER an inactive track (reuse cand_id), else new
        for j, (box, _cen) in enumerate(blobs):
            if j in used:
                continue
            rec = None
            if self.inactive_tol > 0 and self.inactive:
                best, bi = 0.0, -1
                for k, c in enumerate(self.inactive):
                    s, _ = self._match_blob(c, [(box, _cen)], set())
                    if s > best:
                        best, bi = s, k
                if bi >= 0 and best >= 0.001:
                    rec = self.inactive.pop(bi)
            if rec is not None:
                rec.bbox = box
                rec.last_seen = t_now
                rec.hits += 1
                rec.miss = 0
                rec.history.append((t_now, box))
                self.cands.append(rec)
            else:
                cand = Candidate(cand_id=self._next, bbox=box, first_seen=t_now, last_seen=t_now, first_frame=frame_idx)
                cand.history.append((t_now, box))
                self.cands.append(cand)
                self._next += 1

        if len(self.cands) > self.max_cands:
            self.cands.sort(key=lambda c: c.first_seen)
            self.cands = self.cands[: self.max_cands]
        if len(self.inactive) > self.max_cands:
            self.inactive.sort(key=lambda c: c.last_seen, reverse=True)
            self.inactive = self.inactive[: self.max_cands]
        return self.cands

