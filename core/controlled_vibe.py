"""ControlledViBE — a ViBe background-subtraction model that can accept an EXTERNALLY corrected
foreground mask (RT-SBS semantic feedback) instead of only its own decision, and update its
per-pixel sample buffers from that corrected mask. Segmentation is numba-accelerated
(``_vibe_segment_njit``). Used by the instance-feedback / dense-feedback modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # numba optional -> fall back to the cv2 per-sample loop
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True, parallel=True, fastmath=False)
    def _vibe_segment_njit(samples, frame, threshold, matching_number):
        """Per-pixel ViBE match with early-exit; bit-identical decision to the cv2 loop.

        fg = (#samples with L1 colour distance <= threshold) < matching_number.
        """
        s_n, h, w, _ = samples.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for y in prange(h):
            for x in range(w):
                b = np.int32(frame[y, x, 0])
                g = np.int32(frame[y, x, 1])
                r = np.int32(frame[y, x, 2])
                cnt = 0
                for s in range(s_n):
                    d = (abs(np.int32(samples[s, y, x, 0]) - b)
                         + abs(np.int32(samples[s, y, x, 1]) - g)
                         + abs(np.int32(samples[s, y, x, 2]) - r))
                    if d <= threshold:
                        cnt += 1
                        if cnt >= matching_number:
                            break
                if cnt < matching_number:
                    out[y, x] = 255
        return out


@dataclass(frozen=True)
class ViBEConfig:
    samples: int = 30
    matching_threshold: int = 10
    matching_number: int = 2
    update_factor: int = 8
    color_threshold_multiplier: float = 4.5
    neighborhood_radius: int = 1
    init_noise_low: int = -20
    init_noise_high: int = 20
    seed: int = 12345
    foreground_timeout: int = 0


class ControlledViBE:
    """RT-SBS-compatible CPU ViBE with an explicit update mask.

    pybgs.ViBe exposes apply(), which segments and updates internally. RT-SBS needs
    the opposite shape: segment first, let semantic rules correct the mask, then
    update only where the corrected mask says background.
    """

    def __init__(self, init_frames: Iterable[np.ndarray], cfg: ViBEConfig | None = None):
        self.cfg = cfg or ViBEConfig()
        frames = [self._as_bgr_uint8(f) for f in init_frames]
        if not frames:
            raise ValueError("ControlledViBE needs at least one init frame")

        h, w = frames[0].shape[:2]
        for f in frames:
            if f.shape[:2] != (h, w):
                raise ValueError("All init frames must have the same size")

        self.height = h
        self.width = w
        self.rng = np.random.default_rng(self.cfg.seed)
        # uint8 sample buffer (values are always in 0..255): lets segmentation use
        # OpenCV's C++ absdiff directly instead of a slow per-sample numpy loop.
        self.samples = np.empty((self.cfg.samples, h, w, 3), dtype=np.uint8)
        self._sum_channels = np.ones((1, 3), dtype=np.float32)
        self.fg_age = np.zeros((h, w), dtype=np.int32)
        self._init_history(frames[0])
        self._init_update_schedule()

    def _init_history(self, image: np.ndarray) -> None:
        """Match RT-SBS ViBeGPU init: first matches are exact, the rest are noisy."""
        exact = min(self.cfg.matching_number, self.cfg.samples)
        self.samples[:exact] = image

        base = image.astype(np.int16)
        for si in range(exact, self.cfg.samples):
            noise = self.rng.integers(
                self.cfg.init_noise_low,
                self.cfg.init_noise_high,
                size=image.shape,
                dtype=np.int16,
            )
            self.samples[si] = np.clip(base + noise, 0, 255).astype(np.uint8)

    def _init_update_schedule(self) -> None:
        total = self.height * self.width
        self.update_vector = self.rng.random(total) <= (1.0 / float(self.cfg.update_factor))
        amount = int(self.update_vector.sum())
        radius = int(self.cfg.neighborhood_radius)
        self.neighbor_row = self.rng.integers(-radius, radius + 1, size=amount, dtype=np.int16)
        self.neighbor_col = self.rng.integers(-radius, radius + 1, size=amount, dtype=np.int16)
        self.position = self.rng.integers(0, self.cfg.samples, size=amount, dtype=np.int16)

    @staticmethod
    def _as_bgr_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("ViBE expects BGR uint8 frames")
        return frame

    def segmentation(self, frame: np.ndarray) -> np.ndarray:
        frame = self._as_bgr_uint8(frame)
        threshold = float(self.cfg.matching_threshold) * float(self.cfg.color_threshold_multiplier)

        if _HAS_NUMBA:
            # Single-pass per-pixel match with early-exit (JIT, parallel). ~10x the
            # cv2 per-sample loop, and produces the identical fg mask.
            return _vibe_segment_njit(
                self.samples, np.ascontiguousarray(frame), threshold, int(self.cfg.matching_number)
            )

        # Fallback (no numba): per-sample L1 colour distance via OpenCV (C++). The
        # original loop's early-stop only caps the match counter and never changes
        # the final ``count < matching_number`` test, so this is bit-identical.
        match = np.zeros((self.height, self.width), dtype=np.int32)
        for s in range(self.samples.shape[0]):
            diff = cv2.absdiff(self.samples[s], frame)
            l1 = cv2.transform(diff, self._sum_channels)
            match += l1 <= threshold

        fg = match < self.cfg.matching_number
        return fg.astype(np.uint8) * 255

    def update(self, frame: np.ndarray, foreground_mask: np.ndarray) -> None:
        """Update only background pixels from the RT-SBS corrected foreground mask."""
        frame = self._as_bgr_uint8(frame)
        if self.position.size == 0:
            return

        is_fg = foreground_mask > 0
        self.fg_age = np.where(is_fg, self.fg_age + 1, 0)

        # Force update for timed-out foreground pixels
        if self.cfg.foreground_timeout > 0:
            timeout_mask = self.fg_age >= self.cfg.foreground_timeout
            ty, tx = np.where(timeout_mask)
            if ty.size > 0:
                # Force replace a random sample with the current frame color
                pos = self.rng.integers(0, self.cfg.samples, size=ty.size).astype(np.intp)
                self.samples[pos, ty, tx] = frame[ty, tx]
                self.fg_age[timeout_mask] = 0
                is_fg[timeout_mask] = False  # treat them as background for standard updates too

        r = int(self.rng.integers(0, self.update_vector.size))
        r2 = self.rng.integers(0, self.position.size, size=3)
        update_image = np.roll(self.update_vector, r).reshape(self.height, self.width)
        self.neighbor_row = np.roll(self.neighbor_row, int(r2[0]))
        self.neighbor_col = np.roll(self.neighbor_col, int(r2[1]))
        self.position = np.roll(self.position, int(r2[2]))

        update_mask = update_image & ~is_fg
        ys, xs = np.where(update_mask)
        num_updates = ys.size
        if num_updates == 0:
            return

        pos_self = self.position[:num_updates].astype(np.intp)
        self.samples[pos_self, ys, xs] = frame[ys, xs]

        dst_y = np.clip(ys + self.neighbor_row[:num_updates], 0, self.height - 1).astype(np.intp)
        dst_x = np.clip(xs + self.neighbor_col[:num_updates], 0, self.width - 1).astype(np.intp)
        pos_neighbor = np.roll(self.position, int(r2[0]))[:num_updates].astype(np.intp)
        self.samples[pos_neighbor, dst_y, dst_x] = frame[ys, xs]
