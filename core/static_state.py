"""Per-pixel static-foreground state machine for abandoned-object detection (AOD).

The detector compares every frame against a frozen *clean background* (the per-pixel median
of the warm-up window, before any object is dropped). A pixel that (a) differs from the clean
background, (b) is not currently moving, and (c) is not a person/vehicle ("animate") becomes a
candidate; once it has stayed that way for ``t_static_s`` seconds it is reported as *abandoned*.

Semantic-probability maps (animate / object / stuff) come in as float arrays scaled to
``0 .. SEMANTIC_MAX`` (see ``core.semantic_classes``); we call these "score" maps. Thresholds with a
``_score_thresh`` suffix live on the same 0..SEMANTIC_MAX scale.

Terms used throughout:
  * ``animate``  - person / vehicle / animal: a self-moving actor that must NOT be flagged as an
                   abandoned object.
  * ``object``   - bag / suitcase / umbrella / ...: positive evidence that a static blob IS a
                   leave-behind (overrides a weak animate reject).
  * ``stuff``    - wall / floor / water / ...: scene background; reject as a candidate and let the
                   clean background absorb it (dense models only).
  * ``newdiff``  - mask of pixels that differ from the clean background.
  * ``static_fg``- newdiff AND not moving (a still foreground region).
  * ``tight``    - newdiff AND not framediff: the FULL still object, used for the bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.semantic_classes import SEMANTIC_MAX


@dataclass
class StaticStateConfig:
    th_diff: int = 40                 # |frame - clean_bg| >= this (8-bit luma) = "different"
    update_lr: float = 0.0008         # clean_bg slow-update learning rate on non-object pixels
    t_static_s: float = 1.0           # seconds a pixel must stay static (debounce before abandoned)
    fps: float = 30.0
    # Semantic-gate thresholds, as probabilities in 0..1 (scaled to SEMANTIC_MAX internally):
    tau_animate: float = 0.25         # P(person/vehicle) >= tau -> animate (NOT abandoned)
    tau_object: float = 0.30          # P(bag/umbrella/...) >= tau -> positive keep evidence
    tau_stuff: float = 0.50           # P(wall/floor/water/...) >= tau -> scene background, reject (dense models)
    dilate_motion: int = 1            # radius (px) to dilate an external moving mask
    dilate_animate: int = 2           # radius (px) to dilate the animate reject mask
    moving_decay: float = 3.0         # static_age decays this many seconds-worth per frame under motion
    # --- v2c StaticDiffBG features ---
    motion_source: str = "framediff"  # framediff | external (rtsbs/raw-vibe mask passed in)
    persist_s: float = 2.0            # persistent-diff >= this -> protect clean_bg from absorbing it
    light_comp: bool = True           # re-baseline clean_bg when global lighting coverage is high
    heal_cov: float = 0.15            # light-comp triggers when this fraction of pixels differ at once
    heal_alpha: float = 1.0           # light-comp absorb rate (1.0 = replace) in normal light
    heal_alpha_dark: float = 0.05     # light-comp absorb rate when the scene is very desaturated (night/IR)
    dark_s_thresh: float = 15.0       # mean HSV-saturation below this = "dark/IR" -> use heal_alpha_dark
    relight: bool = True              # rebuild clean_bg on a global lighting-mode switch
    relight_dv: float = 20.0          # |cur_value - ref_value| over this (cumulative since last rebuild) = diverged
    relight_ds: float = 12.0          # |cur_sat - ref_sat| over this = diverged (saturation channel)
    relight_stable_dv: float = 2.0    # only rebuild once the new lighting has STABILISED (frame-to-frame
                                      #   |d value| <= this), so we re-median the final lit scene, not a mid-ramp
                                      #   value. Detection is paused (not rebuilt) while still ramping.
    relight_hold: int = 15            # frames the (stable) divergence must hold before a rebuild starts
    relearn_s: float = 2.0            # seconds of frames to median into the rebuilt clean_bg
    motion_to_static: bool = False    # a candidate must have had motion (deposited) before going static
    motion_reset_s: float = 1.0       # back-to-bg must persist this long before clearing the moved latch
    motion_latch_dilate: int = 4      # dilate the motion latch so motion near a deposited object counts


class StaticForegroundState:
    """Per-pixel static-foreground state machine for abandoned-object detection.

    Combines demo2's semantic keep-gate (animate/object) with demo1 v2c's StaticDiffBG:
      - clean_bg diff -> ``newdiff`` (object vs the clean warm-up background)
      - moving gate (framediff by default, or an external rtsbs/raw-vibe mask)
      - ``static_fg`` = newdiff AND NOT moving; aged + semantic-gated -> abandoned
      - ``tight_mask`` = newdiff AND NOT framediff (keeps the FULL static object for the bbox,
        because framediff never fires on a just-deposited still object, unlike ViBe which "eats"
        it until absorbed)
      - persist-protect, light-comp heal and re-light (day/night, lights on/off) so the
        conservative clean_bg neither drifts nor absorbs a real static object.
    """

    def __init__(
        self,
        clean_bg_gray: np.ndarray,
        cfg: StaticStateConfig | None = None,
        clean_bg_color: np.ndarray | None = None,
    ):
        self.cfg = cfg or StaticStateConfig()
        self.clean_bg = clean_bg_gray.astype(np.float32)               # frozen grayscale reference
        self.clean_bg_color = clean_bg_color.astype(np.float32) if clean_bg_color is not None else None
        h, w = self.clean_bg.shape[:2]
        self.height, self.width = h, w
        self.static_age = np.zeros((h, w), dtype=np.float32)           # seconds each pixel has been static
        self.dt = 1.0 / max(1e-3, float(self.cfg.fps))                 # seconds per frame
        # Semantic thresholds on the 0..SEMANTIC_MAX score scale (prob * SEMANTIC_MAX):
        self.animate_score_thresh = float(self.cfg.tau_animate) * float(SEMANTIC_MAX)
        self.object_score_thresh = float(self.cfg.tau_object) * float(SEMANTIC_MAX)
        self.stuff_score_thresh = float(self.cfg.tau_stuff) * float(SEMANTIC_MAX)
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kmotion = self._odd_kernel(self.cfg.dilate_motion)
        self._kanimate = self._odd_kernel(self.cfg.dilate_animate)
        self.persist_thresh = max(1, int(self.cfg.persist_s * self.cfg.fps))
        self.relearn_frames = max(5, int(self.cfg.relearn_s * self.cfg.fps))
        self._persist = np.zeros((h, w), dtype=np.float32)             # consecutive frames a pixel stayed different
        # motion-to-static latch: pixels that have shown motion while continuously different from the
        # clean background (a deposited object); cleared only after a SUSTAINED return to background
        # (survives brief occlusion/reveal gaps).
        self._moved = np.zeros((h, w), dtype=bool)
        self._bg_run = np.zeros((h, w), dtype=np.float32)              # consecutive frames a pixel matched bg
        self.motion_reset_frames = max(1, int(self.cfg.motion_reset_s * self.cfg.fps))
        self._klatch = self._odd_kernel(self.cfg.motion_latch_dilate)
        self._prev_gray: np.ndarray | None = None                     # previous frame (for framediff)
        self.tight_mask = np.zeros((h, w), dtype=np.uint8)
        # relight state (global lighting-mode switch): ref_value/ref_sat are the clean_bg's mean HSV
        # Value/Saturation at the last rebuild; the scene is "diverged" when it drifts far from them.
        self.ref_value: float | None = None
        self.ref_sat: float | None = None
        self._prev_value: float | None = None   # previous-frame mean Value, to detect when lighting stabilised
        self._stable_diverged_count = 0          # consecutive STABLE+diverged frames (rebuild after relight_hold)
        self._relearning = False                 # currently re-medianing a new clean_bg
        self._relearn_left = 0
        self._relearn_buf_gray: list[np.ndarray] = []
        self._relearn_buf_color: list[np.ndarray] = []
        self.relearning = False                  # public flag: detection paused this frame
        # last outputs, kept for inspection / debugging
        self.fgbg = np.zeros((h, w), dtype=bool)        # "different from clean_bg" mask (a.k.a. newdiff>0)
        self.moving = np.zeros((h, w), dtype=bool)
        self.static_fg = np.zeros((h, w), dtype=bool)
        self.keep = np.ones((h, w), dtype=bool)

    @staticmethod
    def _odd_kernel(radius: int):
        """Elliptical structuring element of the given pixel radius (None if radius == 0)."""
        r = max(0, int(radius))
        if r == 0:
            return None
        k = 2 * r + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ---- re-light (global lighting-mode switch: day/night, lights on/off) ----
    def _relight_step(self, frame: np.ndarray, gray: np.ndarray, protect_mask: np.ndarray | None = None) -> bool:
        """Detect a global lighting switch and rebuild clean_bg for the new lighting.

        Returns True while a lighting transition is in progress (the caller pauses detection).
        A rebuild only starts once the new lighting has STABILISED (frame-to-frame Value change
        small), so we re-median the final lit scene instead of a mid-ramp / dark value.
        """
        if self.ref_value is None:                       # first frame: seed the reference from clean_bg
            self.ref_value = float(self.clean_bg.mean())
            self.ref_sat = float(cv2.cvtColor(self.clean_bg_color.astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].mean())
        if self._relearning:                             # collecting frames for the new clean_bg
            self._relearn_buf_gray.append(gray.copy())
            self._relearn_buf_color.append(frame.astype(np.float32))
            self._relearn_left -= 1
            if self._relearn_left <= 0:
                self._finish_relearn(protect_mask)
            self._prev_gray = gray.copy()
            return True
        cur_value = float(gray.mean())
        cur_sat = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())
        diverged = (abs(cur_value - self.ref_value) > self.cfg.relight_dv) \
            or (abs(cur_sat - self.ref_sat) > self.cfg.relight_ds)
        # frame-to-frame stability: while the lights are still RAMPING, |d value| is large -> wait;
        # only when the new level has plateaued do we re-median it (avoids locking in a mid-ramp /
        # dark value the way a fixed hold-count would).
        stable = (self._prev_value is None) or (abs(cur_value - self._prev_value) <= self.cfg.relight_stable_dv)
        self._prev_value = cur_value
        if diverged:
            if stable:
                self._stable_diverged_count += 1         # count consecutive STABLE diverged frames
            else:
                self._stable_diverged_count = 0          # still ramping -> restart the plateau hold
            if self._stable_diverged_count >= self.cfg.relight_hold:
                self._relearning = True
                self._relearn_left = self.relearn_frames - 1
                self._stable_diverged_count = 0
                self._relearn_buf_gray = [gray.copy()]
                self._relearn_buf_color = [frame.astype(np.float32)]
                print(f"\n[RELIGHT] lighting switch (dV={abs(cur_value-self.ref_value):.0f} "
                      f"dS={abs(cur_sat-self.ref_sat):.0f}, stabilised) -> rebuilding clean_bg "
                      f"{self.relearn_frames}f...", flush=True)
            self._prev_gray = gray.copy()
            return True                                  # pause detection throughout the lighting transition
        self._stable_diverged_count = 0
        return False

    def _finish_relearn(self, protect_mask: np.ndarray | None = None) -> None:
        """Replace clean_bg with the median of the collected transition frames (the new lit scene)."""
        new_gray = np.median(np.stack(self._relearn_buf_gray), axis=0).astype(np.float32)
        new_color = np.median(np.stack(self._relearn_buf_color), axis=0).astype(np.float32)
        if protect_mask is not None:                     # keep already-alerted object pixels frozen
            new_gray[protect_mask] = self.clean_bg[protect_mask]
            if self.clean_bg_color is not None:
                new_color[protect_mask] = self.clean_bg_color[protect_mask]
        self.clean_bg = new_gray
        self.clean_bg_color = new_color
        self.ref_value = float(new_gray.mean())
        self.ref_sat = float(cv2.cvtColor(new_color.astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].mean())
        self._relearning = False
        self._relearn_buf_gray = []
        self._relearn_buf_color = []
        print(f"[RELIGHT] done: new clean_bg (V={self.ref_value:.0f} S={self.ref_sat:.0f})", flush=True)

    def _zeros_result(self) -> dict[str, np.ndarray]:
        """All-zero output dict, returned while detection is paused (clean_bg rebuild)."""
        zeros = np.zeros((self.height, self.width), dtype=np.uint8)
        self.tight_mask = zeros.copy()
        return {
            "abandoned": zeros.copy(), "static_fg": zeros.copy(), "moving": zeros.copy(),
            "newdiff": zeros.copy(), "keep": zeros.copy(), "stuff": zeros.copy(), "age": zeros.copy(),
            "tight": zeros.copy(), "relearning": True,
        }

    def update(
        self,
        frame_bgr: np.ndarray,
        motion_mask: np.ndarray | None = None,
        animate_score: np.ndarray | None = None,   # P(person/vehicle) * SEMANTIC_MAX, HxW float
        object_score: np.ndarray | None = None,    # P(bag/umbrella/...) * SEMANTIC_MAX
        stuff_score: np.ndarray | None = None,     # P(wall/floor/...) * SEMANTIC_MAX (dense models)
        protect_mask: np.ndarray | None = None,    # pixels never absorbed into clean_bg (already alerted)
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # re-light: pause detection while rebuilding clean_bg for a new lighting mode
        self.relearning = False
        if cfg.relight and self.clean_bg_color is not None:
            if self._relight_step(frame_bgr, gray, protect_mask):
                self.relearning = True
                return self._zeros_result()

        # scene "stuff"/background (wall/floor/water/...) -> reject as candidate AND let
        # clean_bg absorb it (so a puddle/wall that differs from the warm-up bg stops firing).
        if stuff_score is not None:
            is_stuff = stuff_score >= self.stuff_score_thresh
        else:
            is_stuff = np.zeros((self.height, self.width), dtype=bool)

        protect_persist = (self._persist >= self.persist_thresh) & (~is_stuff)

        diff = cv2.absdiff(gray, self.clean_bg)
        newdiff = (diff >= cfg.th_diff).astype(np.uint8)

        # light-comp: a global lighting event changes many regions at once (high coverage); absorb
        # them into clean_bg (re-baseline), but never the persistent/protected object.
        if cfg.light_comp and float(newdiff.mean()) > cfg.heal_cov:
            absorb = (newdiff > 0) & (~protect_persist)
            if protect_mask is not None:
                absorb &= ~protect_mask
            alpha = cfg.heal_alpha_dark if (self.ref_sat is not None and self.ref_sat < cfg.dark_s_thresh) else cfg.heal_alpha
            self.clean_bg[absorb] = (1.0 - alpha) * self.clean_bg[absorb] + alpha * gray[absorb]
            if self.clean_bg_color is not None:
                frame_f = frame_bgr.astype(np.float32)
                self.clean_bg_color[absorb] = (1.0 - alpha) * self.clean_bg_color[absorb] + alpha * frame_f[absorb]
            diff = cv2.absdiff(gray, self.clean_bg)
            newdiff = (diff >= cfg.th_diff).astype(np.uint8)
        newdiff = cv2.morphologyEx(newdiff, cv2.MORPH_OPEN, self.kernel3)
        is_diff = newdiff > 0                        # pixel differs from the clean background

        # persistence counter (still-different region grows; back-to-background resets)
        self._persist = np.where(is_diff, self._persist + 1, 0.0)

        # frame difference (instantaneous motion) -> never eats a still object; feeds tight_mask
        if self._prev_gray is None:
            frame_diff = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            frame_diff = (cv2.absdiff(gray, self._prev_gray) >= cfg.th_diff).astype(np.uint8)
        self._prev_gray = gray.copy()
        frame_diff_dil = cv2.dilate(frame_diff, self.kernel3)

        # moving gate: framediff by default, or an external (ViBe/rtsbs) motion mask
        if cfg.motion_source == "framediff" or motion_mask is None:
            moving = frame_diff_dil
        else:
            moving = (motion_mask > 0).astype(np.uint8)
            if self._kmotion is not None:
                moving = cv2.dilate(moving, self._kmotion)
        is_moving = moving > 0

        static_fg = is_diff & ~is_moving             # still foreground = different AND not moving

        # motion-to-static latch: a deposited object shows motion (carried in) while it differs from
        # clean_bg, then goes static. Pixels that became different WITHOUT any local motion
        # (lighting/reflection/scene diffs) never latch -> rejected. The latch is dilated (motion
        # near the object counts) and only cleared after a SUSTAINED return to background, so an
        # occlusion/reveal gap doesn't erase it.
        moved_now = is_moving
        if self._klatch is not None:
            moved_now = cv2.dilate(is_moving.astype(np.uint8), self._klatch) > 0
        self._moved |= moved_now
        self._bg_run = np.where(is_diff, 0.0, self._bg_run + 1.0)
        self._moved[self._bg_run >= self.motion_reset_frames] = False

        # semantic keep gate (animate -> reject person/vehicle; object -> positive keep)
        if animate_score is not None:
            animate = (animate_score >= self.animate_score_thresh).astype(np.uint8)
            if self._kanimate is not None:
                animate = cv2.dilate(animate, self._kanimate)
            is_animate = animate > 0
        else:
            is_animate = np.zeros_like(static_fg)
        keep = ~is_animate
        if object_score is not None:
            keep = keep | (object_score >= self.object_score_thresh)

        valid = static_fg & keep & (~is_stuff)
        if cfg.motion_to_static:
            valid &= self._moved

        # age: grow valid; decay under transient motion; reset where back-to-bg/animate/stuff
        self.static_age[valid] += self.dt
        decaying = is_moving & ~valid
        self.static_age[decaying] = np.maximum(0.0, self.static_age[decaying] - cfg.moving_decay * self.dt)
        reset = (~is_diff) | is_animate | is_stuff
        self.static_age[reset] = 0.0
        abandoned = (self.static_age >= cfg.t_static_s).astype(np.uint8) * 255

        # tight mask for the bbox = different-from-clean AND not currently moving (framediff)
        tight = cv2.morphologyEx(((is_diff) & (frame_diff_dil == 0)).astype(np.uint8), cv2.MORPH_OPEN, self.kernel3)
        self.tight_mask = tight * 255

        # conservative clean_bg slow-update: NOT a static-new object AND NOT persistent; but DO
        # absorb confirmed scene-stuff (puddle/wall) so it stops firing as newdiff.
        keep_update = ((~static_fg) | is_stuff) & (~protect_persist)
        if protect_mask is not None:
            keep_update &= ~protect_mask
        lr = cfg.update_lr
        self.clean_bg[keep_update] = (1.0 - lr) * self.clean_bg[keep_update] + lr * gray[keep_update]
        if self.clean_bg_color is not None:
            frame_f = frame_bgr.astype(np.float32)
            self.clean_bg_color[keep_update] = (1.0 - lr) * self.clean_bg_color[keep_update] + lr * frame_f[keep_update]

        self.fgbg = is_diff
        self.moving = is_moving
        self.static_fg = static_fg
        self.keep = keep

        return {
            "abandoned": abandoned,
            "static_fg": static_fg.astype(np.uint8) * 255,
            "moving": is_moving.astype(np.uint8) * 255,
            "newdiff": newdiff.astype(np.uint8) * 255,
            "keep": (keep & ~is_stuff).astype(np.uint8) * 255,
            "stuff": is_stuff.astype(np.uint8) * 255,
            "age": np.clip(self.static_age / max(1e-3, cfg.t_static_s) * 255.0, 0, 255).astype(np.uint8),
            "tight": self.tight_mask,
            "relearning": False,
        }
