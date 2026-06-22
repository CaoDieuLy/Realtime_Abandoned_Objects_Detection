from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.semantic_lut import SEMANTIC_MAX


@dataclass
class StaticStateConfig:
    th_diff: int = 40
    update_lr: float = 0.0008
    t_static_s: float = 1.0           # seconds a pixel must stay static (debounce before abandoned)
    fps: float = 30.0
    tau_animate: float = 0.25         # P(person/vehicle) >= tau -> animate (NOT abandoned)
    tau_object: float = 0.30          # P(bag/umbrella/...) >= tau -> positive keep evidence
    tau_stuff: float = 0.50           # P(wall/floor/water/...) >= tau -> scene background, reject (dense models)
    dilate_motion: int = 1
    dilate_animate: int = 2
    moving_decay: float = 3.0
    # --- v2c StaticDiffBG features ---
    motion_source: str = "framediff"  # framediff | external (rtsbs/raw-vibe passed in)
    persist_s: float = 2.0            # persistent-diff >= this -> protect clean_bg from absorbing it
    light_comp: bool = True           # re-baseline clean_bg when global lighting coverage is high
    heal_cov: float = 0.15
    heal_alpha: float = 1.0
    heal_alpha_dark: float = 0.05
    dark_s_thresh: float = 15.0
    relight: bool = True              # rebuild clean_bg on a global lighting-mode switch
    relight_dv: float = 30.0
    relight_ds: float = 12.0
    relight_hold: int = 15
    relearn_s: float = 2.0
    motion_to_static: bool = False    # a candidate must have had motion (deposited) before going static
    motion_reset_s: float = 1.0       # back-to-bg must persist this long before clearing the moved latch
    motion_latch_dilate: int = 4      # dilate the motion latch so motion near a deposited object counts


class StaticForegroundState:
    """Per-pixel static-foreground state machine for abandoned-object detection.

    Merges demo2's semantic keep-gate (animate/object) with demo1 v2c's StaticDiffBG:
      - clean_bg diff -> newdiff (object vs the clean warmup background)
      - moving gate (framediff by default, or an external rtsbs/raw-vibe mask)
      - static_fg = newdiff AND NOT moving ; aged + semantic-gated -> abandoned
      - tight_mask = newdiff AND NOT framediff  (keeps the FULL static object for the bbox,
        because framediff never fires on a just-deposited still object, unlike ViBe which
        "eats" it until absorbed)
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
        self.clean_bg = clean_bg_gray.astype(np.float32)
        self.clean_bg_color = clean_bg_color.astype(np.float32) if clean_bg_color is not None else None
        h, w = self.clean_bg.shape[:2]
        self.height, self.width = h, w
        self.static_age = np.zeros((h, w), dtype=np.float32)
        self.dt = 1.0 / max(1e-3, float(self.cfg.fps))
        self.tau_animate16 = float(self.cfg.tau_animate) * float(SEMANTIC_MAX)
        self.tau_object16 = float(self.cfg.tau_object) * float(SEMANTIC_MAX)
        self.tau_stuff16 = float(self.cfg.tau_stuff) * float(SEMANTIC_MAX)
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kmotion = self._odd_kernel(self.cfg.dilate_motion)
        self._kanimate = self._odd_kernel(self.cfg.dilate_animate)
        self.persist_thresh = max(1, int(self.cfg.persist_s * self.cfg.fps))
        self.relearn_frames = max(5, int(self.cfg.relearn_s * self.cfg.fps))
        self._persist = np.zeros((h, w), dtype=np.float32)
        # motion-to-static: latch pixels that have shown motion while continuously different
        # from the clean background (a deposited object); cleared only after a SUSTAINED
        # return to background (survives brief occlusion/reveal gaps).
        self._moved = np.zeros((h, w), dtype=bool)
        self._bg_run = np.zeros((h, w), dtype=np.float32)
        self.motion_reset_frames = max(1, int(self.cfg.motion_reset_s * self.cfg.fps))
        self._klatch = self._odd_kernel(self.cfg.motion_latch_dilate)
        self._prev_gray: np.ndarray | None = None
        self.tight_mask = np.zeros((h, w), dtype=np.uint8)
        # relight state
        self.ref_V: float | None = None
        self.ref_S: float | None = None
        self._switch_count = 0
        self._relearning = False
        self._relearn_left = 0
        self._relearn_buf: list[np.ndarray] = []
        self._relearn_buf_c: list[np.ndarray] = []
        self.relearning = False
        # last outputs for inspection
        self.fgbg = np.zeros((h, w), dtype=bool)
        self.moving = np.zeros((h, w), dtype=bool)
        self.static_fg = np.zeros((h, w), dtype=bool)
        self.keep = np.ones((h, w), dtype=bool)

    @staticmethod
    def _odd_kernel(radius: int):
        r = max(0, int(radius))
        if r == 0:
            return None
        k = 2 * r + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ---- re-light (global lighting-mode switch) ----
    def _relight_step(self, frame: np.ndarray, gray: np.ndarray, protect_mask: np.ndarray | None = None):
        if self.ref_V is None:
            self.ref_V = float(self.clean_bg.mean())
            self.ref_S = float(cv2.cvtColor(self.clean_bg_color.astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].mean())
        if self._relearning:
            self._relearn_buf.append(gray.copy())
            self._relearn_buf_c.append(frame.astype(np.float32))
            self._relearn_left -= 1
            if self._relearn_left <= 0:
                self._finish_relearn(protect_mask)
            self._prev_gray = gray.copy()
            return True
        curV = float(gray.mean())
        curS = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())
        diverging = (abs(curV - self.ref_V) > self.cfg.relight_dv) or (abs(curS - self.ref_S) > self.cfg.relight_ds)
        if diverging:
            self._switch_count += 1
            if self._switch_count >= self.cfg.relight_hold:
                self._relearning = True
                self._relearn_left = self.relearn_frames - 1
                self._switch_count = 0
                self._relearn_buf = [gray.copy()]
                self._relearn_buf_c = [frame.astype(np.float32)]
                print(f"\n[RELIGHT] lighting switch (dV={abs(curV-self.ref_V):.0f} "
                      f"dS={abs(curS-self.ref_S):.0f}) -> rebuilding clean_bg {self.relearn_frames}f...", flush=True)
            self._prev_gray = gray.copy()
            return True
        self._switch_count = 0
        return False

    def _finish_relearn(self, protect_mask: np.ndarray | None = None):
        newbg = np.median(np.stack(self._relearn_buf), axis=0).astype(np.float32)
        newbg_c = np.median(np.stack(self._relearn_buf_c), axis=0).astype(np.float32)
        if protect_mask is not None:
            newbg[protect_mask] = self.clean_bg[protect_mask]
            if self.clean_bg_color is not None:
                newbg_c[protect_mask] = self.clean_bg_color[protect_mask]
        self.clean_bg = newbg
        self.clean_bg_color = newbg_c
        self.ref_V = float(newbg.mean())
        self.ref_S = float(cv2.cvtColor(newbg_c.astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1].mean())
        self._relearning = False
        self._relearn_buf = []
        self._relearn_buf_c = []
        print(f"[RELIGHT] done: new clean_bg (V={self.ref_V:.0f} S={self.ref_S:.0f})", flush=True)

    def _zeros_result(self) -> dict[str, np.ndarray]:
        z = np.zeros((self.height, self.width), dtype=np.uint8)
        self.tight_mask = z.copy()
        return {
            "abandoned": z.copy(), "static_fg": z.copy(), "moving": z.copy(),
            "newdiff": z.copy(), "keep": z.copy(), "stuff": z.copy(), "age": z.copy(),
            "tight": z.copy(), "relearning": True,
        }

    def update(
        self,
        frame_bgr: np.ndarray,
        motion_mask: np.ndarray | None = None,
        animate_prob16: np.ndarray | None = None,
        object_prob16: np.ndarray | None = None,
        stuff_prob16: np.ndarray | None = None,
        protect_mask: np.ndarray | None = None,
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
        # clean_bg absorb it (so a puddle/wall that differs from the warmup bg stops firing).
        if stuff_prob16 is not None:
            stuff_b = stuff_prob16 >= self.tau_stuff16
        else:
            stuff_b = np.zeros((self.height, self.width), dtype=bool)

        protect_persist = (self._persist >= self.persist_thresh) & (~stuff_b)

        diff = cv2.absdiff(gray, self.clean_bg)
        newdiff = (diff >= cfg.th_diff).astype(np.uint8)

        # light-comp: a global lighting event changes many regions at once (high coverage);
        # absorb them into clean_bg (re-baseline), but never the persistent/protected object.
        if cfg.light_comp and float(newdiff.mean()) > cfg.heal_cov:
            m = (newdiff > 0) & (~protect_persist)
            if protect_mask is not None:
                m &= ~protect_mask
            a = cfg.heal_alpha_dark if (self.ref_S is not None and self.ref_S < cfg.dark_s_thresh) else cfg.heal_alpha
            self.clean_bg[m] = (1.0 - a) * self.clean_bg[m] + a * gray[m]
            if self.clean_bg_color is not None:
                fc = frame_bgr.astype(np.float32)
                self.clean_bg_color[m] = (1.0 - a) * self.clean_bg_color[m] + a * fc[m]
            diff = cv2.absdiff(gray, self.clean_bg)
            newdiff = (diff >= cfg.th_diff).astype(np.uint8)
        newdiff = cv2.morphologyEx(newdiff, cv2.MORPH_OPEN, self.kernel3)
        fgbg_b = newdiff > 0

        # persistence counter (still-different region grows; back-to-background resets)
        self._persist = np.where(fgbg_b, self._persist + 1, 0.0)

        # frame difference (instantaneous motion) -> tight_mask; never eats a still object
        if self._prev_gray is None:
            fd = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            fd = (cv2.absdiff(gray, self._prev_gray) >= cfg.th_diff).astype(np.uint8)
        self._prev_gray = gray.copy()
        fd_d = cv2.dilate(fd, self.kernel3)

        # moving gate
        if cfg.motion_source == "framediff" or motion_mask is None:
            moving = fd_d
        else:
            moving = (motion_mask > 0).astype(np.uint8)
            if self._kmotion is not None:
                moving = cv2.dilate(moving, self._kmotion)
        moving_b = moving > 0

        static_fg = fgbg_b & ~moving_b

        # motion-to-static latch: a deposited object shows motion (carried in) while it
        # differs from clean_bg, then goes static. Pixels that became different WITHOUT
        # any local motion (lighting/reflection/scene diffs) never latch -> rejected.
        # Latch is dilated (motion near the object counts) and only cleared after a
        # SUSTAINED return to background, so an occlusion/reveal gap doesn't erase it.
        moved_now = moving_b
        if self._klatch is not None:
            moved_now = cv2.dilate(moving_b.astype(np.uint8), self._klatch) > 0
        self._moved |= moved_now
        self._bg_run = np.where(fgbg_b, 0.0, self._bg_run + 1.0)
        self._moved[self._bg_run >= self.motion_reset_frames] = False

        # semantic keep gate (animate -> reject person/vehicle; object -> positive keep)
        if animate_prob16 is not None:
            animate = (animate_prob16 >= self.tau_animate16).astype(np.uint8)
            if self._kanimate is not None:
                animate = cv2.dilate(animate, self._kanimate)
            animate_b = animate > 0
        else:
            animate_b = np.zeros_like(static_fg)
        keep = ~animate_b
        if object_prob16 is not None:
            keep = keep | (object_prob16 >= self.tau_object16)

        valid = static_fg & keep & (~stuff_b)
        if cfg.motion_to_static:
            valid &= self._moved

        # age: grow valid; decay under transient motion; reset where back-to-bg/animate/stuff
        self.static_age[valid] += self.dt
        decaying = moving_b & ~valid
        self.static_age[decaying] = np.maximum(0.0, self.static_age[decaying] - cfg.moving_decay * self.dt)
        gone = (~fgbg_b) | animate_b | stuff_b
        self.static_age[gone] = 0.0
        abandoned = (self.static_age >= cfg.t_static_s).astype(np.uint8) * 255

        # tight mask for the bbox = different-from-clean AND not currently moving (framediff)
        tight = cv2.morphologyEx(((fgbg_b) & (fd_d == 0)).astype(np.uint8), cv2.MORPH_OPEN, self.kernel3)
        self.tight_mask = tight * 255

        # conservative clean_bg slow-update: NOT a static-new object AND NOT persistent;
        # but DO absorb confirmed scene-stuff (puddle/wall) so it stops firing as newdiff.
        keep_update = ((~static_fg) | stuff_b) & (~protect_persist)
        if protect_mask is not None:
            keep_update &= ~protect_mask
        lr = cfg.update_lr
        self.clean_bg[keep_update] = (1.0 - lr) * self.clean_bg[keep_update] + lr * gray[keep_update]
        if self.clean_bg_color is not None:
            fcol = frame_bgr.astype(np.float32)
            self.clean_bg_color[keep_update] = (1.0 - lr) * self.clean_bg_color[keep_update] + lr * fcol[keep_update]

        self.fgbg = fgbg_b
        self.moving = moving_b
        self.static_fg = static_fg
        self.keep = keep

        return {
            "abandoned": abandoned,
            "static_fg": static_fg.astype(np.uint8) * 255,
            "moving": moving_b.astype(np.uint8) * 255,
            "newdiff": newdiff.astype(np.uint8) * 255,
            "keep": (keep & ~stuff_b).astype(np.uint8) * 255,
            "stuff": stuff_b.astype(np.uint8) * 255,
            "age": np.clip(self.static_age / max(1e-3, cfg.t_static_s) * 255.0, 0, 255).astype(np.uint8),
            "tight": self.tight_mask,
            "relearning": False,
        }
