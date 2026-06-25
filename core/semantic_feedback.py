"""RT-SBS semantic decision table: combine the BGS foreground with the semantic map into a CORRECTED
foreground mask fed back into ViBe (segment -> correct -> update). Implements the background/
foreground keep-drop rules (tau_bg/tau_fg, optional two-sided tau_bg*/tau_fg*).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticFeedbackConfig:
    tau_bg: int = 300
    tau_fg: int = 175
    tau_bg_star: int = 65
    tau_fg_star: int = 115
    model_update_factor: int = 256
    seed: int = 54321
    enable_bg_rule: bool = True  # False -> abstain on low-semantic pixels (for sparse instance maps)


class SemanticFeedback:
    """Original RT-SBS semantic rule maps for dense 16-bit semantic images."""

    def __init__(
        self,
        height: int,
        width: int,
        cfg: SemanticFeedbackConfig | None = None,
        initial_semantic: np.ndarray | None = None,
    ):
        self.height = height
        self.width = width
        self.cfg = cfg or SemanticFeedbackConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        if initial_semantic is None:
            self.semantic_model = np.zeros((height, width), dtype=np.float32)
        else:
            self.semantic_model = self._as_semantic(initial_semantic)
        self.rule_map_bg = np.zeros((height, width), dtype=bool)
        self.rule_map_fg = np.zeros((height, width), dtype=bool)
        self.applied_rule_bg = np.zeros((height, width), dtype=bool)
        self.applied_rule_fg = np.zeros((height, width), dtype=bool)
        self.color_map: np.ndarray | None = None

    def segment_with_semantics(
        self,
        frame: np.ndarray,
        bgs_mask: np.ndarray,
        semantic: np.ndarray,
    ) -> np.ndarray:
        self.color_map = frame.copy()
        sem = self._as_semantic(semantic)

        if self.cfg.enable_bg_rule:
            self.rule_map_bg = sem <= float(self.cfg.tau_bg)
        else:
            # Sparse instance maps: "no detection" (sem==0) is NOT confident background,
            # so abstain (defer to BGS) instead of forcing BG.
            self.rule_map_bg = np.zeros((self.height, self.width), dtype=bool)
        semantic_increase = sem - self.semantic_model
        self.rule_map_fg = semantic_increase >= float(self.cfg.tau_fg * 256)
        self.applied_rule_bg = self.rule_map_bg
        self.applied_rule_fg = self.rule_map_fg

        out = self._apply_rules(bgs_mask, self.rule_map_bg, self.rule_map_fg)
        self._update_semantic_model(out, sem)
        return out

    def segment_without_semantics(self, frame: np.ndarray, bgs_mask: np.ndarray) -> np.ndarray:
        if self.color_map is None:
            return bgs_mask.copy()

        color_diff = np.abs(frame.astype(np.int16) - self.color_map.astype(np.int16)).sum(axis=2)
        relevant_bg = self.rule_map_bg & (color_diff <= self.cfg.tau_bg_star)
        relevant_fg = self.rule_map_fg & (color_diff <= self.cfg.tau_fg_star)
        self.applied_rule_bg = relevant_bg
        self.applied_rule_fg = relevant_fg
        return self._apply_rules(bgs_mask, relevant_bg, relevant_fg)

    def _as_semantic(self, semantic: np.ndarray) -> np.ndarray:
        if semantic.shape != (self.height, self.width):
            raise ValueError(
                f"Semantic map shape {semantic.shape} does not match {(self.height, self.width)}"
            )
        return semantic.astype(np.float32, copy=False)

    @staticmethod
    def _apply_rules(bgs_mask: np.ndarray, bg_rule: np.ndarray, fg_rule: np.ndarray) -> np.ndarray:
        out = bgs_mask.copy()
        out[bg_rule] = 0
        out[fg_rule & ~bg_rule] = 255
        return out

    def _update_semantic_model(self, final_mask: np.ndarray, semantic: np.ndarray) -> None:
        bg_pixels = final_mask == 0
        update = bg_pixels & (
            self.rng.integers(0, self.cfg.model_update_factor, size=bg_pixels.shape) == 0
        )
        self.semantic_model[update] = semantic[update]
