"""Per-alert open-vocabulary VERIFIER (MobileCLIP2 via open_clip).

A zero-shot CLIP classifier run ONCE per alert candidate (not per frame) on the candidate's bbox
crop. It judges the *content* of the crop against two prompt groups -- OBJECT (bag/suitcase/box/...
= a leave-behind) and NOT-OBJECT (a standing person / a crowd / bare floor / wall / door / chair /
parked car / shadow / glare). It only ever SUPPRESSES an alert; it never creates one. By design it
is recall-SAFE: it suppresses *only* when the crop is confidently NOT an object AND has little
object-likeness, otherwise it abstains (keep -> alert). This dodges the two traps that sank the
earlier YOLO-person alert-gate (see solution_analysis): (a) a crowd FP that fires after the person
left shows bare floor -> CLIP confidently "empty floor" -> suppress correctly (no real object lost);
(b) a real object with its owner still standing nearby keeps a high object score -> NOT suppressed.

Cost: one CLIP image-encode per *candidate id* (cached), so a few inferences per video -- negligible
vs the per-frame YOLO. Runs on CPU. MobileCLIP2-B image encode ~tens of ms on a laptop CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Prompt classes. Kept as full phrases (the template only wraps them) so articles read naturally.
OBJECT_PROMPTS = [
    "a backpack on the ground",
    "a suitcase on the ground",
    "a handbag on the ground",
    "a duffel bag on the ground",
    "a cardboard box on the ground",
    "a piece of luggage",
    "an abandoned bag",
    "a shopping bag on the ground",
    "an umbrella on the ground",
    "a suitcase or trolley",
]
NOT_OBJECT_PROMPTS = [
    "a person standing",
    "a crowd of people",
    "people walking",
    "an empty floor",
    "an empty tiled floor",
    "a bare concrete ground",
    "a plain wall",
    "a door",
    "an empty chair",
    "a parked car",
    "a shadow on the floor",
    "a light reflection on the floor",
]
TEMPLATES = [
    "a photo of {}.",
    "a surveillance camera photo of {}.",
]


@dataclass
class ClipResult:
    suppress: bool
    label: str            # top-1 class phrase
    top1: float           # top-1 softmax prob (over all classes)
    p_object: float       # grouped prob mass on OBJECT classes
    p_not_object: float   # grouped prob mass on NOT-OBJECT classes


class ClipVerifier:
    """Loads MobileCLIP2 (open_clip), caches text embeddings, verifies alert crops.

    Decision (recall-safe): SUPPRESS only if all of
        * grouped p_object < ``keep_conf``      (little object-likeness), AND
        * the top-1 class is a NOT-OBJECT class, AND
        * top-1 softmax prob >= ``suppress_conf`` (model is *confident* it's a specific non-object).
    Otherwise KEEP (abstain). So ambiguity / any decent object signal -> alert.
    """

    def __init__(
        self,
        model_name: str = "MobileCLIP2-B",
        pretrained: str = "",
        device: str = "cpu",
        keep_conf: float = 0.25,
        suppress_conf: float = 0.30,
        pad: int = 8,
    ):
        import torch
        import open_clip

        self.torch = torch
        self.device = device
        self.keep_conf = float(keep_conf)
        self.suppress_conf = float(suppress_conf)
        self.pad = int(pad)

        model_name, pretrained = self._resolve(open_clip, model_name, pretrained)
        self.model_name, self.pretrained = model_name, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.logit_scale = float(self.model.logit_scale.exp().item())

        self.prompts = OBJECT_PROMPTS + NOT_OBJECT_PROMPTS
        self.is_object = np.array([True] * len(OBJECT_PROMPTS) + [False] * len(NOT_OBJECT_PROMPTS))
        self.text_feats = self._encode_text(self.prompts)   # (C, D) normalized
        self.n_verify = 0           # how many image-encodes ran (cost transparency)

    @staticmethod
    def _resolve(open_clip, model_name: str, pretrained: str) -> tuple[str, str]:
        """Pick an installed model name + a pretrained tag, tolerant to the exact open_clip registry.
        Tries the requested MobileCLIP2-B, then sensible fallbacks, and auto-fills the pretrained tag."""
        pairs = open_clip.list_pretrained()                 # [(model, tag), ...]
        by_model: dict[str, list[str]] = {}
        for m, t in pairs:
            by_model.setdefault(m, []).append(t)
        candidates = [model_name, "MobileCLIP2-B", "MobileCLIP2-S4", "MobileCLIP2-S2",
                      "MobileCLIP-B", "MobileCLIP-S2", "ViT-B-16"]
        # case-insensitive match against available model names
        avail = {m.lower(): m for m in by_model}
        for cand in candidates:
            real = avail.get(cand.lower())
            if real:
                tag = pretrained if (pretrained and pretrained in by_model[real]) else by_model[real][0]
                return real, tag
        raise RuntimeError(
            f"No usable open_clip model found (requested '{model_name}'). "
            f"Available sample: {list(by_model)[:8]}"
        )

    def _encode_text(self, prompts: list[str]):
        torch = self.torch
        feats = []
        with torch.no_grad():
            for p in prompts:
                toks = self.tokenizer([t.format(p) for t in TEMPLATES]).to(self.device)
                tf = self.model.encode_text(toks)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                tf = tf.mean(dim=0)                          # ensemble templates
                tf = tf / tf.norm()
                feats.append(tf)
        return torch.stack(feats, dim=0)                    # (C, D)

    def _crop(self, frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
        x2, y2 = min(w, x2 + self.pad), min(h, y2 + self.pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def verify(self, frame_bgr: np.ndarray, bbox) -> ClipResult:
        """Classify the crop; return a ClipResult. On any failure, abstain (suppress=False)."""
        torch = self.torch
        crop = self._crop(frame_bgr, bbox)
        if crop is None:
            return ClipResult(False, "(too-small)", 0.0, 0.0, 0.0)
        from PIL import Image

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        self.n_verify += 1
        with torch.no_grad():
            f = self.model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * (f @ self.text_feats.T)    # (1, C)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        p_object = float(probs[self.is_object].sum())
        p_not = float(probs[~self.is_object].sum())
        top = int(probs.argmax())
        top1 = float(probs[top])
        top_is_not = not bool(self.is_object[top])
        suppress = bool(p_object < self.keep_conf and top_is_not and top1 >= self.suppress_conf)
        return ClipResult(suppress, self.prompts[top], top1, p_object, p_not)
