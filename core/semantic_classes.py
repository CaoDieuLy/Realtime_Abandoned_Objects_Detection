"""Class lookups + score scaling for the semantic sources. Builds the ANIMATE (person/vehicle/animal
-> reject), OBJECT (bag/umbrella/... -> keep) and STUFF (wall/floor/... -> background) class-id sets
from a model's label map, and converts probabilities to the 0..SEMANTIC_MAX integer 'score' scale
used throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEMANTIC_MAX = 65535
BG_VALUE = 0
FG_VALUE = SEMANTIC_MAX


MOVING_OBJECT_TERMS = {
    "person",
    "individual",
    "someone",
    "somebody",
    "car",
    "auto",
    "automobile",
    "motorcar",
    "bus",
    "truck",
    "van",
    "airplane",
    "boat",
    "ship",
    "bicycle",
    "minibike",
    "motorcycle",
    "animal",
}


# Classes that an abandoned object is likely to be. Used as a positive "keep"
# signal for the AOD static-object channel. Note: the ADE20K-150 label set used
# by SegFormer has no "umbrella" class, so an umbrella is kept only via the
# complement of MOVING_OBJECT_TERMS (low animate probability). These terms still
# help for bag/box/bottle targets (ABODA video1-10) and richer label sets.
STATIC_OBJECT_TERMS = {
    "umbrella",
    "bag",
    "handbag",
    "backpack",
    "rucksack",
    "knapsack",
    "suitcase",
    "luggage",
    "baggage",
    "briefcase",
    "box",
    "carton",
    "case",
    "package",
    "parcel",
    "bottle",
    "trolley",
    "cart",
}


# Scene "stuff"/background classes (dense semantic seg only; instance detectors lack
# these). A static candidate whose pixels are confidently one of these is part of the
# scene (a puddle, a wall, the floor) -> reject it as an abandoned object. Needs a
# stuff-labelling model (SegFormer/PSPNet/panoptic); YOLO-COCO has none of them.
STUFF_BACKGROUND_TERMS = {
    "wall",
    "building",
    "edifice",
    "house",
    "skyscraper",
    "sky",
    "floor",
    "flooring",
    "ceiling",
    "road",
    "route",
    "street",
    "sidewalk",
    "pavement",
    "path",
    "runway",
    "earth",
    "ground",
    "field",
    "land",
    "dirt",
    "sand",
    "grass",
    "plant",
    "tree",
    "mountain",
    "mount",
    "hill",
    "rock",
    "stone",
    "water",
    "sea",
    "river",
    "lake",
    "waterfall",
    "fence",
    "railing",
    "column",
    "pole",
    "terrain",
    "stairs",
    "stairway",
}


@dataclass(frozen=True)
class MovingClassSet:
    mask: np.ndarray
    ids: list[int]
    labels: list[str]


def norm_label(label: str) -> list[str]:
    label = label.lower().replace("-", " ").replace("_", " ")
    parts: list[str] = []
    for chunk in label.replace("/", ",").split(","):
        chunk = " ".join(chunk.strip().split())
        if chunk:
            parts.append(chunk)
            parts.extend([p for p in chunk.split(" ") if p])
    return parts


def _id2label(labels: dict[int, str] | list[str] | tuple[str, ...]) -> dict[int, str]:
    if isinstance(labels, dict):
        id2label = {int(k): str(v) for k, v in labels.items()}
    else:
        id2label = {i: str(v) for i, v in enumerate(labels)}
    return id2label


def build_class_set(
    labels: dict[int, str] | list[str] | tuple[str, ...],
    terms: set[str],
) -> MovingClassSet:
    """Build the set of class ids whose labels match any of ``terms``."""

    id2label = _id2label(labels)
    n = max(id2label) + 1 if id2label else 0
    mask = np.zeros((n,), dtype=bool)
    ids: list[int] = []
    matched_labels: list[str] = []
    for cid in range(n):
        label = str(id2label.get(cid, cid))
        if set(norm_label(label)) & terms:
            mask[cid] = True
            ids.append(cid)
            matched_labels.append(f"{cid}:{label}")
    return MovingClassSet(mask=mask, ids=ids, labels=matched_labels)


def build_moving_class_set(labels: dict[int, str] | list[str] | tuple[str, ...]) -> MovingClassSet:
    """Classes whose probabilities are summed into the RT-SBS semantic signal.

    The RT-SBS paper does not feed a hard semantic class map into SBS. It uses
    the probability that each pixel belongs to objects likely to be moving, then
    thresholds that 16-bit scalar signal into BG/FG/"?" at runtime.
    """

    return build_class_set(labels, MOVING_OBJECT_TERMS)


def build_object_class_set(labels: dict[int, str] | list[str] | tuple[str, ...]) -> MovingClassSet:
    """Classes likely to be an abandoned object (positive keep signal for AOD)."""

    return build_class_set(labels, STATIC_OBJECT_TERMS)


def build_stuff_class_set(labels: dict[int, str] | list[str] | tuple[str, ...]) -> MovingClassSet:
    """Scene/background 'stuff' classes (reject signal for AOD; dense models only)."""

    return build_class_set(labels, STUFF_BACKGROUND_TERMS)


def probability_to_score(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    return np.rint(prob * float(SEMANTIC_MAX)).astype(np.uint16)
