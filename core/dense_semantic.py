from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np


SEMANTIC_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".npy", ".npz"}


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class DenseSemanticSequence:
    """Frame-indexed dense semantic maps, matching the RT-SBS PSPNet-mask setup."""

    def __init__(self, semantic_dir: str, height: int, width: int, strict_frame_index: bool = True):
        self.semantic_dir = Path(semantic_dir)
        self.height = height
        self.width = width
        self.strict_frame_index = bool(strict_frame_index)
        if not self.semantic_dir.exists():
            raise FileNotFoundError(f"semantic-dir does not exist: {semantic_dir}")
        if not self.semantic_dir.is_dir():
            raise NotADirectoryError(f"semantic-dir is not a directory: {semantic_dir}")

        self.files = sorted(
            [p for p in self.semantic_dir.iterdir() if p.suffix.lower() in SEMANTIC_EXTS],
            key=_natural_key,
        )
        if not self.files:
            raise FileNotFoundError(f"No semantic maps found in: {semantic_dir}")
        self.by_frame: dict[int, Path] = {}
        for path in self.files:
            nums = re.findall(r"\d+", path.stem)
            if nums:
                self.by_frame[int(nums[-1])] = path

    def __len__(self) -> int:
        return len(self.files)

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0:
            raise IndexError(
                f"Missing semantic map for frame {frame_idx}. "
                f"Found {len(self.files)} files in {os.fspath(self.semantic_dir)}"
            )
        path = self.by_frame.get(frame_idx)
        if path is None and not self.strict_frame_index and frame_idx < len(self.files):
            path = self.files[frame_idx]
        if path is None:
            raise IndexError(
                f"Missing semantic map for frame {frame_idx}. "
                f"Found {len(self.files)} files in {os.fspath(self.semantic_dir)}"
            )
        return self._read_file(path)

    def _read_file(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".npz":
            data = np.load(path)
            arr = data[data.files[0]]
        else:
            arr = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(f"Cannot read semantic map: {path}")

        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.shape[:2] != (self.height, self.width):
            arr = cv2.resize(arr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return arr.astype(np.float32, copy=False)
