"""
decord compatibility shim for aarch64 (Linux/ARM).

PyPI's `decord` and `eva-decord` only ship x86_64 wheels.
This module re-implements the subset of the decord API used by PAI-Bench-C
on top of cv2 (opencv-python-headless), which has proper aarch64 wheels.

Supported API surface
─────────────────────
  decord.bridge.set_bridge("torch" | "native")
  decord.VideoReader(source, width=-1, height=-1)
    len(reader)                          → int
    reader.get_avg_fps()                 → float
    reader[idx]                          → torch.Tensor (H,W,C) or _Frame
    reader.get_batch(indices)            → _Batch  (.asnumpy() / .numpy() / .shape)
"""

from __future__ import annotations

import os
import tempfile
from typing import Union

import cv2
import numpy as np

# ── Bridge ─────────────────────────────────────────────────────────────────────

_bridge_mode: str = "native"


class _BridgeNamespace:
    @staticmethod
    def set_bridge(mode: str) -> None:
        global _bridge_mode
        _bridge_mode = mode


bridge = _BridgeNamespace()


# ── Array wrappers ─────────────────────────────────────────────────────────────

class _Frame:
    """Single-frame wrapper with .asnumpy() for compatibility."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    # decord returns NDArray-like objects; callers use .asnumpy() or index .shape
    def asnumpy(self) -> np.ndarray:
        return self._arr

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def shape(self) -> tuple:
        return self._arr.shape


class _Batch:
    """Multi-frame wrapper returned by get_batch()."""

    def __init__(self, arr: np.ndarray) -> None:
        # arr shape: (N, H, W, C) uint8 RGB
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def shape(self) -> tuple:
        return self._arr.shape

    def __iter__(self):
        return iter(self._arr)

    def __len__(self) -> int:
        return len(self._arr)


# ── VideoReader ────────────────────────────────────────────────────────────────

class VideoReader:
    """
    Minimal decord.VideoReader replacement backed by cv2.VideoCapture.

    Eagerly loads all frames into a list of uint8 (H, W, 3) RGB numpy arrays.
    This is fine for PAI-Bench-C's typical 121-frame clips at 720p.
    """

    def __init__(
        self,
        source: Union[str, os.PathLike, bytes, "io.IOBase"],
        width: int = -1,
        height: int = -1,
        **kwargs,  # absorb any extra decord kwargs silently
    ) -> None:
        self._width = width
        self._height = height
        self._tmp_path: str | None = None

        # Resolve source to a file path cv2 can open
        if isinstance(source, (str, os.PathLike)):
            path = str(source)
        else:
            # Bytes or file-like (BytesIO)
            if hasattr(source, "read"):
                data: bytes = source.read()
            else:
                data = bytes(source)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            self._tmp_path = tmp.name
            path = self._tmp_path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"decord shim: cv2 could not open video: {path}")

        self._fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames: list[np.ndarray] = []
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if width > 0 and height > 0:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
        cap.release()

        self._frames = frames

    # ── Public API ──────────────────────────────────────────────────────────

    def get_avg_fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        """Return a single frame.

        With bridge="torch" (DOVER path) → torch.Tensor (H, W, C) uint8.
        Otherwise → _Frame wrapping a numpy array.
        """
        arr = self._frames[int(idx)]
        if _bridge_mode == "torch":
            import torch
            return torch.from_numpy(arr)
        return _Frame(arr)

    def get_batch(self, indices) -> _Batch:
        """Return multiple frames as a (N, H, W, C) uint8 ndarray wrapped in _Batch."""
        batch = np.stack([self._frames[int(i)] for i in indices], axis=0)
        return _Batch(batch)

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        if getattr(self, "_tmp_path", None):
            try:
                os.unlink(self._tmp_path)
            except Exception:
                pass
