from pathlib import Path
from typing import Iterator, Tuple, Optional
import cv2
import numpy as np


def iter_video_frames(
    video_path: Path,
    step_ms: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yields (ts_ms, frame). step_ms=None → все кадры; step_ms=1000 → 1 fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    try:
        if step_ms is None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                yield ts, frame
        else:
            duration_ms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) /
                              max(cap.get(cv2.CAP_PROP_FPS), 1) * 1000)
            for ts in range(0, duration_ms, step_ms):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts)
                ok, frame = cap.read()
                if not ok:
                    continue
                yield ts, frame
    finally:
        cap.release()


def read_frame_at(video_path: Path, ts_ms: int) -> Optional[np.ndarray]:
    """Один кадр по точному timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
