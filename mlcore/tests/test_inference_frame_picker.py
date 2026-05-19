import numpy as np
from src.inference.detector import Track, FrameDet
from src.inference.frame_picker import pick_A


def _make_frame(noise: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.normal(128, noise * 60, size=(200, 200, 3))
              .clip(0, 255).astype(np.uint8))


def test_pick_A_returns_sharpest():
    """Кадр с большим шумом даёт больший Laplacian variance."""
    blurry = _make_frame(0.1)
    sharp = _make_frame(1.0)
    track = Track(track_id=1, frames=[
        FrameDet(ts_ms=0, bbox=(10, 10, 190, 190), conf=0.9, frame=blurry),
        FrameDet(ts_ms=100, bbox=(10, 10, 190, 190), conf=0.9, frame=sharp),
    ])
    picked = pick_A(track)
    assert picked.ts_ms == 100


def test_pick_A_single_frame():
    track = Track(track_id=1, frames=[
        FrameDet(ts_ms=42, bbox=(0, 0, 50, 50), conf=0.9, frame=_make_frame(0.5))
    ])
    assert pick_A(track).ts_ms == 42


def test_pick_A_empty_track_returns_none():
    track = Track(track_id=1, frames=[])
    assert pick_A(track) is None
