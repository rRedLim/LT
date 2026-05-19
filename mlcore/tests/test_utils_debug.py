"""Тесты для DebugRecorder и NoOpRecorder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.debug import DebugRecorder, NoOpRecorder, make_recorder


def _dummy_frame(h: int = 200, w: int = 300) -> np.ndarray:
    return (np.random.default_rng(42).integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_noop_recorder_does_nothing(tmp_path):
    """NoOp должен молча проглатывать все вызовы, включая close()."""
    r = NoOpRecorder()
    assert r.enabled is False
    r.setup("video1")
    r.log("anything")
    r.debug("anything")
    r.warning("anything")
    r.save_frame_with_bboxes(_dummy_frame(), 0, [{"track_id": 1, "bbox": (0, 0, 50, 50), "conf": 0.9}])
    r.save_track_artifacts(1, _dummy_frame(), (0, 0, 50, 50), _dummy_frame(),
                           vlm_raw="raw", vlm_parsed={"x": 1}, final_row={"y": 2})
    r.log_barcode_attempt(1, 5, "4670025474665")
    r.save_summary({"foo": "bar"})
    with r.timer("anything"):
        pass
    # pipeline.py вызывает recorder.close() в finally — должно работать без AttributeError
    r.close()
    # Никаких файлов создаваться не должно
    assert not list(tmp_path.iterdir())


def test_make_recorder_factory(tmp_path):
    """make_recorder(False) → NoOp; True → DebugRecorder."""
    assert isinstance(make_recorder(False), NoOpRecorder)
    r = make_recorder(True, debug_root=tmp_path)
    assert isinstance(r, DebugRecorder)


def test_debug_recorder_setup_creates_dirs(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("test_video")
    video_dir = tmp_path / "test_video"
    assert video_dir.exists()
    assert (video_dir / "frames_with_bbox").exists()
    assert (video_dir / "tracks").exists()
    assert (video_dir / "pipeline.log").exists()
    assert (video_dir / "barcode.log").exists()
    r.close()


def test_debug_recorder_logs_to_file(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("test_video")
    r.log("info message %s", "with args")
    r.debug("debug message")
    r.warning("warning here")
    r.close()
    text = (tmp_path / "test_video" / "pipeline.log").read_text(encoding="utf-8")
    assert "info message with args" in text
    assert "debug message" in text
    assert "warning here" in text


def test_debug_recorder_timer_accumulates(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("v")
    with r.timer("phase_a"):
        pass
    with r.timer("phase_a"):
        pass
    with r.timer("phase_b"):
        pass
    r.save_summary({"n": 3})
    summary = json.loads((tmp_path / "v" / "summary.json").read_text(encoding="utf-8"))
    assert "timings_seconds" in summary
    assert "phase_a" in summary["timings_seconds"]
    assert "phase_b" in summary["timings_seconds"]
    r.close()


def test_save_frame_with_bboxes_writes_image(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("v")
    # every_n_frames=1 → каждый кадр пишется
    r.save_frame_with_bboxes(
        _dummy_frame(),
        ts_ms=1000,
        detections=[
            {"track_id": 1, "bbox": (10, 20, 100, 80), "conf": 0.9},
            {"track_id": 2, "bbox": (150, 30, 280, 90), "conf": 0.7},
        ],
        every_n_frames=1,
    )
    files = list((tmp_path / "v" / "frames_with_bbox").glob("*.jpg"))
    assert len(files) == 1
    assert files[0].name == "f00001000.jpg"
    r.close()


def test_save_frame_skips_when_not_nth(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("v")
    # вызываем 3 раза с every_n_frames=10 → должно сохраниться 0 (счётчик не дойдёт до 10)
    for ts in [100, 200, 300]:
        r.save_frame_with_bboxes(_dummy_frame(), ts, [], every_n_frames=10)
    files = list((tmp_path / "v" / "frames_with_bbox").glob("*.jpg"))
    assert len(files) == 0
    r.close()


def test_save_track_artifacts_writes_all(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("v")
    r.save_track_artifacts(
        track_id=42,
        pick_a_frame=_dummy_frame(500, 800),
        bbox=(100.0, 200.0, 300.0, 400.0),
        crop_for_vlm=_dummy_frame(200, 100),
        vlm_raw='{"product_name": "Сыр"}',
        vlm_parsed={"product_name": "Сыр", "color": "red"},
        final_row={"product_name": "Сыр", "color": "red", "barcode": "нет"},
    )
    tdir = tmp_path / "v" / "tracks"
    assert (tdir / "trk_0042_frame.jpg").exists()
    assert (tdir / "trk_0042_crop.jpg").exists()
    raw = (tdir / "trk_0042_vlm_raw.txt").read_text(encoding="utf-8")
    assert "Сыр" in raw
    parsed = json.loads((tdir / "trk_0042_vlm.json").read_text(encoding="utf-8"))
    assert parsed["color"] == "red"
    final = json.loads((tdir / "trk_0042_final.json").read_text(encoding="utf-8"))
    assert final["barcode"] == "нет"
    r.close()


def test_log_barcode_attempt_appends(tmp_path):
    r = DebugRecorder(tmp_path)
    r.setup("v")
    r.log_barcode_attempt(1, n_frames_tried=30, decoded="4670025474665", source="zxing")
    r.log_barcode_attempt(2, n_frames_tried=30, decoded=None, source="zxing", note="failed all")
    r.log_barcode_attempt(3, n_frames_tried=3, decoded="1234567890123", source="vlm_fallback")
    r.close()
    text = (tmp_path / "v" / "barcode.log").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 3
    assert "4670025474665" in lines[0]
    assert "NONE" in lines[1]
    assert "failed all" in lines[1]
    assert "vlm_fallback" in lines[2]


def test_setup_can_be_called_multiple_times(tmp_path):
    """Для batch-режима: один DebugRecorder, несколько видео по очереди."""
    r = DebugRecorder(tmp_path)
    r.setup("video_a")
    r.log("msg in a")
    r.setup("video_b")  # переключение на другое видео
    r.log("msg in b")
    r.close()
    text_a = (tmp_path / "video_a" / "pipeline.log").read_text(encoding="utf-8")
    text_b = (tmp_path / "video_b" / "pipeline.log").read_text(encoding="utf-8")
    assert "msg in a" in text_a
    assert "msg in b" in text_b
    assert "msg in b" not in text_a
