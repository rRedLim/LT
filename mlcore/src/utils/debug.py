"""DebugRecorder — артефакты для дебага inference-пайплайна.

Включается через `--debug` флаг в run_inference.py. Если не включён —
все методы no-op (нулевая стоимость в проде).

Структура артефактов:
    debug/<video_stem>/
        pipeline.log                 # все DEBUG-логи + timing
        frames_with_bbox/
            f0000000.jpg             # каждый N-ый кадр с нарисованными bbox+track_id
            f0001000.jpg
            ...
        tracks/
            trk_001_crop.jpg         # Pick A кроп (повёрнутый, ушёл в VLM)
            trk_001_frame.jpg        # Pick A кадр целиком с подсветкой
            trk_001_vlm_raw.txt      # сырой текст от Qwen-VL до парсера
            trk_001_vlm.json         # распарсенный dict
            trk_001_final.json       # финальный routed row (после template_router)
            ...
        barcode.log                  # log попыток zxing по каждому треку
        summary.json                 # сводка: n_tracks, n_with_barcode, timings
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


_LOG_FMT = "%(asctime)s %(levelname)-7s %(name)-25s %(message)s"


class NoOpRecorder:
    """Заглушка когда debug выключен. Все методы — no-op."""

    enabled: bool = False

    def setup(self, video_stem: str) -> None: ...
    def close(self) -> None: ...  # pipeline.py всегда зовёт close() в финале

    def log(self, msg: str, *args: Any) -> None: ...
    def debug(self, msg: str, *args: Any) -> None: ...
    def warning(self, msg: str, *args: Any) -> None: ...

    def save_frame_with_bboxes(
        self, frame: np.ndarray, ts_ms: int, detections: list, every_n_frames: int = 30
    ) -> None: ...

    def save_track_artifacts(
        self,
        track_id: int,
        pick_a_frame: np.ndarray,
        bbox: tuple,
        crop_for_vlm: np.ndarray,
        vlm_raw: str = "",
        vlm_parsed: Optional[dict] = None,
        final_row: Optional[dict] = None,
    ) -> None: ...

    def log_barcode_attempt(
        self,
        track_id: int,
        n_frames_tried: int,
        decoded: Optional[str],
        source: str = "zxing",
        note: str = "",
    ) -> None: ...

    def save_summary(self, summary: dict) -> None: ...

    @contextmanager
    def timer(self, label: str):
        yield


class DebugRecorder:
    """Полноценный рекордер артефактов для дебага."""

    enabled: bool = True

    def __init__(self, root: Path):
        self.root = Path(root)
        self.video_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
        self.tracks_dir: Optional[Path] = None
        self.logger: Optional[logging.Logger] = None
        self.barcode_log_path: Optional[Path] = None
        self._barcode_log_fh = None
        self._frame_counter = 0
        self._timings: dict[str, float] = {}

    def setup(self, video_stem: str) -> None:
        """Создаёт структуру папок под одно видео. Можно вызывать многократно
        (для batch-режима с несколькими видео)."""
        # Закрыть прошлый файл-handler если есть
        self._cleanup()

        self.video_dir = self.root / video_stem
        self.frames_dir = self.video_dir / "frames_with_bbox"
        self.tracks_dir = self.video_dir / "tracks"
        for d in [self.video_dir, self.frames_dir, self.tracks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Logger именно для этого видео — файл pipeline.log
        self.logger = logging.getLogger(f"debug.{video_stem}")
        self.logger.setLevel(logging.DEBUG)
        # Удаляем предыдущие хэндлеры если они были (повторный setup)
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        fh = logging.FileHandler(self.video_dir / "pipeline.log",
                                 mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FMT))
        self.logger.addHandler(fh)
        self.logger.propagate = False  # не дублируем в root

        self.barcode_log_path = self.video_dir / "barcode.log"
        self._barcode_log_fh = open(self.barcode_log_path, "w", encoding="utf-8")
        self._barcode_log_fh.write(
            "# track_id, n_frames_tried, source, decoded, note\n"
        )

        self._frame_counter = 0
        self._timings = {}
        self.logger.info("DebugRecorder ready for %s", video_stem)

    def _cleanup(self) -> None:
        if self._barcode_log_fh is not None:
            try:
                self._barcode_log_fh.close()
            except Exception:
                pass
            self._barcode_log_fh = None
        if self.logger is not None:
            for h in list(self.logger.handlers):
                try:
                    h.close()
                except Exception:
                    pass
                self.logger.removeHandler(h)

    def close(self) -> None:
        if self.logger is not None:
            self.logger.info("DebugRecorder closing. Total timings: %s", self._timings)
        self._cleanup()

    # ── логирование ──────────────────────────────────────────────────────

    def log(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.info(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.debug(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.warning(msg, *args)

    @contextmanager
    def timer(self, label: str):
        """Контекстный менеджер для замера времени блока.

        Использование:
            with debug.timer("detect"):
                tracks = detect_and_track(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._timings[label] = self._timings.get(label, 0.0) + dt
            if self.logger is not None:
                self.logger.info("[timer] %s: %.3fs", label, dt)

    # ── визуальный дебаг детектора ───────────────────────────────────────

    def save_frame_with_bboxes(
        self,
        frame: np.ndarray,
        ts_ms: int,
        detections: list,
        every_n_frames: int = 30,
    ) -> None:
        """Рисует bbox+track_id поверх кадра. detections — список dict с ключами
        track_id, bbox=(x_min,y_min,x_max,y_max), conf.

        Сохраняет каждый every_n_frames-ный кадр (1 раз в ~секунду при 30fps).
        """
        self._frame_counter += 1
        if self._frame_counter % every_n_frames != 0:
            return
        if self.frames_dir is None:
            return
        img = frame.copy()
        for det in detections:
            tid = det.get("track_id", -1)
            bbox = det.get("bbox", (0, 0, 0, 0))
            conf = det.get("conf", 0.0)
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"id={tid} {conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        out = self.frames_dir / f"f{ts_ms:08d}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # ── артефакты по треку ───────────────────────────────────────────────

    def save_track_artifacts(
        self,
        track_id: int,
        pick_a_frame: np.ndarray,
        bbox: tuple,
        crop_for_vlm: np.ndarray,
        vlm_raw: str = "",
        vlm_parsed: Optional[dict] = None,
        final_row: Optional[dict] = None,
    ) -> None:
        """Сохраняет:
        - trk_NNN_frame.jpg — pick_a кадр с подсветкой bbox
        - trk_NNN_crop.jpg — то что реально ушло в VLM (повёрнутый кроп)
        - trk_NNN_vlm_raw.txt — сырой ответ VLM
        - trk_NNN_vlm.json — распарсенный dict
        - trk_NNN_final.json — финальный row после template_router
        """
        if self.tracks_dir is None:
            return
        stem = f"trk_{track_id:04d}"

        # frame с подсветкой.
        # После RAM-оптимизации pick_a_frame — это уже КРОП (а не полный кадр),
        # поэтому bbox в координатах исходного кадра больше не валиден.
        # Рисуем рамку по периметру кропа + подпись trk_id сверху.
        if pick_a_frame is not None:
            highlighted = pick_a_frame.copy()
            h, w = highlighted.shape[:2]
            cv2.rectangle(highlighted, (2, 2), (w - 3, h - 3), (0, 0, 255), 3)
            cv2.putText(highlighted, f"trk={track_id}", (8, max(20, h // 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(self.tracks_dir / f"{stem}_frame.jpg"),
                        highlighted, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # crop_for_vlm
        if crop_for_vlm is not None and crop_for_vlm.size > 0:
            cv2.imwrite(str(self.tracks_dir / f"{stem}_crop.jpg"),
                        crop_for_vlm, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # vlm raw
        if vlm_raw:
            (self.tracks_dir / f"{stem}_vlm_raw.txt").write_text(
                vlm_raw, encoding="utf-8"
            )

        # vlm parsed
        if vlm_parsed is not None:
            (self.tracks_dir / f"{stem}_vlm.json").write_text(
                json.dumps(vlm_parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # final row
        if final_row is not None:
            (self.tracks_dir / f"{stem}_final.json").write_text(
                json.dumps(final_row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── barcode log ──────────────────────────────────────────────────────

    def log_barcode_attempt(
        self,
        track_id: int,
        n_frames_tried: int,
        decoded: Optional[str],
        source: str = "zxing",
        note: str = "",
    ) -> None:
        if self._barcode_log_fh is None:
            return
        decoded_str = decoded if decoded else "NONE"
        line = f"trk={track_id:04d}  tried={n_frames_tried:3d}  source={source:12s}  decoded={decoded_str:16s}  {note}\n"
        self._barcode_log_fh.write(line)
        self._barcode_log_fh.flush()

    # ── финальная сводка ─────────────────────────────────────────────────

    def save_summary(self, summary: dict) -> None:
        if self.video_dir is None:
            return
        summary = dict(summary)
        summary["timings_seconds"] = dict(self._timings)
        (self.video_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def make_recorder(enabled: bool, debug_root: Optional[Path] = None):
    """Фабрика. enabled=False → NoOp. Иначе создаём DebugRecorder."""
    if not enabled:
        return NoOpRecorder()
    if debug_root is None:
        debug_root = Path("debug")
    return DebugRecorder(debug_root)
