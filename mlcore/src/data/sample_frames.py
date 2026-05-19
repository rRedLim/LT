"""Нарезка видео на кадры для последующей разметки в Label Studio.

1 кадр в секунду (дефолт) — оптимально для bbox-разметки: ценники почти
не двигаются между секундами. JPG @ max-side=1920 — компромисс качества/
скорости загрузки в LS.

Имена файлов: t<ms7>.jpg, где ms7 = таймстемп в миллисекундах, padded до 7 цифр.

Запуск:
    cd ml_core
    python3 scripts/prepare_labeling.py sample-frames \\
        --data ../dataset/dataset_myself \\
        --out ../labeling/frames \\
        --fps 1.0
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2


log = logging.getLogger(__name__)


def resize_keep_aspect(img, max_side: int):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    nw, nh = int(round(w * s)), int(round(h * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def sample_video(
    video_path: Path,
    out_dir: Path,
    fps: float,
    max_side: int,
    jpg_quality: int,
    skip_existing: bool = True,
) -> int:
    """Нарезает одно видео на кадры. Возвращает кол-во сохранённых кадров."""
    if skip_existing and out_dir.exists():
        existing = list(out_dir.glob("t*.jpg"))
        if existing:
            log.info("[%s] уже нарезано (%d кадров) — пропускаю",
                     video_path.name, len(existing))
            return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("[%s] не открылось", video_path.name)
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / src_fps if src_fps > 0 else 0
    step_ms = int(1000.0 / fps)
    log.info("[%s] src_fps=%.1f dur=%.1fs frames=%d → шаг %dms",
             video_path.name, src_fps, duration_s, n_frames, step_ms)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    next_ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if ts_ms < next_ts:
            continue
        small = resize_keep_aspect(frame, max_side)
        name = f"t{ts_ms:07d}.jpg"
        ok = cv2.imwrite(
            str(out_dir / name), small,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
        )
        if ok:
            saved += 1
        next_ts = ts_ms + step_ms
    cap.release()
    log.info("[%s] сохранено %d кадров → %s", video_path.name, saved, out_dir)
    return saved


def sample_all(
    data_dir: Path,
    out_dir: Path,
    fps: float = 1.0,
    max_side: int = 1920,
    jpg_quality: int = 92,
    skip_existing: bool = True,
) -> int:
    """Нарезает все .mov/.mp4/.m4v в data_dir. Возвращает суммарный count."""
    if not data_dir.exists():
        log.error("не найдено: %s", data_dir)
        return 0
    videos = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mov", ".mp4", ".m4v")
    )
    log.info("будет обработано видео: %d", len(videos))
    for v in videos:
        log.info("  • %s", v.name)

    total = 0
    for v in videos:
        per_video_out = out_dir / v.name  # out/<имя_видео>/t*.jpg
        total += sample_video(v, per_video_out, fps, max_side, jpg_quality,
                              skip_existing=skip_existing)
    log.info("Итого сохранено: %d кадров в %s", total, out_dir)
    return total


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="папка с .mov/.mp4 (например ../dataset/dataset_myself)")
    p.add_argument("--out", type=Path, required=True,
                   help="куда сложить кадры (например ../labeling/frames)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--jpg-quality", type=int, default=92)
    p.add_argument("--no-skip-existing", action="store_true",
                   help="перенарезать видео, даже если кадры уже есть")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sample_all(
        data_dir=args.data, out_dir=args.out, fps=args.fps,
        max_side=args.max_side, jpg_quality=args.jpg_quality,
        skip_existing=not args.no_skip_existing,
    )
