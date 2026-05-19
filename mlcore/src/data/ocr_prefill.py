"""Прогоняет YOLO+VLM на каждое видео из dataset/dataset_myself/ → __editable.csv.

После авто-prefill пользователь правит руками __editable.csv в Excel UTF-8,
затем `finalize_csv.py` отрезает служебные колонки.

Использует `src.inference.pipeline.run()` напрямую (без subprocess к legacy).
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from src.inference.pipeline import run as run_inference


log = logging.getLogger(__name__)

# Служебные колонки для Excel-flow — добавляются справа после 29 эталонных
SERVICE_COLUMNS = ["__trk_id", "__preview_crop", "__preview_frame"]


def _safe_relative_to(target: Path, base: Path) -> str:
    """relative_to с защитой от разных дисков на Windows.

    При невозможности построить относительный путь (`ValueError` от Path.relative_to)
    возвращаем абсолютный путь — Excel сможет открыть его как гиперссылку.
    """
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target.resolve())


def _collect_track_meta(tracks_dir: Path) -> dict[tuple[int, str], int]:
    """Сканирует tracks_dir/trk_NNNN_final.json и возвращает маппинг
    (frame_timestamp, "x1.x_y1.y_x2.x_y2.y") → track_id.

    Это нужно потому что track_id из ByteTrack нелинейный (1, 3, 7...) и не
    совпадает с порядковым номером строки в CSV. Маппим строки CSV на
    debug-артефакты по точному совпадению ts + bbox.
    """
    out: dict[tuple[int, str], int] = {}
    if not tracks_dir.exists():
        return out
    for fp in tracks_dir.glob("trk_*_final.json"):
        try:
            stem = fp.stem  # trk_0042_final
            parts = stem.split("_")
            trk_id = int(parts[1])
        except (ValueError, IndexError):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ts = data.get("frame_timestamp")
        try:
            ts_int = int(float(str(ts).replace(",", ".")))
        except (ValueError, TypeError):
            continue
        bbox_key = _bbox_key(
            data.get("x_min"), data.get("y_min"),
            data.get("x_max"), data.get("y_max"),
        )
        out[(ts_int, bbox_key)] = trk_id
    return out


def _bbox_key(x_min, y_min, x_max, y_max) -> str:
    """Нормализует bbox в строку для словарного ключа. Округляем до 0.1 —
    достаточно для сопоставления строки CSV с debug-артефактом."""
    def _norm(v):
        try:
            return f"{float(str(v).replace(',', '.')):.1f}"
        except (ValueError, TypeError):
            return ""
    return f"{_norm(x_min)}_{_norm(y_min)}_{_norm(x_max)}_{_norm(y_max)}"


def _add_service_columns(csv_path: Path, previews_dir: Path,
                         track_meta: dict[tuple[int, str], int]) -> None:
    """Дописываем 3 служебные колонки к 29 эталонным → <csv>__editable.csv.

    track_id определяется по совпадению (frame_timestamp, bbox) с debug-артефактом.
    Если совпадения нет — __trk_id="" и превью-ссылки пустые.

    Гиперссылки в Excel читаются относительно расположения CSV.
    """
    # utf-8-sig читает и без BOM, и с BOM (Excel может пересохранить с BOM)
    rows: list[list[str]] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    # Найдём индексы нужных колонок в header
    try:
        idx_ts = header.index("frame_timestamp")
        idx_x1 = header.index("x_min")
        idx_y1 = header.index("y_min")
        idx_x2 = header.index("x_max")
        idx_y2 = header.index("y_max")
    except ValueError as e:
        log.error("CSV %s missing required column: %s", csv_path, e)
        return

    out_path = csv_path.with_name(csv_path.stem + "__editable.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + SERVICE_COLUMNS)
        n_matched = 0
        for row in rows:
            try:
                ts_int = int(float(row[idx_ts].replace(",", ".")))
            except (ValueError, IndexError):
                writer.writerow(row + ["", "", ""])
                continue
            bbox_k = _bbox_key(row[idx_x1], row[idx_y1], row[idx_x2], row[idx_y2])
            trk_id = track_meta.get((ts_int, bbox_k))
            if trk_id is None:
                writer.writerow(row + ["", "", ""])
                continue
            n_matched += 1
            crop_path = previews_dir / f"trk_{trk_id:04d}_crop.jpg"
            frame_path = previews_dir / f"trk_{trk_id:04d}_frame.jpg"
            crop_rel = (
                _safe_relative_to(crop_path, csv_path.parent)
                if crop_path.exists() else ""
            )
            frame_rel = (
                _safe_relative_to(frame_path, csv_path.parent)
                if frame_path.exists() else ""
            )
            writer.writerow(row + [str(trk_id), crop_rel, frame_rel])
    log.info("editable CSV: %s (matched %d/%d rows to track artifacts)",
             out_path, n_matched, len(rows))


def run_prefill(
    data_dir: Path,
    yolo_weights: Path,
    lora_adapter: Path,
    out_dir: Path,
    only_videos: Optional[list[str]] = None,
) -> int:
    """Прогоняет inference на каждом видео из data_dir.

    data_dir — папка с плоской структурой видео-файлов (../dataset/dataset_myself).
    yolo_weights, lora_adapter — пути к обученным весам.
    out_dir — куда писать <video>.csv, <video>__editable.csv и <video>_previews/.
    only_videos — если задан, фильтрует видео по stem (без расширения), напр. ["1","2","5"].

    Возвращает количество успешно обработанных видео.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    videos: List[Path] = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}
    )
    if only_videos:
        wanted = set(only_videos)
        videos = [v for v in videos if v.stem in wanted]
    if not videos:
        log.warning("В %s не нашлось .mp4/.mov файлов (filter=%s)",
                    data_dir, only_videos)
        return 0

    debug_root = out_dir / "_debug"
    n_ok = 0
    for v in videos:
        csv_out = out_dir / f"{v.stem}.csv"
        previews_dir = out_dir / f"{v.stem}_previews"
        try:
            result = run_inference(
                v, yolo_weights, lora_adapter, csv_out,
                debug=True, debug_dir=debug_root,
                # Отключаем сохранение frames_with_bbox (большие файлы) —
                # для prefill нужны только crop+frame на каждый track.
                debug_every_n_frames=10**9,
                # Skip barcode VLM fallback — 3 VLM-вызова на трек × десятки
                # треков × 16 видео = часы. Для prefill barcode правится в Excel.
                skip_barcode_vlm_fallback=True,
            )
        except Exception as exc:
            log.warning("prefill failed for %s: %s", v.name, exc)
            continue

        # Переносим crop/frame из debug/<stem>/tracks/ → <stem>_previews/.
        src_tracks = debug_root / v.stem / "tracks"
        track_meta: dict[tuple[int, str], int] = _collect_track_meta(src_tracks)
        if src_tracks.exists():
            previews_dir.mkdir(parents=True, exist_ok=True)
            for f in src_tracks.glob("trk_*_crop.jpg"):
                shutil.copy2(f, previews_dir / f.name)
            for f in src_tracks.glob("trk_*_frame.jpg"):
                shutil.copy2(f, previews_dir / f.name)

        _add_service_columns(csv_out, previews_dir, track_meta)
        log.info("OK %s: rows=%s tracks=%s", v.name,
                 result.get("rows", "?"), result.get("tracks", "?"))
        n_ok += 1

    # Чистим _debug — там же лежали pipeline.log и summary.json, для prefill не нужны.
    if debug_root.exists():
        shutil.rmtree(debug_root, ignore_errors=True)
    return n_ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("../dataset/dataset_myself"),
                   help="папка с видео (плоская структура)")
    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--lora-adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("labels_myself"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    n = run_prefill(args.data, args.yolo_weights, args.lora_adapter, args.out)
    print(f"Processed {n} videos.")
