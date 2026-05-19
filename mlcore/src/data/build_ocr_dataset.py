from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import csv
import json
import cv2
import numpy as np

from src.utils.crop import extract_crop
from src.utils.io import read_frame_at


# Источник истины — vlm_reader: и промпт, и список полей одинаковые между
# тренировкой и инференсом (иначе модель учится на одних полях, выдаёт другие).
from src.inference.vlm_reader import UNIVERSAL_PROMPT, FIELDS as FIELD_ORDER  # noqa: E402


def passes_filters(crop_h: int, sharpness: float, h_min: int = 120,
                   sharpness_min: float = 40.0) -> bool:
    """Return True only when crop height AND sharpness meet minimum thresholds."""
    return crop_h >= h_min and sharpness >= sharpness_min


def laplacian_var(crop: np.ndarray) -> float:
    """Variance of the Laplacian — proxy for image sharpness."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def row_to_messages(row: dict, image_path: str) -> list[dict]:
    """Build a two-message list (user + assistant) for LoRA fine-tuning.

    Формат `content` для user — список блоков (image + text), как ожидает
    Qwen2.5-VL processor.apply_chat_template. Это критично для совместимости
    с inference-стороной (vlm_reader.read_with_raw).
    """
    out = {k: (row.get(k) or "").strip() or "" for k in FIELD_ORDER}
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": UNIVERSAL_PROMPT},
        ]},
        {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
    ]


def _parse_coord(s: str) -> float:
    return float(str(s).strip().replace(",", "."))


def _crop_from_csv_row(video_path: Path, ts_ms: int, bbox) -> Optional[np.ndarray]:
    frame = read_frame_at(video_path, ts_ms)
    if frame is None:
        return None
    return extract_crop(frame, bbox, rotate=270)


def build_from_csv(
    csv_path: Path,
    video_path: Path,
    out_crops_dir: Path,
    h_min: int = 120,
    sharpness_min: float = 40.0,
) -> List[dict]:
    """Read CSV, cut crops, filter by quality, return list of JSONL records."""
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    out: List[dict] = []
    # utf-8-sig — на случай если Excel пересохранил CSV с BOM
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                # int(float(...)) — устойчиво к "1500" и "1500.0" (Excel может
                # пересохранить timestamp как float-строку при ручной правке).
                ts = int(float(str(row["frame_timestamp"]).replace(",", ".")))
                bbox = (
                    _parse_coord(row["x_min"]),
                    _parse_coord(row["y_min"]),
                    _parse_coord(row["x_max"]),
                    _parse_coord(row["y_max"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
            crop = _crop_from_csv_row(video_path, ts, bbox)
            if crop is None or crop.size == 0:
                continue
            sh = laplacian_var(crop)
            if not passes_filters(crop.shape[0], sh, h_min, sharpness_min):
                continue
            crop_path = out_crops_dir / f"{video_path.stem}_t{ts:07d}_b{i:04d}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out.append({
                "image": str(crop_path),
                "messages": row_to_messages(row, str(crop_path)),
            })
    return out


def build_from_photos(
    photos_dir: Path,
    out_crops_dir: Path,
    csv_path: Optional[Path] = None,
) -> List[dict]:
    """Собрать JSONL-записи из папки с фотками и единым CSV-разметкой.

    Формат входа:
        photos_dir/
            labels.csv            ← 29 колонок как у орг; filename — имя фотки
            001.jpg
            002.jpg
            ...

    Поле `filename` в каждой строке CSV ищется как `photos_dir/<filename>`
    (поддерживаются подпапки). Записи без найденного файла пропускаются.

    Фильтр sharpness НЕ применяется (фотки заведомо хороши; кропы орг и
    видео уже отфильтрованы выше).
    """
    out: List[dict] = []
    if not photos_dir.exists():
        return out
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csvs = sorted(p for p in photos_dir.glob("*.csv")
                      if not p.name.endswith(".bak"))
        if not csvs:
            print(f"  PHOTOS: в {photos_dir} нет .csv — пропускаю")
            return out
        if len(csvs) > 1:
            print(f"  PHOTOS: несколько CSV в {photos_dir}: {[c.name for c in csvs]}. "
                  f"Беру первый: {csvs[0].name}")
        csv_path = csvs[0]

    n_no_photo = 0
    n_rows = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, raw_row in enumerate(reader):
            n_rows += 1
            fname = (raw_row.get("filename") or "").strip()
            if not fname:
                continue
            # Резолвим путь: сначала как есть, потом basename
            src_img = photos_dir / fname
            if not src_img.exists():
                src_img = photos_dir / Path(fname).name
            if not src_img.exists():
                n_no_photo += 1
                continue

            # Берём 23 FIELDS из row (пустые → "").
            row = {k: (str(raw_row.get(k, "") or "").strip()) for k in FIELD_ORDER}

            # Копируем фотку в crops_dir под уникальным именем (на случай если
            # в CSV две строки на одну фотку — но обычно одна).
            dst = out_crops_dir / f"photo_{i:04d}_{Path(fname).name}"
            import shutil
            if dst.resolve() != src_img.resolve():
                shutil.copyfile(src_img, dst)

            out.append({
                "image": str(dst),
                "messages": row_to_messages(row, str(dst)),
            })

    if n_no_photo:
        print(f"  PHOTOS: {n_no_photo} строк без файла-фотки — пропущены")
    print(f"  PHOTOS: {len(out)}/{n_rows} строк из {csv_path.name} → JSONL")
    return out


def build(
    our_videos_root: Path,
    our_csvs_root: Path,
    organizer_root: Path,
    out_root: Path,
    val_frac: float = 0.2,
    h_min: int = 120,
    sharpness_min: float = 40.0,
    photos_dir: Optional[Path] = None,
    organizer_videos: Optional[List[str]] = None,
) -> dict:
    """Assemble train.jsonl and val.jsonl from:
    - our N videos: dataset_myself + datasets/ocr_raw/*.csv (опционально)
    - organiser videos: список через `organizer_videos`; дефолт — все 5
    - твои ручные фотки: `photos_dir` (опционально)

    Кропы из видео фильтруются sharpness≥`sharpness_min` + h≥`h_min`.
    Фотки в `photos_dir` НЕ фильтруются (заведомо хорошее качество).
    """
    crops_dir = out_root / "crops"
    out_root.mkdir(parents=True, exist_ok=True)
    all_records: List[dict] = []

    # ── Our videos (наши, до N штук — если CSV для них вообще есть) ──────────
    if our_csvs_root.exists():
        for csv_path in sorted(our_csvs_root.glob("*.csv")):
            vname = csv_path.stem
            cand = None
            for ext in [".mov", ".mp4", ".MOV", ".MP4"]:
                p = our_videos_root / f"{vname}{ext}"
                if p.exists():
                    cand = p
                    break
            if cand is None:
                print(f"  WARN: video not found for {csv_path.name}")
                continue
            recs = build_from_csv(csv_path, cand, crops_dir, h_min, sharpness_min)
            all_records.extend(recs)
            print(f"  ours/{csv_path.name}: {len(recs)} records")

    # ── Organiser videos ────────────────────────────────────────────────────
    if organizer_videos is None:
        organizer_videos = ["25_12-20", "26_12-20", "25_2-10", "43_15", "49_5"]
    for v in organizer_videos:
        vd = organizer_root / v
        csv_file = vd / f"{v}.csv"
        mp4_file = vd / f"{v}.mp4"
        if not csv_file.exists():
            print(f"  SKIP ORG {v}: csv not found")
            continue
        recs = build_from_csv(csv_file, mp4_file, crops_dir, h_min, sharpness_min)
        all_records.extend(recs)
        print(f"  ORG {v}: {len(recs)} records")

    # ── Manual photos (без фильтра sharpness/h) ─────────────────────────────
    if photos_dir is not None:
        recs = build_from_photos(photos_dir, crops_dir)
        all_records.extend(recs)
        print(f"  PHOTOS: {len(recs)} records (от {photos_dir})")

    # ── Stratified split: every val_step-th record → val ────────────────────
    val_step = int(round(1 / val_frac)) if val_frac > 0 else 0
    train, val = [], []
    for i, r in enumerate(all_records):
        if val_step and i % val_step == 0:
            val.append(r)
        else:
            train.append(r)

    with open(out_root / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_root / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "total": len(all_records),
        "train": len(train),
        "val": len(val),
        "crops_dir": str(crops_dir),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Build OCR LoRA fine-tune JSONL dataset from video + CSV annotations."
    )
    p.add_argument("--our-videos", type=Path,
                   default=Path("../dataset/dataset_myself"))
    p.add_argument("--our-csvs", type=Path,
                   default=Path("datasets/ocr_raw"))
    p.add_argument("--organizer-root", type=Path,
                   default=Path("../dataset/dataset_orig"))
    p.add_argument("--photos-dir", type=Path, default=None,
                   help="папка с парами image.jpg + image.json (23 поля). "
                        "Без фильтра sharpness/h.")
    p.add_argument("--organizer-videos", type=str, default=None,
                   help="comma-list имён видео орг. По умолчанию все 5.")
    p.add_argument("--out", type=Path, default=Path("datasets/ocr_v3"))
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--h-min", type=int, default=120)
    p.add_argument("--sharp-min", type=float, default=40.0)
    args = p.parse_args()

    org_vids = (
        [s.strip() for s in args.organizer_videos.split(",") if s.strip()]
        if args.organizer_videos else None
    )
    stats = build(
        args.our_videos, args.our_csvs, args.organizer_root, args.out,
        args.val_frac, args.h_min, args.sharp_min,
        photos_dir=args.photos_dir,
        organizer_videos=org_vids,
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")
