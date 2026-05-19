"""Auto-prefill QR-полей в CSV-разметке твоих фоток через декодирование QR.

Использование:
    py scripts/qr_prefill_photos.py --photos-dir ../dataset/my_photos

Структура `--photos-dir`:
    my_photos/
    ├── labels.csv                ← один CSV с 29 колонками (порядок как у орг)
    │                                в колонке `filename` — имя фотки в этой папке
    ├── 001.jpg                   ← или с подпапками: filename=`sub/001.jpg`
    ├── 002.jpg
    └── ...

Что делает:
    1. Читает `labels.csv` (UTF-8, авто-определяет имя если файл один).
    2. Для каждой строки находит фотку по `filename`, пытается декодить QR
       (zxing-cpp → cv2 → препроцессинги).
    3. Парсит payload в 11 канонических полей (qr_code_barcode, price{1..4}_qr,
       wholesale_*, action_*). Если в QR пришёл штрихкод, дополнительно
       заполняет основное `barcode` поле (если оно пустое).
    4. **Не затирает** уже заполненные значения — только пустые ячейки.
       Поля со значением "нет" тоже не трогаем (физически отсутствуют).
    5. Перезаписывает CSV через `csv_writer.write_csv` (правильная локаль:
       price{1..4}_qr с точкой, основные цены с запятой).
       Делает бэкап исходника: `labels.csv.bak`.
    6. Печатает отчёт: сколько фоток, у скольких QR расшифрован, сколько
       ячеек реально дозаполнено по каждой колонке.

После работы скрипта правишь оставшиеся поля (product_name, color, etc.)
руками в Excel UTF-8, потом запускаешь `train_lora.py` напрямую.
"""
from __future__ import annotations

# sys.path bootstrap
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
import csv
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.data.qr_decode import (
    decode_qr_from_file,
    parse_qr_payload,
    ALL_QR_FIELDS,
)
from src.inference.csv_writer import COLUMNS, write_csv


log = logging.getLogger(__name__)


def _find_single_csv(photos_dir: Path) -> Path:
    """Найти ровно один *.csv в photos_dir. Если их несколько — ошибка."""
    csvs = sorted(p for p in photos_dir.glob("*.csv") if not p.name.endswith(".bak"))
    if not csvs:
        raise FileNotFoundError(
            f"В {photos_dir} не найдено ни одного .csv с разметкой. "
            f"Положи CSV с 29 колонками (filename, product_name, ...)."
        )
    if len(csvs) > 1:
        raise RuntimeError(
            f"В {photos_dir} найдено несколько CSV: {[c.name for c in csvs]}. "
            f"Оставь один или укажи --csv-path."
        )
    return csvs[0]


def _is_blank(v: str) -> bool:
    """Пустое значение — то, которое можно заполнить из QR.
    Значение 'нет' НЕ заполняем (поле отсутствует физически)."""
    s = (v or "").strip()
    if s == "":
        return True
    return False


def _read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Прочитать CSV → (header, rows). Header валидируется на 29 колонок."""
    # utf-8-sig — поддержка BOM (Excel может пересохранить с BOM)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        missing = [c for c in COLUMNS if c not in header]
        extra = [c for c in header if c not in COLUMNS]
        if missing:
            raise ValueError(
                f"В CSV не хватает колонок: {missing}. "
                f"Ожидаемый порядок: {COLUMNS}"
            )
        if extra:
            log.warning("Игнорируем лишние колонки в CSV: %s", extra)
        rows = [{k: (r.get(k) or "") for k in COLUMNS} for r in reader]
    return header, rows


def _resolve_photo_path(photos_dir: Path, filename: str) -> Path | None:
    """filename в CSV может быть относительным к photos_dir или с подпапкой."""
    if not filename:
        return None
    # 1) Прямо в photos_dir
    cand = photos_dir / filename
    if cand.exists():
        return cand
    # 2) basename — если в CSV записан путь типа "subdir/001.jpg" а фотка лежит в корне
    cand2 = photos_dir / Path(filename).name
    if cand2.exists():
        return cand2
    return None


def prefill_csv(
    photos_dir: Path,
    csv_path: Path | None = None,
    backup_suffix: str = ".bak",
) -> dict:
    """Основная логика. См. docstring модуля.

    Возвращает dict со статистикой: rows, photos_found, qr_decoded,
    filled_per_field, sources.
    """
    if csv_path is None:
        csv_path = _find_single_csv(photos_dir)
    log.info("Reading CSV: %s", csv_path)
    header, rows = _read_rows(csv_path)
    log.info("Got %d rows × %d columns", len(rows), len(header))

    # Поля, которые ЕДИНСТВЕННО можно дописать из QR
    # qr_code_barcode и 10 qr-полей. Плюс главный `barcode` дублируется
    # из qr_code_barcode когда тот пустой (часто значения совпадают).
    fillable = list(ALL_QR_FIELDS)

    filled_per_field: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    n_photos_found = 0
    n_qr_decoded = 0
    n_no_photo = 0

    for i, row in enumerate(rows):
        fname = (row.get("filename") or "").strip()
        photo_path = _resolve_photo_path(photos_dir, fname)
        if photo_path is None:
            n_no_photo += 1
            log.debug("row %d: no photo for filename=%r", i, fname)
            continue
        n_photos_found += 1

        payload, source = decode_qr_from_file(photo_path)
        if not payload:
            sources["no_decode"] += 1
            continue
        n_qr_decoded += 1
        sources[source] += 1

        parsed = parse_qr_payload(payload)
        if not parsed:
            log.debug("row %d: payload=%r не распарсился", i, payload[:80])
            continue

        # Заполняем только пустые ячейки. 'нет' не трогаем.
        for field in fillable:
            new_val = parsed.get(field)
            if new_val is None or str(new_val).strip() == "":
                continue
            if not _is_blank(row.get(field, "")):
                continue
            row[field] = str(new_val).strip()
            filled_per_field[field] += 1

        # Дублирование штрихкода: если основной `barcode` пустой,
        # а из QR пришёл qr_code_barcode — переносим. Барк-код один и тот же.
        if _is_blank(row.get("barcode", "")) and not _is_blank(row.get("qr_code_barcode", "")):
            row["barcode"] = row["qr_code_barcode"]
            filled_per_field["barcode"] += 1

    # Бэкап
    backup_path = csv_path.with_suffix(csv_path.suffix + backup_suffix)
    shutil.copy2(csv_path, backup_path)
    log.info("Backup: %s", backup_path)

    # Перезапись через write_csv — правильное форматирование локалей
    write_csv(rows, csv_path)
    log.info("Wrote updated CSV: %s", csv_path)

    return {
        "csv_path": str(csv_path),
        "backup_path": str(backup_path),
        "rows": len(rows),
        "photos_found": n_photos_found,
        "photos_missing": n_no_photo,
        "qr_decoded": n_qr_decoded,
        "qr_decode_rate": (round(n_qr_decoded / n_photos_found, 3)
                           if n_photos_found else 0.0),
        "filled_per_field": dict(filled_per_field),
        "decode_sources": dict(sources),
    }


def _print_report(stats: dict) -> None:
    print()
    print("=" * 64)
    print(f"  rows in CSV:           {stats['rows']}")
    print(f"  photos found on disk:  {stats['photos_found']}")
    print(f"  photos missing:        {stats['photos_missing']}")
    print(f"  QR decoded:            {stats['qr_decoded']}"
          f" ({stats['qr_decode_rate']*100:.1f}%)")
    print()
    print("  Cells filled per field:")
    if not stats["filled_per_field"]:
        print("    (none — все qr-поля уже были заполнены, или QR не декодировался)")
    for field, n in sorted(stats["filled_per_field"].items(),
                           key=lambda kv: -kv[1]):
        print(f"    {field:<28} {n}")
    print()
    print("  Decode sources:")
    for src, n in sorted(stats["decode_sources"].items(),
                         key=lambda kv: -kv[1]):
        print(f"    {src:<28} {n}")
    print("=" * 64)
    print(f"  Backup: {stats['backup_path']}")
    print(f"  Output: {stats['csv_path']}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--photos-dir", type=Path, required=True,
                   help="папка с фотками и CSV-разметкой")
    p.add_argument("--csv-path", type=Path, default=None,
                   help="явный путь к CSV. По умолчанию — единственный *.csv в --photos-dir.")
    p.add_argument("--no-backup", action="store_true",
                   help="не делать .bak копию исходного CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    stats = prefill_csv(
        args.photos_dir,
        csv_path=args.csv_path,
        backup_suffix="" if args.no_backup else ".bak",
    )
    _print_report(stats)
