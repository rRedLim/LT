"""ml_core/src/data/finalize_csv.py — финализирует CSV после ручной правки в Excel.

Делает 2 вещи:
    1) валидирует формат (через src.data.validate_csv)
    2) если ок — отрезает служебные колонки __trk_id, __preview_crop,
       __preview_frame и пишет «чистый» CSV в формате организаторов.

Пример:
    py -m src.data.finalize_csv --in labels_myself/1__editable.csv --out datasets/ocr_raw/1.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from src.data.validate_csv import validate_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="editable CSV (с __trk_id / __preview_*)")
    ap.add_argument("--out", type=Path, required=True,
                    help="чистый CSV в формате организаторов")
    ap.add_argument("--skip-validation", action="store_true",
                    help="не валидировать (для отладки)")
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"не найдено: {args.inp}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_validation:
        rc = validate_file(args.inp)
        if rc != 0:
            print(f"\nERROR: {args.inp} не прошёл валидацию — финал не создан",
                  file=sys.stderr)
            sys.exit(2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Читаем через csv.reader и пишем вручную (без csv.writer — он сам решает
    # когда кавычить, а нам нужно сохранить точный формат CSV организаторов
    # с двумя локалями: запятая+кавычки vs точка-без-кавычек).
    with open(args.inp, encoding="utf-8", newline="") as fi, \
         open(args.out, "w", encoding="utf-8", newline="") as fo:
        rdr = csv.reader(fi)
        hdr = next(rdr)
        keep = [i for i, c in enumerate(hdr) if not c.startswith("__")]
        fo.write(",".join(hdr[i] for i in keep) + "\n")
        for row in rdr:
            cells = [row[i] for i in keep]
            # safety net: если значение содержит запятую и не обёрнуто в кавычки —
            # обернём (Excel мог вернуть кавычки как-то иначе).
            fixed = []
            for v in cells:
                if "," in v and not (v.startswith('"') and v.endswith('"')):
                    v = '"' + v.replace('"', '""') + '"'
                fixed.append(v)
            fo.write(",".join(fixed) + "\n")

    print(f"  ✓ финал: {args.out}")


if __name__ == "__main__":
    main()
