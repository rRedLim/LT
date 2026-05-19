#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_core/src/data/validate_csv.py — проверяет, что после правок в Excel CSV
по-прежнему валидный.

Чек-лист (из ORGANIZER_REQUIREMENTS.md):
    - 29 колонок строго в правильном порядке
    - frame_timestamp = целое число (int, не "0,0")
    - price_default/price_card/price_discount: число с запятой, в кавычках
    - x_min/y_min/x_max/y_max: число с запятой, в кавычках
    - *_qr / wholesale_level_*_price / action_price_qr: число с точкой, без кавычек
    - barcode: 13 цифр + EAN-13 checksum (если не "" и не "нет")
    - id_sku: только цифры или "нет"/""
    - print_datetime: dd.mm.yyyy HH:MM или "нет"/""
    - discount_amount: -NN% или "нет"/""

Запуск:
    python -m src.data.validate_csv labels_myself/1__editable.csv

Возвращает exit-code 0 если всё ок, 1 если ошибки.
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path


CSV_COLUMNS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max", "qr_code_barcode",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]
COMMA_DECIMAL = {"price_default", "price_card", "price_discount",
                 "x_min", "y_min", "x_max", "y_max"}
DOT_DECIMAL = {"price1_qr", "price2_qr", "price3_qr", "price4_qr",
               "wholesale_level_1_price", "wholesale_level_2_price",
               "action_price_qr"}
INT_FIELDS = {"frame_timestamp"}


def is_int(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", s.strip()))


def is_comma_decimal(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+,\d+", s.strip()))


def is_dot_decimal(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+\.\d+", s.strip()))


def ean13_ok(s: str) -> bool:
    if not re.fullmatch(r"\d{13}", s): return False
    digits = [int(c) for c in s]
    s_ = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:-1]))
    return (10 - s_ % 10) % 10 == digits[-1]


def check_row(row: dict, lineno: int, errs: list[str]):
    for col in CSV_COLUMNS:
        v = row.get(col, "").strip()
        if v in ("", "нет"):
            continue
        if col in INT_FIELDS:
            if not is_int(v):
                errs.append(f"L{lineno} [{col}]: ожидалось целое, получено {v!r}")
        elif col in COMMA_DECIMAL:
            if not is_comma_decimal(v):
                errs.append(f"L{lineno} [{col}]: ожидалось число с запятой "
                            f"(например '129,99'), получено {v!r}")
        elif col in DOT_DECIMAL:
            if not is_dot_decimal(v):
                errs.append(f"L{lineno} [{col}]: ожидалось число с точкой "
                            f"(например '129.99'), получено {v!r}")
        elif col == "barcode":
            if not ean13_ok(v):
                errs.append(f"L{lineno} [{col}]: невалидный EAN-13 {v!r}")
        elif col == "qr_code_barcode":
            if not ean13_ok(v):
                errs.append(f"L{lineno} [{col}]: невалидный EAN-13 {v!r}")
        elif col == "id_sku":
            if not re.fullmatch(r"\d+", v):
                errs.append(f"L{lineno} [{col}]: должны быть только цифры, "
                            f"получено {v!r}")
        elif col == "discount_amount":
            if not re.fullmatch(r"-\d{1,2}%", v):
                errs.append(f"L{lineno} [{col}]: ожидался '-NN%', получено {v!r}")
        elif col == "print_datetime":
            if not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}:\d{2}", v):
                errs.append(f"L{lineno} [{col}]: ожидался 'dd.mm.yyyy HH:MM', "
                            f"получено {v!r}")


def validate_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    rdr = csv.DictReader(text.splitlines())
    cols = list(rdr.fieldnames or [])
    # editable CSV содержит __trk_id, __preview_crop, __preview_frame в начале
    editable_extras = {"__trk_id", "__preview_crop", "__preview_frame"}
    base_cols = [c for c in cols if c not in editable_extras]

    errs: list[str] = []
    if base_cols != CSV_COLUMNS:
        errs.append(f"Несоответствие схемы колонок:")
        errs.append(f"  ожидалось: {CSV_COLUMNS}")
        errs.append(f"  получено:  {base_cols}")
        # дальше всё равно проверим строки

    for i, row in enumerate(rdr, start=2):  # с 2-й строки в файле
        check_row(row, i, errs)

    print(f"\n=== {path} ===")
    if errs:
        for e in errs:
            print(f"  ✗ {e}")
        print(f"\nОшибок: {len(errs)}")
        return 1
    print(f"  ✓ всё ок ({path.name})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path,
                    help="один или несколько CSV для проверки")
    args = ap.parse_args()
    rc = 0
    for p in args.paths:
        if not p.exists():
            print(f"не найдено: {p}", file=sys.stderr)
            rc = 1; continue
        rc |= validate_file(p)
    sys.exit(rc)


if __name__ == "__main__":
    main()
