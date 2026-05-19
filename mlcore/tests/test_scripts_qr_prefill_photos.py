"""Integration-тест для scripts/qr_prefill_photos.py: синтезируем папку с
фотками + CSV, прогоняем prefill, проверяем что пустые QR-поля заполнились,
а уже заполненные / 'нет' — нет."""
from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Бутстрапим scripts/ в sys.path так же, как это делает сам скрипт
_ML_CORE = str(Path(__file__).resolve().parents[1])
if _ML_CORE not in sys.path:
    sys.path.insert(0, _ML_CORE)
_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from qr_prefill_photos import prefill_csv  # type: ignore
from src.inference.csv_writer import COLUMNS


def _make_qr_image(payload: str) -> np.ndarray:
    import qrcode
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10, border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return np.array(pil)[:, :, ::-1].copy()


def _write_csv(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def _read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(_csv.DictReader(f))


def test_prefill_fills_blank_fields(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()

    # 3 фотки: первая с полным QR, вторая с одним только штрихкодом, третья без QR
    cv2.imwrite(str(photos / "001.jpg"),
                _make_qr_image("b=4670025474665|p1=252.63|p4=129.99|aP=129.99"))
    cv2.imwrite(str(photos / "002.jpg"),
                _make_qr_image("4607054890123"))
    cv2.imwrite(str(photos / "003.jpg"),
                np.full((400, 400, 3), 255, dtype=np.uint8))

    # CSV: все qr-поля пустые, есть одно "нет" (его не трогаем) и одно
    # уже заполненное (его не трогаем).
    _write_csv(photos / "labels.csv", [
        {"filename": "001.jpg",
         "product_name": "Молоко",
         "price1_qr": "999.99",        # уже заполнено — не трогаем
         "price2_qr": "нет",            # явное отсутствие — не трогаем
         "color": "yellow"},
        {"filename": "002.jpg",
         "product_name": "Хлеб"},
        {"filename": "003.jpg",
         "product_name": "Призрак"},
    ])

    stats = prefill_csv(photos)
    assert stats["rows"] == 3
    assert stats["photos_found"] == 3
    assert stats["qr_decoded"] == 2          # 001 и 002 декодированы, 003 — нет
    assert stats["photos_missing"] == 0

    # Бэкап создан
    assert (photos / "labels.csv.bak").exists()

    rows = _read_csv(photos / "labels.csv")
    # Строка 1: 001.jpg
    r1 = rows[0]
    assert r1["filename"] == "001.jpg"
    assert r1["qr_code_barcode"] == "4670025474665"      # дозаполнено
    assert r1["barcode"] == "4670025474665"              # тоже (из qr_code_barcode)
    assert r1["price1_qr"] == "999.99" or r1["price1_qr"] == '999,99'  # НЕ затёрто
    # price2_qr был "нет" → должен остаться "нет"
    assert r1["price2_qr"] == "нет"
    # price4_qr / action_price_qr дозаполнены (формат с точкой т.к. DOT_NOQUOTE)
    assert r1["price4_qr"] == "129.99"
    assert r1["action_price_qr"] == "129.99"

    # Строка 2: 002.jpg (только штрихкод)
    r2 = rows[1]
    assert r2["qr_code_barcode"] == "4607054890123"
    assert r2["barcode"] == "4607054890123"
    assert r2["price1_qr"] == ""                          # QR не дал — пусто

    # Строка 3: 003.jpg (без QR)
    r3 = rows[2]
    assert r3["qr_code_barcode"] == ""
    assert r3["barcode"] == ""

    # Поля статистики по полям
    filled = stats["filled_per_field"]
    # qr_code_barcode дозаполнено в обеих первых строках
    assert filled.get("qr_code_barcode") == 2
    # barcode дозаполнено в обеих
    assert filled.get("barcode") == 2
    # price1_qr дозаполнено НЕ должно быть (в r1 было заполнено, в r2 нет в payload)
    assert "price1_qr" not in filled
    # price4_qr — только в r1
    assert filled.get("price4_qr") == 1


def test_prefill_missing_csv_raises(tmp_path: Path):
    photos = tmp_path / "no_csv"
    photos.mkdir()
    with pytest.raises(FileNotFoundError):
        prefill_csv(photos)


def test_prefill_missing_photo_counted(tmp_path: Path):
    """В CSV строка с filename которого нет на диске — учитывается в photos_missing."""
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "001.jpg"), _make_qr_image("b=4670025474665"))
    _write_csv(photos / "labels.csv", [
        {"filename": "001.jpg", "product_name": "Молоко"},
        {"filename": "999.jpg", "product_name": "Несуществующая"},
    ])
    stats = prefill_csv(photos)
    assert stats["photos_found"] == 1
    assert stats["photos_missing"] == 1
    assert stats["qr_decoded"] == 1
