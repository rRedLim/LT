import json
from pathlib import Path
import numpy as np
import cv2
from src.data.build_ocr_dataset import (
    row_to_messages, passes_filters, build_from_photos,
)


def test_passes_filters_ok():
    assert passes_filters(crop_h=200, sharpness=80.0, h_min=120, sharpness_min=40)


def test_passes_filters_low_h():
    assert not passes_filters(crop_h=100, sharpness=80.0, h_min=120, sharpness_min=40)


def test_passes_filters_low_sharp():
    assert not passes_filters(crop_h=200, sharpness=20.0, h_min=120, sharpness_min=40)


def test_row_to_messages_structure():
    row = {
        "product_name": "Сыр",
        "price_card": "129,99",
        "color": "red",
        "barcode": "4670025474665",
    }
    msgs = row_to_messages(row, image_path="/some/crop.jpg")
    # user-message: content = list блоков (image + text), как ожидает Qwen2.5-VL
    assert msgs[0]["role"] == "user"
    user_content = msgs[0]["content"]
    assert isinstance(user_content, list)
    types = [block.get("type") for block in user_content]
    assert "image" in types and "text" in types
    image_block = next(b for b in user_content if b["type"] == "image")
    assert image_block["image"] == "/some/crop.jpg"
    # assistant-message: content = JSON-строка с полями
    assert msgs[-1]["role"] == "assistant"
    parsed = json.loads(msgs[-1]["content"])
    assert parsed["product_name"] == "Сыр"
    # Все 23 поля FIELDS должны быть в parsed (даже если пустые)
    assert "code" in parsed
    assert len(parsed) == 23


def _write_jpg(p: Path, h: int = 200, w: int = 300) -> None:
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(p), img)


def _write_csv(p: Path, rows: list[dict]) -> None:
    """Записать CSV с 29 колонками в формате орг."""
    from src.inference.csv_writer import COLUMNS
    import csv as _csv
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            full = {c: r.get(c, "") for c in COLUMNS}
            w.writerow(full)


def test_build_from_photos_basic(tmp_path: Path):
    """Папка с CSV + фотками → JSONL с правильной структурой."""
    photos_dir = tmp_path / "photos"
    crops_dir = tmp_path / "crops"
    photos_dir.mkdir()

    _write_jpg(photos_dir / "001.jpg")
    _write_jpg(photos_dir / "002.png")
    _write_jpg(photos_dir / "003.jpg")  # есть фото — нет строки в CSV

    _write_csv(photos_dir / "labels.csv", [
        {"filename": "001.jpg", "product_name": "Молоко",
         "price_card": "89,99", "barcode": "4607054890123", "color": "yellow"},
        {"filename": "002.png", "product_name": "Хлеб",
         "price_default": "55,00"},
        # 003.jpg в CSV нет → не попадает
        # есть строка с filename которой нет на диске → пропускается
        {"filename": "999.jpg", "product_name": "Призрак"},
    ])

    records = build_from_photos(photos_dir, crops_dir)

    # Только строки с реальными фотками
    assert len(records) == 2
    # crops скопированы
    assert sum(1 for _ in crops_dir.iterdir()) == 2

    # Первая запись
    r0 = records[0]
    msgs = r0["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    parsed = json.loads(msgs[-1]["content"])
    assert parsed["product_name"] == "Молоко"
    assert parsed["barcode"] == "4607054890123"
    # отсутствующее поле → пустая строка
    assert parsed["id_sku"] == ""
    assert len(parsed) == 23


def test_build_from_photos_no_csv(tmp_path: Path):
    """Папка с фотками без CSV → пустой список."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _write_jpg(photos_dir / "001.jpg")
    assert build_from_photos(photos_dir, tmp_path / "crops") == []


def test_build_from_photos_nonexistent(tmp_path: Path):
    """Несуществующая папка → пустой список."""
    assert build_from_photos(tmp_path / "nope", tmp_path / "crops") == []
