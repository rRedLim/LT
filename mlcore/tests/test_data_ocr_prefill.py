"""Тесты для маппинга trk_id и _bbox_key в ocr_prefill."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.ocr_prefill import (
    _bbox_key,
    _collect_track_meta,
    _add_service_columns,
    _safe_relative_to,
)


def test_bbox_key_round_trips_comma_decimal():
    """CSV пишет "2063,1", final.json пишет 2063.1 — оба должны дать один ключ."""
    k1 = _bbox_key("2063,1", "1929,8", "2296,6", "2126,1")
    k2 = _bbox_key(2063.123, 1929.8, 2296.6, 2126.1)
    assert k1 == k2 == "2063.1_1929.8_2296.6_2126.1"


def test_bbox_key_handles_invalid():
    """Невалидные значения не падают, дают пустые строки."""
    k = _bbox_key("", "", "", "")
    assert k == "_" * 3  # 4 пустых поля = "___"


def test_collect_track_meta_reads_final_json(tmp_path):
    """Создаём fake tracks/ с двумя trk_*_final.json и проверяем маппинг."""
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    (tracks_dir / "trk_0001_final.json").write_text(json.dumps({
        "frame_timestamp": 1500,
        "x_min": 100.0, "y_min": 200.0, "x_max": 300.0, "y_max": 400.0,
    }), encoding="utf-8")
    (tracks_dir / "trk_0007_final.json").write_text(json.dumps({
        "frame_timestamp": 3000,
        "x_min": 50.5, "y_min": 60.5, "x_max": 150.5, "y_max": 160.5,
    }), encoding="utf-8")
    meta = _collect_track_meta(tracks_dir)
    assert meta[(1500, "100.0_200.0_300.0_400.0")] == 1
    assert meta[(3000, "50.5_60.5_150.5_160.5")] == 7


def test_collect_track_meta_missing_dir():
    """Несуществующая папка → пустой dict, не падает."""
    assert _collect_track_meta(Path("/no/such/path")) == {}


def test_add_service_columns_matches_real_track_ids(tmp_path):
    """Главный тест на CRITICAL #1: track_id из ByteTrack нелинейный (1, 7),
    а строки CSV идут в своём порядке. Маппинг должен работать."""
    # CSV с 2 строками. Формат как у write_csv (29 колонок).
    csv_path = tmp_path / "v.csv"
    cols = (
        "filename,product_name,price_default,price_card,price_discount,"
        "barcode,discount_amount,id_sku,print_datetime,code,"
        "additional_info,color,special_symbols,frame_timestamp,"
        "x_min,y_min,x_max,y_max,qr_code_barcode,"
        "price1_qr,price2_qr,price3_qr,price4_qr,"
        "wholesale_level_1_count,wholesale_level_1_price,"
        "wholesale_level_2_count,wholesale_level_2_price,"
        "action_price_qr,action_code_qr"
    ).split(",")
    assert len(cols) == 29

    def _empty_row(ts: str, x_min: str, y_min: str, x_max: str, y_max: str) -> list[str]:
        row = [""] * 29
        # Заполняем нужные поля
        row[cols.index("filename")] = "video.mp4"
        row[cols.index("frame_timestamp")] = ts
        row[cols.index("x_min")] = f'"{x_min}"'   # как пишет write_csv
        row[cols.index("y_min")] = f'"{y_min}"'
        row[cols.index("x_max")] = f'"{x_max}"'
        row[cols.index("y_max")] = f'"{y_max}"'
        return row

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        # Заголовок без кавычек, строки с кавычками вокруг bbox — как write_csv
        f.write(",".join(cols) + "\n")
        # row 0: ts=1500, bbox=100..400 → trk_id=1
        # row 1: ts=3000, bbox=50.5..160.5 → trk_id=7
        # (track_id нелинейный — CRITICAL #1 баг был именно в этом)
        f.write(",".join(_empty_row("1500", "100,0", "200,0", "300,0", "400,0")) + "\n")
        f.write(",".join(_empty_row("3000", "50,5", "60,5", "150,5", "160,5")) + "\n")

    previews_dir = tmp_path / "v_previews"
    previews_dir.mkdir()
    # Создаём fake превью для trk_0001 и trk_0007
    (previews_dir / "trk_0001_crop.jpg").write_bytes(b"")
    (previews_dir / "trk_0001_frame.jpg").write_bytes(b"")
    (previews_dir / "trk_0007_crop.jpg").write_bytes(b"")
    (previews_dir / "trk_0007_frame.jpg").write_bytes(b"")

    track_meta = {
        (1500, "100.0_200.0_300.0_400.0"): 1,
        (3000, "50.5_60.5_150.5_160.5"): 7,
    }
    _add_service_columns(csv_path, previews_dir, track_meta)

    editable_path = csv_path.with_name(csv_path.stem + "__editable.csv")
    assert editable_path.exists()
    with open(editable_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header[-3:] == ["__trk_id", "__preview_crop", "__preview_frame"]
    assert len(body) == 2
    # Главная проверка: trk_id в строках = 1 и 7 (РЕАЛЬНЫЕ ID), а НЕ 0 и 1
    assert body[0][-3] == "1"
    assert body[1][-3] == "7"
    # Превью-пути не пустые
    assert "trk_0001_crop.jpg" in body[0][-2]
    assert "trk_0007_crop.jpg" in body[1][-2]


def test_safe_relative_to_handles_different_drives(tmp_path):
    """relative_to падает на разных дисках Windows — наш wrapper должен вернуть абсолютный."""
    # На любой ОС: если base и target в одной директории — относительный
    base = tmp_path / "a"
    target = tmp_path / "a" / "b" / "c.jpg"
    base.mkdir(exist_ok=True)
    (tmp_path / "a" / "b").mkdir(exist_ok=True)
    target.write_bytes(b"")
    rel = _safe_relative_to(target, base)
    assert "c.jpg" in rel

    # При невозможности — возвращает строку (не падает с ValueError)
    weird_target = Path("/some/totally/unrelated/path/file.jpg")
    result = _safe_relative_to(weird_target, base)
    assert isinstance(result, str)
    assert "file.jpg" in result
