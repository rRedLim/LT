"""Юнит-тесты для qr_decode: парсер payload и базовое декодирование."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.qr_decode import (
    parse_qr_payload,
    decode_qr_from_image,
    decode_qr_from_file,
    _normalize_price,
    ALL_QR_FIELDS,
)


# ── parse_qr_payload ────────────────────────────────────────────────────────
def test_parse_full_aliases():
    p = "barcode=4670025474665|price1=252.63|price2=239.99|price4=129.99|actionPrice=129.99|actionCode=PROMO5"
    out = parse_qr_payload(p)
    assert out["qr_code_barcode"] == "4670025474665"
    assert out["price1_qr"] == "252.63"
    assert out["price2_qr"] == "239.99"
    assert out["price4_qr"] == "129.99"
    assert out["action_price_qr"] == "129.99"
    assert out["action_code_qr"] == "PROMO5"


def test_parse_short_aliases():
    p = "b=4607054890123|p1=89.99|p4=79.99|aP=79.99"
    out = parse_qr_payload(p)
    assert out["qr_code_barcode"] == "4607054890123"
    assert out["price1_qr"] == "89.99"
    assert out["price4_qr"] == "79.99"
    assert out["action_price_qr"] == "79.99"


def test_parse_wholesale_levels():
    p = "b=4607054890123|wL1C=3|wL1P=99.00|wL2C=10|wL2P=89.00"
    out = parse_qr_payload(p)
    assert out["wholesale_level_1_count"] == "3"
    assert out["wholesale_level_1_price"] == "99.00"
    assert out["wholesale_level_2_count"] == "10"
    assert out["wholesale_level_2_price"] == "89.00"


def test_parse_bare_ean13():
    out = parse_qr_payload("4670025474665")
    assert out == {"qr_code_barcode": "4670025474665"}


def test_parse_bare_text_goes_to_extra():
    out = parse_qr_payload("just some text without keys")
    assert "_extra" in out
    assert "qr_code_barcode" not in out


def test_parse_empty():
    assert parse_qr_payload("") == {}
    assert parse_qr_payload(None) == {}  # type: ignore[arg-type]


def test_parse_url_encoded():
    # `b=4670025474665|p1=252.63` URL-encoded
    p = "b%3D4670025474665%7Cp1%3D252.63"
    out = parse_qr_payload(p)
    assert out["qr_code_barcode"] == "4670025474665"
    assert out["price1_qr"] == "252.63"


def test_parse_colon_separator():
    """Альтернативный разделитель — ':'."""
    p = "barcode:4670025474665;price1:252.63"
    out = parse_qr_payload(p)
    assert out["qr_code_barcode"] == "4670025474665"
    assert out["price1_qr"] == "252.63"


def test_parse_price_normalization():
    # запятая вместо точки в payload
    out = parse_qr_payload("p1=252,63")
    assert out["price1_qr"] == "252.63"
    # целое
    out = parse_qr_payload("p1=129")
    assert out["price1_qr"] == "129.00"
    # один знак после
    out = parse_qr_payload("p1=129.9")
    assert out["price1_qr"] == "129.90"


def test_parse_unknown_keys_go_to_extra():
    p = "b=4670025474665|unknownKey=hello|p1=99.99"
    out = parse_qr_payload(p)
    assert out["qr_code_barcode"] == "4670025474665"
    assert out["price1_qr"] == "99.99"
    assert "_extra" in out
    assert "unknownKey" in out["_extra"]


def test_all_qr_fields_constant():
    # На случай рефакторинга — констант должно быть 11
    assert len(ALL_QR_FIELDS) == 11
    assert "qr_code_barcode" in ALL_QR_FIELDS
    assert "action_code_qr" in ALL_QR_FIELDS


def test_normalize_price_edge_cases():
    assert _normalize_price("129.99") == "129.99"
    assert _normalize_price("129,99") == "129.99"
    assert _normalize_price("129") == "129.00"
    assert _normalize_price("129.9 руб") == "129.90"
    # не цена — возвращается как есть
    assert _normalize_price("hello") == "hello"


# ── decode_qr_from_image (с реальным синтезированным QR) ────────────────────
def _make_qr_image(payload: str, box_size: int = 10) -> np.ndarray:
    """Создать BGR-картинку с QR-кодом и payload."""
    import qrcode
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1].copy()  # RGB → BGR
    return arr


def test_decode_qr_synthetic_simple():
    """cv2.QRCodeDetector должен прочитать чистый синтетический QR."""
    payload = "b=4670025474665|p1=252.63|p4=129.99"
    img = _make_qr_image(payload, box_size=10)
    decoded, source = decode_qr_from_image(img)
    assert decoded == payload, f"decoded={decoded!r} source={source!r}"
    assert source != ""


def test_decode_qr_empty_image():
    """Картинка без QR → ('', '')."""
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    decoded, source = decode_qr_from_image(img)
    assert decoded == ""
    assert source == ""


def test_decode_qr_none_or_empty():
    """None / пустой массив → ('', '')."""
    assert decode_qr_from_image(None) == ("", "")  # type: ignore[arg-type]
    assert decode_qr_from_image(np.array([])) == ("", "")


def test_decode_qr_from_file(tmp_path: Path):
    """decode_qr_from_file: сохранили PNG, прочитали."""
    payload = "b=4607054890123"
    img = _make_qr_image(payload, box_size=10)
    p = tmp_path / "qr.png"
    cv2.imwrite(str(p), img)
    decoded, source = decode_qr_from_file(p)
    assert decoded == payload, f"decoded={decoded!r} source={source!r}"


def test_decode_qr_from_missing_file(tmp_path: Path):
    """Несуществующий файл → ('', '')."""
    decoded, source = decode_qr_from_file(tmp_path / "nope.png")
    assert decoded == ""
    assert source == ""
