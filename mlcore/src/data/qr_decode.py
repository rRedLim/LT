"""QR-декодер для фоток ценников + парсер payload в поля CSV.

Используется в scripts/qr_prefill_photos.py чтобы автоматически дополнить
qr-поля (qr_code_barcode, price1_qr..price4_qr, wholesale_*, action_*) в CSV
с разметкой твоих фоток до запуска LoRA-тюна.

Логика:
1. decode_qr_from_image(img) — пытается извлечь payload из BGR-картинки
   несколькими способами (zxing-cpp → cv2.QRCodeDetector → препроцессинги).
2. parse_qr_payload(payload) — разбирает строку payload по алиасам
   (b=..., p1=..., wL1C=...) в dict {canonical_field: value}.

Не зависит от тяжёлой логики из qr_extract_v2.py (multi-frame SR, finder-pattern
детектор) — фотки заведомо лучше кадров из видео, и базового декодирования
достаточно.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import urllib.parse
from typing import Optional

import cv2
import numpy as np


log = logging.getLogger(__name__)


# ── Маппинг ключей payload → канонические колонки CSV ───────────────────────
QR_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "qr_code_barcode":           ("barcode", "b"),
    "price1_qr":                 ("price1", "p1"),
    "price2_qr":                 ("price2", "p2"),
    "price3_qr":                 ("price3", "p3"),
    "price4_qr":                 ("price4", "p4"),
    "wholesale_level_1_count":   ("wholesaleLevel1Count", "wL1C"),
    "wholesale_level_1_price":   ("wholesaleLevel1Price", "wL1P"),
    "wholesale_level_2_count":   ("wholesaleLevel2Count", "wL2C"),
    "wholesale_level_2_price":   ("wholesaleLevel2Price", "wL2P"),
    "action_price_qr":           ("actionPrice", "aP"),
    "action_code_qr":            ("actionCode", "aC"),
}

_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in QR_KEY_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANON[_a.lower()] = _canon

ALL_QR_FIELDS: tuple[str, ...] = tuple(QR_KEY_ALIASES.keys())

# Цены в формате CSV орг — с запятой, два знака после ("129,99"). Если payload
# даёт "129.99" / "129" — приводим.
PRICE_FIELDS: frozenset[str] = frozenset({
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_price", "wholesale_level_2_price",
    "action_price_qr",
})

_KV_SPLIT_RE = re.compile(r"[|&;\n]+")
_KV_PAIR_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s*[=:]\s*(.+)$")
_EAN13_RE = re.compile(r"^\d{12,14}$")


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", str(s))


def _normalize_price(v: str) -> str:
    """Привести цену к каноничному виду '129.99' (с точкой, 2 знака).

    Точка — это формат для QR-полей в csv_writer.DOT_NOQUOTE. Конечное
    форматирование (точка/запятая, кавычки) делает csv_writer.write_csv.
    Здесь только нормализуем числовое значение.
    """
    s = str(v).strip()
    # быстрый путь: уже число
    m = re.search(r"(\d{1,5})[.,](\d{1,2})\b", s)
    if m:
        cents = m.group(2).ljust(2, "0")[:2]
        return f"{m.group(1)}.{cents}"
    if s.isdigit():
        return f"{s}.00"
    return s


def parse_qr_payload(payload: str) -> dict[str, str]:
    """Разбирает payload QR в dict {canonical_field: value}.

    Поддерживает:
      • `b=4670...|p1=252.63|p4=129.99` — основной формат Ленты;
      • `barcode:4670...;price1:252,63` — альтернативные разделители;
      • `4670025474665` — голый EAN-13, mapped в qr_code_barcode;
      • URL-encoded payload (`%3D`, `%7C` и т.п.).

    Цены приводятся к формату CSV (`'129,99'`), штрихкод — только цифры.
    Неизвестные ключи кладутся в `_extra` (не используется в prefill).
    """
    out: dict[str, str] = {}
    if not payload:
        return out
    raw = unicodedata.normalize("NFKC", str(payload)).strip()
    if "%" in raw:
        try:
            raw = urllib.parse.unquote(raw)
        except Exception:  # noqa: BLE001
            pass

    # Голый EAN-13/UPC
    if _EAN13_RE.match(raw):
        out["qr_code_barcode"] = raw
        return out

    if "=" not in raw and ":" not in raw:
        digits = _digits_only(raw)
        if 12 <= len(digits) <= 14:
            out["qr_code_barcode"] = digits
            return out
        out["_extra"] = raw
        return out

    extra: list[str] = []
    for part in (p.strip() for p in _KV_SPLIT_RE.split(raw) if p.strip()):
        m = _KV_PAIR_RE.match(part)
        if not m:
            extra.append(part)
            continue
        key_raw, val = m.group(1), m.group(2).strip()
        canon = _ALIAS_TO_CANON.get(key_raw.lower())
        if canon is None:
            extra.append(part)
            continue
        if canon == "qr_code_barcode":
            digits = _digits_only(val)
            out[canon] = digits or val
        elif canon in PRICE_FIELDS:
            out[canon] = _normalize_price(val)
        else:
            out[canon] = val
    if extra:
        out["_extra"] = " | ".join(extra)
    return out


# ── Декодирование изображения ───────────────────────────────────────────────
def _try_zxing(img: np.ndarray) -> Optional[str]:
    """zxing-cpp умеет читать QR + EAN, более устойчив чем cv2 на мелких QR."""
    try:
        import zxingcpp  # type: ignore
    except ImportError:
        return None
    try:
        results = zxingcpp.read_barcodes(img)
    except Exception:  # noqa: BLE001
        return None
    for r in results or []:
        if r.format == zxingcpp.BarcodeFormat.QRCode and r.text:
            return r.text
    # Fallback: вернём что угодно (EAN-13 на QR-меньшем — лучше чем ничего)
    for r in results or []:
        if r.text:
            return r.text
    return None


def _try_cv2(img: np.ndarray) -> Optional[str]:
    """cv2.QRCodeDetector — встроенный, быстрый."""
    try:
        det = cv2.QRCodeDetector()
        data, _, _ = det.detectAndDecode(img)
        if data:
            return data
    except cv2.error:
        return None
    return None


def _preprocess_variants(img: np.ndarray) -> list[np.ndarray]:
    """Лёгкие препроцессинги для деградированных QR.

    Не дублируем тяжёлый аппарат из qr_extract_v2 (multi-frame SR, custom
    finder-pattern) — на фотках обычно один-двух препроцессингов хватает.
    """
    variants: list[np.ndarray] = [img]
    h, w = img.shape[:2]

    # Апскейл до 1500 px по короткой стороне (cv2.QRCodeDetector любит крупные)
    short = min(h, w)
    if short < 1500 and short > 0:
        scale = 1500 / short
        up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(up)

    # Grayscale + sharpen unsharp
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else sharp)

    # Adaptive threshold (бинаризация — модели QR чёрно-белые)
    binar = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10,
    )
    variants.append(cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else binar)

    return variants


def decode_qr_from_image(img: np.ndarray) -> tuple[str, str]:
    """Извлечь payload QR из BGR-картинки.

    Возвращает `(payload, source)`, где source — какой бэкенд сработал
    ('zxing' / 'cv2' / 'cv2+upscale' / ...). Если ничего не нашли — ('', '').
    """
    if img is None or img.size == 0:
        return "", ""

    # Сначала zxing на оригинале — самый надёжный
    payload = _try_zxing(img)
    if payload:
        return payload, "zxing"
    payload = _try_cv2(img)
    if payload:
        return payload, "cv2"

    # Препроцессинги
    for i, variant in enumerate(_preprocess_variants(img)[1:], start=1):
        p = _try_zxing(variant)
        if p:
            return p, f"zxing+preproc{i}"
        p = _try_cv2(variant)
        if p:
            return p, f"cv2+preproc{i}"
    return "", ""


def decode_qr_from_file(path) -> tuple[str, str]:
    """Удобная обёртка: путь → (payload, source)."""
    from pathlib import Path
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        return "", ""
    return decode_qr_from_image(img)
