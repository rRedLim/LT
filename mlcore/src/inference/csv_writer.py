from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable


# 29 колонок строго в порядке эталонного CSV организаторов
# (см. Данные/<video>/<video>.csv, header)
COLUMNS: list[str] = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max", "qr_code_barcode",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]
assert len(COLUMNS) == 29

COMMA_QUOTED_PRICE = {"price_default", "price_card", "price_discount"}
COMMA_QUOTED_COORD = {"x_min", "y_min", "x_max", "y_max"}
DOT_NOQUOTE = {"price1_qr", "price2_qr", "price3_qr", "price4_qr",
               "action_price_qr",
               "wholesale_level_1_price", "wholesale_level_2_price"}


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_csv_value(field: str, value: Any) -> str:
    s = "" if value is None else str(value).strip()
    if s in ("нет", ""):
        return s
    if field in COMMA_QUOTED_COORD:
        f = _to_float(value)
        if f is None:
            return s
        return '"' + f"{f:.1f}".replace(".", ",") + '"'
    if field in COMMA_QUOTED_PRICE:
        f = _to_float(value)
        if f is None:
            return s
        return '"' + f"{f:.2f}".replace(".", ",") + '"'
    if field in DOT_NOQUOTE:
        f = _to_float(value)
        if f is None:
            return s
        return f"{f:.2f}"
    if field == "frame_timestamp":
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    # product_name / additional_info / special_symbols / code / color и т.п.:
    # кавычки только при наличии запятой или " внутри
    if "," in s or '"' in s:
        s_esc = s.replace('"', '""')
        return f'"{s_esc}"'
    return s


def write_csv(rows: Iterable[dict[str, Any]], out_path: Path) -> None:
    """Пишет CSV строго в формате организаторов (29 колонок, две локали, LF)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            cells = [_fmt_csv_value(col, row.get(col, "")) for col in COLUMNS]
            f.write(",".join(cells) + "\n")
