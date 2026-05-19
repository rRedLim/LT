from pathlib import Path
from src.inference.csv_writer import _fmt_csv_value, write_csv, COLUMNS


def test_columns_count():
    assert len(COLUMNS) == 29

def test_fmt_price_card_comma_with_quotes():
    assert _fmt_csv_value("price_card", "129.99") == '"129,99"'

def test_fmt_price1_qr_dot_no_quotes():
    assert _fmt_csv_value("price1_qr", "129.99") == "129.99"

def test_fmt_x_min_one_decimal_comma_quoted():
    assert _fmt_csv_value("x_min", 2063.123) == '"2063,1"'

def test_fmt_frame_timestamp_int_no_quotes():
    assert _fmt_csv_value("frame_timestamp", 6595) == "6595"

def test_fmt_barcode_no_quotes():
    assert _fmt_csv_value("barcode", "4670025474665") == "4670025474665"

def test_fmt_product_name_no_quotes_when_no_comma():
    assert _fmt_csv_value("product_name", "Молоко Простоквашино 1л") == "Молоко Простоквашино 1л"

def test_fmt_product_name_quoted_when_comma():
    s = _fmt_csv_value("product_name", "Тонизирующий, 0.25L")
    assert s.startswith('"')

def test_fmt_net_no_quotes():
    assert _fmt_csv_value("price_card", "нет") == "нет"

def test_fmt_empty_no_quotes():
    assert _fmt_csv_value("price_card", "") == ""

def test_write_csv_29_columns(tmp_path):
    out = tmp_path / "o.csv"
    row = {c: "" for c in COLUMNS}
    row.update({"filename": "video.mp4", "product_name": "Сыр",
                "price_card": "129.99", "color": "red",
                "frame_timestamp": 6595,
                "x_min": 100.5, "y_min": 200.5, "x_max": 300.5, "y_max": 400.5})
    write_csv([row], out)
    text = out.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert len(lines) == 2
    # header — точно 29 колонок (разделено запятой; никаких кавычек в header)
    assert len(lines[0].split(",")) == 29
    # значение с запятой ("129,99") должно быть в кавычках
    assert '"129,99"' in lines[1]

def test_header_matches_organizer_csv():
    """Header должен СТРОГО соответствовать эталону Данные/25_12-20/25_12-20.csv"""
    expected = ("filename,product_name,price_default,price_card,price_discount,"
                "barcode,discount_amount,id_sku,print_datetime,code,"
                "additional_info,color,special_symbols,frame_timestamp,"
                "x_min,y_min,x_max,y_max,qr_code_barcode,"
                "price1_qr,price2_qr,price3_qr,price4_qr,"
                "wholesale_level_1_count,wholesale_level_1_price,"
                "wholesale_level_2_count,wholesale_level_2_price,"
                "action_price_qr,action_code_qr")
    assert ",".join(COLUMNS) == expected
