from src.utils.metrics import find_prices, barcode_match, fuzzy_contains, ean13_checksum_ok


# ── find_prices ──────────────────────────────────────────────────────────────

def test_find_prices_dot_separator():
    assert 129.99 in find_prices("Цена 129.99 руб")

def test_find_prices_comma_separator():
    assert 129.99 in find_prices("Цена 129,99 руб")

def test_find_prices_space_separator():
    """129 99 (пробел вместо разделителя — типичный OCR-выход)"""
    assert 129.99 in find_prices("129 99")

def test_find_prices_no_separator():
    """12999 → 129.99 (OCR теряет точку)"""
    assert 129.99 in find_prices("12999")

def test_find_prices_multiple():
    found = find_prices("129.99 и 252.63")
    assert 129.99 in found and 252.63 in found


# ── barcode_match ─────────────────────────────────────────────────────────────

def test_barcode_match_exact():
    assert barcode_match("4670025474665", "4670025474665")

def test_barcode_match_one_digit_off():
    """OCR ошибся в одной цифре — допуск Levenshtein ≤2"""
    assert barcode_match("4670025474665", "4670025474675")

def test_barcode_match_two_digits_off():
    assert barcode_match("4670025474665", "4670025474676")

def test_barcode_match_three_digits_off():
    assert not barcode_match("4670025474665", "4670025474778")

def test_barcode_match_substring():
    """OCR обрезал первую цифру → fallback на голову/хвост"""
    assert barcode_match("4670025474665", "670025474665")

def test_barcode_match_empty():
    assert not barcode_match("", "4670025474665")
    assert not barcode_match("4670025474665", "")


# ── fuzzy_contains ────────────────────────────────────────────────────────────

def test_fuzzy_contains_exact():
    assert fuzzy_contains("Молоко Простоквашино 1л", "Молоко Простоквашино 1л") >= 0.99

def test_fuzzy_contains_partial():
    """OCR ломает имя на несколько строк, но 4 из 5 слов есть"""
    score = fuzzy_contains("Молоко Простоквашино жирное 3.2% 1л",
                           "Молоко Простоквашино жирное 1л")
    assert score >= 0.7

def test_fuzzy_contains_unrelated():
    assert fuzzy_contains("Молоко", "Картофель фри") < 0.3


# ── ean13_checksum_ok ─────────────────────────────────────────────────────────

def test_ean13_valid():
    assert ean13_checksum_ok("4670025474665")

def test_ean13_invalid_checksum():
    assert not ean13_checksum_ok("4670025474666")

def test_ean13_wrong_length():
    assert not ean13_checksum_ok("46700254746")
    assert not ean13_checksum_ok("46700254746651")

def test_ean13_non_digits():
    assert not ean13_checksum_ok("467002547466A")
