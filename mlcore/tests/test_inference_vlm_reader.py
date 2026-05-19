from src.inference.vlm_reader import _parse_json_from_raw, FIELDS


def test_parse_valid_json():
    raw = '{"product_name": "Сыр", "color": "red"}'
    out = _parse_json_from_raw(raw)
    assert out["product_name"] == "Сыр"


def test_parse_with_surrounding_text():
    raw = 'Sure! Here is the JSON: {"color": "red"} hope that helps.'
    out = _parse_json_from_raw(raw)
    assert out["color"] == "red"


def test_parse_invalid_returns_empty():
    assert _parse_json_from_raw("no json here") == {}
    assert _parse_json_from_raw("{broken json}") == {}
    assert _parse_json_from_raw("") == {}


def test_fields_count():
    # 23 поля: 22 базовых + "code" (добавлено после review CRITICAL #6:
    # без него колонка "code" в финальном CSV всегда оставалась пустой).
    assert len(FIELDS) == 23
    assert "code" in FIELDS
