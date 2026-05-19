from src.utils.gallery import Gallery


def test_gallery_exact_match():
    g = Gallery(["Молоко Простоквашино", "Сыр Российский"])
    assert g.match("Молоко Простоквашино") == "Молоко Простоквашино"


def test_gallery_close_match():
    """1 буква отличается, threshold 0.55"""
    g = Gallery(["Молоко Простоквашино"], threshold=0.55)
    assert g.match("Молоко Простаквашино") == "Молоко Простоквашино"


def test_gallery_far_returns_original():
    """Слишком далеко — оставляем исходное"""
    g = Gallery(["Молоко Простоквашино"], threshold=0.55)
    assert g.match("Картофель фри") == "Картофель фри"


def test_gallery_empty_input():
    g = Gallery(["Молоко"], threshold=0.55)
    assert g.match("") == ""


def test_gallery_from_json(tmp_path):
    j = tmp_path / "g.json"
    j.write_text('{"names": ["Сыр Российский"]}', encoding="utf-8")
    g = Gallery.from_json(j)
    assert g.match("Сыр Российский") == "Сыр Российский"
