import pytest
from pathlib import Path
from src.inference.template_router import TemplateRouter


SCHEMA = """
red:
  required: [product_name, price_card, color]
  optional: [discount_amount, id_sku]
yellow:
  required: [product_name, price_card, color, discount_amount]
  optional: [id_sku]
"""


@pytest.fixture
def router(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(SCHEMA, encoding="utf-8")
    return TemplateRouter(p)


def test_route_red_missing_required(router):
    out = router.route({"product_name": "Сыр", "price_card": "", "color": "red"})
    assert out["price_card"] == ""

def test_route_red_optional_missing_becomes_net(router):
    out = router.route({"product_name": "Сыр", "price_card": "129,99", "color": "red"})
    assert out["discount_amount"] == "нет"
    assert out["id_sku"] == "нет"

def test_route_yellow_has_discount(router):
    out = router.route({"product_name": "Сыр", "price_card": "129,99",
                        "discount_amount": "-30%", "color": "yellow"})
    assert out["discount_amount"] == "-30%"

def test_route_unknown_color_falls_back_to_blanks(router):
    out = router.route({"product_name": "Сыр", "color": "magenta"})
    assert out["product_name"] == "Сыр"
