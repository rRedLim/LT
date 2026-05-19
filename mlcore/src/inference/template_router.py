from __future__ import annotations
from pathlib import Path
from typing import Dict
import yaml


class TemplateRouter:
    """Маршрутизация полей по архетипу (определяется через VLM-color).

    Правило ORGANIZER_REQUIREMENTS §7:
    - "нет" если поле физически отсутствует на этом архетипе
    - ""    если должно быть, но не распознано
    """

    def __init__(self, schema_path: Path):
        self.schema: Dict[str, Dict[str, list[str]]] = yaml.safe_load(
            schema_path.read_text(encoding="utf-8")
        ) or {}

    def route(self, fields: Dict[str, str]) -> Dict[str, str]:
        """Применяет правила архетипа к dict с полями ценника.

        Поля, не упомянутые в схеме архетипа, оставляются как есть.
        Если color неизвестен (нет в schema) — возвращает входной dict без правок.
        """
        out = {k: (v if v is not None else "").strip() for k, v in fields.items()}
        color = (out.get("color") or "").lower().strip()
        arche = self.schema.get(color)
        if not arche:
            return out
        required = set(arche.get("required", []))
        optional = set(arche.get("optional", []))
        for k in required:
            if not out.get(k):
                out[k] = ""
        for k in optional:
            if not out.get(k):
                out[k] = "нет"
        return out
