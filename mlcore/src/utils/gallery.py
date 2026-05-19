import json
import re
from pathlib import Path
from typing import List

import Levenshtein


# Нормализация: \n / лишние пробелы / нижний регистр для сравнения.
# Хранимое значение в .names НЕ изменяется — возвращаем оригинал.
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip().lower()


class Gallery:
    """Nearest-neighbour matching по списку известных product_name.

    Score = 1 - lev(norm(pred), norm(name)) / max(len(...))
    Нормализация: сворачивает `\\n` и множественные пробелы в один пробел,
    приводит к lower-case. Это критично для gallery.json где много имён
    с переносами строк (артефакты OCR) — без нормализации Levenshtein
    бы засчитывал \\n как полноценный символ-различие.

    Если max(score) >= threshold — возвращает ОРИГИНАЛЬНОЕ name (с переносами),
    иначе — исходный pred.
    """

    def __init__(self, names: List[str], threshold: float = 0.55):
        self.names = list(names)
        self._normalized = [_normalize_for_match(n) for n in self.names]
        self.threshold = threshold

    @classmethod
    def from_json(cls, path, threshold: float = 0.55) -> "Gallery":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["names"], threshold=threshold)

    def match(self, pred: str) -> str:
        if not pred:
            return pred
        npred = _normalize_for_match(pred)
        if not npred:
            return pred
        best_name = pred
        best_score = -1.0
        for orig, nname in zip(self.names, self._normalized):
            denom = max(len(npred), len(nname)) or 1
            score = 1.0 - Levenshtein.distance(npred, nname) / denom
            if score > best_score:
                best_score = score
                best_name = orig
        return best_name if best_score >= self.threshold else pred
