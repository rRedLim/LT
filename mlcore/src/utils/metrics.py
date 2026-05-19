"""Общие функции сопоставления полей.

Используются в:
  - training (per-epoch eval)
  - inference (постпроцесс)
  - eval_on_video.py (KPI)

Единственный источник истины — гарантирует согласованность метрик между
train-логом и bench-результатами.
"""

import re
from typing import Set

import Levenshtein


# ── find_prices ───────────────────────────────────────────────────────────────

_PRICE_RE = re.compile(r"(\d{1,5})[.,\s]?(\d{2})\b")


def find_prices(text: str) -> Set[float]:
    """Достать все возможные цены из строки.

    Поддерживает: 129.99, 129,99, 129 99, 12999 (без разделителя).
    Возвращает Set[float].
    """
    out: Set[float] = set()
    if not text:
        return out
    for m in _PRICE_RE.finditer(text):
        rub, kop = m.group(1), m.group(2)
        try:
            out.add(float(f"{rub}.{kop}"))
        except ValueError:
            pass
    return out


# ── barcode_match ─────────────────────────────────────────────────────────────

def barcode_match(pred: str, gt: str, max_dist: int = 2) -> bool:
    """Levenshtein ≤ max_dist по полному значению или по голове/хвосту 10 цифр.

    Обрабатывает типичные OCR-ошибки:
    - замену цифр (полное расстояние Levenshtein ≤ max_dist);
    - обрезание крайних символов OCR-ом (голова/хвост-fallback, только когда
      длины строк различаются, чтобы не давать ложных совпадений).
    """
    if not pred or not gt:
        return False
    if Levenshtein.distance(pred, gt) <= max_dist:
        return True
    # substring-fallback только при разной длине (OCR обрезал цифру)
    if len(pred) != len(gt):
        L = min(len(pred), len(gt), 10)
        if L >= 8:
            if Levenshtein.distance(pred[-L:], gt[-L:]) <= max_dist:
                return True
            if Levenshtein.distance(pred[:L], gt[:L]) <= max_dist:
                return True
    return False


# ── fuzzy_contains ────────────────────────────────────────────────────────────

def _normalize_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def fuzzy_contains(haystack: str, needle: str, word_dist: int = 1) -> float:
    """Доля слов из needle, найденных в haystack с допуском Levenshtein ≤ word_dist.

    Возвращает float в [0.0, 1.0].
    """
    if not needle:
        return 0.0
    h_words = [_normalize_word(w) for w in haystack.split() if _normalize_word(w)]
    n_words = [_normalize_word(w) for w in needle.split() if _normalize_word(w)]
    if not n_words:
        return 0.0
    found = 0
    for nw in n_words:
        for hw in h_words:
            if Levenshtein.distance(hw, nw) <= word_dist:
                found += 1
                break
    return found / len(n_words)


# ── ean13_checksum_ok ─────────────────────────────────────────────────────────

def ean13_checksum_ok(code: str) -> bool:
    """EAN-13 контрольная сумма.

    Возвращает True только если код ровно 13 цифр и контрольная сумма верна.
    """
    if not code or len(code) != 13 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    s = sum(d if i % 2 == 0 else 3 * d for i, d in enumerate(digits[:12]))
    check = (10 - s % 10) % 10
    return check == digits[12]
