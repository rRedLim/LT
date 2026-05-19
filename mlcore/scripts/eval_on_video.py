"""Сравнивает predicted CSV с ground truth (эталоном организаторов).

Метрики:
  - Track-level: recall, precision (match по barcode или spatial-temporal)
  - Field-level (per matched track): per-field accuracy через src.utils.metrics
  - Главный KPI: доля ценников с ≥80% корректных полей
"""
from __future__ import annotations
# sys.path bootstrap для запуска `py scripts/...` (Python добавляет в path
# папку скрипта, а не родителя). Без этого `from src.X import Y` упадёт.
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)
from pathlib import Path
from typing import Optional
import csv
import argparse
import json

from src.utils.metrics import barcode_match, find_prices, fuzzy_contains


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0, (bx2 - bx1) * (by2 - by1))
    return inter / max(a_area + b_area - inter, 1e-9)


def _parse_coord(s: str) -> float:
    try:
        return float(str(s).strip().replace(",", "."))
    except (ValueError, AttributeError):
        return 0.0


def _match_row(pred: dict, gts: list[dict], used: set[int]) -> Optional[int]:
    """Ищет GT-строку для pred по barcode (primary) или spatial-temporal."""
    pred_bc = (pred.get("barcode") or "").strip()
    if pred_bc and len(pred_bc) == 13 and pred_bc.isdigit():
        for i, gt in enumerate(gts):
            if i in used:
                continue
            gt_bc = (gt.get("barcode") or "").strip()
            if barcode_match(pred_bc, gt_bc):
                return i
    # fallback: ts ±1500мс + IoU bbox ≥ 0.3
    try:
        # int(float(...)) — устойчиво к "1500" и "1500.0", и к запятой "1500,0"
        ts = int(float(str(pred.get("frame_timestamp") or "0").replace(",", ".")))
    except (ValueError, TypeError):
        return None
    bx = (_parse_coord(pred.get("x_min", "0")),
          _parse_coord(pred.get("y_min", "0")),
          _parse_coord(pred.get("x_max", "0")),
          _parse_coord(pred.get("y_max", "0")))
    for i, gt in enumerate(gts):
        if i in used:
            continue
        try:
            gts_ts = int(float(str(gt.get("frame_timestamp") or "0").replace(",", ".")))
        except (ValueError, TypeError):
            continue
        gtb = (_parse_coord(gt.get("x_min", "0")),
               _parse_coord(gt.get("y_min", "0")),
               _parse_coord(gt.get("x_max", "0")),
               _parse_coord(gt.get("y_max", "0")))
        if abs(ts - gts_ts) > 1500:
            continue
        if _iou(bx, gtb) >= 0.3:
            return i
    return None


FIELDS_FOR_KPI = [
    "product_name", "price_default", "price_card", "price_discount",
    "barcode", "color", "discount_amount", "print_datetime",
]


def field_correct(field: str, pred: str, gt: str) -> bool:
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if not gt and not pred:
        return True
    if not gt or not pred:
        return False
    if gt == "нет" or pred == "нет":
        return pred == gt
    if field in ("price_default", "price_card", "price_discount"):
        p_set = find_prices(pred)
        g_set = find_prices(gt)
        return any(abs(p - g) < 0.01 for p in p_set for g in g_set)
    if field == "barcode":
        return barcode_match(pred, gt)
    if field == "product_name":
        return fuzzy_contains(pred, gt) >= 0.7
    return pred.lower() == gt.lower()


def evaluate(pred_path: Path, gt_path: Path) -> dict:
    preds = read_csv(pred_path)
    gts = read_csv(gt_path)
    used: set[int] = set()
    matched: list[tuple[dict, dict]] = []
    for p in preds:
        idx = _match_row(p, gts, used)
        if idx is not None:
            used.add(idx)
            matched.append((p, gts[idx]))

    n_kpi_pass = 0
    field_correct_count = {f: 0 for f in FIELDS_FOR_KPI}
    field_total = {f: 0 for f in FIELDS_FOR_KPI}
    for p, g in matched:
        correct = 0
        total = 0
        for f in FIELDS_FOR_KPI:
            if not (g.get(f) or "").strip():
                continue
            total += 1
            field_total[f] += 1
            if field_correct(f, p.get(f, ""), g.get(f, "")):
                correct += 1
                field_correct_count[f] += 1
        if total > 0 and correct / total >= 0.80:
            n_kpi_pass += 1

    return {
        "n_pred": len(preds),
        "n_gt": len(gts),
        "matched": len(matched),
        "recall": len(matched) / max(len(gts), 1),
        "precision": len(matched) / max(len(preds), 1),
        "kpi_pass_rate": n_kpi_pass / max(len(matched), 1),
        "per_field_acc": {f: (field_correct_count[f] / max(field_total[f], 1))
                          for f in FIELDS_FOR_KPI},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    args = p.parse_args()
    r = evaluate(args.pred, args.gt)
    print(json.dumps(r, indent=2, ensure_ascii=False))
