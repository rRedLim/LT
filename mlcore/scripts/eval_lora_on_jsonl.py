"""Per-field accuracy evaluation для LoRA-адаптера на val.jsonl.

Запуск:
    py scripts/eval_lora_on_jsonl.py
    # или явно:
    py scripts/eval_lora_on_jsonl.py \
        --val datasets/ocr_v3/val.jsonl \
        --lora runs/lora/v3/final \
        --out runs/lora/v3/eval_report.json

Логика:
1. Загружает VLMReader (Qwen2.5-VL + LoRA-адаптер, bf16, device="cuda" автоматом).
2. Идёт по каждой записи val.jsonl:
   - читает image, ground-truth берёт из assistant-message (JSON 23 полей);
   - генерит prediction через `reader.read()`;
   - сравнивает field-by-field.
3. Считает acc по правилам:
   - **price-поля** (price_default/_card/_discount/price{1..4}_qr/wholesale_*_price/action_price_qr):
     равенство с tolerance 0.01 после parse;
   - **barcode/id_sku/qr_code_barcode**: barcode_match (Levenshtein ≤ 2);
   - **product_name**: fuzzy_contains (word-level edit distance);
   - **остальные** (color, code, datetime, ...): exact match с нормализацией.
   Метка `"нет"` сравнивается как exact.
4. Печатает таблицу + сохраняет JSON-репорт.

Не использует HF Trainer — поэтому никаких OOM от logits, генерация делается по
одному семплу через `model.generate()` который и так используется в проде.
"""
from __future__ import annotations

# sys.path bootstrap
import sys as _sys
from pathlib import Path as _Path
_ML_CORE = str(_Path(__file__).resolve().parents[1])
if _ML_CORE not in _sys.path:
    _sys.path.insert(0, _ML_CORE)

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.inference.vlm_reader import VLMReader, FIELDS
from src.utils.metrics import (
    find_prices, barcode_match, fuzzy_contains,
)


log = logging.getLogger(__name__)


# ── Категоризация полей по способу сравнения ───────────────────────────────
PRICE_LIKE = {
    "price_default", "price_card", "price_discount",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_price", "wholesale_level_2_price",
    "action_price_qr",
}
BARCODE_LIKE = {"barcode", "qr_code_barcode", "id_sku"}
FUZZY_LIKE = {"product_name", "additional_info"}
EXACT_LIKE = (set(FIELDS) - PRICE_LIKE - BARCODE_LIKE - FUZZY_LIKE)


def _normalize(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _compare_price(pred: str, gt: str, tol: float = 0.01) -> bool:
    """Сравнить два значения цены с допуском. Любая распознанная цена считается."""
    pp = find_prices(pred)
    gp = find_prices(gt)
    if not gp:                      # GT нет цены — считаем совпадающим если pred тоже пустой/без цены
        return not pp
    if not pp:
        return False
    # Хотя бы одна предсказанная цена должна совпадать с любой из GT
    for g in gp:
        for p in pp:
            if abs(g - p) <= tol:
                return True
    return False


def _compare_field(field: str, pred: str, gt: str) -> bool:
    pred_n = _normalize(pred)
    gt_n = _normalize(gt)

    # 'нет' — exact match как маркер физического отсутствия поля
    if gt_n == "нет":
        return pred_n == "нет"
    if pred_n == "нет":
        return False                # GT не "нет", а pred говорит "нет" → промах

    # Пустой GT — считаем совпадающим если pred тоже пустой (нет, чтобы он не клевал)
    if gt_n == "":
        return pred_n == ""
    if pred_n == "":
        return False

    if field in PRICE_LIKE:
        return _compare_price(pred, gt)
    if field in BARCODE_LIKE:
        return barcode_match(pred_n, gt_n, max_dist=2)
    if field in FUZZY_LIKE:
        return fuzzy_contains(gt_n, pred_n, word_dist=1) >= 0.5
    return pred_n == gt_n           # exact


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(rec)
    return out


def _gt_from_record(rec: dict) -> dict[str, str]:
    """Достать ground-truth dict из messages[-1].content (assistant JSON)."""
    msgs = rec.get("messages", [])
    if not msgs:
        return {}
    assistant = msgs[-1].get("content", "")
    if isinstance(assistant, list):
        # legacy format — берём первый text-блок
        assistant = next(
            (b.get("text", "") for b in assistant if isinstance(b, dict)),
            "",
        )
    try:
        d = json.loads(assistant)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {k: str(d.get(k, "")) for k in FIELDS} if isinstance(d, dict) else {}


def evaluate(
    val_path: Path,
    lora_adapter: Path,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_samples: Optional[int] = None,
) -> dict:
    records = _load_jsonl(val_path)
    if max_samples is not None:
        records = records[:max_samples]
    log.info("Loading VLMReader (base=%s, lora=%s)...", base_model, lora_adapter)
    reader = VLMReader(
        base_model=base_model,
        adapter=lora_adapter if lora_adapter.exists() else None,
    )
    if not lora_adapter.exists():
        log.warning("LoRA adapter %s не существует — eval на base-модели!", lora_adapter)

    log.info("Evaluating %d samples...", len(records))
    per_field_ok: Counter[str] = Counter()
    per_field_total: Counter[str] = Counter()
    per_sample_avg_acc: list[float] = []
    failures_per_field: dict[str, list[dict]] = defaultdict(list)

    t0 = time.time()
    for i, rec in enumerate(records):
        img_path = rec.get("image")
        if not img_path or not Path(img_path).exists():
            log.warning("sample %d: image %r не найден", i, img_path)
            continue
        gt = _gt_from_record(rec)
        if not gt:
            log.warning("sample %d: GT не распарсился", i)
            continue
        img = cv2.imread(img_path)
        if img is None:
            log.warning("sample %d: cv2 не прочёл %s", i, img_path)
            continue

        pred = reader.read(img)

        n_ok = 0
        n_total = 0
        for field in FIELDS:
            gt_v = gt.get(field, "")
            pred_v = pred.get(field, "")
            # Не считаем поля где и GT и pred пустые — это не «правильно», а
            # «вопроса не было». Если GT пустой а pred что-то выдал — это FP, считаем.
            if not gt_v.strip() and not pred_v.strip():
                continue
            ok = _compare_field(field, pred_v, gt_v)
            per_field_total[field] += 1
            if ok:
                per_field_ok[field] += 1
            else:
                if len(failures_per_field[field]) < 5:   # сохраняем первые 5 фейлов для дебага
                    failures_per_field[field].append({
                        "sample_idx": i,
                        "image": img_path,
                        "gt": gt_v,
                        "pred": pred_v,
                    })
            n_total += 1
            n_ok += int(ok)
        if n_total:
            per_sample_avg_acc.append(n_ok / n_total)

        if (i + 1) % 5 == 0 or i + 1 == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info("[%3d/%d] avg_field_acc_so_far=%.3f  rate=%.2f sample/s",
                     i + 1, len(records),
                     np.mean(per_sample_avg_acc) if per_sample_avg_acc else 0.0,
                     rate)

    # Отчёт
    field_acc: dict[str, dict] = {}
    for field in FIELDS:
        total = per_field_total[field]
        ok = per_field_ok[field]
        field_acc[field] = {
            "acc": (ok / total if total else None),
            "ok": ok,
            "total": total,
        }
    macro_acc = (
        sum(v["acc"] for v in field_acc.values() if v["acc"] is not None)
        / max(1, sum(1 for v in field_acc.values() if v["acc"] is not None))
    )
    micro_total = sum(per_field_total.values())
    micro_ok = sum(per_field_ok.values())
    micro_acc = micro_ok / micro_total if micro_total else 0.0

    return {
        "n_samples": len(records),
        "n_evaluated": len(per_sample_avg_acc),
        "field_acc": field_acc,
        "macro_field_acc": macro_acc,
        "micro_field_acc": micro_acc,
        "avg_sample_acc": float(np.mean(per_sample_avg_acc)) if per_sample_avg_acc else 0.0,
        "failures_per_field": dict(failures_per_field),
        "elapsed_sec": time.time() - t0,
    }


def _print_report(stats: dict) -> None:
    print()
    print("=" * 70)
    print(f"  n_samples evaluated:    {stats['n_evaluated']}/{stats['n_samples']}")
    print(f"  elapsed:                {stats['elapsed_sec']:.1f}s "
          f"({stats['elapsed_sec']/max(1, stats['n_evaluated']):.2f}s/sample)")
    print()
    print(f"  macro_field_acc:        {stats['macro_field_acc']*100:.1f}%")
    print(f"  micro_field_acc:        {stats['micro_field_acc']*100:.1f}%")
    print(f"  avg_per_sample_acc:     {stats['avg_sample_acc']*100:.1f}%")
    print()
    print("  Per-field accuracy:")
    print(f"    {'field':<28} {'acc':>7}   {'ok/total':>10}")
    print(f"    {'-'*28} {'-'*7}   {'-'*10}")
    rows = sorted(
        stats["field_acc"].items(),
        key=lambda kv: (-(kv[1]["acc"] if kv[1]["acc"] is not None else -1)),
    )
    for field, v in rows:
        if v["acc"] is None:
            print(f"    {field:<28} {'  N/A':>7}   {0:>4}/{0:<4}")
            continue
        print(f"    {field:<28} {v['acc']*100:>6.1f}%   {v['ok']:>4}/{v['total']:<4}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", type=Path, default=Path("datasets/ocr_v3/val.jsonl"))
    p.add_argument("--lora", type=Path, default=Path("runs/lora/v3/final"))
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--out", type=Path, default=Path("runs/lora/v3/eval_report.json"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="ограничить N (для быстрой проверки)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    stats = evaluate(args.val, args.lora, args.base_model, args.max_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("Saved report: %s", args.out)
    _print_report(stats)
