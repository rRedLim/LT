"""Простой progress-helper, работает в любом окружении (cloud-jupyter, ssh, cron).

tqdm/rich требуют TTY, на cloud-нотбуках часто не виден. Этот модуль использует
обычный print с явным flush — гарантированно работает везде.

Использование:
    items = list(...)
    for i, item in enumerate(progress(items, "fisheye[medium]"), 1):
        ...  # обработка
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Sized, TypeVar

T = TypeVar("T")


def progress(
    items: Iterable[T],
    desc: str,
    *,
    every_pct: float = 5.0,
    every_n: int | None = None,
) -> Iterator[T]:
    """Итератор с прогрессом через `print(..., flush=True)`.

    Параметры
    ---------
    items     — итерируемое (желательно с `__len__`).
    desc      — заголовок (например "fisheye[medium]").
    every_pct — печатать примерно каждые `every_pct`% (по умолчанию 5%).
    every_n   — альтернатива: фиксированный шаг.

    Печатает в формате:
        [fisheye[medium]]   0/374 (0%)  0.0s
        [fisheye[medium]]  19/374 (5%)  2.1s, ETA 39.8s
        [fisheye[medium]]  37/374 (10%) 4.1s, ETA 36.9s
        ...
        [fisheye[medium]] 374/374 DONE in 41.2s
    """
    try:
        total = len(items)  # type: ignore[arg-type]
    except TypeError:
        total = None

    if every_n is None:
        if total and total > 0:
            every_n = max(1, int(total * every_pct / 100))
        else:
            every_n = 50  # fallback для итераторов без len

    t_start = time.perf_counter()
    last_print = -every_n  # чтобы первый принт сразу после i=0

    print(f"[{desc}]   0/{total or '?'} (0%)  0.0s", flush=True)

    n = 0
    for n, item in enumerate(items, 1):
        yield item
        if n - last_print >= every_n or n == total:
            elapsed = time.perf_counter() - t_start
            if total:
                pct = int(n * 100 / total)
                rate = n / elapsed if elapsed > 0 else 0
                remaining = (total - n) / rate if rate > 0 else 0
                print(f"[{desc}] {n:>4}/{total} ({pct:>3}%) "
                      f"{elapsed:.1f}s, ETA {remaining:.1f}s",
                      flush=True)
            else:
                print(f"[{desc}] {n:>4} processed  {elapsed:.1f}s",
                      flush=True)
            last_print = n

    elapsed = time.perf_counter() - t_start
    print(f"[{desc}] {n}/{total or n} DONE in {elapsed:.1f}s", flush=True)
