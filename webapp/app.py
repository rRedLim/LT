"""Lenta Tech — веб-интерфейс распознавания ценников.

Front + back в одном файле на Gradio. Оборачивает боевой ML-пайплайн
`mlcore/src/inference/pipeline.py` (`video.mp4 → CSV`, 29 колонок организаторов).

Поток: пользователь загружает видео → пайплайн (YOLO+ByteTrack → Qwen2.5-VL +
LoRA → barcode → template-router) → CSV в формате организаторов на скачивание
+ превью таблицы + сводка по прогону + хвост лога.

Конфигурация через env (см. .env.example):
    LENTA_YOLO_WEIGHTS   путь к YOLO .pt           (в Docker: /models/yolo/best.pt)
    LENTA_LORA_ADAPTER   путь к LoRA-адаптеру      (в Docker: /models/lora/final)
    LENTA_BASE_MODEL     HF id базовой VLM          (Qwen/Qwen2.5-VL-7B-Instruct)
    LENTA_OUTPUT_DIR     куда складывать CSV
    GRADIO_SERVER_NAME   интерфейс прослушки        (0.0.0.0)
    GRADIO_SERVER_PORT   порт                       (7860)

Тяжёлые импорты (torch/transformers/ultralytics) выполняются ЛЕНИВО внутри
обработчика — модуль и UI поднимаются мгновенно даже без ML-стека (удобно для
локальной отладки фронта).
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr

# --------------------------------------------------------------------------- #
#  Пути и конфигурация
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
MLCORE_ROOT = REPO_ROOT / "mlcore"

# Чтобы работал `from src.inference.pipeline import run` (пакет — это mlcore/).
if str(MLCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLCORE_ROOT))


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


YOLO_WEIGHTS = _env_path(
    "LENTA_YOLO_WEIGHTS", MLCORE_ROOT / "runs" / "yolo" / "v3" / "weights" / "best.pt"
)
LORA_ADAPTER = _env_path(
    "LENTA_LORA_ADAPTER", MLCORE_ROOT / "runs" / "lora" / "v3" / "final"
)
BASE_MODEL = os.environ.get("LENTA_BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
OUTPUT_DIR = _env_path("LENTA_OUTPUT_DIR", MLCORE_ROOT / "outputs")
JOBS_DIR = Path(os.environ.get("LENTA_JOBS_DIR", "/tmp/lenta_jobs"))

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Перехват stdout/stderr пайплайна в буфер (для живого лога в UI)
# --------------------------------------------------------------------------- #


class _Tee(io.TextIOBase):
    """Пишет и в реальный поток (docker logs), и в потокобезопасный буфер."""

    def __init__(self, real, buf: io.StringIO, lock: threading.Lock):
        self._real = real
        self._buf = buf
        self._lock = lock

    def write(self, s: str) -> int:  # noqa: D401
        with self._lock:
            self._buf.write(s)
        try:
            self._real.write(s)
            self._real.flush()
        except Exception:  # noqa: BLE001  (битый дескриптор не должен ронять прогон)
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:  # noqa: BLE001
            pass


def _parse_progress(text: str) -> float:
    """Достаёт долю выполнения из хвоста stdout пайплайна.

    Пайплайн печатает две фазы:
      [YOLO+ByteTrack <stem>] N/T (P%) ...   ← детекция, мапим в 0..50%
      [infer <stem>] N/T (P%) ...            ← VLM-чтение,  мапим в 50..95%
    """
    import re

    last_pct = 0.0
    phase_base, phase_span = 0.0, 0.50
    # YOLO-фаза печатает "(50%)" (detector.py), infer-фаза — "( 50%)" с
    # right-pad (utils/progress.py). Поэтому допускаем пробелы внутри скобок.
    for m in re.finditer(r"\[(YOLO\+ByteTrack|infer)[^\]]*\][^\(]*\(\s*(\d+)%\)", text):
        kind, pct = m.group(1), int(m.group(2))
        if kind == "infer":
            phase_base, phase_span = 0.50, 0.45
        else:
            phase_base, phase_span = 0.0, 0.50
        last_pct = phase_base + phase_span * (pct / 100.0)
    return min(0.95, last_pct)


# --------------------------------------------------------------------------- #
#  Запуск пайплайна
# --------------------------------------------------------------------------- #


def _resolve_weights_status() -> tuple[bool, bool, str]:
    """(yolo_ok, lora_ok, человекочитаемый статус) для шапки UI."""
    yolo_ok = YOLO_WEIGHTS.exists()
    lora_ok = LORA_ADAPTER.exists()
    lines = [
        f"- **YOLO weights:** `{YOLO_WEIGHTS}` — "
        + ("✅ найдены" if yolo_ok else "❌ не найдены"),
        f"- **LoRA adapter:** `{LORA_ADAPTER}` — "
        + ("✅ найден" if lora_ok else "⚠️ не найден (VLM пойдёт без адаптера)"),
        f"- **Base VLM:** `{BASE_MODEL}`",
    ]
    return yolo_ok, lora_ok, "\n".join(lines)


def _prepare_input(video_path: str) -> tuple[Path, Path]:
    """Кладёт загруженное видео в `<job>/<stem>/<filename>`.

    Пайплайн формирует колонку `filename` как `<имя_папки>/<имя_файла>`
    (формат эталона организаторов, напр. `25_12-20/2.mp4`). Поэтому помещаем
    файл в подпапку с именем = stem видео.
    """
    src = Path(video_path)
    stem = src.stem or "video"
    job_dir = JOBS_DIR / uuid.uuid4().hex[:12]
    vid_dir = job_dir / stem
    vid_dir.mkdir(parents=True, exist_ok=True)
    dst = vid_dir / src.name
    shutil.copy2(src, dst)
    return dst, job_dir


def _read_csv_preview(csv_path: Path, max_rows: int = 200):
    """CSV → (pandas.DataFrame для превью, n_rows). На ошибке — пустой df."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        n = len(df)
        return df.head(max_rows), n
    except Exception:  # noqa: BLE001
        import pandas as pd

        return pd.DataFrame(), 0


def process(video_file, mode, progress=gr.Progress()):
    """Главный обработчик. Генератор: стримит статус/лог, в конце — результат.

    Выходы (в порядке компонентов): status_md, csv_file, preview_df,
    summary_json, log_box.
    """
    import pandas as pd

    empty_df = pd.DataFrame()

    if not video_file:
        yield ("❗ Загрузите видеофайл.", None, empty_df, {}, "")
        return

    video_path = video_file if isinstance(video_file, str) else video_file.name
    ext = Path(video_path).suffix.lower()
    if ext not in VIDEO_EXTS:
        yield (
            f"❗ Неподдерживаемое расширение `{ext}`. "
            f"Допустимо: {', '.join(sorted(VIDEO_EXTS))}.",
            None,
            empty_df,
            {},
            "",
        )
        return

    smoke = mode.startswith("Быстрый")

    yolo_ok, lora_ok, _ = _resolve_weights_status()
    if not yolo_ok:
        yield (
            f"❌ Не найдены веса YOLO: `{YOLO_WEIGHTS}`.\n\n"
            "Положите `best.pt` по этому пути (в Docker — в `./models/yolo/best.pt`) "
            "и перезапустите контейнер.",
            None,
            empty_df,
            {},
            "",
        )
        return

    progress(0.02, desc="Подготовка входных данных…")
    in_video, job_dir = _prepare_input(video_path)
    stem = in_video.parent.name
    out_csv = OUTPUT_DIR / f"{stem}.csv"

    # --- запуск пайплайна в отдельном потоке + стриминг лога ---------------- #
    buf = io.StringIO()
    lock = threading.Lock()
    result_box: dict = {}
    error_box: dict = {}

    def _worker():
        from src.inference.pipeline import run  # тяжёлый импорт — лениво

        try:
            result_box["result"] = run(
                in_video,
                YOLO_WEIGHTS,
                LORA_ADAPTER,
                out_csv,
                base_model=BASE_MODEL,
                smoke=smoke,
            )
        except Exception as exc:  # noqa: BLE001
            error_box["error"] = exc
            error_box["tb"] = traceback.format_exc()

    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_out, buf, lock)
    sys.stdout.write(
        f"=== Lenta inference: video={in_video.name} mode="
        f"{'smoke' if smoke else 'full'} ===\n"
    )
    sys.stderr = sys.stdout

    th = threading.Thread(target=_worker, daemon=True)
    started = time.perf_counter()
    try:
        th.start()
        while th.is_alive():
            th.join(timeout=1.5)
            with lock:
                log_text = buf.getvalue()
            frac = _parse_progress(log_text)
            elapsed = time.perf_counter() - started
            phase = "детекция + трекинг" if frac < 0.5 else "чтение ценников (VLM)"
            progress(max(0.03, frac), desc=f"{phase} · {elapsed:.0f}s")
            yield (
                f"⏳ Идёт обработка **{in_video.name}** "
                f"({'smoke' if smoke else 'full'} режим)… "
                f"{int(frac * 100)}% · {elapsed:.0f}s",
                None,
                empty_df,
                {},
                _tail(log_text),
            )
    finally:
        sys.stdout, sys.stderr = real_out, real_err

    with lock:
        log_text = buf.getvalue()

    # --- ошибка прогона ---------------------------------------------------- #
    if "error" in error_box:
        yield (
            f"❌ Ошибка во время инференса: `{error_box['error']}`",
            None,
            empty_df,
            {"error": str(error_box["error"])},
            _tail(log_text + "\n" + error_box.get("tb", ""), n=120),
        )
        _cleanup(job_dir)
        return

    result = result_box.get("result", {})

    if not out_csv.exists():
        yield (
            "❌ Пайплайн завершился, но CSV не создан. Смотрите лог.",
            None,
            empty_df,
            result,
            _tail(log_text, n=120),
        )
        _cleanup(job_dir)
        return

    progress(0.97, desc="Формирование результата…")
    preview_df, n_rows = _read_csv_preview(out_csv)

    summary = {
        "video": in_video.name,
        "mode": "smoke" if smoke else "full",
        "tracks_detected": result.get("tracks"),
        "csv_rows": result.get("rows", n_rows),
        "barcode": result.get("barcode"),
        "lora_adapter_used": (not smoke) and lora_ok,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "csv_path": str(out_csv),
    }

    note = ""
    if not smoke and not lora_ok:
        note = (
            "\n\n> ⚠️ LoRA-адаптер не найден — VLM работала на базовой модели "
            "(качество распознавания ниже)."
        )

    progress(1.0, desc="Готово")
    yield (
        f"✅ Готово. Обнаружено ценников: **{summary['csv_rows']}** "
        f"(треков: {summary['tracks_detected']}). "
        f"Время: {summary['elapsed_sec']} c.{note}",
        str(out_csv),
        preview_df,
        summary,
        _tail(log_text, n=120),
    )
    _cleanup(job_dir)


def _tail(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _cleanup(job_dir: Path) -> None:
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #

_DESC = """
# 🛒 Lenta Tech — распознавание ценников с видео робота

Загрузите видео стеллажа, снятое роботом, — система детектирует ценники,
читает их мультимодальной моделью и отдаёт **CSV в формате организаторов
(29 колонок)** на скачивание.

**Пайплайн:** YOLO11s + ByteTrack → Qwen2.5-VL-7B + LoRA → штрихкод
(zxing-cpp / VLM-fallback) → template-router → CSV.
"""


def build_demo() -> gr.Blocks:
    _, _, weights_md = _resolve_weights_status()

    with gr.Blocks(
        title="Lenta Tech — распознавание ценников",
        theme=gr.themes.Soft(primary_hue="blue"),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(_DESC)

        with gr.Accordion("⚙️ Состояние моделей", open=False):
            gr.Markdown(weights_md)

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.File(
                    label="Видео со стеллажа",
                    file_count="single",
                    file_types=["video", ".mp4", ".mov", ".avi", ".mkv"],
                )
                mode_in = gr.Radio(
                    choices=[
                        "Полный (детекция + VLM + штрихкод)",
                        "Быстрый (только детекция, без VLM — smoke)",
                    ],
                    value="Полный (детекция + VLM + штрихкод)",
                    label="Режим распознавания",
                    info="Быстрый режим не требует VLM — полезен для проверки I/O "
                    "и демонстрации без GPU.",
                )
                run_btn = gr.Button("▶️ Распознать", variant="primary")
                status = gr.Markdown("Ожидание загрузки видео…")

            with gr.Column(scale=2):
                csv_out = gr.File(label="📄 Результат — CSV (29 колонок)")
                preview = gr.Dataframe(
                    label="Превью результата (до 200 строк)",
                    wrap=True,
                    interactive=False,
                )
                summary_out = gr.JSON(label="Сводка по прогону")
                with gr.Accordion("📋 Лог пайплайна", open=False):
                    log_out = gr.Textbox(
                        label="", lines=18, max_lines=18, show_copy_button=True
                    )

        run_btn.click(
            fn=process,
            inputs=[video_in, mode_in],
            outputs=[status, csv_out, preview, summary_out, log_out],
            concurrency_limit=1,  # GPU обслуживает один прогон за раз
        )

        gr.Markdown(
            "---\n*Lenta Tech Life Hack · «Полка под контролем». "
            "Локальный контур, без обращения к облачным API.*"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="4gb",
        show_api=False,
    )
