# Lenta Tech — ML Core

Боевой ML-пайплайн от подготовки датасета через файнтюн моделей до inference.
`video.mp4 → CSV` в формате организаторов (29 колонок).

> Спек: `../docs/superpowers/specs/2026-05-16-ml-pipeline-design.md`
> План: `../docs/superpowers/plans/2026-05-16-ml-pipeline.md`

## Структура рабочей директории на облаке

Запуск из директории, в которой лежит `ml_core` и `dataset`:

```
<wd>/                              ← cwd
├── ml_core/                       ← наш пакет
│   ├── src/{data,training,inference,utils}/
│   ├── scripts/
│   ├── configs/
│   ├── tests/
│   ├── datasets/                  ← подготовленные данные (detection_v3, ocr_v3)
│   ├── runs/{yolo,lora}/v3/       ← обученные веса
│   ├── outputs/                   ← финальные CSV
│   └── debug/                     ← debug-артефакты при --debug
└── dataset/
    ├── dataset_orig/              ← видео орг (бывшая «Данные/»)
    │   ├── 25_12-20/{25_12-20.mp4, 25_12-20.csv}
    │   ├── 25_2-10/...
    │   ├── 26_12-20/...
    │   ├── 43_15/...              (motion-видео, val для YOLO; НЕ берётся в train LoRA)
    │   └── 49_5/...
    └── dataset_myself/            ← 13 наших видео (плоская структура)
        ├── 1.mov
        ├── 2.mov
        └── ...
```

Все скрипты запускаются из `<wd>/ml_core/`. Пути к данным внутри — относительные `../dataset/...`.

## Требования

- Python 3.10+
- CUDA-GPU (RTX 4090 рекомендован) для `train_lora` и inference
- ~20 GB GPU RAM для Qwen2.5-VL-7B в bf16
- `py -m pip install -r requirements.txt`

## Команды воспроизведения

Все из `<wd>/ml_core/`.

### 1. Подготовка detection-датасета
```bash
py scripts/prepare_detection_data.py
```
Без аргументов берёт дефолты: `--organizer-root ../dataset/dataset_orig`,
`--ls-export ../labeling/ls_export.json`, `--frames ../labeling/frames`,
`--out datasets/detection_v3`.

> Fisheye/motion-аугментация применяется ТОЛЬКО к нашим 13 видео (у орг fisheye
> уже есть, повторное применение ухудшит модель).

### 2. Тренировка YOLO11s v3
```bash
py scripts/train_yolo.py --config configs/yolo_v3.yaml
```
60 эпох, warm-start от `yolo11s.pt`. Веса → `runs/yolo/v3/weights/best.pt`.

### 3. Подготовка OCR-датасета (3 шага)
```bash
# 3.1 Авто-prefill на 13 наших видео через обученный YOLO+VLM
py scripts/prepare_ocr_data.py prefill \
    --yolo runs/yolo/v3/weights/best.pt \
    --lora runs/lora/v3/final
# (на первом проходе LoRA ещё не обучен — можно опустить --lora,
#  тогда VLM работает без адаптера. Качество prefill будет ниже,
#  но для последующей ручной правки этого хватает.)

# 3.2 РУЧНОЙ ШАГ: открой каждый labels_myself/<video>__editable.csv в Excel UTF-8
#     - в служебной колонке __preview_frame есть Ctrl+клик-гиперссылка
#       на полный кадр с подсветкой бокса
#     - в __preview_crop — что реально ушло в VLM (повёрнутый кроп)
#     - правь поля, удаляй ложно-позитивные строки

# 3.3 Финализация (отрезает служебные колонки) + сборка JSONL
py scripts/prepare_ocr_data.py finalize
py scripts/prepare_ocr_data.py build
```
На выходе `datasets/ocr_v3/{train,val}.jsonl` + crops.

### 4. Тренировка LoRA v3
```bash
py scripts/train_lora.py --config configs/lora_v3.yaml
```
3 эпохи, ~30 мин на 4090. Адаптер → `runs/lora/v3/final/`.

> Эта тренировка использует **наши 13 видео + static-видео орг** (через JSONL
> из шага 3.3).

### 5. Inference
```bash
# одно видео
py scripts/run_inference.py video \
    --video ../dataset/dataset_orig/43_15/43_15.mp4 \
    --out outputs/43_15.csv

# batch на 5 видео орг
py scripts/run_inference.py batch \
    --data ../dataset/dataset_orig \
    --out-dir outputs
```

### 6. Eval
Bash (Git Bash / WSL / Linux):
```bash
for v in 25_12-20 25_2-10 26_12-20 43_15 49_5; do
    py scripts/eval_on_video.py \
        --pred outputs/$v.csv \
        --gt ../dataset/dataset_orig/$v/$v.csv
done
```
PowerShell:
```powershell
foreach ($v in @("25_12-20","25_2-10","26_12-20","43_15","49_5")) {
    py scripts/eval_on_video.py --pred "outputs/$v.csv" --gt "../dataset/dataset_orig/$v/$v.csv"
}
```

## Debug-режим

Все inference-скрипты принимают `--debug`:

```bash
py scripts/run_inference.py video \
    --video ../dataset/dataset_orig/25_12-20/25_12-20.mp4 \
    --out outputs/25_12-20.csv \
    --debug                       # включить
    --debug-dir debug             # корень (по умолчанию ml_core/debug/)
    --debug-every-n-frames 30     # каждый 30-й кадр (1/сек при 30fps)
```

На выходе в `debug/<video_stem>/`:
```
pipeline.log                      все DEBUG-логи + timing по фазам
frames_with_bbox/f0000030.jpg     кадры с нарисованными bbox+track_id+conf
tracks/
    trk_0001_frame.jpg            Pick A кадр с подсветкой
    trk_0001_crop.jpg             что реально ушло в VLM (повёрнутый)
    trk_0001_vlm_raw.txt          сырой ответ VLM
    trk_0001_vlm.json             распарсенный dict (23 поля)
    trk_0001_final.json           после template_router
barcode.log                       попытки zxing/VLM-fallback по трекам
summary.json                      итоги + timings
```

barcode.log:
```
trk=0001  tried=  5  source=zxing         decoded=4670025474665     preproc=equalize_hist_rot90cw
trk=0002  tried= 28  source=zxing         decoded=NONE              no_decode_in_any_frame
trk=0003  tried=  3  source=vlm_fallback  decoded=4607054890123     votes={'4607054890123': 2, ...}
```

Без `--debug` все вызовы — no-op (нулевая стоимость в проде).

## Ключевые решения

См. `../docs/superpowers/specs/2026-05-16-ml-pipeline-design.md` §1.2.

- **Detector тренируется на всём** — все 5 видео орг + наши 13 с fisheye-аугами
  (только к нашим!). Задача детектора — найти bbox; читаемость текста ему безразлична.
- **Reader (LoRA) тренируется только на читаемых** — фильтр `sharpness≥40, h≥120`.
  Static-видео орг + все 13 наших. Motion-видео орг (`43_15`, `49_5`) НЕ берём.
- **На нечитаемых кадрах в submission `""`** — по ORGANIZER §4 это не списывается
  как наша вина.
- **Barcode**: zxing-cpp на ВСЕХ кадрах трека (28 препроцессингов × 4 поворота).
  VLM-fallback при пустом zxing.
- **DR + GAL включены** (эксп. 5: +9.2 п.п. product_name). **SR (Real-ESRGAN)
  выключен** — на эксп. 5 ухудшил price_card на −7.6 п.п.

## Что НЕ работает (проверено, не повторять)

Подробности — `../PIPELINE_NOTES.md`:
- PaddleOCR / PP-OCRv5 (mobile/server) — floor RapidOCR на русском шрифте
- Niblack price recognition (Aliev 2019) — found 40%
- Multi-frame voting на VLM — везде ≤ sharp_k1
- Centered эвристика для текста — не догоняет naive
- Two-prompt (NUMERIC/TEXTUAL) — равно universal при 2× cost
- HSV color classifier — после LoRA color=0.96, не нужен
- Real-ESRGAN SR — −7.6 п.п. price_card
- Procedural-синтетика (PIL.Draw) — −29 п.п. color из-за RGB-mismatch палитры

## Тесты

```bash
py -m pytest tests/ -v
```
Текущее покрытие: **77 passed, 1 skipped** (zxing-cpp на dev-машине не установлен — на облаке после `pip install -r requirements.txt` все 78 будут зелёные).

