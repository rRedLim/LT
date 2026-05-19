# Веса моделей

Веса **не** хранятся в репозитории и **не** зашиваются в Docker-образ —
они монтируются сюда volume'ом (`./models:/models:ro`) на сервере.

Ожидаемая структура:

```
models/
├── yolo/
│   └── best.pt          ← детектор YOLO11s v3
│                           (mlcore: runs/yolo/v3/weights/best.pt)
└── lora/
    └── final/           ← LoRA-адаптер Qwen2.5-VL (ПАПКА адаптера)
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...           (mlcore: runs/lora/v3/final/)
```

## Куда какой файл

| Источник в `mlcore`                | Кладём сюда            |
|------------------------------------|------------------------|
| `runs/yolo/v3/weights/best.pt`     | `models/yolo/best.pt`  |
| `runs/lora/v3/final/` (вся папка)  | `models/lora/final/`   |

## Минимально необходимое

- **`models/yolo/best.pt`** — обязателен. Без него не запустится даже
  быстрый (smoke) режим.
- **`models/lora/final/`** — желателен. Если папки нет, VLM работает на
  базовой модели без адаптера (качество ниже, сервис всё равно работает —
  в UI будет предупреждение).
- Базовая VLM (`Qwen/Qwen2.5-VL-7B-Instruct`) качается автоматически с
  HuggingFace в `hf-cache` volume при первом запуске (~16 ГБ, один раз).

Пути переопределяются переменными `LENTA_YOLO_WEIGHTS` /
`LENTA_LORA_ADAPTER` (см. `docker-compose.yml`).
