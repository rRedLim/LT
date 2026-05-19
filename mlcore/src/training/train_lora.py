"""LoRA fine-tune для Qwen2.5-VL-7B на ценниках Ленты.

Полностью самостоятельный модуль (без legacy `lora2_new_dataset.py`).

Источник данных: JSONL-файлы, собранные через `src/data/build_ocr_dataset.py`
(13 наших видео + static-видео орг, фильтр sharpness≥40, h≥120).

Формат JSONL-записи:
    {"image": "path/to/crop.jpg",
     "messages": [
         {"role": "user", "content": [
             {"type": "image", "image": "path/to/crop.jpg"},
             {"type": "text", "text": UNIVERSAL_PROMPT}
         ]},
         {"role": "assistant", "content": "<JSON dump 23 полей>"}
     ]}

Использует DR (dynamic resolution через image_block min/max_pixels) +
GAL отключён в train (он применяется только на inference).

Запуск:
    cd ml_core
    py scripts/train_lora.py --config configs/lora_v3.yaml
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


log = logging.getLogger(__name__)


_ML_CORE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Sample:
    image: Path
    messages: list[dict]


def _load_jsonl(path: Path) -> list[_Sample]:
    out: list[_Sample] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            img = rec.get("image")
            msgs = rec.get("messages")
            if not img or not msgs:
                continue
            out.append(_Sample(image=Path(img), messages=msgs))
    return out


def _resolve_path(p: str | Path, anchor: Path) -> Path:
    """Резолвит путь: абсолютный — как есть, относительный — от anchor."""
    pth = Path(p)
    return pth if pth.is_absolute() else (anchor / pth).resolve()


class _QwenVLDataset:
    """Torch Dataset для Qwen2.5-VL. Применяет аугментации к кропу,
    собирает input_ids/labels/pixel_values."""

    def __init__(
        self,
        samples: list[_Sample],
        processor,
        augment: bool,
        max_len: int = 1280,
        min_pixels: int = 200_000,
        max_pixels: int = 1_600_000,
    ):
        self.samples = samples
        self.proc = processor
        self.augment = augment
        self.max_len = max_len
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._augmenter = self._build_augmenter() if augment else None

    @staticmethod
    def _build_augmenter():
        """Аугментации, имитирующие реальные искажения с робота."""
        import cv2
        import albumentations as A

        def _try(*alts):
            last_err = None
            for ctor in alts:
                try:
                    return ctor()
                except (TypeError, ValueError) as e:
                    last_err = e
            raise last_err  # type: ignore[misc]

        return A.Compose([
            A.MotionBlur(blur_limit=(3, 11), p=0.4),
            A.Defocus(radius=(1, 3), p=0.2),
            _try(
                lambda: A.GaussNoise(std_range=(0.02, 0.15), p=0.3),
                lambda: A.GaussNoise(var_limit=(5.0, 40.0), p=0.3),
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20,
                                 val_shift_limit=15, p=0.4),
            A.Perspective(scale=(0.02, 0.08), p=0.35),
            A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.35),
            _try(
                lambda: A.ImageCompression(quality_range=(55, 95), p=0.4),
                lambda: A.ImageCompression(quality_lower=55, quality_upper=95, p=0.4),
            ),
            _try(
                lambda: A.CoarseDropout(num_holes_range=(1, 3),
                                        hole_height_range=(8, 24),
                                        hole_width_range=(8, 24),
                                        fill=0, p=0.2),
                lambda: A.CoarseDropout(max_holes=3, max_height=24, max_width=24,
                                        fill_value=0, p=0.2),
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import cv2
        from PIL import Image

        sample = self.samples[idx]
        img_bgr = cv2.imread(str(sample.image))
        if img_bgr is None:
            img_bgr = np.zeros((600, 400, 3), dtype=np.uint8)

        if self.augment and self._augmenter is not None:
            img_bgr = self._augmenter(image=img_bgr)["image"]

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Берём messages из JSONL, но подменяем image-объект на свежий PIL
        # (в JSONL хранится строковый путь — заменим на PIL для процессора).
        user_msg = sample.messages[0]
        assistant_msg = sample.messages[1] if len(sample.messages) > 1 else \
            {"role": "assistant", "content": "{}"}

        # Строим content для user с актуальным min/max_pixels
        user_blocks = []
        for block in user_msg["content"]:
            if isinstance(block, dict) and block.get("type") == "image":
                user_blocks.append({
                    "type": "image",
                    "image": pil,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                })
            else:
                user_blocks.append(block)

        # assistant.content может быть str (наш формат) или list[dict] (legacy).
        # Нормализуем в str.
        ac = assistant_msg["content"]
        if isinstance(ac, list):
            ac = "\n".join(
                blk.get("text", "") for blk in ac if isinstance(blk, dict)
            )
        messages_full = [
            {"role": "user", "content": user_blocks},
            {"role": "assistant", "content": ac},
        ]
        messages_prefix = [messages_full[0]]

        full_text = self.proc.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False,
        )
        prefix_text = self.proc.apply_chat_template(
            messages_prefix, tokenize=False, add_generation_prompt=True,
        )

        proc_kwargs = dict(text=[full_text], images=[pil],
                           return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len)
        try:
            inputs_full = self.proc(
                **proc_kwargs,
                min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            )
        except TypeError:
            inputs_full = self.proc(**proc_kwargs)
        try:
            inputs_prefix = self.proc(
                text=[prefix_text], images=[pil], return_tensors="pt",
                padding=False, truncation=True, max_length=self.max_len,
                min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            )
        except TypeError:
            inputs_prefix = self.proc(
                text=[prefix_text], images=[pil], return_tensors="pt",
                padding=False, truncation=True, max_length=self.max_len,
            )

        input_ids = inputs_full.input_ids[0]
        attention_mask = inputs_full.attention_mask[0]
        prefix_len = inputs_prefix.input_ids.shape[1]
        # Если prefix не короче full (большая картинка + DR съели весь max_len) —
        # это значит assistant-часть обрезана truncation'ом. В этом случае
        # семпл бесполезен для обучения (labels все будут -100). Сигнализируем
        # warning, но НЕ маскируем последний токен (хотя бы 1 токен останется
        # с реальным loss — это лучше чем тихий ноль-градиент).
        n_total = int(input_ids.shape[0])
        if prefix_len >= n_total:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "sample idx=%d: prefix_len=%d >= input_len=%d; "
                "assistant truncated. Уменьши max_pixels или увеличь max_len.",
                idx, prefix_len, n_total,
            )
            prefix_len = max(0, n_total - 1)
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if hasattr(inputs_full, "pixel_values") and inputs_full.pixel_values is not None:
            item["pixel_values"] = inputs_full.pixel_values
        if hasattr(inputs_full, "image_grid_thw") and inputs_full.image_grid_thw is not None:
            item["image_grid_thw"] = inputs_full.image_grid_thw
        return item


def _collate_fn(batch: list[dict]) -> dict:
    """Паддинг текстовых тензоров + конкатенация image-токенов.

    pixel_values/image_grid_thw гейтятся через `all(...)` — если хотя бы один
    элемент батча без изображения (битый JSONL / np.zeros fallback в _QwenVLDataset),
    весь батч обрабатывается как текстовый. Альтернатива — `any()` + дозаполнение
    нулями — сложнее, чёрная картинка модели не нужна.
    """
    import torch
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out: dict = {}
    for k in ("input_ids", "attention_mask", "labels"):
        pad_val = -100 if k == "labels" else 0
        t = torch.full((len(batch), max_len), pad_val, dtype=batch[0][k].dtype)
        for i, b in enumerate(batch):
            t[i, :b[k].shape[0]] = b[k]
        out[k] = t
    if all("pixel_values" in b for b in batch):
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if all("image_grid_thw" in b for b in batch):
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return out


def train(config_path: Path) -> dict:
    """Читает YAML, запускает LoRA-тренировку напрямую через transformers.Trainer."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Резолвинг путей относительно ml_core/
    train_jsonl = _resolve_path(cfg.get("train_data", "datasets/ocr_v3/train.jsonl"),
                                _ML_CORE_ROOT)
    val_jsonl = _resolve_path(cfg.get("val_data", "datasets/ocr_v3/val.jsonl"),
                              _ML_CORE_ROOT)
    out_dir = _resolve_path(cfg.get("out_dir", "runs/lora/v3"), _ML_CORE_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # Heavy imports — поздние
    import torch
    from transformers import (
        AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        log.warning("CUDA недоступна — обучение будет очень медленным")

    proc_cfg = cfg.get("processor", {})
    min_pixels = int(proc_cfg.get("min_pixels", 200_000))
    max_pixels = int(proc_cfg.get("max_pixels", 1_600_000))

    base_model = cfg.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained(base_model)

    # Multi-GPU detection: если запущены под `torchrun` (DDP), у каждого процесса
    # будет переменная LOCAL_RANK. В этом случае модель целиком грузим на свою
    # карту (data-parallel), а НЕ шардируем через "auto" (model-parallel конфликтует
    # с DDP-обёрткой Trainer'а — будет CUDA-кеш-промах и зависание на all_reduce).
    import os
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        device_map_arg: Any = {"": local_rank}
        log.info("DDP detected: LOCAL_RANK=%d, device_map={\"\": %d}",
                 local_rank, local_rank)
    else:
        device_map_arg = "auto"
        log.info("Single-process mode: device_map=\"auto\"")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device_map_arg,
    )

    lora_cfg_raw = cfg.get("lora", {})
    lora_cfg = LoraConfig(
        r=int(lora_cfg_raw.get("r", 16)),
        lora_alpha=int(lora_cfg_raw.get("alpha", 32)),
        lora_dropout=float(lora_cfg_raw.get("dropout", 0.05)),
        target_modules=lora_cfg_raw.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Датасеты
    train_samples = _load_jsonl(train_jsonl)
    val_samples = _load_jsonl(val_jsonl)
    if not train_samples:
        raise RuntimeError(
            f"train.jsonl пустой или не найден: {train_jsonl}. "
            "Сначала запусти `py scripts/prepare_ocr_data.py build`."
        )
    log.info("train_samples=%d val_samples=%d", len(train_samples), len(val_samples))

    max_len = int(cfg.get("max_len", 1280))
    augment = bool(cfg.get("augment", True))
    train_ds = _QwenVLDataset(train_samples, processor, augment=augment,
                              max_len=max_len, min_pixels=min_pixels,
                              max_pixels=max_pixels)
    val_ds = _QwenVLDataset(val_samples, processor, augment=False,
                            max_len=max_len, min_pixels=min_pixels,
                            max_pixels=max_pixels) if val_samples else None

    train_cfg = cfg.get("training", {})
    # При DDP: эффективный batch = per_device_train_batch_size × world_size ×
    # gradient_accumulation_steps. По умолчанию (batch=1, grad_accum=8, 2 GPU) =
    # 16 — это уже сильнее одиночного 4090 (8). При желании можно понизить
    # grad_accum в configs/lora_v3.yaml до 4 чтобы сохранить тот же эффективный
    # batch и пройти в ~2 раза быстрее.
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=int(train_cfg.get("epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        # eval_strategy: явное "no" из YAML отключает eval (OOM на 4090 24GB,
        # logits [B, L, vocab=152K] не влезают). Trainer всё равно сохраняет
        # адаптер по save_strategy.
        eval_strategy=(train_cfg.get("evaluation_strategy", "no")
                       if val_ds else "no"),
        # На случай если позже включат eval: чанкуем logits на CPU чтобы не
        # держать [B, L, vocab] на GPU целиком. Существенно снижает OOM-риск.
        eval_accumulation_steps=int(train_cfg.get("eval_accumulation_steps", 1)),
        logging_steps=10,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        # DDP: LoRA замораживает base-параметры, DDP должен их игнорить, иначе
        # будет RuntimeError "Expected to mark a variable ready only once".
        ddp_find_unused_parameters=False,
        # Чтобы IO не был бутылочным горлом при двух картах
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=_collate_fn,
    )
    trainer.train()
    # При DDP save_model() безопасен на всех rank'ах (Trainer сам гейтит на rank-0),
    # но явный гейт надёжнее — peft на старых версиях может писать дублирующиеся
    # adapter_config.json от обоих процессов и испортить контрольную сумму.
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "final"))
        log.info("LoRA saved to %s/final", out_dir)
    return {"out_dir": str(out_dir), "n_train": len(train_samples),
            "n_val": len(val_samples)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/lora_v3.yaml"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    r = train(args.config)
    print(r)
