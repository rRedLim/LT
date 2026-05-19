from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import re
import numpy as np
from PIL import Image

from src.utils.gallery import Gallery


UNIVERSAL_PROMPT = (
    "Это фото ценника из магазина «Лента». Извлеки из него следующие поля "
    "в JSON: product_name, price_default, price_card, price_discount, barcode, "
    "id_sku, code, color, special_symbols, additional_info, print_datetime, "
    "discount_amount, "
    "price1_qr, price2_qr, price3_qr, price4_qr, action_price_qr, action_code_qr, "
    "qr_code_barcode, wholesale_level_1_count, wholesale_level_1_price, "
    "wholesale_level_2_count, wholesale_level_2_price. "
    'Если поле физически отсутствует — пиши "нет". Если не можешь прочесть — пиши "".'
)


FIELDS = [
    "product_name", "price_default", "price_card", "price_discount", "barcode",
    "id_sku", "code", "color", "special_symbols", "additional_info",
    "print_datetime", "discount_amount",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr", "action_price_qr",
    "action_code_qr", "qr_code_barcode",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
]


def _parse_json_from_raw(raw: str) -> dict[str, str]:
    """Достать JSON между первой `{` и последней `}`. На ошибки — пустой dict."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class VLMReader:
    """Qwen2.5-VL-7B + LoRA-адаптер с DR (dynamic resolution) и GAL (gallery match).

    SR (Real-ESRGAN) НЕ используется — в эксп. 5 он ухудшил price_card на −7.6 п.п.

    Внимание: `min_pixels`/`max_pixels` НЕ передаются в `AutoProcessor.from_pretrained`
    (он их молча игнорирует). Вместо этого они инжектятся в `image_block` каждого
    запроса и пробрасываются в processor через kwargs — это правильный паттерн
    для Qwen2.5-VL (как в pipeline_v1.py и lora2_new_dataset.py).
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter: Optional[Path] = None,
        gallery_path: Optional[Path] = None,
        gallery_threshold: float = 0.55,
        min_pixels: int = 200_000,
        max_pixels: int = 1_600_000,
        device: str = "auto",
    ):
        # Heavy imports — внутри __init__, чтобы модуль импортировался без torch
        import os
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        # AutoProcessor НЕ принимает min/max_pixels — пробрасываем их per-call.
        self.processor = AutoProcessor.from_pretrained(base_model)

        # 4-bit (NF4) по env LENTA_VLM_4BIT — для GPU < ~16 ГБ (4070 Super 12 ГБ):
        # 7B с ~16 ГБ → ~6 ГБ. По умолчанию ВЫКЛ → на проде остаётся bf16.
        _use_4bit = os.environ.get("LENTA_VLM_4BIT", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        model_kwargs: dict = dict(torch_dtype=torch.bfloat16, device_map=device)
        if _use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if device in ("auto", None, "cpu"):
                model_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, **model_kwargs,
        )
        if adapter is not None and Path(adapter).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.eval()

        # При device_map="auto" у dispatch_model нет .device — берём device первого параметра.
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        self.gallery: Optional[Gallery] = None
        if gallery_path is not None and Path(gallery_path).exists():
            self.gallery = Gallery.from_json(gallery_path, threshold=gallery_threshold)

    def read(self, crop_bgr: np.ndarray) -> dict[str, str]:
        """Прогоняет один кроп через VLM, возвращает dict с 23 полями (FIELDS)."""
        result, _raw = self.read_with_raw(crop_bgr)
        return result

    def read_with_raw(self, crop_bgr: np.ndarray) -> tuple[dict[str, str], str]:
        """Как `read`, но дополнительно возвращает сырой текст ответа VLM.

        Используется debug-режимом для записи в trk_NNN_vlm_raw.txt.
        """
        import torch
        img = Image.fromarray(crop_bgr[..., ::-1].copy())  # BGR → RGB

        # min/max_pixels — В блоке image (паттерн pipeline_v1.py:643-659):
        image_block = {
            "type": "image",
            "image": img,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        messages = [
            {"role": "user", "content": [
                image_block,
                {"type": "text", "text": UNIVERSAL_PROMPT},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Дублируем min/max_pixels в processor kwargs для совместимости со старыми
        # версиями transformers, которые не поддерживают per-image overrides.
        proc_kwargs = dict(text=[text], images=[img], return_tensors="pt")
        try:
            inputs = self.processor(
                **proc_kwargs,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        except TypeError:
            # Старые версии transformers без поддержки этих kwargs
            inputs = self.processor(**proc_kwargs)

        inputs = inputs.to(self._device)

        # 23 поля × ~10 токенов на поле + JSON-кавычки ≈ 250 токенов реальных.
        # Дефолт 512 был с запасом, но на средней скорости 50 tok/s каждый
        # лишний токен = 20мс/трек. Урезаем до 320; override через env LENTA_MAX_TOKENS.
        import os as _os
        max_new = int(_os.environ.get("LENTA_MAX_TOKENS", "320"))
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        raw = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        parsed = _parse_json_from_raw(raw)
        result = {k: str(parsed.get(k, "")).strip() for k in FIELDS}
        if self.gallery is not None and result["product_name"]:
            result["product_name"] = self.gallery.match(result["product_name"])
        return result, raw
