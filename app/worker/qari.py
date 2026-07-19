"""Qari-OCR inference (Arabic-specialist Qwen2-VL fine-tune), CPU.

Loaded lazily so mineru-only jobs never pay the ~4.5 GB memory cost, and the
MinerU subprocess has already exited before the model loads (the Docker VM has
< 8 GB — both cannot be resident at once).
"""

import gc
import re

from PIL import Image

from app import config

_model = None
_processor = None

# Qwen2-VL patches are 28px; give tiny line crops some headroom.
_MIN_SIDE = 56
# Cap huge crops (full-page handwritten blocks): vision tokens grow with pixels
# and the memory spike OOM-kills the worker inside the small Docker VM.
_MAX_SIDE = int(__import__("os").environ.get("QARI_MAX_SIDE", "1540"))
_TAG_RE = re.compile(r"<[^>]+>")
# A short chunk repeated many times in a row = degeneration loop, not document text.
_LOOP_RE = re.compile(r"(.{2,16}?)\1{5,}", re.S)


def ocr_image(image: Image.Image) -> str:
    import torch

    model, processor = _load()
    image = _ensure_min_size(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": config.QARI_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=config.QARI_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=config.QARI_REPETITION_PENALTY,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    return _clean(text)


def unload() -> None:
    global _model, _processor
    _model = None
    _processor = None
    gc.collect()


def _load():
    global _model, _processor
    if _model is None:
        import os

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        # Fewer threads = smaller per-thread scratch buffers; matters in the small VM.
        torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.QARI_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _model.eval()
        _processor = AutoProcessor.from_pretrained(config.QARI_MODEL_ID)
    return _model, _processor


def _ensure_min_size(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    longest = max(image.width, image.height)
    if longest > _MAX_SIDE:
        factor = _MAX_SIDE / longest
        image = image.resize((round(image.width * factor), round(image.height * factor)))
    shortest = min(image.width, image.height)
    if shortest >= _MIN_SIDE:
        return image
    factor = _MIN_SIDE / shortest
    return image.resize((round(image.width * factor), round(image.height * factor)))


def _clean(text: str) -> str:
    text = _TAG_RE.sub(" ", text)  # v0.3 may emit structural HTML markup
    text = _LOOP_RE.sub(lambda m: m.group(1), text)  # collapse degeneration loops
    return re.sub(r"[ \t]+", " ", text).strip()
