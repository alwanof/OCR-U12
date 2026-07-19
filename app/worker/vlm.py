"""Ollama-backed VLM OCR and judge calls for the ensemble engine."""

import base64
import io
import json

import requests
from PIL import Image

from app import config

OCR_PROMPT = (
    "انسخ النص العربي المكتوب في هذه الصورة كما هو تماماً، سطراً سطراً، "
    "دون أي شرح أو تعليق. أخرج النص فقط."
)

JUDGE_SYSTEM = (
    "أنت خبير في تدقيق نصوص المستندات العربية. لديك نسختان (X وY) لنفس المقطع، "
    "أنتجهما نظاما تعرف ضوئي مختلفان وكلاهما قد يحتوي أخطاء. "
    "استنتج النص الأصح بالاختيار من النسختين أو الدمج بينهما. "
    "لا تضف أي كلمة أو رقم غير موجود في إحدى النسختين، ولا تشرح."
)


def vlm_ocr(image: Image.Image) -> str:
    payload = {
        "model": config.ENSEMBLE_VLM_MODEL,
        "stream": False,
        "keep_alive": "1m",  # free VM memory for the next model promptly
        "options": {"temperature": 0},
        "messages": [
            {"role": "user", "content": OCR_PROMPT, "images": [_b64(image)]}
        ],
    }
    return _chat(payload)


def judge(x: str, y: str, image: Image.Image | None = None) -> str:
    user = (
        f"النسخة X:\n{x or '(فارغة)'}\n\nالنسخة Y:\n{y or '(فارغة)'}\n\n"
        "أرجع النص النهائي الأصح بصيغة JSON."
    )
    message: dict = {"role": "user", "content": user}
    if config.ENSEMBLE_JUDGE_MODE == "vision" and image is not None:
        model = config.ENSEMBLE_VLM_MODEL
        message["images"] = [_b64(image)]
    else:
        model = config.EXTRACT_MODEL
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "1m",
        "format": {
            "type": "object",
            "properties": {"final": {"type": "string"}},
            "required": ["final"],
        },
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, message],
    }
    return json.loads(_chat(payload))["final"].strip()


def _chat(payload: dict) -> str:
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.EXTRACT_TIMEOUT
    )
    if resp.status_code == 400 and "think" in payload:
        payload.pop("think")
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.EXTRACT_TIMEOUT
        )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


_MAX_SIDE = 1280  # bound ollama's vision memory inside the small Docker VM


def _b64(image: Image.Image) -> str:
    image = image.convert("RGB")
    longest = max(image.width, image.height)
    if longest > _MAX_SIDE:
        f = _MAX_SIDE / longest
        image = image.resize((round(image.width * f), round(image.height * f)))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()
