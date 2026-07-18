"""Field extraction: converted markdown + user field list -> values via a local
LLM (Ollama, structured JSON output) -> CSV/XLSX.

Anti-hallucination measures: temperature 0, a schema that forces every field to
string-or-null, and a system prompt instructing "answer only from the document,
null if absent". Long documents are chunked; the first non-empty value per
field wins across chunks.
"""

import csv
import json

import requests

from app import config, db

SYSTEM_PROMPT = (
    "أنت أداة لاستخراج البيانات من المستندات. استخرج قيم الحقول المطلوبة من نص "
    "المستند فقط، حرفياً كما وردت، دون أي تخمين أو معلومات خارجية. "
    "إذا لم توجد قيمة لحقلٍ ما في النص فأرجع null لذلك الحقل."
)


def run_extract(ext_id: str) -> None:
    db.init_db()
    db.mark_extraction(ext_id, "processing")
    try:
        ext = db.get_extraction(ext_id)
        if ext is None:
            raise RuntimeError(f"extraction {ext_id} not found")
        md_file = config.RESULTS_DIR / ext["job_id"] / "document.md"
        text = md_file.read_text(encoding="utf-8")
        fields = json.loads(ext["fields"])
        result = _extract(text, fields)
        out_dir = config.RESULTS_DIR / ext["job_id"]
        _write_csv(out_dir / f"extract_{ext_id}.csv", fields, result)
        _write_xlsx(out_dir / f"extract_{ext_id}.xlsx", fields, result)
        db.finish_extraction(ext_id, json.dumps(result, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - job must record any failure
        db.fail_extraction(ext_id, f"{type(exc).__name__}: {exc}")
        raise


def _extract(text: str, fields: list[dict]) -> dict:
    merged: dict = {f["name"]: None for f in fields}
    for chunk in _chunks(text, config.EXTRACT_MAX_CHARS):
        missing = [f for f in fields if merged[f["name"]] in (None, "")]
        if not missing:
            break
        data = _ask_ollama(chunk, missing)
        for f in missing:
            value = data.get(f["name"])
            if value not in (None, "", "null"):
                merged[f["name"]] = str(value).strip()
    return merged


def _ask_ollama(text: str, fields: list[dict]) -> dict:
    schema = {
        "type": "object",
        "properties": {f["name"]: {"type": ["string", "null"]} for f in fields},
        "required": [f["name"] for f in fields],
    }
    field_lines = "\n".join(
        f"- {f['name']}" + (f": {f['desc']}" if f.get("desc") else "") for f in fields
    )
    payload = {
        "model": config.EXTRACT_MODEL,
        "stream": False,
        "think": False,
        "format": schema,
        "keep_alive": "2m",  # free VM memory promptly; hybrid OCR may need it next
        "options": {"temperature": 0, "num_ctx": config.EXTRACT_NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"نص المستند:\n---\n{text}\n---\n"
                    f"استخرج الحقول التالية وأرجع النتيجة بصيغة JSON:\n{field_lines}"
                ),
            },
        ],
    }
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.EXTRACT_TIMEOUT
    )
    if resp.status_code == 400:
        # Older Ollama without the `think` parameter.
        payload.pop("think", None)
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.EXTRACT_TIMEOUT
        )
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts, current = [], []
    length = 0
    for para in text.split("\n\n"):
        if length + len(para) > size and current:
            parts.append("\n\n".join(current))
            current, length = [], 0
        current.append(para)
        length += len(para) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


def _write_csv(path, fields: list[dict], result: dict) -> None:
    # utf-8-sig so Arabic opens correctly in Excel.
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        names = [f["name"] for f in fields]
        writer.writerow(names)
        writer.writerow([result.get(n) or "" for n in names])


def _write_xlsx(path, fields: list[dict], result: dict) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "البيانات"
    ws.sheet_view.rightToLeft = True
    names = [f["name"] for f in fields]
    ws.append(names)
    ws.append([result.get(n) or "" for n in names])
    for col in ws.columns:
        width = max(len(str(c.value or "")) for c in col) + 4
        ws.column_dimensions[col[0].column_letter].width = min(width, 60)
    wb.save(path)
