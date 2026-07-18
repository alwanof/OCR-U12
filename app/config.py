import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "app.db"

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# OCR settings
MINERU_LANG = os.environ.get("MINERU_LANG", "arabic")
# Force real OCR by default: Arabic PDFs' embedded text layers are frequently stored
# in presentation forms / visual order and extract as garbled text.
MINERU_METHOD = os.environ.get("MINERU_METHOD", "ocr")
# Formula model hallucinates LaTeX from Arabic calligraphy/decorative regions.
MINERU_FORMULA = os.environ.get("MINERU_FORMULA", "false")
MINERU_TABLE = os.environ.get("MINERU_TABLE", "true")
OCR_JOB_TIMEOUT = int(os.environ.get("OCR_JOB_TIMEOUT", "3600"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_ENGINES = {"mineru-only", "hybrid"}

# Qari-OCR (hybrid engine)
QARI_MODEL_ID = os.environ.get("QARI_MODEL_ID", "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct")
QARI_PROMPT = os.environ.get(
    "QARI_PROMPT",
    "Extract and return only the text visible in this image, exactly as written, as plain text.",
)
QARI_MAX_NEW_TOKENS = int(os.environ.get("QARI_MAX_NEW_TOKENS", "512"))
# Render scale for cropping blocks out of PDF pages (multiples of 72 dpi).
QARI_RENDER_SCALE = float(os.environ.get("QARI_RENDER_SCALE", "2.8"))

# Field extraction (Ollama)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "qwen3:4b")
EXTRACT_MAX_CHARS = int(os.environ.get("EXTRACT_MAX_CHARS", "12000"))
EXTRACT_NUM_CTX = int(os.environ.get("EXTRACT_NUM_CTX", "16384"))
EXTRACT_TIMEOUT = int(os.environ.get("EXTRACT_TIMEOUT", "900"))
MAX_FIELDS = 30


def ensure_dirs() -> None:
    for d in (UPLOADS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
