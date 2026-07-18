# OCR-U12 — Offline Arabic Document OCR & Data Extraction

A fully-offline web app that converts **Arabic PDFs and photos** into clean **Markdown**, and extracts **user-defined fields** into **CSV / Excel** — no cloud APIs, all AI models run locally.

**Pipeline:** [MinerU](https://github.com/opendatalab/MinerU) (layout, reading order, tables) → [Qari-OCR](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct) (Arabic-specialist recognition) → Markdown → [Qwen3](https://ollama.com/library/qwen3) via Ollama (field extraction → CSV/XLSX).

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for architecture, decisions, and roadmap.

## Features

- 📄 **PDF / photo → Markdown** — upload in the browser, download `.md` (+ ZIP with images)
- 🎯 Two OCR engines, selectable per job:
  - `MinerU` — fast baseline (~20 s for a test page on CPU)
  - `Hybrid (MinerU + Qari)` — noticeably better Arabic: recovers numbers, diacritics, and words the baseline drops (much slower on CPU; fast with a GPU)
- 📊 **Field extraction** — define fields in plain Arabic ("رقم الفاتورة", "التاريخ"…), get values as a table + **CSV / Excel** download. Anti-hallucination: values come only from the document; absent fields return empty, never invented
- 🗂 Reusable extraction **templates**
- 🌐 Arabic-first RTL web UI
- 🔒 **Fully offline** after the one-time model download — nothing leaves your machine

## Requirements

- **Docker** (Docker Desktop on macOS/Windows, or Docker Engine + Compose on Linux)
- Give the Docker VM at least **8 GB RAM** (Docker Desktop → Settings → Resources)
- ~**20 GB free disk** for images + model weights
- No GPU required (CPU works; NVIDIA GPU makes the hybrid engine dramatically faster)

## Quick start

```bash
git clone https://github.com/alwanof/OCR-U12.git
cd OCR-U12

# 1. Build and start everything (first build takes several minutes)
docker compose up -d --build

# 2. Pull the extraction model into the Ollama container (one time, ~2.5 GB)
docker compose exec ollama ollama pull qwen3:4b

# 3. Open the app
open http://localhost:8000        # or just visit it in your browser
```

Then upload an Arabic PDF or photo (JPG/PNG) and pick an engine:

| First use of… | Downloads (one time, into Docker volumes) |
|---|---|
| Any conversion | MinerU model weights (~2–3 GB) |
| Hybrid engine | Qari-OCR weights (~4.5 GB) |
| Field extraction | already pulled in step 2 |

After these downloads, the stack runs **fully offline** — you can disable networking on the containers and everything keeps working.

### Using the app

1. **Convert:** drag & drop a PDF/photo → the job page updates live → preview the Markdown (RTL) → download `.md` or ZIP.
2. **Extract:** on a finished job, open «استخراج بيانات», write one field per line:
   ```
   رقم الفاتورة
   التاريخ : تاريخ إصدار المستند
   الإجمالي
   ```
   Optionally save the list as a named template for reuse. Submit → values table appears → download CSV or Excel (Arabic-safe encoding, RTL sheet).

## Services

| Service | Role |
|---|---|
| `web` | FastAPI + Jinja2/HTMX UI (Arabic, RTL) on port 8000 |
| `worker` | RQ worker: MinerU (layout/OCR) + Qari-OCR (hybrid engine) |
| `ollama` | Local LLM runtime for field extraction (qwen3:4b) |
| `redis` | Job queue |

Volumes: `ocr-data` (uploads, results, SQLite DB), `ocr-models` (MinerU + Qari weights), `ollama-models` (Qwen3).

## Configuration & upgrading models

All models and engine settings live in a `.env` file (see [.env.example](.env.example) for the full documented list):

```bash
cp .env.example .env   # then edit and: docker compose up -d
```

The app runs with sensible defaults when no `.env` exists. Key variables:

| Variable | Default | Upgrade path |
|---|---|---|
| `QARI_MODEL_ID` | `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` | `...Qari-OCR-0.4.0-VL-4B-Instruct` (needs ~9 GB RAM / GPU) |
| `EXTRACT_MODEL` | `qwen3:4b` | `qwen3:8b` / `qwen3:14b` (pull via `docker compose exec ollama ollama pull <model>` first) |
| `MINERU_DEVICE` | `cpu` | `cuda` on an NVIDIA server |
| `MINERU_BACKEND` | `pipeline` | `vlm-auto-engine` (GPU, higher accuracy) |
| `MINERU_METHOD` | `ocr` | `ocr` forces real OCR — Arabic PDF text layers are unreliable |
| `MINERU_FORMULA` | `false` | off: hallucinates LaTeX from Arabic calligraphy |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |

## Performance expectations

- **On CPU (any machine, incl. Apple Silicon under Docker):** MinerU engine ≈ 10–20 s/page; Hybrid engine ≈ **minutes per page** (every text block runs through a 2B vision model). Fine for evaluation; not for volume.
- **On a Linux server with an NVIDIA GPU (16 GB+):** same compose file, 20–50× faster; the production plan moves Qari to vLLM with batched inference and extraction to `qwen3:8b`.

## Arabic-specific engineering notes

Three issues you will hit with off-the-shelf tools, already handled here:

1. **Embedded PDF text layers** in Arabic docs are often stored in presentation forms / visual order and extract as garbage — so real OCR is forced (`MINERU_METHOD=ocr`).
2. **MinerU/PaddleOCR emit Arabic lines character-reversed** ([MinerU#677](https://github.com/opendatalab/MinerU/issues/677)). Fixed at the span level in `middle.json` via the Unicode bidi transform, then markdown is reassembled with MinerU's own builder ([app/worker/rtl.py](app/worker/rtl.py)). Heuristic-gated: becomes a no-op if upstream fixes RTL.
3. **Formula models hallucinate LaTeX** from decorative Arabic calligraphy — disabled by default.

## Licenses of the models used

| Component | License |
|---|---|
| This app's code | — (set by repo owner) |
| MinerU | Custom Apache-2.0-based (no revenue caps) |
| Qari-OCR v0.3 | Apache 2.0 |
| Qwen3 | Apache 2.0 |

## Troubleshooting

- **Job stuck in "في الانتظار"** — worker still downloading models on first run: `docker compose logs -f worker`
- **Hybrid job fails / container OOM-killed** — increase Docker VM memory to 8 GB+; make sure no other heavy containers are running
- **Extraction fails with connection error** — Ollama model not pulled: `docker compose exec ollama ollama pull qwen3:4b`
- **Reset everything** — `docker compose down -v` (deletes jobs, results, and downloaded models)
