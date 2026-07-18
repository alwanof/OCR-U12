# Arabic OCR Web App — Implementation Plan

**Goal:** A simple web app where a user uploads an Arabic PDF or photo and gets (1) a well-formatted Markdown file, and (2) optionally, a user-defined list of fields extracted into Excel/CSV.

**Hard constraint:** 100% offline — no cloud/online AI models. All inference runs on our own hardware.

---

## 1. Chosen OCR Architecture — Hybrid Pipeline

**Decision:** combine the best layout engine with the best Arabic recognizer instead of using either alone.

```
        PDF / Photo
             │
             ▼
   ┌───────────────────┐
   │      MinerU        │  page rendering, preprocessing,
   │  (pipeline backend)│  layout analysis (text/title/table/
   │                    │  figure/formula blocks), reading
   │                    │  order, table structure, formulas
   └─────────┬─────────┘
             │  layout JSON + page images
             ▼
   ┌───────────────────┐
   │   Block Cropper    │  crop each text-bearing block
   │                    │  (and each table cell) from the
   │                    │  page image, batch them
   └─────────┬─────────┘
             │  image crops
             ▼
   ┌───────────────────┐
   │     Qari-OCR       │  Arabic-specialist recognition
   │ (Qwen2-VL 2B FT,   │  (CER ~0.06 on Arabic print vs
   │  served via vLLM)  │  0.44 Tesseract, 0.64 AIN)
   └─────────┬─────────┘
             │  text per block/cell
             ▼
   ┌───────────────────┐
   │ Markdown Assembler │  layout labels → md structure:
   │                    │  title→#, text→¶, table→md/HTML
   │                    │  table, formula→LaTeX, figure→img
   │                    │  — in MinerU's reading order
   └─────────┬─────────┘
             ▼
        document.md
```

### Why this split

- **MinerU** (pipeline backend): mature layout detection, reading order, table *structure* recognition, formula recognition; permissive custom Apache-based license with **no revenue caps**; runs on CPU or ~4 GB VRAM. Its weak point for us is Arabic *text recognition* (PaddleOCR).
- **Qari-OCR** ([NAMAA-Space](https://huggingface.co/collections/NAMAA-Space/qari-ocr-a-high-accuracy-model-for-arabic-optical-character), Qwen2-VL 2B fine-tune): open-source SOTA for Arabic print — CER 0.061 / WER 0.16, strong on tashkeel, varied fonts, low-res scans. Its weak point is that it's recognition-only — no layout, no tables. Exactly what MinerU provides.
- Together: MinerU decides *where* things are and *what kind* they are; Qari decides *what the Arabic text says*; our assembler produces the Markdown.

### Integration design details

1. **MinerU as layout provider.** Run MinerU's pipeline backend and consume its intermediate output (`middle.json`-style block list: bbox, block type, reading-order index, table cell grid) plus rendered page images. We do **not** use MinerU's final markdown — only its analysis. If hooking the intermediate output proves brittle across MinerU versions, fall back to calling its underlying models directly (DocLayout-YOLO for layout + its table/formula models) behind our own thin wrapper.
2. **Cropping strategy.** Crop at **block/paragraph level** (Qari is trained on page/paragraph images — don't slice into single lines). Add small padding, upscale tiny crops before inference. Tables: crop **per cell** using MinerU's cell grid, then rebuild the table from structure + cell texts.
3. **Qari serving.** Serve Qari via **vLLM** (local OpenAI-compatible endpoint) so block crops from all pages are batched — critical for throughput; a 2B model on vLLM handles dozens of crops/sec on a mid-range GPU.
4. **Non-Arabic / numeric blocks.** Qari handles mixed AR/EN reasonably, but keep MinerU's PaddleOCR text as a per-block fallback and for pages detected as fully Latin — cheap insurance, already computed.
5. **Formulas** use MinerU's formula model output (LaTeX) as-is; **figures** are extracted as images and embedded with relative links.
6. **Assembler** maps block types to Markdown (`title→#/##` by level, `text→paragraph`, `list→-`, `table→`Markdown table, falling back to HTML for merged cells, `formula→$$...$$`), ordered by MinerU's reading order. Unicode NFC normalization; optional diacritics-stripping toggle.
7. **Escape hatch.** Keep a `engine=mineru-only` mode (MinerU end-to-end, no Qari) — one config flag. Useful for A/B accuracy comparison, non-Arabic docs, and as a fallback if the hybrid misbehaves on some input.

### Known risks of the hybrid (and mitigations)

| Risk | Mitigation |
|---|---|
| Qari hallucinates fluent Arabic on bad crops (VLM failure mode) | Confidence heuristics (compare against PaddleOCR fallback text; flag blocks with high divergence), human-review step before export, upscale/denoise crops |
| MinerU intermediate format changes between versions | Pin MinerU version; wrapper isolates the dependency; regression suite reruns on every upgrade |
| Per-cell table OCR is slow on dense tables | vLLM batching; cap table size, fall back to whole-table MinerU text above the cap |
| Reading order wrong on complex RTL layouts | Phase 0 test set includes multi-column RTL pages; assembler allows manual order override later |
| Two models + LLM = more VRAM | Qari 2B (~5 GB) + MinerU layout (~4 GB) + Qwen3 8B quant (~6 GB) → fits a 16 GB GPU if loaded sensibly (MinerU layout can run CPU); see Hardware |

## 2. Information Extraction (Feature 2)

Unchanged by the OCR decision: user defines fields ("رقم الفاتورة، التاريخ، الإجمالي…") → local LLM extracts them from the Markdown.

- **Ollama + Qwen3 (8B, Q4 quant)** — strong Arabic understanding, structured JSON output mode, fully offline.
- Pipeline: Markdown + field list → strict-JSON-schema prompt ("answer only from the document, `null` if absent") → chunk if over context window and merge → Pydantic validation (one retry on schema failure) → CSV/XLSX via pandas + openpyxl, **UTF-8-BOM** so Arabic opens correctly in Excel.
- Batch mode: one field template applied to N documents → one spreadsheet, a row per document.
- Anti-hallucination: store the source snippet/bbox for each extracted value and show it in the UI for verification before export.

## 3. System Architecture

```
┌──────────────┐     ┌─────────────────────────────────────────────┐
│   Frontend    │     │              Backend (FastAPI)              │
│  (React/Vite  │────▶│   /upload  /extract  /jobs/{id}  /download  │
│   or HTMX)    │     │                                             │
└──────────────┘     │  ┌──────────────┐   ┌───────────────────┐   │
                     │  │  Job Queue    │──▶│  OCR Worker        │   │
                     │  │ (Redis + RQ)  │   │  MinerU → crops →  │   │
                     │  └──────────────┘   │  Qari (vLLM) → .md │   │
                     │                     └─────────┬─────────┘   │
                     │                               ▼             │
                     │                     ┌───────────────────┐   │
                     │                     │ Extraction Worker  │   │
                     │                     │ Ollama (Qwen3)     │   │
                     │                     │ .md → JSON → CSV/  │   │
                     │                     │ XLSX               │   │
                     │                     └───────────────────┘   │
                     └─────────────────────────────────────────────┘
                          Local disk: uploads / crops / results
```

### Tech stack (finalized)

| Layer | Choice | Notes |
|---|---|---|
| Backend language | Python 3.11+ | Forced: MinerU & Qari are Python libraries |
| API | FastAPI + Uvicorn | Async uploads, SSE progress, Pydantic validation |
| Frontend | **HTMX + Jinja2**, served by FastAPI | Server-rendered, no Node/build step; **Arabic-first RTL UI** (`dir="rtl"`, Noto Naskh) |
| Layout/tables/formulas | MinerU (pipeline backend, pinned version) | Imported as a library in the OCR worker; intermediate JSON + page images |
| Arabic recognition | Qari-OCR 2B — PoC: CPU (transformers); prod: vLLM | In-worker for PoC; local OpenAI-compatible vLLM endpoint (batched) in prod |
| Markdown assembly | Our Python module | Block types + reading order → md |
| Extraction LLM | Ollama + Qwen3 — PoC: 4B quant; prod: 8B | Own container; structured JSON output; model swap is a tag change |
| Queue | **Redis + RQ** | Survives restarts, dashboard, same setup in prod; serializes heavy inference |
| Job metadata / templates | SQLite | Single file, zero setup; Postgres only if prod needs it |
| File storage | Local disk volumes | uploads / crops / results |
| Tabular output | pandas + openpyxl | CSV + XLSX, UTF-8-BOM for Excel-safe Arabic |
| Python tooling | uv (deps + lockfile), Ruff (lint/format) | Fast reproducible Docker builds |
| Packaging | Docker Compose, 4 services: `web`, `worker`, `redis`, `ollama` | `web`+`worker` share one image; shared `models/` + `data/` volumes; no runtime downloads after first pull; prod adds NVIDIA Container Toolkit + vLLM service |

### API sketch

```
POST /api/upload            multipart: file (pdf|jpg|png), options {engine: hybrid|mineru-only}
                            → { job_id }
POST /api/extract           { job_id | file, fields: [...], output: "csv"|"xlsx" }
                            → { job_id }
GET  /api/jobs/{id}         → { status, progress, error? }
GET  /api/download/{id}.md | .csv | .xlsx
```

### Hardware

**PoC (now):** developer Mac — Apple M4, 16 GB RAM, Docker Desktop. Docker on macOS has **no GPU access** (containers run in a Linux VM), so all inference is CPU-only: expect ~1–3 min/page for the hybrid pipeline. Fine for demos on small documents. 16 GB RAM also means models load **sequentially per stage** (never resident together), and the extraction LLM is the small **Qwen3 4B** quant instead of 8B.

**Production (later):** **one server with a 16 GB+ NVIDIA GPU** (e.g., RTX 4060 Ti 16GB / 4080 / A4000-class). The same Docker Compose file deploys unchanged and runs 20–50× faster.
- Qari on vLLM: ~5 GB VRAM (PoC uses CPU inference instead of vLLM).
- Qwen3 8B Q4 on Ollama: ~6 GB (PoC: 4B).
- MinerU layout/table/formula models: GPU (~4 GB) or CPU.

### Licensing

| Component | License | Commercial notes |
|---|---|---|
| MinerU | Custom Apache-2.0-based | No revenue caps — clean |
| Qari-OCR | Fine-tune of Qwen2-VL 2B (Apache 2.0 base) | **Verify NAMAA's model-card license before launch** |
| Qwen3 (extraction) | Apache 2.0 | Clean |
| vLLM / Ollama / FastAPI | Apache/MIT | Clean |

No Datalab (Surya/Marker) components → no $2M/$5M weight-license thresholds.

## 4. Roadmap — PoC first, then production

**Decision:** build the PoC as a fully dockerized app on the dev Mac (Option A: everything in containers, CPU-only), staged so each stage produces a working demo. The same Compose file is the production deployment artifact later.

### PoC — dockerized, on the dev Mac

**Stage 1 — MinerU-only pipeline, end-to-end skeleton** ✅ **DONE (2026-07-18)**

Findings from the build (test doc: 3 pages of the UN UDHR Arabic booklet):
- CPU speed is far better than feared: **3 pages in ~40 s** (models cached; first run +~5 min for weight download).
- **`-m ocr` must be forced**: Arabic PDFs' embedded text layers extract as presentation-form/visual-order garbage in `auto` mode.
- **Formula parsing must be off** for Arabic docs — it hallucinates LaTeX from calligraphy (`MINERU_FORMULA=false`).
- **MinerU emits Arabic OCR lines in reversed (visual) order** ([MinerU#677](https://github.com/opendatalab/MinerU/issues/677)). Fixed in [app/worker/rtl.py](app/worker/rtl.py): flip each span in `middle.json` via the bidi display transform, then reassemble markdown with MinerU's own `union_make`. Heuristic-gated (auto-noop if upstream fixes RTL). Line-level markdown fallback exists but scrambles paragraph line order — span-level is the real fix.

- Docker Compose: `web` (FastAPI + minimal upload page), `redis`, `ocr-worker` (MinerU pipeline backend, CPU).
- Flow proven in the browser: upload Arabic PDF/photo → job queued → progress → download/preview `.md`.
- Model weights downloaded once into a mounted volume; after that the stack runs with network disabled.
- Image preprocessing for photos (EXIF auto-rotate, deskew, contrast normalize).
- ✅ Demo: upload → Markdown in the browser.

**Stage 2 — Qari hybrid behind `engine=hybrid` flag** ✅ **DONE (2026-07-18)** — full doc-set accuracy scoring still pending real documents

A/B on the same UDHR page (hybrid vs MinerU-only): hybrid recovered content MinerU dropped
(**the year "1948"**, **"٢٢٠ لغة"**), fixed misrecognitions (`ويُتاح` vs `ويتَاحح`,
`للأمم` vs `لأمم`), and produced correct diacritics and punctuation. Model:
Qari-OCR v0.3 2B (**Apache-2.0** — license question resolved), bf16 CPU, lazy-loaded
after the MinerU subprocess exits (fits the 7.6 GiB Docker VM). Cost: ~11 min/page on
CPU (vs 20 s MinerU-only) — vLLM+GPU in production fixes this. Gotcha fixed along the
way: FastAPI form fields need `Form(...)` — a plain str param is a query param, so the
engine selection was silently ignored.
- Add Qari-OCR (CPU inference in the container — slow but demonstrable): MinerU layout JSON → block cropper → Qari → Markdown assembler.
- `engine=mineru-only` remains the default fallback; the flag enables A/B comparison on the same document.
- Collect 15–25 representative Arabic docs (scanned + digital PDFs, phone photos, tables, multi-column, mixed AR/EN) and score both engines: CER/WER on sampled blocks, reading order, table fidelity.
- ⚠️ **Accuracy gate:** hybrid must beat MinerU-only on Arabic text without breaking layout/tables. Also validates crop granularity and Qari hallucination rate on real scans.
- ✅ Demo: same document, both engines, visibly better Arabic from hybrid.

**Stage 3 — Field extraction → CSV/XLSX** ✅ **DONE (2026-07-18)**

Verified on the hybrid-OCR'd UDHR page: extracted سنة الاعتماد → `1948`,
عدد اللغات → `٢٢٠`, الجهة → `الجمعية العامة للأمم المتحدة` — all correct, and a
negative test (رقم الفاتورة on a non-invoice) correctly returned `null` (no
hallucination). Ollama + qwen3:4b, JSON-schema-forced output, temperature 0,
`think:false`, keep_alive 2m to free VM memory. CSV is UTF-8-BOM; XLSX sheet is RTL.
Templates save/load works. Extraction shares the OCR queue so heavy model work
stays serialized on the 7.6 GiB VM.
- Add `ollama` service with **Qwen3 4B** quant (16 GB RAM budget; models load sequentially per stage).
- UI to define field lists; extraction worker: schema prompt → JSON → validation → CSV/XLSX (UTF-8-BOM).
- ✅ Demo: upload invoice + field list → spreadsheet with extracted values.

**PoC constraints (Mac):** Docker has no GPU on macOS → CPU-only, ~1–3 min/page; demo on small documents. This validates *correctness and UX*, not speed.

### Production — after PoC acceptance

**Phase P1 — Performance & robustness (~1.5 weeks)**
- Deploy the same Compose stack to a Linux server with a 16 GB+ NVIDIA GPU (NVIDIA Container Toolkit).
- Switch Qari to **vLLM** serving with batched crops; extraction LLM to Qwen3 8B; MinerU on GPU.
- vLLM client retries, queue limits, timeouts; regression suite = Stage-2 doc set with expected outputs, rerun on any upgrade.

**Phase P2 — Full UI polish (3–5 days)**
- Side-by-side preview (page thumbnails vs rendered RTL Markdown), Arabic UI labels, reusable extraction templates, batch mode across documents, per-value source snippets for verification.

**Phase P3 — Hardening & launch (~1 week)**
- Auth, per-user job isolation, file retention/cleanup.
- Error surfaces: corrupt/password-protected PDFs, blank images, huge files.
- Behind Traefik/Nginx; **block network egress on inference containers** to prove offline operation.

**Estimate: ~2–3 weeks PoC + ~3 weeks production = 5–6 weeks total.**

## 5. Out of scope (v1)
- Handwritten Arabic
- In-app Markdown editing (view + download only; v2 candidate)
- Multi-tenant SaaS features, billing

## 6. Open questions
1. Dominant document types (invoices, contracts, IDs, books?) — drives the Phase-0 set and extraction templates.
2. Volume (docs/day) and acceptable turnaround per document?
3. Do we already have the GPU server, or provision one?
4. Qari model-card license confirmation for commercial use.
