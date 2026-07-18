import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app import config, db, queue

app = FastAPI(title="Arabic OCR")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
import json as _json  # noqa: E402

templates.env.filters["fromjson"] = _json.loads

STATUS_LABELS = {
    "queued": "في الانتظار",
    "processing": "قيد المعالجة",
    "done": "مكتمل",
    "failed": "فشل",
}

_md_renderer = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def _render(request: Request, template: str, **ctx) -> HTMLResponse:
    ctx.setdefault("status_labels", STATUS_LABELS)
    return templates.TemplateResponse(request, template, ctx)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render(request, "index.html", jobs=db.list_jobs())


@app.post("/upload")
async def upload(file: UploadFile, engine: str = Form("mineru-only")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, "نوع الملف غير مدعوم. الأنواع المسموحة: PDF, PNG, JPG")
    if engine not in config.ALLOWED_ENGINES:
        raise HTTPException(400, "محرك تحويل غير معروف")

    job_id = uuid.uuid4().hex[:12]
    upload_dir = config.UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"upload{suffix}"

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(413, f"حجم الملف يتجاوز الحد الأقصى ({config.MAX_UPLOAD_MB} م.ب)")
            out.write(chunk)
    if written == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(400, "الملف فارغ")

    db.create_job(job_id, file.filename or dest.name, engine)
    queue.enqueue_ocr(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "المهمة غير موجودة")
    return _render(
        request,
        "job.html",
        job=job,
        content=_job_content(job_id, job),
        extractions=[_ext_view(e) for e in db.list_extractions(job_id)],
        templates_list=db.list_templates(),
    )


@app.get("/jobs/{job_id}/fragment", response_class=HTMLResponse)
def job_fragment(request: Request, job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404)
    return _render(request, "partials/job_content.html", job=job, content=_job_content(job_id, job))


def _job_content(job_id: str, job) -> str | None:
    """Rendered HTML preview of the markdown, only for finished jobs."""
    if job["status"] != "done":
        return None
    md_file = config.RESULTS_DIR / job_id / "document.md"
    if not md_file.exists():
        return None
    text = md_file.read_text(encoding="utf-8")
    html = _md_renderer.render(text)
    # MinerU markdown references images relatively; point them at our static mount.
    return html.replace('src="images/', f'src="/results/{job_id}/images/')


def _ext_view(ext) -> dict:
    """Row -> dict with parsed JSON for templates."""
    import json

    return {
        "id": ext["id"],
        "job_id": ext["job_id"],
        "status": ext["status"],
        "error": ext["error"],
        "created_at": ext["created_at"],
        "fields": json.loads(ext["fields"]),
        "result": json.loads(ext["result"]) if ext["result"] else None,
    }


def _parse_fields(raw: str) -> list[dict]:
    """One field per line: 'name' or 'name : description'."""
    fields = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•").strip()
        if not line:
            continue
        name, _, desc = line.partition(":")
        if not name.strip():
            continue
        fields.append({"name": name.strip(), "desc": desc.strip()})
    return fields[: config.MAX_FIELDS]


@app.post("/jobs/{job_id}/extract")
def start_extract(job_id: str, fields: str = Form(...), template_name: str = Form("")):
    import json

    job = db.get_job(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(400, "المهمة غير مكتملة")
    parsed = _parse_fields(fields)
    if not parsed:
        raise HTTPException(400, "أدخل حقلاً واحداً على الأقل")
    fields_json = json.dumps(parsed, ensure_ascii=False)
    if template_name.strip():
        db.save_template(uuid.uuid4().hex[:12], template_name.strip(), fields_json)
    ext_id = uuid.uuid4().hex[:12]
    db.create_extraction(ext_id, job_id, fields_json)
    queue.enqueue_extract(ext_id)
    return RedirectResponse(url=f"/jobs/{job_id}#extractions", status_code=303)


@app.get("/extractions/{ext_id}/row", response_class=HTMLResponse)
def extraction_row(request: Request, ext_id: str):
    ext = db.get_extraction(ext_id)
    if ext is None:
        raise HTTPException(404)
    return _render(request, "partials/extraction_row.html", e=_ext_view(ext))


@app.get("/extractions/{ext_id}/download.{fmt}")
def download_extraction(ext_id: str, fmt: str):
    if fmt not in ("csv", "xlsx"):
        raise HTTPException(404)
    ext = db.get_extraction(ext_id)
    if ext is None:
        raise HTTPException(404)
    path = config.RESULTS_DIR / ext["job_id"] / f"extract_{ext_id}.{fmt}"
    if not path.exists():
        raise HTTPException(404)
    media = "text/csv" if fmt == "csv" else (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return FileResponse(path, filename=f"extracted_{ext_id}.{fmt}", media_type=media)


@app.get("/download/{job_id}/document.md")
def download_md(job_id: str):
    md_file = config.RESULTS_DIR / job_id / "document.md"
    if not md_file.exists():
        raise HTTPException(404)
    job = db.get_job(job_id)
    stem = Path(job["filename"]).stem if job else job_id
    return FileResponse(md_file, filename=f"{stem}.md", media_type="text/markdown")


@app.get("/download/{job_id}/bundle.zip")
def download_zip(job_id: str):
    result_dir = config.RESULTS_DIR / job_id
    if not (result_dir / "document.md").exists():
        raise HTTPException(404)
    zip_path = result_dir.parent / f"{job_id}_bundle"
    archive = shutil.make_archive(str(zip_path), "zip", result_dir)
    job = db.get_job(job_id)
    stem = Path(job["filename"]).stem if job else job_id
    return FileResponse(archive, filename=f"{stem}.zip", media_type="application/zip")


@app.get("/health")
def health():
    return {"status": "ok"}
