"""OCR worker task: input file -> MinerU (pipeline, CPU) -> normalized result dir.

Stage 1: engine 'mineru-only'. Stage 2 will add the 'hybrid' engine
(MinerU layout -> block crops -> Qari-OCR -> assembler) behind the same interface.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from app import config, db
from app.worker import rtl

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def run_ocr(job_id: str) -> None:
    db.init_db()
    db.mark_processing(job_id)
    try:
        job = db.get_job(job_id)
        engine = job["engine"] if job else "mineru-only"
        input_file = _find_upload(job_id)
        prepared = _preprocess(input_file)
        result_dir = config.RESULTS_DIR / job_id
        _run_mineru(prepared, result_dir, engine)
        md_path = result_dir / "document.md"
        db.mark_done(job_id, str(md_path))
    except Exception as exc:  # noqa: BLE001 - job must record any failure
        db.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        raise


def _find_upload(job_id: str) -> Path:
    upload_dir = config.UPLOADS_DIR / job_id
    files = [p for p in upload_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise FileNotFoundError(f"no uploaded file for job {job_id}")
    return files[0]


def _preprocess(input_file: Path) -> Path:
    """Normalize photos: apply EXIF orientation, force RGB. PDFs pass through."""
    if input_file.suffix.lower() not in IMAGE_EXTENSIONS:
        return input_file
    with Image.open(input_file) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        normalized = input_file.with_name(f"normalized_{input_file.stem}.png")
        im.save(normalized, format="PNG")
    return normalized


def _run_mineru(input_file: Path, result_dir: Path, engine: str = "mineru-only") -> None:
    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
        tmp_out = Path(tmp)
        cmd = [
            "mineru",
            "-p", str(input_file),
            "-o", str(tmp_out),
            "-b", config.MINERU_BACKEND,
            "-d", config.MINERU_DEVICE,
            "-m", config.MINERU_METHOD,
            "-f", config.MINERU_FORMULA,
            "-t", config.MINERU_TABLE,
        ]
        if config.MINERU_LANG:
            proc = _exec(cmd + ["-l", config.MINERU_LANG])
            if proc.returncode != 0:
                # Language tag may be unsupported by this MinerU version; retry with auto-detect.
                proc = _exec(cmd)
        else:
            proc = _exec(cmd)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise RuntimeError(f"mineru exited with code {proc.returncode}: {tail}")
        _collect_output(tmp_out, result_dir, engine, input_file)


def _exec(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=config.OCR_JOB_TIMEOUT - 60,
    )


def _collect_output(mineru_out: Path, result_dir: Path, engine: str, input_file: Path) -> None:
    """Copy MinerU's markdown + images into a flat, stable layout:
    result_dir/document.md and result_dir/images/ (markdown references images/...)."""
    md_files = sorted(mineru_out.rglob("*.md"))
    if not md_files:
        raise RuntimeError("mineru produced no markdown output")
    src_md = md_files[0]

    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)
    md_text = _finalize_markdown(src_md, engine, input_file)
    (result_dir / "document.md").write_text(md_text, encoding="utf-8")

    src_images = src_md.parent / "images"
    if src_images.is_dir():
        shutil.copytree(src_images, result_dir / "images")


def _finalize_markdown(src_md: Path, engine: str, input_file: Path) -> str:
    """Fix reversed-RTL OCR output and (for hybrid) swap block text for Qari's.

    Preferred path: mutate spans in middle.json and let MinerU reassemble the
    markdown (preserves line order inside paragraphs). Fallback for mineru-only:
    line-level bidi fix on the markdown text. Hybrid has no fallback — it must
    fail loudly rather than silently return MinerU-quality output."""
    middle_files = sorted(src_md.parent.glob("*middle.json"))
    if not middle_files and engine == "hybrid":
        raise RuntimeError("hybrid engine requires middle.json from mineru")
    if middle_files:
        try:
            import json

            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
            from mineru.utils.enum_class import MakeMode

            middle = json.loads(middle_files[0].read_text(encoding="utf-8"))
            changed = rtl.fix_middle_json(middle)
            if engine == "hybrid":
                from app.worker import hybrid

                stats = hybrid.apply_qari(middle, input_file)
                if not stats["replaced"]:
                    raise RuntimeError(f"qari replaced no blocks: {stats}")
                changed = True
            if changed:
                made = union_make(middle["pdf_info"], MakeMode.MM_MD, "images")
                return made if isinstance(made, str) else "\n\n".join(made)
            return src_md.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 - md-level fallback is mineru-only
            if engine == "hybrid":
                raise
    return rtl.fix_rtl_markdown(src_md.read_text(encoding="utf-8"))
