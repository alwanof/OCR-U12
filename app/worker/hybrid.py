"""Hybrid engine: MinerU layout + Qari-OCR recognition.

Takes MinerU's middle.json (after the RTL fix), re-renders each page, crops
every text-bearing block, runs Qari on the crop, and replaces the block's
text content in place. Tables/figures/formulas keep MinerU's output (per-cell
Qari OCR is a production-phase item). Markdown is then reassembled by the
caller via MinerU's union_make, so reading order and structure are preserved.
"""

import logging
from pathlib import Path

from PIL import Image

from app import config
from app.worker import qari

logger = logging.getLogger(__name__)

# Block types whose text Qari replaces. Tables/images/formulas stay MinerU's.
REPLACE_TYPES = {"text", "title", "list", "index"}


def apply_qari(middle: dict, input_file: Path) -> dict:
    stats = {"replaced": 0, "kept_mineru": 0, "failed": 0}
    try:
        for block, crop in iter_text_blocks(middle, input_file):
            try:
                text = qari.ocr_image(crop)
            except Exception:  # noqa: BLE001 - keep MinerU text for this block
                logger.exception("qari failed on block %s", block["bbox"])
                stats["failed"] += 1
                continue
            if not text:
                stats["kept_mineru"] += 1
                continue
            set_block_text(block, text)
            stats["replaced"] += 1
    finally:
        qari.unload()
    logger.info("hybrid stats: %s", stats)
    return stats


def iter_text_blocks(middle: dict, input_file: Path):
    """Yield (block, crop_image) for every text-bearing block, in order."""
    pages = middle.get("pdf_info", [])
    for page, image in zip(pages, _render_pages(input_file, len(pages))):
        if image is None:
            continue
        page_w, page_h = page.get("page_size") or (image.width, image.height)
        sx, sy = image.width / page_w, image.height / page_h
        for block in page.get("para_blocks") or []:
            if block.get("type") not in REPLACE_TYPES:
                continue
            x0, y0, x1, y1 = block["bbox"]
            crop = image.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
            if crop.width < 8 or crop.height < 8:
                continue
            yield block, crop


def set_block_text(block: dict, text: str) -> None:
    block["lines"] = [
        {
            "bbox": block["bbox"],
            "spans": [
                {"bbox": block["bbox"], "type": "text", "content": text, "score": 1.0}
            ],
        }
    ]


def _render_pages(input_file: Path, page_count: int):
    if input_file.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(input_file))
        try:
            for i in range(page_count):
                if i >= len(doc):
                    yield None
                    continue
                yield doc[i].render(scale=config.QARI_RENDER_SCALE).to_pil().convert("RGB")
        finally:
            doc.close()
    else:
        with Image.open(input_file) as im:
            yield im.convert("RGB")
        for _ in range(page_count - 1):
            yield None
