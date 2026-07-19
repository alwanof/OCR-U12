"""Ensemble engine: Qari (X) + Qwen3-VL via Ollama (Y) -> judge -> final text.

Runs in three phases per job, not per block, because the Docker VM cannot hold
Qari (~5 GB in the worker) and qwen3-vl (~6 GB in the Ollama container) at
once: first all Qari OCR (then unload), then all VLM OCR, then all judging.

Judge mode is configurable (ENSEMBLE_JUDGE_MODE): "text" gives the judge only
the two candidates; "vision" also gives it the block image.
"""

import logging
from pathlib import Path

from app.worker import hybrid, qari, vlm

logger = logging.getLogger(__name__)


def apply_ensemble(middle: dict, input_file: Path) -> dict:
    stats = {"replaced": 0, "qari_failed": 0, "vlm_failed": 0, "judge_failed": 0}

    # Phase 1: Qari on every block, then free its memory.
    items = []  # (block, crop, x_text)
    try:
        for block, crop in hybrid.iter_text_blocks(middle, input_file):
            try:
                x_text = qari.ocr_image(crop)
            except Exception:  # noqa: BLE001
                logger.exception("qari failed on block %s", block["bbox"])
                stats["qari_failed"] += 1
                x_text = ""
            items.append((block, crop, x_text))
    finally:
        qari.unload()

    # Phase 2: VLM on every block.
    with_y = []
    for block, crop, x_text in items:
        try:
            y_text = vlm.vlm_ocr(crop)
        except Exception:  # noqa: BLE001
            logger.exception("vlm failed on block %s", block["bbox"])
            stats["vlm_failed"] += 1
            y_text = ""
        with_y.append((block, crop, x_text, y_text))

    # Phase 3: judge X vs Y per block.
    for block, crop, x_text, y_text in with_y:
        if not x_text and not y_text:
            continue
        if not x_text or not y_text:
            final = x_text or y_text
        else:
            try:
                final = vlm.judge(x_text, y_text, image=crop)
            except Exception:  # noqa: BLE001
                logger.exception("judge failed on block %s", block["bbox"])
                stats["judge_failed"] += 1
                final = y_text  # VLM output as fallback
        if final:
            hybrid.set_block_text(block, final)
            stats["replaced"] += 1

    logger.info("ensemble stats: %s", stats)
    return stats
