import redis
from rq import Queue

from app import config

_redis = redis.from_url(config.REDIS_URL)
ocr_queue = Queue("ocr", connection=_redis, default_timeout=config.OCR_JOB_TIMEOUT)


def enqueue_ocr(job_id: str) -> None:
    ocr_queue.enqueue(
        "app.worker.ocr.run_ocr",
        job_id,
        job_timeout=config.OCR_JOB_TIMEOUT,
        result_ttl=300,
        failure_ttl=86400,
    )


def enqueue_extract(ext_id: str) -> None:
    # Same queue as OCR: heavy model work stays serialized on the small VM.
    ocr_queue.enqueue(
        "app.worker.extract.run_extract",
        ext_id,
        job_timeout=config.EXTRACT_TIMEOUT + 120,
        result_ttl=300,
        failure_ttl=86400,
    )
