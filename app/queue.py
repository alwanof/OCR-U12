import redis
from rq import Callback, Queue

from app import config

_redis = redis.from_url(config.REDIS_URL)
ocr_queue = Queue("ocr", connection=_redis, default_timeout=config.OCR_JOB_TIMEOUT)

_ON_FAILURE = Callback("app.worker.callbacks.on_job_failure")


def enqueue_ocr(job_id: str) -> None:
    ocr_queue.enqueue(
        "app.worker.ocr.run_ocr",
        job_id,
        job_timeout=config.OCR_JOB_TIMEOUT,
        result_ttl=300,
        failure_ttl=86400,
        on_failure=_ON_FAILURE,
    )


def active_task_ids() -> set[str]:
    """First args (our job/extraction ids) of tasks currently held by any worker."""
    from rq import Worker

    ids = set()
    for w in Worker.all(connection=_redis):
        job = w.get_current_job()
        if job and job.args:
            ids.add(str(job.args[0]))
    return ids


def queued_task_ids() -> set[str]:
    ids = set()
    for job in ocr_queue.jobs:
        if job.args:
            ids.add(str(job.args[0]))
    return ids


def enqueue_extract(ext_id: str) -> None:
    # Same queue as OCR: heavy model work stays serialized on the small VM.
    ocr_queue.enqueue(
        "app.worker.extract.run_extract",
        ext_id,
        job_timeout=config.EXTRACT_TIMEOUT + 120,
        result_ttl=300,
        failure_ttl=86400,
        on_failure=_ON_FAILURE,
    )
