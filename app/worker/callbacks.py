"""RQ failure callbacks: run in the worker's main process, so they fire even
when the work-horse is OOM-killed (SIGKILL) and the task's own error handling
never executes — without this, jobs stay 'processing' in the DB forever."""

from app import db


def on_job_failure(job, connection, exc_type, exc_value, tb) -> None:
    db.init_db()
    target_id = job.args[0] if job.args else None
    if not target_id:
        return
    reason = f"{getattr(exc_type, '__name__', exc_type)}: {exc_value}"
    if "run_extract" in (job.func_name or ""):
        ext = db.get_extraction(target_id)
        if ext is not None and ext["status"] != "failed":
            db.fail_extraction(target_id, reason)
    else:
        j = db.get_job(target_id)
        if j is not None and j["status"] not in ("failed", "done"):
            db.mark_failed(target_id, reason)
