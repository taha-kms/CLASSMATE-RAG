"""A tiny in-memory registry for the admin operations that take minutes.

`rebuild` re-embeds the whole corpus and `restore` rewrites it. Neither can sit
in a request that blocks for ten minutes, so they run on a thread and the
caller polls.

Deliberately not durable. Jobs die with the process, which is correct here: one
student, one machine, one process, and a rebuild interrupted by a restart has
to be started again anyway. Persisting them would mean reasoning about jobs
that claim to be running with nothing behind them.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

log = logging.getLogger(__name__)

JobState = Literal["running", "succeeded", "failed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    operation: str
    state: JobState = "running"
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRegistry:
    """Runs one operation per thread and remembers how it went."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, operation: str, fn: Callable[[], dict[str, Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex, operation=operation)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001 - a job records its failure rather than killing the thread
                # There is no caller to propagate to: this runs on its own
                # thread, and an escaping exception would leave the job stuck
                # at "running" forever.
                #
                # The traceback goes to the log rather than into the response.
                # It names internal paths and frames, and the operator running
                # `rag serve` is watching that terminal anyway; `error` carries
                # the one line a UI needs to show.
                log.exception("job %s (%s) failed", job.id, operation)
                with self._lock:
                    job.state = "failed"
                    job.error = f"{type(e).__name__}: {e}"
                    job.finished_at = _now()
            else:
                with self._lock:
                    job.state = "succeeded"
                    job.result = result
                    job.finished_at = _now()

        threading.Thread(target=run, name=f"job-{operation}-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
