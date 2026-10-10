"""A single processing thread, bounded pending work, and explicit artifact finalization."""

import json
import shutil
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Thread
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from astroagent.errors import AstroError, ImageIOError, PipelineValidationError, ResourceLimitError
from astroagent.execution import ExecutionCancelled, ExecutionContext, Progress
from astroagent.io.datasets import write_json
from astroagent.worker.protocol import Transport, WorkerError

JobState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


def safe_error(exc: Exception) -> WorkerError:
    """Classify failures without exposing untrusted exception strings or credentials."""
    if isinstance(exc, WorkerError):
        return exc
    if isinstance(exc, PipelineValidationError):
        return WorkerError(
            "INVALID_PARAMS",
            "Pipeline parameters or transitions are invalid.",
            {"locations": exc.locations},
        )
    if isinstance(exc, MemoryError | ResourceLimitError):
        return WorkerError(
            "RESOURCE_LIMIT", "Memory was exhausted; reduce image size or tile/memory budgets."
        )
    if isinstance(exc, OSError | ImageIOError) or isinstance(exc.__cause__, OSError):
        return WorkerError(
            "IO_ERROR", "File operation failed; check permissions, free space, and input format."
        )
    if isinstance(exc, AstroError | ValueError):
        return WorkerError(
            "INCOMPATIBLE_DATA",
            "Input or pipeline is incompatible; check layout, metadata, and parameters.",
        )
    return WorkerError(
        "INTERNAL_ERROR",
        "Processing failed internally; restart the worker and retain the incomplete workspace.",
    )


@dataclass
class Job:
    """Bounded state with no retained pixel arrays, credentials, or provider clients."""

    id: str
    request_id: str
    method: str
    workspace: Path
    context: ExecutionContext
    state: JobState = "QUEUED"
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    finalizing: bool = False
    last_progress_time: float = 0
    last_phase: str | None = None

    def status(self) -> dict[str, Any]:
        """Expose only compact state and references, including recoverable incomplete work."""
        return {
            "job_id": self.id,
            "state": self.state,
            "cancellation_requested": self.context.cancellation.requested,
            "finalizing": self.finalizing,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "workspace": str(self.workspace),
            "artifacts_complete": self.state == "COMPLETED",
        }


class JobManager:
    """Accept control traffic concurrently with one serial numerical worker."""

    def __init__(self, transport: Transport, *, queue_limit: int = 8, retention: int = 128) -> None:
        """Start exactly one worker thread with bounded completed-state retention."""
        if queue_limit < 1 or retention < 1:
            raise ValueError("Job queue and retention limits must be positive.")
        self.transport = transport
        self.queue_limit, self.retention = queue_limit, retention
        self.condition = Condition()
        self.pending: deque[tuple[Job, Callable[[Job], dict[str, Any]]]] = deque()
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.stopping = False
        self.thread = Thread(target=self._loop, name="aia-processing", daemon=False)
        self.thread.start()

    def submit(
        self,
        request_id: str,
        method: str,
        root: Path,
        runner: Callable[[Job], dict[str, Any]],
        *,
        memory_mb: int | None = None,
        scratch_bytes: int | None = None,
    ) -> str:
        """Send acceptance under the scheduler lock before any events can be emitted."""
        with self.condition:
            if self.stopping:
                raise WorkerError("SHUTTING_DOWN", "Worker is shutting down.")
            if len(self.pending) >= self.queue_limit:
                raise WorkerError(
                    "QUEUE_FULL", "Processing queue is full; wait for a job to finish."
                )
            job_id = "job-" + uuid4().hex
            workspace = root / job_id
            context = ExecutionContext(memory_mb=memory_mb, scratch_bytes=scratch_bytes)
            job = Job(job_id, request_id, method, workspace, context)
            context.progress = lambda progress: self._progress(job, progress)
            self.jobs[job_id] = job
            self.pending.append((job, runner))
            self.transport.response(request_id, {"job_id": job_id, "state": "QUEUED"})
            self.condition.notify()
            return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        """Return a bounded snapshot or an explicit expired/unknown-job error."""
        with self.condition:
            return self._find(job_id).status()

    def _find(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise WorkerError("JOB_NOT_FOUND", "Job is unknown or its retained status has expired.")
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Acknowledge a request; terminal outcomes and sealed finalization cannot change."""
        with self.condition:
            job = self._find(job_id)
            accepted = job.state not in TERMINAL and not job.finalizing
            if accepted:
                job.context.cancellation.cancel()
            return {
                "job_id": job.id,
                "accepted": accepted,
                "state": job.state,
                "cancellation_requested": job.context.cancellation.requested,
                "finalizing": job.finalizing,
            }

    def shutdown(self) -> None:
        """Stop accepting jobs and request cancellation of all unsealed work."""
        with self.condition:
            self.stopping = True
            for job in self.jobs.values():
                if job.state not in TERMINAL and not job.finalizing:
                    job.context.cancellation.cancel()
            self.condition.notify_all()

    def join(self) -> None:
        """Wait for cooperative stop; indivisible kernels/finite provider calls may delay it."""
        self.thread.join()

    def _event(self, job: Job, event: str, data: dict[str, Any]) -> None:
        self.transport.send(
            {
                "type": "event",
                "request_id": job.request_id,
                "job_id": job.id,
                "event": event,
                "data": data,
            }
        )

    def _progress(self, job: Job, progress: Progress) -> None:
        now = monotonic()
        with self.condition:
            payload = progress.as_dict()
            job.progress = payload
            complete = (
                progress.total_units is not None
                and progress.completed_units == progress.total_units
            )
            if progress.phase != job.last_phase or complete or now - job.last_progress_time >= 0.1:
                job.last_progress_time, job.last_phase = now, progress.phase
                self._event(job, "job.progress", payload)

    def _finish(self, job: Job, state: JobState) -> None:
        job.state = state
        self._event(job, f"job.{state.lower()}", job.status())
        completed = [key for key, value in self.jobs.items() if value.state in TERMINAL]
        for key in completed[: -self.retention]:
            del self.jobs[key]
        self.condition.notify_all()

    def _loop(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: bool(self.pending) or self.stopping)
                if not self.pending:
                    return
                job, runner = self.pending.popleft()
                if job.context.cancellation.requested:
                    self._finish(job, "CANCELLED")
                    continue
                job.state = "RUNNING"
                self._event(job, "job.started", {"state": "RUNNING"})
            try:
                job.context.cancellation.check()
                job.workspace.mkdir(parents=True, exist_ok=False)
                scratch = job.workspace / "scratch"
                scratch.mkdir()
                job.context.scratch_directory = scratch
                write_json(
                    job.workspace / "ownership.json", {"job_id": job.id, "method": job.method}
                )
                with job.context.activate():
                    result = runner(job)
                if len(json.dumps(result, allow_nan=False).encode("utf-8")) > 16384:
                    raise WorkerError(
                        "RESULT_TOO_LARGE", "Job results must be compact artifact references."
                    )
                self._clean_scratch(job)
                # Seal under the same lock as cancellation. All processing and artifact
                # writes have finished; only the bounded completion marker remains.
                with self.condition:
                    job.context.cancellation.check()
                    job.finalizing = True
                write_json(job.workspace / "complete.json", {"job_id": job.id, "result": result})
                with self.condition:
                    job.result = {**result, "manifest": str(job.workspace / "complete.json")}
                outcome: JobState = "COMPLETED"
            except ExecutionCancelled:
                outcome = "CANCELLED"
            except Exception as exc:
                job.error = safe_error(exc).payload()
                outcome = "FAILED"
            finally:
                try:
                    self._clean_scratch(job)
                except OSError:
                    # Retain failed cleanup as explicitly incomplete recoverable work.
                    job.error = WorkerError(
                        "IO_ERROR", "Scratch cleanup failed; retain the job-owned workspace."
                    ).payload()
                del runner  # Release closures containing in-memory credentials.
            with self.condition:
                self._finish(job, outcome)

    def _clean_scratch(self, job: Job) -> None:
        scratch = job.context.scratch_directory
        if scratch is None or not scratch.exists():
            return
        # Check resolved ancestry immediately before any recursive deletion.
        resolved = scratch.resolve(strict=True)
        workspace = job.workspace.resolve(strict=True)
        if scratch.is_symlink() or resolved.parent != workspace or scratch.name != "scratch":
            raise OSError("Scratch ownership or ancestry changed.")
        shutil.rmtree(scratch)
