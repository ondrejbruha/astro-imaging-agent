"""Provider-independent cooperative execution, also usable by custom tools."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from threading import Event
from typing import Any, cast


class ExecutionCancelled(Exception):
    """Signal cooperative cancellation without being mistaken for a bad frame."""


class CancellationToken:
    """Thread-safe, idempotent cancellation signal checked at safe boundaries."""

    def __init__(self) -> None:
        """Create an initially unset signal."""
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation; this does not imply that processing has stopped."""
        self._event.set()

    @property
    def requested(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def check(self) -> None:
        """Raise the dedicated signal when cancellation was requested."""
        if self.requested:
            raise ExecutionCancelled("Execution was cancelled.")


@dataclass(frozen=True)
class Progress:
    """Units describe work counts, never elapsed-time fractions; None is indeterminate."""

    phase: str
    completed_units: int | None = None
    total_units: int | None = None
    unit: str | None = None
    step_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a small JSON-compatible callback payload."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class ExecutionContext:
    """Optional execution services; budgets are estimates rather than process RSS caps."""

    cancellation: CancellationToken = field(default_factory=CancellationToken)
    progress: Callable[[Progress], None] | None = None
    scratch_directory: Path | None = None
    memory_mb: int | None = None
    scratch_bytes: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid optional budgets before creating temporary processing artifacts."""
        for name in ("memory_mb", "scratch_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Propagate context through existing tools without changing process signatures."""
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)


_current: ContextVar[ExecutionContext | None] = ContextVar("astroagent_execution", default=None)


def current_context() -> ExecutionContext | None:
    """Return the explicitly activated context for this thread/task, if present."""
    return _current.get()


def checkpoint() -> None:
    """Check cancellation without requiring context-aware custom tool methods."""
    context = current_context()
    if context is not None:
        context.cancellation.check()


def emit_progress(
    phase: str,
    completed: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    *,
    step_index: int | None = None,
) -> None:
    """Emit truthful progress; each phase occurrence starts a new counter sequence."""
    checkpoint()
    context = current_context()
    if context is not None and context.progress is not None:
        context.progress(Progress(phase, completed, total, unit, step_index))


def execution_scope[F: Callable[..., Any]](function: F) -> F:
    """Activate an optional explicit context while retaining existing signatures."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context") or current_context()
        if context is None:
            return function(*args, **kwargs)
        with context.activate():
            checkpoint()
            result = function(*args, **kwargs)
            checkpoint()
            return result

    return cast(F, wrapped)
