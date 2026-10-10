"""Strict envelopes, bounded UTF-8 framing, and serialized protocol writes."""

import json
from collections.abc import Callable
from threading import Lock
from typing import Any, BinaryIO, Literal

from pydantic import Field, JsonValue, StrictInt, StrictStr, model_validator

from astroagent.models.base import SchemaModel

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_IDENTIFIER_LENGTH = 128


class Request(SchemaModel):
    """Version-one method request; identifiers are unique throughout a session."""

    protocol_version: StrictInt
    type: Literal["request"]
    id: StrictStr = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    method: StrictStr = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class ErrorInfo(SchemaModel):
    """Safe machine-readable failure; details contain locations rather than input values."""

    code: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Response(SchemaModel):
    """Correlated success/error, or null ID when the request cannot be recovered."""

    protocol_version: Literal[1]
    type: Literal["response"]
    id: str | None = Field(max_length=MAX_IDENTIFIER_LENGTH)
    result: JsonValue = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def exclusive_outcome(self) -> "Response":
        """Require exactly one response outcome field."""
        if ("result" in self.model_fields_set) == ("error" in self.model_fields_set):
            raise ValueError("Response requires exactly one result or error.")
        return self


class JobEvent(SchemaModel):
    """Structured event with explicit originating request and generated job identifiers."""

    protocol_version: Literal[1]
    type: Literal["event"]
    request_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    job_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    event: Literal["job.started", "job.progress", "job.completed", "job.failed", "job.cancelled"]
    data: dict[str, JsonValue]


class WorkerError(Exception):
    """Static safe error message and optional parameter locations, never SDK bodies."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Construct a transport-safe failure."""
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}

    def payload(self) -> dict[str, Any]:
        """Return the documented error envelope content."""
        return {"code": self.code, "message": self.message, "details": self.details}


def decode_line(line: bytes) -> Any:
    """Reject nonfinite numbers and duplicate keys, including nested JSON objects."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field.")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("Nonfinite JSON number.")

    return json.loads(line.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


class Transport:
    """One lock and one stream serialize complete bounded messages with prompt flushing."""

    def __init__(self, output: BinaryIO, *, fatal: Callable[[], None] | None = None) -> None:
        """Use the original stdout stream, independent of redirected library prints."""
        self.output = output
        self.lock = Lock()
        self.fatal = fatal

    def send(self, payload: dict[str, Any]) -> None:
        """Write one finite JSON object; refuse arrays/binary through the JSON encoder."""
        envelope = {"protocol_version": PROTOCOL_VERSION, **payload}
        schema = JobEvent if payload.get("type") == "event" else Response
        schema.model_validate(envelope)
        data = (
            json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(data) > MAX_MESSAGE_BYTES:
            raise WorkerError(
                "RESULT_TOO_LARGE", "Result exceeds the message limit; use artifact references."
            )
        with self.lock:
            try:
                self.output.write(data)
                self.output.flush()
            except OSError:
                if self.fatal is not None:
                    self.fatal()
                raise

    def response(self, request_id: str | None, result: Any) -> None:
        """Respond to a correlated request."""
        self.send({"type": "response", "id": request_id, "result": result})

    def error(self, request_id: str | None, error: WorkerError) -> None:
        """Report malformed uncorrelated input using a null response identifier."""
        self.send({"type": "response", "id": request_id, "error": error.payload()})
