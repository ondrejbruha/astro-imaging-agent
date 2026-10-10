"""Allowlisted worker methods backed by existing deterministic APIs and planners."""

import hashlib
import math
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO
from uuid import uuid4

from pydantic import ValidationError

from astroagent import __version__
from astroagent.agent.executor import AgentExecutor
from astroagent.agent.models import PlanResult
from astroagent.agent.providers import create_planner
from astroagent.analysis.statistics import ImageMetrics, inspect_image
from astroagent.calibration.session import inspect_session_frames
from astroagent.errors import AstroError, PipelineValidationError, ProviderError
from astroagent.execution import ExecutionCancelled, checkpoint, emit_progress
from astroagent.io.datasets import write_json
from astroagent.io.fits import list_hdus
from astroagent.io.images import image_format, load_image
from astroagent.models.base import SchemaModel
from astroagent.models.dataset import AstroDataset, DatasetMetrics
from astroagent.pipeline.dataset_executor import execute_dataset_pipeline
from astroagent.pipeline.executor import PipelineExecutor, save_result
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import load_pipeline, save_pipeline
from astroagent.preview import PreviewOptions, create_preview
from astroagent.registration.engine import analyze_frames
from astroagent.worker.jobs import Job, JobManager, safe_error
from astroagent.worker.models import (
    DatasetInput,
    DatasetParams,
    EmptyParams,
    ExecuteParams,
    HandshakeParams,
    ImageInput,
    InspectParams,
    JobParams,
    PathParams,
    PipelineLoadParams,
    PipelineParams,
    PipelineSaveParams,
    PlanParams,
    PreviewParams,
    ProviderConfig,
    TestProviderParams,
)
from astroagent.worker.protocol import (
    MAX_MESSAGE_BYTES,
    JobEvent,
    Request,
    Response,
    Transport,
    WorkerError,
    decode_line,
)

METHODS: dict[str, type[SchemaModel]] = {
    "worker.handshake": HandshakeParams,
    "worker.shutdown": EmptyParams,
    "tools.list": EmptyParams,
    "pipeline.validate": PipelineParams,
    "pipeline.load": PipelineLoadParams,
    "pipeline.save": PipelineSaveParams,
    "pipeline.execute": ExecuteParams,
    "image.hdus": PathParams,
    "image.inspect": InspectParams,
    "image.preview": PreviewParams,
    "session.inspect": DatasetParams,
    "dataset.analyze": DatasetParams,
    "agent.plan": PlanParams,
    "llm.test": TestProviderParams,
    "job.status": JobParams,
    "job.cancel": JobParams,
}
FAST_METHODS = {
    "worker.handshake",
    "worker.shutdown",
    "tools.list",
    "pipeline.validate",
    "job.status",
    "job.cancel",
}


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _redact(value: Any, key: str | None) -> Any:
    if isinstance(value, str) and key:
        return value.replace(key, "[redacted]")
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, dict):
        return {str(name): _redact(item, key) for name, item in value.items()}
    return value


class Server:
    """Session-scoped filesystem policy and read loop independent of processing work."""

    def __init__(self, input_stream: BinaryIO, transport: Transport) -> None:
        """Create bounded scheduling without accessing providers or input files."""
        self.input_stream, self.transport = input_stream, transport
        self.executor = PipelineExecutor()
        self.jobs = JobManager(transport)
        self.ids: set[str] = set()
        self.config: HandshakeParams | None = None
        self.stopping = False
        self.plans: dict[str, tuple[float, dict[str, Any]]] = {}
        self.previews: dict[str, tuple[int, str]] = {}

    def run(self) -> None:
        """Accept fragmented/batched LF or CRLF lines; EOF cancels all unsealed work."""
        try:
            while not self.stopping:
                line = self.input_stream.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES:
                    while line and not line.endswith(b"\n"):
                        line = self.input_stream.readline(MAX_MESSAGE_BYTES + 1)
                    self.transport.error(
                        None, WorkerError("MESSAGE_TOO_LARGE", "Request exceeds the byte limit.")
                    )
                    continue
                self.dispatch(line)
        finally:
            self.jobs.shutdown()
            self.jobs.join()

    def dispatch(self, line: bytes) -> None:
        """Validate envelopes/version/uniqueness before any method side effects."""
        request_id = None
        try:
            try:
                document = decode_line(line)
                if not _finite(document):
                    raise ValueError("Nonfinite number.")
            except (ValueError, UnicodeError, RecursionError):
                raise WorkerError(
                    "INVALID_JSON", "Provide one finite UTF-8 JSON object per line."
                ) from None
            if isinstance(document, dict):
                candidate = document.get("id")
                if isinstance(candidate, str) and 0 < len(candidate) <= 128:
                    request_id = candidate
            try:
                request = Request.model_validate(document)
            except ValidationError:
                raise WorkerError(
                    "INVALID_REQUEST",
                    "Request envelope is invalid; check fields and identifier lengths.",
                ) from None
            request_id = request.id
            if request.protocol_version != 1:
                raise WorkerError("UNSUPPORTED_VERSION", "Only protocol version 1 is supported.")
            if request.id in self.ids:
                raise WorkerError(
                    "DUPLICATE_ID", "Request ID was already used in this worker session."
                )
            if len(self.ids) >= 10000:
                raise WorkerError(
                    "SESSION_LIMIT", "Session request limit reached; start a fresh worker."
                )
            self.ids.add(request.id)
            if request.method not in METHODS:
                raise WorkerError("METHOD_NOT_FOUND", "Method is not supported by this worker.")
            try:
                params = METHODS[request.method].model_validate(request.params)
            except ValidationError as exc:
                locations: list[dict[str, Any]] = [
                    {"field": list(error["loc"]), "type": error["type"]}
                    for error in exc.errors(include_input=False, include_context=False)
                ]
                for location in locations:
                    field = location["field"]
                    if (
                        isinstance(field, list)
                        and len(field) >= 3
                        and field[:2] == ["pipeline", "steps"]
                        and isinstance(field[2], int)
                    ):
                        location["step_index"] = field[2] + 1
                        location["field"] = field[3:]
                supplied_config = request.params.get("config")
                supplied_key = (
                    supplied_config.get("api_key") if isinstance(supplied_config, dict) else None
                )
                locations = _redact(
                    locations, supplied_key if isinstance(supplied_key, str) else None
                )
                raise WorkerError(
                    "INVALID_PARAMS", "Method parameters are invalid.", {"locations": locations}
                ) from None
            if request.method == "worker.handshake":
                assert isinstance(params, HandshakeParams)
                self.transport.response(request.id, self._handshake(params))
            elif self.config is None:
                raise WorkerError(
                    "HANDSHAKE_REQUIRED", "Complete worker.handshake before calling other methods."
                )
            elif request.method == "worker.shutdown":
                self.stopping = True
                self.jobs.shutdown()
                self.transport.response(
                    request.id, {"shutdown_requested": True, "policy": "cancel-unsealed-jobs"}
                )
            elif request.method == "tools.list":
                self.transport.response(request.id, {"tools": self._catalog()})
            elif request.method == "pipeline.validate":
                assert isinstance(params, PipelineParams)
                self.transport.response(
                    request.id,
                    {
                        "pipeline": self._resolved(params.pipeline, params.input_kind).model_dump(
                            mode="json"
                        )
                    },
                )
            elif request.method in {"job.status", "job.cancel"}:
                assert isinstance(params, JobParams)
                operation = self.jobs.get if request.method == "job.status" else self.jobs.cancel
                self.transport.response(request.id, operation(params.job_id))
            else:
                if self.config.workspace_root is None or not self.config.read_roots:
                    raise WorkerError(
                        "PATH_POLICY_REQUIRED",
                        "Handshake must configure absolute read_roots and workspace_root for jobs.",
                    )
                with self.jobs.condition:
                    if isinstance(params, ExecuteParams):
                        self._resolved(params.pipeline, params.input.kind)
                        if params.input.kind == "image" or any(
                            step.tool == "stack_frames" for step in params.pipeline.steps
                        ):
                            image_format(params.output_name)
                    previous = None
                    if isinstance(params, PreviewParams):
                        previous = self.previews.get(params.channel)
                        if previous and params.generation <= previous[0]:
                            raise WorkerError(
                                "STALE_PREVIEW",
                                "Preview generation must increase within its channel.",
                            )
                        if params.channel not in self.previews and len(self.previews) >= 64:
                            raise WorkerError(
                                "RESOURCE_LIMIT",
                                "Preview channel limit reached; restart the worker.",
                            )
                    job_id = self.jobs.submit(
                        request.id,
                        request.method,
                        self.config.workspace_root,
                        lambda job: self._execute(request.method, params, job),
                        memory_mb=self.config.memory_mb,
                        scratch_bytes=self.config.scratch_bytes,
                    )
                    if isinstance(params, PreviewParams):
                        self.previews[params.channel] = (params.generation, job_id)
                        if previous:
                            try:
                                self.jobs.cancel(previous[1])
                            except WorkerError:
                                pass  # Previous completed status may already have expired.
        except WorkerError as exc:
            self.transport.error(request_id, exc)
        except Exception as exc:
            self.transport.error(request_id, safe_error(exc))

    def _handshake(self, params: HandshakeParams) -> dict[str, Any]:
        paths = [
            *params.read_roots,
            *([] if params.workspace_root is None else [params.workspace_root]),
        ]
        if any(not path.is_absolute() or len(str(path)) > 4096 for path in paths):
            raise WorkerError(
                "INVALID_PARAMS", "Handshake paths must be absolute host-selected locations."
            )
        normalized = params.model_copy(
            update={
                "read_roots": [path.resolve() for path in params.read_roots],
                "workspace_root": None
                if params.workspace_root is None
                else params.workspace_root.resolve(),
            }
        )
        if self.config is not None and self.config != normalized:
            raise WorkerError(
                "CONFIG_CONFLICT", "Worker configuration is immutable; restart to change it."
            )
        self.config = normalized
        return {
            "protocol_version": 1,
            "supported_protocol_versions": [1],
            "aia_version": __version__,
            "dependency_versions": {
                name: version(name)
                for name in ("numpy", "scipy", "astropy", "photutils", "pydantic", "tifffile")
            },
            "capabilities": {
                "methods": list(METHODS),
                "providers": ["rules", "openai", "anthropic", "gemini"],
                "provider_sdks_optional": True,
                "selected_fits_hdu": True,
                "hfr": True,
                "preview_strategy": "full-resolution-then-crop-resize",
                "preview_cache": False,
                "single_image_formats": ["fits", "tiff", "png", "jpeg", "webp", "bmp"],
                "dataset_formats": ["fits"],
                "cooperative_cancellation": True,
                "token_streaming": False,
                "endpoint_customization": False,
            },
            "limits": {
                "message_bytes": MAX_MESSAGE_BYTES,
                "identifier_characters": 128,
                "queued_jobs": self.jobs.queue_limit,
                "processing_jobs": 1,
                "completed_job_retention": self.jobs.retention,
                "session_requests": 10000,
                "plan_retention": 32,
                "plan_ttl_seconds": 300,
                "preview_channels": 64,
                "memory_mb": params.memory_mb,
                "scratch_bytes": params.scratch_bytes,
            },
            "schemas": {
                "request": Request.model_json_schema(),
                "response": Response.model_json_schema(),
                "event": JobEvent.model_json_schema(),
                "methods": {method: model.model_json_schema() for method, model in METHODS.items()},
            },
        }

    def _catalog(self) -> list[dict[str, Any]]:
        return [
            description.model_dump(mode="json") for description in self.executor.registry.describe()
        ]

    def _resolved(self, pipeline: PipelineDefinition, kind: Any) -> PipelineDefinition:
        resolved = self.executor.validate(pipeline, input_kind=kind)
        return PipelineDefinition(
            steps=[
                PipelineStep(tool=tool.name, params=params.model_dump(mode="json"))
                for tool, params in resolved
            ]
        )

    def _read_path(self, path: Path) -> Path:
        assert self.config is not None
        if not path.is_absolute():
            raise WorkerError("PATH_DENIED", "Input paths must be absolute.")
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(root) for root in self.config.read_roots):
            raise WorkerError("PATH_DENIED", "Input is outside the host-selected read locations.")
        if not resolved.is_file():
            raise WorkerError("INVALID_PARAMS", "Select an explicit input file, not a directory.")
        return resolved

    def _dataset(self, reference: DatasetInput) -> AstroDataset:
        paths = [self._read_path(path) for path in reference.frames]
        if len(set(paths)) != len(paths) or len({path.stat().st_ino for path in paths}) != len(
            paths
        ):
            raise WorkerError(
                "INVALID_PARAMS", "Dataset selection contains duplicate files or aliases."
            )
        if any(
            not (
                path.name.lower().endswith(
                    (".fit", ".fits", ".fts", ".fit.gz", ".fits.gz", ".fts.gz")
                )
            )
            for path in paths
        ):
            raise WorkerError(
                "UNSUPPORTED_CAPABILITY", "Session and dataset operations support FITS frames only."
            )
        selected = None if reference.reference is None else self._read_path(reference.reference)
        if selected is not None and selected not in paths:
            raise WorkerError(
                "INVALID_PARAMS", "Reference must belong to the explicit selected dataset."
            )
        purposes = {
            str(self._read_path(Path(name))): purpose.value
            for name, purpose in reference.purposes.items()
        }
        if not set(purposes).issubset(set(map(str, paths))):
            raise WorkerError("INVALID_PARAMS", "Purpose overrides must identify selected frames.")
        return AstroDataset(
            paths,
            reference=selected,
            purpose_overrides=purposes,
            reports={"selection": {"frames": list(map(str, paths)), "purposes": purposes}},
        )

    def _identity(
        self, reference: ImageInput | DatasetInput, *, hash_contents: bool = True
    ) -> dict[str, Any]:
        paths = [reference.path] if isinstance(reference, ImageInput) else reference.frames
        entries = []
        for item in paths:
            path = self._read_path(item)
            before = path.stat()
            digest = None
            if hash_contents:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        checkpoint()
                        digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise WorkerError(
                    "INPUT_CHANGED", "Input changed during inspection; inspect and review it again."
                )
            entries.append(
                {
                    "path": str(path),
                    "size": after.st_size,
                    "mtime_ns": after.st_mtime_ns,
                    "sha256": None if digest is None else digest.hexdigest(),
                }
            )
        return {"reference": reference.model_dump(mode="json"), "files": entries}

    def _planner(self, config: ProviderConfig) -> Any:
        key = None if config.api_key is None else config.api_key.get_secret_value()
        return create_planner(config.provider, config.model, timeout=config.timeout, api_key=key)

    def _execute(self, method: str, params: SchemaModel, job: Job) -> dict[str, Any]:
        emit_progress("validating")
        if isinstance(params, PipelineSaveParams):
            pipeline = self._resolved(params.pipeline, params.input_kind)
            path = save_pipeline(pipeline, job.workspace / params.filename)
            return {"pipeline": str(path)}
        if isinstance(params, PipelineLoadParams):
            source = self._read_path(params.path)
            if source.stat().st_size > MAX_MESSAGE_BYTES:
                raise WorkerError("RESOURCE_LIMIT", "Pipeline YAML exceeds the worker byte limit.")
            pipeline = self._resolved(load_pipeline(source), params.input_kind)
            path = save_pipeline(pipeline, job.workspace / "pipeline.yaml")
            return {"pipeline": str(path)}
        if isinstance(params, PathParams):
            entries = list_hdus(self._read_path(params.path))
            path = write_json(job.workspace / "hdus.json", {"hdus": entries})
            return {"hdus": str(path), "count": len(entries)}
        if isinstance(params, TestProviderParams):
            emit_progress("provider-testing")
            if params.config.provider == "rules":
                return {"provider": "rules", "tested": True, "network_call": False}
            # One existing planner call, no image load, tools, or scientific output.
            test_metrics = DatasetMetrics(number_of_frames=0, session_counts={}, frames=[])
            try:
                self._planner(params.config).create_plan(
                    "Connection test. Return an empty pipeline and no alternatives.",
                    test_metrics,
                    [],
                )
            except Exception:
                checkpoint()
                raise WorkerError(
                    "PROVIDER_ERROR",
                    "Provider test failed; check key, model access, timeout, and connection.",
                ) from None
            return {
                "provider": params.config.provider,
                "model": _redact(
                    params.config.model,
                    params.config.api_key.get_secret_value()
                    if params.config.api_key is not None
                    else None,
                ),
                "tested": True,
                "network_call": True,
            }
        assert isinstance(
            params, ExecuteParams | InspectParams | DatasetParams | PlanParams | PreviewParams
        )
        identity = self._identity(params.input, hash_contents=method != "session.inspect")
        if isinstance(params, ExecuteParams):
            pipeline = self._resolved(params.pipeline, params.input.kind)
            if params.plan_id is not None:
                saved = self.plans.get(params.plan_id)
                if saved is None or monotonic() - saved[0] > 300:
                    raise WorkerError(
                        "PLAN_EXPIRED", "Reviewed plan identity expired; inspect and plan again."
                    )
                if saved[1] != identity:
                    raise WorkerError(
                        "INPUT_CHANGED",
                        "Input changed after planning; inspect and review a fresh plan.",
                    )
            # An explicit session parameter would rediscover deselected files.
            for step in pipeline.steps:
                if step.tool == "build_masters" and step.params.get("session") is not None:
                    raise WorkerError(
                        "INVALID_PARAMS",
                        "Worker uses selected frames; omit the session directory override.",
                    )
            output = job.workspace / params.output_name

            # Validate real Path-typed tool parameters through their backend models.
            def check_paths(value: Any) -> None:
                if isinstance(value, Path):
                    self._read_path(value)
                elif isinstance(value, dict):
                    for item in value.values():
                        check_paths(item)
                elif isinstance(value, list | tuple):
                    for item in value:
                        check_paths(item)

            for _, resolved_params in self.executor.validate(
                pipeline, input_kind=params.input.kind
            ):
                check_paths(resolved_params.model_dump())
            if isinstance(params.input, ImageInput):
                emit_progress("loading")
                image = load_image(self._read_path(params.input.path), hdu=params.input.hdu)
                result = self.executor.execute(image, pipeline)
                emit_progress("saving")
                save_result(result, output)
            else:
                dataset = self._dataset(params.input)
                execute_dataset_pipeline(
                    dataset,
                    pipeline,
                    self.executor.validate(pipeline, input_kind="dataset"),
                    output,
                )
            if self._identity(params.input) != identity:
                raise WorkerError(
                    "INPUT_CHANGED", "Input changed during processing; review it and retry."
                )
            path = write_json(job.workspace / "input-identity.json", identity)
            return {
                "output": str(output),
                "input_identity": str(path),
                "workspace": str(job.workspace),
            }
        if isinstance(params, DatasetParams):
            dataset = self._dataset(params.input)
            if method == "session.inspect":
                session = inspect_session_frames(dataset.frames, purposes=dataset.purpose_overrides)
                payload = session.model_dump(mode="json")
                payload["resources"] = {
                    "estimated_stack_spool_bytes": sum(
                        frame.width * frame.height * (3 if frame.layout.value == "rgb" else 1) * 4
                        for frame in session.frames
                    ),
                    "memory_mb": job.context.memory_mb,
                    "scratch_bytes": job.context.scratch_bytes,
                    "memory_is_rss_cap": False,
                }
            else:
                analyze_frames(dataset, params.detection)
                payload = dataset.reports["analysis"]
            path = write_json(job.workspace / "analysis.json", payload)
            if self._identity(params.input, hash_contents=method != "session.inspect") != identity:
                raise WorkerError(
                    "INPUT_CHANGED", "Input changed during analysis; inspect it again."
                )
            identity_path = write_json(job.workspace / "input-identity.json", identity)
            return {"analysis": str(path), "input_identity": str(identity_path)}
        if isinstance(params, InspectParams):
            emit_progress("loading")
            image = load_image(self._read_path(params.input.path), hdu=params.input.hdu)
            inspection: dict[str, Any] = {
                "metrics": inspect_image(image).model_dump(mode="json"),
                "input_hdu": image.input_hdu,
                "input_identity": identity,
            }
            if params.quality:
                from astroagent.analysis.frame_quality import measure_frame

                quality, catalog = measure_frame(image, detection=params.detection)
                inspection.update(
                    quality=quality.model_dump(mode="json"), catalog=catalog.model_dump(mode="json")
                )
            path = write_json(job.workspace / "inspection.json", inspection)
            if self._identity(params.input) != identity:
                raise WorkerError(
                    "INPUT_CHANGED", "Input changed during inspection; inspect it again."
                )
            return {"inspection": str(path), "input_hdu": image.input_hdu}
        if isinstance(params, PreviewParams):
            if any(
                description.input_kind == "dataset"
                for description in self.executor.registry.describe()
                if description.name in {step.tool for step in params.pipeline.steps}
            ):
                raise WorkerError(
                    "UNSUPPORTED_CAPABILITY",
                    "Registration/stacking preview requires a selected dataset pipeline execution.",
                )
            image = load_image(self._read_path(params.input.path), hdu=params.input.hdu)
            preview = create_preview(
                image,
                params.pipeline,
                job.workspace / "preview.png",
                options=PreviewOptions(roi=params.roi, scale=params.scale),
                executor=self.executor,
                input_identity=identity,
            )
            with self.jobs.condition:
                if self.previews.get(params.channel) != (params.generation, job.id):
                    raise ExecutionCancelled("Preview was superseded.")
            if self._identity(params.input) != identity:
                raise WorkerError(
                    "INPUT_CHANGED", "Preview input changed; request a fresh preview."
                )
            return {
                "display": str(preview.display),
                "provenance": str(preview.provenance),
                "approximate": preview.approximate,
                "generation": params.generation,
                "channel": params.channel,
            }
        assert isinstance(params, PlanParams)
        key = None if params.config.api_key is None else params.config.api_key.get_secret_value()
        planner = self._planner(params.config)
        metrics: ImageMetrics | DatasetMetrics
        try:
            if isinstance(params.input, ImageInput):
                prepared = AgentExecutor(planner, self.executor).prepare(
                    self._read_path(params.input.path), params.request, hdu=params.input.hdu
                )
                plan, metrics = prepared.plan, prepared.metrics
            else:
                dataset = self._dataset(params.input)
                selected_plan = AgentExecutor(planner, self.executor).prepare_dataset(
                    dataset, params.request
                )
                plan, metrics = selected_plan.plan, selected_plan.metrics
            emit_progress("validating")
            plan = PlanResult(
                pipeline=self._resolved(plan.pipeline, params.input.kind),
                reasoning=plan.reasoning,
                alternatives=[
                    self._resolved(candidate, params.input.kind) for candidate in plan.alternatives
                ],
            )
        except PipelineValidationError as exc:
            raise WorkerError(
                "INVALID_PARAMS",
                "Planned pipeline parameters are invalid.",
                {"locations": _redact(exc.locations, key)},
            ) from None
        except ExecutionCancelled:
            raise
        except ProviderError:
            raise WorkerError(
                "PROVIDER_ERROR", "Planning provider failed; check configuration and connection."
            ) from None
        except AstroError:
            raise
        except Exception:
            if params.config.provider != "rules":
                raise WorkerError(
                    "PROVIDER_ERROR",
                    "Planning failed; check provider configuration and input eligibility.",
                ) from None
            raise
        if self._identity(params.input) != identity:
            raise WorkerError(
                "INPUT_CHANGED", "Input changed during planning; inspect and plan again."
            )
        for expired in [
            identifier for identifier, saved in self.plans.items() if monotonic() - saved[0] > 300
        ]:
            del self.plans[expired]
        plan_id = "plan-" + uuid4().hex
        if len(self.plans) >= 32:
            del self.plans[next(iter(self.plans))]
        self.plans[plan_id] = (monotonic(), identity)
        path = write_json(
            job.workspace / "plan.json",
            _redact(
                {
                    "plan_id": plan_id,
                    "input_identity": identity,
                    "plan": plan.model_dump(mode="json"),
                    "warnings": metrics.warnings,
                    "aia_version": __version__,
                },
                key,
            ),
        )
        return {"plan_id": plan_id, "plan": str(path), "expires_in_seconds": 300}
