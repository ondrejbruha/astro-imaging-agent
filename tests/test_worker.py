import io
import json
import subprocess
import sys
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from astropy.io import fits
from synthetic import make_session, star_image

from astroagent.execution import checkpoint
from astroagent.io.fits import load_fits, save_fits
from astroagent.pipeline.serialization import load_pipeline
from astroagent.worker.jobs import JobManager
from astroagent.worker.protocol import MAX_MESSAGE_BYTES, Transport, WorkerError
from astroagent.worker.server import Server


def envelope(identifier, method, params=None, version=1):
    return {
        "protocol_version": version,
        "type": "request",
        "id": identifier,
        "method": method,
        "params": params or {},
    }


def messages(transport):
    with transport.lock:
        return [json.loads(line) for line in transport.output.getvalue().splitlines()]


def wait_job(manager, job_id):
    with manager.condition:
        assert manager.condition.wait_for(
            lambda: manager.jobs[job_id].state in {"COMPLETED", "FAILED", "CANCELLED"}, timeout=20
        )
    return manager.get(job_id)


@pytest.fixture
def server(tmp_path):
    transport = Transport(io.BytesIO())
    server = Server(io.BytesIO(), transport)
    server.dispatch(
        json.dumps(
            envelope(
                "hello",
                "worker.handshake",
                {
                    "read_roots": [str(tmp_path)],
                    "workspace_root": str(tmp_path / "work"),
                },
            )
        ).encode()
    )
    yield server
    server.jobs.shutdown()
    server.jobs.join()


def dispatch(server, identifier, method, params=None):
    server.dispatch(json.dumps(envelope(identifier, method, params)).encode())
    return next(
        message
        for message in reversed(messages(server.transport))
        if message.get("id") == identifier
    )


def run_job(server, identifier, method, params):
    accepted = dispatch(server, identifier, method, params)
    assert "result" in accepted, accepted
    return wait_job(server.jobs, accepted["result"]["job_id"])


def test_strict_transport_validation_and_catalog(server):
    assert messages(server.transport)[0]["result"]["protocol_version"] == 1
    for i, (line, code) in enumerate(
        [
            (b"{broken", "INVALID_JSON"),
            (b'{"value":NaN}', "INVALID_JSON"),
            (b'{"x":1,"x":2}', "INVALID_JSON"),
            (b'{"x":1e999}', "INVALID_JSON"),
            (b"\xff", "INVALID_JSON"),
            (
                json.dumps(envelope("old", "worker.handshake", version=2)).encode(),
                "UNSUPPORTED_VERSION",
            ),
            (
                json.dumps({**envelope("extra", "tools.list"), "secret": "fake-key"}).encode(),
                "INVALID_REQUEST",
            ),
        ]
    ):
        server.dispatch(line)
        assert messages(server.transport)[-1]["error"]["code"] == code, i
    assert dispatch(server, "hello", "tools.list")["error"]["code"] == "DUPLICATE_ID"
    assert dispatch(server, "unknown", "module.import")["error"]["code"] == "METHOD_NOT_FOUND"
    assert (
        dispatch(server, "bad", "tools.list", {"key": "fake-key"})["error"]["code"]
        == "INVALID_PARAMS"
    )
    catalog = dispatch(server, "catalog", "tools.list")["result"]["tools"]
    register = next(tool for tool in catalog if tool["name"] == "register_frames")
    assert "$defs" in register["parameters"]
    assert "$ref" in register["parameters"]["properties"]["detection"]
    assert register["preview_strategies"] == []
    calibrate = next(tool for tool in catalog if tool["name"] == "calibrate_frames")
    assert "anyOf" in calibrate["parameters"]["properties"]["master_bias"]
    assert calibrate["ui_hints"]["master_bias"]["widget"] == "file"
    invalid = dispatch(
        server,
        "validate",
        "pipeline.validate",
        {"input_kind": "dataset", "pipeline": {"steps": [{"tool": "normalize"}]}},
    )
    assert invalid["error"]["details"]["locations"][0]["step_index"] == 1
    nested = dispatch(
        server,
        "nested",
        "pipeline.validate",
        {
            "input_kind": "dataset",
            "pipeline": {
                "steps": [{"tool": "register_frames", "params": {"detection": {"fwhm": -1}}}]
            },
        },
    )
    assert nested["error"]["details"]["locations"][0]["field"] == ["detection", "fwhm"]
    resolved = dispatch(
        server, "defaults", "pipeline.validate", {"pipeline": {"steps": [{"tool": "denoise"}]}}
    )
    assert resolved["result"]["pipeline"]["steps"][0]["params"]["sigma"] == 0.8
    assert b"fake-key" not in server.transport.output.getvalue()
    unknown_tool = dispatch(
        server,
        "unknown-tool",
        "pipeline.validate",
        {
            "pipeline": {"steps": [{"tool": "unregistered"}]},
        },
    )
    assert unknown_tool["error"]["code"] == "INVALID_PARAMS"
    assert unknown_tool["error"]["details"]["locations"][0]["field"] == ["tool"]
    gui_metadata = dispatch(
        server,
        "gui-metadata",
        "pipeline.validate",
        {
            "pipeline": {"steps": [{"tool": "normalize", "enabled": True}]},
        },
    )
    assert gui_metadata["error"]["details"]["locations"][0]["step_index"] == 1


def test_real_image_jobs_selected_hdu_preview_and_plan(server, tmp_path):
    path = tmp_path / "input.fit"
    image = star_image()
    fits.HDUList([fits.PrimaryHDU(np.ones(image.data.shape)), fits.ImageHDU(image.data)]).writeto(
        path
    )
    reference = {"kind": "image", "path": str(path), "hdu": 1}
    before = path.read_bytes()
    listed = run_job(server, "hdus", "image.hdus", {"path": str(path)})
    assert json.loads(Path(listed["result"]["hdus"]).read_text())["hdus"][1]["supported"]
    inspected = run_job(server, "inspect", "image.inspect", {"input": reference, "quality": True})
    report = json.loads(Path(inspected["result"]["inspection"]).read_text())
    assert report["input_hdu"] == 1 and report["quality"]["star_count"] >= 20
    planned = run_job(
        server, "plan", "agent.plan", {"input": reference, "request": "normalize image"}
    )
    assert planned["state"] == "COMPLETED", planned
    plan = json.loads(Path(planned["result"]["plan"]).read_text())
    assert not list(Path(planned["workspace"]).glob("*.fit"))
    pipeline = {"steps": [{"tool": "normalize"}]}
    executed = run_job(
        server,
        "execute",
        "pipeline.execute",
        {
            "input": reference,
            "pipeline": pipeline,
            "plan_id": plan["plan_id"],
        },
    )
    assert executed["state"] == "COMPLETED", executed
    output = Path(executed["result"]["output"])
    assert load_fits(output).data.max() == 1
    assert json.loads(output.with_suffix(".processing.json").read_text())["input_hdu"] == 1
    preview = run_job(
        server,
        "preview",
        "image.preview",
        {
            "input": reference,
            "pipeline": pipeline,
            "scale": 0.5,
            "generation": 1,
        },
    )
    assert preview["state"] == "COMPLETED", preview
    assert preview["result"]["approximate"] and Path(preview["result"]["display"]).exists()
    stale = dispatch(
        server,
        "stale",
        "image.preview",
        {"input": reference, "pipeline": pipeline, "generation": 1},
    )
    assert stale["error"]["code"] == "STALE_PREVIEW"
    unsupported = run_job(
        server,
        "unsupported",
        "image.preview",
        {
            "input": reference,
            "pipeline": {"steps": [{"tool": "register_frames"}]},
            "generation": 2,
        },
    )
    assert unsupported["error"]["code"] == "UNSUPPORTED_CAPABILITY"
    assert path.read_bytes() == before
    fits.setval(path, "OBJECT", value="Changed")
    changed = run_job(
        server,
        "changed",
        "pipeline.execute",
        {
            "input": reference,
            "pipeline": pipeline,
            "plan_id": plan["plan_id"],
        },
    )
    assert changed["error"]["code"] == "INPUT_CHANGED"
    assert not (Path(changed["workspace"]) / "complete.json").exists()
    sent = messages(server.transport)
    acceptance = next(i for i, msg in enumerate(sent) if msg.get("id") == "execute")
    events = [(i, msg) for i, msg in enumerate(sent) if msg.get("job_id") == executed["job_id"]]
    assert all(i > acceptance for i, _ in events)
    assert (
        sum(msg["event"] in {"job.completed", "job.failed", "job.cancelled"} for _, msg in events)
        == 1
    )


def test_pipeline_save_load_and_path_policy(server, tmp_path):
    saved = run_job(
        server, "save", "pipeline.save", {"pipeline": {"steps": [{"tool": "normalize"}]}}
    )
    assert saved["state"] == "COMPLETED"
    loaded = run_job(server, "load", "pipeline.load", {"path": saved["result"]["pipeline"]})
    assert load_pipeline(loaded["result"]["pipeline"]).steps[0].params == {"lower": 0, "upper": 1}
    path = tmp_path / "image.fit"
    save_fits(star_image(), path)
    invalid = run_job(
        server, "relative", "image.inspect", {"input": {"kind": "image", "path": "image.fit"}}
    )
    assert invalid["error"]["code"] == "PATH_DENIED"
    reference = {"kind": "dataset", "frames": [str(path), str(path)]}
    duplicate = run_job(server, "duplicate", "dataset.analyze", {"input": reference})
    assert duplicate["error"]["code"] == "INVALID_PARAMS"
    conflict = dispatch(server, "conflict", "worker.handshake", {})
    assert conflict["error"]["code"] == "CONFIG_CONFLICT"


@pytest.mark.parametrize("cfa", [False, True])
def test_explicit_dataset_full_pipeline_and_replay(server, tmp_path, cfa):
    session = tmp_path / "session"
    make_session(session, cfa=cfa)
    selected = sorted(session.rglob("*.fit"))
    excluded = selected.pop(
        next(i for i, path in enumerate(selected) if path.name == "light_2.fit")
    )
    original = {path: path.read_bytes() for path in [*selected, excluded]}
    reference = {
        "kind": "dataset",
        "frames": list(map(str, selected)),
        "reference": str(session / "lights" / "light_1.fit"),
    }
    inspected = run_job(server, "session", "session.inspect", {"input": reference})
    payload = json.loads(Path(inspected["result"]["analysis"]).read_text())
    assert len(payload["frames"]) == len(selected)
    assert payload["resources"]["estimated_stack_spool_bytes"] > 0
    planned = run_job(
        server, "plan-dataset", "agent.plan", {"input": reference, "request": "create master"}
    )
    assert planned["state"] == "COMPLETED", planned
    steps = [{"tool": "build_masters"}, {"tool": "calibrate_frames"}]
    if cfa:
        steps.append({"tool": "debayer_frames"})
    steps.extend(
        [{"tool": "register_frames"}, {"tool": "stack_frames", "params": {"method": "mean"}}]
    )
    completed = run_job(
        server,
        "dataset-execute",
        "pipeline.execute",
        {"input": reference, "pipeline": {"steps": steps}},
    )
    assert completed["state"] == "COMPLETED", completed
    output = Path(completed["result"]["output"])
    master = load_fits(output)
    assert master.header["NCOMBINE"] == 2
    assert master.channels == (3 if cfa else 1)
    report = json.loads(output.with_suffix(".processing.json").read_text())
    assert report["used_frames"] == 2
    assert "light_1" in report["reference"]
    assert str(excluded) not in json.dumps(report)
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert output.with_suffix(".pipeline.yaml").exists()
    assert not (Path(completed["workspace"]) / "scratch").exists()


def test_provider_configuration_and_sensitive_failures(server, tmp_path, monkeypatch):
    from astroagent.worker import server as module

    key = "fake-sensitive-api-key"
    calls = []

    class FakePlanner:
        def create_plan(self, *args, **kwargs):
            calls.append(True)
            raise RuntimeError(key + " private SDK response body")

    def fake_factory(provider, model, **kwargs):
        assert kwargs["api_key"] == key
        assert kwargs["timeout"] == 1
        return FakePlanner()

    monkeypatch.setattr(module, "create_planner", fake_factory)
    config = {"provider": "openai", "model": "fake-model", "api_key": key, "timeout": 1}
    result = run_job(server, "test", "llm.test", {"config": config})
    assert result["error"]["code"] == "PROVIDER_ERROR"
    path = save_fits(star_image(), tmp_path / "image.fit")
    result = run_job(
        server,
        "provider-plan",
        "agent.plan",
        {
            "input": {"kind": "image", "path": str(path)},
            "request": "process",
            "config": config,
        },
    )
    assert result["error"]["code"] == "PROVIDER_ERROR"
    assert calls == [True, True]
    assert key not in server.transport.output.getvalue().decode()
    for file in (tmp_path / "work").rglob("*.json"):
        assert key not in file.read_text()
    bad = dispatch(
        server, "missing-model", "llm.test", {"config": {"provider": "openai", "api_key": key}}
    )
    assert bad["error"]["code"] == "INVALID_PARAMS"


def test_queue_control_cancellation_and_retention(tmp_path):
    transport = Transport(io.BytesIO())
    manager = JobManager(transport, queue_limit=1, retention=1)
    running, release = Event(), Event()
    calls = []

    def blocked(job):
        running.set()
        assert release.wait(10)
        checkpoint()
        calls.append(job.id)
        return {"value": "done"}

    try:
        first = manager.submit("first", "blocked", tmp_path, blocked)
        assert running.wait(10)
        second = manager.submit("second", "queued", tmp_path, blocked)
        with pytest.raises(WorkerError, match="queue is full"):
            manager.submit("third", "excess", tmp_path, blocked)
        assert manager.get(first)["state"] == "RUNNING"
        assert manager.cancel(second)["accepted"]
        assert manager.cancel(second)["accepted"]
        assert manager.cancel(first)["accepted"]
        release.set()
        wait_job(manager, second)
        assert manager.get(second)["state"] == "CANCELLED"
        assert not manager.cancel(second)["accepted"]
        assert not calls
        assert not (tmp_path / second).exists()
        assert len(manager.jobs) == 1
        terminal = [msg for msg in messages(transport) if msg.get("event") == "job.cancelled"]
        assert len(terminal) == 2
        with pytest.raises(WorkerError, match="expired"):
            manager.get(first)
    finally:
        release.set()
        manager.shutdown()
        manager.join()


def test_finalization_cancel_race_and_write_failure(tmp_path, monkeypatch):
    from astroagent.worker import jobs as module

    write = module.write_json
    sealed, release = Event(), Event()

    def paused(path, payload, **kwargs):
        if path.name == "complete.json":
            sealed.set()
            assert release.wait(10)
        return write(path, payload, **kwargs)

    monkeypatch.setattr(module, "write_json", paused)
    transport = Transport(io.BytesIO())
    manager = JobManager(transport)
    try:
        job_id = manager.submit("seal", "simple", tmp_path, lambda job: {"saved": True})
        assert sealed.wait(10)
        assert manager.cancel(job_id) == {
            "job_id": job_id,
            "accepted": False,
            "state": "RUNNING",
            "cancellation_requested": False,
            "finalizing": True,
        }
        release.set()
        assert wait_job(manager, job_id)["state"] == "COMPLETED"

        def fail(path, payload, **kwargs):
            if path.name == "complete.json":
                raise OSError("disk full secret must not leak")
            return write(path, payload, **kwargs)

        monkeypatch.setattr(module, "write_json", fail)
        failed_id = manager.submit("fail", "simple", tmp_path, lambda job: {"saved": True})
        status = wait_job(manager, failed_id)
        assert status["state"] == "FAILED" and status["error"]["code"] == "IO_ERROR"
        assert not (Path(status["workspace"]) / "complete.json").exists()
        assert b"secret" not in transport.output.getvalue()
    finally:
        release.set()
        manager.shutdown()
        manager.join()


def test_eof_shutdown_and_message_limits(tmp_path):
    request = json.dumps(envelope("hello", "worker.handshake")).encode()
    stream = io.BytesIO(b"x" * (MAX_MESSAGE_BYTES + 2) + b"\n" + request + b"\r\n")
    transport = Transport(io.BytesIO())
    Server(stream, transport).run()
    replies = messages(transport)
    assert replies[0]["error"]["code"] == "MESSAGE_TOO_LARGE"
    assert replies[1]["id"] == "hello"
    assert replies[1]["result"]["aia_version"]


def test_module_subprocess_fragmented_utf8_batched_lines_and_clean_stdout(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-m", "astroagent.worker"],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        line = (
            json.dumps(envelope("žluťoučký", "worker.handshake"), ensure_ascii=False).encode()
            + b"\r\n"
        )
        process.stdin.write(line[:19])
        process.stdin.flush()
        process.stdin.write(
            line[19:] + json.dumps(envelope("shutdown", "worker.shutdown")).encode() + b"\n"
        )
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=15)
        replies = [json.loads(row) for row in stdout.splitlines()]
        assert [row["id"] for row in replies] == ["žluťoučký", "shutdown"]
        assert process.returncode == 0 and stderr == b""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_worker_shutdown_cancels_running_and_queued_work(server, tmp_path):
    running, release = Event(), Event()

    def blocked(job):
        running.set()
        assert release.wait(10)
        checkpoint()
        return {}

    first = server.jobs.submit("active", "test", tmp_path, blocked)
    assert running.wait(10)
    second = server.jobs.submit("queued", "test", tmp_path, blocked)
    response = dispatch(server, "shutdown", "worker.shutdown")
    assert response["result"]["shutdown_requested"]
    release.set()
    assert wait_job(server.jobs, second)["state"] == "CANCELLED"
    assert server.jobs.get(first)["state"] == "CANCELLED"
    with pytest.raises(WorkerError, match="shutting down"):
        server.jobs.submit("late", "test", tmp_path, blocked)


def test_partial_artifact_failure_and_source_immutability(server, tmp_path, monkeypatch):
    from astroagent.errors import PipelineError
    from astroagent.pipeline import executor

    path = save_fits(star_image(), tmp_path / "source.fit")
    before = path.read_bytes()

    def fail(*args, **kwargs):
        try:
            raise OSError("sensitive disk details")
        except OSError as exc:
            raise PipelineError("sensitive wrapper") from exc

    monkeypatch.setattr(executor, "save_pipeline", fail)
    result = run_job(
        server,
        "partial",
        "pipeline.execute",
        {
            "input": {"kind": "image", "path": str(path)},
            "pipeline": {"steps": []},
        },
    )
    assert result["state"] == "FAILED" and result["error"]["code"] == "IO_ERROR"
    assert (Path(result["workspace"]) / "output.fit").is_file()
    assert not (Path(result["workspace"]) / "complete.json").exists()
    assert result["result"] is None and path.read_bytes() == before
    assert b"sensitive" not in server.transport.output.getvalue()


def test_planning_alternatives_secrets_and_invalid_backend_parameters(
    server, tmp_path, monkeypatch
):
    from astroagent.agent.models import PlanResult
    from astroagent.pipeline.models import PipelineDefinition, PipelineStep

    key = "fake-plan-key-never-save"

    class FakePlanner:
        invalid = False

        def create_plan(self, *args, **kwargs):
            return PlanResult(
                pipeline=PipelineDefinition(steps=[PipelineStep(tool="normalize")]),
                alternatives=[
                    PipelineDefinition(
                        steps=[
                            PipelineStep(
                                tool="denoise", params={"sigma": -1 if self.invalid else 1}
                            )
                        ]
                    )
                ],
                reasoning=["Safe explanation " + key],
            )

    planner = FakePlanner()
    monkeypatch.setattr(server, "_planner", lambda config: planner)
    path = save_fits(star_image(), tmp_path / "source.fit")
    request = {
        "input": {"kind": "image", "path": str(path)},
        "request": "process",
        "config": {"provider": "openai", "model": "fake", "api_key": key},
    }
    completed = run_job(server, "valid-plan", "agent.plan", request)
    assert completed["state"] == "COMPLETED"
    text = Path(completed["result"]["plan"]).read_text()
    assert key not in text and "[redacted]" in text
    assert len(json.loads(text)["plan"]["alternatives"]) == 1
    assert not list(Path(completed["workspace"]).glob("*.fit"))
    planner.invalid = True
    invalid = run_job(server, "invalid-plan", "agent.plan", request)
    assert invalid["error"]["code"] == "INVALID_PARAMS"
    assert not list(Path(invalid["workspace"]).glob("*.fit"))


def test_backend_transition_validation_precedes_workspace_creation(server):
    accepted = dispatch(
        server,
        "bad-transition",
        "pipeline.execute",
        {
            "input": {"kind": "dataset", "frames": ["C:/not-read.fit"]},
            "pipeline": {"steps": [{"tool": "normalize"}]},
        },
    )
    assert accepted["error"]["code"] == "INVALID_PARAMS"
    assert not server.config.workspace_root.exists()


def test_preview_supersession_during_processing(server, tmp_path, monkeypatch):
    from astroagent.worker import server as module

    path = save_fits(star_image(), tmp_path / "source.fit")
    original = module.create_preview
    running, release = Event(), Event()

    def delayed(*args, **kwargs):
        running.set()
        assert release.wait(10)
        checkpoint()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "create_preview", delayed)
    params = {
        "input": {"kind": "image", "path": str(path)},
        "pipeline": {"steps": []},
        "generation": 1,
    }
    first = dispatch(server, "preview-1", "image.preview", params)["result"]["job_id"]
    assert running.wait(10)
    second = dispatch(server, "preview-2", "image.preview", {**params, "generation": 2})["result"][
        "job_id"
    ]
    release.set()
    assert wait_job(server.jobs, first)["state"] == "CANCELLED"
    assert wait_job(server.jobs, second)["state"] == "COMPLETED"
    assert not (Path(server.jobs.get(first)["workspace"]) / "complete.json").exists()


def test_public_protocol_schemas_output_limits_and_fatal_transport():
    from astroagent.worker.protocol import Response

    with pytest.raises(ValueError):
        Response.model_validate({"protocol_version": 1, "type": "response", "id": "x"})
    transport = Transport(io.BytesIO())
    with pytest.raises(WorkerError, match="message limit"):
        transport.response("large", {"text": "x" * MAX_MESSAGE_BYTES})
    called = []

    class Broken(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError("secret")

    with pytest.raises(BrokenPipeError):
        Transport(Broken(), fatal=lambda: called.append(True)).response("x", {})
    assert called == [True]


def test_main_suppresses_sensitive_prints_and_restores_streams(monkeypatch):
    from astroagent.worker import __main__ as module

    original_stdout, original_stderr, original_stdin = sys.stdout, sys.stderr, sys.stdin
    output = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    diagnostics = io.StringIO()

    class FakeServer:
        def __init__(self, *args):
            pass

        def run(self):
            print("fake-sdk-key")
            print("fake-sdk-key", file=sys.stderr)
            raise RuntimeError("fake-sdk-key")

    monkeypatch.setattr(module, "Server", FakeServer)
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostics)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    assert module.main() == 1
    assert output.buffer.getvalue() == b""
    assert "fake-sdk-key" not in diagnostics.getvalue()
    sys.stdout, sys.stderr, sys.stdin = original_stdout, original_stderr, original_stdin


def test_request_local_provider_key_overrides_environment_without_mutation(monkeypatch):
    import os
    from types import SimpleNamespace

    from astroagent.agent import providers

    captured = []

    def constructor(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setattr(
        providers, "import_module", lambda name: SimpleNamespace(OpenAI=constructor)
    )
    providers._sdk_client(
        "openai", "OpenAI", "openai", "OPENAI_API_KEY", api_key="request-key", timeout=1
    )
    assert captured[0]["api_key"] == "request-key"
    assert os.environ["OPENAI_API_KEY"] == "environment-key"


def test_provider_test_success_and_model_secret_redaction(server, monkeypatch):
    from astroagent.agent.models import PlanResult
    from astroagent.pipeline.models import PipelineDefinition

    class FakePlanner:
        def create_plan(self, request, metrics, tools):
            assert not tools and metrics.number_of_frames == 0
            return PlanResult(pipeline=PipelineDefinition())

    monkeypatch.setattr(server, "_planner", lambda config: FakePlanner())
    key = "model-and-key-same-value"
    result = run_job(
        server,
        "provider-success",
        "llm.test",
        {
            "config": {"provider": "openai", "model": key, "api_key": key},
        },
    )
    assert result["state"] == "COMPLETED" and result["result"]["model"] == "[redacted]"
    assert key not in server.transport.output.getvalue().decode()


def test_session_inspection_uses_headers_without_full_file_hashes(server, tmp_path, monkeypatch):
    from astroagent.worker import server as module

    path = save_fits(star_image(), tmp_path / "light.fit")

    def fail(*args, **kwargs):
        raise AssertionError("Header-only session inspection hashed pixel bytes.")

    monkeypatch.setattr(module.hashlib, "sha256", fail)
    result = run_job(
        server,
        "headers-only",
        "session.inspect",
        {
            "input": {"kind": "dataset", "frames": [str(path)]},
        },
    )
    assert result["state"] == "COMPLETED", result
    identity = json.loads(Path(result["result"]["input_identity"]).read_text())
    assert identity["files"][0]["sha256"] is None


def test_planning_preserves_cfa_and_layout_eligibility(server, tmp_path, monkeypatch):
    from astroagent.agent.models import PlanResult
    from astroagent.pipeline.models import PipelineDefinition, PipelineStep

    path = tmp_path / "raw.fit"
    fits.writeto(path, np.ones((32, 32)), header=fits.Header({"BAYERPAT": "RGGB"}))
    rejected = run_job(
        server,
        "raw-plan",
        "agent.plan",
        {
            "input": {"kind": "image", "path": str(path)},
            "request": "process",
        },
    )
    assert rejected["error"]["code"] == "INCOMPATIBLE_DATA"
    mono = save_fits(star_image(), tmp_path / "mono.fit")

    class FakePlanner:
        def create_plan(self, request, metrics, tools):
            assert "color_adjust" not in {tool.name for tool in tools}
            return PlanResult(
                pipeline=PipelineDefinition(steps=[PipelineStep(tool="color_adjust")])
            )

    monkeypatch.setattr(server, "_planner", lambda config: FakePlanner())
    rejected = run_job(
        server,
        "mono-color-plan",
        "agent.plan",
        {
            "input": {"kind": "image", "path": str(mono)},
            "request": "color edit",
        },
    )
    assert rejected["error"]["code"] == "INVALID_PARAMS"


def test_real_provider_error_class_maps_to_provider_failure(server, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from astroagent.agent.providers import OpenAIPlanner

    key = "sensitive-fake-sdk-response"

    def failure(**kwargs):
        raise RuntimeError(key)

    planner = OpenAIPlanner(
        "fake-model", client=SimpleNamespace(responses=SimpleNamespace(create=failure))
    )
    monkeypatch.setattr(server, "_planner", lambda config: planner)
    path = save_fits(star_image(), tmp_path / "source.fit")
    result = run_job(
        server,
        "sdk-error",
        "agent.plan",
        {
            "input": {"kind": "image", "path": str(path)},
            "request": "process",
            "config": {"provider": "openai", "model": "fake-model", "api_key": key},
        },
    )
    assert result["error"]["code"] == "PROVIDER_ERROR"
    assert key not in server.transport.output.getvalue().decode()
