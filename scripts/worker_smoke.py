"""Exercise an installed worker with the Python supplied by a desktop bundle."""

import argparse
import json
import subprocess
from pathlib import Path
from queue import Queue
from threading import Thread


class WorkerClient:
    """Small host-side smoke harness with timed reads and deterministic process cleanup."""

    def __init__(self, python: Path, directory: Path) -> None:
        """Start an installed module outside the repository using pipe transport."""
        self.process = subprocess.Popen(
            [str(python), "-m", "astroagent.worker"],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.messages: Queue[dict[str, object] | None] = Queue()
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.messages.put(json.loads(line))
        self.messages.put(None)

    def send(self, identifier: str, method: str, params: dict[str, object]) -> None:
        """Write one UTF-8 version-one request."""
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(
                {
                    "protocol_version": 1,
                    "type": "request",
                    "id": identifier,
                    "method": method,
                    "params": params,
                }
            ).encode()
            + b"\n"
        )
        self.process.stdin.flush()

    def receive(self) -> dict[str, object]:
        """Fail instead of hanging the build if a process crashes or stops responding."""
        message = self.messages.get(timeout=60)
        assert message is not None, "Worker exited unexpectedly."
        assert message["protocol_version"] == 1
        return message

    def close(self) -> None:
        """Kill only this owned child when a smoke assertion interrupted normal shutdown."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=10)


def smoke_worker(python: Path, directory: Path) -> None:
    """Verify handshake, real processing, clean stdout, and death/restart isolation."""
    configuration = {"read_roots": [str(directory)], "workspace_root": str(directory / "jobs")}
    client = WorkerClient(python, directory)
    try:
        client.send("hello", "worker.handshake", configuration)
        handshake = client.receive()
        assert handshake["id"] == "hello" and handshake["result"]["aia_version"]
        assert handshake["result"]["dependency_versions"]["numpy"]
        client.send(
            "run",
            "pipeline.execute",
            {
                "input": {"kind": "image", "path": str(directory / "input.fit")},
                "pipeline": {"steps": [{"tool": "normalize"}]},
            },
        )
        accepted = client.receive()
        assert accepted["id"] == "run" and "result" in accepted
        job_id = accepted["result"]["job_id"]
        while True:
            event = client.receive()
            assert event["job_id"] == job_id and event["request_id"] == "run"
            if event["event"] in {"job.completed", "job.failed", "job.cancelled"}:
                assert event["event"] == "job.completed", event
                output = Path(event["data"]["result"]["output"])
                assert output.is_file() and output.with_suffix(".pipeline.yaml").exists()
                assert Path(event["data"]["result"]["manifest"]).is_file()
                break
        client.send("bye", "worker.shutdown", {})
        assert client.receive()["id"] == "bye"
        assert client.process.wait(timeout=15) == 0
        assert client.process.stderr.read() == b""
    finally:
        client.close()

    interrupted = WorkerClient(python, directory)
    try:
        interrupted.send("hello", "worker.handshake", configuration)
        interrupted.receive()
        interrupted.send(
            "interrupted",
            "pipeline.execute",
            {
                "input": {"kind": "image", "path": str(directory / "input.fit")},
                "pipeline": {"steps": [{"tool": "denoise", "params": {"sigma": 10}}] * 100},
            },
        )
        abandoned = interrupted.receive()["result"]["job_id"]
        # A started event precedes input loading; kill before any terminal outcome.
        assert interrupted.receive()["event"] == "job.started"
        interrupted.close()
        assert not (directory / "jobs" / abandoned / "complete.json").exists()
    finally:
        interrupted.close()

    restarted = WorkerClient(python, directory)
    try:
        restarted.send("hello", "worker.handshake", configuration)
        assert restarted.receive()["id"] == "hello"
        restarted.send("old", "job.status", {"job_id": abandoned})
        assert restarted.receive()["error"]["code"] == "JOB_NOT_FOUND"
        restarted.send("bye", "worker.shutdown", {})
        assert restarted.receive()["id"] == "bye"
        assert restarted.process.wait(timeout=15) == 0
    finally:
        restarted.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Writable absolute directory containing a synthetic input.fit.",
    )
    options = parser.parse_args()
    smoke_worker(options.python.resolve(), options.directory.resolve())
