"""Launch with the host's Python, independently of the checkout or working directory."""

import logging
import os
import sys
from typing import TextIO

from astroagent.worker.protocol import Transport
from astroagent.worker.server import Server


class DiagnosticSink:
    """Suppress untrusted library prints, which can include provider response bodies."""

    def write(self, text: str) -> int:
        """Discard text instead of retaining or echoing potential credentials."""
        return len(text)

    def flush(self) -> None:
        """Provide the file-like interface used by progress bars."""


def main() -> int:
    """Reserve original stdout for protocol and emit only static fatal diagnostics."""
    protocol_output = sys.stdout.buffer
    diagnostics: TextIO = sys.stderr
    original_stdout, original_stderr = sys.stdout, sys.stderr
    previous_disable = logging.root.manager.disable
    sink = DiagnosticSink()
    logging.disable(logging.CRITICAL)
    sys.stdout = sink
    sys.stderr = sink

    def fatal_transport() -> None:
        # A broken transport cannot reliably deliver terminal events. Exit even
        # when the failure occurred on the processing thread while stdin blocks.
        os._exit(1)

    try:
        Server(sys.stdin.buffer, Transport(protocol_output, fatal=fatal_transport)).run()
        return 0
    except BrokenPipeError:
        return 1
    except Exception:
        diagnostics.write("AIA worker stopped after a fatal transport or internal failure.\n")
        diagnostics.flush()
        return 1
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        logging.disable(previous_disable)


if __name__ == "__main__":
    raise SystemExit(main())
