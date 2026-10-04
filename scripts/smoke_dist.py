"""Install the built wheel into an isolated environment and exercise its CLI."""

import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    """Verify the entry point outside the checkout using declared runtime dependencies."""
    wheel = next(Path("dist").glob("*.whl")).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        venv.create(root / "env", with_pip=True)
        python = root / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True)
        subprocess.run([str(python), "-m", "astroagent", "--help"], cwd=root, check=True)
        aia = python.with_name("aia.exe" if os.name == "nt" else "aia")
        subprocess.run([str(aia), "--version"], cwd=root, check=True)
        subprocess.run([str(aia), "tools"], cwd=root, check=True, stdout=subprocess.DEVNULL)
        fixture = """import numpy as np
from astropy.io import fits
y, x = np.indices((32, 32))
data = 100 + x + y + np.random.default_rng(7).normal(0, .1, (32, 32))
fits.writeto('input.fit', data, header=fits.Header({'OBJECT': 'Package smoke'}))
"""
        subprocess.run([str(python), "-c", fixture], cwd=root, check=True)
        inspect = subprocess.run(
            [str(aia), "inspect", "input.fit", "--json"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(inspect.stdout)["dimensions"] == [32, 32]
        subprocess.run(
            [str(aia), "agent", "input.fit", "process", "--dry-run", "--explain"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run([str(aia), "stretch", "input.fit", "out.png"], cwd=root, check=True)
        subprocess.run(
            [
                str(aia),
                "run",
                "out.pipeline.yaml",
                "--input",
                "input.fit",
                "--output",
                "replay.png",
            ],
            cwd=root,
            check=True,
        )
        first = json.loads((root / "out.processing.json").read_text())
        replay = json.loads((root / "replay.processing.json").read_text())
        assert first["output_sha256"] == replay["output_sha256"]
        source = next(wheel.parent.glob("*.tar.gz"))
        subprocess.run(
            [str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(source)],
            cwd=root,
            check=True,
        )
        subprocess.run([str(aia), "--version"], cwd=root, check=True)


if __name__ == "__main__":
    main()
