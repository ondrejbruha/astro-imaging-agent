"""Check distribution metadata, license payloads, and release tag consistency."""

import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path

from packaging.version import Version


def main() -> None:
    """Fail before publication when an artifact is malformed or mislabeled."""
    distributions = sorted(Path("dist").glob("*"))
    wheels = [path for path in distributions if path.suffix == ".whl"]
    sources = [path for path in distributions if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution in dist/.")
    subprocess.run([sys.executable, "-m", "twine", "check", *map(str, distributions)], check=True)
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        assert metadata["Name"] == "astro-imaging-agent"
        entry_points = ConfigParser()
        entry_points.read_string(
            archive.read(next(name for name in names if name.endswith("entry_points.txt"))).decode()
        )
        assert entry_points["console_scripts"]["aia"] == "astroagent.cli.main:app"
        assert "astroagent/py.typed" in names
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/NOTICE") for name in names)
        assert not any(name.startswith("tests/") for name in names)
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        assert any(name.endswith("/README.md") for name in names)
        assert any("/tests/" in name for name in names)
        assert any(name.endswith("/examples/pipeline.yaml") for name in names)
        pyproject_name = next(name for name in names if name.endswith("/pyproject.toml"))
        pyproject_file = archive.extractfile(pyproject_name)
        assert pyproject_file is not None
        frozen = tomllib.loads(pyproject_file.read().decode("utf-8"))
        assert frozen["project"]["version"] == metadata["Version"]
    tag = os.environ.get("RELEASE_TAG")
    if tag is not None:
        if not tag.startswith("v") or Version(metadata["Version"]) != Version(tag[1:]):
            raise SystemExit(
                f"Artifact version {metadata['Version']} does not match release tag {tag}."
            )
    print(f"Verified wheel and sdist: {metadata['Name']} {metadata['Version']}")


if __name__ == "__main__":
    main()
