import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from astroagent.errors import PipelineError


def write_text(path: Path, content: str, *, overwrite: bool = False) -> Path:
    """Atomically write UTF-8 artifacts, refusing accidental overwrites."""
    temporary: Path | None = None
    try:
        if path.exists() and not overwrite:
            raise PipelineError(f"Output already exists: {path}. Use --overwrite to replace it.")
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        return path
    except OSError as exc:
        raise PipelineError(f"Cannot write '{path}': {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
