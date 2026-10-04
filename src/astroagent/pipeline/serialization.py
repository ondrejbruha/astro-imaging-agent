from pathlib import Path

import yaml
from pydantic import ValidationError

from astroagent.errors import PipelineError
from astroagent.io.artifacts import write_text
from astroagent.pipeline.models import PipelineDefinition, ProcessingReport


def dump_pipeline(pipeline: PipelineDefinition) -> str:
    """Serialize pipeline fields in stable, human-readable YAML order."""
    return yaml.safe_dump(pipeline.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def load_pipeline(path: Path | str) -> PipelineDefinition:
    """Safely parse YAML and reject invalid or unsupported pipeline definitions."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return PipelineDefinition.model_validate(document)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise PipelineError(f"Cannot load pipeline '{path}': {exc}") from exc


def save_pipeline(
    pipeline: PipelineDefinition, path: Path | str, *, overwrite: bool = False
) -> Path:
    """Save the exact resolved execution parameters for later replay."""
    return write_text(Path(path), dump_pipeline(pipeline), overwrite=overwrite)


def save_report(report: ProcessingReport, path: Path | str, *, overwrite: bool = False) -> Path:
    """Save a JSON report without NaN or unserializable NumPy objects."""
    return write_text(Path(path), report.model_dump_json(indent=2) + "\n", overwrite=overwrite)
