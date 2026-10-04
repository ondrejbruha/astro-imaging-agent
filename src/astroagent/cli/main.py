import logging
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import ValidationError
from typer.core import TyperGroup

from astroagent import __version__
from astroagent.agent.autonomous import AgentOptions, AutonomousAgent
from astroagent.agent.providers import ProviderName, create_planner
from astroagent.analysis.background import analyze_background
from astroagent.analysis.stars import analyze_stars
from astroagent.analysis.statistics import inspect_image
from astroagent.errors import AstroError
from astroagent.io.images import load_image
from astroagent.pipeline.dataset_executor import DatasetPipelineResult
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import dump_pipeline, load_pipeline
from astroagent.tools.registry import default_registry

logger = logging.getLogger(__name__)


class ErrorHandlingGroup(TyperGroup):
    """Present expected CLI failures concisely, with tracebacks only at -vv."""

    def invoke(self, ctx: Any) -> Any:
        """Apply one error boundary around all command handlers."""
        try:
            return super().invoke(ctx)
        except (AstroError, ValidationError, ValueError, OSError) as exc:
            if isinstance(ctx.obj, dict) and ctx.obj.get("debug"):
                logger.exception("Command failed")
            typer.echo(f"Error: {exc}", err=True)
            ctx.exit(1)


app = typer.Typer(
    name="aia",
    cls=ErrorHandlingGroup,
    no_args_is_help=True,
    help="Astronomical image processing with deterministic, replayable pipelines.",
    pretty_exceptions_enable=False,
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Use -v for steps, -vv for debug tracebacks."
    ),
    show_version: bool = typer.Option(
        False, "--version", is_eager=True, help="Show the installed package version."
    ),
) -> None:
    """Configure logging once; verbosity options precede the subcommand."""
    root_logger = logging.getLogger()
    previous_handlers, previous_level = list(root_logger.handlers), root_logger.level
    for handler in previous_handlers:
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    def restore_logging() -> None:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)

    ctx.call_on_close(restore_logging)
    ctx.obj = {"debug": verbose >= 2}
    for name in ("openai", "anthropic", "google.genai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if show_version:
        typer.echo(f"astro-imaging-agent {__version__}")
        raise typer.Exit()


def _output_path(source: Path, output: Path | None) -> Path:
    return output if output is not None else source.with_name(f"{source.stem}.processed.fit")


def _process(
    source: Path, output: Path | None, tool: str, params: dict[str, Any], overwrite: bool
) -> None:
    destination = _output_path(source, output)
    pipeline = PipelineDefinition(steps=[PipelineStep(tool=tool, params=params)])
    result = PipelineExecutor().run(pipeline, source, destination, overwrite=overwrite)
    assert not isinstance(result, DatasetPipelineResult)
    typer.echo(f"Saved {destination}")
    for warning in result.report.warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command("inspect")
def inspect_command(
    image: Path = typer.Argument(
        ..., help="Input mono or RGB image (FITS, TIFF, PNG, JPEG, WebP, BMP)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON suitable for scripts."),
) -> None:
    """Inspect dimensions, finite statistics, saturation, and FITS metadata."""
    metrics = inspect_image(load_image(image))
    if as_json:
        typer.echo(metrics.model_dump_json(indent=2))
        return
    for name, value in metrics.model_dump().items():
        if value is not None and name not in {"warnings", "fits_metadata"}:
            typer.echo(f"{name}: {value}")
    typer.echo(f"FITS metadata: {metrics.fits_metadata}")
    for warning in metrics.warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command("analyze-background")
def background_command(image: Path) -> None:
    """Emit robust background level, residual noise, and gradient as JSON."""
    typer.echo(analyze_background(load_image(image)).model_dump_json(indent=2))


@app.command("analyze-stars")
def stars_command(
    image: Path,
    threshold_sigma: float = typer.Option(
        5.0, min=0.01, help="Detection threshold in sky-noise sigma."
    ),
    fwhm: float = typer.Option(3.0, min=0.1, help="Expected FWHM in pixels."),
) -> None:
    """Emit approximate photutils star counts and moment-based shapes as JSON."""
    typer.echo(
        analyze_stars(
            load_image(image), threshold_sigma=threshold_sigma, fwhm=fwhm
        ).model_dump_json(indent=2)
    )


@app.command("normalize")
def normalize_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    lower: float = typer.Option(0.0),
    upper: float = typer.Option(1.0),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
) -> None:
    """Normalize globally into 0..1 (or a specified subinterval)."""
    _process(image, output, "normalize", {"lower": lower, "upper": upper}, overwrite)


@app.command("stretch")
def stretch_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    method: Literal["linear", "asinh"] = typer.Option("asinh"),
    strength: float = typer.Option(0.6),
    black_point: float | None = typer.Option(None),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
) -> None:
    """Apply a bounded linear or asinh stretch, preserving the brightest endpoint."""
    _process(
        image,
        output,
        "stretch",
        {"method": method, "strength": strength, "black_point": black_point},
        overwrite,
    )


@app.command("denoise")
def denoise_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    sigma: float = typer.Option(0.8),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
) -> None:
    """Apply deterministic spatial Gaussian denoising."""
    _process(image, output, "denoise", {"method": "gaussian", "sigma": sigma}, overwrite)


@app.command("local-contrast")
def local_contrast_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    radius: float = typer.Option(12),
    amount: float = typer.Option(0.3),
    protect_highlights: bool = typer.Option(True),
    overwrite: bool = typer.Option(False),
) -> None:
    """Enhance local luminance contrast after normalization or stretching."""
    _process(
        image,
        output,
        "local_contrast",
        {"radius": radius, "amount": amount, "protect_highlights": protect_highlights},
        overwrite,
    )


@app.command("sharpen")
def sharpen_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    radius: float = typer.Option(1),
    amount: float = typer.Option(0.4),
    threshold: float = typer.Option(0.01),
    protect_highlights: bool = typer.Option(True),
    overwrite: bool = typer.Option(False),
) -> None:
    """Apply thresholded luminance unsharp masking to normalized samples."""
    _process(
        image,
        output,
        "sharpen",
        {
            "radius": radius,
            "amount": amount,
            "threshold": threshold,
            "protect_highlights": protect_highlights,
        },
        overwrite,
    )


@app.command("color-adjust")
def color_adjust_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    red_gain: float = typer.Option(1),
    green_gain: float = typer.Option(1),
    blue_gain: float = typer.Option(1),
    saturation: float = typer.Option(1),
    hue_shift: float = typer.Option(0),
    target_hue: float | None = typer.Option(None),
    hue_width: float = typer.Option(40),
    overwrite: bool = typer.Option(False),
) -> None:
    """Adjust RGB gains, saturation and hue globally or in a selected hue band."""
    _process(
        image,
        output,
        "color_adjust",
        {
            "red_gain": red_gain,
            "green_gain": green_gain,
            "blue_gain": blue_gain,
            "saturation": saturation,
            "hue_shift": hue_shift,
            "target_hue": target_hue,
            "hue_width": hue_width,
        },
        overwrite,
    )


@app.command("background-extract")
def background_extract_command(
    image: Path,
    output: Path | None = typer.Argument(None, help="Output image; extension selects format."),
    grid_size: int = typer.Option(8),
    polynomial_degree: int = typer.Option(2),
    sigma_clipping_threshold: float = typer.Option(3.0),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
) -> None:
    """Subtract a sigma-clipped tiled polynomial sky model."""
    _process(
        image,
        output,
        "background_extract",
        {
            "grid_size": grid_size,
            "polynomial_degree": polynomial_degree,
            "sigma_clipping_threshold": sigma_clipping_threshold,
        },
        overwrite,
    )


@app.command("run")
def run_command(
    pipeline: Path,
    input_path: Path = typer.Option(..., "--input", help="Input image."),
    output: Path = typer.Option(..., "--output", help="Output image; extension selects format."),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
) -> None:
    """Execute a saved YAML pipeline without any planner or LLM."""
    result = PipelineExecutor().run(
        load_pipeline(pipeline), input_path, output, overwrite=overwrite
    )
    typer.echo(f"Saved {output}")
    warnings = (
        result.report.get("warnings", [])
        if isinstance(result, DatasetPipelineResult)
        else result.report.warnings
    )
    for warning in warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command("tools")
def tools_command() -> None:
    """List names, descriptions, and parameter JSON schemas."""
    import json

    typer.echo(
        json.dumps(
            [tool.model_dump(mode="json") for tool in default_registry().describe()], indent=2
        )
    )


@app.command("agent")
def agent_command(
    image: Path,
    request: str,
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Default: <input>.processed.fit."
    ),
    dry_run: bool = typer.Option(False, help="Analyze and display the plan without writing files."),
    explain: bool = typer.Option(
        False, help="Display short user-facing reasons for selected steps."
    ),
    overwrite: bool = typer.Option(False, help="Replace existing output artifacts."),
    provider: ProviderName = typer.Option(
        "rules", help="Planner: rules, openai, anthropic, or gemini."
    ),
    model: str | None = typer.Option(None, help="Required model ID for LLM providers."),
    timeout: float = typer.Option(60.0, min=0.1, help="LLM API timeout in seconds."),
    max_iterations: int = typer.Option(
        3, min=1, max=20, help="Maximum plan/execute/measure rounds."
    ),
    max_candidates: int = typer.Option(
        3, min=1, max=3, help="Maximum alternative pipelines per round."
    ),
) -> None:
    """Autonomously plan, execute, compare results and refine image or session pipelines."""
    agent = AutonomousAgent(planner=create_planner(provider, model, timeout=timeout))
    prepared = agent.prepare(image, request)
    typer.echo(dump_pipeline(prepared.plan.pipeline), nl=False)
    if explain:
        for reason in prepared.plan.reasoning:
            typer.echo(f"Reason: {reason}", err=True)
        if provider == "rules":
            typer.echo(
                "Rules use numerical metrics and limited editing keywords; "
                "LLM providers interpret free-form goals.",
                err=True,
            )
    for warning in prepared.metrics.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if dry_run:
        return
    destination = (
        (output or image.parent / f"{image.name}-master.fit")
        if image.is_dir()
        else _output_path(image, output)
    )
    result = agent.run(
        prepared,
        destination,
        options=AgentOptions(max_iterations=max_iterations, max_candidates=max_candidates),
        overwrite=overwrite,
    )
    typer.echo(f"Saved {destination}")
    for warning in result.report["warnings"]:
        typer.echo(f"Warning: {warning}", err=True)


from astroagent.cli.datasets import attach_dataset_commands  # noqa: E402

attach_dataset_commands(app)
