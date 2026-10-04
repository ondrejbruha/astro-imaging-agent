import json
import sys
from pathlib import Path
from typing import Any, Literal

import typer

from astroagent.calibration.debayer import debayer_image
from astroagent.calibration.masters import build_master, build_masters
from astroagent.calibration.models import CalibrationPlan, FrameType
from astroagent.calibration.session import discover_session, inspect_frame
from astroagent.calibration.workflow import calibrate_frames, plan_calibration
from astroagent.errors import PipelineError
from astroagent.io.datasets import discover_fits, write_json
from astroagent.io.export import export_image
from astroagent.io.fits import load_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.layout import CFAMetadata
from astroagent.registration.engine import RegistrationParams, analyze_frames, register_frames
from astroagent.registration.reference import select_reference
from astroagent.stacking.rejection import RejectionParams
from astroagent.stacking.stack import CombineParams, StackMethod
from astroagent.stacking.workflow import StackParams, save_stack, stack_frames
from astroagent.workflow import process_session


def _dataset(source: Path, *, recursive: bool = False) -> AstroDataset:
    return AstroDataset(discover_fits(source, recursive=recursive), source=source)


def _messages(report: dict[str, Any]) -> None:
    for warning in report.get("warnings", []):
        typer.echo(f"Warning: {warning}", err=True)
    for frame in report.get("frames", []):
        if frame.get("warning"):
            typer.echo(f"Warning: {frame['path']}: {frame['warning']}", err=True)
        for warning in frame.get("warnings", []):
            typer.echo(f"Warning: {frame['path']}: {warning}", err=True)


def attach_dataset_commands(app: typer.Typer) -> None:
    """Attach thin command handlers backed by independently usable deterministic services."""
    session_app = typer.Typer(help="Discover and inspect astronomical sessions.")
    master_app = typer.Typer(help="Build floating point master calibration frames.")
    app.add_typer(session_app, name="session")
    app.add_typer(master_app, name="master")

    @app.command("analyze-frames")
    def analyze(source: Path, as_json: bool = typer.Option(False, "--json")) -> None:
        """Analyze FITS frame quality and dataset-relative percentile scores."""
        dataset = analyze_frames(_dataset(source))
        if as_json:
            typer.echo(json.dumps(dataset.reports["analysis"], indent=2))
        else:
            typer.echo("Frame                         Stars   FWHM    Ecc     Noise    Quality")
            for q in dataset.qualities:
                fwhm = f"{q.median_fwhm:.2f}" if q.median_fwhm is not None else "n/a"
                ecc = f"{q.median_eccentricity:.2f}" if q.median_eccentricity is not None else "n/a"
                typer.echo(
                    f"{Path(q.path).name:28} {q.star_count:5d} {fwhm:>7} {ecc:>7} "
                    f"{q.background_sigma:9.3g} {q.quality_score:8.3f}"
                )

    @app.command("select-reference")
    def reference(source: Path, as_json: bool = typer.Option(False, "--json")) -> None:
        """Select the best eligible star-bearing frame, with deterministic tie breaking."""
        dataset = analyze_frames(_dataset(source))
        selected = select_reference(dataset.qualities)
        alternatives = sorted(
            (q for q in dataset.qualities if q.path != selected.path),
            key=lambda q: (-(q.quality_score or 0), q.path),
        )
        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "selected": selected.path,
                        "quality_score": selected.quality_score,
                        "alternatives": [
                            {"path": q.path, "quality_score": q.quality_score} for q in alternatives
                        ],
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(f"{selected.path} (quality {selected.quality_score:.3f})")

    @app.command("register")
    def register(
        source: Path,
        output: Path = typer.Option(..., "--output", "-o"),
        reference: str = typer.Option("auto"),
        interpolation: Literal["nearest", "bilinear", "bicubic"] = typer.Option("bicubic"),
        model: Literal["similarity", "affine"] = typer.Option("similarity"),
        min_matches: int = typer.Option(6, min=3),
        max_residual_rms: float = typer.Option(1, min=0.001),
        random_seed: int = typer.Option(0, min=0),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Match stars and register prepared mono/RGB frames onto reference pixels."""
        dataset = _dataset(source)
        params = RegistrationParams(
            reference=reference,
            interpolation=interpolation,
            model=model,
            min_matches=min_matches,
            min_inliers=min_matches,
            max_residual_rms=max_residual_rms,
            random_seed=random_seed,
        )
        with typer.progressbar(
            length=len(dataset.frames),
            label="Registering frames",
            show_pos=True,
            file=sys.stderr,
            hidden=not sys.stderr.isatty(),
        ) as bar:
            result = register_frames(
                dataset, output, params, overwrite=overwrite, progress=lambda: bar.update(1)
            )
        _messages(result.reports["registration"])
        typer.echo(f"Registered {len(result.frames)} frames to {output}")

    @app.command("stack")
    def stack(
        source: Path,
        output: Path = typer.Option(
            ...,
            "--output",
            "-o",
            help="FITS/TIFF scientific master or PNG/JPEG/WebP/BMP display export.",
        ),
        method: StackMethod = typer.Option("weighted-sigma-clipped"),
        register: bool = typer.Option(False, "--register"),
        reference: str = typer.Option("auto"),
        reject_worst: str = typer.Option("0%", help="Fraction, e.g. 0.1, or percentage, e.g. 10%."),
        min_quality: float | None = typer.Option(None),
        max_fwhm: float | None = typer.Option(None),
        max_eccentricity: float | None = typer.Option(None),
        max_background_sigma: float | None = typer.Option(None),
        max_saturation: float | None = typer.Option(None),
        max_registration_residual: float | None = typer.Option(None),
        sigma_low: float = typer.Option(3),
        sigma_high: float = typer.Option(3),
        max_iterations: int = typer.Option(5),
        normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
        tile_rows: int = typer.Option(128),
        random_seed: int = typer.Option(0, min=0),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Reject poor frames and combine with NaN-aware, tiled pixel statistics."""
        fraction = (
            float(reject_worst[:-1]) / 100 if reject_worst.endswith("%") else float(reject_worst)
        )
        params = StackParams(
            method=method,
            normalize=normalize,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            max_iterations=max_iterations,
            tile_rows=tile_rows,
            rejection=RejectionParams(
                reject_worst_fraction=fraction,
                min_quality=min_quality,
                max_fwhm=max_fwhm,
                max_eccentricity=max_eccentricity,
                max_background_sigma=max_background_sigma,
                max_saturation=max_saturation,
                max_registration_residual=max_registration_residual,
            ),
        )
        dataset = _dataset(source)
        if output.resolve() in {p.resolve() for p in dataset.frames}:
            raise PipelineError("Stack output must not replace an input frame.")
        base = output.with_suffix("") if output.suffix.lower() == ".gz" else output
        if not overwrite and any(
            p.exists() for p in (output, base.with_suffix(".processing.json"))
        ):
            raise PipelineError(f"Output already exists: {output} or its processing report.")
        if register:
            dataset = register_frames(
                dataset,
                output.parent / f"{output.stem}.registered",
                RegistrationParams(reference=reference, random_seed=random_seed),
                overwrite=overwrite,
            )
        image, report = stack_frames(dataset, params)
        save_stack(image, report, output, overwrite=overwrite)
        _messages(report)
        typer.echo(
            f"Stacked {report['used_frames']} frames; rejected "
            f"{report['rejected_frames']}. Saved {output}"
        )

    @session_app.command("inspect")
    def inspect(
        source: Path,
        as_json: bool = typer.Option(False, "--json"),
        cfa_pattern: str | None = typer.Option(None),
    ) -> None:
        """Classify frames using FITS aliases and conservative path heuristics."""
        session = discover_session(source, cfa_pattern=cfa_pattern)
        if as_json:
            typer.echo(session.model_dump_json(indent=2))
        else:
            for kind, count in session.counts().items():
                typer.echo(f"{kind.upper()}: {count}")
            for info in session.frames:
                typer.echo(
                    f"{info.path}: {info.frame_type} / {info.exposure}s / gain "
                    f"{info.gain} / {info.temperature} C / filter {info.filter_name} / "
                    f"{info.layout} {info.cfa.pattern if info.cfa else ''}"
                )
            _messages({"warnings": session.warnings})

    def make_master(
        kind: FrameType,
        source: Path,
        output: Path,
        method: Literal["mean", "median", "sigma-clipped"],
        bias: Path | None,
        dark: Path | None,
        overwrite: bool,
    ) -> None:
        frames = [inspect_frame(p) for p in discover_fits(source)]
        frames = [
            f.model_copy(update={"frame_type": kind})
            for f in frames
            if f.frame_type in (kind, FrameType.UNKNOWN)
        ]
        result = build_master(
            frames,
            kind,
            output,
            params=CombineParams(method=method),
            bias=load_fits(bias) if bias else None,
            dark=load_fits(dark) if dark else None,
            overwrite=overwrite,
        )
        typer.echo(f"Saved {result.path}")

    @master_app.command("bias")
    def bias(
        source: Path,
        output: Path = typer.Option(..., "--output"),
        method: Literal["mean", "median", "sigma-clipped"] = typer.Option("sigma-clipped"),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Build the readout-offset master without intensity normalization."""
        make_master(FrameType.BIAS, source, output, method, None, None, overwrite)

    @master_app.command("dark")
    def dark(
        source: Path,
        output: Path = typer.Option(..., "--output"),
        bias: Path | None = typer.Option(None),
        method: Literal["mean", "median", "sigma-clipped"] = typer.Option("sigma-clipped"),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Subtract optional bias from each dark before master combination."""
        make_master(FrameType.DARK, source, output, method, bias, None, overwrite)

    @master_app.command("flat")
    def flat(
        source: Path,
        output: Path = typer.Option(..., "--output"),
        bias: Path | None = typer.Option(None),
        dark: Path | None = typer.Option(None),
        method: Literal["mean", "median", "sigma-clipped"] = typer.Option("sigma-clipped"),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Correct flats and normalize each channel/CFA phase before and after combination."""
        make_master(FrameType.FLAT, source, output, method, bias, dark, overwrite)

    @master_app.command("build")
    def build(
        source: Path,
        output: Path = typer.Option(..., "--output"),
        cfa_pattern: str | None = typer.Option(None),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Build all compatible bias, exposure-dark, and filter-flat master groups."""
        masters = build_masters(
            discover_session(source, cfa_pattern=cfa_pattern), output, overwrite=overwrite
        )
        typer.echo(f"Built {len(masters)} masters in {output}")

    @app.command("calibrate")
    def calibrate(
        source: Path,
        output: Path | None = typer.Option(None, "--output", "-o"),
        bias: Path | None = typer.Option(None),
        dark: Path | None = typer.Option(None),
        flat: Path | None = typer.Option(None),
        scale_dark: bool = typer.Option(False),
        cosmetic_correction: bool = typer.Option(False),
        dark_contains_bias: bool | None = typer.Option(None, "--dark-contains-bias/--dark-no-bias"),
        cfa_pattern: str | None = typer.Option(None),
        debayer: bool = typer.Option(True, "--debayer/--no-debayer"),
        flat_min_fraction: float = typer.Option(0.05),
        temperature_threshold: float = typer.Option(3),
        dry_run: bool = typer.Option(False),
        explain: bool = typer.Option(False),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Discover a session, build missing masters, calibrate lights, then optionally debayer."""
        plan = CalibrationPlan(
            master_bias=bias,
            master_dark=dark,
            master_flat=flat,
            dark_scaling=scale_dark,
            cosmetic_correction=cosmetic_correction,
            dark_contains_bias=dark_contains_bias,
            cfa_pattern=cfa_pattern,
            flat_min_fraction=flat_min_fraction,
            temperature_threshold=temperature_threshold,
        )
        session = discover_session(source, cfa_pattern=cfa_pattern)
        if explain:
            typer.echo(
                "Bias removes readout offset. Dark removes thermal current and fixed "
                "sensor signal. Flat corrects pixel response and vignetting. "
                "Calibrate original CFA samples before debayering.",
                err=True,
            )
        if dry_run:
            typer.echo(json.dumps(plan_calibration(session, plan), indent=2))
            return
        if output is None:
            raise PipelineError("--output is required unless --dry-run is used.")
        with typer.progressbar(
            length=len(session.of_type(FrameType.LIGHT)),
            label="Calibrating lights",
            show_pos=True,
            file=sys.stderr,
            hidden=not sys.stderr.isatty(),
        ) as bar:
            result = calibrate_frames(
                AstroDataset([f.path for f in session.frames if not f.master], source=source),
                output,
                plan,
                debayer=debayer,
                overwrite=overwrite,
                progress=lambda: bar.update(1),
            )
        _messages(result.reports["calibration"])
        typer.echo(f"Calibrated {len(result.frames)} lights to {output}")

    @app.command("debayer")
    def debayer_one(
        source: Path,
        output: Path = typer.Option(..., "--output"),
        pattern: str | None = typer.Option(None),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Bilinearly demosaic a single raw CFA frame into RGB."""
        image = load_fits(source)
        cfa = (
            image.cfa
            if pattern is None
            else CFAMetadata(
                pattern=pattern,
                x_offset=int(image.header.get("XBAYROFF", 0)),
                y_offset=int(image.header.get("YBAYROFF", 0)),
            )
        )
        if source.resolve() == output.resolve():
            raise PipelineError("Debayer output must not replace the input image.")
        result = debayer_image(image, cfa)
        if not overwrite and output.with_suffix(".processing.json").exists():
            raise PipelineError("Debayer processing report already exists; use --overwrite.")
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input": str(source),
            "cfa": cfa.model_dump() if cfa else None,
            "method": "bilinear",
        }
        report["export"] = export_image(result, output, overwrite=overwrite)
        write_json(output.with_suffix(".processing.json"), report, overwrite=overwrite)
        typer.echo(f"Saved {output}")

    @app.command("process")
    def process(
        source: Path,
        output: Path = typer.Option(..., "--output", "-o"),
        cfa_pattern: str | None = typer.Option(None),
        overwrite: bool = typer.Option(False),
    ) -> None:
        """Run a complete deterministic session workflow and save its replay YAML."""
        result = process_session(source, output, cfa_pattern=cfa_pattern, overwrite=overwrite)
        _messages(result.report)
        typer.echo(f"Saved {output / 'master.fit'}")
