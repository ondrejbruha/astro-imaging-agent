# aiaGUI backend implementation report

Prepared on 2026-10-10 against `fc1d3d0`. The starting working tree was clean.
The existing processing algorithms, planners, registries and serializers were extended;
no second engine, GUI application, new provider framework or runtime dependency was added.

## Changed modules

Paths in this table are relative to `src/astroagent/` unless stated otherwise.

| Area | Modules/files | Result |
| --- | --- | --- |
| Worker | `worker/__init__.py`, `__main__.py`, `protocol.py`, `models.py`, `jobs.py`, `server.py` | Installed-module launch, typed protocol/method schemas, one processing thread, bounded queue/status, control requests, all assigned methods |
| Execution | `execution.py`, `errors.py` | Thread-safe cancellation, scoped context, structured progress, safe validation/resource/provider error categories |
| Pipelines | `pipeline/executor.py`, `dataset_executor.py`, `models.py` | Context propagation, input-kind transition validation, resolved defaults/locations, selected-HDU provenance, source alias protection |
| I/O/models | `io/__init__.py`, `fits.py`, `images.py`; `models/image.py`, `dataset.py` | Header-only HDU listing, explicit selection, propagated identity, purpose overrides |
| Catalog/tools | `tools/base.py`, `dataset.py`, `registry.py`, `color.py` | Real Pydantic schemas with nested references, input/output/layout/NaN facts, path hints, implemented preview strategies; original custom process signatures retained |
| Preview | `preview.py` | Existing-tool processing followed by crop/resize, separately recorded display mapping and atomic PNG export |
| Quality | `analysis/__init__.py`, `hfr.py`, `frame_quality.py`, `stars.py`, `statistics.py`, `quality.py` | Aperture HFR/median fields with old-report defaults, eligibility and cancellation, unchanged quality weights |
| Calibration | `calibration/debayer.py`, `engine.py`, `masters.py`, `models.py`, `session.py`, `workflow.py` | Cooperative checkpoints, selected purpose/reference propagation, header/WCS session metadata |
| Registration | `registration/engine.py`, `stars.py`, `matching.py`, `transform.py` | Frame/source/matching/RANSAC checkpoints, selected reference, HFR catalog measurement; cancellation propagates through frame catches |
| Stacking | `stacking/stack.py`, `workflow.py` | Same memmap/tile numerics, configurable scratch, spool/free-disk estimates and cancellation |
| Planning/providers | `agent/executor.py`, `dataset_planner.py`, `providers.py` | Read-only selected-image/dataset preparation, validated alternatives/eligibility, explicit request-local provider keys and finite timeouts |
| CLI | `cli/main.py` | Additive `inspect --hdu` and `run --hdu` replay, existing commands preserved |
| Tests | `tests/test_execution_context.py`, `test_hfr_preview_hdu.py`, `test_worker.py` | Synthetic transport/jobs/artifacts/HDU/preview/planning/secrets/HFR/dataset/compatibility coverage |
| Package/CI | `pyproject.toml`; `scripts/check_dist.py`, `smoke_dist.py`, `worker_smoke.py`; `.github/workflows/ci.yml`, `release.yml` | Documentation in sdist; installed-wheel worker processing, death/restart checks on existing Linux/Windows matrix |
| Documentation/rules | `README.md`, `development.md`, `AGENTS.md`, `docs/worker-protocol.md`, `execution-and-cancellation.md`, `preview-and-quality.md`, `release-handoff.md`, this report | Public contracts, limits, ownership, compatibility and unreleased handoff |

## API decisions

Protocol and processing YAML remain version 1. Request/response/event and method
parameter schemas are exported during handshake. `tools.list` preserves the actual
registry's complete Pydantic schemas rather than flattening them into a GUI registry.
Names/defaults/cross-field rules and the full dataset/image transition chain are
validated before processing output directories are created. One-based step indices
and nested field paths are returned without parameter values.

Existing path-based Python/CLI calls remain supported. Optional `context=` uses
scoped propagation so custom tools keep their original `process()`/dataset signatures.
Cancellation is a dedicated exception and is never an excluded frame. A job result
is complete only after all required writers/cleanup succeed and the sealed atomic
completion manifest is written. Multi-file output is explicitly not a filesystem
transaction; earlier files can survive a later failed write as incomplete artifacts.

The worker accepts host-selected absolute read locations and a working volume at
handshake; it writes only unique job workspaces. Source HDUs are consistent across
inspection, preview, planning and execution, with CLI replay. Explicit selected FITS
frames, reference choices and purpose overrides are preserved. Header-only session
inspection does not hash/read all pixel bytes; planning and execution identities use
streamed SHA-256 plus stat information. No prepared full images remain cached.

Planning is read-only and separate from reviewed execution. It returns validated
primary/alternative pipelines and explanations, associated with expiring input
identities; changed input conflicts rather than silently reusing approval. Provider
credentials are request-local and are excluded/redacted from outputs. Provider SDK
errors and numerical incompatibility have distinct codes. `llm.test` is one explicitly
requested existing planner call for an empty plan with no tools/images; fake clients
cover automated tests. No paid calls were made.

Previews use full-resolution processing before output crop/resampling, so global
statistics and pixel-radius operations stay correct. PNG display mapping is separate
provenance, not an implicit scientific stretch. HFR uses background-subtracted
aperture curves of growth and reliable-source medians rather than multiplying FWHM;
existing quality/reference/stack weighting semantics remain unchanged.

## Checks actually run

The repository's existing isolated Poetry 2.3.1 with dynamic-versioning 1.10.0 was used
after the documented global Poetry installation failed plugin dependency resolution.
No dependency-manager replacement was introduced.

| Check | Observed result |
| --- | --- |
| Poetry install with all extras | Passed through `.tools/poetry/Scripts/poetry.exe`; no lock/dependency changes |
| `poetry check --lock` | Passed |
| Ruff format/check | Passed |
| mypy | Passed, 79 source files |
| Full pytest with coverage, Windows Python 3.13.7 | 327 passed, coverage 93.00%, existing 85% threshold preserved |
| Build wheel/sdist | Passed; version `0.0.2.post1.dev0+fc1d3d0` derived from Git |
| `scripts/check_dist.py --dist dist/gui-backend` | Passed wheel/sdist metadata, license/NOTICE, required modules and frozen version checks |
| `scripts/smoke_dist.py --dist dist/gui-backend`, Windows | Passed installed wheel CLI/replay, real worker execution, interruption/restart, then sdist reinstall outside checkout |
| Installed-wheel smoke, Ubuntu WSL Python 3.14.4 | Passed CLI, handshake, real processing and interrupted-job restart outside checkout, without optional provider SDKs |
| Read-only Git whitespace checks | Passed working-tree and existing index checks |

The sole full-suite warning is the existing Astropy FITS `BAYERPATN` long-card warning.
Local Ubuntu lacked ensurepip/venv support for the generic distribution smoke. The
Linux installed-wheel scenario instead used an isolated temporary runtime with pip
bootstrapped there; it did not install system packages. Full Linux pytest and the
complete remote Python-version matrix were not run locally; the existing matrix
remains configured with the added worker smoke. Real-data benchmarks, real provider
connections and Qt host tests were not run: synthetic/fake transports are the assigned
test boundary, and Qt management belongs to aiaGUI.

## Remaining limitations and release boundary

- Cancellation is cooperative; indivisible numerical kernels/file writes and finite
  synchronous provider calls delay it. A broken transport or process death can prevent
  terminal-event delivery and requires host restart/recovery.
- Preview processing requires the full image. Accelerated ROI processing, reduced-input
  processing, persistent preview caching, endpoint customization and token streaming
  are not advertised or implemented. The GUI must honor returned preview generations.
- Memory budgets are approximate tile budgets, not RSS caps. One frame/output and
  numerical temporaries still need RAM. Scratch estimates cover the stack spool;
  intermediate/final products need additional disk. Owned artifacts remain until
  host cleanup, including deliberately retained incomplete work.
- HFR is finite-aperture, discretized, noise/blend-sensitive image-quality measurement,
  not calibrated photometry. Unsupported/crowded/masked/saturated/border sources are null.
- Dataset/session/calibration input support remains FITS only. TIFF support applies to
  single-image operations and does not imply TIFF sessions.

Artifacts are prepared in `dist/gui-backend/`. No release or PyPI publication was
created and no mutating Git commands were run. Proposed maintainer release target is
**0.0.3**, not an already published version. aiaGUI must pin
`astro-imaging-agent[llm]==0.0.3` only if the maintainer approves and publishes that
version containing these changes; otherwise use the actual approved released version.
See [release-handoff.md](release-handoff.md) for the existing publication workflow.
