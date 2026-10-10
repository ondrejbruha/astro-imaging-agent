# Execution context and cancellation API

The provider-independent services live in `astroagent.execution`. Existing callers
and custom `ImageTool.process(image, params)` implementations continue to work.
There is no new abstract method, GUI dependency, mandatory callback or provider.

```python
from pathlib import Path
from astroagent.execution import ExecutionContext, Progress
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.serialization import load_pipeline


def progress(update: Progress) -> None:
    print(update.as_dict())


context = ExecutionContext(
    progress=progress,
    scratch_directory=Path("/work/scratch"),
    memory_mb=128,
    scratch_bytes=8 * 1024**3,
)
result = PipelineExecutor().run(
    load_pipeline("pipeline.yaml"), "input.fit", "output.fit", hdu=2, context=context
)
```

The scratch directory must exist for direct Python calls. A worker creates its own
scratch child on the configured working volume. Call `context.cancellation.cancel()`
from another thread to request a stop. Catch `ExecutionCancelled` at your operation
boundary; it is intentionally neither `AstroError`, `ValueError`, nor `RuntimeError`.
Frame-level calibration, master analysis and registration error handling must not
swallow it as an excluded frame.

Explicit `context=` is supported by image/dataset pipeline execution, image tool
execution, frame/star measurement, master creation, image/frame calibration,
image/frame debayering, registration, stacking, session inspection, read-only plan
preparation and preview generation. `context.activate()` also propagates the context
through nested existing tool calls using a context variable. It does not automatically
propagate to threads created by custom tools. Such tools can accept a captured context
or activate it explicitly; standard AIA processing is serial within a worker job.

Custom tools can call `checkpoint()`, `current_context()` and `emit_progress(...)`
without changing their `process()` signature. `execution_scope` activates an optional
explicit context for the duration of a public call and checks at entry/return. All
numerical operations remain independently usable without worker or agent imports.

## Progress conventions

`Progress` contains a phase, optional completed/total counts, unit, and one-based
pipeline step index. `None` counts mean indeterminate work. Units are steps, frames,
channels, or stack tiles rather than estimates of runtime. A step count is not a
time percentage. Counters increase within each occurrence of a phase; nested phases
may interrupt/resume the parent pipeline counter. A new occurrence of the same phase
starts a new sequence, for example a new master's spooling phase. Failed/excluded
frames count as visited/completed analysis or registration units. Unknown totals and
synchronous provider/kernel phases do not invent progress percentages or tokens.

Legacy calibration/registration `progress: Callable[[], None]` callbacks still fire
once per visited light/frame. Structured progress additionally reports the phase and
count. Worker callbacks retain only the latest update, throttle repeated intermediate
updates to roughly ten per second, and promptly emit transitions/final counters.
There is no unbounded event buffer. The host must continuously drain stdout; a blocked
pipe applies backpressure and can delay all transport traffic.

## Safe checkpoints and latency

Checkpoints exist before/after public context calls, between pipeline steps, session
headers, frame analyses, calibration lights, master quality passes/groups/corrected
input units, detected/HFR sources, debayer channels/frames, registration frames and
before their saves, stack input spooling and row tiles. Existing per-frame broad catches
do not catch cancellation. An indivisible SciPy/Astropy/NumPy kernel, file write or
SDK call may finish before cancellation is observed. Stop latency therefore depends
on image dimensions, tile size, I/O and provider timeout, not just callback frequency.
No immediate kernel interruption or process RSS ceiling is promised.

Direct Python/CLI writes retain the existing per-file atomic behavior. An image,
YAML, and processing report are independent atomic files, not a transaction. A later
write failure or cancellation can leave recoverable earlier files. Worker jobs wrap
these existing writers in unique workspaces and require `complete.json` before
publishing result references. See [worker-protocol.md](worker-protocol.md) for the
sealed finalization/cancellation race and cleanup rules.

## FITS selection and replay

`load_fits(path, hdu=None)` and `load_image(path, hdu=None)` retain first-image defaults.
`hdu=N` selects exactly the zero-based FITS HDU; empty primary/table/out-of-range HDUs
are errors. Non-FITS input rejects explicit HDU selection. `list_hdus(path)` reads
headers only, returning index, name/version, HDU type, NumPy-order dimensions,
BITPIX/storage precision, scaling and supported-layout status. Ambiguous cubes are
unsupported unless the existing `ASTRCHAX` declaration resolves their channel axis.
Mono and channel-first/channel-last RGB follow the existing loader rules.

Primary and selected-extension metadata are merged, with the extension taking
precedence. Astropy applies integer scaling and BLANK masks; floating NaNs, FITS
headers, WCS and RGB storage orientation retain their existing behavior. The actual
selected index is `AstroImage.input_hdu`, propagated by `with_data()` and recorded
as `ProcessingReport.input_hdu` (default null for older saved reports).

```bash
aia inspect multi.fit --hdu 2 --json
aia run output.pipeline.yaml --input multi.fit --hdu 2 --output replay.fit
```

`PipelineExecutor.run(..., hdu=2)` and `AgentExecutor.prepare(..., hdu=2)` use the same
selection. Input selection stays outside processing YAML version 1. Directory dataset
replay rejects `--hdu` instead of silently discarding it. To replay a worker's explicit
selection, use `execute_dataset_pipeline` with `AstroDataset(selected_paths,
reference=..., purpose_overrides=...)` and the resolved pipeline, or send those inputs
back through the worker. CLI directory discovery remains available for existing users
and can include additional frames; it is not a substitute for a GUI's explicit selection.

## Resources and ownership

Stacking still spools float32 inputs to a disk memmap then combines row tiles with
float64 accumulation and float32 output. Estimated spool storage is
`N * height * width * channels * 4` bytes plus a small NumPy header. The existing
`tile_rows` and `memory_mb` tool parameters are authoritative. The execution context
can lower the tile memory budget; `scratch_bytes` limits the spool estimate and a
free-disk check precedes allocation. The estimate excludes final output, calibration
intermediates and filesystem overhead. Disk exhaustion can still happen later and
must be handled as a job failure. Session inspection surfaces the approximate spool
estimate and configured budgets before processing.

`memory_mb` is an approximate tile working budget, not a process RSS cap. Input and
output images, float conversions, masks, clipping/interpolation temporaries and
libraries need additional RAM. One full frame still needs to fit in memory. The
adapter does not construct an in-memory frame cube. Catalogs are pixel-free and
bounded by validated detection/selection limits; completed jobs retain references
only, and prepared planning images are released after their job. There is no preview
cache. The host owns retention and cleanup of completed/incomplete job directories;
the worker automatically cleans only its verified scratch child.
