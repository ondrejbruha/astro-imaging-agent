# astro-imaging-agent

**Made by Alpha Codes s.r.o.** Author and maintainer: **Ondřej Brůha**
([ondrej.bruha@alphacodes.eu](mailto:ondrej.bruha@alphacodes.eu)).

`aia` is a Python CLI and modular astronomical image-processing library. It measures
images, applies deterministic processing tools, and saves replayable YAML pipelines
with JSON processing reports. The default rule-based planner runs entirely locally.
Optional OpenAI, Anthropic and Google Gemini planners interpret natural-language goals.

The purpose of the LLM integration is **planning and orchestration**.
A generative model never modifies pixels. The processing engine accepts a validated
pipeline and remains usable independently of any agent.

## Installation

Requires Python **3.12+**. Package name: `astro-imaging-agent`; Python import:
`astroagent`; CLI command: **`aia`**.

After the first PyPI release:

```bash
python -m pip install astro-imaging-agent
aia --help
```

For development from source, see [development.md](development.md). The core runs
locally; optional LLM planners require their SDK extra and API key.

## Supported image formats

| Format | Reading | Writing | Precision and metadata |
| --- | --- | --- | --- |
| FITS (`.fit`, `.fits`, `.fts`, also `.gz`) | Mono and unambiguous RGB image HDUs | Mono/RGB | Integer or float; scientific header, WCS, COMMENT and HISTORY preserved; structural, scaling and checksum cards regenerated |
| TIFF (`.tif`, `.tiff`) | Single mono/RGB image; contiguous or planar RGB | Mono/RGB | Integer or float arrays preserved; own exports embed original metadata and the complete FITS header in ImageDescription |
| PNG | Mono/RGB, including **16-bit RGB** and opaque palettes | Mono/RGB | uint8 stays 8-bit; uint16/normalized floats export as 16-bit; lossless compression, float quantization |
| JPEG (`.jpg`, `.jpeg`) | Mono/RGB | Mono/RGB | 8-bit **lossy**, quality 95 with no chroma subsampling |
| WebP | Single mono/RGB frame | 8-bit RGB | Lossless compression; floats/uint16 quantized to 8-bit; mono exports decode as RGB |
| BMP | Mono/RGB | Mono/RGB | 8-bit, lossless compression; floats/uint16 quantized to 8-bit |

PNG/JPEG/WebP/BMP export accepts float pixels in **0..1**, or unsigned 8/16-bit
integer input scaled by its dtype range. It never silently stretches raw scientific
data. Add `normalize` or `stretch` before exporting an unbounded float image.
TIFF and FITS preserve unbounded floats and negative sky-subtracted residuals.
Standard raster exports retain original metadata in the processing JSON sidecar;
they do not embed a FITS header. JPEG/WebP/BMP are display products, not scientific
intermediates. Alpha channels, animated images, TIFF stacks and non-RGB FITS cubes
are rejected with a readable error. Ordinary JPEG/WebP/BMP EXIF orientation is applied
when loading. Existing tone mapping in display images cannot be reliably inferred.

FITS RGB is internally `(height, width, 3)`. A cube without an `ASTRCHAX` card
must have exactly one end axis of length three; a cube with both end axes of length
three is ambiguous. Own exports declare the NumPy storage channel axis in
`ASTRCHAX`. Loaded FITS storage orientation is retained, preserving WCS axis mapping.

## CLI examples

```bash
aia inspect image.fit
aia inspect image.fit --json
aia analyze-background image.fit
aia analyze-stars image.fit --fwhm 3 --threshold-sigma 5

aia normalize image.fit normalized.fit
aia stretch image.fit stretched.fit --method asinh --strength 0.6
aia denoise image.fit smoothed.fit --sigma 0.8
aia background-extract image.fit corrected.fit --grid-size 8 --polynomial-degree 2

# The output extension selects the format.
aia stretch image.fit preview.png --strength 0.6
aia normalize photo.tiff preview.jpg
aia inspect preview.png --json

aia run examples/pipeline.yaml --input image.fit --output processed.fit
aia agent image.fit "zpracuj tento snímek přirozeně a nepřepal hvězdy" --dry-run --explain
aia agent image.fit "zpracuj tento obrázek" --output processed.fit --explain

aia tools
aia -v run examples/pipeline.yaml --input image.fit --output processed.fit
aia -vv inspect missing.fit
```

Verbosity flags precede the command: `-v` logs steps, `-vv` includes debug tracebacks.
Expected errors show `Error: ...` with exit code 1; normal execution has no debug logs.
Single-tool commands can omit the output, defaulting to `<input-stem>.processed.fit`.
Use `--overwrite` to replace output artifacts; input replacement is always refused.
`python -m astroagent` provides the same CLI.

## Pipeline and processing reports

```yaml
version: 1
steps:
  - tool: background_extract
    params:
      grid_size: 8
      polynomial_degree: 2
      sigma_clipping_threshold: 3.0
  - tool: denoise
    params:
      method: gaussian
      sigma: 0.8
  - tool: stretch
    params:
      method: asinh
      strength: 0.6
```

Every processing command saves:

```text
processed.fit
processed.pipeline.yaml
processed.processing.json
```

The saved YAML contains all **resolved defaults**, so changes in future defaults do
not change the plan. Replay it with `aia run processed.pipeline.yaml --input image.fit
--output replay.fit`. All steps and parameters are validated before processing starts;
unknown tools, misspelled fields, unsupported versions and invalid values are rejected.
The executor stops at the first failure and identifies its step number.

The JSON report contains input/output paths, package/dependency versions, pixel
SHA-256 hashes, resolved pipeline, per-step before/after metrics, elapsed milliseconds,
warnings and (for agent runs) the request, short explanations, and full analysis of
the initial/final images. Pixel hashes canonicalize data as little-endian float64
with shape; the output-file hash identifies the encoded artifact. Raster export
quantization is also measured separately. Timings are informational and vary between
runs. Replay equality requires the same inputs, tool versions, parameters and numerical
environment; cross-platform floating-point bitwise identity is not promised.

Outputs are checked for collisions before execution. Each artifact is written
atomically, but the three-file bundle is not a transaction: a disk failure may leave
a partial bundle. Intermediate files are not saved.

## Architecture

```text
src/astroagent/
  models/       ndarray-based AstroImage and shared strict schema model
  io/           FITS/raster adapters and metadata handling
  analysis/     finite statistics, robust background, photutils stars, quality
  tools/        typed deterministic tools and immutable per-executor registry
  pipeline/     Pydantic definitions, YAML/JSON, executor and execution history
  agent/        planner protocols, rules, and analysis/plan/execute orchestration
  cli/          Typer argument parsing, presentation and error boundary
```

The central boundary is:

```text
Planner -> PlanResult(PipelineDefinition, explanations)
       -> PipelineExecutor -> ImageTool -> ToolResult
```

`PipelineExecutor` does not import the agent layer. `AstroImage` is a dataclass so
large NumPy arrays do not pass through Pydantic. Schemas, metrics, plans and reports
use Pydantic v2. Tools operate in float64, return new arrays, copy metadata/headers,
and provide before/after metrics and warnings. There is no global mutable registry.
SciPy supplies Gaussian filtering and NumPy supplies polynomial fitting; scikit-image
is not needed for this implementation. Optional future backends (OpenCV, CuPy,
PyTorch, SEP, astroalign, ccdproc) can live behind new tools/adapters.

## Analysis and numerical behavior

- `inspect` reports dimensions `(height, width)`, channels, dtype, finite min/max,
  mean/median/std, percentiles 1/5/50/95/99, saturation and JSON-safe metadata.
  RGB summary values aggregate channel samples. NaN/Inf are excluded with a warning;
  processing rejects them until a future repair/masking tool is provided.
- Saturation uses `SATURATE`, then the integer dtype ceiling, or 1 for bounded float
  images. Otherwise its fraction is `null` with a warning. It counts **channel samples**,
  not whole RGB pixels. A normalized maximum is not evidence of physical detector
  saturation; saturation fractions after stretching are display-range measurements.
- Background uses sigma-clipped tile medians and a fitted plane. `sigma` is clipped
  residual noise; `gradient_estimate` is the fitted surface's 95th–5th percentile
  range in input units. RGB analysis uses equal-weight channel averaging.
- Stars use photutils `DAOStarFinder`. Positive local second moments give approximate
  Gaussian FWHM in pixels, ellipticity `1-b/a`, and eccentricity `sqrt(1-(b/a)^2)`.
  Failed detection or unreliable shapes return `null` and warnings. This is a rough
  quality indicator, not precision PSF fitting or calibrated photometry.
- Normalize uses one global min/max scale across RGB; constant images map to the
  lower endpoint. The target interval must satisfy `0 <= lower < upper <= 1`.
- Stretch also uses global endpoints. `black_point` is in input units and must be
  below the maximum. `linear` ignores strength; `asinh` uses
  `gain = 10**(3*strength)-1` and `asinh(gain*x)/asinh(gain)`, with strength zero
  reducing to linear. Strength is in 0..1. Explicit black points can clip shadows.
- Gaussian denoising uses reflected spatial edges and never mixes RGB channels.
  It softens stars; the report makes that limitation explicit.
- Background extraction subtracts a total-degree 0–3 polynomial from each channel
  independently. It removes offset as well as gradient and preserves negative
  values. Extended nebulosity, gradients within tiles and crowded fields can bias
  automatic samples; rank-deficient grids are rejected. It is not a full DBE tool.

## Agent flow

`AgentExecutor` loads an image, inspects it, estimates sky and stars, then sends only
`ImageMetrics` and `ToolDescription` schemas to `Planner.create_plan`. The CLI prints
the pipeline, executes it through the ordinary executor, reanalyzes, and saves all
artifacts. `--dry-run` displays the plan without writing any files. `--explain`
displays concise explicit reasons; no hidden model reasoning is stored.

`RuleBasedPlanner` compares gradient/noise against the robust 99th–1st percentile
range: gradient over 15% selects a degree-1 background model, noise over 8% selects
mild Gaussian denoising, and a dark median suggests an asinh stretch. `ASTRSTR`
marks images stretched by this tool. Out-of-range non-linear images are normalized.
Constant images have empty plans. Only advertised tools can be selected.

**The rule-based MVP does not understand the free-text request.** It records it for
provenance and plans from measurements. Its thresholds and linearity inference are
heuristics, so review `--dry-run --explain` before relying on them. `LLMPlanner` provides the common boundary for `OpenAIPlanner`, `AnthropicPlanner`
and `GeminiPlanner`; each returns the same validated `PlanResult` and has no access
to image arrays or engine internals.

## LLM planners

Install only the provider you need, or all three:

```bash
python -m pip install 'astro-imaging-agent[openai]'
python -m pip install 'astro-imaging-agent[anthropic]'
python -m pip install 'astro-imaging-agent[gemini]'
python -m pip install 'astro-imaging-agent[llm]'
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` in your environment.
Choose a model ID available to your account explicitly; no model is hardcoded:

```bash
aia agent image.fit "preserve stars and bring out faint detail" --provider openai --model YOUR_GPT_MODEL --dry-run --explain
aia agent image.fit "natural processing" --provider anthropic --model YOUR_CLAUDE_MODEL --output processed.fit
aia agent image.fit "remove the sky gradient" --provider gemini --model YOUR_GEMINI_MODEL --dry-run
```

`--provider rules` is the default. `--timeout` controls SDK request timeout (default
60 seconds). LLM dry-run calls the selected provider to generate a plan but never
writes or processes an image. The provider receives the request, image measurements,
metadata and tool schemas; image pixels/files and API keys are not included in the
planning payload. The report records provider and model, while the saved YAML replays
offline with `aia run`. Planning itself can vary between API calls.

OpenAI uses [Responses JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs),
Claude uses a [forced plan-submission tool call](https://platform.claude.com/docs/en/api/python/messages/create),
and Gemini uses [Google GenAI JSON generation](https://github.com/googleapis/python-genai).
Extensible parameter dictionaries are checked locally with Pydantic and the registry;
provider JSON mode alone does not guarantee schema validity. Malformed, incomplete,
unknown-tool or invalid-parameter plans are rejected before processing. There is no
automatic fallback to rules when an API call fails. API errors omit SDK response
bodies; keys are read from the environment and are never written to artifacts.
Tests use fake clients and do not make paid API calls.

## Contributing

See [development.md](development.md) for Poetry setup, test commands, adding tools,
versioning, and PyPI/GitHub release configuration.

## MVP scope and next steps

Implemented: multi-format I/O, inspection, background/star analysis, normalization,
linear/asinh stretching, Gaussian denoising, polynomial sky subtraction, tool schemas,
YAML replay, processing history, rule-based orchestration and packaging automation.

Not included: stacking, bias/dark/flat calibration, debayering, plate solving,
photometric color calibration, neural processing, GPU acceleration or distribution.
The next useful step is explicit sky/star masks and stronger quality checks for
crowded fields and nebulosity, followed by dataset-based evaluation of LLM-generated
plans. Scientific FITS/TIFF artifacts
should remain the processing intermediates.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
