# Local worker protocol 1

Launch the host's bundled Python as `python -m astroagent.worker`. The installed
wheel needs no checkout, activated environment, Poetry, Qt, keys, network service,
or writable package directory. Use binary stdin/stdout pipes and UTF-8 JSON Lines.
Do not launch through a shell or attach a terminal. LF and CRLF, fragmented writes,
and multiple lines in one write work. Flush each request. Read stdout continuously.
Library prints/logs are suppressed; stderr contains only static fatal diagnostics.

This is a local trusted-host protocol, not an authentication boundary or a network
service. The host selects filesystem locations during the first handshake. CLI and
Python callers retain their existing unrestricted path behavior.

## Envelopes and negotiation

```json
{"protocol_version":1,"type":"request","id":"hello","method":"worker.handshake","params":{}}
```

The response has `protocol_version:1`, `type:"response"`, matching `id`, and either
`result` or `error`. A successful handshake includes installed `aia_version`,
`supported_protocol_versions`, numerical `dependency_versions`, capabilities, and
limits. An empty handshake enables catalogs/validation/control only. For filesystem
or provider jobs, initially configure absolute host-selected locations:

```json
{"protocol_version":1,"type":"request","id":"hello","method":"worker.handshake","params":{"read_roots":["C:/observations"],"workspace_root":"C:/AIA-work","memory_mb":256,"scratch_bytes":34359738368}}
```

Use native absolute paths on Linux. Read locations may be files or directories;
inputs must be explicit files whose canonical paths are within those locations.
Symlinks are resolved before authorization. Workspace roots need not already exist;
job-owned directories are created there. Configuration is immutable for a process;
repeat the same handshake or restart to change it. Root paths are at most 4096
characters. Input/output host choices should be stable while a job runs; protection
against a hostile process replacing filesystem entries is outside this local protocol.

```json
{"protocol_version":1,"type":"response","id":"bad","error":{"code":"INVALID_PARAMS","message":"Pipeline parameters or transitions are invalid.","details":{"locations":[{"step_index":1,"field":["detection","fwhm"],"type":"greater_than"}]}}}
```

```json
{"protocol_version":1,"type":"response","id":"run","result":{"job_id":"job-<generated-uuid>","state":"QUEUED"}}
```

```json
{"protocol_version":1,"type":"event","request_id":"run","job_id":"job-<generated-uuid>","event":"job.progress","data":{"phase":"pipeline","completed_units":1,"total_units":3,"unit":"step","step_index":1}}
```

Job acceptance always precedes its events. Other request responses and job events
may interleave. IDs are nonempty strings of at most 128 characters and unique within
the process. Restart resets IDs/status; old jobs are never replayed. Unsupported
versions are rejected before side effects. Unrecoverable IDs, malformed UTF-8/JSON,
and oversized lines produce a response with `id:null`. Duplicate JSON keys,
nonfinite numbers (including exponent overflow), extra envelope/parameter fields,
and invalid parameter types are rejected. Inputs and output messages are bounded
to 1 MiB including newline; oversized lines are drained to their next newline.
An EOF-terminated final JSON line is accepted; subsequent EOF initiates shutdown.

The server uses one processing thread and an eight-job pending queue. Excess jobs
return `QUEUE_FULL`. Session request IDs are bounded to 10,000; restart at that
limit. Keep at most 128 terminal statuses, 32 input-identity plans with 300-second
eligibility, and 64 preview channels. Job result dictionaries are limited to 16 KiB;
large measurements, catalogs, plans, and reports are files. No image arrays, binary
contents, credentials, NaN, or Infinity are protocol outputs. Filesystem failures
are sanitized; structured validation locations contain field paths, never values.

## Methods and parameters

Parameter models live in `astroagent.worker.models`; handshake returns a `schemas`
map for request/response/event envelopes and every method's parameters. These are also
available through `model_json_schema()` in Python. Processing parameters come
directly from the tool registry. All tables below describe implemented behavior.

| Method | Parameters | Delivery and result |
| --- | --- | --- |
| `worker.handshake` | Optional `read_roots`, `workspace_root`, `memory_mb`, `scratch_bytes` | Direct; versions, capabilities, limits |
| `worker.shutdown` | `{}` | Direct acknowledgement; cancel queued/running unsealed jobs, then exit |
| `tools.list` | `{}` | Direct; registry schemas, input/output kinds, layouts, NaN behavior, path hints, preview strategies |
| `pipeline.validate` | `pipeline`, `input_kind` (`image` default or `dataset`) | Direct; resolved definition; checks full transition chain |
| `pipeline.load` | `path`, optional `input_kind` | Job; safe-load YAML (at most 1 MiB), validate, return resolved YAML path |
| `pipeline.save` | `pipeline`, optional `input_kind`, `filename` (`pipeline.yaml`) | Job; validate/save resolved ordinary YAML into owned workspace |
| `pipeline.execute` | `input`, `pipeline`, optional `output_name` (`output.fit`), `plan_id` | Job; existing executor, image/dataset products, identity and completion manifest |
| `image.hdus` | `path` | Job; path to header-only HDU listing and count |
| `image.inspect` | `input` image reference, optional `quality`, `detection` | Job; inspection JSON; optional existing quality/catalog measurements including HFR |
| `image.preview` | image `input`, `pipeline`, optional `roi`, `scale`, `generation`, `channel` | Job; display PNG, provenance, generation, approximation flag |
| `session.inspect` | dataset `input`, optional `detection` | Job; header metadata/selection and estimated stack spool requirement; detection is used only by analysis |
| `dataset.analyze` | dataset `input`, optional `detection` | Job; one-frame-at-a-time measurements/catalogs, scoring and report path |
| `agent.plan` | `input`, `request`, optional `config` | Job; validated primary/alternative plans, reasons, warnings, input identity and expiring `plan_id` |
| `llm.test` | `config` | Job; explicitly requested provider test; rules are local, SDK providers make one existing planner call asking for an empty plan with no tools/images |
| `job.status` | `job_id` | Direct; bounded state/progress/result references |
| `job.cancel` | `job_id` | Direct acknowledgement; wait for the terminal event |

Only catalog, validation, handshake, shutdown, and job controls are synchronous.
No job inputs can select imports, executable code, shell commands, or unregistered
tools. `output_name` and pipeline filenames are basenames, never arbitrary output
paths. Scientific products remain within unique job workspaces; the host explicitly
exports/copies completed products if another destination is desired.

Image reference:

```json
{"kind":"image","path":"/observations/master.fit","hdu":2}
```

Dataset reference (at most 10,000 explicit FITS paths):

```json
{"kind":"dataset","frames":["/observations/a.fit","/observations/b.fit"],"reference":"/observations/b.fit","purposes":{"/observations/a.fit":"light","/observations/b.fit":"light"}}
```

The optional reference must be selected. Duplicate files/aliases are rejected.
Purpose overrides use `light`, `bias`, `dark`, `dark_flat`, `flat`, or `unknown`, are
recorded in provenance, and do not invent exposure, gain, or CFA information.
Single-image methods support FITS/TIFF/PNG/JPEG/WebP/BMP subject to existing layout
rules. Dataset/session/calibration methods support FITS only. `build_masters` cannot
use its CLI `session` directory override in the worker; explicit selections govern.
Path-typed master dependencies are also checked against authorized read locations.

## States, shutdown and artifacts

States are `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.
`cancellation_requested` is independent of terminal state. `job.started`,
`job.progress`, and exactly one of `job.completed`, `job.failed`, `job.cancelled`
are emitted for each accepted job during normal transport operation. Cancellation
of queued work never invokes it. Repeated cancellation is idempotent. Terminal
jobs and jobs with `finalizing:true` reject cancellation with `accepted:false`.

All processing, artifact writes and scratch cleanup precede finalization sealing.
Under the scheduler/cancellation lock the worker checks cancellation and seals the
job, then atomically writes `complete.json`. A successful manifest is the commit
boundary: later cancel requests do not relabel success. Manifest write failures are
failures. No multi-file filesystem transaction is claimed. A cancelled/failed job
can retain scientific-looking files, but has no completed result or valid completion
manifest. Its workspace is explicitly recoverable/incomplete, not a product.

`ownership.json` identifies the unique job directory. Only its verified `scratch`
child is cleaned automatically. Products and incomplete intermediates stay until
the host removes the exact owned directory according to its retention policy.
Completed status retention does not delete files. On process death, any workspace
without `complete.json` is incomplete; the host must not infer success from an image
file alone. Restart establishes a fresh session and cannot report an old job as
completed. A manifest written just before process death can be recovered by explicit
host inspection, never automatic replay.

Shutdown and EOF stop acceptance, request cancellation of all unsealed jobs, drain
queued cancellations, and wait for running work to stop. Indivisible kernels and
synchronous SDK calls delay cooperative shutdown. Provider timeouts are finite
(0.1–120 seconds, SDK retries disabled). If a hung kernel/transport requires forced
termination, the host kills/restarts the process; that is crash recovery, not a
successful cooperative cancellation. Fatal transport failures may prevent delivery
of terminal events and require host recovery.

## Planning and credentials

`config` has `provider` (`rules`, `openai`, `anthropic`, `gemini`), explicit `model`,
`timeout` and request-local `api_key` for SDK providers. Rules require neither model
nor key. Optional SDKs are loaded only when used. There is no endpoint customization,
Ollama, token streaming, persistent credential file, or environment mutation.
Existing CLI environment-variable configuration remains supported. Logging and
third-party stdout/stderr prints are suppressed in the worker process to avoid SDK
body leakage; actionable static errors replace raw exception strings. Plan artifact
text is scrubbed of the supplied credential. Keys are neither returned nor written.
The GUI retains ownership of credential storage and the approval interaction.

Planning calls `AgentExecutor.prepare`/`prepare_dataset`, validates every candidate,
and writes only a planning artifact. It does not execute candidate pipelines or call
the autonomous loop. Reviewed execution is a separate request carrying the pipeline
and returned `plan_id`. Input identity covers canonical file paths, size, nanosecond
mtime, streaming SHA-256, selected HDU/reference/purpose information. Modified input
returns `INPUT_CHANGED`; expired identities require re-planning. No prepared image
arrays are retained between jobs. Manual execution can omit `plan_id`.
Header-only `session.inspect` uses stat identity with null SHA-256 rather than reading
all pixel bytes; reviewed plans/execution/previews use the stronger streamed identity.

Preview generations must increase per channel; accepting a newer request cancels
older unsealed work. Completed events carry channel/generation; the host must present
only its current generation, including when a previous completion races a new request.
There is no persistent preview cache, so changed source data cannot reuse stale pixels.

## Stable errors

| Code | Host action |
| --- | --- |
| `INVALID_JSON`, `INVALID_REQUEST`, `INVALID_PARAMS` | Repair JSON/envelope/fields; use safe locations |
| `UNSUPPORTED_VERSION` | Use compatible AIA and protocol versions |
| `METHOD_NOT_FOUND`, `UNSUPPORTED_CAPABILITY` | Use advertised methods/strategies/formats |
| `HANDSHAKE_REQUIRED`, `PATH_POLICY_REQUIRED` | Establish the host's session/configuration |
| `CONFIG_CONFLICT` | Restart to change configuration |
| `DUPLICATE_ID`, `SESSION_LIMIT` | Use unique IDs or restart at the session limit |
| `QUEUE_FULL`, `RESOURCE_LIMIT`, `RESULT_TOO_LARGE` | Wait, reduce budgets/selection, or use artifacts |
| `PATH_DENIED`, `IO_ERROR` | Select authorized readable files; check permissions/free disk |
| `INCOMPATIBLE_DATA` | Check dimensions, sampling, metadata and operation requirements |
| `PROVIDER_ERROR` | Check supplied key/model/access/timeout/connection |
| `JOB_NOT_FOUND`, `PLAN_EXPIRED` | Reinspect/replan; retained process state expired |
| `INPUT_CHANGED`, `STALE_PREVIEW` | Review changed input or submit a newer generation |
| `SHUTTING_DOWN`, `INTERNAL_ERROR` | Await exit or restart and inspect incomplete work |

Cancellation uses `job.cancelled` rather than an ordinary error response.
