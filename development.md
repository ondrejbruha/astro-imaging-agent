# Development

Made by Alpha Codes s.r.o. Author and maintainer: Ondřej Brůha <ondrej.bruha@alphacodes.eu>.

## Local setup

Use Python 3.12+ and Poetry 2.2+ (CI uses 2.3.1). From a Git checkout:

```bash
poetry install --all-extras
poetry run aia --help
poetry run pytest
```

Omit `--all-extras` for the offline core, or select `-E openai`, `-E anthropic`,
or `-E gemini`. Poetry installs the project's declared dynamic-versioning plugin.
The environment is local to `.venv/`; dependencies are frozen in `poetry.lock`.
Use `poetry run aia` inside this environment; installed-package users invoke `aia`
directly. No API keys are needed for tests, including provider contract tests.

Install Poetry in a dedicated environment (for example with pipx) so its plugin
resolver does not inherit unrelated application dependencies. On the machine used
to prepare this project, the global Poetry environment had conflicting LangChain
dependencies; an isolated local Poetry installation is already available at
`.tools/poetry/Scripts/poetry.exe`. On that machine, use:

```powershell
.\.tools\poetry\Scripts\poetry.exe install --all-extras
.\.tools\poetry\Scripts\poetry.exe run pytest
```

`.tools/` is ignored by Git and is not needed by installed-package users or CI.

## Adding a processing tool

Define a validated schema, implement an `ImageTool`, and register a fresh instance:

```python
import numpy as np
from pydantic import Field
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool
from astroagent.tools.registry import ToolRegistry, default_registry


class OffsetParams(SchemaModel):
    amount: float = Field(default=0.0, ge=-1, le=1)


class OffsetTool(ImageTool[OffsetParams]):
    name = "offset"
    description = "Add a fixed intensity offset."
    params_model = OffsetParams

    def process(self, image: AstroImage, params: OffsetParams) -> tuple[AstroImage, list[str]]:
        return image.with_data(np.asarray(image.data, dtype=np.float64) + params.amount), []


defaults = default_registry()
registry = ToolRegistry([*(defaults.get(item.name) for item in defaults.describe()), OffsetTool()])
```

Pass the registry to `PipelineExecutor(registry)`. Tool descriptions automatically
expose Pydantic JSON schemas. Add synthetic-image tests for the algorithm, validation,
input immutability, metadata and replay. No executor `if/elif` or provider changes
are required. Direct usage is also supported, e.g. `StretchTool().execute(image,
StretchParams(strength=0.6))`.

## Development and publishing

```bash
poetry check --lock
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy
poetry run pytest --cov=astroagent --cov-report=term-missing
poetry build
poetry run twine check dist/*
```

CI runs these checks on Python 3.12, 3.13 and 3.14 on Linux and Python 3.13 on Windows,
then verifies a built wheel can be installed and its CLI invoked outside the checkout.
Tests use small synthetic arrays, including realistic Gaussian stars and polynomial
sky gradients; no external datasets are fetched.

Versions come from Git tags through [poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning).
`[project].dynamic = ["version"]` declares that boundary; `[tool.poetry].version =
"0.0.0"` is only a required placeholder. A release tag such as `v0.1.0` produces
version `0.1.0`; untagged checkouts get a development version. Source distributions
freeze the derived version, so installation from PyPI does not need Git.

`.github/workflows/release.yml` builds and tests a published release tag (or a manual
run on a tag), verifies the artifact version matches the tag, checks both wheel and
sdist, and publishes through PyPI Trusted Publishing. Before the first release:

1. Configure the GitHub environment **`pypi`**, optionally with required reviewers.
2. Register a PyPI pending/trusted publisher for your repository, workflow
   **`release.yml`**, and environment **`pypi`**. Follow the
   [PyPI documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
3. Create and publish a GitHub release for the desired `vX.Y.Z` tag. The workflow
   also supports `workflow_dispatch` on an existing tag. No stored PyPI token is required.

This repository preparation does not itself publish a package or create a release.
Project instructions in `AGENTS.md` prohibit commits/pushes during agent work and
require tests and staged, reviewable changes.
