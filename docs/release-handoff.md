# aiaGUI backend release handoff

Baseline inspected: local `fc1d3d0`, matching the previously verified 0.0.2 wheel.
These changes are unreleased. Do not pin 0.0.2 for worker integration: that release
does not contain this protocol. Proposed maintainer release target: **0.0.3**, subject
to approval and publication of a tag containing these changes. It is not claimed to
exist on PyPI. The GUI build must pin the actual approved released version, for example
`astro-imaging-agent[llm]==0.0.3` only after that version is published and verified.

Versions still derive from Git tags through Poetry dynamic versioning. No hardcoded
library version, tag, release, commit or PyPI publication is created by this assignment.
The installed package version and numerical dependency versions appear in handshake
and processing provenance. Protocol version 1 and processing YAML version 1 are
separate compatibility boundaries. The GUI should reject incompatible protocol/package
versions and maintain an exact tested dependency/runtime lock in its own repository.

## Maintainer verification and release

Use a clean distribution destination to avoid mixing historical builds:

```bash
poetry install --all-extras
poetry check --lock
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy
poetry run pytest --cov=astroagent --cov-report=term-missing
poetry build --output dist/gui-backend
poetry run python scripts/check_dist.py --dist dist/gui-backend
poetry run python scripts/smoke_dist.py --dist dist/gui-backend
```

On machines with the documented global Poetry/plugin conflict, use
`.tools/poetry/Scripts/poetry.exe` for the same commands. No dependency-manager change
is needed. Review artifacts and CI on Python 3.12/3.13/3.14 Linux and 3.13 Windows.
The existing coverage threshold remains 85%. CI's installed-wheel smoke now exercises
the real worker and crash/restart isolation on Linux and Windows. Tests use synthetic
images/fake providers; no paid calls or downloaded astronomical datasets are needed.

After maintainer approval, use the existing `release.yml` workflow with an approved
`vX.Y.Z` tag containing the reviewed changes. It verifies tag/artifact version equality,
checks wheel/sdist and their installed behavior, then publishes using the configured
PyPI Trusted Publishing environment. Verify the published wheel independently before
updating aiaGUI's exact dependency pin.

## Bundled Python smoke scenario

The host bundle needs Python plus the released AIA wheel and its locked runtime
dependencies. Poetry, Git and SDK API keys are not required at runtime. The ordinary
distribution smoke installs the base wheel without provider extras and executes the
worker outside the checkout, proving manual processing independence from SDKs.

For a GUI runtime, place a small synthetic `input.fit` in an absolute writable test
directory, then run this host-side harness from a development environment:

```bash
python scripts/worker_smoke.py --python /absolute/bundle/python --directory /absolute/smoke-data
```

On Windows pass the bundled `python.exe`. The script launches the selected Python as
`-m astroagent.worker` with cwd at the selected data directory, handshakes, performs
registered normalization, verifies output/YAML/manifest, shuts down, interrupts a
fresh accepted job, and verifies a restarted process knows no old job. It does not
use repository-relative resources in the worker. The harness belongs to development
scripts and is included in the sdist, not a worker runtime dependency.

The GUI must continuously drain stdout, manage process exit/restart and retain only
current preview generations. Qt process management tests remain in aiaGUI. Exporting
completed job products and retention of incomplete workspaces are host responsibilities.

Desktop packaging remains outside AIA: Linux AppImage only; no DEB/APT/R2 workflow.
Python, Qt, GUI OpenCV libraries and dependencies are bundled by aiaGUI. Windows
DigiCert signing and Edge Ant licensing remain in aiaGUI. AIA adds no Qt/OpenCV/GPU,
new LLM framework, or licensing dependency and retains Apache-2.0/NOTICE.
