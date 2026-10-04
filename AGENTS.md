# Project instructions

- Never create a Git commit or push changes unless the user explicitly changes this instruction.
- Prepare completed changes in the Git staging area. Preserve unrelated existing edits.
- Use Poetry for dependency management and execution; do not introduce uv.
- Write meaningful tests for new behavior and bug fixes, using small synthetic images.
- Run pytest, Ruff formatting/linting, mypy, and package build checks before handing off work.
- Keep image tools, analysis, pipeline execution, CLI, and planning independent.
- A saved PipelineDefinition must execute without an agent or an LLM provider.
- Processing must be deterministic and must preserve relevant FITS metadata.
- Document public APIs and behavior, including numerical limitations.
- Do not add GPU processing, external LLM integrations, or unrelated dependencies without a task requiring them.
