from collections.abc import Iterable
from types import MappingProxyType
from typing import Any

from astroagent.errors import PipelineError
from astroagent.tools.background import BackgroundExtractTool
from astroagent.tools.base import ImageTool, ToolDescription
from astroagent.tools.denoise import DenoiseTool
from astroagent.tools.normalization import NormalizeTool
from astroagent.tools.stretch import StretchTool


class ToolRegistry:
    """An immutable per-executor registry of independently constructed tools."""

    def __init__(self, tools: Iterable[ImageTool[Any]]) -> None:
        """Reject duplicate names instead of silently replacing tool definitions."""
        entries: dict[str, ImageTool[Any]] = {}
        for tool in tools:
            if tool.name in entries:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            entries[tool.name] = tool
        self._tools = MappingProxyType(entries)

    def get(self, name: str) -> ImageTool[Any]:
        """Return a tool or a user-facing error listing available names."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise PipelineError(
                f"Unknown tool '{name}'. Available: {', '.join(self._tools)}"
            ) from exc

    def describe(self) -> list[ToolDescription]:
        """List tool descriptions and schemas in stable registration order."""
        return [tool.describe() for tool in self._tools.values()]


def default_registry() -> ToolRegistry:
    """Build a fresh registry; no global mutable tool state is shared."""
    return ToolRegistry([NormalizeTool(), StretchTool(), DenoiseTool(), BackgroundExtractTool()])
