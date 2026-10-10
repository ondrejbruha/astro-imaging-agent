"""Errors that can be presented directly to CLI users."""


class AstroError(Exception):
    """Base exception for expected input and processing errors."""


class ImageIOError(AstroError):
    """An image cannot be loaded or saved safely."""


class AnalysisError(AstroError):
    """An image cannot provide the requested measurements."""


class PipelineError(AstroError):
    """A pipeline or one of its processing steps is invalid."""


class PipelineValidationError(PipelineError):
    """Pipeline validation failure with safe one-based step/field locations."""

    def __init__(self, message: str, locations: list[dict[str, object]]) -> None:
        """Keep parameter values out of structured locations."""
        super().__init__(message)
        self.locations = locations


class ResourceLimitError(PipelineError):
    """Configured scratch, memory, or display resources cannot support the operation."""


class ProviderError(PipelineError):
    """An optional planning provider failed without a numerical processing failure."""
