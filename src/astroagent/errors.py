"""Errors that can be presented directly to CLI users."""


class AstroError(Exception):
    """Base exception for expected input and processing errors."""


class ImageIOError(AstroError):
    """An image cannot be loaded or saved safely."""


class AnalysisError(AstroError):
    """An image cannot provide the requested measurements."""


class PipelineError(AstroError):
    """A pipeline or one of its processing steps is invalid."""
