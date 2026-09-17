"""STTL's reusable, importable building blocks.

The command-line scripts at the repository root remain supported during the
incremental package migration. New shared code belongs here so it can be
typed, tested, and reused without copying implementation details between
scripts.
"""

from .version import PIPELINE_VERSION, __version__

__all__ = ["PIPELINE_VERSION", "__version__"]
