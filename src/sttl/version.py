"""Version identifiers shared by STTL artifacts and package metadata."""

# This version identifies the persisted pipeline contract. Keep it in one
# importable module so production code does not duplicate artifact versions.
PIPELINE_VERSION = "1.10"
__version__ = PIPELINE_VERSION
