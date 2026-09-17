"""Import support for legacy root-level commands in a source checkout.

The repository historically used ``sttl/`` as a virtual-environment directory.
That directory has no ``__init__.py`` and can therefore be discovered as a
namespace package before the new ``src/sttl`` package.  Root command shims use
this loader so they remain usable before an editable installation is made.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_sttl_module(module_name: str) -> ModuleType:
    """Load an STTL module, preferring the package in this source checkout."""
    source_root = Path(__file__).resolve().parent / "src"
    if source_root.is_dir():
        source_root_text = str(source_root)
        if source_root_text not in sys.path:
            sys.path.insert(0, source_root_text)

        # Discard only the legacy namespace package, never an installed proper
        # package.  This handles callers which imported ``sttl`` before a shim.
        package = sys.modules.get("sttl")
        if package is not None and getattr(package, "__file__", None) is None:
            del sys.modules["sttl"]

    return importlib.import_module(f"sttl.{module_name}")


def run_sttl_main(module: ModuleType) -> int:
    """Run a legacy module's main function with process-safe return semantics."""
    result = module.main()
    return result if isinstance(result, int) else 0
