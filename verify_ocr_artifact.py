#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.verify`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_verify = load_sttl_module("verify")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_verify))

sys.modules[__name__] = _verify
