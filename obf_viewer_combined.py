#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Olgierd Nowakowski
# SPDX-License-Identifier: Apache-2.0

"""Convenience entry-point for running the combined viewer directly.

Usage (no install required, but the package must be on PYTHONPATH):
    python obf_viewer_combined.py path/to/build.obf [options]

For installed usage run:
    obfviewer-combined path/to/build.obf [options]
"""

import sys

# Ensure src/ is importable when executed directly from the project root
from pathlib import Path
_src = Path(__file__).parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from obfviewer.combined import main

if __name__ == "__main__":
    sys.exit(main())
