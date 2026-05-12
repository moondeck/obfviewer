# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""Main application setup for OBP viewer."""

from __future__ import annotations

import tkinter
from typing import TYPE_CHECKING

from obfviewer.gui.viewer import ObpFrame

if TYPE_CHECKING:
    from obfviewer.models import Data


def run_viewer(
    data: list[Data],
    title: str = "OBF Viewer",
    build_info: dict | None = None,
    layer_numbers: list[int] | None = None,
    scan_types: list[str] | None = None,
    slice_size: int = 50000,
    index: int = 0,
) -> None:
    """Run the OBF viewer application.

    Args:
        data: List of Data objects for each layer
        title: Window title
        build_info: Optional build analysis information
        layer_numbers: List mapping each data index to its physical layer number (1-indexed)
        scan_types: List mapping each data index to its scan type (e.g., "melt", "preheat")
        slice_size: Number of paths to display at once
        index: Initial path index
    """
    root = tkinter.Tk()
    root.title(title)

    # Default to sequential 1-indexed if not provided
    if layer_numbers is None:
        layer_numbers = list(range(1, len(data) + 1))
    
    # Default scan types to "unknown" if not provided
    if scan_types is None:
        scan_types = ["unknown"] * len(data)

    num_physical_layers = max(layer_numbers) if layer_numbers else 0
    print(f"Loaded {len(data)} scans across {num_physical_layers} physical layers")

    frame = ObpFrame(
        root,
        data,
        len(data),
        build_info=build_info,
        layer_numbers=layer_numbers,
        scan_types=scan_types,
        slice_size=slice_size,
        index=index,
    )
    frame.grid(row=0, column=0, sticky="NSWE", padx=5, pady=5)
    frame.setup_grid()

    root.mainloop()
