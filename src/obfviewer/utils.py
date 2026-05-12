# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for OBP viewer."""

from __future__ import annotations

import atexit
import os
import pathlib
import shutil
import tempfile
import zipfile
from typing import TYPE_CHECKING

import obanalyser.get_build_order as get_build_order

if TYPE_CHECKING:
    pass


def extract_obf_archive(obf_path: pathlib.Path) -> pathlib.Path:
    """Extract an OBF archive to a temporary directory.

    The temporary directory will be automatically cleaned up when the
    program exits.

    Args:
        obf_path: Path to the OBF file

    Returns:
        Path to the temporary directory containing extracted files
    """
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="obfviewer_"))

    # Register cleanup on exit
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    with zipfile.ZipFile(obf_path, "r") as zip_ref:
        # Guard against Zip Slip: reject members with path-traversal components.
        dest = temp_dir.resolve()
        for member in zip_ref.infolist():
            member_path = (dest / member.filename).resolve()
            if not str(member_path).startswith(str(dest) + os.sep) and member_path != dest:
                raise ValueError(f"Unsafe path in archive: {member.filename!r}")
        zip_ref.extractall(dest)

    return temp_dir


def get_layer_sequence(
    temp_dir: pathlib.Path,
    melt_only: bool = True,
) -> list[pathlib.Path]:
    """Get the sequence of OBP layer files from a build.

    Args:
        temp_dir: Path to the extracted OBF directory
        melt_only: If True, only include melt files

    Returns:
        List of paths to OBP files in execution order
    """
    build_info_path = temp_dir / "buildInfo.json"
    sequence = get_build_order.get_layer_execution_sequence(str(build_info_path))

    layer_sequence: list[pathlib.Path] = []

    for item in sequence[0]:
        layers = item
        for layer in layers:
            scanpaths = layer
            # scanpaths is a tuple of (path, repetitions), not a list
            (obp_path, repetitions) = scanpaths
            obp_path = pathlib.Path(obp_path)

            if melt_only and _extract_scan_type(obp_path.name) != "melt":
                continue

            for _ in range(repetitions):
                layer_sequence.append(obp_path)

    return layer_sequence


def _extract_scan_type(filename: str) -> str:
    """Extract the scan type from a filename.
    
    Args:
        filename: Name of the OBP file (e.g., "layer199melt1.obp" or "layer199.obp")
        
    Returns:
        Scan type string (e.g., "melt", "preheat", "jumpsafe", "heatBalance", "idle")
    """
    import re
    name_lower = filename.lower()
    
    # Check for specific scan type keywords FIRST (before "melt")
    # Order matters: "idle" must come before "melt" to handle "PostMelt_IdleScan"
    if "idle" in name_lower:
        return "idle"
    if "jumpsafe" in name_lower or "jump" in name_lower:
        return "jumpsafe"
    if "preheat" in name_lower:
        return "preheat"
    if "heatbalance" in name_lower or "balance" in name_lower:
        return "heatBalance"
    if "bse" in name_lower:
        return "BSE"
    if "measure" in name_lower:
        return "measure"
    
    # "melt" in name = melt
    if "melt" in name_lower:
        return "melt"
    
    # Default: files like "layer199.obp" with no type keyword are melt files
    if re.match(r"^layer\d+\.obp$", name_lower):
        return "melt"
    
    print(f"Warning: Unknown scan type in filename '{filename}'")
    return "unknown"


def get_layer_sequence_with_info(
    temp_dir: pathlib.Path,
    melt_only: bool = True,
) -> tuple[list[pathlib.Path], list[int], list[str]]:
    """Get the sequence of OBP layer files with their physical layer numbers and scan types.

    Args:
        temp_dir: Path to the extracted OBF directory
        melt_only: If True, only include melt files

    Returns:
        Tuple of (file_paths, layer_numbers, scan_types) where:
        - layer_numbers[i] is the 1-indexed physical layer number that file_paths[i] belongs to.
        - scan_types[i] is the scan type name (e.g., "melt", "preheat", "jumpsafe")
    """
    build_info_path = temp_dir / "buildInfo.json"
    sequence = get_build_order.get_layer_execution_sequence(str(build_info_path))

    layer_sequence: list[pathlib.Path] = []
    layer_numbers: list[int] = []
    scan_types: list[str] = []

    physical_layer = 0  # Will be incremented to 1 for first layer

    for item in sequence[0]:
        physical_layer += 1  # Each 'item' is a physical layer (1-indexed)
        
        layers = item
        for layer in layers:
            scanpaths = layer
            # scanpaths is a tuple of (path, repetitions), not a list
            (obp_path, repetitions) = scanpaths
            obp_path = pathlib.Path(obp_path)

            scan_type = _extract_scan_type(obp_path.name)
            if melt_only and scan_type != "melt":
                continue

            for _ in range(repetitions):
                layer_sequence.append(obp_path)
                layer_numbers.append(physical_layer)
                scan_types.append(scan_type)

    return layer_sequence, layer_numbers, scan_types


def get_grouped_layer_sequence(
    temp_dir: pathlib.Path,
    melt_only: bool = True,
) -> list[list[pathlib.Path]]:
    """Get the sequence of OBP layer files grouped by layer.

    Unlike get_layer_sequence, this groups all scans (jumpsafe, preheat, melt)
    within a layer together, so each "layer" in the UI represents a physical
    build layer rather than individual scan passes.

    Args:
        temp_dir: Path to the extracted OBF directory
        melt_only: If True, only include melt files within each layer

    Returns:
        List of lists, where each inner list contains all OBP files for one layer
    """
    build_info_path = temp_dir / "buildInfo.json"
    sequence = get_build_order.get_layer_execution_sequence(str(build_info_path))

    grouped_layers: list[list[pathlib.Path]] = []

    for item in sequence[0]:
        # Each 'item' represents one physical layer
        layer_files: list[pathlib.Path] = []

        layers = item
        for layer in layers:
            scanpaths = layer
            # scanpaths is a tuple of (path, repetitions), not a list
            (obp_path, repetitions) = scanpaths
            obp_path = pathlib.Path(obp_path)

            if melt_only and _extract_scan_type(obp_path.name) != "melt":
                continue

            for _ in range(repetitions):
                layer_files.append(obp_path)

        # Only add if this layer has any matching files
        if layer_files:
            grouped_layers.append(layer_files)

    return grouped_layers
