# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for OBP viewer."""

from __future__ import annotations

import argparse
import pathlib
import sys

# Configure matplotlib before importing pyplot
import matplotlib

matplotlib.rcParams["agg.path.chunksize"] = 50000
matplotlib.rcParams["path.simplify"] = False
matplotlib.rcParams["path.simplify_threshold"] = 0.0

import matplotlib.pyplot as plt

plt.style.use("dark_background")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional list of arguments (defaults to sys.argv)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="OBF/OBP data viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "obf_file",
        type=pathlib.Path,
        help="Path to OBF archive file",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=50000,
        help="Number of paths to display at once",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Initial path index",
    )
    parser.add_argument(
        "--no-melt",
        action="store_true",
        help="Include non-melt scan files",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for the OBP viewer.

    Args:
        args: Optional list of arguments

    Returns:
        Exit code (0 for success)
    """
    parsed = parse_args(args)

    # Import here to avoid slow startup for --help
    import obanalyser.analyse_build as analyse_build

    from obfviewer.gui.app import run_viewer
    from obfviewer.loaders import load_obp_files_parallel
    from obfviewer.utils import extract_obf_archive, get_layer_sequence_with_info

    # Validate input file
    if not parsed.obf_file.exists():
        print(f"Error: File not found: {parsed.obf_file}", file=sys.stderr)
        return 1

    # Extract archive
    print(f"Extracting {parsed.obf_file.name}...")
    temp_dir = extract_obf_archive(parsed.obf_file)

    # Get build analysis
    build_info_path = temp_dir / "buildInfo.json"
    analysis = analyse_build.analyse_build(str(build_info_path))

    # Get layer sequence with physical layer numbers and scan types
    # When --no-melt is not set, default to melt_only=True (only melt scans).
    # When --no-melt is set, include all scan types.
    melt_only = not parsed.no_melt
    layer_sequence, layer_numbers, scan_types = get_layer_sequence_with_info(temp_dir, melt_only=melt_only)

    if not layer_sequence:
        print("No .obp files found in the provided .obf archive.", file=sys.stderr)
        return 1

    num_physical_layers = max(layer_numbers) if layer_numbers else 0
    print(f"Found {len(layer_sequence)} .obp files across {num_physical_layers} physical layers")

    # Load data
    data = load_obp_files_parallel(layer_sequence)

    # Run viewer with layer number mapping and scan types
    run_viewer(
        data,
        title=f"OBF Viewer - {parsed.obf_file.name}",
        build_info=analysis,
        layer_numbers=layer_numbers,
        scan_types=scan_types,
        slice_size=parsed.slice_size,
        index=parsed.index,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
