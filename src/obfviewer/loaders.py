# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""OBP file loading functions."""

from __future__ import annotations

import gzip
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Generator, Iterable

import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
from matplotlib.path import Path
from obplib import OBP_pb2 as obp
from rich.progress import Progress

from obfviewer.models import Data, TimedPoint


def load_obp_objects(filepath: pathlib.Path) -> Generator:
    """Load OBP objects from a file.

    Args:
        filepath: Path to the OBP file (can be .obp or .obp.gz)

    Yields:
        OBP packet payloads
    """
    # Determine if we need to decompress
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rb") as fh:
            data = fh.read()
    else:
        with open(filepath, "rb") as fh:
            data = fh.read()

    # Cache the Packet class to avoid repeated lookups
    packet_class = obp.Packet
    data_len = len(data)
    consumed = 0

    while consumed < data_len:
        msg_len, new_pos = _DecodeVarint32(data, consumed)
        msg_end = new_pos + msg_len
        msg_buf = data[new_pos:msg_end]
        consumed = msg_end

        packet = packet_class()
        packet.ParseFromString(msg_buf)
        attr = packet.WhichOneof("payload")
        yield getattr(packet, attr)


def merge_obp_objects(obp_objects: Iterable) -> list:
    """Merge consecutive TimedPoints objects.

    Args:
        obp_objects: Iterable of OBP objects

    Returns:
        List with merged TimedPoints
    """
    merged = []
    for obj in obp_objects:
        if isinstance(obj, obp.TimedPoints):
            if merged and isinstance(merged[-1], obp.TimedPoints):
                merged[-1].points.extend(obj.points)
            else:
                merged.append(obj)
        else:
            merged.append(obj)
    return merged


def _unpack_tp(obp_objects: Iterable) -> Generator:
    """Unpack TimedPoints into individual TimedPoint objects.

    Args:
        obp_objects: Iterable of OBP objects

    Yields:
        Individual OBP objects with TimedPoints expanded
    """
    for obj in obp_objects:
        if isinstance(obj, obp.TimedPoints):
            t = 0
            for point in obj.points:
                tp = TimedPoint()
                tp.x = point.x
                tp.y = point.y
                if point.t == 0:
                    point.t = t
                tp.t = t = point.t
                tp.params = obj.params
                yield tp
        else:
            yield obj


# Pre-computed path codes (immutable, can be shared)
_LINE_CODES = np.array([Path.MOVETO, Path.LINETO], dtype=np.uint8)
_CURVE_CODES = np.array([Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4], dtype=np.uint8)
_DIAMOND_CODES = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.MOVETO], dtype=np.uint8)


def load_artist_data_fast(obp_objects: Iterable) -> Data:
    """Convert OBP objects to matplotlib-compatible artist data (optimized).

    This version collects raw vertex data first, then batch-creates paths.

    Args:
        obp_objects: Iterable of OBP objects

    Returns:
        Data object containing paths and associated metadata
    """
    # Collect raw data in lists first (faster than creating Path objects in loop)
    line_vertices: list[tuple[float, float, float, float]] = []
    curve_vertices: list[tuple[float, ...]] = []
    diamond_vertices: list[tuple[float, float]] = []

    # Track which type each path is and its index in the type-specific list
    path_order: list[tuple[str, int]] = []  # ('line', idx), ('curve', idx), ('diamond', idx)

    speeds: list[float] = []
    dwell_times: list[float] = []
    spotsizes: list[float] = []
    beampowers: list[float] = []
    restores: list[int] = []
    syncpoints: dict[str, list[int]] = {}
    _lastseen: dict[str, int] = {}
    _restore = 0

    # Cache class references for faster isinstance checks
    Line = obp.Line
    AcceleratingLine = obp.AcceleratingLine
    Curve = obp.Curve
    AcceleratingCurve = obp.AcceleratingCurve
    SyncPoint = obp.SyncPoint
    Restore = obp.Restore

    scale = 1e-6  # Pre-compute scale factor

    for obj in _unpack_tp(obp_objects):
        if isinstance(obj, (Line, AcceleratingLine)):
            line_vertices.append((obj.x0, obj.y0, obj.x1, obj.y1))
            path_order.append(('L', len(line_vertices) - 1))
            speeds.append(obj.speed * scale if isinstance(obj, Line) else obj.sf)
            dwell_times.append(getattr(obj.params, "dwell_time", 0) * scale)  # Convert μs to s

        elif isinstance(obj, (Curve, AcceleratingCurve)):
            curve_vertices.append((
                obj.p0.x, obj.p0.y,
                obj.p1.x, obj.p1.y,
                obj.p2.x, obj.p2.y,
                obj.p3.x, obj.p3.y,
            ))
            path_order.append(('C', len(curve_vertices) - 1))
            speeds.append(obj.speed * scale if isinstance(obj, Curve) else obj.sf)
            dwell_times.append(getattr(obj.params, "dwell_time", 0) * scale)  # Convert μs to s

        elif isinstance(obj, TimedPoint):
            diamond_vertices.append((obj.x, obj.y))
            path_order.append(('D', len(diamond_vertices) - 1))
            speeds.append(0)
            dwell_times.append(obj.t * scale)  # Already in μs, convert to s

        elif isinstance(obj, SyncPoint):
            if obj.endpoint not in syncpoints:
                syncpoints[obj.endpoint] = [0] * len(path_order)
            _lastseen[obj.endpoint] = int(obj.value)
            continue

        elif isinstance(obj, Restore):
            _restore = 1
            continue

        else:
            continue

        spotsizes.append(obj.params.spot_size)
        beampowers.append(obj.params.beam_power)
        for k, v in _lastseen.items():
            syncpoints[k].append(v)
        restores.append(_restore)
        _restore = 0

    # Batch create numpy arrays for vertices
    if line_vertices:
        line_arr = np.array(line_vertices, dtype=np.float64) * scale
    else:
        line_arr = np.empty((0, 4), dtype=np.float64)

    if curve_vertices:
        curve_arr = np.array(curve_vertices, dtype=np.float64).reshape(-1, 4, 2) * scale
    else:
        curve_arr = np.empty((0, 4, 2), dtype=np.float64)

    if diamond_vertices:
        diamond_arr = np.array(diamond_vertices, dtype=np.float64) * scale
    else:
        diamond_arr = np.empty((0, 2), dtype=np.float64)

    # Pre-build all paths by type, then reorder
    line_paths = [
        Path(line_arr[i].reshape(2, 2), _LINE_CODES, readonly=True)
        for i in range(len(line_vertices))
    ]

    curve_paths = [
        Path(curve_arr[i], _CURVE_CODES, readonly=True)
        for i in range(len(curve_vertices))
    ]

    # Diamond paths need vertex expansion
    diamond_paths = []
    for i in range(len(diamond_vertices)):
        x, y = diamond_arr[i]
        d = 100 * scale
        verts = np.array([
            [x - d, y],
            [x, y + d],
            [x + d, y],
            [x, y - d],
            [x - d, y],
            [x, y],
        ], dtype=np.float64)
        diamond_paths.append(Path(verts, _DIAMOND_CODES, readonly=True))

    # Reconstruct paths in original order
    path_lookup = {'L': line_paths, 'C': curve_paths, 'D': diamond_paths}
    paths = [path_lookup[ptype][idx] for ptype, idx in path_order]

    # Convert syncpoints lists to numpy arrays
    syncpoints_np = {key: np.array(val, dtype=np.int32) for key, val in syncpoints.items()}

    return Data(
        paths=paths,
        speeds=np.array(speeds, dtype=np.float64),
        dwell_times=np.array(dwell_times, dtype=np.float64),
        spotsizes=np.array(spotsizes, dtype=np.float64),
        beampowers=np.array(beampowers, dtype=np.float64),
        syncpoints=syncpoints_np,
        restores=np.array(restores, dtype=np.int8),
    )


def load_obp_raw_geometry(filepath: pathlib.Path) -> dict:
    """Load OBP file and extract raw geometry (no Path objects).
    
    Much faster than load_artist_data_fast since it skips expensive
    matplotlib Path object creation. Returns raw vertices directly.
    
    Args:
        filepath: Path to the OBP file (can be .obp or .obp.gz)
    
    Returns:
        Dict with keys:
        - 'lines': list of [x0, y0, x1, y1, speed] 
        - 'spots': list of [x, y, t]
        - 'spotsizes': numpy array of spot sizes
        - 'beampowers': numpy array of beam powers
    """
    obp_objects = load_obp_objects(filepath)
    
    # Cache class references for faster isinstance checks
    Line = obp.Line
    AcceleratingLine = obp.AcceleratingLine
    Curve = obp.Curve
    AcceleratingCurve = obp.AcceleratingCurve
    
    scale = 1e-6  # Pre-compute scale factor
    
    lines = []
    spots = []
    spotsizes = []
    beampowers = []
    
    for obj in _unpack_tp(obp_objects):
        if isinstance(obj, (Line, AcceleratingLine)):
            # Extract line endpoints directly
            speed = obj.speed * scale if isinstance(obj, Line) else obj.sf
            lines.append((obj.x0, obj.y0, obj.x1, obj.y1, speed))
            spotsizes.append(obj.params.spot_size)
            beampowers.append(obj.params.beam_power)
            
        elif isinstance(obj, (Curve, AcceleratingCurve)):
            # For curves, use start and end points
            speed = obj.speed * scale if isinstance(obj, Curve) else obj.sf
            lines.append((obj.p0.x, obj.p0.y, obj.p3.x, obj.p3.y, speed))
            spotsizes.append(obj.params.spot_size)
            beampowers.append(obj.params.beam_power)
            
        elif isinstance(obj, TimedPoint):
            # Store timed points as spots
            spots.append((obj.x, obj.y, obj.t))
    
    return {
        'lines': lines,
        'spots': spots,
        'spotsizes': np.array(spotsizes, dtype=np.float64),
        'beampowers': np.array(beampowers, dtype=np.float64),
    }


def load_obp_unified(filepath: pathlib.Path) -> tuple[Data, dict]:
    """Load OBP file ONCE and return both Data (for 2D) and raw geometry (for 3D).
    
    This avoids parsing the same file twice.
    
    Args:
        filepath: Path to the OBP file
        
    Returns:
        Tuple of (Data object for 2D, raw geometry dict for 3D)
    """
    # Read file once
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rb") as fh:
            file_data = fh.read()
    else:
        with open(filepath, "rb") as fh:
            file_data = fh.read()
    
    # Parse once, build both outputs
    scale = 1e-6
    
    # For Data (2D)
    line_vertices = []
    curve_vertices = []
    diamond_vertices = []
    path_order = []
    speeds = []
    dwell_times = []
    spotsizes = []
    beampowers = []
    restores = []
    syncpoints = {}
    _lastseen = {}
    _restore = 0
    
    # For raw geometry (3D)
    raw_lines = []
    raw_spots = []
    
    # Cache classes
    Line = obp.Line
    AcceleratingLine = obp.AcceleratingLine
    Curve = obp.Curve
    AcceleratingCurve = obp.AcceleratingCurve
    SyncPoint = obp.SyncPoint
    Restore = obp.Restore
    
    packet = obp.Packet()
    consumed = 0
    data_len = len(file_data)
    
    while consumed < data_len:
        msg_len, new_pos = _DecodeVarint32(file_data, consumed)
        packet.ParseFromString(file_data[new_pos:new_pos + msg_len])
        consumed = new_pos + msg_len
        
        attr = packet.WhichOneof("payload")
        obj = getattr(packet, attr)
        
        if isinstance(obj, (Line, AcceleratingLine)):
            line_vertices.append((obj.x0, obj.y0, obj.x1, obj.y1))
            path_order.append(('L', len(line_vertices) - 1))
            speed = obj.speed * scale if isinstance(obj, Line) else obj.sf
            speeds.append(speed)
            dwell_times.append(getattr(obj.params, "dwell_time", 0) * scale)
            spotsizes.append(obj.params.spot_size)
            beampowers.append(obj.params.beam_power)
            raw_lines.append((obj.x0, obj.y0, obj.x1, obj.y1, speed))
            for k, v in _lastseen.items():
                syncpoints[k].append(v)
            restores.append(_restore)
            _restore = 0
            
        elif isinstance(obj, (Curve, AcceleratingCurve)):
            curve_vertices.append((obj.p0.x, obj.p0.y, obj.p1.x, obj.p1.y, 
                                   obj.p2.x, obj.p2.y, obj.p3.x, obj.p3.y))
            path_order.append(('C', len(curve_vertices) - 1))
            speed = obj.speed * scale if isinstance(obj, Curve) else obj.sf
            speeds.append(speed)
            dwell_times.append(getattr(obj.params, "dwell_time", 0) * scale)
            spotsizes.append(obj.params.spot_size)
            beampowers.append(obj.params.beam_power)
            raw_lines.append((obj.p0.x, obj.p0.y, obj.p3.x, obj.p3.y, speed))
            for k, v in _lastseen.items():
                syncpoints[k].append(v)
            restores.append(_restore)
            _restore = 0
            
        elif isinstance(obj, obp.TimedPoints):
            t = 0
            for point in obj.points:
                if point.t == 0:
                    point.t = t
                t = point.t
                diamond_vertices.append((point.x, point.y))
                path_order.append(('D', len(diamond_vertices) - 1))
                speeds.append(0)
                dwell_times.append(t * scale)
                spotsizes.append(obj.params.spot_size)
                beampowers.append(obj.params.beam_power)
                raw_spots.append((point.x, point.y, t))
                for k, v in _lastseen.items():
                    syncpoints[k].append(v)
                restores.append(_restore)
                _restore = 0
                
        elif isinstance(obj, SyncPoint):
            if obj.endpoint not in syncpoints:
                syncpoints[obj.endpoint] = [0] * len(path_order)
            _lastseen[obj.endpoint] = int(obj.value)
            
        elif isinstance(obj, Restore):
            _restore = 1
    
    # Build Data object paths
    if line_vertices:
        line_arr = np.array(line_vertices, dtype=np.float64) * scale
    else:
        line_arr = np.empty((0, 4), dtype=np.float64)
    
    if curve_vertices:
        curve_arr = np.array(curve_vertices, dtype=np.float64).reshape(-1, 4, 2) * scale
    else:
        curve_arr = np.empty((0, 4, 2), dtype=np.float64)
    
    if diamond_vertices:
        diamond_arr = np.array(diamond_vertices, dtype=np.float64) * scale
    else:
        diamond_arr = np.empty((0, 2), dtype=np.float64)
    
    line_paths = [Path(line_arr[i].reshape(2, 2), _LINE_CODES, readonly=True)
                  for i in range(len(line_vertices))]
    curve_paths = [Path(curve_arr[i], _CURVE_CODES, readonly=True)
                   for i in range(len(curve_vertices))]
    
    diamond_paths = []
    d = 100 * scale
    for i in range(len(diamond_vertices)):
        x, y = diamond_arr[i]
        verts = np.array([[x-d, y], [x, y+d], [x+d, y], [x, y-d], [x-d, y], [x, y]], dtype=np.float64)
        diamond_paths.append(Path(verts, _DIAMOND_CODES, readonly=True))
    
    path_lookup = {'L': line_paths, 'C': curve_paths, 'D': diamond_paths}
    paths = [path_lookup[ptype][idx] for ptype, idx in path_order]
    
    syncpoints_np = {key: np.array(val, dtype=np.int32) for key, val in syncpoints.items()}
    
    data_obj = Data(
        paths=paths,
        speeds=np.array(speeds, dtype=np.float64),
        dwell_times=np.array(dwell_times, dtype=np.float64),
        spotsizes=np.array(spotsizes, dtype=np.float64),
        beampowers=np.array(beampowers, dtype=np.float64),
        syncpoints=syncpoints_np,
        restores=np.array(restores, dtype=np.int8),
    )
    
    raw_geom = {
        'lines': raw_lines,
        'spots': raw_spots,
        'spotsizes': np.array(spotsizes, dtype=np.float64),
        'beampowers': np.array(beampowers, dtype=np.float64),
    }
    
    return data_obj, raw_geom


# Keep old function as alias
load_artist_data = load_artist_data_fast


def load_obp_worker(file: pathlib.Path) -> Data:
    """Worker function to load a single OBP file.

    Args:
        file: Path to the OBP file

    Returns:
        Loaded Data object
    """
    obp_objects = load_obp_objects(file)
    return load_artist_data(obp_objects)


def load_obp_files_parallel(
    layer_sequence: list[pathlib.Path],
    show_progress: bool = True,
    max_workers: int | None = None,
) -> list[Data]:
    """Load OBP files in parallel, caching repeated files.

    This function optimizes loading by identifying unique files and loading
    each only once. Repeated files in the sequence will reference the same
    Data object.

    Uses ProcessPoolExecutor for true parallelism since protobuf parsing
    is CPU-bound.

    Args:
        layer_sequence: List of paths to OBP files (may contain duplicates)
        show_progress: Whether to show a progress bar
        max_workers: Maximum number of parallel workers (None = CPU count)

    Returns:
        List of Data objects in the same order as input
    """
    if not layer_sequence:
        return []

    # Identify unique files to avoid loading duplicates
    unique_files = list(dict.fromkeys(layer_sequence))  # Preserves order
    unique_count = len(unique_files)
    total_count = len(layer_sequence)
    skipped = total_count - unique_count

    if show_progress:
        print(f"Loading {unique_count} unique files ({skipped} duplicates skipped)")

    # Load only unique files
    cache: dict[pathlib.Path, Data] = {}

    # Use ProcessPoolExecutor for CPU-bound protobuf parsing (GIL limits ThreadPool)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_obp_worker, file): file
            for file in unique_files
        }

        if show_progress:
            with Progress() as progress:
                task = progress.add_task("Loading layers...", total=unique_count)
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        cache[file] = future.result()
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        raise
                    progress.update(task, advance=1)
        else:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                cache[file] = future.result()

    # Build the full sequence from cache
    return [cache[file] for file in layer_sequence]


def merge_data(data_list: list[Data]) -> Data:
    """Merge multiple Data objects into one.

    This combines all paths, speeds, etc. from multiple scan files into
    a single Data object representing a complete layer.

    Args:
        data_list: List of Data objects to merge

    Returns:
        Single merged Data object
    """
    if not data_list:
        return Data(
            paths=[],
            speeds=np.array([], dtype=np.float64),
            dwell_times=np.array([], dtype=np.float64),
            spotsizes=np.array([], dtype=np.float64),
            beampowers=np.array([], dtype=np.float64),
            syncpoints={},
            restores=np.array([], dtype=np.int8),
        )

    if len(data_list) == 1:
        return data_list[0]

    # Merge all arrays
    all_paths = []
    all_speeds = []
    all_dwell_times = []
    all_spotsizes = []
    all_beampowers = []
    all_restores = []

    # Collect all syncpoint keys
    all_syncpoint_keys: set[str] = set()
    for data in data_list:
        all_syncpoint_keys.update(data.syncpoints.keys())

    # Initialize syncpoints dict
    merged_syncpoints: dict[str, list] = {key: [] for key in all_syncpoint_keys}

    for data in data_list:
        all_paths.extend(data.paths)
        all_speeds.append(data.speeds)
        all_dwell_times.append(data.dwell_times)
        all_spotsizes.append(data.spotsizes)
        all_beampowers.append(data.beampowers)
        all_restores.append(data.restores)

        # Handle syncpoints - fill missing keys with zeros
        n_paths = len(data.paths)
        for key in all_syncpoint_keys:
            if key in data.syncpoints:
                merged_syncpoints[key].extend(data.syncpoints[key].tolist())
            else:
                merged_syncpoints[key].extend([0] * n_paths)

    return Data(
        paths=all_paths,
        speeds=np.concatenate(all_speeds) if all_speeds else np.array([], dtype=np.float64),
        dwell_times=np.concatenate(all_dwell_times) if all_dwell_times else np.array([], dtype=np.float64),
        spotsizes=np.concatenate(all_spotsizes) if all_spotsizes else np.array([], dtype=np.float64),
        beampowers=np.concatenate(all_beampowers) if all_beampowers else np.array([], dtype=np.float64),
        syncpoints={k: np.array(v, dtype=np.int32) for k, v in merged_syncpoints.items()},
        restores=np.concatenate(all_restores) if all_restores else np.array([], dtype=np.int8),
    )


def load_grouped_layers_parallel(
    grouped_layers: list[list[pathlib.Path]],
    show_progress: bool = True,
    max_workers: int | None = None,
) -> list[Data]:
    """Load grouped layer files in parallel, merging scans within each layer.

    This function loads all unique files, then merges the data for files
    that belong to the same layer.

    Args:
        grouped_layers: List of lists, where each inner list contains
                       all OBP files for one physical layer
        show_progress: Whether to show a progress bar
        max_workers: Maximum number of parallel workers

    Returns:
        List of Data objects, one per physical layer
    """
    if not grouped_layers:
        return []

    # Flatten to get all unique files
    all_files = [file for layer_files in grouped_layers for file in layer_files]
    unique_files = list(dict.fromkeys(all_files))

    unique_count = len(unique_files)
    total_files = len(all_files)
    layer_count = len(grouped_layers)

    if show_progress:
        print(
            f"Loading {unique_count} unique files "
            f"({total_files - unique_count} duplicates) "
            f"for {layer_count} layers"
        )

    # Load all unique files
    cache: dict[pathlib.Path, Data] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_obp_worker, file): file
            for file in unique_files
        }

        if show_progress:
            with Progress() as progress:
                task = progress.add_task("Loading files...", total=unique_count)
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        cache[file] = future.result()
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        raise
                    progress.update(task, advance=1)
        else:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                cache[file] = future.result()

    # Merge files for each layer
    merged_layers: list[Data] = []
    for layer_files in grouped_layers:
        layer_data = [cache[file] for file in layer_files]
        merged_layers.append(merge_data(layer_data))

    if show_progress:
        print(f"Merged into {len(merged_layers)} layers")

    return merged_layers
