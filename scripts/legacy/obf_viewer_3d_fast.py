#!/usr/bin/env python3
"""ULTRA-FAST 3D visualization for OBF/OBP build files.

Extreme optimizations:
- Single unified mesh for entire build (no per-layer overhead)
- Batch all geometry into one VTK call
- Parallel protobuf parsing with minimal allocations
- numpy vectorized operations throughout
- Layer visibility via scalar filtering (not actor toggling)

Usage:
    python obf_viewer_3d_fast.py path/to/file.obf
"""

from __future__ import annotations

import argparse
import gzip
import mmap
import pathlib
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def parse_obp_ultra_fast(filepath: pathlib.Path, z: float, scale: float, max_elements: int):
    """Ultra-fast OBP parser - returns raw numpy arrays only."""
    from google.protobuf.internal.decoder import _DecodeVarint32
    from obplib import OBP_pb2 as obp
    
    try:
        if str(filepath).endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
    except:
        return None
    
    # Pre-allocate with estimated size (will trim later)
    est_size = len(data) // 50  # rough estimate
    lines = np.zeros((est_size, 7), dtype=np.float32)  # x0,y0,x1,y1,z,speed,layer
    spots = np.zeros((est_size, 4), dtype=np.float32)  # x,y,z,speed
    
    line_idx = 0
    spot_idx = 0
    pscale = 1e-6 * scale
    
    packet = obp.Packet()
    consumed = 0
    data_len = len(data)
    
    while consumed < data_len:
        msg_len, new_pos = _DecodeVarint32(data, consumed)
        packet.ParseFromString(data[new_pos:new_pos + msg_len])
        consumed = new_pos + msg_len
        
        attr = packet.WhichOneof("payload")
        
        if attr == "line":
            obj = packet.line
            if line_idx < est_size:
                lines[line_idx] = (obj.x0*pscale, obj.y0*pscale, obj.x1*pscale, obj.y1*pscale, z, obj.speed*1e-6, z)
                line_idx += 1
        elif attr == "accelerating_line":
            obj = packet.accelerating_line
            if line_idx < est_size:
                lines[line_idx] = (obj.x0*pscale, obj.y0*pscale, obj.x1*pscale, obj.y1*pscale, z, obj.sf, z)
                line_idx += 1
        elif attr == "curve":
            obj = packet.curve
            if line_idx < est_size:
                lines[line_idx] = (obj.p0.x*pscale, obj.p0.y*pscale, obj.p3.x*pscale, obj.p3.y*pscale, z, obj.speed*1e-6, z)
                line_idx += 1
        elif attr == "accelerating_curve":
            obj = packet.accelerating_curve
            if line_idx < est_size:
                lines[line_idx] = (obj.p0.x*pscale, obj.p0.y*pscale, obj.p3.x*pscale, obj.p3.y*pscale, z, obj.sf, z)
                line_idx += 1
        elif attr == "timed_points":
            for pt in packet.timed_points.points:
                if spot_idx < est_size:
                    spots[spot_idx] = (pt.x*pscale, pt.y*pscale, z, pt.t*1e-6 if pt.t else 0)
                    spot_idx += 1
    
    # Trim and sample
    lines = lines[:line_idx]
    spots = spots[:spot_idx]
    
    if len(lines) > max_elements:
        idx = np.linspace(0, len(lines)-1, max_elements, dtype=np.int32)
        lines = lines[idx]
    
    if len(spots) > max_elements:
        idx = np.linspace(0, len(spots)-1, max_elements, dtype=np.int32)
        spots = spots[idx]
    
    return lines, spots


def worker(args):
    """Process one layer file."""
    filepath, z, scale, max_elem = args
    return parse_obp_ultra_fast(filepath, z, scale, max_elem)


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast 3D OBF viewer")
    parser.add_argument("obf_file", type=pathlib.Path)
    parser.add_argument("--layer-height", type=float, default=70.0)
    parser.add_argument("--max-per-file", type=int, default=300, help="Max lines per file")
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--all-scans", action="store_true")
    args = parser.parse_args()
    
    if not args.obf_file.exists():
        print(f"File not found: {args.obf_file}")
        return 1
    
    # Extract
    print(f"Extracting {args.obf_file.name}...")
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="obf_"))
    with zipfile.ZipFile(args.obf_file) as zf:
        zf.extractall(temp_dir)
    
    # Find files
    layer_pattern = re.compile(r"layer(\d+)", re.IGNORECASE)
    obp_files = sorted(temp_dir.rglob("*.obp"))
    
    work = []
    layer_height_mm = args.layer_height * 1e-3
    scale = 1e3  # m to mm
    
    for f in obp_files:
        name = f.stem.lower()
        if not args.all_scans:
            if "idle" in name or "melt" not in name:
                continue
        match = layer_pattern.search(f.stem)
        if match:
            layer_num = int(match.group(1))
            z = layer_num * layer_height_mm
            work.append((f, z, scale, args.max_per_file))
    
    if not work:
        print("No matching files found")
        return 1
    
    print(f"Loading {len(work)} files in parallel...")
    
    # Parallel load - collect all results
    all_lines = []
    all_spots = []
    
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(worker, w): i for i, w in enumerate(work)}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                lines, spots = result
                if len(lines) > 0:
                    all_lines.append(lines)
                if len(spots) > 0:
                    all_spots.append(spots)
            done += 1
            print(f"\r  {done}/{len(work)}", end="", flush=True)
    
    print()
    
    # Concatenate all data into single arrays
    print("Building unified mesh...")
    
    if all_lines:
        all_lines = np.vstack(all_lines)
    else:
        all_lines = np.zeros((0, 7), dtype=np.float32)
    
    if all_spots:
        all_spots = np.vstack(all_spots)
    else:
        all_spots = np.zeros((0, 4), dtype=np.float32)
    
    n_lines = len(all_lines)
    n_spots = len(all_spots)
    print(f"  {n_lines} lines, {n_spots} spots")
    
    # Build VTK geometry - ONE mesh for everything
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    
    meshes = []
    
    if n_lines > 0:
        # Build line mesh: 2 points per line
        points = np.zeros((n_lines * 2, 3), dtype=np.float32)
        points[0::2, 0] = all_lines[:, 0]  # x0
        points[0::2, 1] = all_lines[:, 1]  # y0
        points[0::2, 2] = all_lines[:, 4]  # z
        points[1::2, 0] = all_lines[:, 2]  # x1
        points[1::2, 1] = all_lines[:, 3]  # y1
        points[1::2, 2] = all_lines[:, 4]  # z
        
        # Line connectivity
        cells = np.zeros(n_lines * 3, dtype=np.int64)
        cells[0::3] = 2
        cells[1::3] = np.arange(0, n_lines * 2, 2)
        cells[2::3] = np.arange(1, n_lines * 2, 2)
        
        mesh = pv.PolyData(points)
        mesh.lines = cells
        mesh["speed"] = np.repeat(all_lines[:, 5], 2)
        mesh["layer"] = np.repeat(all_lines[:, 6], 2)
        meshes.append(("lines", mesh))
        
        # Free memory
        del all_lines, points, cells
    
    if n_spots > 0:
        points = np.zeros((n_spots, 3), dtype=np.float32)
        points[:, 0] = all_spots[:, 0]
        points[:, 1] = all_spots[:, 1]
        points[:, 2] = all_spots[:, 2]
        
        mesh = pv.PolyData(points)
        mesh["speed"] = all_spots[:, 3]
        mesh["layer"] = all_spots[:, 2]  # z = layer height
        meshes.append(("spots", mesh))
        
        del all_spots, points
    
    if not meshes:
        print("No geometry to display")
        return 1
    
    # Get layer range from data
    all_z = []
    for name, mesh in meshes:
        all_z.append(mesh["layer"].min())
        all_z.append(mesh["layer"].max())
    z_min, z_max = min(all_z), max(all_z)
    
    # Get speed range
    speed_max = max(mesh["speed"].max() for _, mesh in meshes)
    speed_max = max(speed_max, 0.1)  # Avoid zero
    
    print(f"Starting viewer...")
    
    # Create plotter and add ONE mesh (instant!)
    plotter = BackgroundPlotter(title=f"OBF 3D - {args.obf_file.name}", window_size=(1400, 1000))
    plotter.set_background("white", top="lightblue")
    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    
    actors = []
    for name, mesh in meshes:
        if name == "lines":
            actor = plotter.add_mesh(
                mesh,
                scalars="speed",
                cmap="rainbow",
                clim=[0, speed_max],
                show_scalar_bar=True,
                scalar_bar_args={"title": "Speed (m/s)", "color": "black"},
                line_width=args.line_width,
                name="lines",
            )
        else:
            actor = plotter.add_mesh(
                mesh,
                scalars="speed", 
                cmap="rainbow",
                clim=[0, speed_max],
                show_scalar_bar=False,
                point_size=2,
                render_points_as_spheres=False,
                name="spots",
            )
        actors.append((name, mesh, actor))
    
    # Store original meshes for filtering
    original_meshes = {name: mesh.copy() for name, mesh, _ in actors}
    
    # Layer slider - uses threshold filter for instant updates
    current_z_max = [z_max]  # mutable container for closure
    
    def update_layer(value):
        z_threshold = value
        if abs(z_threshold - current_z_max[0]) < 0.001:
            return
        current_z_max[0] = z_threshold
        
        for name, orig_mesh, actor in actors:
            # Threshold by layer (z value)
            mask = orig_mesh["layer"] <= z_threshold
            if name == "lines":
                # For lines, we need pairs of points
                n_lines = len(mask) // 2
                line_mask = mask[::2]  # One bool per line
                
                if line_mask.sum() == 0:
                    actor.SetVisibility(False)
                else:
                    actor.SetVisibility(True)
                    # Filter points
                    keep_points = np.repeat(line_mask, 2)
                    new_points = orig_mesh.points[keep_points]
                    
                    # Rebuild connectivity
                    n_new = line_mask.sum()
                    new_cells = np.zeros(n_new * 3, dtype=np.int64)
                    new_cells[0::3] = 2
                    new_cells[1::3] = np.arange(0, n_new * 2, 2)
                    new_cells[2::3] = np.arange(1, n_new * 2, 2)
                    
                    new_mesh = pv.PolyData(new_points)
                    new_mesh.lines = new_cells
                    new_mesh["speed"] = orig_mesh["speed"][keep_points]
                    
                    # Update actor's mapper
                    actor.mapper.SetInputData(new_mesh)
            else:
                # Points - simple filter
                if mask.sum() == 0:
                    actor.SetVisibility(False)
                else:
                    actor.SetVisibility(True)
                    new_mesh = pv.PolyData(orig_mesh.points[mask])
                    new_mesh["speed"] = orig_mesh["speed"][mask]
                    actor.mapper.SetInputData(new_mesh)
        
        plotter.update()
    
    # Add slider - layer number approximation
    layer_min = int(z_min / layer_height_mm)
    layer_max = int(z_max / layer_height_mm)
    
    def slider_callback(value):
        z = value * layer_height_mm
        update_layer(z)
    
    plotter.add_slider_widget(
        slider_callback,
        [layer_min, layer_max],
        value=layer_max,
        title="Layer",
        pointa=(0.65, 0.92),
        pointb=(0.95, 0.92),
        style="modern",
    )
    
    plotter.reset_camera()
    plotter.view_isometric()
    plotter.app.exec_()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
