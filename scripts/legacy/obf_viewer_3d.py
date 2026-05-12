#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""
Standalone 3D visualization tool for OBF build files.

This tool visualizes the entire build in 3D, with each layer at its
physical Z-height. Uses PyVista for proper 3D rendering with volumetric
tube representations of scan paths.

Usage:
    python obf_viewer_3d.py <path_to_obf_file> [options]

Controls:
    - Mouse drag: Rotate view
    - Scroll: Zoom in/out
    - Layer slider: Show layers up to selected layer
    - Play button: Animate build layer by layer

Requirements:
    pip install pyvista pyvistaqt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Check for PyVista availability
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    HAS_PYVISTA = True
except ImportError as e:
    HAS_PYVISTA = False
    PYVISTA_ERROR = str(e)
    print(f"Warning: PyVista/Qt not available: {e}")
    print("Install with: pip install pyvista pyvistaqt pyqt6")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="3D OBF build visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "obf_file",
        type=pathlib.Path,
        help="Path to OBF archive file",
    )
    parser.add_argument(
        "--layer-height",
        type=float,
        default=50.0,
        help="Layer height in micrometers",
    )
    parser.add_argument(
        "--track-width",
        type=float,
        default=100.0,
        help="Melt track width in micrometers (tube diameter)",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=2000,
        help="Maximum paths to display per layer (for performance)",
    )
    parser.add_argument(
        "--tube-sides",
        type=int,
        default=6,
        help="Number of sides for tube geometry (lower = faster, higher = smoother)",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Use simple lines instead of tubes (much faster, lower memory)",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width when using --wireframe mode",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Point size for spot/dwell point rendering",
    )
    parser.add_argument(
        "--max-spots",
        type=int,
        default=50000,
        help="Maximum number of spots/dwell points to render per layer (default 50000)",
    )
    parser.add_argument(
        "--decimate",
        type=float,
        default=1.0,
        help="Decimation factor (0.1 = keep 10%% of geometry, 1.0 = keep all)",
    )
    parser.add_argument(
        "--melt-only",
        action="store_true",
        default=True,
        help="Only show melt layers (default)",
    )
    parser.add_argument(
        "--all-scans",
        action="store_true",
        help="Show all scan types (jumpsafe, preheat, melt, etc.)",
    )

    return parser.parse_args(args)


class Viewer3D:
    """3D visualization viewer for OBF build data using PyVista."""

    def __init__(
        self,
        data: list,
        layer_numbers: list[int],
        layer_height_um: float = 70.0,
        track_width_um: float = 500.0,
        max_paths_per_layer: int = 2000,
        tube_sides: int = 5,
        wireframe: bool = False,
        line_width: float = 2.0,
        point_size: float = 5.0,
        max_spots: int = 50000,
        decimate: float = 1.0,
        title: str = "OBF 3D Viewer",
    ):
        """Initialize the 3D viewer.

        Args:
            data: List of Data objects for each layer
            layer_numbers: Physical layer numbers (1-indexed) for each data entry
            layer_height_um: Layer height in micrometers
            track_width_um: Track/tube width in micrometers
            max_paths_per_layer: Max paths to show per layer for performance
            tube_sides: Number of sides for tube geometry
            wireframe: Use simple lines instead of tubes (faster, less memory)
            line_width: Line width for wireframe mode
            point_size: Point size for spot/dwell rendering
            max_spots: Maximum number of spots to render per layer
            decimate: Decimation factor for mesh simplification
            title: Window title
        """
        self.data = data
        self.layer_numbers = layer_numbers
        # Convert to mm for better display (OBP data is in meters already, layer height in μm)
        self.layer_height = layer_height_um * 1e-3  # μm to mm
        self.track_width = track_width_um * 1e-3   # μm to mm
        self.scale_factor = 1e3  # m to mm conversion for path data
        self.max_paths = max_paths_per_layer
        self.tube_sides = tube_sides
        self.wireframe = wireframe
        self.line_width = line_width
        self.point_size = point_size
        self.max_spots = max_spots
        self.decimate = decimate

        # Get unique physical layers
        self.unique_layers = sorted(set(layer_numbers))
        self.max_layer = max(self.unique_layers) if self.unique_layers else 1
        self.current_max_layer = self.max_layer

        # Pre-compute layer meshes
        self._precompute_layers()

        # Create the plotter with a gradient background
        self.plotter = BackgroundPlotter(title=title, window_size=(1200, 900))
        self.plotter.set_background("white", top="lightblue")

        # Add coordinate axes with labels
        self.plotter.add_axes(
            xlabel="X (mm)",
            ylabel="Y (mm)", 
            zlabel="Z (mm)",
            line_width=2,
        )

        # Store mesh actors for layer visibility control (list of actors per layer)
        self.layer_actors: dict[int, list] = {}

        # Add all layer meshes
        self._add_all_layers()

        # Add UI controls
        self._add_controls()

        # Set proper scaling - use equal aspect ratio
        self.plotter.reset_camera()
        self._set_equal_aspect()
        self.plotter.view_isometric()

    def _precompute_layers(self) -> None:
        """Pre-compute 3D meshes for each physical layer."""
        self.layer_meshes: dict[int, pv.PolyData] = {}
        self.layer_point_clouds: dict[int, pv.PolyData] = {}  # For spot-based data
        self.layer_speed_arrays: dict[int, np.ndarray] = {}

        # Group data by physical layer
        layer_data_map: dict[int, list] = {}
        for i, layer_num in enumerate(self.layer_numbers):
            if layer_num not in layer_data_map:
                layer_data_map[layer_num] = []
            layer_data_map[layer_num].append(self.data[i])

        mode = "wireframe" if self.wireframe else "tubes"
        print(f"Pre-computing 3D {mode} for {len(layer_data_map)} layers...")

        total_segments = 0
        total_points = 0
        
        for layer_num in sorted(layer_data_map.keys()):
            z = layer_num * self.layer_height  # Z height based on layer number

            all_lines = []
            all_points_list = []  # For spots/points
            all_point_speeds = []
            point_offset = 0

            for layer_data in layer_data_map[layer_num]:
                # Sample paths if too many
                paths = layer_data.paths
                layer_speeds = layer_data.speeds

                # Apply decimation - reduce number of paths
                effective_max = int(self.max_paths * self.decimate)
                if len(paths) > effective_max:
                    indices = np.linspace(0, len(paths) - 1, effective_max, dtype=int)
                    paths = [paths[i] for i in indices]
                    layer_speeds = layer_speeds[indices]

                for path, speed in zip(paths, layer_speeds):
                    verts = path.vertices
                    n_verts = len(verts)
                    
                    # Detect spots: single point, or small closed shapes (diamonds, etc.)
                    # A spot is when all vertices are within a small area (e.g., < 0.1mm)
                    is_spot = False
                    if n_verts == 1:
                        is_spot = True
                    elif n_verts >= 2:
                        # Check if path is a small closed shape (spot/dwell marker)
                        path_extent = np.ptp(verts, axis=0)  # range in x and y
                        max_extent = np.max(path_extent) * self.scale_factor  # in mm
                        if max_extent < 0.2:  # Less than 0.2mm extent = spot
                            is_spot = True
                    
                    if is_spot:
                        # Use centroid as the spot location
                        centroid = np.mean(verts, axis=0)
                        all_points_list.append([
                            centroid[0] * self.scale_factor,
                            centroid[1] * self.scale_factor,
                            z
                        ])
                        all_point_speeds.append(speed)
                    elif n_verts >= 2:
                        # Line segment - create 3D path
                        points_3d = np.column_stack([
                            verts[:, 0] * self.scale_factor,  # X in mm
                            verts[:, 1] * self.scale_factor,  # Y in mm
                            np.full(n_verts, z),  # Z already in mm
                        ])

                        # Create line connectivity
                        line = np.zeros(n_verts + 1, dtype=np.int64)
                        line[0] = n_verts
                        line[1:] = np.arange(point_offset, point_offset + n_verts)

                        all_lines.append((points_3d, line, speed))
                        point_offset += n_verts

            # Create line mesh if we have lines
            if all_lines:
                all_points = np.vstack([item[0] for item in all_lines])
                lines_list = [item[1] for item in all_lines]
                lines_array = np.concatenate(lines_list)

                line_mesh = pv.PolyData()
                line_mesh.points = all_points
                line_mesh.lines = lines_array

                speeds = np.array([item[2] for item in all_lines])

                if self.wireframe:
                    # Wireframe mode - just use lines
                    n_line_points = line_mesh.n_points
                    n_original_lines = len(speeds)
                    
                    if n_original_lines > 0 and n_line_points > 0:
                        points_per_line = n_line_points // n_original_lines
                        if points_per_line > 0:
                            speed_array = np.repeat(speeds, points_per_line)
                            if len(speed_array) < n_line_points:
                                speed_array = np.pad(speed_array, (0, n_line_points - len(speed_array)), 
                                                    mode='edge')
                            else:
                                speed_array = speed_array[:n_line_points]
                            line_mesh["speed"] = speed_array
                    
                    self.layer_meshes[layer_num] = line_mesh
                else:
                    # Tube mode
                    tube_mesh = line_mesh.tube(
                        radius=self.track_width / 2,
                        n_sides=self.tube_sides,
                        capping=False,
                    )

                    n_tube_points = tube_mesh.n_points
                    n_original_lines = len(speeds)
                    
                    if n_original_lines > 0 and n_tube_points > 0:
                        points_per_tube = n_tube_points // n_original_lines
                        if points_per_tube > 0:
                            speed_array = np.repeat(speeds, points_per_tube)
                            if len(speed_array) < n_tube_points:
                                speed_array = np.pad(speed_array, (0, n_tube_points - len(speed_array)), 
                                                    mode='edge')
                            else:
                                speed_array = speed_array[:n_tube_points]
                            tube_mesh["speed"] = speed_array

                    self.layer_meshes[layer_num] = tube_mesh
                
                total_segments += len(all_lines)
                del all_lines, all_points, lines_array

            # Create point cloud for spots/dwell points
            if all_points_list:
                points_arr = np.array(all_points_list)
                speeds_arr = np.array(all_point_speeds)
                
                # Decimate spots if too many
                n_spots = len(points_arr)
                if n_spots > self.max_spots:
                    indices = np.linspace(0, n_spots - 1, self.max_spots, dtype=int)
                    points_arr = points_arr[indices]
                    speeds_arr = speeds_arr[indices]
                    print(f"  Layer {layer_num}: decimated {n_spots} spots to {self.max_spots}")
                
                point_cloud = pv.PolyData(points_arr)
                point_cloud["speed"] = speeds_arr
                self.layer_point_clouds[layer_num] = point_cloud
                total_points += len(points_arr)

        print(f"Pre-computed {total_segments} line segments, {total_points} points across {len(self.layer_meshes) + len(self.layer_point_clouds)} layers")

        # Calculate global speed range for consistent coloring
        all_speeds = []
        for layer_data_list in layer_data_map.values():
            for layer_data in layer_data_list:
                all_speeds.extend(layer_data.speeds)
        
        if all_speeds:
            self.speed_min = 0
            self.speed_max = max(all_speeds)
        else:
            self.speed_min = 0
            self.speed_max = 1

    def _add_all_layers(self) -> None:
        """Add all layer meshes and point clouds to the plotter."""
        # Determine which layer shows the scalar bar (prefer one with tube mesh)
        scalar_bar_layer = self.max_layer
        scalar_bar_shown = False
        
        # Add tube/line meshes
        for layer_num, mesh in self.layer_meshes.items():
            show_bar = (layer_num == scalar_bar_layer) and not scalar_bar_shown
            
            if "speed" in mesh.array_names:
                if self.wireframe:
                    # Wireframe mode - render as lines
                    actor = self.plotter.add_mesh(
                        mesh,
                        scalars="speed",
                        cmap="rainbow",
                        clim=[self.speed_min, self.speed_max],
                        show_scalar_bar=show_bar,
                        scalar_bar_args={
                            "title": "Speed (m/s)",
                            "color": "black",
                            "title_font_size": 12,
                            "label_font_size": 10,
                        },
                        line_width=self.line_width,
                        render_lines_as_tubes=False,
                        name=f"layer_{layer_num}",
                    )
                else:
                    # Tube mode - render as surfaces
                    actor = self.plotter.add_mesh(
                        mesh,
                        scalars="speed",
                        cmap="rainbow",
                        clim=[self.speed_min, self.speed_max],
                        show_scalar_bar=show_bar,
                        scalar_bar_args={
                            "title": "Speed (m/s)",
                            "color": "black",
                            "title_font_size": 12,
                            "label_font_size": 10,
                        },
                        name=f"layer_{layer_num}",
                    )
                if show_bar:
                    scalar_bar_shown = True
            else:
                actor = self.plotter.add_mesh(
                    mesh,
                    color="orange",
                    line_width=self.line_width if self.wireframe else 1,
                    name=f"layer_{layer_num}",
                )
            
            if layer_num not in self.layer_actors:
                self.layer_actors[layer_num] = []
            self.layer_actors[layer_num].append(actor)

        # Add point clouds for spot/dwell data
        for layer_num, point_cloud in self.layer_point_clouds.items():
            show_bar = (layer_num == scalar_bar_layer) and not scalar_bar_shown
            
            if "speed" in point_cloud.array_names:
                actor = self.plotter.add_points(
                    point_cloud,
                    scalars="speed",
                    cmap="rainbow",
                    clim=[self.speed_min, self.speed_max],
                    show_scalar_bar=show_bar,
                    scalar_bar_args={
                        "title": "Speed (m/s)",
                        "color": "black",
                        "title_font_size": 12,
                        "label_font_size": 10,
                    },
                    point_size=self.point_size,
                    render_points_as_spheres=False,  # Flat points are much faster
                    name=f"layer_{layer_num}_points",
                )
                if show_bar:
                    scalar_bar_shown = True
            else:
                actor = self.plotter.add_points(
                    point_cloud,
                    color="red",
                    point_size=self.point_size,
                    render_points_as_spheres=False,  # Flat points are much faster
                    name=f"layer_{layer_num}_points",
                )
            
            if layer_num not in self.layer_actors:
                self.layer_actors[layer_num] = []
            self.layer_actors[layer_num].append(actor)

    def _set_equal_aspect(self) -> None:
        """Set equal aspect ratio for all axes."""
        # Get the bounds of all meshes and point clouds
        if not self.layer_meshes and not self.layer_point_clouds:
            return
        
        # Combine bounds from all layers
        all_bounds = []
        for mesh in self.layer_meshes.values():
            all_bounds.append(mesh.bounds)
        for pc in self.layer_point_clouds.values():
            all_bounds.append(pc.bounds)
        
        if not all_bounds:
            return
            
        bounds = np.array(all_bounds)
        x_min, x_max = bounds[:, 0].min(), bounds[:, 1].max()
        y_min, y_max = bounds[:, 2].min(), bounds[:, 3].max()
        z_min, z_max = bounds[:, 4].min(), bounds[:, 5].max()
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Find the maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Set camera to view the full build with equal scaling
        # Add some padding
        padding = max_range * 0.1
        
        self.plotter.camera.focal_point = (x_center, y_center, z_center)
        self.plotter.reset_camera()

    def _add_controls(self) -> None:
        """Add UI controls to the plotter."""
        # Layer slider
        def update_layers(value):
            self.current_max_layer = int(value)
            for layer_num, actors in self.layer_actors.items():
                visible = layer_num <= self.current_max_layer
                for actor in actors:
                    actor.SetVisibility(visible)
            self.plotter.update()

        self.plotter.add_slider_widget(
            update_layers,
            [1, self.max_layer],
            value=self.max_layer,
            title="Max Layer",
            pointa=(0.1, 0.92),
            pointb=(0.9, 0.92),
            style="modern",
        )

        # Add text showing layer info
        self.plotter.add_text(
            f"Layers: {len(self.unique_layers)} | "
            f"Track width: {self.track_width*1e3:.0f}μm | "
            f"Layer height: {self.layer_height*1e3:.0f}μm",
            position="lower_left",
            font_size=10,
            color="black",
        )

    def show(self) -> None:
        """Show the viewer (blocking)."""
        self.plotter.app.exec_()


def main(args: list[str] | None = None) -> int:
    """Main entry point for the 3D viewer."""
    parsed = parse_args(args)

    if not HAS_PYVISTA:
        print("Error: PyVista is required for 3D visualization.")
        print("Install with: pip install pyvista pyvistaqt pyqt6")
        print(f"Original error: {PYVISTA_ERROR}")
        return 1

    # Import obfviewer components
    try:
        from obfviewer.loaders import load_obp_files_parallel
        from obfviewer.utils import extract_obf_archive, get_layer_sequence_with_info
    except ImportError:
        print("Error: obfviewer package not found. Install with: pip install -e .")
        return 1

    # Validate input file
    if not parsed.obf_file.exists():
        print(f"Error: File not found: {parsed.obf_file}", file=sys.stderr)
        return 1

    # Extract archive
    print(f"Extracting {parsed.obf_file.name}...")
    temp_dir = extract_obf_archive(parsed.obf_file)

    # Get layer sequence
    melt_only = not parsed.all_scans
    layer_sequence, layer_numbers, scan_types = get_layer_sequence_with_info(
        temp_dir, melt_only=melt_only
    )

    if not layer_sequence:
        print("No .obp files found in the provided .obf archive.", file=sys.stderr)
        return 1

    num_physical_layers = max(layer_numbers) if layer_numbers else 0
    print(f"Found {len(layer_sequence)} scans across {num_physical_layers} physical layers")

    # Load data
    print("Loading layer data...")
    data = load_obp_files_parallel(layer_sequence)

    # Create and show viewer
    print("Starting 3D viewer...")
    viewer = Viewer3D(
        data,
        layer_numbers,
        layer_height_um=parsed.layer_height,
        track_width_um=parsed.track_width,
        max_paths_per_layer=parsed.max_paths,
        tube_sides=parsed.tube_sides,
        wireframe=parsed.wireframe,
        line_width=parsed.line_width,
        point_size=parsed.point_size,
        max_spots=parsed.max_spots,
        decimate=parsed.decimate,
        title=f"OBF 3D Viewer - {parsed.obf_file.name}",
    )
    viewer.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
