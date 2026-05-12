# SPDX-FileCopyrightText: 2025-2026 Olgierd Nowakowski
# SPDX-License-Identifier: Apache-2.0

"""Combined 2D/3D OBF Viewer — main application entry point.

Features:
- 2D matplotlib view for detailed layer inspection (path-by-path)
- 3D PyVista view for full build visualization
- Synchronized layer navigation between views
- Dark mode support
- Fast parallel loading with per-file deduplication caching

Usage (installed):
    obfviewer-combined path/to/file.obf

Usage (direct):
    python -m obfviewer.combined path/to/file.obf
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Qt imports
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal

# Matplotlib for 2D view
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

# PyVista for 3D view
import pyvista as pv
from pyvistaqt import QtInteractor

from obfviewer.models import Data


# =============================================================================
# Fast OBP Parsing for 3D view
# =============================================================================


def _parse_obp_for_3d(
    filepath: pathlib.Path, z: float, scale: float, max_elements: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse an OBP file and return geometry arrays for 3D rendering.

    Args:
        filepath: Path to the OBP file.
        z: Z-height in mm to assign to this layer's geometry.
        scale: Coordinate scale factor (use 1e3 to convert m→mm).
        max_elements: Maximum number of elements to return (subsampled evenly).

    Returns:
        Tuple of (lines, spots) numpy arrays, or None on error.
        lines shape: (N, 7) — x0, y0, x1, y1, z, speed, layer_z
        spots shape: (M, 4) — x, y, z, dwell_time
    """
    from google.protobuf.internal.decoder import _DecodeVarint32
    from obplib import OBP_pb2 as obp

    try:
        if str(filepath).endswith(".gz"):
            with gzip.open(filepath, "rb") as f:
                data = f.read()
        else:
            with open(filepath, "rb") as f:
                data = f.read()
    except Exception:
        return None

    lines_list: list = []
    spots_list: list = []
    pscale = 1e-6 * scale

    packet = obp.Packet()
    consumed = 0
    data_len = len(data)

    while consumed < data_len:
        msg_len, new_pos = _DecodeVarint32(data, consumed)
        packet.ParseFromString(data[new_pos : new_pos + msg_len])
        consumed = new_pos + msg_len

        attr = packet.WhichOneof("payload")

        if attr == "line":
            obj = packet.line
            lines_list.append(
                (obj.x0 * pscale, obj.y0 * pscale, obj.x1 * pscale, obj.y1 * pscale, z, obj.speed * 1e-6, z)
            )
        elif attr == "accelerating_line":
            obj = packet.accelerating_line
            lines_list.append(
                (obj.x0 * pscale, obj.y0 * pscale, obj.x1 * pscale, obj.y1 * pscale, z, obj.sf, z)
            )
        elif attr == "curve":
            obj = packet.curve
            lines_list.append(
                (obj.p0.x * pscale, obj.p0.y * pscale, obj.p3.x * pscale, obj.p3.y * pscale, z, obj.speed * 1e-6, z)
            )
        elif attr == "accelerating_curve":
            obj = packet.accelerating_curve
            lines_list.append(
                (obj.p0.x * pscale, obj.p0.y * pscale, obj.p3.x * pscale, obj.p3.y * pscale, z, obj.sf, z)
            )
        elif attr == "timed_points":
            for pt in packet.timed_points.points:
                spots_list.append((pt.x * pscale, pt.y * pscale, z, pt.t * 1e-6 if pt.t else 0))

    lines = np.array(lines_list, dtype=np.float32) if lines_list else np.zeros((0, 7), dtype=np.float32)
    spots = np.array(spots_list, dtype=np.float32) if spots_list else np.zeros((0, 4), dtype=np.float32)

    if len(lines) > max_elements:
        idx = np.linspace(0, len(lines) - 1, max_elements, dtype=np.int32)
        lines = lines[idx]
    if len(spots) > max_elements:
        idx = np.linspace(0, len(spots) - 1, max_elements, dtype=np.int32)
        spots = spots[idx]

    return lines, spots


def _worker_3d(args: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    """ProcessPoolExecutor-compatible wrapper for `_parse_obp_for_3d`."""
    filepath, z, scale, max_elem = args
    return _parse_obp_for_3d(filepath, z, scale, max_elem)


# =============================================================================
# 2D Layer View Widget
# =============================================================================


class LayerView2D(QWidget):
    """2D matplotlib view for detailed layer inspection (path-by-path)."""

    layer_changed = pyqtSignal(int)  # Emits physical layer number

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: list[Data] | None = None
        self.layer_numbers: list[int] = []
        self.scan_types: list[str] = []
        self.current_idx = 0
        self.current_indices: list[int] = []
        self.current_path_idx = 0
        self.marker = None
        self.has_dwell = False
        self.max_dwell = 1.0
        self.max_speed = 1.0
        self.dark_mode_enabled = False
        self.current_layer_data: Data | None = None

        # Debounce rapid path-slider drags so the expensive matplotlib
        # canvas redraw only fires after the user pauses (~80 ms).
        self._path_timer = QTimer(singleShot=True, interval=80)
        self._path_timer.timeout.connect(self._flush_path_change)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 7), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Layer:"))
        self.layer_spin = QSpinBox()
        self.layer_spin.setMinimum(1)
        self.layer_spin.valueChanged.connect(self._on_layer_spin_changed)
        ctrl_layout.addWidget(self.layer_spin)

        ctrl_layout.addWidget(QLabel("Scan:"))
        self.scan_combo = QComboBox()
        self.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        ctrl_layout.addWidget(self.scan_combo)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        self.path_slider = QSlider(Qt.Orientation.Horizontal)
        self.path_slider.setMinimum(0)
        self.path_slider.valueChanged.connect(self._on_path_changed)
        path_layout.addWidget(self.path_slider, stretch=1)
        self.path_label = QLabel("0/0")
        self.path_label.setMinimumWidth(80)
        path_layout.addWidget(self.path_label)
        layout.addLayout(path_layout)

        self.info_label = QLabel("Speed: - | Power: - | Dwell: - | Spot: -")
        self.info_label.setStyleSheet("font-family: monospace; padding: 5px; background: #f0f0f0;")
        layout.addWidget(self.info_label)

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
        self.ax.yaxis.set_major_formatter(EngFormatter(unit="m"))

        self.path_collection = None
        self.colorbar = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_data(self, data: list[Data], layer_numbers: list[int], scan_types: list[str]) -> None:
        """Load data and initialise the view."""
        self.data = data
        self.layer_numbers = layer_numbers
        self.scan_types = scan_types

        if not data:
            return

        self.layer_spin.setMinimum(min(layer_numbers))
        self.layer_spin.setMaximum(max(layer_numbers))
        self.layer_spin.blockSignals(True)
        self.layer_spin.setValue(min(layer_numbers))
        self.layer_spin.blockSignals(False)

        self._update_scan_options()
        self._update_view()

    def set_layer(self, layer_num: int) -> None:
        """Navigate to the given physical layer number (1-indexed)."""
        if self.layer_spin.value() != layer_num:
            self.layer_spin.setValue(layer_num)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_layer_spin_changed(self, layer_num: int) -> None:
        self.current_path_idx = -1
        self._update_scan_options()
        self._update_view()
        self.layer_changed.emit(layer_num)

    def _update_scan_options(self) -> None:
        """Populate the scan-type combo for the currently selected layer."""
        layer_num = self.layer_spin.value()
        scans = [
            (i, st)
            for i, (ln, st) in enumerate(zip(self.layer_numbers, self.scan_types))
            if ln == layer_num
        ]
        melt_indices = [i for i, st in scans if "melt" in st.lower()]

        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()

        if len(melt_indices) > 1:
            self.scan_combo.addItem(f"All Melts ({len(melt_indices)})", melt_indices)

        for idx, scan_type in scans:
            self.scan_combo.addItem(scan_type, [idx])

        # Default: prefer "All Melts" or first melt scan
        if len(melt_indices) > 1:
            self.scan_combo.setCurrentIndex(0)
        else:
            melt_idx = next(
                (i for i in range(self.scan_combo.count()) if "melt" in self.scan_combo.itemText(i).lower()),
                0 if self.scan_combo.count() > 0 else -1,
            )
            if melt_idx >= 0:
                self.scan_combo.setCurrentIndex(melt_idx)

        self.current_indices = self.scan_combo.currentData() or []
        self.scan_combo.blockSignals(False)

    def _on_scan_changed(self, combo_idx: int) -> None:
        if combo_idx >= 0 and self.scan_combo.count() > 0:
            data = self.scan_combo.itemData(combo_idx)
            if data is not None:
                self.current_indices = data
                self.current_path_idx = -1
                self._update_view()

    def _on_path_changed(self, value: int) -> None:
        self.current_path_idx = value
        # Update the counter label immediately so the UI feels responsive,
        # then debounce the expensive canvas redraw.
        if self.current_layer_data is not None:
            n = len(self.current_layer_data.paths)
            self.path_label.setText(f"{value + 1}/{n}")
        self._path_timer.start()

    def _flush_path_change(self) -> None:
        self._update_display(self.current_path_idx)

    def _merge_scan_data(self, indices: list[int]) -> Data | None:
        """Merge one or more scan Data objects into a single Data object."""
        if not indices or self.data is None:
            return None
        if len(indices) == 1:
            return self.data[indices[0]]

        all_paths: list = []
        all_speeds: list[np.ndarray] = []
        all_dwell_times: list[np.ndarray] = []
        all_spotsizes: list[np.ndarray] = []
        all_beampowers: list[np.ndarray] = []

        for idx in indices:
            d = self.data[idx]
            all_paths.extend(d.paths)
            all_speeds.append(d.speeds)
            all_dwell_times.append(d.dwell_times)
            all_spotsizes.append(d.spotsizes)
            all_beampowers.append(d.beampowers)

        return Data(
            paths=all_paths,
            speeds=np.concatenate(all_speeds) if all_speeds else np.array([]),
            dwell_times=np.concatenate(all_dwell_times) if all_dwell_times else np.array([]),
            spotsizes=np.concatenate(all_spotsizes) if all_spotsizes else np.array([]),
            beampowers=np.concatenate(all_beampowers) if all_beampowers else np.array([]),
            syncpoints={},
            restores=np.array([]),
        )

    def _update_view(self) -> None:
        """Rebuild the 2D matplotlib canvas for the current layer/scan selection."""
        if not self.data or not self.current_indices:
            return

        layer_data = self._merge_scan_data(self.current_indices)
        if layer_data is None:
            return

        self.current_idx = self.current_indices[0]
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
        self.ax.yaxis.set_major_formatter(EngFormatter(unit="m"))

        # Build-plate reference circle (100 mm diameter)
        self.ax.add_patch(
            plt.Circle(
                (0, 0), 0.05, fill=False, color="gray",
                linestyle="--", linewidth=1.5, label="Build plate (∅100 mm)",
            )
        )

        n_paths = len(layer_data.paths)
        if n_paths == 0:
            self.path_slider.setMaximum(0)
            self.path_label.setText("0/0")
            self.info_label.setText("No paths in this scan")
            self.ax.set_xlim(-0.06, 0.06)
            self.ax.set_ylim(-0.06, 0.06)
            if self.dark_mode_enabled:
                self.set_dark_mode(True)
            self.canvas.draw()
            return

        self.current_layer_data = layer_data
        self.n_paths = n_paths

        has_dwell = len(layer_data.dwell_times) > 0 and np.any(layer_data.dwell_times > 0)

        if has_dwell:
            dwell_us = layer_data.dwell_times * 1e6
            self.max_dwell = max(float(dwell_us.max()), 1.0)
            self.path_collection = mcoll.PathCollection(
                layer_data.paths, facecolors="none",
                transform=self.ax.transData, cmap=plt.cm.plasma,
                norm=plt.Normalize(vmin=0, vmax=self.max_dwell),
            )
            self.path_collection.set_array(dwell_us)
            self.ax.add_collection(self.path_collection)
            self.colorbar = self.figure.colorbar(
                self.path_collection, ax=self.ax, pad=0.01, aspect=40,
                label="Dwell time (μs)",
            )
        else:
            self.max_speed = max(float(layer_data.speeds.max()) if len(layer_data.speeds) > 0 else 0, 1.0)
            self.path_collection = mcoll.PathCollection(
                layer_data.paths, facecolors="none",
                transform=self.ax.transData, cmap=plt.cm.rainbow,
                norm=plt.Normalize(vmin=0, vmax=self.max_speed),
            )
            self.path_collection.set_array(layer_data.speeds)
            self.ax.add_collection(self.path_collection)
            self.colorbar = self.figure.colorbar(
                self.path_collection, ax=self.ax, pad=0.01, aspect=40,
                format=EngFormatter(unit="m/s"),
            )

        self.has_dwell = has_dwell

        path_idx = max(min(self.current_path_idx, n_paths - 1), 0)
        seg = layer_data.paths[path_idx]
        if len(seg.vertices) > 0:
            self.marker = self.ax.scatter(
                *seg.vertices[-1], c="white", edgecolors="black", s=100, marker="*", zorder=10
            )

        # Compute axis bounds from paths + build-plate
        all_x = [-0.05, 0.05]
        all_y = [-0.05, 0.05]
        for p in layer_data.paths:
            if len(p.vertices) > 0:
                all_x.extend(p.vertices[:, 0])
                all_y.extend(p.vertices[:, 1])
        margin = 0.005
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        if self.current_path_idx < 0:
            self.current_path_idx = n_paths - 1

        self.path_slider.blockSignals(True)
        self.path_slider.setMaximum(n_paths - 1)
        self.path_slider.setValue(min(self.current_path_idx, n_paths - 1))
        self.path_slider.blockSignals(False)

        self._update_display(self.path_slider.value())

        if self.dark_mode_enabled:
            self.set_dark_mode(True)

        self.canvas.draw()

    def _update_display(self, path_idx: int) -> None:
        """Refresh info label, path visibility, and marker position."""
        if self.current_layer_data is None:
            return

        layer_data = self.current_layer_data
        n_paths = len(layer_data.paths)
        if n_paths == 0:
            return

        path_idx = min(path_idx, n_paths - 1)
        self.path_label.setText(f"{path_idx + 1}/{n_paths}")

        speed = layer_data.speeds[path_idx] if path_idx < len(layer_data.speeds) else 0.0
        power = layer_data.beampowers[path_idx] if path_idx < len(layer_data.beampowers) else 0.0
        dwell = layer_data.dwell_times[path_idx] if path_idx < len(layer_data.dwell_times) else 0.0
        spot = layer_data.spotsizes[path_idx] if path_idx < len(layer_data.spotsizes) else 0.0
        self.info_label.setText(
            f"Speed: {speed:.2f} m/s | Power: {power:.0f} W | Dwell: {dwell * 1e6:.1f} μs | Spot: {spot:.0f} μm"
        )

        if self.path_collection is not None:
            if self.has_dwell:
                dwell_us = layer_data.dwell_times * 1e6
                colors = plt.cm.plasma(plt.Normalize(0, self.max_dwell)(dwell_us))
            else:
                colors = plt.cm.rainbow(plt.Normalize(0, self.max_speed)(layer_data.speeds))
            colors[path_idx + 1 :, 3] = 0
            self.path_collection.set_edgecolors(colors)

        if self.marker is not None and path_idx < len(layer_data.paths):
            seg = layer_data.paths[path_idx]
            if len(seg.vertices) > 0:
                self.marker.set_offsets([seg.vertices[-1]])

        self.canvas.draw_idle()

    def set_dark_mode(self, dark: bool) -> None:
        """Apply or remove dark styling to the matplotlib figure."""
        self.dark_mode_enabled = dark
        if dark:
            bg, fg, frame = "#1e1e1e", "#e0e0e0", "#555555"
            ax_bg = "#252526"
            info_style = "font-family: monospace; padding: 5px; background: #3a3a3a; color: #e0e0e0;"
        else:
            bg, fg, frame = "white", "black", "black"
            ax_bg = "white"
            info_style = "font-family: monospace; padding: 5px; background: #f0f0f0; color: black;"

        self.figure.set_facecolor(bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.tick_params(colors=fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        for spine in self.ax.spines.values():
            spine.set_color(frame)
        self.info_label.setStyleSheet(info_style)

        if self.colorbar is not None:
            self.colorbar.ax.tick_params(colors=fg)
            self.colorbar.ax.xaxis.label.set_color(fg)
            self.colorbar.ax.yaxis.label.set_color(fg)
            for spine in self.colorbar.ax.spines.values():
                spine.set_color(frame)
            self.colorbar.outline.set_color(frame)

        self.canvas.draw()


# =============================================================================
# 3D Build View Widget
# =============================================================================


class BuildView3D(QWidget):
    """3D PyVista view for full build visualisation."""

    layer_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh_data: dict | None = None
        self.lines_mesh = None
        self.lines_actor = None
        self.spots_mesh = None
        self.spots_actor = None
        self.layer_height_mm = 0.07
        self.z_min = 0.0
        self.z_max = 1.0
        self._last_z_max: float | None = None
        self.dark_mode_enabled = False

        # Debounce the cross-panel layer sync: the clip-plane move is cheap
        # and happens on every tick, but triggering a full 2D matplotlib
        # redraw on every tick makes the slider feel jerky.
        self._sync_timer = QTimer(singleShot=True, interval=120)
        self._sync_timer.timeout.connect(self._flush_layer_sync)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pv.global_theme.allow_empty_mesh = True
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white", top="lightblue")
        self.plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        # FXAA is a 2D post-process filter — nearly free vs SSAA which renders
        # the scene at 2× resolution (4× fragment cost) before downsampling.
        self.plotter.enable_anti_aliasing("fxaa")
        self.plotter.enable_parallel_projection()
        layout.addWidget(self.plotter.interactor, stretch=1)

        # Viewport preset buttons
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))
        for label, fn in [
            ("Iso", lambda: self.plotter.view_isometric()),
            ("Top", lambda: self.plotter.view_xy()),
            ("Front", lambda: self.plotter.view_xz()),
            ("Side", lambda: self.plotter.view_yz()),
        ]:
            btn = QPushButton(label)
            btn.setMaximumWidth(50)
            btn.clicked.connect(fn)
            view_layout.addWidget(btn)

        view_layout.addSpacing(20)

        self.ortho_check = QCheckBox("Ortho")
        self.ortho_check.setChecked(True)
        self.ortho_check.toggled.connect(self._toggle_projection)
        view_layout.addWidget(self.ortho_check)

        self.dark_check = QCheckBox("Dark")
        self.dark_check.toggled.connect(self._toggle_dark_mode)
        view_layout.addWidget(self.dark_check)

        btn_reset = QPushButton("Reset")
        btn_reset.setMaximumWidth(50)
        btn_reset.clicked.connect(lambda: self.plotter.reset_camera())
        view_layout.addWidget(btn_reset)
        view_layout.addStretch()
        layout.addLayout(view_layout)

        # Layer Z-clip slider
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Show up to layer:"))
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(1)
        self.layer_slider.valueChanged.connect(self._on_slider_changed)
        ctrl_layout.addWidget(self.layer_slider, stretch=1)
        self.layer_label = QLabel("1")
        self.layer_label.setMinimumWidth(50)
        ctrl_layout.addWidget(self.layer_label)
        layout.addLayout(ctrl_layout)

    def _toggle_projection(self, ortho: bool) -> None:
        if ortho:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.update()

    def _toggle_dark_mode(self, dark: bool) -> None:
        self.dark_mode_enabled = dark
        if dark:
            self.plotter.set_background("black", top="#1a1a2e")
        else:
            self.plotter.set_background("white", top="lightblue")
        self.plotter.update()

    def set_mesh_data(self, lines: np.ndarray, spots: np.ndarray, layer_height_mm: float) -> None:
        """Load pre-computed 3D geometry arrays."""
        self.layer_height_mm = layer_height_mm
        self.mesh_data = {"lines": lines, "spots": spots}

        all_z = []
        if len(lines) > 0:
            all_z += [float(lines[:, 4].min()), float(lines[:, 4].max())]
        if len(spots) > 0:
            all_z += [float(spots[:, 2].min()), float(spots[:, 2].max())]

        if not all_z:
            return

        self.z_min, self.z_max = min(all_z), max(all_z)
        layer_min = max(1, int(self.z_min / layer_height_mm))
        layer_max = int(self.z_max / layer_height_mm)

        self.layer_slider.setMinimum(layer_min)
        self.layer_slider.setMaximum(layer_max)
        self.layer_slider.setValue(layer_max)
        self.layer_label.setText(str(layer_max))

        self._build_meshes()

    def _build_meshes(self) -> None:
        """Construct VTK tube/sphere meshes and add them to the plotter."""
        lines = self.mesh_data["lines"]
        spots = self.mesh_data["spots"]

        speed_max = 0.1
        if len(lines) > 0:
            speed_max = max(speed_max, float(lines[:, 5].max()))
        if len(spots) > 0:
            speed_max = max(speed_max, float(spots[:, 3].max()))
        self.speed_max = speed_max
        self._last_z_max = None

        sb_color = "#e0e0e0" if self.dark_mode_enabled else "black"

        # Line geometry — stored as raw PolyData lines (no tube tessellation).
        # render_lines_as_tubes=True tells the OpenGL shader to draw them as
        # 3D tubes entirely on the GPU, which is instant compared to VTK's
        # CPU-side .tube() tessellation.
        if len(lines) > 0:
            n = len(lines)
            pts = np.zeros((n * 2, 3), dtype=np.float32)
            pts[0::2, 0] = lines[:, 0]
            pts[0::2, 1] = lines[:, 1]
            pts[0::2, 2] = lines[:, 4]
            pts[1::2, 0] = lines[:, 2]
            pts[1::2, 1] = lines[:, 3]
            pts[1::2, 2] = lines[:, 4]

            cells = np.zeros(n * 3, dtype=np.int64)
            cells[0::3] = 2
            cells[1::3] = np.arange(0, n * 2, 2)
            cells[2::3] = np.arange(1, n * 2, 2)

            lm = pv.PolyData(pts)
            lm.lines = cells
            lm.point_data["speed"] = np.repeat(lines[:, 5].astype(np.float32), 2)
            lm.point_data["z"] = np.repeat(lines[:, 4].astype(np.float32), 2)

            self.lines_mesh = lm
            self.lines_actor = self.plotter.add_mesh(
                lm, scalars="speed", cmap="rainbow",
                clim=[0, speed_max], show_scalar_bar=True,
                scalar_bar_args={"title": "Speed (m/s)", "color": sb_color},
                render_lines_as_tubes=True,
                line_width=6,
                name="lines",
            )
        else:
            self.lines_mesh = self.lines_actor = None

        # Spot geometry — stored as raw PolyData point cloud (no glyph tessellation).
        # render_points_as_spheres=True uses the GPU shader to draw spheres.
        if len(spots) > 0:
            max_spots = 500_000
            s = spots if len(spots) <= max_spots else spots[np.linspace(0, len(spots) - 1, max_spots, dtype=np.int32)]
            pc = pv.PolyData(s[:, :3].astype(np.float32))
            pc.point_data["speed"] = s[:, 3].astype(np.float32)
            pc.point_data["z"] = s[:, 2].astype(np.float32)

            self.spots_mesh = pc
            self.spots_actor = self.plotter.add_mesh(
                pc, scalars="speed", cmap="rainbow",
                clim=[0, speed_max], show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=8,
                name="spots",
            )
        else:
            self.spots_mesh = self.spots_actor = None

        # Build-plate representation
        plate = pv.Cylinder(radius=50, height=10, center=(0, 0, -5), direction=(0, 0, 1), resolution=64)
        self.plotter.add_mesh(plate, color="silver", specular=0.5, specular_power=20, name="build_plate")

        self._setup_clipping()
        self._setup_lod()
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def _setup_lod(self) -> None:
        """Configure interactive LOD for smooth dragging.

        Uses vtkInteractorStyle's StartInteractionEvent / EndInteractionEvent
        rather than raw button events so the callback fires reliably at the
        *style* level (rotate/pan/zoom gestures) rather than raw mouse clicks.
        During interaction the tube geometry shader is disabled so VTK renders
        plain lines — still visible, just not 3D. Restored on release.
        """
        iren = self.plotter.interactor
        iren.SetDesiredUpdateRate(60.0)
        iren.SetStillUpdateRate(0.01)

        style = iren.GetInteractorStyle()
        style.AddObserver("StartInteractionEvent", lambda *_: self._lod_mode(True))
        style.AddObserver("EndInteractionEvent", lambda *_: self._lod_mode(False))

    def _lod_mode(self, fast: bool) -> None:
        """Toggle between interaction quality (fast) and still quality."""
        if self.lines_actor is not None:
            prop = self.lines_actor.GetProperty()
            if fast:
                prop.RenderLinesAsTubesOff()
            else:
                prop.RenderLinesAsTubesOn()
        if not fast:
            self.plotter.render()

    def _setup_clipping(self) -> None:
        """Attach a GPU clipping plane for fast layer-by-layer Z filtering."""
        import vtk

        self._clip_plane = vtk.vtkPlane()
        self._clip_plane.SetOrigin(0, 0, self.z_max)
        self._clip_plane.SetNormal(0, 0, -1)

        if self.lines_actor is not None:
            self.lines_actor.mapper.AddClippingPlane(self._clip_plane)
        if self.spots_actor is not None:
            self.spots_actor.mapper.AddClippingPlane(self._clip_plane)

    def set_layer(self, layer_num: int) -> None:
        if self.layer_slider.value() != layer_num:
            self.layer_slider.setValue(layer_num)

    def _on_slider_changed(self, layer_num: int) -> None:
        self.layer_label.setText(str(layer_num))
        self._filter_by_z(layer_num * self.layer_height_mm)  # cheap — just moves the clip plane
        self._sync_timer.start()  # debounce the expensive 2D cross-sync

    def _flush_layer_sync(self) -> None:
        self.layer_changed.emit(self.layer_slider.value())

    def _filter_by_z(self, z_max: float) -> None:
        """Move the GPU clipping plane to reveal layers up to z_max."""
        if self._last_z_max == z_max:
            return
        self._last_z_max = z_max
        if hasattr(self, "_clip_plane"):
            self._clip_plane.SetOrigin(0, 0, z_max)
            self.plotter.update()


# =============================================================================
# Background Loader Thread
# =============================================================================


class LoaderThread(QThread):
    """Loads an OBF file's 2D and 3D data in a background thread."""

    progress = pyqtSignal(int, int)
    finished_2d = pyqtSignal(list, list, list)
    finished_3d = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)

    def __init__(
        self,
        obf_path: pathlib.Path,
        layer_height_um: float,
        max_per_file: int,
        melt_only: bool,
    ):
        super().__init__()
        self.obf_path = obf_path
        self.layer_height_um = layer_height_um
        self.max_per_file = max_per_file
        self.melt_only = melt_only

    def run(self) -> None:
        try:
            from obfviewer.loaders import load_obp_files_parallel
            from obfviewer.utils import extract_obf_archive, get_layer_sequence_with_info

            temp_dir = extract_obf_archive(self.obf_path)
            files, layer_nums, scan_types = get_layer_sequence_with_info(temp_dir, melt_only=self.melt_only)

            if not files:
                self.error.emit("No matching OBP files found in the archive.")
                return

            data_2d = load_obp_files_parallel(files, show_progress=False)
            self.finished_2d.emit(data_2d, layer_nums, scan_types)

            layer_height_mm = self.layer_height_um * 1e-3
            scale = 1e3  # m → mm

            work = [
                (f, ln * layer_height_mm, scale, self.max_per_file)
                for f, ln, st in zip(files, layer_nums, scan_types)
                if st == "melt"
            ]

            all_lines: list[np.ndarray] = []
            all_spots: list[np.ndarray] = []

            with ProcessPoolExecutor() as ex:
                futures = {ex.submit(_worker_3d, w): i for i, w in enumerate(work)}
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
                    self.progress.emit(done, len(work))

            stacked_lines = np.vstack(all_lines) if all_lines else np.zeros((0, 7), dtype=np.float32)
            stacked_spots = np.vstack(all_spots) if all_spots else np.zeros((0, 4), dtype=np.float32)
            self.finished_3d.emit(stacked_lines, stacked_spots)

        except Exception as exc:
            self.error.emit(str(exc))


# =============================================================================
# Main Window
# =============================================================================

_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QLabel { color: #e0e0e0; }
QPushButton {
    background-color: #3a3a3a; color: #e0e0e0;
    border: 1px solid #555; padding: 4px 8px; border-radius: 3px;
}
QPushButton:hover { background-color: #4a4a4a; }
QPushButton:pressed { background-color: #2a2a2a; }
QSlider::groove:horizontal { background: #3a3a3a; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #0078d4; width: 16px; margin: -5px 0; border-radius: 8px; }
QComboBox, QSpinBox {
    background-color: #3a3a3a; color: #e0e0e0;
    border: 1px solid #555; padding: 3px;
}
QComboBox::drop-down { border: none; }
QCheckBox { color: #e0e0e0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QFrame { background-color: #252526; border: 1px solid #3a3a3a; }
QStatusBar { background-color: #007acc; color: white; }
QProgressBar { background-color: #3a3a3a; border: none; border-radius: 3px; }
QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
QSplitter::handle { background-color: #3a3a3a; }
"""


class CombinedViewer(QMainWindow):
    """Main window combining the 2D layer view and the 3D build view."""

    def __init__(
        self,
        obf_path: pathlib.Path,
        layer_height_um: float = 70.0,
        max_per_file: int = 50_000,
        melt_only: bool = True,
        no_3d: bool = False,
    ):
        super().__init__()
        self.obf_path = obf_path
        self.layer_height_um = layer_height_um
        self.max_per_file = max_per_file
        self.sync_layers = True
        self.no_3d = no_3d

        self.setWindowTitle(f"OBF Viewer — {obf_path.name}")
        self.resize(1600 if not no_3d else 900, 900)

        self._setup_ui()
        self._start_loader(obf_path, max_per_file, melt_only)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Menu bar
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Load New File…", self._load_new_file)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        self.menuBar().addMenu("Help").addAction("About", self._show_about)

        # Top bar
        top_layout = QHBoxLayout()
        if not self.no_3d:
            self.sync_check = QCheckBox("Sync layer between views")
            self.sync_check.setChecked(True)
            self.sync_check.toggled.connect(lambda c: setattr(self, "sync_layers", c))
            top_layout.addWidget(self.sync_check)
        top_layout.addStretch()
        self.dark_mode_check = QCheckBox("Dark Mode")
        self.dark_mode_check.toggled.connect(self._toggle_global_dark_mode)
        top_layout.addWidget(self.dark_mode_check)
        layout.addLayout(top_layout)

        # 2D panel
        frame_2d = QFrame()
        frame_2d.setFrameStyle(QFrame.Shape.StyledPanel)
        lyt_2d = QVBoxLayout(frame_2d)
        lyt_2d.setContentsMargins(0, 0, 0, 0)
        hdr_2d = QLabel("2D Layer View")
        hdr_2d.setStyleSheet("font-weight: bold; padding: 5px;")
        lyt_2d.addWidget(hdr_2d)
        self.view_2d = LayerView2D()
        self.view_2d.layer_changed.connect(self._on_2d_layer_changed)
        lyt_2d.addWidget(self.view_2d)

        if not self.no_3d:
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(frame_2d)

            frame_3d = QFrame()
            frame_3d.setFrameStyle(QFrame.Shape.StyledPanel)
            lyt_3d = QVBoxLayout(frame_3d)
            lyt_3d.setContentsMargins(0, 0, 0, 0)
            hdr_3d = QLabel("3D Build View")
            hdr_3d.setStyleSheet("font-weight: bold; padding: 5px;")
            lyt_3d.addWidget(hdr_3d)
            self.view_3d = BuildView3D()
            self.view_3d.layer_changed.connect(self._on_3d_layer_changed)
            lyt_3d.addWidget(self.view_3d)

            splitter.addWidget(frame_3d)
            splitter.setSizes([800, 800])
            layout.addWidget(splitter)
        else:
            layout.addWidget(frame_2d)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading…")

    def _start_loader(
        self, obf_path: pathlib.Path, max_per_file: int, melt_only: bool
    ) -> None:
        self.loader = LoaderThread(obf_path, self.layer_height_um, max_per_file, melt_only)
        self.loader.progress.connect(self._on_progress)
        self.loader.finished_2d.connect(self._on_2d_loaded)
        if not self.no_3d:
            self.loader.finished_3d.connect(self._on_3d_loaded)
        self.loader.error.connect(self._on_error)
        self.loader.start()

    # ------------------------------------------------------------------
    # Loader callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status.showMessage(f"Loading 3D data: {current}/{total} files")

    def _on_2d_loaded(self, data: list, layer_nums: list, scan_types: list) -> None:
        self.view_2d.set_data(data, layer_nums, scan_types)
        from obfviewer.gpu import gpu_info, using_gpu
        gpu_label = f" | GPU: {gpu_info()}" if using_gpu else ""
        self.status.showMessage(
            f"2D view ready — {len(data)} scans, types: {set(scan_types)}{gpu_label}"
        )

    def _on_3d_loaded(self, lines: np.ndarray, spots: np.ndarray) -> None:
        self.view_3d.set_mesh_data(lines, spots, self.layer_height_um * 1e-3)
        self.progress_bar.setVisible(False)
        self.status.showMessage(f"Ready — {len(lines):,} lines, {len(spots):,} spots in 3D view")

    def _on_error(self, msg: str) -> None:
        self.status.showMessage(f"Error: {msg}")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Loading Error", msg)

    # ------------------------------------------------------------------
    # Layer sync
    # ------------------------------------------------------------------

    def _on_2d_layer_changed(self, layer_num: int) -> None:
        if not self.no_3d and self.sync_layers:
            self.view_3d.set_layer(layer_num)

    def _on_3d_layer_changed(self, layer_num: int) -> None:
        if not self.no_3d and self.sync_layers:
            self.view_2d.set_layer(layer_num)

    # ------------------------------------------------------------------
    # Dark mode
    # ------------------------------------------------------------------

    def _toggle_global_dark_mode(self, dark: bool) -> None:
        self.setStyleSheet(_DARK_STYLESHEET if dark else "")
        if not self.no_3d:
            self.view_3d.dark_check.setChecked(dark)
        self.view_2d.set_dark_mode(dark)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_new_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open OBF File", str(self.obf_path.parent), "OBF Files (*.obf);;All Files (*)"
        )
        if not file_path:
            return

        dark = self.dark_mode_check.isChecked()
        sync = self.sync_check.isChecked() if not self.no_3d else False

        if self.loader.isRunning():
            self.loader.quit()
            self.loader.wait()

        self.obf_path = pathlib.Path(file_path)
        self.setWindowTitle(f"OBF Viewer — {self.obf_path.name}")

        self.view_2d.data = None
        if not self.no_3d:
            self.view_3d.mesh_data = None
            for actor_attr in ("lines_actor", "spots_actor"):
                actor = getattr(self.view_3d, actor_attr, None)
                if actor is not None:
                    self.view_3d.plotter.remove_actor(actor)
                    setattr(self.view_3d, actor_attr, None)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status.showMessage("Loading…")
        self._start_loader(self.obf_path, self.max_per_file, True)

        if not self.no_3d:
            self.sync_check.setChecked(sync)
        self.dark_mode_check.setChecked(dark)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About OBF Viewer",
            "<b>OBF Viewer</b><br><br>"
            "An open-source viewer for Freemelt OBF/OBP build files.<br><br>"
            "Based on <i>obpviewer</i> by Freemelt AB and modifications "
            "by Anton Wiberg (Linköping University).<br><br>"
            "© 2025–2026 Olgierd Nowakowski<br>"
            "Licensed under the Apache License 2.0",
        )

    def closeEvent(self, event) -> None:
        if not self.no_3d:
            self.view_3d.plotter.close()
        super().closeEvent(event)


# =============================================================================
# CLI entry point
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """Entry point for the combined OBF viewer."""
    parser = argparse.ArgumentParser(
        description="Combined 2D/3D OBF build-file viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("obf_file", type=pathlib.Path, help="Path to .obf archive")
    parser.add_argument("--layer-height", type=float, default=70.0, metavar="UM",
                        help="Layer height in micrometres")
    parser.add_argument("--max-per-file", type=int, default=50_000,
                        help="Max 3D geometry elements per file. The 3D view uses GPU-shader"
                             " lines/points so higher values are cheap to render.")
    parser.add_argument("--all-scans", action="store_true",
                        help="Load all scan types, not just melt scans")
    parser.add_argument("--no-3d", action="store_true",
                        help="Disable the 3D viewer (faster startup, lower memory usage)")

    parsed = parser.parse_args(args)

    if not parsed.obf_file.exists():
        print(f"Error: file not found: {parsed.obf_file}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    viewer = CombinedViewer(
        parsed.obf_file,
        layer_height_um=parsed.layer_height,
        max_per_file=parsed.max_per_file,
        melt_only=not parsed.all_scans,
        no_3d=parsed.no_3d,
    )
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
