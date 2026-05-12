# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""OBP Viewer Frame widget."""

from __future__ import annotations

import tkinter
from tkinter import ttk
from typing import TYPE_CHECKING

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import EngFormatter

if TYPE_CHECKING:
    from obfviewer.models import Data


class ObpFrame(ttk.Frame):
    """Main viewer frame for OBP data visualization."""

    def __init__(
        self,
        master: tkinter.Tk,
        data: list[Data],
        layer_count: int,
        build_info: dict | None = None,
        layer_numbers: list[int] | None = None,
        scan_types: list[str] | None = None,
        slice_size: int = 15000,
        index: int | None = None,
        **kwargs,
    ):
        """Initialize the OBP viewer frame.

        Args:
            master: Parent Tk window
            data: List of Data objects for each layer
            layer_count: Total number of data entries (scans)
            build_info: Optional build analysis information
            layer_numbers: List mapping each data index to its physical layer number (1-indexed)
            scan_types: List mapping each data index to its scan type (e.g., "melt", "preheat")
            slice_size: Number of paths to display at once
            index: Initial path index
            **kwargs: Additional arguments for ttk.Frame
        """
        super().__init__(master, **kwargs)

        self.data = data
        self.layer_index = 0  # Internal 0-indexed position in data list
        self.layer_count = layer_count
        self.build_info = build_info
        
        # layer_numbers maps data index -> physical layer number (1-indexed)
        # Default to sequential 1-indexed if not provided
        self.layer_numbers = layer_numbers if layer_numbers else list(range(1, len(data) + 1))
        self.max_physical_layer = max(self.layer_numbers) if self.layer_numbers else 1
        
        # scan_types maps data index -> scan type name
        self.scan_types = scan_types if scan_types else ["unknown"] * len(data)

        index = index if index is not None else slice_size
        self._create_cap_function()
        index = self.cap(index)
        slice_ = slice(self.cap(index + 1 - slice_size), self.cap(index) + 1)

        # Create figure and axes
        fig = Figure(figsize=(9, 8), constrained_layout=True)
        ax = fig.add_subplot(111)

        # Setup axes
        self._setup_axes(ax)

        # Create path collection
        speeds = self.data[self.layer_index].speeds
        vmax = float(max(speeds)) if len(speeds) > 0 else 1.0
        self.path_collection = mcoll.PathCollection(
            self.data[self.layer_index].paths[slice_],
            facecolors="none",
            transform=ax.transData,
            cmap=plt.cm.rainbow,
            norm=plt.Normalize(vmin=0, vmax=vmax),
        )
        self.path_collection.set_array(self.data[self.layer_index].speeds[slice_])
        ax.add_collection(self.path_collection)

        # Add colorbar
        cbar = fig.colorbar(
            self.path_collection, ax=ax, pad=0, aspect=60, format=EngFormatter(unit="m/s")
        )
        cbar.ax.tick_params(axis="y", labelsize=8)

        # Create marker for current position (hidden when layer has no paths)
        paths = self.data[self.layer_index].paths
        if paths:
            seg = paths[index]
            self.marker = ax.scatter(*seg.vertices[-1], c="white", marker="*", zorder=2)
        else:
            self.marker = ax.scatter(0, 0, c="white", marker="*", zorder=2, visible=False)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.mpl_connect("key_press_event", self._on_keypress)

        # Create control variables
        self._slice_size = tkinter.IntVar(value=slice_size)
        self._index = tkinter.IntVar(value=index)
        # Slider shows physical layer number (1-indexed)
        self._layer_index = tkinter.IntVar(value=self.layer_numbers[self.layer_index])

        # Create widgets
        self._create_widgets()

    def _create_cap_function(self) -> None:
        """Create the index capping function for current layer."""
        self.cap = lambda i: max(
            0, min(len(self.data[self.layer_index].paths) - 1, int(i))
        )

    def _setup_axes(self, ax) -> None:
        """Setup the matplotlib axes with formatting."""
        ax.axhline(0, linewidth=1, zorder=0)
        ax.axvline(0, linewidth=1, zorder=0)
        ax.add_patch(Circle((0, 0), 0.04, edgecolor="white", facecolor="none"))
        ax.add_patch(
            Circle((0, 0), 0.05, edgecolor="grey", facecolor="none", linestyle="--")
        )
        ax.set_xlim([-0.05, 0.05])
        ax.set_ylim([-0.05, 0.05])

        si_meter = EngFormatter(unit="m")
        ax.xaxis.set_major_formatter(si_meter)
        ax.yaxis.set_major_formatter(si_meter)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Slice size spinbox
        self._slice_size_spinbox = ttk.Spinbox(
            self,
            from_=0,
            to=len(self.data[self.layer_index].paths) - 1,
            textvariable=self._slice_size,
            command=self.update_index,
            width=6,
        )
        self._slice_size_spinbox.bind("<KeyRelease>", self.update_index)

        # Layer control frame
        self.layer_control_frame = ttk.Frame(self)
        self._layer_index_scale = tkinter.Scale(
            self.layer_control_frame,
            from_=1,  # Physical layers are 1-indexed
            to=self.max_physical_layer,
            orient=tkinter.HORIZONTAL,
            variable=self._layer_index,
            command=self.update_layer_index,
        )
        self.layer_control_frame.grid_columnconfigure(0, weight=1)
        self._layer_index_scale.grid(row=0, column=0, sticky="EW", padx=(0, 6))

        # Navigation buttons
        self._prev_layer_btn = ttk.Button(
            self.layer_control_frame, text="◀", width=3, command=self.prev_layer
        )
        self._next_layer_btn = ttk.Button(
            self.layer_control_frame, text="▶", width=3, command=self.next_layer
        )
        self._prev_layer_btn.grid(row=0, column=1, sticky="E")
        self._next_layer_btn.grid(row=0, column=2, sticky="E")
        
        # Scan type selector
        self._scan_type_var = tkinter.StringVar()
        self._scan_type_label = ttk.Label(self.layer_control_frame, text="Scan:")
        self._scan_type_combo = ttk.Combobox(
            self.layer_control_frame,
            textvariable=self._scan_type_var,
            state="readonly",
            width=12,
        )
        self._scan_type_combo.bind("<<ComboboxSelected>>", self._on_scan_type_selected)
        self._scan_type_label.grid(row=0, column=3, sticky="E", padx=(10, 2))
        self._scan_type_combo.grid(row=0, column=4, sticky="E")
        
        # Initialize scan type options for current layer
        self._update_scan_type_options()

        # Index controls
        self._index_scale = tkinter.Scale(
            self,
            from_=0,
            to=len(self.data[self.layer_index].paths) - 1,
            orient=tkinter.HORIZONTAL,
            variable=self._index,
            command=self.update_index,
        )
        self._index_spinbox = ttk.Spinbox(
            self,
            from_=0,
            to=len(self.data[self.layer_index].paths) - 1,
            textvariable=self._index,
            command=self.update_index,
            width=6,
        )
        self._index_spinbox.bind("<KeyRelease>", self.update_index)

        # Info label
        self.info_value = tkinter.StringVar(
            value=",  ".join(self._get_info(self._index.get()))
        )
        self.info_label = ttk.Label(self, textvariable=self.info_value)

        # Quit button
        self.button_quit = ttk.Button(self, text="Quit", command=self.master.quit)

        # Toolbar
        self.toolbar_frame = ttk.Frame(master=self)
        NavigationToolbar2Tk(self.canvas, self.toolbar_frame).update()

    def _get_info(self, index: int) -> list[str]:
        """Get info strings for the current path.

        Args:
            index: Path index

        Returns:
            List of formatted info strings
        """
        layer_data = self.data[self.layer_index]
        info = [f"{k}={v[index]}" for k, v in layer_data.syncpoints.items()]
        info.append(f"Restore={int(layer_data.restores[index])}")
        info.append(f"BeamPower(W)={int(layer_data.beampowers[index])}")
        info.append(f"SpotSize(μm)={int(layer_data.spotsizes[index])}")
        info.append(f"Speed(m/s)={layer_data.speeds[index]:.3f}")
        info.append(f"DwellTime(ms)={layer_data.dwell_times[index]:.5f}")
        return info

    def _get_scans_for_layer(self, physical_layer: int) -> list[tuple[int, str]]:
        """Get all scan indices and types for a given physical layer.
        
        Args:
            physical_layer: The 1-indexed physical layer number
            
        Returns:
            List of (data_index, scan_type) tuples for scans in this layer
        """
        scans = []
        for i, (layer_num, scan_type) in enumerate(zip(self.layer_numbers, self.scan_types)):
            if layer_num == physical_layer:
                scans.append((i, scan_type))
        return scans
    
    def _update_scan_type_options(self) -> None:
        """Update the scan type combobox options for the current physical layer."""
        physical_layer = self._layer_index.get()
        scans = self._get_scans_for_layer(physical_layer)
        
        # Build options list - include index for duplicate scan types (e.g., melt1, melt2)
        type_counts: dict[str, int] = {}
        options = []
        self._scan_type_to_index: dict[str, int] = {}  # Maps display name -> data index
        
        for data_idx, scan_type in scans:
            # Count occurrences of each type for numbering
            count = type_counts.get(scan_type, 0) + 1
            type_counts[scan_type] = count
            
            # Create display name
            display_name = f"{scan_type} {count}" if count > 1 or type_counts[scan_type] != count else scan_type
            options.append(display_name)
            self._scan_type_to_index[display_name] = data_idx
        
        # Handle case where same type appears multiple times - add numbers retroactively
        if any(c > 1 for c in type_counts.values()):
            # Rebuild with numbered entries
            type_counts = {}
            options = []
            self._scan_type_to_index = {}
            
            for data_idx, scan_type in scans:
                count = type_counts.get(scan_type, 0) + 1
                type_counts[scan_type] = count
                display_name = f"{scan_type} {count}"
                options.append(display_name)
                self._scan_type_to_index[display_name] = data_idx
        
        self._scan_type_combo["values"] = options
        
        # Select default: prefer "melt" if available, otherwise first option
        if options:
            # Try to find a melt option
            melt_option = None
            for display_name in options:
                if display_name.lower().startswith("melt"):
                    melt_option = display_name
                    break
            
            if melt_option:
                self._scan_type_var.set(melt_option)
            else:
                self._scan_type_var.set(options[0])
    
    def _on_scan_type_selected(self, _=None) -> None:
        """Handle scan type selection from combobox."""
        selected = self._scan_type_var.get()
        if selected and selected in self._scan_type_to_index:
            new_index = self._scan_type_to_index[selected]
            if new_index != self.layer_index:
                self.layer_index = new_index
                self._apply_layer_change()

    def update_index(self, _=None) -> None:
        """Update the display for the current path index."""
        index = self.cap(self._index.get())
        ss = self._slice_size.get() or 1
        slice_ = slice(self.cap(index + 1 - ss), self.cap(index) + 1)
        segs = self.data[self.layer_index].paths[slice_]

        if segs:
            self.path_collection.set_paths(segs)
            self.path_collection.set_array(self.data[self.layer_index].speeds[slice_])
            self.marker.set_offsets(segs[-1].vertices[-1])
            self.canvas.draw()

        self.info_value.set(",  ".join(self._get_info(index)))

    def update_layer_index(self, _=None) -> None:
        """Update the display for the current physical layer."""
        # Get the selected physical layer number (1-indexed)
        physical_layer = self._layer_index.get()
        
        # Update scan type options for the new layer
        self._update_scan_type_options()
        
        # Find the first data index that belongs to this physical layer
        for i, layer_num in enumerate(self.layer_numbers):
            if layer_num == physical_layer:
                self.layer_index = i
                break
        
        # Check if combo has selection, use that index instead
        selected = self._scan_type_var.get()
        if selected and selected in self._scan_type_to_index:
            self.layer_index = self._scan_type_to_index[selected]
        
        self._apply_layer_change()
    
    def _apply_layer_change(self) -> None:
        """Apply changes after switching to a new layer/scan."""
        physical_layer = self.layer_numbers[self.layer_index]
        scan_type = self.scan_types[self.layer_index]
        
        self._create_cap_function()
        self.master.title(f"OBP Viewer - Layer {physical_layer} ({scan_type})")

        self._index_scale.config(to=len(self.data[self.layer_index].paths) - 1)
        if self._index.get() != len(self.data[self.layer_index].paths):
            self._index.set(len(self.data[self.layer_index].paths) - 1)

        print(
            f"Switched to layer {physical_layer} ({scan_type}) "
            f"with {len(self.data[self.layer_index].paths)} paths."
        )
        self.update_index()

    def prev_layer(self) -> None:
        """Navigate to the previous layer."""
        v = max(1, self._layer_index.get() - 1)  # Minimum is 1 (1-indexed)
        self._layer_index.set(v)
        self.update_layer_index()

    def next_layer(self) -> None:
        """Navigate to the next layer."""
        upper = int(self._layer_index_scale.cget("to"))  # Already 1-indexed
        v = min(upper, self._layer_index.get() + 1)
        self._layer_index.set(v)
        self.update_layer_index()

    def _on_keypress(self, event) -> None:
        """Handle keyboard navigation.

        Args:
            event: Matplotlib key press event
        """
        key = event.key.lower()
        stepsize = {"": 1, "shift": 10, "ctrl": 100, "alt": 1000}.get(
            event.key.split("+")[0], 1
        )

        if key in {"right", "p"}:
            self._index.set(self.cap(self._index.get() + stepsize))
        elif key in {"left", "n"}:
            self._index.set(self.cap(self._index.get() - stepsize))
        elif key == "a":
            self._index.set(0)
        elif key == "e":
            self._index.set(len(self.data[self.layer_index].paths) - 1)
        elif key in "0123456789":
            n = int(key)
            for i, k in enumerate(self.data[self.layer_index].syncpoints):
                if i + 1 == n:
                    self._next_different(self.data[self.layer_index].syncpoints[k])
        elif key == "r":
            self._next_different(self.data[self.layer_index].restores)
        elif key == "b":
            self._next_different(self.data[self.layer_index].beampowers)
        elif key == "s":
            self._next_different(self.data[self.layer_index].spotsizes)

        self.update_index()

    def _next_different(self, array: np.ndarray) -> None:
        """Jump to the next index where the array value changes.

        Args:
            array: Array to search for value changes
        """
        start = self.cap(self._index.get())
        diff = array[start:] != array[start]
        if diff.any():
            self._index.set(start + np.argmax(diff))

    def setup_grid(self) -> None:
        """Configure the grid layout for all widgets."""
        self.canvas.get_tk_widget().grid(row=0, columnspan=4, sticky="NSWE")
        self.layer_control_frame.grid(row=1, columnspan=4, sticky="NSWE")
        self._index_scale.grid(row=2, columnspan=4, sticky="NSWE")
        self.info_label.grid(row=3, column=0, sticky="SW")
        self._slice_size_spinbox.grid(row=3, column=1, sticky="SE")
        self._index_spinbox.grid(row=3, column=2, sticky="SE")
        self.button_quit.grid(row=3, column=3, sticky="SE")
        self.toolbar_frame.grid(row=4, columnspan=4, sticky="NSWE")
