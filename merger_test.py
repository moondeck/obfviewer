import argparse
import dataclasses
from logging import root
import pathlib
import gzip
import sys
import tkinter
from tkinter import ttk

# Freemelt
from obplib import OBP_pb2 as obp
from obplib import write_obp, read_obp
import obflib

import obanalyser.analyse_build as analyse_build
import obanalyser.plotters.plot_build_data as plot_build_data
import obanalyser.get_build_order as get_build_order
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed


# PyPI

try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "Error: matplotlib is not installed. Try:\n"
        "  $ sudo apt install python3-matplotlib\n"
        "or\n"
        "  $ python3 -m pip install matplotlib"
    )

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from matplotlib.path import Path
import matplotlib.collections as mcoll
import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
from matplotlib.patches import Circle

matplotlib.rcParams['agg.path.chunksize'] = 50000
matplotlib.rcParams['path.simplify'] = False
matplotlib.rcParams['path.simplify_threshold'] = 0.0

plt.style.use("dark_background")

@dataclasses.dataclass
class Data:
    paths: list
    speeds: np.ndarray
    dwell_times: np.ndarray
    spotsizes: np.ndarray
    beampowers: np.ndarray
    syncpoints: dict
    restores: np.ndarray

@dataclasses.dataclass
class scanInfo:
    scan_number: int
    layer_number: int
    scan_type: str

class TimedPoint:
    pass

def merge_obp_objects(obp_objects):
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

def load_obp_objects(file_path):
    obp_objects = []
    print(f"Loading OBP objects from {file_path}")
    with (gzip.open(file_path, 'rb') if file_path.suffix == '.gz' else open(file_path, 'rb')) as f:
        data = f.read()
        pos = 0
        while pos < len(data):
            msg_len, new_pos = _DecodeVarint32(data, pos)
            msg_data = data[new_pos:new_pos + msg_len]
            pos = new_pos + msg_len

            obp_obj = obp.O()
            obp_obj.ParseFromString(msg_data)
            obp_objects.append(obp_obj)
    return obp_objects

# main function
def main():
    file1 = pathlib.Path("layer199melt1.obp")
    file2 = pathlib.Path("layer199melt2.obp")

    obp_objects1 = read_obp(file1)
    obp_objects2 = read_obp(file2)

    merged_objects = merge_obp_objects([*obp_objects1, *obp_objects2])
    merged_file = pathlib.Path("merged.obp")
    write_obp(merged_objects, merged_file)
    print(f"Merged OBP objects saved to {merged_file}")

if __name__ == "__main__":
    main()