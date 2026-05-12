# obfviewer

An open-source viewer for **Freemelt OBF / OBP** build files (electron-beam powder-bed fusion).

The combined viewer provides:

- **2D layer view** — inspect scan paths one-by-one, colour-coded by speed or dwell time  
- **3D build view** — visualise the entire build stack in 3D with layer-by-layer reveal  
- **Synchronised navigation** — scroll layers in either view and the other follows  
- **Dark mode** — toggle a dark theme for the whole UI  
- **Fast loading** — parallel protobuf parsing with per-file deduplication cache

This project is based on the original *obpviewer* by Freemelt AB and enhancements by
Anton Wiberg (Linköping University / [OBPlanner](https://github.com/wiberganton/OBPlanner)).

---

## Requirements

### Python ≥ 3.9

### Freemelt private libraries (not on PyPI)

Install these from your Freemelt distribution **before** installing obfviewer:

| Library | Purpose |
|---------|---------|
| `obplib` | OBP protobuf definitions |
| `obflib` | OBF archive helpers |
| `obanalyser` | Build-order analysis |

### PyPI dependencies (installed automatically)

```
matplotlib  numpy  rich  protobuf  PyQt6  pyvista  pyvistaqt  PyOpenGL
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/moondeck/obfviewer.git
cd obfviewer

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install (editable mode is handy for development)
pip install -e .
```

---

## Usage

### Combined 2D + 3D viewer (recommended)

```bash
obfviewer-combined path/to/build.obf
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--layer-height UM` | `70` | Layer height in micrometres |
| `--max-per-file N` | `50000` | Max 3D geometry elements per file |
| `--all-scans` | off | Include all scan types (not just melt) |
| `--no-3d` | off | Disable the 3D view (faster startup) |

### 2D-only CLI viewer

```bash
obfviewer path/to/build.obf [--slice-size N] [--index N] [--no-melt]
```

### Run directly without installing

```bash
python obf_viewer_combined.py path/to/build.obf
```

---

## Keyboard shortcuts (2D viewer)

| Key | Action |
|-----|--------|
| `→` / `p` | Next path (+1) |
| `←` / `n` | Previous path (−1) |
| `Shift+→` | +10 paths |
| `Ctrl+→` | +100 paths |
| `Alt+→` | +1000 paths |
| `a` | Jump to first path |
| `e` | Jump to last path |
| `r` | Jump to next restore point |
| `b` | Jump to next beam-power change |
| `s` | Jump to next spot-size change |
| `0`–`9` | Jump to next sync-point change |

---

## Project layout

```
src/obfviewer/
    combined.py     ← Combined 2D/3D viewer (main application)
    cli.py          ← 2D-only CLI entry point
    loaders.py      ← Parallel OBP file loading
    models.py       ← Data model dataclasses
    utils.py        ← OBF archive extraction, build-order helpers
    gui/
        app.py      ← Tkinter app runner (legacy 2D viewer)
        viewer.py   ← Tkinter ObpFrame widget (legacy 2D viewer)
tests/
    fixtures/       ← Small OBP sample files for tests
scripts/
    merge_obp.py    ← Utility: merge two OBP files into one
    legacy/         ← Prototype scripts (superseded by src/obfviewer)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## GPU acceleration (experimental)

See the `feature/gpu-acceleration` branch for optional CuPy-based acceleration of
array operations. Requires a CUDA (NVIDIA) or ROCm (AMD) runtime.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Original *obpviewer* © 2022 Freemelt AB.  
Modifications © 2025–2026 Olgierd Nowakowski.
