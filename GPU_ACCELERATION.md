# GPU Acceleration

This branch (`feature/gpu-acceleration`) adds **optional** GPU-accelerated array
operations to the data-loading pipeline.

## What is accelerated

| Operation | Normal | With GPU |
|-----------|--------|----------|
| Vertex coordinate scaling (`× 1e-6`) | numpy | CuPy (GPU) |
| Array stacking (lines, curves, spots) | numpy | CuPy (GPU) |
| Speed/power/spotsize array creation | numpy | CuPy (GPU) |
| Protobuf parsing | CPU (required) | CPU (unchanged) |
| Matplotlib Path object creation | CPU (required) | CPU (unchanged) |
| 3D VTK/PyVista rendering | GPU via OpenGL | GPU via OpenGL |

## Why not full GPU parsing?

Protobuf parsing in Python is inherently sequential and CPU-bound.  The
`ProcessPoolExecutor` in `load_obp_files_parallel` provides parallelism there.
The GPU acceleration targets the secondary bottleneck: building the large numpy
arrays from the parsed results.

## Installation

1. Install a CuPy wheel matching your GPU driver:

   ```bash
   # NVIDIA CUDA 12.x
   pip install cupy-cuda12x

   # NVIDIA CUDA 11.x
   pip install cupy-cuda11x

   # AMD ROCm 5.0
   pip install cupy-rocm-5-0
   ```

2. Install obfviewer with the GPU extras:

   ```bash
   pip install -e ".[gpu]"
   ```

3. Run as usual — if CuPy initialises successfully the status bar will show the
   active GPU name.  Set `OBFVIEWER_NO_GPU=1` to force CPU-only mode.

## Key module: `obfviewer.gpu`

```python
from obfviewer.gpu import xp, using_gpu, gpu_info, to_numpy

# xp is cupy if GPU is available, numpy otherwise
arr = xp.array([1.0, 2.0, 3.0])
result = xp.sqrt(arr)

# Always convert back to numpy before passing to matplotlib / VTK
numpy_arr = to_numpy(result)

print(using_gpu)   # True / False
print(gpu_info())  # "NVIDIA GeForce RTX 3060 — 12.0 GB VRAM, CUDA 12.2"
```

## Fallback behaviour

If CuPy is not installed, or the GPU runtime fails to initialise, the module
silently falls back to numpy.  A `RuntimeWarning` is emitted only when a GPU
was detected but could not be initialised (e.g. driver mismatch).
