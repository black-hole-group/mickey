# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Mickey is

Mickey is a Python toolkit for analyzing and visualizing MHD simulations from the [Pluto code](https://github.com/black-hole-group/pluto). It wraps `pyPLUTO` (binary file reader) with physics computations (accretion rates, mass, angular momentum) and coordinate transformations between spherical and Cartesian grids, including optional GPU acceleration via OpenACC.

## Build Commands

```bash
# Install (recommended for development — picks up source edits immediately)
python setup.py develop

# C extensions (GPU/CPU acceleration for regridding)
cd src && make          # GPU build (requires PGI compiler pgcc + NVIDIA GPU)
cd src && make CPU=1    # CPU-only build
cd src && make clean    # Remove build artifacts
```

The `setup.py develop` call also runs `make -C src` automatically.

Required environment variables (add to shell profile):
```bash
export PLUTO_DIR=/path/to/pluto
export PYTHONPATH=/path/to/pyPLUTO:/path/to/mickey/src:$PYTHONPATH
```

## Testing

No pytest/unittest framework. Tests are manual:
```bash
./mickey/tests           # Binary test executable
python src/testKernel.py # OpenCL kernel test
```

## Architecture

```
mickey/mickey.py   — Pluto class (central object): loads simulation frames via pyPLUTO,
                     computes derived fields (entropy, Mach, Bernoulli), accretion rates,
                     and performs coordinate regridding
mickey/plot.py     — 2D plotting functions (density, mesh, streamplot) using pcolormesh
mickey/hplots.py   — 1D/history plot helpers
mickey/threed.py   — 3D visualization via mayavi (cut planes, volume rendering)
src/fastregrid.c   — OpenACC C kernel for fast spherical→Cartesian regridding
src/fastregrid.i   — SWIG interface file that generates Python bindings
```

**Data flow**: `pyPLUTO.pload(i)` → `Pluto(i)` stores raw fields (`rho`, `p`, `v1/v2/v3`) plus derived 2D meshgrid arrays (`X1`, `X2`, `X`, `Y`) and computed quantities (`mdot`, `mdotin`, `mdotout`, `mass`, `cs`, `mach`, `entropy`, `Be`). Coordinate transformation (`regrid` / `regridFast` / `regridGPU`) returns a *new* empty `Pluto()` object populated with Cartesian-grid arrays.

**Regridding variants**:
- `regrid()` — pure Python, slow (nested loops)
- `regridFast()` — calls the SWIG-wrapped C extension `fastregrid`
- `regridGPU()` — uses PyOpenCL kernel (`src/fastregrid.cl`) for NVIDIA GPU (~100× speedup vs pure Python)

`getgamma()` reads `pluto.ini` from the current working directory; scripts must be run from the simulation directory or have `pluto.ini` accessible.

## Code Conventions

- **Tabs** for indentation (not spaces)
- `import numpy` (not `import numpy as np`)
- `import matplotlib.pyplot as pylab` (project convention)
- Function names: `camelCase` (`angleAvg`, `getmdot`, `regridFast`)
- Class names: `PascalCase` (`Pluto`)
- Docstrings use `:param name:` / `:returns:` notation with a `>>> example` line
