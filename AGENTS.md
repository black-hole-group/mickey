# AGENTS.md: Coding Guidelines for Mickey

Mickey is a Python package with C extensions for analyzing Pluto MHD code output.

## Build Commands

### C Extensions (GPU/CPU acceleration)
```bash
cd src
make              # Build with GPU support (NVIDIA)
make CPU=1        # Build CPU-only version
make clean        # Clean build artifacts
```

### Python Package
```bash
python setup.py install   # Standard install
python setup.py develop   # Symlink install (recommended for development)
pip install -e .          # Editable install
```

### Environment Setup
```bash
export PLUTO_DIR=/path/to/pluto
export PYTHONPATH=/path/to/pyPLUTO:/path/to/mickey/src:$PYTHONPATH
```

## Testing

### Run Tests
```bash
./mickey/tests           # Binary test executable
python src/testKernel.py # OpenCL kernel test
```

**Note**: No pytest/unittest framework. Tests are manual/binary executables.

## Code Style Guidelines

### Python

**Imports**:
- Group: standard lib, third-party, local
- Use `import numpy` not `import numpy as np`
- Use `import matplotlib.pyplot as pylab` (project convention)
- Example:
```python
import numpy
import os
import matplotlib.pyplot as pylab
from scipy import ndimage
import scipy.interpolate
import pyPLUTO as pp
import nmmn.misc
```

**Formatting**:
- **Tabs for indentation** (not spaces)
- No line length limit enforced
- Docstrings use triple quotes with `:param:` notation

**Naming**:
- Functions: `camelCase` (e.g., `angleAvg`, `getmdot`)
- Classes: `PascalCase` (e.g., `Pluto`)
- Variables: lowercase with descriptive names
- Private methods: no underscore prefix (project style)

**Documentation**:
```python
def functionName(param1, param2):
    """
    Brief description.
    
    :param param1: Description of param1
    :param param2: Description of param2
    :returns: Description of return value
    
    Example:
    >>> result = functionName(value1, value2)
    """
```

**Error Handling**:
```python
try:
    f = open("pluto.ini","r")
except IOError as e: 
    print(e)
```

### C Code

**Style**:
- Use tabs for indentation
- Braces on new lines (Allman style)
- OpenACC pragmas for GPU acceleration:
```c
#pragma acc routine seq
#pragma acc data copyout(...) copyin(...)
#pragma acc parallel loop collapse(2)
```

**Comments**:
- Use `/* */` for multi-line, `//` for single-line allowed
- Document function purpose and parameters

## Project Structure

```
mickey/
├── mickey/          # Python package
│   ├── __init__.py
│   ├── mickey.py    # Main Pluto class
│   ├── plot.py      # Plotting functions
│   ├── hplots.py    # History/1D plots
│   └── threed.py    # 3D visualization
├── src/             # C extensions
│   ├── fastregrid.c # OpenACC C code
│   ├── fastregrid.h
│   ├── fastregrid.i # SWIG interface
│   └── Makefile
├── examples/        # Jupyter notebook tutorial
├── setup.py
└── requirements.txt
```

## Dependencies

- `pyPLUTO`: Read Pluto binary files
- `nmmn`: Scientific utilities (`nmmn.misc`, `nmmn.lsd`)
- `numpy`, `scipy`, `matplotlib`
- Optional: PGI compiler (`pgcc`), SWIG, PyOpenCL

## Git Workflow

- Main branch: `master`
- Feature branches: `openacc`, `swig`, `openmp`, `opencl`
- This is an older project (2017-2020); uses simple git workflow

## Important Notes

1. **GPU acceleration**: Code uses OpenACC for NVIDIA GPUs; can fallback to CPU
2. **SWIG bindings**: C extensions use SWIG to generate Python bindings
3. **Scientific code**: Focus on correctness over style; prefer readability
4. **Pluto integration**: Requires `pyPLUTO` and environment variables set
