# Technology Stack

**Analysis Date:** 2026-04-09

## Languages

**Primary:**
- Python 3.13+ - Main language for scientific computing framework
- C 99 - Low-level computational kernels and extensions

**Secondary:**
- CMake - Build system for C/Fortran extensions

## Runtime

**Environment:**
- CPython 3.13+ (supports 3.8-3.14 per pyproject.toml)

**Package Manager:**
- pip / setuptools
- Conda (conda_build_config.yaml supports Python 3.8-3.13)
- Lockfile: Not enforced (dependencies specified in pyproject.toml)

## Frameworks

**Core:**
- setuptools 61.0+ - Python package building and distribution

**Testing:**
- pytest - Test runner and framework
- pytest-cov - Code coverage measurement
- pytest-timer - Test timing statistics
- flake8 3.7.0+ - Python linting
- ruff - Fast Python linter and formatter
- pycodestyle - Style conformance checking

**Build/Dev:**
- CMake 3.22+ - C extension build system
- wheel - Binary package distribution

## Key Dependencies

**Critical:**
- numpy >=1.13,!=1.16,!=1.17 - Array computing and numerical operations
- scipy >=1.6.0 - Scientific computing algorithms
- h5py >=2.7 - HDF5 file format I/O for molecular data
- scipy !=1.5.0,!=1.5.1 - Known bug exclusions

**Infrastructure (C Libraries via CMake):**
- libcint - Gaussian integral library for quantum chemistry (GIT: https://github.com/sunqm/libcint)
- libxc 7.0.0 - Density functional approximations exchange-correlation library
- xcfun - Extended-functional library for DFT (GIT: https://github.com/dftlibs/xcfun, commit a89b783)
- libxsmm 1.17 - Small matrix multiply library for performance (optional)
- FFTW 3.3.10 - Fast Fourier Transform library (optional)
- OpenMP - Parallel computation support (auto-detected, enabled by default)

**Optional Plugin Dependencies:**
- pyscf-forge - Advanced functionality plugin
- pyberny >=0.6.2 - Geometry optimization
- geometric >=0.9.7.2 - Advanced molecular structure optimization
- pyscf-qsdopt - QSD optimizer
- pyscf-doci - Doubly occupied CI
- pyscf-properties - Property calculations
- pyscf-semiempirical - Semiempirical methods
- cppe - Coupled Cluster for Polarizable Embedding
- pyqmc - Quantum Monte Carlo
- basis-set-exchange - Basis set database access
- pyscf-dispersion - Dispersion corrections
- coupled-cluster-py - Coupled cluster methods

**BLAS/LAPACK:**
- System BLAS provider (Intel MKL, OpenBLAS, or system LAPACK) - Required for linear algebra operations

## Configuration

**Environment:**
- Configuration via `.pyscf_conf.py` file in current directory or home directory
- Environment variable override: `PYSCF_CONFIG_FILE`
- Key environment variables:
  - `PYSCF_MAX_MEMORY` - Memory limit in MB (default: 4000)
  - `PYSCF_TMPDIR` - Temporary directory (default: system temp dir)
  - `PYSCF_ARGPARSE` - Enable argument parsing (default: False)
  - `PYSCF_EXT_PATH` - Path to plugin modules
  - `CMAKE_CONFIGURE_ARGS` - Custom CMake configuration
  - `CMAKE_BUILD_ARGS` - Custom CMake build arguments
  - `CMAKE_OSX_ARCHITECTURES` - macOS architecture targeting
  - `CMAKE_BUILD_PARALLEL_LEVEL` - Parallel build jobs (note: high levels can cause OOM)

**Build:**
- `pyproject.toml` - Modern Python packaging configuration
- `setup.py` - Custom build commands (CMakeBuildPy class)
- `CMakeLists.txt` at `pyscf/lib/CMakeLists.txt` - C extension build configuration
- `pytest.ini` - Test runner configuration
- `.ruff.toml` - Ruff linter configuration
- `.flake8` - Flake8 linter configuration
- `.coveragerc` - Code coverage configuration

## Platform Requirements

**Development:**
- CMake 3.22 or later
- C compiler (GCC, Clang, or Intel compiler)
- BLAS library (Intel MKL, OpenBLAS, or system LAPACK)
- OpenMP-compatible compiler
- Git (for downloading external dependencies)

**Tested Platforms:**
- Linux (Ubuntu latest, aarch64 via Docker/QEMU)
- macOS (latest, with universal2 support for Apple Silicon and Intel)
- Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

**Production:**
- Linux/macOS with system BLAS/LAPACK
- Python 3.13+ required per pyproject.toml
- Memory: 4GB default allocation (configurable via PYSCF_MAX_MEMORY)

---

*Stack analysis: 2026-04-09*
