# External Integrations

**Analysis Date:** 2026-04-09

## APIs & External Services

**Chemistry Data:**
- Basis Set Exchange - Optional basis set database access via `basis-set-exchange` package
- NIST Data - Local chemical element data in `pyscf/data/nist.py` and `pyscf/data/nuclear_g_factor.dat`

## Data Storage

**Databases:**
- HDF5 via h5py >=2.7 - Primary persistent storage for molecular checkpoint files and calculation results
  - Connection: File-based, no server required
  - Client: h5py library
  - Usage: `pyscf/lib/chkfile.py` for checkpoint management

**File Storage:**
- Local filesystem only - No cloud storage integration
- Checkpoint files stored as HDF5 format
- Data files: Element properties, solvent properties in `pyscf/data/`
- Basis set data: `pyscf/gto/basis/` directory contains basis function definitions

**Caching:**
- None - In-memory caching via Python objects during runtime

## Authentication & Identity

**Auth Provider:**
- None - PySCF is a computational library with no user authentication

## Monitoring & Observability

**Error Tracking:**
- None - No remote error tracking integration

**Logs:**
- Python logging module via `pyscf/lib/logger.py`
- Configurable verbosity via `__config__.py` VERBOSE setting
- Default: logger.NOTE level (VERBOSE=3)
- No external log aggregation

## CI/CD & Deployment

**Hosting:**
- GitHub repository: https://github.com/pyscf/pyscf
- PyPI distribution: pyscf package

**CI Pipeline:**
- GitHub Actions for continuous integration:
  - `.github/workflows/ci.yml` - Main CI pipeline
  - `.github/workflows/lint.yml` - Code quality checks (ruff, flake8, pycodestyle)
  - `.github/workflows/ci_conda.yml` - Conda environment testing
  - `.github/workflows/publish.yml` - Package release and distribution
  - Runs on: ubuntu-latest (Python 3.8, 3.12), macos-latest (Python 3.13), aarch64 via Docker
  - Test coverage: Codecov integration with token (CODECOV_TOKEN secret)
  - Coverage target: 75% with 5% threshold
  - Parallel matrix testing with fail-fast: false to catch platform-specific issues

**Release Process:**
- `.github/workflows/release_tag.yml` - Automated release tagging
- Wheel building for multiple platforms (aarch64 via manylinux2014)
- Published to PyPI for pip distribution

## Environment Configuration

**Required env vars:**
- `PYSCF_MAX_MEMORY` - Memory limit (default: 4000 MB)
- `PYSCF_TMPDIR` - Temporary directory location
- `PYSCF_EXT_PATH` - Plugin module paths

**Optional env vars:**
- `PYSCF_CONFIG_FILE` - Configuration file path
- `PYSCF_ARGPARSE` - Enable command-line argument parsing
- `CMAKE_CONFIGURE_ARGS` - CMake configuration options
- `CMAKE_BUILD_ARGS` - CMake build options
- `CMAKE_BUILD_PARALLEL_LEVEL` - Parallel build concurrency
- `CMAKE_OSX_ARCHITECTURES` - macOS architecture targeting
- `OMP_NUM_THREADS` - OpenMP thread limit
- `PYTHONPATH` - Python module search path
- `LDFLAGS` - Linker flags
- `CODECOV_TOKEN` - CI coverage reporting (secret)

**Secrets location:**
- GitHub Actions secrets: CODECOV_TOKEN

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- Codecov webhook for test coverage reporting to GitHub PR comments

## Plugin Architecture

**Extension Points:**
- `PYSCF_EXT_PATH` environment variable supports loading external namespace packages as plugins
- Optional modules can be installed independently: pyscf-forge, pyscf-doci, pyscf-properties, etc.
- Loaded dynamically at runtime in `pyscf/__init__.py`

## External Computational Libraries

**Integral Evaluation:**
- libcint - C library for Gaussian basis function integral evaluation
  - Source: https://github.com/sunqm/libcint
  - Built via CMake external project
  - Provides: Electron repulsion integrals, derivative integrals

**Density Functional Theory:**
- libxc 7.0.0 - Exchange-correlation functional library
  - Source: https://gitlab.com/libxc/libxc/-/archive/7.0.0/
  - Built via CMake external project
  - Provides: Standard DFT functionals (LDA, GGA, meta-GGA, hybrid)

- xcfun - Extended-functional library
  - Source: https://github.com/dftlibs/xcfun (commit a89b783)
  - Applied patch for derivative order support (up to 5)
  - Built via CMake external project
  - Provides: XC functional derivatives

**Linear Algebra Acceleration (Optional):**
- libxsmm 1.17 - Small matrix multiply acceleration
  - Source: https://github.com/hfp/libxsmm
  - Optional, improves tensor contraction performance

**Fourier Transforms (Optional):**
- FFTW 3.3.10 - Fast Fourier Transform library
  - Source: https://www.fftw.org/fftw-3.3.10.tar.gz
  - Optional, enables efficient FFT-based operations in periodic systems
  - Configuration: Single-threaded, shared library build

## System Dependencies

**Required:**
- BLAS library (auto-detected via CMake, required for linear algebra)
- C compiler with C99 support
- OpenMP support

**Linux (CI):**
- openblas-devel
- gcc
- cmake
- curl (for dependency downloads)

**macOS:**
- Xcode Command Line Tools
- Homebrew-provided BLAS/LAPACK

---

*Integration audit: 2026-04-09*
