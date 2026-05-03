# Codebase Structure

**Analysis Date:** 2026-04-09

## Directory Layout

```
pyscf_rs/
├── pyscf/                   # Main package source code
│   ├── __init__.py          # Package initialization and module imports
│   ├── __config__.py        # Default configuration parameters
│   ├── __all__.py           # Public API exports
│   ├── post_scf.py          # Convenience imports for post-SCF methods
│   ├── gto/                 # Gaussian Type Orbitals (molecular basis sets)
│   ├── scf/                 # Self-consistent field methods (HF, DFT base)
│   ├── dft/                 # Density functional theory
│   ├── ao2mo/               # AO to MO integral transformation
│   ├── cc/                  # Coupled-cluster methods
│   ├── mp/                  # Møller-Plesset perturbation theory
│   ├── fci/                 # Full configuration interaction
│   ├── ci/                  # General configuration interaction
│   ├── agf2/                # Algebraic Green's function theory
│   ├── adc/                 # Algebraic diagrammatic construction
│   ├── gw/                  # GW approximation
│   ├── tdscf/               # Time-dependent SCF
│   ├── tddft/               # Time-dependent DFT
│   ├── grad/                # Analytical gradients
│   ├── hessian/             # Analytical Hessians
│   ├── x2c/                 # Exact two-component relativistic methods
│   ├── solvent/             # Solvent effects (PCM, CPCM)
│   ├── geomopt/             # Geometry optimization interface
│   ├── mcscf/               # Multi-configurational SCF
│   ├── mcpdft/              # Multi-configurational pair-density functional theory
│   ├── mrpt/                # Multi-reference perturbation theory
│   ├── symm/                # Point group symmetry
│   ├── pbc/                 # Periodic boundary conditions
│   ├── qmmm/                # QM/MM interface
│   ├── soscf/               # Second-order SCF
│   ├── sgx/                 # Screened exact exchange
│   ├── nac/                 # Non-adiabatic coupling
│   ├── eph/                 # Electron-phonon coupling
│   ├── lo/                  # Localized orbitals (IAO, IBO, Pipek-Mezey, etc.)
│   ├── tools/               # Utility tools and wavefunction analysis
│   ├── data/                # Atomic and nuclear data
│   ├── lib/                 # Low-level utilities and C extensions
│   └── [module]/test/       # Test directories parallel to module structure
├── examples/                # Example scripts organized by method
├── doc_legacy/              # Legacy documentation
├── docker/                  # Docker configuration files
├── tools/                   # Project-level utility scripts
├── .github/                 # GitHub workflows (CI/CD)
├── .planning/               # GSD planning and analysis documents
├── pyproject.toml           # Modern Python project configuration
├── setup.py                 # setuptools configuration with CMake build
├── pytest.ini               # pytest configuration
└── [config files]           # .ruff.toml, .flake8, .coveragerc, etc.
```

## Directory Purposes

**pyscf/gto/**
- Purpose: Molecular geometry and Gaussian type orbital basis sets
- Contains: Mole class, basis loading, integral evaluation interface
- Key files: `mole.py`, `moleintor.py`, `eval_gto.py`, `ecp.py`, `basis/`
- Test: `gto/test/test_mole.py`, `test_moleintor.py`

**pyscf/scf/**
- Purpose: Hartree-Fock and self-consistent field methods
- Contains: RHF, UHF, GHF, RHF with symmetry, UHF with symmetry
- Key files: `hf.py` (main kernel), `_vhf.py` (Fock matrix), `diis.py`, `addons.py`
- Test: `scf/test/test_rhf.py`, `test_uhf.py`

**pyscf/dft/**
- Purpose: Kohn-Sham density functional theory
- Contains: RKS, UKS, GKS with various xc functionals
- Key files: `rks.py`, `uks.py`, `numint.py` (grid integration), `gen_grid.py`, `libxc.py`
- Test: `dft/test/` - multiple test files for functionals and grids

**pyscf/ao2mo/**
- Purpose: Atomic orbital to molecular orbital integral transformation
- Contains: In-core and out-of-core transformation algorithms
- Key files: `kernel.py`, `incore.py`, `outcore.py`
- Test: `ao2mo/test/test_incore.py`, `test_outcore.py`

**pyscf/cc/**
- Purpose: Coupled-cluster methods (CCSD, RCCSD, UCCSD, etc.)
- Contains: Ground-state and excited-state CC methods
- Key files: `ccsd.py`, `rccsd.py`, `uccsd.py`
- Test: `cc/test/test_ccsd*.py`

**pyscf/mp/**
- Purpose: Møller-Plesset perturbation theory
- Contains: MP2, MP3, MP4 implementations
- Key files: `mp2.py`, `mp3.py`, etc.
- Test: `mp/test/test_mp2*.py`

**pyscf/fci/**
- Purpose: Full configuration interaction
- Contains: FCI solver, selected CI (SCI), DCI
- Key files: `direct_spin0.py`, `direct_spin1.py`
- Test: `fci/test/test_fci*.py`

**pyscf/agf2/**
- Purpose: Algebraic Green's function approximation
- Contains: AGF2(0,0), AGF2(1,0), etc. implementations
- Key files: `ragf2.py`, `uagf2.py`, `dfragf2.py`
- Test: `agf2/test/test_ragf2*.py`, `test_uagf2*.py`

**pyscf/lib/**
- Purpose: Low-level utilities, C extension wrappers, linear algebra
- Contains: numpy helpers, linalg wrappers, logging, checkpoint I/O
- Key subdirs:
  - `numpy_helper.py`: einsum variants, array operations
  - `linalg_helper.py`: eigendecomposition, linear system solvers
  - `misc.py`: general utilities
  - `logger.py`: logging system
  - `diis.py`: convergence acceleration
  - `chkfile.py`: HDF5 I/O
  - `vhf/`: Virtual Hartree-Fock (2-electron integrals) - C code
  - `np_helper/`: NumPy performance helpers - C code
  - `gto/`: Integral evaluation - C code (libcint interface)
  - `dft/`: DFT grid and functional evaluation - C code (libxc interface)
  - `ao2mo/`: Integral transformation - C code
- Test: `lib/test/`

**pyscf/tools/**
- Purpose: Wavefunction analysis, format conversion, utilities
- Contains: Orbital analysis, molecular properties, format I/O
- Key files: Various utility modules
- Test: `tools/test/`

**pyscf/data/**
- Purpose: Atomic and nuclear constants
- Contains: Element data, nuclear spin constants from NIST
- Key files: Database files

**pyscf/pbc/**
- Purpose: Periodic boundary conditions (crystal systems)
- Contains: Mirrored structure of main modules (pbc/scf, pbc/dft, etc.)
- Key files: Domain-specific for k-point sampling
- Test: Multiple test directories

**pyscf/grad/** and **pyscf/hessian/**
- Purpose: Analytical derivatives for geometry optimization
- Contains: Gradient computation for various methods
- Key files: Derivative kernels
- Test: `grad/test/`, `hessian/test/`

**examples/**
- Purpose: Demonstration scripts
- Organization: One directory per method (agf2/, cc/, dft/, gto/, etc.)
- Naming: `00-*.py`, `01-*.py` (numbered for learning progression)

**pyscf/lib/test/**
- Purpose: Unit tests for utility functions
- Contains: Test files mirroring lib structure
- Naming: `test_*.py`

## Key File Locations

**Entry Points:**
- `pyscf/__init__.py`: Package initialization, module imports
- `pyscf/__config__.py`: Default configuration values
- `setup.py`: Installation and build configuration (CMake for C extensions)
- `pyproject.toml`: Modern project metadata (PEP 517/518)

**Configuration:**
- `pytest.ini`: Test runner configuration (excludes slow/high-cost tests)
- `.ruff.toml`: Code formatter/linter configuration
- `.flake8`: Flake8 linter configuration
- `.style.yapf`: YAPF formatter configuration
- `.coveragerc`: Code coverage configuration
- `pyscf/__config__.py`: Runtime defaults

**Core Logic - Molecular Representation:**
- `pyscf/gto/mole.py`: Mole class definition (~4300 lines)
- `pyscf/gto/moleintor.py`: libcint interface
- `pyscf/gto/basis/`: Basis set data files

**Core Logic - SCF:**
- `pyscf/scf/hf.py`: HF kernel and RHF/UHF/GHF classes
- `pyscf/scf/_vhf.py`: Fock matrix builder
- `pyscf/scf/diis.py`: DIIS wrapper

**Core Logic - DFT:**
- `pyscf/dft/rks.py`: Restricted Kohn-Sham
- `pyscf/dft/numint.py`: Numerical integration (~2500 lines)
- `pyscf/dft/libxc.py`: Functional library interface

**Core Logic - Post-SCF:**
- `pyscf/cc/ccsd.py`: Coupled-cluster singles-doubles
- `pyscf/mp/mp2.py`: Møller-Plesset 2nd order
- `pyscf/fci/direct_spin0.py`: FCI singlet

**Utilities:**
- `pyscf/lib/misc.py`: StreamObject, general utilities
- `pyscf/lib/numpy_helper.py`: einsum, array operations
- `pyscf/lib/linalg_helper.py`: eigendecomposition, solvers
- `pyscf/lib/logger.py`: Logging with verbosity
- `pyscf/lib/chkfile.py`: HDF5 checkpoint I/O

**Testing:**
- `pyscf/scf/test/test_rhf.py`: RHF test (defines setUpModule with mol, mf)
- `pyscf/dft/test/test_rks.py`: RKS test
- `pytest.ini`: pytest configuration

## Naming Conventions

**Files:**
- Module files: `lowercase_with_underscores.py`
- Test files: `test_*.py` or `*_test.py` (convention: `test_*.py`)
- Basis data: No extension, named by element (e.g., `cc-pvdz`, `6-31g`)
- C source: `.c`, `.h` in subdirectories under `lib/`
- CMake: `CMakeLists.txt` at each level

**Directories:**
- Domain modules: `lowercase` (gto, scf, dft, cc, mp, fci)
- Data folders: `lowercase` with hyphens allowed (e.g., `f12-basis`, `dyall-basis`)
- Test directories: Always named `test/` parallel to module directory
- C code: Organized in `lib/{subdomain}/` mirroring Python structure

**Classes:**
- Main classes: CamelCase (Mole, RHF, UHF, RKS, UKS, CCSD)
- Mixins and base classes: CamelCase with descriptive suffix (KohnShamDFT, StreamObject)
- Internal classes: _CamelCase with leading underscore

**Functions:**
- Public functions: lowercase_with_underscores
- Private functions: _lowercase_with_underscores
- Factory functions: lowercase or lowercase_mixed (e.g., M(), kernel())

**Variables:**
- Global constants: UPPERCASE_WITH_UNDERSCORES
- Instance attributes: lowercase_with_underscores
- Numpy arrays: Descriptive lowercase (e.g., mo_coeff, mo_energy, density_matrix)

## Where to Add New Code

**New Quantum Chemistry Method:**
- Create directory: `pyscf/{method_name}/`
- Primary code: `pyscf/{method_name}/{method_name}.py` or domain-specific variants
- Examples:
  - New post-SCF method: `pyscf/{method_name}/base.py` with main class
  - If multiple variants: `pyscf/{method_name}/r{method}.py`, `u{method}.py` for restricted/unrestricted
- Tests: `pyscf/{method_name}/test/test_*.py`
- Examples: `examples/{method_name}/00-simple.py`, etc.

**New SCF Variant (HF-like):**
- File: `pyscf/scf/{variant}.py` (e.g., `dhf.py` for Dirac-Hartree-Fock)
- Inheritance: Inherit from `scf.hf.SCF` base class
- Override methods: `get_hcore()`, `get_veff()`, `get_fock()`, etc.
- Register in: `pyscf/scf/__init__.py` imports

**New DFT Functional:**
- No code change needed if using libxc
- If custom functional: Add to `pyscf/dft/libxc.py` or `xcfun.py`
- Register in functional dictionary: `pyscf/dft/__init__.py:XC`

**New Helper Function:**
- General utilities: `pyscf/lib/misc.py` (add function, update __all__)
- Linear algebra: `pyscf/lib/linalg_helper.py`
- NumPy operations: `pyscf/lib/numpy_helper.py`

**New Module-Level Utility:**
- Location: `pyscf/tools/{tool_name}.py`
- Example: Orbital analysis, format conversion

**C/C++ Extensions:**
- Location: `pyscf/lib/{domain_name}/` (e.g., `lib/vhf/`, `lib/gto/`)
- Build config: `pyscf/lib/CMakeLists.txt`
- Python wrapper: Pair each C module with Python `.py` file that imports it

**Tests for New Code:**
- Location: Co-located test directory, e.g., `pyscf/{module}/test/test_{module}.py`
- Pattern: Use unittest.TestCase with setUpModule() for global fixtures
- Example setup: `mol = gto.M(atom='...', basis='...')` then test against it
- Naming: `def test_*()` for methods, `class Test*()` for classes

## Special Directories

**pyscf/lib/deps/**
- Purpose: External C/C++ dependencies (libcint, libxc, etc.)
- Generated: Yes (populated during CMake build)
- Committed: No (gitignored, downloaded during build)

**pyscf/gto/basis/**
- Purpose: Basis set data files
- Generated: No
- Committed: Yes (critical for runtime)
- Contents: Pople basis (6-31g, etc.), Dunning (cc-pVDZ, etc.), ECP, F12

**build/**
- Purpose: Compilation artifacts and C extension binaries
- Generated: Yes (CMake output)
- Committed: No

**pyscf.egg-info/**
- Purpose: Package metadata (created by setuptools)
- Generated: Yes
- Committed: No

**Examples/** directories:
- `examples/{method}/` - One directory per computational method
- `*.py` files: Demonstration scripts
- Naming: `00-simple.py`, `01-advanced.py`, etc. (numbered for learning)
- Purpose: Educational; tested separately or in documentation

**pyscf/lib/vhf/, pyscf/lib/gto/, etc.**
- Purpose: C extension modules
- Files: `.c`, `.h` source files plus Python wrappers
- Build: Compiled by CMake into `.so` (shared objects)

---

*Structure analysis: 2026-04-09*
