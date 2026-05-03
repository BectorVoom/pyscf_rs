# Architecture

**Analysis Date:** 2026-04-09

## Pattern Overview

**Overall:** Modular layered architecture with plugin support for quantum chemistry computations

**Key Characteristics:**
- Domain-driven: Modules organized by computational chemistry domain (SCF, DFT, coupled-cluster, etc.)
- Plugin-extensible: Support for namespace plugins via PYSCF_EXT_PATH environment variable
- Physics-first API: Thin wrapper around low-level C extensions for computational kernels
- Composable workflows: Post-SCF methods stack on top of SCF results to build complex calculations

## Layers

**Application / User API Layer:**
- Purpose: High-level Python interface for molecular simulations
- Location: `pyscf/` (root level modules and convenience functions)
- Contains: Convenience constructors (e.g., `gto.M()`, `scf.RHF()`, `dft.RKS()`)
- Depends on: Core modules (gto, scf, dft, etc.) and lib layer
- Used by: User scripts and application code

**Core Domain Modules:**
- Purpose: Implement major computational domains
- Location: `pyscf/{gto,scf,dft,cc,fci,agf2,mp,td*,x2c,solvent,etc.}`
- Contains: Physics-specific classes and algorithms
- Depends on: lib, _vhf, integral evaluation, linear algebra helpers
- Used by: Application layer and post-SCF methods

**SCF Base Layer:**
- Purpose: Hartree-Fock and generalized self-consistent field methods
- Location: `pyscf/scf/`
- Key files:
  - `hf.py`: SCF kernel and HF methods (RHF, UHF, GHF)
  - `_vhf.py`: Fock matrix construction wrapper
  - `diis.py`: DIIS convergence acceleration (Bash-based)
  - `addons.py`: SCF enhancements (dispersion, smearing, level-shift)
- Depends on: gto, lib, integral engine
- Used by: DFT, MCSCF, post-SCF methods

**Density Functional Theory (DFT) Layer:**
- Purpose: Kohn-Sham density functional theory
- Location: `pyscf/dft/`
- Key files:
  - `rks.py`: Restricted Kohn-Sham
  - `uks.py`: Unrestricted Kohn-Sham
  - `numint.py`: Numerical integration on grids (large file, ~2500 lines)
  - `gen_grid.py`: Grid generation and pruning
  - `libxc.py`: Interface to libxc functional library
  - `LebedevGrid.py`: Lebedev grid data (large, ~4000 lines)
- Depends on: scf, gto, lib, libxc (C extension)
- Used by: Application layer, TDDFT, post-DFT methods

**Post-SCF Methods:**
- Purpose: Correlation effects beyond mean-field (coupled-cluster, CI, etc.)
- Location: `pyscf/{cc,fci,mp,agf2,adc,mrpt,cc,gw}`
- Examples:
  - `cc/`: Coupled-cluster methods (CCSD, RCCSD, UCCSD)
  - `mp/`: Møller-Plesset perturbation theory (MP2, MP3, etc.)
  - `fci/`: Full configuration interaction
  - `agf2/`: Algebraic Green's function theory
  - `adc/`: Algebraic diagrammatic construction
- Depends on: scf output (MO coefficients, energies), ao2mo, lib
- Used by: Application layer

**Basis Sets and Molecular Geometry:**
- Purpose: Molecular representation and basis functions
- Location: `pyscf/gto/`
- Key files:
  - `mole.py`: Mole class for molecular geometry, basis setup (~4300 lines)
  - `moleintor.py`: Interface to libcint integral library
  - `eval_gto.py`: Evaluate GTOs at arbitrary points
  - `ecp.py`: Effective core potentials
  - `basis/`: Basis set data (pople, dyall, f12, ccecp, etc.)
- Depends on: lib, libcint (C extension)
- Used by: All quantum chemistry modules

**Linear Algebra & Utilities (lib):**
- Purpose: Low-level support: BLAS/LAPACK wrappers, numpy helpers, logging
- Location: `pyscf/lib/`
- Key files:
  - `numpy_helper.py`: numpy array operations and einsum variants (~2000 lines)
  - `linalg_helper.py`: LAPACK wrappers, eigendecomposition (~2000 lines)
  - `misc.py`: General utilities, temporary file handling, logger support (~1800 lines)
  - `logger.py`: Logging system (with verbosity control)
  - `diis.py`: DIIS convergence helper
  - `chkfile.py`: HDF5 checkpoint file I/O
  - Subdirectories: `np_helper/`, `vhf/`, `ao2mo/`, `dft/`, `gto/` (C extension modules)
- Depends on: numpy, scipy, h5py, libcint, libxc
- Used by: All domain modules

**Molecular Orbital Transformation:**
- Purpose: AO to MO integral transformation
- Location: `pyscf/ao2mo/`
- Key files: `outcore.py`, `incore.py`, `kernel.py` (C extensions in `lib/ao2mo/`)
- Depends on: gto, lib
- Used by: Post-SCF methods (cc, fci, mp, etc.)

**Periodic Systems (PBC):**
- Purpose: Plane-wave expansion for crystal systems
- Location: `pyscf/pbc/`
- Contains: All domain modules mirrored (pbc/scf, pbc/dft, pbc/cc, etc.)
- Depends on: Core modules, gto (adapted for k-points)
- Used by: Solid-state quantum chemistry applications

**Helper Modules:**
- `grad/`: Analytical gradient computation for molecular geometry optimization
- `hessian/`: Second derivatives
- `td*/ `and `tdscf/`: Time-dependent methods (linear response, TDDFT)
- `tools/`: Data manipulation, wavefunction analysis, format conversion
- `solvent/`: Solvent effects (PCM, CPCM)
- `geomopt/`: Geometry optimization (interface to external optimizers)
- `data/`: Atomic data (elements, nuclear data via nist)

## Data Flow

**Typical Single-Point Calculation:**

1. Molecule Setup
   - User calls `gto.M()` → creates `Mole` object with geometry and basis
   - `Mole.build()` → parses atoms, bases, loads basis from `gto/basis/`
   - Integrals stored as `_atm`, `_bas`, `_env` arrays (libcint convention)

2. SCF Initialization
   - User creates SCF object: `scf.RHF(mol)` → instantiates `RHF` class
   - Setting SCF parameters (conv_tol, max_cycle, diis_space, etc.)

3. SCF Kernel Execution
   - `mf.run()` or `mf.kernel()` → calls `scf.hf.kernel()`
   - Loop: 
     - Get initial guess density matrix: `mf.get_init_guess()`
     - Build Fock matrix: `mf.get_fock()` → uses `_vhf.kernel()` for 2-electron integrals
     - Solve eigenvalue problem: `mf.eig()` (scipy.linalg.eigh)
     - Compute density matrix: `mf.make_rdm1(mo_coeff, mo_occ)`
     - Check convergence: `mf.get_grad()`
     - DIIS acceleration via `lib.diis.DIIS`
   - Save results to checkpoint file (HDF5)

4. Post-SCF Method (optional)
   - Take converged SCF object and MO coefficients
   - Example: `cc.CCSD(mf)` → accesses `mf.mo_coeff`, `mf.mo_energy`
   - Transform integrals: `ao2mo.kernel(mol, mo_coeff)`
   - Run correlation method-specific algorithm

**State Management:**
- Mole object holds: atom list, basis functions, integral environment
- SCF object holds: Mole reference, SCF parameters, converged MOs and densities
- Post-SCF objects hold: SCF reference, method-specific parameters and results
- Checkpoint files: Store converged MOs, densities, SCF energy for restart

## Key Abstractions

**Mole (Molecular System):**
- Purpose: Central representation of molecular geometry and quantum basis
- Examples: `pyscf/gto/mole.py:Mole` class (~3700 lines)
- Pattern: Lazy evaluation with caching (atomic basis loaded on demand)
- Entry point: `gto.M()` factory function

**SCF Base Class:**
- Purpose: Abstract interface for self-consistent field methods
- Examples: `pyscf/scf/hf.py:SCF`, `pyscf/scf/hf.py:RHF`, `pyscf/scf/hf.py:UHF`
- Pattern: Inheritance hierarchy with method overriding for variants
- Extensible: Users can subclass to modify `get_hcore()`, `get_veff()`, `get_fock()`, etc.

**KohnShamDFT:**
- Purpose: Mixin class adding DFT-specific methods to SCF classes
- Examples: `pyscf/dft/rks.py:KohnShamDFT`
- Pattern: Multiple inheritance - combines SCF with grid-based numerical integration

**StreamObject:**
- Purpose: Base class with logging and attribute management
- Examples: `pyscf/lib/misc.py:StreamObject`
- Pattern: Provides `.set()` chaining, verbose logging, automatic attributes

**DIIS (Direct Inversion in the Iterative Subspace):**
- Purpose: Convergence acceleration for SCF
- Examples: `pyscf/lib/diis.py:DIIS`, `pyscf/scf/diis.py`
- Pattern: Stores error vectors and solution vectors, linear extrapolation

## Entry Points

**Module Initialization:**
- Location: `pyscf/__init__.py`
- Triggers: Import of pyscf package
- Responsibilities: Import core modules (gto, scf, dft, etc.), set up logging, load plugins

**Main API Functions:**
- `gto.M()` - Create molecule
  - Location: `pyscf/gto/mole.py:M()`
  - Invokes: `Mole.build()` and other initialization
  
- `scf.RHF()` / `scf.UHF()` - Create SCF object
  - Location: `pyscf/scf/__init__.py`
  - Invokes: Class instantiation with mol parameter

- `dft.RKS()` / `dft.UKS()` - Create DFT object
  - Location: `pyscf/dft/__init__.py`
  - Invokes: RKS/UKS class with xc functional selection

- `.run()` or `.kernel()` - Execute calculation
  - Location: Method on SCF/DFT/post-SCF objects
  - Invokes: Domain-specific algorithm kernel

**Configuration:**
- Environment variables: PYSCF_EXT_PATH (plugin loading), CMAKE_* (build-time)
- Config file: `pyscf/__config__.py` (default parameters)
- Per-object: Attributes like `mol.verbose`, `mf.conv_tol`, `mf.max_memory`

## Error Handling

**Strategy:** Exceptions with context, fallback to warnings

**Patterns:**
- Custom exceptions in `pyscf/lib/exceptions.py`: `BasisNotFoundError`, `PointGroupSymmetryError`
- Logging with verbosity levels (0-9): `logger.warn()`, `logger.error()`, `logger.debug()`
- Assertion-based validation in `Mole.build()` for input sanity checks
- Try-except in optional feature imports (libxc, xcfun) with graceful degradation

## Cross-Cutting Concerns

**Logging:** 
- System: `pyscf/lib/logger.py` with per-object verbosity
- Pattern: `logger.info(obj, format_str, *args)` where obj holds `verbose` attribute

**Symmetry:**
- System: `pyscf/symm/` module for point group operations
- Pattern: Optional; Mole can be created with `symmetry=True`
- Used by: SCF and DFT to reduce computational cost

**Memory Management:**
- System: `mf.max_memory` parameter controls scratch disk vs. in-core
- Pattern: ao2mo and DFT numerical integration check available memory

**Integral Caching:**
- System: libcint C extension computes integrals on-demand
- Pattern: Direct SCF (default) recomputes integrals each cycle
- Alternative: Store integrals to disk with `mf.direct_scf = False`

**Checkpoint I/O:**
- System: HDF5 files via `pyscf/lib/chkfile.py`
- Pattern: `mf.chkfile = 'filename.chk'` enables auto-save; restart with `scf.chkfile.load_scf()`

---

*Architecture analysis: 2026-04-09*
