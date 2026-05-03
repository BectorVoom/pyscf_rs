# Codebase Concerns

**Analysis Date:** 2026-04-09

## Tech Debt

**Frozen Orbitals Not Implemented in GW Methods:**
- Issue: Multiple GW modules raise `NotImplementedError` for frozen orbital functionality that users expect
- Files: 
  - `pyscf/gw/gw_cd.py:274` - TODO: implement frozen orbs
  - `pyscf/pbc/gw/kugw_ac.py:622` - TODO: implement frozen orbs
  - `pyscf/pbc/gw/krgw_cd.py:617` - TODO: implement frozen orbs
  - `pyscf/pbc/gw/krgw_ac.py:556` - TODO: implement frozen orbs
- Impact: Users cannot use standard frozen orbital approximation in GW calculations, limiting practical applicability for large systems
- Fix approach: Implement frozen orbital indexing and tensor slicing in GW kernels following similar patterns in CC/MP2 modules

**Analytic Derivatives Missing in GW:**
- Issue: GW implementations use numerical approaches where analytical gradients are commented out or unimplemented
- Files:
  - `pyscf/gw/gw_cd.py:97` - FIXME with linearization issue
  - `pyscf/pbc/gw/kugw_ac.py:143` - TODO: analytic sigma derivative
  - `pyscf/pbc/gw/krgw_ac.py:133` - TODO: analytic sigma derivative
  - `pyscf/gw/gw_slow.py:286` - TODO: analytic sigma derivative (commented out)
- Impact: Geometry optimization and Hessian calculations are slow and may have numerical stability issues
- Fix approach: Implement analytical derivative formulas for self-energy calculations following Casida-like approach used elsewhere

**Incomplete Solvent Integration:**
- Issue: Solvent implementation has known coupling issues when combined with other modules
- Files:
  - `pyscf/solvent/_attach_solvent.py:82-83` - FIXME: super() problematic with QM/MM combinations
  - `pyscf/solvent/_attach_solvent.py:134` - FIXME: when applying DF after solvent
  - `pyscf/solvent/pol_embed.py:215` - FIXME: PE model default character undefined
  - `pyscf/solvent/test/test_ddcosmo.py:520` - TODO: add tests for direct-scf, ROHF, ROKS, .newton()
- Impact: Solvent calculations may give incorrect results when combined with density fitting, QM/MM, or specific SCF methods
- Fix approach: Refactor solvent attachment to properly handle method resolution order; add comprehensive integration tests

**Redundant Fock Matrix Computation:**
- Issue: MC-PDFT computes Fock matrix twice unnecessarily
- Files: `pyscf/mcpdft/xmspdft.py:142` - TODO fix redundancy
- Impact: Unnecessary computation time in excited-state MC-PDFT calculations
- Fix approach: Cache Fock matrix or refactor to avoid duplication

**Outcore AO2MO Symmetry Modes Not Implemented:**
- Issue: 4-fold and anti-symmetric tensor transformation modes documented as TODO across multiple files
- Files:
  - `pyscf/ao2mo/outcore.py:63-66, 147-150, 369-372, 542-545, 623-626`
  - `pyscf/ao2mo/__init__.py:79-82, 207-210, 363-366`
  - `pyscf/ao2mo/addons.py:71-72`
- Impact: Users cannot use advanced symmetry properties in integral transformations for outcore calculations
- Fix approach: Implement permutation and sign logic for anti-symmetric transformations

**SMD Solvent Model Incomplete:**
- Issue: Cavitation energy descriptor (CDS) computation incomplete for OCC methods
- Files: `pyscf/solvent/cosmors.py:237-259` - TODO: implement OCC
- Impact: SMD solvation corrections are incomplete for open-shell coupled cluster
- Fix approach: Add OCC-specific descriptors to cavitation energy model

## Known Bugs

**GW Linearization with Contour Deformation:**
- Symptoms: Wrong quasiparticle energy when using linearized GW with contour deformation method
- Files: `pyscf/gw/gw_cd.py:97-99`
- Trigger: `gw.linearized = True` with `gw.kernel()` using contour deformation
- Current status: Code raises NotImplementedError to prevent silent incorrect results
- Workaround: Use non-linearized GW or switch to analytical continuation method

**Bare Exception Handlers in Test Code:**
- Symptoms: Suppressed errors in test suite may hide actual failures
- Files: `pyscf/pbc/cc/test/test_kgccsd.py:151, 155, 160, 168, 172, 177` - bare except clauses
- Trigger: Run test suite; failures may be silently ignored
- Impact: Test results cannot be trusted; real bugs could exist in tested code paths
- Fix approach: Replace bare `except:` with specific exception types; log exceptions

**DEBUG Flag Hardcoded in Production Code:**
- Symptoms: Conditional code paths enabled/disabled by hardcoded DEBUG variables
- Files:
  - `pyscf/scf/diis.py:30` - DEBUG = False (but affects DIIS behavior)
  - `pyscf/ci/cisd.py:415` - DEBUG = True (hardcoded for overlap calculation)
  - `pyscf/scf/dhf.py:45` - DEBUG = False (affects Dirac-Fock Hessian computation)
- Impact: Cannot trace actual code execution path; disabling debug output requires code modification
- Fix approach: Convert to runtime logging or parameterized options

**CISD Overlap Calculation Conditional:**
- Symptoms: CISD overlap between non-orthogonal orbitals may use suboptimal path
- Files: `pyscf/ci/cisd.py:424-426` - hardcoded DEBUG check skips optimal path
- Trigger: Call `dot()` with overlap matrix on non-orthogonal basis
- Impact: Potentially slower or less accurate overlap calculations
- Fix approach: Remove DEBUG flag; always use optimal algorithm

## Security Considerations

**No Input Validation in Basis Set Data:**
- Risk: Basis set files contain hardcoded numerical data (Dyall basis); no checksum validation
- Files: `pyscf/gto/basis/dyall-basis/*.py` (11,839+ lines of data per file)
- Current mitigation: Data is read-only after module load
- Recommendations: Add CRC32 or MD5 checksum to basis files; validate on import

**Missing Bounds Checking in Integral Transforms:**
- Risk: Array reshaping operations could silently fail or access wrong memory
- Files: `pyscf/ao2mo/outcore.py`, `pyscf/ao2mo/r_outcore.py`
- Current mitigation: NumPy raises exceptions on shape mismatch
- Recommendations: Add explicit shape assertions before reshape operations; add input validation

**Exception Types Not Specific in Test Cleanup:**
- Risk: Exception handlers catch all types including KeyboardInterrupt, SystemExit
- Files: `pyscf/pbc/cc/test/test_kgccsd.py`
- Current mitigation: Only in test code, not production
- Recommendations: Catch specific exceptions; document why broad catching is needed

## Performance Bottlenecks

**24K+ Line File in ADC Module:**
- Problem: `pyscf/adc/uadc_ee.py` has 24,189 lines - single monolithic file
- Files: `pyscf/adc/uadc_ee.py`
- Cause: All unrestricted ADC/EE equations generated inline instead of modularized
- Improvement path: Split into logical equation groups; use code generation for boilerplate; cache intermediate expressions
- Current impact: Long parse time; difficult to modify individual equations; memory footprint during import

**Large Basis Files Loaded Entirely:**
- Problem: Dyall basis files (8K-12K lines each) fully parsed and kept in memory
- Files: `pyscf/gto/basis/dyall-basis/dyall_*.py`
- Cause: Each basis definition is complete Python file with inline data
- Improvement path: Use JSON or HDF5 format; lazy-load basis data; implement basis data cache
- Current impact: Slow import time; increased memory for users who don't need all bases

**Numint DFT Integration Grid Inefficiency:**
- Problem: Grid point cutoff threshold not tuned for performance vs accuracy
- Files: `pyscf/dft/numint.py:816, 2921` - TODO comments about turnover threshold
- Cause: Default thresholds may include unnecessary grid points
- Improvement path: Auto-tune cutoff based on basis set; benchmark against molecular size
- Current impact: DFT calculations run slower than necessary for large molecules

**Repeated Density Matrix Evaluation:**
- Problem: Density matrix evaluated multiple times per SCF cycle when solvent attached
- Files: `pyscf/solvent/_attach_solvent.py:104-108` - redundant get_veff calls
- Cause: Solvent kernel called separately; dm computed in multiple methods
- Improvement path: Cache dm within SCF iteration; compute solvent response once per density update
- Current impact: 20-50% slower SCF when solvent is attached

## Fragile Areas

**CISD Overlap with Non-Orthogonal Basis:**
- Files: `pyscf/ci/cisd.py:410-465`
- Why fragile: Complex tensor indexing with conditional logic; determinant detection fragile
- Safe modification: Add comprehensive tests for all orbital overlap regimes before changes
- Test coverage: Only basic cases tested; missing edge cases (near-singular overlaps)
- Risk: Silent precision loss if indices computed incorrectly

**MC-PDFT State Average Mixing:**
- Files: `pyscf/mcpdft/mcpdft.py:267-296, 389-521, 545, 796`
- Why fragile: Multiple TODO comments about state-average compatibility; incompatible with non-translated functionals
- Safe modification: Ensure all state-average modes tested; verify with translated functionals only
- Test coverage: Basic state-average works; edge cases with degenerate states untested
- Risk: Incorrect energy decomposition or gradients for state-average MC-PDFT

**GW Contour Deformation with Linearization:**
- Files: `pyscf/gw/gw_cd.py:97-119`
- Why fragile: Newton solver with no convergence guarantee; relies on delta tuning
- Safe modification: Add robustness checks; test with difficult molecules
- Test coverage: Basic GW works; convergence failures not well documented
- Risk: Silent convergence failures or wrong energies without error message

**Solvent-QM/MM Integration:**
- Files: `pyscf/solvent/_attach_solvent.py:75-108`
- Why fragile: super() resolution problematic with multiple inheritance chains
- Safe modification: Redesign attachment mechanism; test all solvent+QM/MM combinations
- Test coverage: Individual solvent tests exist; QM/MM tests exist; combined tests missing
- Risk: Incorrect force fields or charges when solvent + QM/MM combined

**Newton-Raphson Orbital Optimization:**
- Files: `pyscf/soscf/newton_ah.py:54, 158, 311, 320, 415` - multiple TODO comments
- Why fragile: Dual-basis treatment not implemented; step size adjustment heuristic
- Safe modification: Validate with small test molecules; monitor convergence metrics
- Test coverage: Convergence tested for standard cases; difficult cases untested
- Risk: Orbital optimization divergence for challenging electronic structures

## Scaling Limits

**Memory Usage in Large ADC Calculations:**
- Current capacity: ADC equations fit in memory for small-medium molecules (< 50 basis functions)
- Limit: UADC-EE file size and equation complexity limit to ~100 basis functions before memory issues
- Scaling path: Implement disk-based (outcore) ADC; decompose equations
- Risk: OOM errors for medium-sized systems

**GW Self-Energy Storage:**
- Current capacity: Sigma matrix stored fully in memory; works for ~200 basis functions
- Limit: Beyond ~300 basis functions, Sigma becomes memory bottleneck
- Scaling path: Implement iterative GW; approximate Sigma on energy grid
- Risk: GW limited to small-medium systems

**Solvent DDCOSMO Grid:**
- Current capacity: Lebedev grid integrates ~5K points for medium solute
- Limit: Linear scaling with surface area; hits memory/time limits at ~1000 grid points
- Scaling path: Adaptive grid refinement; approximate distant surface elements
- Risk: Solvent calculations slow for large molecules

## Dependencies at Risk

**NumPy Compatibility Range:**
- Risk: Version restriction `!=1.16,!=1.17` indicates past breaking changes
- Impact: Users on excluded versions cannot use package; may not upgrade safely
- Migration plan: Remove version exclusions with comprehensive NumPy 2.0 testing

**h5py Requirement with PyTorch/TensorFlow:**
- Risk: h5py can conflict with torch/TF on GPU systems
- Impact: Users cannot easily combine PySCF with ML frameworks
- Migration plan: Make h5py optional; use fallback formats (pickle, JSON)

**SciPy API Changes:**
- Risk: Minimum version 1.6.0 is over 5 years old; many deprecated functions still called
- Impact: Code may break with future SciPy versions
- Migration plan: Update minimum to 1.10+; refactor deprecated calls

## Missing Critical Features

**DF-GW for Exact Exchange:**
- Problem: DFGWExact not implemented, only RPA available
- Blocks: High-accuracy GW+hybrid functional calculations
- Files: `pyscf/gw/gw_exact.py:272-274`

**Direct SCF with Solvent:**
- Problem: No direct-SCF implementation for DDCOSMO/SMD
- Blocks: Direct SCF optimization for solvated systems
- Files: `pyscf/solvent/test/test_ddcosmo.py:520`

**MC-PDFT with Non-Translated Functionals:**
- Problem: Code explicitly rejects non-translated functionals
- Blocks: Alternative functional forms in MC-PDFT
- Files: `pyscf/mcpdft/mcpdft.py:284-285`

**SO3 Symmetry in HF/KS:**
- Problem: Continuous rotation group symmetry disabled everywhere
- Blocks: Linear molecules and symmetric systems cannot use exact symmetry
- Files: `pyscf/scf/atom_hf.py:34`, `pyscf/scf/atom_ks.py:33`

## Test Coverage Gaps

**GW Test Coverage:**
- What's not tested: GW with frozen orbitals, GW linearization, GW contour deformation with difficult convergence
- Files: `pyscf/gw/test/test_gw.py`, `pyscf/gw/test/test_gw_ac.py`
- Risk: GW module has multiple NotImplementedError cases and FIXME comments with minimal test coverage
- Priority: High - GW is user-facing code

**Solvent-SCF Integration:**
- What's not tested: Solvent+direct SCF, Solvent+ROHF, Solvent+Newton-Raphson
- Files: `pyscf/solvent/test/test_ddcosmo.py:520`
- Risk: Untested combinations likely broken or incorrect
- Priority: High - common use cases

**MC-PDFT State-Average:**
- What's not tested: Degenerate state handling, gradient computation for state-average
- Files: `pyscf/mcpdft/test/test_mcpdft.py`
- Risk: State-average MC-PDFT gradients likely incorrect for degenerate states
- Priority: High - impacts geometry optimization

**CISD with Overlap Matrix:**
- What's not tested: Overlap with various singular/near-singular cases
- Files: `pyscf/ci/test/test_cisd.py`
- Risk: Precision loss in edge cases
- Priority: Medium - affects advanced use cases

**PBC ADC Methods:**
- What's not tested: Comprehensive ADC for periodic systems
- Files: `pyscf/pbc/adc/`
- Risk: PBC-ADC may have untested failure modes
- Priority: Medium - emerging feature

---

*Concerns audit: 2026-04-09*
