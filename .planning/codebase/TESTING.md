# Testing Patterns

**Analysis Date:** 2026-04-09

## Test Framework

**Runner:**
- pytest
- Config: `pytest.ini`
- Import mode: `importlib` (via `--import-mode=importlib` in pytest.ini)

**Assertion Library:**
- unittest assertions via `unittest.TestCase` (PySCF uses unittest, not pytest assertions)

**Run Commands:**
```bash
pytest                                    # Run all tests
pytest -v                                 # Verbose output
pytest pyscf/lo/test/test_iao.py         # Run specific test file
pytest -k "not _high_cost"                # Skip high-cost tests (default behavior)
```

## Test File Organization

**Location:**
- Co-located with source code in `test/` subdirectory
- Pattern: `pyscf/[module]/test/test_*.py`
- Examples: `pyscf/lo/test/test_iao.py`, `pyscf/agf2/test/test_ragf2_h2o.py`, `pyscf/lo/test/test_cholesky.py`

**Naming:**
- Test files: `test_[feature].py`
- Test classes: `KnownValues` (singular class per file is common pattern)
- Test methods: `test_[description]()`

**Structure:**
```
pyscf/[module]/test/
├── test_feature1.py
├── test_feature2.py
└── __init__.py (optional)
```

## Test Structure

**Suite Organization:**
```python
import unittest
from pyscf import gto, scf

def setUpModule():
    """Module-level setup run once before all tests"""
    global mol, mf
    mol = gto.Mole()
    mol.atom = '...'
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.output = None
    mol.build()
    mf = scf.RHF(mol)
    mf.run()

def tearDownModule():
    """Module-level teardown run once after all tests"""
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Class-level setup run once before all methods"""
        cls.mol = gto.M(atom='...', basis='...')
        cls.mf = scf.RHF(cls.mol)
        cls.mf.run()
    
    @classmethod
    def tearDownClass(cls):
        """Class-level teardown run once after all methods"""
        del cls.mol, cls.mf
    
    def setUp(self):
        """Method-level setup run before each test method"""
        self.mol = mol.copy()
        self.mo_coeff = mf.mo_coeff.copy()
    
    def test_feature_behavior(self):
        """Test description: what is being verified"""
        result = compute_something(self.mo_coeff)
        self.assertAlmostEqual(result, expected_value, 6)
```

**Patterns:**
- Module-level `setUpModule()` and `tearDownModule()` used to set up expensive objects (molecule, SCF calculation) once
- Class-level `setUpClass()` and `tearDownClass()` used for test class-specific setup (common when using class attributes)
- Instance-level `setUp()` used for test method-specific setup (copies, state reset)
- Module-level globals (`global mol, mf`) referenced by `setUpModule()` and test methods
- Instance attributes (`self.mol`, `self.mf`) used when class-level setup is employed

## Mocking

**Framework:** unittest.mock (Python standard library)

**Patterns:**
- Not heavily used in core test suite; real objects preferred for integration testing
- When mocking needed: `from unittest import mock`
- Mock file operations: `tempfile.NamedTemporaryFile()` creates temporary files for checkpoint testing

**What to Mock:**
- External I/O operations (file reads/writes using tempfile)
- Optional dependencies not always installed

**What NOT to Mock:**
- PySCF objects (Mole, SCF methods) - real calculations are preferred
- Numpy operations - use actual arrays
- Linear algebra operations - integration testing preferred

## Fixtures and Factories

**Test Data:**
```python
def setUpModule():
    """Fixture: Create test molecule once, reuse across tests"""
    global mol
    mol = gto.Mole()
    mol.atom = '''
    O    0.   0.       0
    H    0.   -0.757   0.587
    H    0.   0.757    0.587'''
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.output = None
    mol.build()
```

**Factory Pattern:**
```python
class KnownValues(unittest.TestCase):
    def setUp(self):
        """Factory: Create fresh copy for each test"""
        self.mol = mol.copy()
        self.mo_coeff = mf.mo_coeff.copy()
```

**Location:**
- Fixtures defined in `setUpModule()` for shared objects
- Factories (methods creating copies/variations) in `setUp()` for test isolation
- Inline creation for test-specific data within test methods

## Coverage

**Requirements:** 
- No explicit coverage minimum enforced by configuration
- Coverage tracking configured in `.coveragerc`

**View Coverage:**
```bash
pytest --cov=pyscf --cov-report=html    # Generate HTML coverage report
pytest --cov=pyscf --cov-report=term    # Print coverage to terminal
```

**Coverage Configuration** (`.coveragerc`):
- Branch coverage enabled
- Excluded directories: `dmrgscf`, `fciqmcscf`, `shciscf`, `xianci`, etc. (experimental/external modules)
- Excluded patterns: slow tests, MPI tests, proxy tests
- Pragma directives recognized: `# pragma: no cover`

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods (e.g., `test_fast_iao_mulliken_pop`, `test_density`)
- Approach: Test with known input/output values or properties
- Example: Verify density preservation, orthonormality of localized orbitals
```python
def test_density(self):
    """Test whether the localized orbitals preserve the density."""
    mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
    rdm_loc = 2 * mo_loc.dot(mo_loc.T)
    matching = numpy.allclose(rdm_loc, self.rdm1_rhf, atol=1.0e-12)
    self.assertTrue(matching)
```

**Integration Tests:**
- Scope: Full workflows (SCF calculation followed by post-SCF method)
- Approach: Run complete calculation and verify final energy/results against known values
- Example: AGF2 calculation starting from SCF
```python
def test_ragf2_h2o_ground_state(self):
    """Tests the ground state AGF2 energies for H2O/cc-pvdz"""
    self.assertTrue(self.gf2.converged)
    self.assertAlmostEqual(self.mf.e_tot, -76.0167894720742, 10)
    self.assertAlmostEqual(self.gf2.e_1b, -75.89108074396137, 6)
```

**E2E Tests:**
- Framework: Not formally structured as separate E2E suite
- High-cost tests marked with `_high_cost` prefix and excluded by default (via pytest.ini)
- Examples: Full geometry optimization, MPI tests (marked `_skip` or in excluded files)

## Common Patterns

**Numeric Assertions:**
```python
self.assertAlmostEqual(computed_value, expected_value, 6)    # 6 decimal places
self.assertAlmostEqual(value, ref, delta=1.0e-6)             # Using delta parameter
numpy.allclose(result, expected, atol=1.0e-12)               # Array comparison
self.assertTrue(condition)                                    # Boolean assertion
```

**Async/Convergence Testing:**
- Not asynchronous framework (synchronous computation)
- Convergence tested via `.converged` attribute: `self.assertTrue(self.gf2.converged)`
- Energy comparison pattern for converged results

**Error Testing:**
```python
with self.assertRaises(NotImplementedError):
    function_that_should_raise()
```

**Test Skipping:**
- Skip patterns in pytest.ini: `--ignore-glob="*test_kproxy*.py"`, `--ignore-glob="*_slow*.py"`
- Test marking: Tests prefixed with `_high_cost` or `_skip` automatically excluded
- Conditional skip: `@unittest.skipIf(condition, "reason")`

**Data Comparison:**
```python
# Fingerprint method for quick identity checks
self.assertAlmostEqual(lib.finger(p), 0.56812564587009806, 5)

# Multi-dimensional array verification
v_ip = [numpy.linalg.norm(v)**2 for v in v_ip]
self.assertAlmostEqual(v_ip[0], 0.9704061235804103, 6)
```

**Checkpoint/Persistence Testing:**
```python
def test_ragf2_outcore(self):
    """Tests checkpoint file support"""
    gf2.dump_chk()
    with h5py.File(gf2.chkfile, 'r') as f:
        self.assertEqual(
            set(f['agf2'].keys()),
            {'e_1b', 'e_2b', 'e_init', 'converged', ...}
        )
    # Load from checkpoint
    gf2 = agf2.RAGF2(self.mf)
    gf2.__dict__.update(agf2.chkfile.load(gf2.chkfile, 'agf2'))
```

## Pytest Configuration

**File:** `pytest.ini`

**Key Settings:**
```ini
[pytest]
addopts = --import-mode=importlib
  -k "not _high_cost and not _skip"
  --ignore=examples
  --ignore-glob="*_slow*.py"
  --ignore-glob="*test_kproxy*.py"
  --ignore-glob="*test_proxy*.py"
  --ignore-glob="*test_bz*"
  --ignore-glob="*pbc/cc/test/*test_h_*.py"
  --ignore-glob="*test_ks_noimport*.py"
```

**Explanation:**
- Tests with `_high_cost` or `_skip` in name excluded by default
- Examples directory ignored
- Slow tests, proxy tests, and problematic test files explicitly excluded
- Examples: Long-running CI tests, KProxy/Proxy system tests, periodic boundary condition tests

---

*Testing analysis: 2026-04-09*
