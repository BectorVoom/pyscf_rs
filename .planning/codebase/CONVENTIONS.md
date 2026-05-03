# Coding Conventions

**Analysis Date:** 2026-04-09

## Naming Patterns

**Files:**
- Module files use lowercase with underscores: `iao.py`, `boys.py`, `cholesky.py`
- Test files follow pattern `test_*.py`: `test_iao.py`, `test_cholesky.py`, `test_ragf2_h2o.py`
- Classes use PascalCase: `KnownValues`, `Mole`, `RHF`

**Functions:**
- Functions use lowercase with underscores: `iao()`, `reference_mol()`, `fast_iao_mullikan_pop()`, `kernel()`
- Private/internal functions use leading underscore: `_high_cost`, `_skip`
- Nested functions are acceptable: `make_iaos()` defined within `iao()`

**Variables:**
- Local variables use lowercase with underscores: `mol`, `orbocc`, `mf`, `mo_coeff`, `nocc`, `rdm1_rhf`
- Boolean variables use verb form: `conv`, `matching`
- Loop variables use simple names: `k`, `imacro`, `it`, `ind`
- Numpy arrays explicitly named: `s1`, `s2`, `s12`, `p12`, `ctild`, `iaos`

**Constants:**
- Module-level constants use UPPERCASE: `MINAO = getattr(__config__, 'lo_iao_minao', 'minao')`
- Configuration values are fetched via `__config__` attribute access

**Types/Classes:**
- Exception types use explicit names: `NotImplementedError`, `RuntimeError`, `KeyError`, `ValueError`
- Tuples for coordinates: `(int, int)` for alpha/beta electron counts

## Code Style

**Formatting:**
- Line length: 120 characters maximum (configured in `.flake8` and `.ruff.toml`)
- Indentation: 4 spaces (not tabs)
- YAPF configured with `COLUMN_LIMIT = 119` and `INDENT_WIDTH = 4`

**Linting:**
- Primary tool: Ruff (configured in `.ruff.toml`)
- Secondary tool: Flake8 (configured in `.flake8`)
- Significant rules ignored (see `.flake8`):
  - Indentation issues: E126, E127, E128, E129
  - Whitespace issues: E201, E202, E203, E211, E221, E222, E225, E226, E228, E231, E241, E251
  - Comment issues: E261, E262, E265, E266
  - Blank line issues: E301, E302, E303, E305, E306
  - Import ordering: E401, E402
  - Multiple statements per line: E701
  - Lambda assignment: E731
  - Wildcard imports: F403
  - Unused imports: F401
  - Complex functions: C901
- Quote style: Single quotes preferred (Ruff configured with `quote-style = "single"`)

## Import Organization

**Order:**
1. Standard library imports (functools, numpy/scipy): `from functools import reduce`, `import numpy`, `import scipy.linalg`
2. PySCF internal imports: `from pyscf import gto`, `from pyscf import scf`, `from pyscf.lib import logger`
3. Configuration imports: `from pyscf import __config__`
4. Submodule imports: `from pyscf.lo.orth import vec_lowdin`, `from pyscf.data.elements import is_ghost_atom`

**Path Aliases:**
- Configuration accessed via attribute: `getattr(__config__, 'lo_iao_minao', 'minao')`
- Internal module relative imports: `from pyscf.lo import orth, cholesky_mos`
- Conditional imports for optional features: `from pyscf.pbc import gto as pbcgto` (inside function when needed for PBC)

## Error Handling

**Patterns:**
- Raise specific exception types with descriptive messages:
  - `raise NotImplementedError('k-points crystal orbitals')`
  - `raise RuntimeError('rank of matrix lower than the number of orbitals')`
  - `raise KeyError('method = %s' % method)`
  - `raise ValueError('PM attribute method is not valid')`
- Catch specific exceptions: `except numpy.linalg.LinAlgError:` for linear algebra failures
- Try-except used for optional feature detection (e.g., Cholesky decomposition fallback to canonical orthogonalization)
- Warning messages via logger: `logger.warn(mol, 'message')`, `logger.info(mol, 'message')`

## Logging

**Framework:** PySCF's built-in `pyscf.lib.logger` module

**Patterns:**
- Import at top: `from pyscf.lib import logger`
- Logger levels: `logger.DEBUG`, `logger.NOTE`, `logger.WARN`
- Create logger instance: `log = logger.new_logger(localizer, verbose=verbose)` when verbose control needed
- Log with context: `logger.warn(mol, 'message')` or `logger.info(mol, 'message')`
- Info logs: `log.info('message formatting %s %d', var1, var2)`
- Timer tracking: `cput0 = (logger.process_clock(), logger.perf_counter())` and `log.timer('operation name', *cput0)`
- Check verbosity level: `if localizer.verbose >= logger.WARN:`
- Suppress output: `mol.output = None` or `mol.output = '/dev/null'`

## Comments

**When to Comment:**
- Algorithm explanations: `# s1 is the one electron overlap integrals (coulomb integrals)`
- Reference citations: `'''Intrinsic Atomic Orbitals. [Ref. JCTC, 9, 4834]'''`
- Complex transformations: `# overlap integrals of the two molecules`
- PBC-specific logic: `# For PBC, we must use the pbc code for evaluating the integrals lest the pbc conditions be ignored.`
- Workarounds and special cases: Comments explain why the special case exists

**Docstring Format:**
- Triple-quoted strings (single or multi-line): Module docstrings use `'''Module description'''`
- Function docstrings use triple quotes with structure:
  - Brief description followed by blank line
  - Longer explanation if needed
  - `Args:` section with type and description on separate lines
  - `Returns:` section describing return value(s)
  - Code example with `>>>` prefix in Returns section if illustrative

## Function Design

**Size:** 
- Functions typically 20-100+ lines for complex calculations
- Nested helper functions used for local computation (e.g., `make_iaos()` within `iao()`)

**Parameters:**
- Use keyword arguments with defaults for optional parameters: `minao=MINAO`, `kpts=None`, `lindep_threshold=1e-8`
- Verbose parameter pattern: `verbose=logger.DEBUG` for functions that support logging
- Callback pattern: `callback=None` for functions supporting user callbacks

**Return Values:**
- Return numpy arrays directly for numerical results
- Return tuples for multiple related values: `(e_ip, v_ip)` from `ipagf2()`
- Return status/boolean for convergence: `converged` attribute
- Always document return type and meaning in docstring

## Module Design

**Exports:**
- No explicit `__all__` in most modules; all public functions are importable
- Convention: functions at module level are public; functions starting with `_` are private
- Import pattern: `from pyscf.lo import iao` imports the `iao` function; `from pyscf.lo.iao import iao` also works

**Barrel Files:**
- Module `__init__.py` files selectively import and expose public API
- Example: `pyscf/scf/__init__.py` imports `RHF`, `UHF`, etc., making them accessible as `scf.RHF`
- Lazy imports used for optional dependencies

---

*Convention analysis: 2026-04-09*
