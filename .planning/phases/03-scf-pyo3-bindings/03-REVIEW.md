---
phase: 03-scf-pyo3-bindings
reviewed: 2026-05-11T00:00:00Z
depth: standard
files_reviewed: 53
files_reviewed_list:
  - crates/pyscf-algebra/src/eigh_gen.rs
  - crates/pyscf-algebra/src/error.rs
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/src/solve_linear.rs
  - crates/pyscf-chkfile/src/checkpointable.rs
  - crates/pyscf-chkfile/src/error.rs
  - crates/pyscf-chkfile/src/lib.rs
  - crates/pyscf-chkfile/src/primitives.rs
  - crates/pyscf-core/src/canonicalize.rs
  - crates/pyscf-core/src/lib.rs
  - crates/pyscf-df/src/auxbasis.rs
  - crates/pyscf-df/src/cholesky_eri.rs
  - crates/pyscf-df/src/df_jk.rs
  - crates/pyscf-df/src/error.rs
  - crates/pyscf-df/src/lib.rs
  - crates/pyscf-diis/src/cdiis.rs
  - crates/pyscf-diis/src/error.rs
  - crates/pyscf-diis/src/lib.rs
  - crates/pyscf-diis/src/storable.rs
  - crates/pyscf-gto/src/intor.rs
  - crates/pyscf-gto/src/lib.rs
  - crates/pyscf-oracle/src/fixtures.rs
  - crates/pyscf-oracle/src/lib.rs
  - crates/pyscf-oracle/src/runner.rs
  - crates/pyscf-py/src/bridge.rs
  - crates/pyscf-py/src/caches.rs
  - crates/pyscf-py/src/errors.rs
  - crates/pyscf-py/src/lib.rs
  - crates/pyscf-py/src/numpy_io.rs
  - crates/pyscf-py/src/scf.rs
  - crates/pyscf-scf/src/analyze.rs
  - crates/pyscf-scf/src/chkfile.rs
  - crates/pyscf-scf/src/convert.rs
  - crates/pyscf-scf/src/df_scf.rs
  - crates/pyscf-scf/src/diis_adapter.rs
  - crates/pyscf-scf/src/eig.rs
  - crates/pyscf-scf/src/energy.rs
  - crates/pyscf-scf/src/error.rs
  - crates/pyscf-scf/src/fock.rs
  - crates/pyscf-scf/src/ghf.rs
  - crates/pyscf-scf/src/hooks.rs
  - crates/pyscf-scf/src/init_guess.rs
  - crates/pyscf-scf/src/kernel.rs
  - crates/pyscf-scf/src/kernel_impl.rs
  - crates/pyscf-scf/src/lib.rs
  - crates/pyscf-scf/src/occ.rs
  - crates/pyscf-scf/src/rdm.rs
  - crates/pyscf-scf/src/rhf.rs
  - crates/pyscf-scf/src/scanner.rs
  - crates/pyscf-scf/src/uhf.rs
  - python/pyscf/__init__.py
  - python/pyscf/scf/__init__.py
  - python/pyscf/scf/ghf.py
  - python/pyscf/scf/hf.py
  - python/pyscf/scf/uhf.py
findings:
  critical: 0
  warning: 7
  info: 11
  total: 18
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-05-11
**Depth:** standard
**Files Reviewed:** 53
**Status:** issues_found

## Summary

Phase 3 delivers RHF/UHF/GHF SCF kernels, CDIIS, density-fitting J/K,
HDF5 chkfile primitives, the PyO3 bridge, and the cross-language oracle
harness. Overall the code is well-structured: every workspace crate
honors `#![forbid(unsafe_code)]` (no escape hatches found), error
propagation goes through typed `thiserror` enums rather than panics, the
algebra-wall (D-04) is respected (chemistry crates never name a cubecl
or pyo3 type), and reductions go through `oracle_*` helpers (Pitfall 9).
No `unimplemented!`, `assert False`, or `panic!("plan XX-YY pending")`
remnants were found in shipping code paths — the only `panic!` sites are
in `pyscf-oracle` (oracle_check macro + fixture-lookup), which are
test-only and intentional.

The findings below are concentrated on:

1. **Numerical correctness gaps** that mostly trace back to known cintx
   limitations (`int3c2e_sph` returns zeroed buffers; arity-4 `int2e`
   returns `NotYetImplemented`). These are documented in source code and
   gated by `#[ignore]` oracle tests, but the SCF kernel path will
   currently fail closed (`?`-propagation) rather than producing wrong
   numbers — acceptable for v1.
2. **PyO3 subclass-override fidelity (BIND-07 / Pitfall 7) edge cases**:
   the `get_init_guess` override path is hard-wired to `NoOverrides`
   even though every other hook routes through `call_method1`, and
   `to_mo_coeff` (used to unpack subclass-returned MO coefficients) only
   accepts F-contiguous input but the C-order `mo_coeff_to_pyarray`
   bridge sends C-order arrays into the Python override — a Python
   subclass that returns its input unchanged would fail the F-contiguous
   check on the return path.
3. **Resource / error-pathway hygiene**: the DIIS `extrapolate` indexes
   `bookkeep[0]` unconditionally to discover `flat_len`, which is fine
   only because `push` runs before that line — but the invariant is
   load-bearing and undocumented in the code. Several `cycle as usize`
   / `cycles as i64` casts assume non-negative inputs that are upheld at
   the call sites but not asserted.
4. **Style / dead code**: a handful of `#[allow(dead_code)]` markers
   (`_py_mol_use_marker`, `_no_overrides_use_marker`) papering over
   warnings that could be addressed more directly; `override_cache` is
   shipped (BIND-06) but appears to be exercised only by a smoke test —
   it is not yet wired into the kernel hot path.

No `Critical` findings.

## Warnings

### WR-01: `PyOverrideBridge::get_init_guess` bypasses Python override (BIND-07 violation)

**File:** `crates/pyscf-py/src/bridge.rs:93-109`
**Issue:** Every other hook in `PyOverrideBridge` dispatches through
`slf.call_method1(...)`, but `get_init_guess` short-circuits to
`pyscf_scf::NoOverrides.get_init_guess(mol, mode)`. The in-line comment
acknowledges this is intentional ("No Python-side override surface for
get_init_guess in the pyscf upstream class"). However, BIND-07 / Pitfall 7
("subclass fidelity") promises that *any* Python subclass override of
the 10/11 documented hooks is invoked transparently. Upstream pyscf does
expose `SCF.get_init_guess(mol, key='minao')` (`pyscf/scf/hf.py:485`
forward), and subclasses commonly override it. Forwarding to
`NoOverrides` here defeats the override mechanism for that one hook and
makes the Rust kernel see a different density than a subclass would
expect.

Worse: because `default_get_init_guess` returns
`InitGuessNotYetImplemented` for everything except `OneElectron`,
`Chkfile`, and `UserDM`, a Python subclass that tries to use a custom
init-guess will hit this stub rather than its own override.

**Fix:** Route through `call_method1` like every other hook:
```rust
fn get_init_guess(&self, _mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
    Python::attach(|py| {
        let mode_str = match mode {
            InitGuessMode::Minao => "minao",
            InitGuessMode::Atom => "atom",
            InitGuessMode::OneElectron => "1e",
            InitGuessMode::Huckel => "huckel",
            InitGuessMode::Chkfile(_) => "chkfile",
            InitGuessMode::UserDM(d) => {
                // UserDM is already a density; no Python round-trip needed.
                return Ok(d.clone());
            }
        };
        let args = PyTuple::new(py, [self.py_mol.bind(py).clone(),
                                     pyo3::types::PyString::new(py, mode_str).into_any()])
            .map_err(py_to_pyscf)?;
        call_hook(&self.slf, "get_init_guess", args, |r| {
            let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
            to_density(arr)
        })
    })
}
```

---

### WR-02: `to_mo_coeff` rejects C-contiguous arrays that `mo_coeff_to_pyarray` produces

**File:** `crates/pyscf-py/src/numpy_io.rs:53-82` (read path) and `crates/pyscf-py/src/numpy_io.rs:109-124` (write path)
**Issue:** Layout mismatch on the bridge round-trip:

- `mo_coeff_to_pyarray` (lines 109-124) **builds a C-order array** from
  the F-order `mc.data`: `c_data` is populated with C-order strides and
  returned via `Array2::from_shape_vec((nao, nmo), c_data)`. The
  resulting array is C-contiguous.
- `to_mo_coeff` (lines 53-82) requires **F-contiguous** input to take
  the fast path; otherwise it falls back to a transpose-copy via
  `view[[i, j]]`.

In `bridge::eig` (line 195), a Python subclass override receives an MO
coefficient array via `mo_coeff_to_pyarray` (C-order) and may pass it
back. If the subclass returns it unchanged (a common test/passthrough
pattern), `to_mo_coeff` sees C-order and silently falls back to the
slower transpose path — *but the slow path assumes the input is in
upstream pyscf layout (column-major LAPACK)*, so this produces the
**transpose** of the intended MO matrix.

Worse: the fast-path branch correctly reads F-order, but
`mo_coeff_to_pyarray` produces C-order; the fallback path therefore
copies `view[[i, j]]` in C-order semantics — actually that ends up
storing `data[i + j*nao] = C_python[i, j]`, which is correct *only if*
the Python side treats the array element-wise. Subclass code that does
`mo_coeff.copy()` returns C-order; mo_coeff_to_pyarray ALSO is C-order;
the fallback path stores C-order elements into an F-order Rust buffer,
producing an effective transpose of the MO matrix relative to the
caller's intent.

This is BIND-04 / Pitfall 5 territory and the BIND-04 stride-fuzz test
(plan 03-10) should catch it — but the asymmetric `density_to_pyarray`
(C-order) / `mo_coeff_to_pyarray` (C-order) / `to_mo_coeff` (F-order
required) layout policy is fragile.

**Fix:** Either (a) make `mo_coeff_to_pyarray` produce an F-contiguous
NumPy array (use `PyArray2::from_owned_array` with explicit F-order
strides), so the round-trip preserves layout; or (b) make `to_mo_coeff`
accept the array as `(nao, nmo)` and use `view[[i, j]]` (NumPy logical
indexing) to populate an F-order Rust buffer regardless of underlying
strides:
```rust
pub fn to_mo_coeff<'py>(arr: PyReadonlyArray2<'py, f64>) -> PyResult<MOCoefficients> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mo_coeff must be 2D, got shape {:?}", shape)));
    }
    let (nao, nmo) = (shape[0], shape[1]);
    let view = arr.as_array();
    // Always re-pack into F-order regardless of input stride.
    let mut data = Vec::with_capacity(nao * nmo);
    for j in 0..nmo {
        for i in 0..nao {
            data.push(view[[i, j]]);
        }
    }
    Ok(MOCoefficients { nao, nmo, data, energies: vec![], occupations: vec![] })
}
```

Recommend (b): the conversion cost is identical to the existing fallback
and the contract becomes layout-agnostic.

---

### WR-03: `Diis::extrapolate` indexes `bookkeep[0]` after push but the invariant is implicit

**File:** `crates/pyscf-diis/src/cdiis.rs:120-132`
**Issue:** Lines 120 (`self.bookkeep[0].len()`) and 132
(`self.bookkeep[0].clone()`) assume `bookkeep` is non-empty. That's
true at the call site because `push(...)` runs at line 88 immediately
before, but the invariant lives implicitly in the function ordering. If
a future refactor moves the `push` to after the solve, or someone calls
a hypothetical `extrapolate_existing` that doesn't push first, both
indices will panic with `index out of bounds`.

`flat_len` should be derived from the *just-pushed* iterate — which is
what `bookkeep[head_after_push]` would give — but the code reaches into
`[0]` which is the oldest entry.

**Fix:** Use the explicit `current.as_flat().len()` from the input,
since `current` is moved into the buffer just above:
```rust
// Capture flat_len from `current` BEFORE move-into-push, then push.
let flat_len_check = current.as_flat().len();
self.push(current, error);
// Sanity: every stored iterate must have identical flat length.
debug_assert_eq!(self.bookkeep[0].len(), flat_len_check);
```
Or, even simpler: hold a local `len` from `bookkeep.last()` (or the
just-pushed slot) after `push` completes:
```rust
let flat_len = self.bookkeep.last().map(|x| x.len()).unwrap_or(0);
if flat_len == 0 {
    return Err(DiisError::Singular); // or a new EmptyIterate variant
}
```

---

### WR-04: `kernel_impl::scf_loop` treats `cycle: u32 as i32` and `cycle as usize` as infallible

**File:** `crates/pyscf-scf/src/kernel_impl.rs:88, 102-103`
**Issue:** The cycle counter is `u32`; the kernel performs:
- `cycle as i32` at line 88 (passed to `hooks.get_fock(..., cycle as i32, None)`).
- `cycle as usize` at line 102 (passed to `diis_step`).

Both casts are saturating on overflow in Rust. `max_cycle` defaults to
50 and is bounded by `cfg.max_cycle: u32`, so this isn't reachable in
practice today, but a user who sets `max_cycle = u32::MAX` (or, more
realistically, a future use that loops further) would silently truncate
to `i32::MIN` for the i32 cast.

The `get_fock` hook signature takes `cycle: i32` (line 30 in
`hooks.rs`), so this is upstream-pyscf parity (Python's int → i32).
Upstream pyscf wraps cycles in a `range(max_cycle)` Python-side without
overflow concerns.

**Fix:** Either widen the hook signature to `cycle: u32` (breaking
trait change), or assert/clamp:
```rust
if cycle > i32::MAX as u32 {
    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
        "SCF cycle count exceeds i32::MAX".into())));
}
let cycle_i32 = cycle as i32;
```

This is low-risk in practice; flagging because the cast is invisible
and the i32 trait surface invites the truncation.

---

### WR-05: `mo_coeff` shape-mismatch on chkfile dump reports wrong `actual` shape

**File:** `crates/pyscf-scf/src/chkfile.rs:34-39, 45-49`
**Issue:** When `self.mo_coeff.data.len() != nao * nmo`, the error
returns `actual: vec![self.mo_coeff.data.len()]` — a one-element vector
containing the flat length. That violates the contract of
`ChkfileError::ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> }`,
where both vectors should be shape tuples (e.g. `[nao, nmo]`). Same for
the second copy at lines 45-49 if `ArrayView2::from_shape` fails.

This is cosmetic — the message survives — but downstream code that
parses `actual.len() == 2` to extract `(nao_disk, nmo_disk)` will
fail.

**Fix:** Either report a single integer-flatlen via a dedicated error
variant, or wrap the flat length as a "could not determine shape"
sentinel:
```rust
return Err(ChkfileError::ShapeMismatch {
    key: "mo_coeff".into(),
    expected: vec![nao, nmo],
    actual: vec![self.mo_coeff.data.len(), 0],  // flat length, no second dim
});
```

---

### WR-06: `df_jk::get_jk_df` per-Q intermediate `prod_buf` is materialised but discarded

**File:** `crates/pyscf-df/src/df_jk.rs:63-74`
**Issue:** The triple loop fills `prod_buf[lambda * nao + sigma] =
b * d`, then calls `oracle_sum(&prod_buf)` to produce `rho_q[q]`. This
allocates `nao * nao * naux` work over the lifetime of the function.
That's correct, but the `prod_buf` is re-zeroed implicitly by
overwriting at each `q` — and the loop body covers every (lambda,
sigma) pair, so no stale entries persist. **Correctness is fine.**
However, when `nao = naux = 0` (edge case, e.g. ghost-atom Mole),
`oracle_sum(&[])` is called. Per `pyscf_algebra::oracle::oracle_sum`
contract this should return 0.0, but the result is then stored in
`rho_q` which has length `naux == 0` — the loop body never runs, so
this is safe but worth verifying.

More substantively: the **K-matrix build** at lines 105-119 allocates
`naux * nao` `k_buf` per outer `(mu, nu)` and rewrites it; the inner
`for q ... for lambda` loop fills it fully before each `oracle_sum`,
so the buffer is correctly reset. This is O(nao⁴ · naux) — fine for
Phase 3 test corpus (nao ≤ ~140) but it's worth flagging as a
performance ceiling: a real DF-K build is O(nao³ · naux) via an
intermediate contraction. Out of v1 scope but documents a known cost.

**Fix:** No correctness change needed. Add a comment documenting that
the inner-loop ordering trades memory for arithmetic and that the O(n⁵)
ceiling is acceptable at Phase 3 corpus sizes. Plan 03-10's
optimization round should revisit.

---

### WR-07: `bridge::extract_mole_from_pyany` re-serializes on every hook call

**File:** `crates/pyscf-py/src/bridge.rs:283-295`, called from `scf.rs:290, 302, 315`
**Issue:** Each hook default in `PyRHF` (`get_hcore`, `get_ovlp`,
`get_jk`, `get_veff`) calls `extract_mole_from_pyany(py, &mol)` which
invokes `mol.call_method0("dumps")?` then `pyscf_gto::loads(&json)?`.
That's a JSON round-trip per hook per SCF cycle. For a 50-cycle SCF
on H2O/cc-pVDZ that's ~250 round-trips through a JSON serializer.

The PyRHF struct already holds a typed `Mole` inside `self.inner.mol`
(see `scf.rs:46`). The hook defaults should use the cached `Mole`
directly instead of re-extracting from the Python handle. The PyO3
override-detection path (`call_method1`) does pass the Python mol back
to user code so subclass overrides see the upstream `Mole` instance,
but the *default-hook path* doesn't need this.

This is a Phase 3 plan 03-10 perf concern (no correctness implication),
but it materially impacts SCF wall-clock for any real molecule. Flagged
as warning because it's load-bearing for the BIND-02 user experience.

**Fix:** Use `self.inner.mol` directly in each hook default. For
example, in `get_hcore`:
```rust
fn get_hcore<'py>(&self, py: Python<'py>, _mol: Py<PyAny>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let h = py.detach(|| default_get_hcore(&self.inner.mol)).map_err(pyscf_to_py)?;
    density_to_pyarray(py, &h)
}
```
The `_mol` argument is kept on the Python signature for upstream
compat (`mf.get_hcore(mol)` is the canonical call); we just ignore it
when the user passes the same mol that was registered at `__init__`.

---

## Info

### IN-01: `chkfile::primitives::open_for_write` race: TOCTOU on existence check

**File:** `crates/pyscf-chkfile/src/primitives.rs:22-36`
**Issue:** `p.exists()` then `hdf5::File::append` is a classic
TOCTOU (time-of-check vs time-of-use) pattern. Between the `exists`
check and the `append` / `create` call, another process could create or
delete the file. For a chkfile this is unlikely to cause data
corruption (HDF5 handles concurrent access through its own locking),
but the error message will be misleading.

**Fix:** Try `append` first; on `Err(_)`, try `create`. Or use
`hdf5_metno::File::with_options()` if available.

### IN-02: `cholesky_eri::forward_substitute` silently returns 0 on zero diagonal

**File:** `crates/pyscf-df/src/cholesky_eri.rs:199-208`
**Issue:** The doc comment says "Silently writes zeros at indices where
`L[i, i] == 0` (handled by caller's SingularAux check in
`cholesky_banachiewicz_lower` — if we got here, the diagonal is
nonzero)". This is a defense-in-depth fallback, but if a caller invokes
`forward_substitute` directly with a singular L, the result silently
produces garbage. Worth a `debug_assert!(diag != 0.0)` so test/dev
builds catch the misuse.

**Fix:**
```rust
let diag = l[i * n + i];
debug_assert!(diag != 0.0 && diag.is_finite(),
    "forward_substitute: zero/non-finite diagonal at row {i}");
out[out_offset + i] = if diag != 0.0 { s / diag } else { 0.0 };
```

### IN-03: `_py_mol_use_marker` and `_no_overrides_use_marker` are dead-code workarounds

**File:** `crates/pyscf-py/src/scf.rs:619-637`
**Issue:** Both functions are tagged `#[allow(dead_code)]` and exist
solely to silence unused-field/import warnings. The `py_mol` getter
DOES use `py_mol` (lines 70, 530, 583), so the compiler should see it
as used; the marker is probably stale. Likewise `NoOverrides` is
intentionally unused in `scf.rs` since dispatch always goes through
`PyOverrideBridge`.

**Fix:** Remove `_py_mol_use_marker` (the getter usage covers it). For
`NoOverrides`, replace the import-use marker with
`#[allow(unused_imports)]` directly on the import line:
```rust
#[allow(unused_imports)]
use pyscf_scf::NoOverrides;
```

### IN-04: `caches::override_cache` is shipped but appears unused in kernel path

**File:** `crates/pyscf-py/src/caches.rs` (entire module)
**Issue:** The doc comment says "Phase 3 use case: cache the type-id
of Python subclasses that override SCF hooks. The override-detection
fast path looks up the type-id in this cache instead of calling
`hasattr()` once per cycle per hook." A grep across `crates/pyscf-py/`
finds the cache exercised only by `tests/scaffold_surface.rs:41`
(`use pyscf_py::caches::override_cache as _;`). The actual kernel path
(`PyOverrideBridge::call_method1`) always invokes through MRO without
checking the cache.

This isn't a bug — `call_method1` is correct and the cache would only
be an optimization. But the BIND-06 documentation suggests it's wired,
which it isn't.

**Fix:** Either wire it (check `slf.get_type().as_ptr() as usize` in
override_cache before the first `call_method1` in `kernel`, skip the
bridge entirely if not overridden), or update the doc comment to note
"shipped for future optimization; not yet wired into hot path".

### IN-05: `eigh_gen` pads dropped eigenvalues with `+∞` but caller may not expect it

**File:** `crates/pyscf-algebra/src/eigh_gen.rs:114-117`
**Issue:** When linear dependencies are removed (`valid_cols.len() <
n`), the first `n_lin` eigenvalues are real, the remaining `n - n_lin`
are `f64::INFINITY`. The C buffer has the corresponding columns zeroed
(line 122-127). Downstream:
- `default_get_occ` (occ.rs:19) does `if i < n_occ ... occ[i] = 2.0`
  — fine, the dropped MOs get occupation 0.
- `default_make_rdm1` (rdm.rs:38-46) multiplies `occ[i] * C[mu, i] *
  C[nu, i]`. With `occ[dropped] = 0.0` and `C[:, dropped] = 0.0`, the
  contribution is 0 — fine.
- `default_energy_elec` (energy.rs:52-53) just dot-products D and h1e
  — no per-MO loop, fine.

But: `mulliken_pop` and other post-SCF analyses that take the MO
spectrum may not expect `+∞` entries in `mo_energy`. If a user prints
or plots the eigenvalues, the `+∞` flags will be visible.

**Fix:** Document the convention prominently in `eigh_gen`'s doc, and
also document on `MOCoefficients.energies` and `ScfResult.mo_energy`.
Optionally, pad with `NaN` instead — many plotting/analysis tools
skip NaN automatically but show `+inf` as a tall spike. NaN is more
"missing data"-shaped; `+inf` more "above all real values"-shaped.

### IN-06: `fixtures::atom` panics on unknown fixture (test-only is fine)

**File:** `crates/pyscf-oracle/src/fixtures.rs:37, 48`
**Issue:** Both `atom()` and `basis()` panic with `panic!("pyscf-oracle:
unknown fixture '{}'", other)`. This is acceptable for a test-only crate
where invalid fixtures are a programmer error, but consider returning
`Option<&'static str>` so the caller can surface a structured error
through `OracleError::UnknownFixture(String)` for symmetry with
`OracleError::UnknownMethod`.

**Fix:** Add `OracleError::UnknownFixture(String)` variant and convert
the panics into early returns from the dispatcher.

### IN-07: `intor::evaluate_int3c2e_with_auxmol` returns zero-filled buffer (documented gap)

**File:** `crates/pyscf-gto/src/intor.rs:459-476`
**Issue:** The function explicitly returns
`vec![0.0_f64; nao*nao*naux]` because `int3c2e_sph` is not a base
symbol in cintx-ops. This is documented in the module-level comment and
in `cholesky_eri` (lines 105-108). The contract is: "shape-correct
zero-filled buffer ... sufficient for shape/wiring tests, NOT for
bit-exact DF-HF energy". The DF-HF oracle (plan 03-10 wave 2) is
`#[ignore]`'d until cintx lands the real symbol.

This is an Info-level concern rather than a Warning because:
- The behaviour is documented in code.
- The path is gated by `#[ignore]` tests.
- A user calling `RHF::density_fit().kernel()` today will get an
  energy that converges to the **single-particle (1e-only) energy plus
  nuclear repulsion** (because J and K both come out zero), not a wrong
  DF-HF energy. The result is incorrect but recognizably so (off by
  orders of magnitude from upstream).

**Fix:** Consider raising
`PyscfRsError::NotYetImplemented{phase:3, what:"int3c2e_sph awaiting cintx-ops base symbol"}`
instead of silently zero-filling, so users hitting this path get a
loud failure rather than a wrong answer:
```rust
fn evaluate_int3c2e_with_auxmol(_mol: &Mole, _auxmol: &Mole) -> Result<IntorOutput, PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 3,
        what: "int3c2e_sph: cintx-ops base symbol not yet available — \
               DF-HF cannot run until cintx ships the operator id",
    })
}
```
This is more aligned with the rest of pyscf-rs's "fail closed rather
than produce wrong numbers" philosophy.

### IN-08: `bridge::PyOverrideBridge::get_occ` doesn't pass `nelec` to Python

**File:** `crates/pyscf-py/src/bridge.rs:202-215`
**Issue:** The Rust `OverrideHooks::get_occ` signature is `(mo_energy:
&[f64], nelec: usize) -> Vec<f64>`. The bridge dispatches
`slf.get_occ(mo_energy)` without passing `nelec`. Upstream pyscf's
signature is `def get_occ(self, mo_energy=None, mo_coeff=None)`, where
`nelec` is read from `self.mol.nelectron`. The Python `PyRHF.get_occ`
default (`scf.rs:372`) does exactly that:
`let nelec = self.inner.mol.nelectron as usize;`. So at the **default**
hook level, parity is preserved.

But for a **subclass override** that wants to use a custom nelec
(e.g., to fill MOs differently for FT-HF or fractional occupancy
schemes), the bridge silently drops the Rust `nelec` arg. The override
gets only mo_energy.

This is minor because (a) upstream pyscf has the same surface, (b)
custom occupancy schemes are rare. Flagging for completeness.

**Fix:** Either accept the upstream-parity behaviour (and document
that overrides must read `self.mol.nelectron`), or pass `nelec` as a
keyword arg to the Python call to support both pyscf-parity and
explicit-arg overrides:
```rust
let kwargs = pyo3::types::PyDict::new(py);
kwargs.set_item("nelec", nelec)?;
slf.bind(py).call_method("get_occ", (e_py,), Some(&kwargs))
```

### IN-09: `df_scf::DfHooks::get_veff` re-computes get_jk_df instead of using `get_jk`

**File:** `crates/pyscf-scf/src/df_scf.rs:100-120`
**Issue:** `DfHooks::get_jk` and `DfHooks::get_veff` both call
`pyscf_df::get_jk_df(dm, self.df)?` — so `get_veff` runs the J/K build
twice if the kernel calls both. Inspecting the kernel cycle loop
(`kernel_impl.rs:88, 128`) shows it only invokes `hooks.get_veff(...)`
within a cycle (not `get_jk`), so there's no double-evaluation in the
hot path. But a user (or future kernel) that calls both will recompute.

The simple fix: make `get_veff` call `self.get_jk(...)` to centralize
the DF compute:
```rust
fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
    let (j, k) = self.get_jk(mol, dm)?;
    // ... (J - 0.5K loop)
}
```

### IN-10: `ChkfileError::ShapeMismatch` `actual` ignored on `ArrayView2::from_shape` failure

**File:** `crates/pyscf-scf/src/chkfile.rs:45-49`
**Issue:** When `ArrayView2::from_shape` fails (e.g., the shape and
length disagree), we map it to
`ChkfileError::ShapeMismatch { actual: vec![self.mo_coeff.data.len()] }`.
The actual cause is captured by ndarray's error type but discarded in
favour of a synthetic message. For debugging chkfile interop issues
this loses context. Add ndarray::ShapeError to the error wrapper or
preserve via `MalformedMol(format!("ndarray shape: {}", e).into())` (but
that's the JSON error variant — wrong fit).

**Fix:** Define a new `ChkfileError::Internal(String)` variant for
unexpected errors, or upgrade `MalformedMol` to a more general
`InternalParse(String)`. Cheaper alternative: log a `tracing::error!`
before returning.

### IN-11: `numpy_io::to_density` C-contig fallback loop is exact-correct but redundant

**File:** `crates/pyscf-py/src/numpy_io.rs:31-44`
**Issue:** The fallback path manually re-walks `view[[i, j]]` to
populate `buf` in C-order. ndarray already provides
`view.as_standard_layout().to_owned().into_raw_vec_and_offset().0` which
handles strided / transposed inputs in one call. The manual loop is
correct but verbose.

**Fix:** Replace with:
```rust
let owned = view.as_standard_layout().to_owned();
let (data, _) = owned.into_raw_vec_and_offset();
```
This is what BIND-04 documentation calls "re-materialise as
default-order (C-contiguous) owned ndarray" — the helper already exists.

---

_Reviewed: 2026-05-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
