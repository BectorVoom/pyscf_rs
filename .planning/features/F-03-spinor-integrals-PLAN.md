# F-03 — Spinor (relativistic 2-component) integral representation

**Status:** SCAFFOLDED (foundation + plan landed; implementation tasks T1–T5 open)
**Source finding:** `.planning/AUDIT-FIX-2026-06-01.md` F-03
**Stub origin:** `crates/pyscf-gto/src/intor.rs` `_spinor` arm (`NotYetImplemented{phase:3}`)
**Owner:** unassigned · **Created:** 2026-06-01

---

## 1. Goal

Make `mol.intor("…_spinor")` return correct **complex** 2-component spinor
integrals (overlap, kinetic, nuclear, ERIs as scoped) matching upstream PySCF
to `atol = 1e-10`, via a dedicated `intor_spinor` surface.

Concretely, the acceptance bar:

```
intor_spinor(H2O/cc-pVDZ, "int1e_ovlp_spinor")  ==  upstream mol.intor("int1e_ovlp_spinor")
    to atol 1e-10, shape [n2c, n2c], n2c = 2·nao_nr
```

## 2. Why this is a feature, not an audit-fix

Three properties push F-03 out of one-shot territory (this is the rationale the
audit-fix sweep recorded for *not* force-fixing it):

1. **Complex output type.** Spinor integrals are complex; the spherical→spinor
   transform couples `2(2l+1)` real spherical harmonics × {α,β} spin into the
   `j = l ± 1/2` basis with **complex** Clebsch–Gordan coefficients. The
   non-relativistic `IntorOutput` is `Vec<f64>` (real). Overloading it would be
   a breaking change to every consumer (scf, mp2, df, PyO3). → new
   `IntorOutputComplex` + new `intor_spinor` entry point.
2. **No in-sandbox oracle.** The only oracle-free invariants (Hermiticity,
   shape) **pass for wrong** phase/ordering/CG conventions. Correctness
   *requires* a live-PySCF byte comparison. Per the project test guideline,
   shipping an unvalidatable complex transform would be exactly the "fake
   implementation that survives the tests" we must design against.
3. **Upstream integrals are cintx-gated.** Spinor operators must come through
   the cintx resolver, which does not currently ship them (integral-family
   ceiling). The transform-from-spherical approach (below) sidesteps needing
   *native* spinor operators from cintx, but still depends on the spherical
   integrals cintx already provides.

## 3. Chosen architecture — transform-from-spherical

Rather than wait for native spinor operators in cintx, build spinor integrals
from the **already-correct spherical** integrals via the c2s-spinor transform:

```
S_2c = U† · (S_sph ⊗ I₂) · U
```

where, per shell of angular momentum `l`, `U` is the `2(2l+1) × 2(2l+1)`
complex matrix mapping (real-spherical-harmonic ⊗ spin) → j-adapted spinors.
`U` is block-diagonal by shell, so the molecule-level transform is a per-shell
congruence applied to the spherical integral blocks.

- **Provenance for `U`:** libcint `c2s.c CINTc2s_bra_spinor` /
  PySCF `pyscf/gto/mole.py sph2spinor_coeff` (Apache-2.0).
- **Why this architecture:** reuses byte-verified spherical integrals
  (`int1e_ovlp_sph`, etc.), confines the new/unverified surface to the *pure*
  `U` coefficient tables + a complex congruence, and keeps cintx out of the
  critical path for the transform itself.
- **Scope for v1 of this feature:** the one-electron families
  (`int1e_ovlp_spinor`, `int1e_kin_spinor`, `int1e_nuc_spinor`). Two-electron
  spinor ERIs (`int2e_spinor`) are a follow-on (much larger; deferred).

## 4. Scaffold landed in this commit (verifiable today)

`crates/pyscf-gto/src/spinor.rs`:
- `IntorOutputComplex { re, im, shape, layout }` — the complex output container
  (parallel F-order `re`/`im` buffers; no `num_complex` dep). **Used + tested.**
- `n2c_per_shell(l, nctr) -> usize` — the `2·(2l+1)·nctr` AO-count formula.
  **Implemented + unit-tested** (s/p/d/f + the `n2c = 2·nao_sph` relation).
- `intor_spinor(mol, name)` — entry point **stub** → `NotYetImplemented`
  (returns the complex type once T2–T4 land).
- `sph2spinor_coeff(l, row, col, spin)` — coefficient **stub** →
  `NotYetImplemented` (the explicit T2 gap).
- Tests: scaffold stubs error cleanly (never panic, never fake numbers); two
  `#[ignore]`d **oracle-contract** tests encode the T4 verification gate in-tree.

Wiring: `pub mod spinor` + `pub use spinor::{IntorOutputComplex, intor_spinor}`;
the `intor`/`intor_cross` `_spinor` arms now redirect to `intor_spinor` and this
plan. `mol.nao_2c` left as the documented `0` stub (T1) — deliberately NOT
edited blind: it needs reliable `_bas` raw-array indexing (ANG_OF/NCTR_OF), best
done as a focused, reviewed task.

## 5. Task DAG

| Task | Depends on | Description | Verification |
|------|-----------|-------------|--------------|
| **T1** | — | Compute real `mol.nao_2c` = Σ `n2c_per_shell(l, nctr)` over shells (walk `_bas` ANG_OF/NCTR_OF, or `_basis` shells). Wire in `build_from`. | Unit: H2/STO-3G → 4; mol with p/d shells → hand counts; `nao_2c == 2·nao_nr` for spherical bases. (oracle-free) |
| **T2** | — | Transcribe `sph2spinor_coeff` CG tables (l = 0..4 at least) from libcint `CINTc2s_bra_spinor`. Real `(re, im)` per `(l, row, col, spin)`. | **Live-PySCF**: per-l `U` block equals `mol.sph2spinor_coeff()` to 1e-12. Plus unitarity `U†U = I`. |
| **T3** | T1, T2 | Implement the per-shell congruence `S_2c = U†(S_sph⊗I₂)U` over spherical integral blocks → assemble `IntorOutputComplex` (F-order, `[n2c, n2c]`). | Unit: shape/Hermiticity; block-diagonal correctness on a 1-atom, 1-shell case computable by hand. |
| **T4** | T3 | Build the live-PySCF spinor oracle harness; wire `intor_spinor` for `int1e_{ovlp,kin,nuc}_spinor`. Un-`#[ignore]` the contract tests. | **GATE:** byte-identity to upstream `mol.intor("int1e_*_spinor")` at atol 1e-10 on H2O/cc-pVDZ + a heavier-l fixture. |
| **T5** | T4 | PyO3 bridge: expose `intor_spinor` returning a complex array (numpy `complex128`). | Python parity test under maturin + live PySCF. |
| T6 (opt) | T4 | `int2e_spinor` (two-electron). Separate, larger effort. | byte-identity vs upstream 2e spinor. |

Critical path: **T1∥T2 → T3 → T4** (→ T5). T4 is the hard gate (needs the
live-PySCF environment that also blocks F-14).

## 6. Risks / open questions

- **Phase & ordering conventions.** libcint vs PySCF Condon–Shortley phase and
  the m-ordering of real spherical harmonics must match the *spherical*
  integrals we transform. This is the most likely source of a subtle, Hermitian-
  but-wrong result — hence T4's byte oracle is non-negotiable.
- **cart vs sph.** Transform is defined from the *spherical* basis; if
  `mol.cart`, derive the spherical integrals explicitly (don't transform from
  cartesian).
- **Environment dependency.** T4/T5 cannot complete in the CPU sandbox; they
  need maturin + live PySCF (shared blocker with F-14).
- **No oracle-free correctness proof exists** for the transform — do not relax
  T4 to invariant-only checks.

## 7. Definition of done

- `intor_spinor("int1e_{ovlp,kin,nuc}_spinor")` byte-matches upstream (atol 1e-10).
- `mol.nao_2c` correct for s/p/d/f bases; `= 2·nao_nr` (spherical).
- Contract tests un-`#[ignore]`d and green under the live-PySCF oracle.
- PyO3 surface returns numpy `complex128` with Python parity.
- No regression to the real `intor`/`IntorOutput` path or its consumers.
