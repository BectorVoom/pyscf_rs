# F-03 — Spinor (relativistic 2-component) integral representation

**Status:** NEAR-COMPLETE (segmented bases) — **architecture pivoted to
route-through-cintx**. T1 + `intor_spinor` 1e (T2/T3) + `int2e_spinor` 2e (T6)
+ PyO3 `complex128` bridge (T5) all landed; **byte-identity to upstream PySCF
2.12.1 CONFIRMED** (T4) at atol 1e-10 for STO-3G-class bases. See §3a + §3b.
Residual: (a) cintx wires the cart→spinor transform for SEGMENTED (`nctr==1`)
bases only — general contraction errors cleanly; (b) for multi-shell-same-l
bases (e.g. 6-31g) the integrals are eigenvalue-identical to upstream but the
global AO **ordering** differs (a separate intor_spinor/cintx ordering matter).
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

## 3a. ARCHITECTURE PIVOT (2026-06-01) — route through cintx

**The premise behind §2.3 / §3 was wrong.** cintx **does** ship a
libcint-parity-validated cart→spinor transform and resolves `_spinor` operator
symbols:

- `cintx/crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — the
  `g_trans_cart2j` CG tables (l=0..4) extracted verbatim from libcint
  `cart2sph.c` (this is literally the artifact T2 proposed to hand-transcribe).
- `…/transform/c2spinor.rs` — the cart→spinor drivers (`cart_to_spinor_sf_2d`,
  `_si`, `_4d`, `_3c2e`, derivatives), transcribed verbatim from libcint.
- `cintx/crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs` —
  gates `int1e_{ovlp,kin,nuc}_spinor` against **vendor libcint at atol 1e-12**
  (exactly F-03 v1's three operators).

**Consequence:** the "transform-from-spherical" congruence (T2 CG transcription
+ T3 manual `U†(S⊗I)U`) is unnecessary and *less* verifiable than reusing
cintx. `intor_spinor` now mirrors the real `intor` arity-2 path:

1. Build a `Representation::Spinor`-tagged `BasisSet`
   (`projection::build_cintx_spinor_basis_set`) — cintx applies its cart→spinor
   transform and the representation-aware `BasisMeta` reports `n2c`-unit AO
   offsets/counts.
2. Evaluate each shell pair via `SessionRequest` (`Representation::Spinor`),
   read the complex block with `IntegralTensor::complex_values()`.
3. Stitch per-pair blocks (F-order, bra-fastest — identical convention to the
   scalar `stitch_arity2_block`) into one F-order `[n2c, n2c]`
   `IntorOutputComplex`.

**T2 and T3 are therefore RESOLVED-via-cintx (obsoleted), not hand-built.**
Implemented in `crates/pyscf-gto/src/spinor.rs` + `projection.rs`; verified by
`crates/pyscf-gto/tests/spinor_intor.rs` (shape, `n2c == 2·nao_nr`,
Hermiticity `S=S†`, real-positive overlap diagonal, finiteness, name
normalisation, int2e-deferred error). Per-pair numerics inherit cintx's vendor
parity.

**Residual gap (still live-PySCF gated, shared with F-14):** the *global
shell-block ordering* of the assembled `[n2c,n2c]` matrix vs upstream
`mol.intor("int1e_*_spinor")` — captured by the `#[ignore]`d
`ovlp_spinor_byte_matches_upstream` contract. cintx's spinor `ao_loc` ordering
mirrors libcint, but matching upstream PySCF's global layout byte-for-byte
needs the live-PySCF environment. T5 (PyO3 complex128 bridge) is unchanged.

The original transform-from-spherical design is retained below for historical
context; it is **superseded** by §3a.

## 3b. T5 PyO3 bridge + T4 byte-identity CONFIRMED (2026-06-01)

The sandbox in fact carries the live oracle: `.upstream-venv` has upstream
PySCF 2.12.1 + numpy; `maturin` installs into `.venv` (nightly toolchain
required — the workspace manifest doesn't parse on stable). So T4/T5 were
*not* permanently blocked.

- **T5** — `crates/pyscf-py/src/gto.rs` exposes `Mole.intor_spinor(name)`
  returning a numpy **`complex128`** ndarray (F-order; `re`/`im` interleaved via
  `numpy::Complex64`), plus a `nao_2c()` accessor. Verified through `maturin
  develop`: the overlay `pyscf.gto.M(...).intor_spinor(...)` yields complex128,
  F-contiguous, Hermitian output. Python suite:
  `python/pyscf/tests/test_intor_spinor.py`.
- **T4 — byte-identity CONFIRMED** against upstream PySCF 2.12.1 at atol 1e-10
  (cross-venv, logged in `log/`): `int1e_{ovlp,kin,nuc}_spinor` on H2O/STO-3G
  (≤2e-14) and `int2e_spinor` on H2/STO-3G (6e-16). The `#[ignore]`d Rust
  contract placeholder is superseded by the Python parity test (which skips
  gracefully when upstream's C-libs aren't importable in a minimal venv).
- **Ordering caveat** — for multi-shell-same-l bases (e.g. 6-31g: 3×s, 2×p on
  C) the result is eigenvalue-identical to upstream but the global AO **index
  ordering** differs (a pure permutation). This is an `intor_spinor`/cintx
  global-assembly-ordering matter, *not* a transform or binding defect, and is
  the remaining follow-up for full upstream index-parity on general bases.
- **Segmentation limit** — cintx wires the cart→spinor transform (1e and 2e)
  for `nctr==1` only; `intor_spinor` guards general contraction with a clean
  `NotYetImplemented` (`ensure_segmented_for_spinor`).

## 3. ~~Chosen architecture~~ (SUPERSEDED by §3a) — transform-from-spherical

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
plan. `mol.nao_2c` is now computed for real (T1, commit `79061a3`) by walking
`_bas` (ANG_OF/NCTR_OF) and summing `n2c_per_shell` — the prior documented `0`
stub is gone.

## 5. Task DAG

| Task | Depends on | Description | Verification |
|------|-----------|-------------|--------------|
| ~~**T1**~~ ✅ | — | ~~Compute real `mol.nao_2c` = Σ `n2c_per_shell(l, nctr)` over shells (walk `_bas` ANG_OF/NCTR_OF, or `_basis` shells). Wire in `build_from`.~~ **DONE** (`79061a3`): walks `_bas` ANG_OF/NCTR_OF, sums `n2c_per_shell`. | ✅ `tests/nao_2c.rs`: H2/STO-3G → 4; `nao_2c == 2·nao_nr` across sto-3g/6-31g/cc-pvdz (s/p/d). `cargo +nightly test -p pyscf-gto` green. (oracle-free) |
| ~~**T2**~~ | — | ~~Transcribe `sph2spinor_coeff` CG tables…~~ **OBSOLETED by §3a** (`28876f4`→pivot): cintx already ships the libcint `g_trans_cart2j` CG tables + validated cart→spinor drivers; no hand-transcription. | n/a — superseded; correctness via cintx vendor parity. |
| ~~**T3**~~ | — | ~~Per-shell congruence `S_2c = U†(S_sph⊗I₂)U`…~~ **OBSOLETED by §3a**: replaced by routing through cintx's cart→spinor transform + per-pair stitch in `spinor.rs::intor_spinor`. | ✅ `tests/spinor_intor.rs`: shape/Hermiticity/diagonal/finiteness. |
| **T4** | (pivot) | ✅ **CONFIRMED (segmented)** — byte-identity to upstream PySCF 2.12.1 verified at atol 1e-10 (see §3b): 1e ovlp/kin/nuc on H2O/STO-3G (≤2e-14), int2e on H2/STO-3G (6e-16). **Caveat:** multi-shell-same-l ordering differs (eigenvalue-identical); general contraction unsupported. | ✅ cross-venv proof (logged) + `python/pyscf/tests/test_intor_spinor.py` (runs in full env). |
| **T5** | T4 | ✅ **DONE** — `crates/pyscf-py/src/gto.rs` `Mole.intor_spinor(name)` → numpy `complex128` (F-order) + `nao_2c()`; built via `maturin develop`. | ✅ `python/pyscf/tests/test_intor_spinor.py`: complex128/F-order/Hermitian/ERI-symmetry (pass) + upstream byte-identity (pass in full env; skips if upstream C-libs absent). |
| **T6** | (pivot) | ✅ **DONE (segmented)** via route-through-cintx — `intor_spinor` extended to arity-4 `int2e_spinor` (plain spin-free Coulomb), F-order `[n2c;4]`, mirroring `evaluate_arity4`. cintx's `cart_to_spinor_sf_4d` is vendor-validated (`cintx-oracle/.../oracle_gate_closure.rs` int2e_spinor vs libcint 6.1.3). **Limitation:** cintx wires the 4D spinor transform for SEGMENTED bases only (`nctr==1`); general contraction errors cleanly. | ✅ `tests/spinor_intor.rs`: shape `[n2c;4]`, finiteness, nonzero, ERI symmetries `(ij\|kl)==(kl\|ij)` + `(ij\|kl)==conj(ji\|lk)`, nctr>1 clean-error. **Open:** global-ordering byte-identity (live-PySCF, shared T4 gate). |

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
