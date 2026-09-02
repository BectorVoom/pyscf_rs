# Phase 18 — Periodic gradients + stress + geomopt — CONTEXT

**Written:** 2026-09-02, before any Phase-18 code.
**Read this before `18-01-PLAN.md`.** Everything here was verified against the
vendored PySCF **2.12.1** tree (`pyscf/__init__.py:38`), the vendored `cintx`
tree (`../cintx`) and the current Rust workspace on 2026-09-02; every claim
carries the file and line that proves it.

`PBC-MASTER-PLAN.md §8.10` sizes this phase at **seven plans**. Its line-count
inventory is **exact** — `wc -l pyscf/pbc/grad/*.py` = 2,848 and
`pyscf/pbc/geomopt/*.py` = 269, and every per-plan figure in its table
(`krhf` 418, `rhf` 188, `kuhf` 124, `uhf` 103, `krks` 141, `kuks` 135,
`krks_stress` 404, `kuks_stress` 308, `rks_stress` 462, `uks_stress` 246,
`geometric_solver` 246) reproduces to the line.

**But the seven-plan table is wrong about the starting state in eight ways, its
gate is the fifth unmeasured, self-contradictory gate this project has found,
and one of its seven plans is scope this port would be inventing.** This
document records what is actually there, what is actually missing, and what
cannot be believed as written.

---

## 1. The scope corrections, in order of consequence

### 1.1 The gamma-point half is a multigrid-v2 program, and five of its entry points do not exist

`§8.10` costs `grad/rhf` (188 l) and `grad/uhf` (103 l) as ordinary
gamma-point ports. Read the source. `pbc/grad/rhf.py:42-47`:

```python
if hasattr(mf, '_numint'):
    ni = mf._numint
    assert isinstance(ni, MultiGridNumInt2)
else:
    ni = mf.with_df
    raise NotImplementedError
```

There is **no non-multigrid branch at all** — the `else` is
`raise NotImplementedError`. Every quantity `grad_elec` then consumes is a
`MultiGridNumInt2` method:

| upstream call | defined at | Rust state |
|---|---|---|
| `ni.get_veff_ip1(dm, xc_code, kpts)` | `multigrid_pair.py:748-860` (113 l) | **absent** |
| `ni.get_nuc_ip1(kpts)` | `multigrid_pair.py:893-895` | **absent** |
| `ni.get_nuc_nuc_grad(dm, kpts)` | `multigrid_pair.py:896-921` | **absent** |
| `ni.get_vpploc_part1_ip1(kpts)` | `pp.py:135-150` (16 l) | **absent** |
| `ni.vpploc_part1_nuc_grad(dm, kpts)` | `pp.py:151-201` (51 l) | **absent** |

`crates/pyscf-pbc-dft/src/multigrid/pair.rs`'s `MultiGridNumInt2` (`:959`)
exposes exactly five methods — `get_nuc` (`:994`), `get_pp` (`:1002`),
`eval_rho_g` (`:1011`), `get_j` (`:1041`), `nr_rks` (`:1067`) — and **none of
the five above**. That is ~209 upstream lines of gradient-only multigrid work
that `§8.10` costs at zero, and it is a hard prerequisite of both `grad/rhf`
and `grad/uhf`. It is plan **18-09**, and 18-10 is blocked on it.

`grad/rks.py` (29 l) and `grad/uks.py` (29 l) are `class Gradients(rhf.Gradients)`
and `class Gradients(uhf.Gradients)` with no body — they inherit the multigrid
assertion wholesale, so they are free once 18-10 lands and are folded into it.

### 1.2 There is exactly ONE density-fitting route with a gradient, and it is not the one a `.density_fit()` user gets

`grep -rn "def get_jk_e1\|def get_j_e1\|def get_k_e1" pyscf/pbc/df/*.py`
returns **three lines, all in `fft.py`**:

```
pyscf/pbc/df/fft.py:324:    def get_jk_e1(self, dm, kpts=None, kpts_band=None, exxdiv=None):
pyscf/pbc/df/fft.py:330:    def get_j_e1(self, dm, kpts=None, kpts_band=None):
pyscf/pbc/df/fft.py:335:    def get_k_e1(self, dm, kpts=None, kpts_band=None, exxdiv=None):
```

and the class graph shows nothing inherits them:

```
pyscf/pbc/df/fft.py:185   class FFTDF(lib.StreamObject)
pyscf/pbc/df/df.py:125    class GDF(lib.StreamObject, aft.AFTDFMixin)
pyscf/pbc/df/aft.py:585   class AFTDF(lib.StreamObject, AFTDFMixin)
pyscf/pbc/df/rsdf.py:75   class RSGDF(GDF)
pyscf/pbc/df/mdf.py:49    class MDF(df.GDF)
```

`GDF`, `RSGDF`, `MDF` and `AFTDF` are **not** subclasses of `FFTDF`. So
`krhf.GradientsBase.get_jk` (`pbc/grad/krhf.py:262`, `self.base.with_df.get_jk_e1(...)`)
raises `AttributeError` on every route except FFTDF, and there is no periodic
analytic gradient for GDF/MDF/RSDF/AFTDF anywhere in PySCF 2.12.1.

This is survivable *for this port* because `KRHF::new` already defaults to
FFTDF (`crates/pyscf-pbc-scf/src/krhf.rs:46`,
`let with_df = Fftdf::new(cell, kpts)`). It is **not** survivable silently: a
user who calls `.density_fit()` and then asks for a gradient must get a loud,
named refusal, never a fallback to a different route whose number would be
plausible and wrong. That is 18-04's Task 4, and `§8.10` does not mention it.

`§8.10`'s own note — *"FFTDF J/K gradients are grid-based, not integral-based …
`int2e_ip1` is never called on the FFTDF path"* — is **correct** and is the
best single observation in the section. It just understates the consequence:
grid-based is not a convenience, it is the only route that exists.

### 1.3 `§8.10`'s two cintx blockers are already resolved, at the same maturity as code this port ships today

`§8.10` says the PP-gradient half is blocked: *"`int3c1e_ip1_r{2,4,6}_origk`
(`pp_int.py:187`) and `int1e_r{2,4}_origi_ip2` (`pp_int.py:454`). These are
cintx Wave 0.5. If they have not landed, plan 18-01 ships the assembly and
`#[ignore]`-gates only the PP-gradient contribution."* `PBC-MASTER-PLAN §2.4`
marks all five ❌ **declared, unverified, no dispatch arm** and
**BLOCKS plan 18-01**.

All five are in the manifest today
(`../cintx/crates/cintx-ops/src/generated/api_manifest.csv`):

```
"int1e_r2_origi_ip2_sph"     oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int1e_r4_origi_ip2_sph"     oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_ip1_r2_origk_sph"   oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_ip1_r4_origk_sph"   oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_ip1_r6_origk_sph"   oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
```

**That is byte-for-byte the same status as the five families this port already
runs GTH pseudopotentials on:**

```
"int1e_r2_origi_sph"         oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int1e_r4_origi_sph"         oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_r2_origk_sph"       oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_r4_origk_sph"       oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
"int3c1e_r6_origk_sph"       oracle_covered=false  profiles="unstable-source"  stability="unstable_source"
```

and the port already carries the feature that turns them on —
`crates/pyscf-pbc-gto/Cargo.toml:70`, `gth-pp = ["cintx-rs/unstable-source-api"]`,
which is **`default = ["gth-pp"]`** (`:54`) because *"every reference system in
PBC-MASTER-PLAN §9.2 uses `gth-pade`, so a pyscf-pbc-gto without this flag
cannot build an `hcore` for any of them."*

**R-14 is already disproved for this exact family class, and the port already
has the test that disproves it.** `Cargo.toml:60-63` records the measured
behaviour: *"without the flag `SessionRequest::evaluate` refuses with
'source-only symbol … requires feature `unstable-source-api`' — it does NOT
fall open to the unweighted parent (the R-13 silent-wrong-answer risk), which
`tests/cintx_moment_weighted_available.rs` re-checks on every run."*

**Ruling:** 18-03 does not `#[ignore]`-gate the PP gradient and does not wait
for a cintx wave. It enables the five `_ip1_`/`_ip2` symbols under the existing
`gth-pp` feature and **extends `cintx_moment_weighted_available.rs` to cover
them**, so the R-14 refusal is re-proved for the derivative half by the same
test that proves it for the energy half. `PBC-MASTER-PLAN §2.4`'s two
"BLOCKS plan 18-01" rows and `§8.10`'s `#[ignore]` fallback are both retired by
this section. What does *not* change: `oracle_covered=false` means the numeric
gate for these five is **upstream PySCF, not cintx's own oracle** — exactly as
`Cargo.toml:68-70` already records for the energy half.

### 1.4 Half of `§8.10`'s plan 18-01 is already shipped

`§8.10`'s 18-01 is *"Gradient integral infrastructure: `pbc_intor` derivative
families (`int1e_ipovlp`, `int1e_ipkin` — both cintx-ready) over lattice
images"*. Both are already in `SUPPORTED_INTORS`
(`crates/pyscf-pbc-gto/src/pbc_intor.rs:275-277`), alongside `int1e_ipnuc`,
with the `ComponentLeadingFOrder { components: 3 }` layout resolved through
`pyscf_gto::layout_table` (`:328-331`) and a shipped test
(`crates/pyscf-pbc-gto/tests/pbc_intor.rs:375-393`,
`int1e_ipovlp returns the 3 component blocks its layout advertises`).

What `§8.10` misses is that the *reuse hook the stress tensor needs* is
shipped too: `intor_cross_with_images(intor, cell1, cell2, kpts, opts, ls,
neighbor_list)` (`pbc_intor.rs:295`), written for plan 10-06 precisely so that
*"rebuilding `Ls` — an `O(nimgs · natm)` filter — per operator is pure waste."*
`18-REVIEW §3.2` (D-PBC-30 clause 2) turns that into the phase's single largest
integral-side saving.

### 1.5 `rks_stress` is the base of the other three, and `§8.10` schedules it third

`§8.10`'s 18-05 lists *"`krks_stress` (404 l), `kuks_stress` (308 l),
`rks_stress` (462 l), `uks_stress` (246 l)"*. The dependency runs the other
way. All three of the others import **the same eight symbols** from
`rks_stress`:

```
pyscf/pbc/grad/krks_stress.py:74-83   from pyscf.pbc.grad.rks_stress import (
pyscf/pbc/grad/kuks_stress.py:26-35       strain_tensor_dispalcement, _finite_diff_cells,
pyscf/pbc/grad/uks_stress.py:24-33        _get_weight_strain_derivatives, _get_coulG_strain_derivatives,
                                          _eval_ao_strain_derivatives,
                                          _get_vpplocG_strain_derivatives,
                                          _get_pp_nonloc_strain_derivatives, ewald)
```

`rks_stress.py` ships first (**18-12**), the other three follow (**18-13**).
This is the same defect shape as `16-CONTEXT §1.4` (`§8.8` built the EOM base
class last, and it is inherited by the other two).

Two further facts about the stress half that `§8.10` does not record:

* **There is no HF stress.** `ls pyscf/pbc/grad/` has `rks_stress`,
  `uks_stress`, `krks_stress`, `kuks_stress` and **no `rhf_stress` /
  `khf_stress`**. Upstream's stress tensor is Kohn–Sham only. A KRHF stress is
  not a port, it is new physics, and this phase does not write it.
* **`§8.10` is right that stress needs no derivative integrals**, and the proof
  is `rks_stress.py:86-111`: `get_ovlp` and `get_kin` central-difference
  `cell1.pbc_intor('int1e_ovlp')` against `cell2.pbc_intor('int1e_ovlp')` at a
  hard-coded `disp = 1e-5`. What it needs instead is a **new AO evaluation
  family that has no Rust counterpart** — see §1.6.

### 1.6 The strain-tensor AO family is a C kernel with no Rust counterpart, and `§8.10` costs it at zero

`_eval_ao_strain_derivatives` (`rks_stress.py:127-147`) calls
`cell.pbc_eval_gto('GTOval_sph_deriv%d_strain_tensor' % deriv, ...)`. That
feval resolves to `pyscf/lib/pbc/grid_ao.c:431`,
`PBCeval_cart_for_strain_tensor_iter`, whose comment block confirms the shape
(`:442`, `int comp_strain_tensor = ncomp * 3 * 3`).

`grep -rn "strain_tensor" crates/` returns **nothing**. This is a new
`pyscf-kernels` collocation family — the 3×3 strain-tensor AO derivative at
`deriv = 0` and `deriv = 1`, sph and cart — required by all four stress
modules and by every one of upstream's own component-level stress tests
(`test_rks_stress.py:125` `test_eval_ao_cart`, `:142` `test_eval_ao_sph`,
`:160` `test_eval_ao_deriv1_cart`, `:178` `test_eval_ao_deriv1_sph`). It is
plan **18-11**, it blocks 18-12 and 18-13, and by `17-CONTEXT`'s ALG-06
corollary — *"`pyscf-pbc-dft` may not name cubecl, so every collocation kernel
goes in `pyscf-kernels`"* — it lives in `pyscf-kernels`, not in
`pyscf-pbc-grad`.

### 1.7 `§8.10`'s 18-06 asks for lattice degrees of freedom. Upstream has none.

`§8.10`'s 18-06: *"`pbc/geomopt/geometric_solver` (246 l) — reuse
`pyscf-geomopt`'s native BFGS+RFO engine; **add lattice degrees of freedom**."*
`ROADMAP.md:464` repeats it as *"periodic geometry/lattice optimization"*.

`pyscf/pbc/geomopt/` is two files — `geometric_solver.py` (246) and
`__init__.py` (23). `grep -rn "lattice" pyscf/pbc/geomopt/*.py` returns **one
hit, `geometric_solver.py:223`, inside the `__main__` example block**. The
optimizer moves atoms and nothing else:

```python
pyscf/pbc/geomopt/geometric_solver.py:80   cell.set_geom_(coords, unit='Bohr')
pyscf/pbc/geomopt/geometric_solver.py:81   energy, gradients = g_scanner(cell)
```

There is no `cell.a` update, no strain coordinate, no variable-cell relaxation
anywhere in `pyscf/pbc`. **Lattice optimization is invented scope**, it has no
upstream oracle, and D-PBC-15 ("add-on layer, never a fork") forbids shipping
it inside the port surface. 18-14 ports what exists — the atom-coordinate
optimizer over `pyscf-geomopt`'s native BFGS+RFO engine — and records the
lattice-DOF request as a post-v2.0 feature whose *precondition* (an analytic
`dE/dε`) 18-12/18-13 deliver. `ROADMAP.md:464` must be restated to match.

A second, smaller trap in the same area: `pbc/grad/rhf.py:169-178` and
`pbc/grad/krhf.py:290-298` define `GradientsBase.optimizer(solver='ase')` whose
**only** accepted value is `'ase'`; `'geometric'` raises
`RuntimeError(f'Optimization solver {solver} not supported')`. The geomeTRIC
path is reachable *only* through `pyscf.pbc.geomopt.optimize(...)`
(`__init__.py:18-23`). Two entry points, one of which cannot reach the module
18-14 ports. Mirror both, including the refusal.

### 1.8 `aoslice_by_atom` returns a 2-tuple; three Phase-18 bodies need the 4-tuple

`crates/pyscf-gto/src/aoslice.rs:33` is
`pub fn aoslice_by_atom(mol: &Mole) -> Result<Vec<(usize, usize)>, PyscfRsError>`
— `(p0, p1)` only. Upstream unpacks four fields in the bodies this phase ports:

```
pyscf/pbc/grad/kuhf.py:37    shl0, shl1, p0, p1 = aoslices[ia]
pyscf/pbc/grad/uhf.py  (via mol_uhf.grad_elec:63)  shl0, shl1, p0, p1 = aoslices[ia]
pyscf/pbc/grad/rhf.py:83     shls_atm = mol._bas[:,ATOM_OF]     # _contract_vhf_dm
```

`aoslice.rs` is **new in the working tree** (`git status`: `A crates/pyscf-gto/src/aoslice.rs`),
so widening it to `(shl0, shl1, p0, p1)` before it has downstream callers is
the cheap moment. 18-02 Task 2.

---

## 2. The gate: `ROADMAP.md`'s 1e-14 is arithmetically impossible, and the stress gate is in the wrong units

`ROADMAP.md:464`: *"**Gate:** always-on `verify_fd` central difference agrees
to **1e-14 Ha/Bohr**; stress agrees with `dE/da` finite difference to
**1e-5**."*

### 2.1 The gradient number contradicts three things at once

1. **The master plan.** `§8.10`'s own 18-07 row says *"`verify_fd`
   central-difference gate at **1e-6 Ha/Bohr**"* — eight orders looser than the
   ROADMAP, for the same quantity. Same contradiction Phases 14, 15, 16 and 17
   each found.
2. **The shipped harness.** `crates/pyscf-grad/src/verify_fd.rs:35` is
   `pub const FD_TOL: f64 = 1e-6`, with the docstring *"A correct analytical
   gradient agrees with the central difference of the energy to well within
   this on every in-scope method."* The gate as written fails against the
   project's own constant.
3. **Upstream's own test suite.** Every periodic FD-vs-analytic assertion in
   `pyscf/pbc/grad/test/` is at **6 decimal places (5e-7)**, and the DFT+U ones
   at **5 (5e-6)**:

   | test | line | assertion | tolerance |
   |---|---|---|---|
   | `test_krhf.py` | `:55` | `g[1,2]` vs `(e1-e2)/disp` | 6 dp = 5e-7 |
   | `test_kuhf.py` | `:54` | same | 6 dp |
   | `test_krks.py` | `:58,72,87` | LDA / GGA / hybrid | 6 dp |
   | `test_kuks.py` | `:58,72,87` | LDA / GGA / hybrid | 6 dp |
   | `test_krkspu.py` | `:87,92` | `lib.fp(g)`, FD | **5 dp = 5e-6** |
   | `test_kukspu.py` | `:67,72` | `lib.fp(g)`, FD | **5 dp** |

**And 1e-14 is below the arithmetic floor of the method the gate names.** A
central difference at step `h` carries a truncation error `O(h²)` and a
cancellation error `≈ 2·ε·|E| / h`. Upstream uses `h = 1e-5` for HF
(`test_krhf.py:38`) and `h = 1e-3` for DFT (`test_krks.py:37`). For a diamond
`gth-szv`/`gth-pade` cell with `|E| ~ 10 Ha`, the cancellation floor alone is
`2 × 2.2e-16 × 10 / 1e-5 ≈ 4e-10`, and the truncation term at the DFT step is
`~1e-6`. **No correct implementation can pass 1e-14**, because no *exactly
correct* gradient can: the number being compared against is itself only
accurate to ~1e-10. The gate does not distinguish a right answer from a wrong
one — it rejects both.

Note also a step-size convention mismatch that will silently double an
apples-to-apples comparison if it is not written down: upstream's `disp` is the
**full** step (`e1` at `+disp/2`, `e2` at `-disp/2`, divided by `disp` —
`test_krhf.py:53-55`), while `verify_fd`'s `disp` is the **half** step
(`plus[ia][c] += disp`, divided by `2*disp` — `verify_fd.rs:90-109`), with
`DEFAULT_DISP = 1e-4` (`:30`). Upstream's `1e-5` is this harness's `5e-6`.

### 2.2 The stress number is in the wrong units

`ROADMAP.md` and `§8.10` both say *"stress vs finite-difference of `E(a)` at
**1e-5 Ha**"*. Upstream's assertion is
(`test_rks_stress.py:406`, `:424`, `:442`, and the `krks`/`kuks`/`uks` mirrors):

```python
assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6
```

The **`/vol`**. Upstream's stress is `dE/dε` divided by the cell volume — a
*pressure*, in Ha/Bohr³, not an energy. For the port's own diamond
(`test_systems.rs`, `vol = 76.55488063251218` Bohr³) upstream's 1e-6 Ha/Bohr³
corresponds to `≈ 7.7e-5 Ha` on the energy derivative, so the ROADMAP's
"1e-5 Ha" is simultaneously the wrong dimension **and**, read as an energy,
~8× tighter than upstream's own number. A gate whose units do not match the
quantity is untestable; this one has both failure modes at once.

The **component-level** strain-derivative tests are a separate, much tighter
tier and must not be collapsed into the same number:
`test_rks_stress.py:388` asserts `abs(dat[i,j] - de/2e-5) < 1e-8` for the
individual `ovlp` / `kin` / `weight` / `coulG` / AO strain derivatives.

### 2.3 The replacement: five gates, and 18-01 measures every floor before any of them is written

Following 17-01's ruling exactly — *measure the floor, then write the gate*.

* **Gate A — component strain derivatives, port vs its own finite difference.**
  `ovlp`, `kin`, `weight`, `coulG`, `vpplocG`, and the strain AO at
  `deriv = 0, 1` in both sph and cart, each against `_finite_diff_cells` at
  `disp = 1e-5`. Target **1e-8**, upstream's own number
  (`test_rks_stress.py:388`). No SCF is involved, so there is no convergence
  path and no second-solution noise — this is the tightest gate in the phase
  and the one that localises a wrong strain AO to a single component.
* **Gate B — analytic gradient vs this port's `verify_fd`, per method.**
  **1e-6 Ha/Bohr** (`FD_TOL`) for KRHF/KUHF/KRKS/KUKS and the gamma bodies;
  **5e-6** for `krkspu`/`kukspu`, because upstream's own DFT+U assertions are
  5-decimal (`test_krkspu.py:87,92`) and a blanket 1e-6 would fail on upstream's
  own numbers. Needs no PySCF.
* **Gate C — analytic gradient vs upstream, per method**, at upstream's
  published `lib.fp(g)` values: `-0.9017171774435333` (KRHF and KUHF,
  `test_krhf.py:50` / `test_kuhf.py:49`), `-0.22166962318360375` (LDA),
  `-0.21844074846755882` (GGA), `-0.19544969829285652` (hybrid),
  `-0.42370983409650914` (DFT+U), each at the decimal count upstream itself
  uses.
* **Gate D — stress vs finite difference of `E(ε)`**, at **1e-6 Ha/Bohr³**,
  i.e. `|dat[i,j] − (E₊−E₋)/2h/vol|`, `h = 1e-3` (`test_rks_stress.py:390-406`).
  Stated in pressure units, with `vol` named in the assertion so the dimension
  cannot drift again.
* **Gate E — the gamma-point multigrid-v2 bodies are NOT held to B or C at the
  same tolerance.** `17-12-SUMMARY.md` records v2's accuracy floor against the
  reference route as the screening floor `precision · EXTRA_PREC` (~1e-6 on
  the electron count), and 17-01 measured v2 carrying a *mesh-independent*
  ~2e-8 (diamond) / 1.5e-7 (si) definitional gap against FFTDF. A gradient
  built on v2 inherits both. 18-01 measures the gamma-path floor and 18-10
  gates against the measurement, never against the k-point number.

`ROADMAP.md:464` and `PBC-MASTER-PLAN §8.10`'s 18-07 row are both restated by
18-01, in one edit, with the measured numbers.

---

## 3. Traps recorded in advance, each with the line that proves it

1. **`hcore_deriv` re-evaluates the whole AO table per atom.**
   `krhf.py:132-147`: the closure body loops `for kn, kpt in enumerate(kpts):
   ao = eval_ao_kpts(cell, coords, kpt)[0]`, and `grad_elec` calls it once per
   atom (`:60-68`). The AO table does not depend on `atm_id`. Ported literally,
   a 2-atom cell pays 2× and an 8-atom cell 8× the full-grid AO cost for
   nothing. Hoist it — see D-PBC-30 clause 6, which also states the memory
   price of hoisting.

2. **The energy path's k-pair conjugate symmetry does NOT transfer to the
   gradient.** `crates/pyscf-pbc-df/src/fft_jk.rs:265-330` documents the
   identity `rho1^{21}[(i,j),g] = conj(rho1^{12}[(j,i),g])`, which holds
   because bra and ket carry the *same* AO table. In `get_k_e1_kpts`
   (`fft_jk.py:391`) the bra carries `ao1T[1:,p0:p1]` — the **derivative** AO —
   and the ket carries `ao2T`. The `(j,i)` swap moves the derivative to the
   ket, and the identity does not close. `kk_symmetry` must be **refused** on
   the gradient route, with a test that asserts the refusal rather than a
   comment that asks for it.

3. **`de[x] /= nkpts` is inside the atom loop, `vppnl_nuc_grad` is outside it.**
   `krhf.py:67` divides each atom row by `nkpts`; `:69` then adds
   `pp_int.vppnl_nuc_grad(cell, dm0, kpts) / nkpts` to the **whole array**, and
   `:68`'s `extra_force` is added *after* the division but *before* that. Three
   different scopes in five lines. Reproduce the order literally; a reordering
   here is invisible to a `lib.fp` comparison that happens to be dominated by
   one term.

4. **`_contract_vhf_dm` is screened by default.**
   `pbc/grad/rhf.py:30`, `SCREEN_VHF_DM_CONTRA = getattr(__config__,
   'pbc_rhf_grad_screen_vhf_dm_contract', True)`. The screened and unscreened
   contractions differ in the last bits. The port has the neighbour list
   already (`crates/pyscf-pbc-gto/src/neighborlist.rs:221`,
   `build_neighbor_list_for_shlpairs`), so both branches are cheap to have —
   but the FD gate must be run against whichever one ships as the default, and
   the default must be stated, not inherited by accident.

5. **`_finite_diff_cells` rebuilds the cell when symmetry is on.**
   `rks_stress.py:82-84`: `if cell.space_group_symmetry: cell1.build(False,
   False)`. 17-CONTEXT already recorded that `check_mesh_symmetry` can
   **enlarge** `cell.mesh` (`symmetry.py:96`). A strain finite difference whose
   two cells rebuilt to different meshes measures the mesh, not the strain.
   Pin the mesh on both sides — the same correction 14-VERIFICATION had to make
   for MDF and 17-CONTEXT for the ksymm gates.

6. **`get_ovlp`/`get_kin` in `rks_stress` hard-code `disp = 1e-5`**
   (`:88`, `:101`) — a *different* step from the `disp = 1e-3` used by the
   end-to-end stress tests. Two finite differences at two steps inside one
   quantity. Do not unify them; the tight one is an integral difference and the
   loose one an SCF difference, and they have different noise floors.

7. **Upstream's `blksize` subtracts `mem_now` twice.** `fft_jk.py:361-363`:
   `max_memory = mydf.max_memory - mem_now`, then
   `blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/3/ngrids/nao)))`.
   The port must respect `PYSCF_MAX_MEMORY` (`aftdf.rs:84-85`) but must not
   port this arithmetic literally — same ruling shape as `16-REVIEW §2.4`
   ("upstream's own memory estimate is a documented TODO — do not port it").

8. **`np.einsum('xkij,kji->x', ...)` transposes the DM index.**
   `krhf.py:63-66` contracts `h1ao[x,k,i,j]` against `dm0[k,j,i]` — the DM is
   indexed `ji`, not `ij`, on all three terms. `crates/pyscf-pbc-scf/src/types.rs:119`
   records that this port's `mo_coeff` is **column-major**; 14-05's
   `decompose_j2c` misread of exactly this kind was worth +6 306 866.73 Ha and
   was invisible to every gate then existing. Write the index order down in the
   Rust doc comment and gate it with a deliberately non-symmetric DM.

---

## 4. The plan set: fifteen plans, four waves

`§8.10`'s seven become fifteen. The additions are §1.1 (multigrid-v2 gradient
entry points), §1.6 (the strain AO kernel), §1.5 (`rks_stress` as its own plan,
ahead of the other three), a measure-first plan (17-01's and 16-01's precedent),
a substrate plan, and the FFTDF gradient-JK plan `§8.10` folded into 18-02.

| plan | content | blocked on |
|---|---|---|
| **18-01** | **MEASURE** — every gate floor in §2.3 (A–E), the memory sizings in `18-REVIEW`, and the D-PBC-30 clause-5 fusion question. Restates `ROADMAP:464` and `§8.10`'s 18-07 row. No production Rust. | — |
| **18-02** | Substrate: `pyscf-pbc-grad` crate surface, `Cell`-aware `verify_fd` + a strain-FD harness, `aoslice_by_atom` → 4-tuple (§1.8), the `mo_coeff`/`mo_occ` DM tag (clause 4), `_contract_vhf_dm` over the existing neighbour list (trap 4) | — |
| **18-03** | PP + Ewald gradient integrals: `vpploc_part2_nuc_grad`, `vppnl_nuc_grad` (`pp_int.py:300-407, 443-510`), `ewald_nuc_grad` (`ewald_methods.py:101-122, 256-292`); extends `cintx_moment_weighted_available.rs` to the five `_ip1_`/`_ip2` symbols (§1.3) | 18-02 |
| **18-04** | FFTDF gradient JK: `get_j_e1_kpts` (`fft_jk.py:113`), `get_k_e1_kpts` (`:310`), `FFTDF::{get_jk_e1,get_j_e1,get_k_e1}`; **named refusal on GDF/MDF/RSDF/AFTDF** (§1.2); clauses 3 and 4 | 18-02 |
| **18-05** | `grad/krhf` (418 l): `grad_elec`, `get_hcore`, `hcore_generator` (trap 1), `grad_nuc`, `GradientsBase`, `as_scanner`/`SCF_GradScanner` | 18-03, 18-04 |
| **18-06** | `grad/kuhf` (124 l) | 18-05 |
| **18-07** | `grad/krks` (141 l) + `grad/kuks` (135 l) | 18-06 |
| **18-08** | `grad/krkspu` (142 l) + `grad/kukspu` (83 l) — `generate_first_order_local_orbitals`, `_hubbard_U_deriv1` | 18-07 |
| **18-09** | **MultiGrid v2 gradient entry points** (§1.1): `get_veff_ip1`, `get_nuc_ip1`, `get_nuc_nuc_grad`, `get_vpploc_part1_ip1`, `vpploc_part1_nuc_grad` | 18-02 |
| **18-10** | `grad/rhf` (188 l) + `grad/uhf` (103 l) gamma + the `rks`/`uks` thin subclasses; Gate E | 18-09 |
| **18-11** | **Strain-tensor AO kernel** in `pyscf-kernels` (§1.6): `deriv = 0, 1`, sph + cart; clause 5 | 18-01 |
| **18-12** | `rks_stress` (462 l) — the base of the other three (§1.5); clauses 1 and 2 | 18-11 |
| **18-13** | `uks_stress` (246 l), `krks_stress` (404 l), `kuks_stress` (308 l) | 18-12 |
| **18-14** | `pbc/geomopt/geometric_solver` (246 l) over `pyscf-geomopt`'s BFGS+RFO; both entry points incl. the `optimizer(solver=)` refusal (§1.7). **No lattice DOF.** | 18-05 |
| **18-15** | Verification rollup: Gates A–E, `FEATURES`, `STATE.md`, `ROADMAP` | all |

**Waves.** Wave 0: 18-01, 18-02. Wave 1: 18-03, 18-04, 18-09, 18-11 (four
independent tracks). Wave 2: 18-05, 18-10, 18-12. Wave 3: 18-06 → 18-07 →
18-08, 18-13, 18-14. Wave 4: 18-15.

**The droppable half, if the phase overruns, is the stress tensor**
(18-11/18-12/18-13). Nothing in Phases 19–20 needs `dE/dε` for correctness, it
is the only half with no consumer inside the milestone, and it is ordered so
that dropping it costs nothing already built. Do **not** drop 18-09/18-10
instead: `grad/rks.py` and `grad/uks.py` are `pyscf.pbc.dft`'s advertised
`Gradients` attribute (`pbc/dft/rks.py:440`, `pbc/dft/uks.py:164`), so dropping
them leaves a broken import on the default surface.

**Deliberate non-ports:** an HF stress tensor (§1.5 — does not exist upstream);
lattice / variable-cell optimization (§1.7 — does not exist upstream); periodic
Hessians (`PBC-MASTER-PLAN §8.13` records that `pyscf/pbc` has no Hessian
module, so the missing cintx Hessian families block nothing here); and the ASE
`optimizer(solver='ase')` branch, which belongs to Phase 20's `tools/pyscf_ase`
(`§8.12`, 20-02) and is mirrored here only as the refusal that upstream's own
`'geometric'` argument produces.
