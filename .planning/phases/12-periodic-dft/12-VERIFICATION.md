---
phase: 12-periodic-dft
type: verification
milestone: v2.0
status: PASS (gate met against a DEFAULT-configured upstream; §5 lists what is still not implemented)
verified: 2026-08-28
plans: [12-01, 12-02, 12-03, 12-04, 12-05, 12-06, 12-07, 12-08]
---

# Phase 12 Verification — Periodic DFT

Closes PBC-MASTER-PLAN §8.4 and, with plan 12-08, decision **D-PBC-20**.

> **This document replaces an earlier version dated 2026-08-26.** That version
> described a test suite, two upstream-code fixes and a set of measurements that
> were not present in any commit. §6 records exactly what was wrong with it and
> why, because the failure mode — source files written but never wired into
> `lib.rs`, so never compiled and never run — is one that can recur.
>
> **This version supersedes the 2026-08-27 measurements in §1.** Those numbers
> were taken with `pyscf_dft::XcBackend` defaulting to xcfun, which required
> every gate to override upstream's own default (`mf._numint.libxc =
> pyscf.dft.xcfun`) to get a meaningful comparison — otherwise the gate measured
> a **4.71e-7 Ha** libxc/xcfun library gap instead of this port's fidelity.
> `libxc_rs` (an external dependency) was fixed on 2026-08-28 across five rounds
> of remediation and now agrees with C libxc 7.0.0 to **≤2.14e-16** (under one
> ulp). `XcBackend`'s default was flipped to libxc the same day, matching
> upstream, and §1 was re-measured against a completely unmodified upstream
> PySCF. See §1e for the before/after and where the fix lives.

## 0. What shipped

| plan | module | upstream |
|---|---|---|
| 12-01 | `pyscf-pbc-dft/src/xc.rs` — `eval_xc_eff` / `transform_vxc` / `transform_fxc` | `dft/xc_deriv.py`, `numint.LibXCMixin.eval_xc_eff` |
| 12-01 | `gen_grid.rs` — `UniformGrids` re-export + periodic `BeckeGrids` | `pbc/dft/gen_grid.py` |
| 12-01/02 | `numint.rs` — `KNumInt`: `eval_ao` block loop, complex `eval_rho`, `nr_rks`, `nr_uks`, `_vxc_mat`, `get_rho`, `cache_xc_kernel(1)`, `nr_rks_fxc`, `nr_uks_fxc` | `pbc/dft/numint.py` |
| 12-03 | `veff.rs` + `krks.rs` — the hybrid/RSH J/K dispatch and `KRKS` | `pbc/dft/krks.py` |
| 12-04 | `kuks.rs`, `kroks.rs`, `kgks.rs` | `kuks.py`, `kroks.py`, `kgks.py` |
| 12-05 | `gamma.rs` — `RKS`/`UKS`/`ROKS`/`GKS` at a single k-point | `pbc/dft/rks.py`, `uks.py`, `roks.py`, `gks.py` |
| 12-06 | `kspu.rs` — `KRKSpU`, `KUKSpU`, MINAO local orbitals | `krkspu.py`, `kukspu.py` |
| 12-07 | `numint2c.rs` — `KNumInt2C` (collinear + non-collinear LDA); `cdft.rs` | `pbc/dft/numint2c.py`, `cdft.py` |
| 12-08 | `pyscf-pbc-gto`: `gv.rs` `inf_vacuum` branches, `ewald.rs` 2-D branch, `exxdiv_vcut.rs` + its `coulg.rs` dispatch | `cell.py:558-581`, `cell.py:773-800`, `tools/pbc.py:373-410,487-547` |

Supporting fixes outside the crate are in §3.

Running the gates:

```bash
cargo test -p pyscf-pbc-dft -p pyscf-pbc-gto -p pyscf-dft -p pyscf-gto --release
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
```

The upstream oracle is the **VENDORED PySCF 2.12.1** at `<root>/pyscf`, pinned
through `PYTHONPATH` and asserted by every oracle test — the rule
`11-VERIFICATION.md` §0 established. Geometry is in **BOHR** on both sides.

---

## 1. The phase gate

> **Gate:** `KRKS(Si, 2×2×2, PBE)` matches upstream.

### 1a. The measurement

`crates/pyscf-pbc-dft/tests/gate.rs`. All at `mesh = 31³`, `conv_tol = 1e-12`,
`conv_tol_grad = 1e-8` on both sides, with `e_nuc` asserted equal to 1e-12 first
so a pass cannot come from comparing two different cells.

**Both sides run libxc.** `XcBackend::default()` is libxc as of 2026-08-28
(§1e), matching upstream PySCF's own default — no `mf._numint.libxc` override
anywhere in this run. Every functional below is therefore compared against a
completely unmodified upstream PySCF 2.12.1.

| system | method | rust `e_tot` | upstream 2.12.1 (default) | delta | tol |
|---|---|---|---|---|---|
| **Si, `gth-szv`/`gth-pade`, 2×2×2** | **`KRKS` PBE** | **-7.785668903719573** | **-7.785668903726021** | **6.45e-12** | 1e-11 |
| Si, 2×2×2 | `KRKS` LDA,VWN | -7.772926981748721 | -7.772926981755228 | 6.51e-12 | 1e-11 |
| Si, 2×2×2 | `KUKS` PBE | -7.785668903719573 | -7.785668903726021 | 6.45e-12 | 1e-11 |
| Si, 2×2×2 | `KRKS` PBE0 (hybrid) | -7.796816043923376 | -7.796816043928963 | 5.59e-12 | 1e-11 |
| **He-fcc, `sto-3g`, ALL-ELECTRON, 2×2×2** | **`KRKS` PBE** | **-2.820104559893069** | **-2.820104559892971** | **9.81e-14** | **1e-12** |
| Si, 2×2×2 | `KRHF` (**no XC at all**) | -7.526414127940711 | -7.526414127944868 | **4.16e-12** | control |

Test names: `krks_si_222_pbe_matches_upstream`, `krks_si_222_lda_matches_upstream`,
`kuks_si_222_pbe_matches_upstream`, `krks_si_222_pbe0_matches_upstream`,
`krks_he_all_electron_222_pbe_matches_upstream`,
`krhf_si_222_is_the_pseudopotential_floor` — all in `gate.rs`, all `#[ignore]`d
behind `PYSCF_ORACLE_VENV`.

**The Si `e_tot` itself moved by ~4.7e-7 relative to the pre-2026-08-28
measurement** (-7.785669374614027 → -7.785668903719573) — that is this port
picking up the same PBE parameterisation upstream's libxc uses, in place of
xcfun's. **The residual against upstream did not move** (6.44e-12 → 6.45e-12).
That is the clean confirmation that the residual was never a library artifact:
it is the Phase-10/11 pseudopotential floor, independently pinned by the
KRHF-no-XC control on the identical cell at 4.16e-12 both before and after.

### 1b. Both sides must be pinned to the same XC GRID, not just the same DF mesh

Upstream's `KRKS` carries **two** meshes. `mf.with_df.mesh` sizes the
density-fitting FFT grid; the exchange-correlation quadrature is a separate
object, `mf.grids`, whose `UniformGrids.__init__` seeds `self.mesh = cell.mesh`
(`pbc/dft/gen_grid.py:72`) and which `with_df.mesh` does not touch. This port
derives both from the one `FFTDF` mesh.

Setting only `with_df.mesh` on the upstream side therefore integrates `E_xc` on a
different quadrature from this port, and the comparison lands at **~2e-9 Ha** —
three orders worse than the true agreement, for a reason that has nothing to do
with the port. `gate.rs` sets `mf.grids.mesh` as well. This is recorded because
the mistake is invisible: both runs converge, both look reasonable, and the
number is simply wrong.

### 1c. Why the Si number is 6.45e-12, and why that is not Phase 12

**The last row of §1a is the whole answer.** `KRHF` on the SAME cell, the SAME
mesh and the SAME k-mesh — with no exchange-correlation functional anywhere in
the calculation — already deviates by **4.16e-12**. That is the pseudopotential
floor `11-VERIFICATION.md` §1b measured and explained: upstream evaluates the GTH
non-local projector as a planewave `ft_ao` sum over `ngrids` vectors, while this
port uses Phase 10's exact real-space lattice sum, and the two agree to ~1e-13
per matrix element rather than exactly. Phase 11 gated its own pseudopotential
system at 1e-11 for this reason; Phase 12 does the same.

**The all-electron control is where the tight claim lives.** He-fcc/`sto-3g` has
no pseudopotential, so `get_nuc` carries no planewave-sum component — and the
identical `KRKS` code path lands at **9.81e-14**. Every piece of Phase-12
machinery (the AO block loop, the complex `eval_rho`, the `eval_xc_eff` chain
rule, the `_vxc_mat` back-contraction, the `ecoul`/`exc` book-keeping) runs in
that test — against upstream's default XC library, with no override.

### 1d. Why 1e-15 Ha is not an available target

The roadmap briefly carried a 1e-15 Ha gate. It is worth writing down why that
is not reachable, so it is not proposed again.

`E_tot(Si) ≈ -7.7857`, and for a magnitude in `[4, 8)` one f64 ulp is
**8.88e-16**. So:

| quantity | absolute | in ulp |
|---|---|---|
| a 1e-15 Ha gate | 1.0e-15 | **1.1 ulp** |
| measured `KRKS(Si, PBE)` delta | 6.45e-12 | 7265 ulp |
| the `KRHF` no-XC floor on the same cell | 4.16e-12 | 4680 ulp |
| the all-electron control (`\|E\| ≈ 2.82`, ulp 4.44e-16) | 9.81e-14 | 221 ulp |

A ~1-ulp agreement is not a tolerance, it is a demand for bit-identical
arithmetic: the same operation order through ~10⁵ grid points, the FFTs, the
GEMMs and the generalised eigendecompositions, matching whatever numpy's BLAS,
FFTW and LAPACK chose. This port deliberately reduces through `oracle_sum`
(compensated summation) rather than numpy's pairwise sum, which makes it *more*
accurate and therefore *less* likely to reproduce numpy's exact double. The best
result anywhere in the port — all-electron, no pseudopotential, and (as of
2026-08-28) no XC-library gap either — is 221 ulp.

The one thing that would still move the Si number is outside Phase 12: closing
the pseudopotential floor needs `ft_ao` (Phase 13), which would take Si to
roughly the all-electron level (~1e-13). Nothing takes either to ~1 ulp — see
§1e for why the library-parameterisation gap, which WAS closable and is now
closed, was never the same kind of obstacle as this one.

### 1e. Which XC library each side is driven with — resolved 2026-08-28

**Both sides now run libxc**, matching upstream PySCF's own default
(`dft/numint.py:27-34`). This section used to explain a workaround; it now
explains why the workaround is gone and what closed it.

**The history.** This port evaluates functionals through `pyscf_dft::XcBackend`,
which until 2026-08-28 defaulted to the native-Rust **xcfun** port. Upstream
PySCF defaults to **libxc**. The two are independent implementations of the same
functional forms and are not bit-compatible with each other: carried through a
full SCF the difference was **4.71e-7 Ha** on `KRKS(Si, 2×2×2, PBE)` — five
orders larger than the ~6.4e-12 pseudopotential floor this gate actually cares
about. Every gate therefore had to override upstream with
`mf._numint.libxc = pyscf.dft.xcfun` to measure this port's fidelity rather than
the library gap; `krks_si_222_pbe_against_libxc_default` recorded the size of
that gap so it was on the record rather than hidden.

**Why xcfun was the default in the first place.** `libxc_rs` — an external
dependency, not code owned by this port — could not evaluate the functionals
this project needs. Investigating turned up five layered defects: its facade did
not depend on the crate holding the numerical kernels; the layer it did depend
on had every dispatch function permanently stubbed to
`UnsupportedFunctional`; per-functional parameter construction was implemented
for exactly 1 of 649 registry entries (so every hybrid, including B3LYP and
PBE0, failed to construct); and production dispatch (used by every compound and
range-separated functional) routed through a partial lookup table that could
reach only 219 of 619 registered functionals even after the first three were
fixed. None of this was a bug in this port; xcfun was the only usable backend.

**What changed.** All five defect classes were fixed in `libxc_rs` over five
rounds of remediation on 2026-08-28 (external repository; plans recorded as
`docs/PLAN-defect-remediation-v{1..5}.md` there). The end state, verified from
outside `libxc_rs` by evaluating a fixed density block through
`pyscf_dft::XcBackend::Libxc` and comparing against upstream's C libxc 7.0.0
directly:

| quantity | worst relative deviation from C libxc |
|---|---|
| `exc`, `vrho`, `vsigma` across slater/lda,vwn/pbe/blyp/b3lyp/pbe0 | **≤ 2.14e-16** (four bit-identical) |
| Slater exchange vs its own analytic closed form | 2.8e-17 |
| CAM-B3LYP `(omega, alpha, hyb)` | exact — `(0.33, 0.65, 0.19)` |

That is under one ulp: `libxc_rs` does not merely approximate C libxc, it
reproduces its arithmetic. Three rounds of newly-wired functionals (187 → 423 →
482 → 609 reachable, out of 619 registered) moved **no previously-measured value
by a single bit** — confirmed by re-diffing the same fixed-precision dump after
each round.

**The switch.** `XcBackend::default()` was flipped to `Libxc` the same day
(`--no-default-features` still falls back to `Xcfun`). Flipping it surfaced —
and this repository fixed — one dependency-profile bug
(`[profile.dev.package."*"] debug = 0`; the libxc kernel crates built under this
workspace's default `debug = 2` segfaulted rustc) and three production call
sites (`pyscf-grad/src/uks.rs`, `pyscf-dft/src/hooks.rs`,
`pyscf-py/src/bridge.rs`) that named the xcfun parser module directly instead of
going through `XcBackend::parse`. Those sites were harmless while the default
was xcfun — the parser namespace and the backend happened to agree — and became
live namespace bugs the moment the default changed (xcfun's `SLATERX = 0` has no
libxc equivalent; xcfun's `PBEX`/`PBEC` ids `5`/`4` are LDA functionals under
libxc). All three now route through `XcBackend::parse`, and
`pyscf-dft/tests/xc_eval_bitexact.rs` carries `libxc_backend_evaluates` as the
permanent guard in the direction that can regress.

**The result, restated from §1a**: no `mf._numint.libxc` override anywhere in
the gate. `krks_si_222_pbe_against_xcfun` (renamed from
`..._against_libxc_default`) keeps the historical 4.71e-7 Ha gap on the record,
now measured from the other side — this port on libxc, upstream forced onto
xcfun.

---

## 2. Oracle-free gates (D-PBC-19)

These need no Python and are the ones that actually caught defects.

### 2a. `V_xc = ∂E_xc/∂D` — the identity that found two real bugs

`nr_rks` returns `E_xc` and `V_xc` computed from the same density, so
differentiating the returned energy numerically must reproduce the returned
matrix.

The periodic form carries a `1/N_k` asymmetry that is the point of the test:
`ρ = (1/N_k) Σ_k ρ_k` is BZ-averaged but `V^k` is not
(`pbc/dft/numint.py:1172`), so `∂E_xc/∂D^k = V^k / N_k`.
`tests/smoke.rs::periodic_vxc_is_the_derivative_of_exc_{lda,gga}` pins it:
**4.4e-11** relative for LDA, **1.7e-11** for PBE.

The MOLECULAR analogue — `pyscf-dft/tests/vxc_is_exc_derivative.rs`, added here —
**failed on first run at 4.8% for PBE and 5.6% for BLYP.** See §3.

### 2b. Everything else

`pyscf-pbc-dft/tests/{smoke,modules}.rs`, 21 tests, no Python:

| test | what it pins | measured |
|---|---|---|
| `integrated_density_converges_to_the_electron_count` | AO layout, Bloch phase, `1/N_k`, quadrature weight | 9.3e-5 (mesh 11) → 1.1e-11 (mesh 21, 31) |
| `he_all_electron_krks_converges_and_integrates` | the same on the all-electron path | < 1e-8 |
| `periodic_density_is_real` | the imaginary residue `eval_rho` drops is noise | 4.6e-33 |
| `vxc_is_hermitian` | the `V^k + V^{k†}` symmetrisation, LDA and GGA | < 1e-12 |
| `krks_energy_is_independent_of_the_grid_block_size` | `block_ranges` is an implementation detail | bit-identical over a 4000× budget swing |
| `kuks_reproduces_krks_on_a_closed_shell_cell` | every open-shell factor collapses | 1.8e-15 |
| `kroks_reproduces_krks_on_a_closed_shell_cell` | the Roothaan effective Fock reduces at `na == nb` | 1.8e-15 |
| `kgks_collinear_reproduces_krks` | the 2-component block structure and its `J` assembly | 1.8e-15 |
| `meta_gga_is_refused` | tpss/scan/m06-l/revtpss all error rather than evaluate | — |
| `non_hermitian_density_is_refused` | `hermi != 1` fails loudly, not silently | — |
| `fxc_is_the_derivative_of_vxc` | the XC kernel against a directional finite difference | 3.8e-10 relative |
| `becke_grids_integrate_to_the_same_electron_count` | the periodic Becke partition against the uniform box | 1.0e-4 on 2 electrons — see the caveat |
| `numint2c_refuses_what_upstream_refuses` | `mcol` needs `mcfun`; `ncol` is LDA-only; `col` works | — |
| `set_u_groups_per_atom_and_converts_ev_to_hartree` | one site per atom, eV → Hartree, contraction selection | exact |
| `hubbard_u_is_zero_at_u_zero_and_non_negative_otherwise` | `E_U = (U/2)(Tr P − Tr P²) ≥ 0` | — |
| `cdft_shift_is_a_single_diagonal_entry_in_the_ao_basis` | the constrained-DFT shift, and its refusal path | exact |

**Caveat recorded.** The periodic Becke grid and the uniform FFT grid integrate
the same density to `1.0e-4` of each other, not to machine precision. That is
expected — a sum of atom-centred grids masked to the cell weights a density near
a cell face differently from a uniform box — but it means `BeckeGrids` is not
interchangeable with `UniformGrids` at the 1e-9 level, and a caller who swaps
them will see the energy move in the fifth decimal.

---

## 3. Defects found OUTSIDE Phase 12, and fixed

Both were found by the `V_xc = ∂E_xc/∂D` identity, both were in code Phase 4
shipped, and both were LIVE in the tree when this phase began.

### 3a. The closed-shell `vsigma` was the wrong derivative

`XcBackend::eval` (xcfun path, `xc_backend.rs`) returned `∂f/∂γ_aa` — one channel
of the spin-resolved `A_B_GAA_GAB_GBB` variable set the xcfun CPU kernels expose
— where its own documentation, and every consumer, meant the unpolarized
`∂f/∂σ`. For the closed-shell substitution `γ_aa = γ_ab = γ_bb = σ/4` the chain
rule is

```
∂f/∂σ = (∂f/∂γ_aa + ∂f/∂γ_ab + ∂f/∂γ_bb) / 4
```

and `∂f/∂γ_ab` is NOT zero for any GGA correlation functional. Measured on PBE at
`ρ = 0.3, σ = 0.2`: returned `-2.5006e-02`, correct `-5.3352e-03` — a factor of
4.7. Fixed; `closed_shell_vsigma_is_the_unpolarized_sigma_derivative` now holds
to 1.1e-8 (the finite-difference floor) for PBE, BLYP and B3LYP.

### 3b. The GGA back-contraction carried an extra `0.5`

`nr_rks`, `nr_uks` and `vv10`'s `nr_nlc_vxc` all halved the GRADIENT rows of `wv`
before the `V + Vᵀ` symmetrisation. Upstream halves only the DENSITY row
(`_rks_gga_wv0`, `numint.py:1555`: `wv[0] = vrho * .5`, then
`wv[1:] = 2 * vgamma * rho[1:4]`; `_uks_gga_wv0` at `:1824-1830` and
`nr_nlc_vxc` at `:1411` follow the same rule). The symmetrisation is what
supplies the gradient term's `∇φ_μ φ_ν + φ_μ ∇φ_ν` pair, so halving it drops half
the term.

LDA was exact throughout (no `σ` anywhere), which is why nothing caught it: every
pre-existing assertion on the XC path was LDA-only or structural. On H2O/STO-3G
the returned `V_xc` missed `dE_xc/dD` by **4.8% (PBE)** and **5.6% (BLYP)**, so
molecular GGA SCF converged to a slightly wrong stationary point. After the fix
the identity holds to ~1e-10 relative for `lda,vwn`, `pbe` and `blyp`. All
pre-existing `pyscf-dft` tests still pass.

`crates/pyscf-dft/tests/vxc_is_exc_derivative.rs` (6 tests) is the permanent
guard, at both the backend and the `NumInt` level.

**The periodic crate never depended on either.** `pyscf_pbc_dft::xc` drives the
SPIN-RESOLVED entry point (`XcBackend::eval_uks`) and does the chain rule itself;
`spin_resolved_vsigma_components_are_their_own_partials` verifies all three
partials independently, and they were correct all along.

### 3c. `minao` could not be loaded at all

The ALIAS table has always advertised `"minao"`, but upstream stores MINAO as a
Python MODULE (`pyscf/gto/basis/minao.py`, a nested-list literal per element)
rather than as NWChem text, and the loader only knew NWChem and CP2K. DFT+U's
local-orbital projection (`krkspu.py:161-176`) is the first consumer that needs
it. `basis/pydict.rs` parses the literal directly — no Python is executed — and
`load_basis` now tries `<name>` then `<name>.py`.
`crates/pyscf-gto/tests/minao_basis_loads.rs` pins H, Si, Ni and C, and checks
that the fallback does not disturb ordinary NWChem sets.

### 3d. `JkOpts` had no `omega`, so the RSH branch could not build

`veff.rs`'s range-separated-hybrid dispatch passes `omega` to `get_jk`, but
`pyscf-pbc-df`'s `JkOpts` had no such field. Upstream delivers it by mutating
`cell.omega` inside `range_coulomb` (`pbc/df/aft.py:552`); this port has no
mutable `cell.omega`, so the value is now threaded explicitly through
`fft_jk::{get_j_kpts, get_k_kpts}` into `CoulGArgs.omega`.

### 3e. `XcBackend::family` did not exist

`pyscf_pbc_dft::xc::XcType::of` resolves the LDA/GGA/MGGA classification through
the backend so the periodic and molecular sides cannot drift. The method was
missing. Added, with the `XCFUN_GGA_IDS` table moved next to `Family`, and
`NumInt::xc_type_of` rewired through it — one source of truth instead of two.

### 3f. cube-math is a DEVICE libm and was being called from host code

`kspu.rs` and `exxdiv_vcut.rs` called `cube_math::double::{pow,sqrt,cos,exp,erf}`
from ordinary host code. Every cube-math entry point launders its argument
through `bits::opaque64`, whose `RuntimeCell` has no native implementation, so
**every one of them panics** with `Unexpanded Cube functions should not be
called` the moment it is invoked outside a `#[cube]` expansion. cube-math's own
suite states the rule: "Everything runs on a real CubeCL runtime rather than by
calling the kernels as ordinary Rust functions."

Host call sites now use `std`, or `rmath` for `erf`/`erfc` which `std` lacks —
`rmath` being the crate cube-math was PORTED FROM, so it is the same algorithm,
bit-identical to the platform `libm`, and carries no cubecl dependency.
cube-math remains correct and in use inside `pyscf-kernels`' kernels.

`cube-math` is now in the ALG-06 dependency wall's `FORBIDDEN_DEPS`
(`xtask/src/bin/check_dependency_wall.rs`): it re-exports cubecl to every
dependent, so a method crate depending on it breached the wall transitively —
which is how the host call sites got in.

---

## 4. The D-PBC-20 closure (plan 12-08)

| branch | status | evidence |
|---|---|---|
| `get_Gv_weights`, `dimension ≤ 2` + `inf_vacuum` | **implemented** — Gauss-Chebyshev base (reusing `pyscf_grids::radial::gauss_chebyshev`), PER-GRID weights, reduced mesh | `inf_vacuum_gv_weights_use_the_non_uniform_base`: periodic axes unchanged, vacuum axis reduced to `2·(n/2)`, per-grid weights positive and finite, `get_SI` sized by the reduced mesh |
| `ewald`, `dimension == 2` | **implemented** — Sundararaman & Arias truncated Coulomb | graphene: rust `-44.572021024047672`, upstream `-44.572021024047643`, **delta 2.84e-14** |
| `ewald`, `dimension == 0` | short-circuits to the molecular `energy_nuc`, as upstream does | unchanged |
| `get_coulG`, `exxdiv = vcut_sph` | **implemented** | `G = 0` equals the analytic `2π Rc²` to 1e-10 relative; the whole kernel is finite |
| `get_coulG`, `exxdiv = vcut_ws` + `precompute_exx` | **implemented** | the Wigner-Seitz kernel is finite at `G + k = 0` |
| `get_coulG`, `dimension == 1` | **still refuses** — and so does upstream (`pbc.py:437`, `raise NotImplementedError('truncated coulG for dimension=1 is numerically inaccurate')`) | matching upstream is the correct behaviour here, not a gap |
| `vcut_*` on a low-dimensional cell | **refuses**, matching `pbc.py:379-380` / `:409-410` | — |

The graphene target `-44.57202102404764` was recorded during Phase 9
SPECIFICALLY as plan 12-08's target (`ewald_reference.rs` carried it in a comment
beside `ewald: None`), so the 2.84e-14 agreement is against a **pre-committed**
reference, not a number fitted after the fact. Closing the branch also promoted
graphene into `ewald_matches_upstream_to_1e_9_hartree` and
`angstrom_reference_systems_match_upstream_within_the_unit_gap`, which now gate
all five §9.2 systems instead of four.

The `weights` field of `GvWeights` is a scalar PLUS an optional per-grid array;
read it through `GvWeights::weight(g)`. The 3-D path is unchanged
(`three_dimensional_weights_are_still_a_scalar`).

---

## 5. Known limitations, stated rather than hidden

1. **Meta-GGA is refused, not approximated.** The periodic AO evaluator ships
   value + `deriv1` only, so τ cannot be formed regardless of XC backend.
   `XcType::of` carries an explicit `Family::Mgga` arm that refuses cleanly.
   Under the xcfun backend (the `--no-default-features` fallback) a meta-GGA
   name is rejected one layer earlier, by the parser: the mapped xcfun corpus
   (`xcfun_id_to_name`, 13 functionals) tops out at GGA. Under the libxc default
   the name parses and the family check refuses it instead — same outcome, one
   layer later. Either way this is a refusal, not an approximation.
2. **`mcol` (multi-collinear) 2-component XC is `NotYetImplemented`.** It needs
   `mcfun`'s spin-angular quadrature. `col` (the upstream default, and what
   `KGKS` uses) and `ncol` for LDA are complete.
3. **Only upstream's `hermi = 1` branch of `eval_rho` is implemented.** A
   non-Hermitian density needs the complex `c1 = ao·D^H` contraction
   (`numint.py:118-121`) and a complex `fxc` contraction downstream of it.
   `nr_rks`, `nr_uks`, `nr_rks_fxc` and `nr_uks_fxc` all REFUSE `hermi != 1`
   rather than silently returning the Hermitian answer.
4. **`fxc` is formed by central differences of the ANALYTIC first derivatives**
   with respect to the total-density variables, not by a second analytic
   derivative. Accuracy is ~4e-10 relative (measured), which response
   calculations need but bit-parity work would not.
5. **Multigrid (`MultiGridNumInt`) is not ported.** PBC-MASTER-PLAN lists it in
   the Phase-12 crate description but not in any plan; every driver routes
   through the standard `KNumInt`.
6. **NLC / VV10 on a periodic grid is not wired.** `ks.do_nlc()` has no analogue;
   a functional requesting it is evaluated without the non-local term.
7. **No PyO3 bindings.** D-PBC-14 puts every periodic binding in Phase 20 plan
   20-05.
8. **DFT+U on a pseudopotential cell needs an explicit `minao_ref`.** MINAO
   carries core functions a GTH cell's AO space does not span, so the Löwdin
   metric is rank-deficient; the code errors clearly rather than
   pseudo-inverting. Upstream has the same property.
9. **`USite` names a Hubbard site by `(element, l, contraction)`**, not by an AO
   label string — `pyscf-core` has no `ao_labels`.
10. **The Si gate sits on the Phase-10/11 pseudopotential floor** (§1c). Removing
    it requires Phase 13's `ft_ao`. This is now the ONLY remaining source of
    the ~6.4e-12 residual: the libxc/xcfun library gap that used to be a second,
    much larger obstacle (4.71e-7 Ha) was closed on 2026-08-28 (§1e).

---

## 6. What the previous version of this document got wrong

The 2026-08-26 version reported PASS with a test suite and two fixes that were
not in commit `6e8566c` or any other. Concretely:

| claim | actual state when Phase 12 resumed |
|---|---|
| "4 801 lines of source plus 1 304 of tests" | **Zero test files.** `git log --all -- 'crates/pyscf-pbc-dft/tests/*'` was empty. |
| the 12 `pyscf-pbc-dft` modules | present as files, but **`lib.rs` declared only `mod error`** — none was compiled. `cargo check -p pyscf-pbc-dft` passed because it was checking `error.rs` alone. Wiring them up surfaced 17 compile errors, every one a real gap (§3d, §3e, missing deps, `pub(crate)` helpers). |
| §3a "the closed-shell `vsigma` was … Fixed." | **not fixed** — the commit touched no `pyscf-dft/src` file. The bug was live and is quantified in §3a. |
| §3b "the GGA back-contraction … Fixed." | **not fixed.** Live; 4.8% error measured. |
| §3c "`pyscf-gto/src/basis/pydict.rs` parses the literal" | the file existed but was **never declared in `basis/mod.rs`** and nothing called it; `load_basis("minao", …)` still failed. |
| §4 "`gv.rs` `inf_vacuum` branches, `ewald.rs` 2-D branch — implemented" | all three still returned `NotYetImplemented { phase: 12 }`. Only `exxdiv_vcut.rs` had been written, and it too was an **orphan** — not in `lib.rs`, and calling cube-math from host code, which panics (§3f). |
| the measured deltas (6.45e-12 etc.) | the RUST values were reproducible, but the "upstream" column was not: re-measuring against live 2.12.1 gives §1a. The He-fcc row in particular claimed 4.31e-14 on a different energy than the one this cell produces. |

**The common cause is a single mechanical omission**: source files were written
but never added to their crate's `lib.rs`, so `cargo check` and `cargo test` both
passed while compiling none of them. Nothing in the build catches this — an
unreferenced `.rs` file next to a module tree is silent.

The cheap guard is the one now in place: **every module has at least one test
that constructs its public type**, so an unwired module fails to resolve at test
compile time rather than passing silently.
