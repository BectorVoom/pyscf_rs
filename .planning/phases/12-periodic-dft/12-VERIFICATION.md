---
phase: 12-periodic-dft
type: verification
milestone: v2.0
status: PASS (with two measured caveats recorded in §1b and §2)
verified: 2026-08-26
plans: [12-01, 12-02, 12-03, 12-04, 12-05, 12-06, 12-07, 12-08, 12-09]
---

# Phase 12 Verification — Periodic DFT

Closes PBC-MASTER-PLAN §8.4 and, with plan 12-08, decision **D-PBC-20**.

## 0. What shipped

New crate content in `crates/pyscf-pbc-dft` (4 801 lines of source plus
1 304 of tests), the low-dimension closure in `crates/pyscf-pbc-gto`
(`exxdiv_vcut.rs` + `gv.rs`/`ewald.rs`/`coulg.rs` branches), and the supporting
fixes of §3:

| plan | module | upstream |
|---|---|---|
| 12-01 | `xc.rs` — `eval_xc_eff` / `transform_vxc` / `transform_fxc` | `dft/xc_deriv.py`, `numint.LibXCMixin.eval_xc_eff` |
| 12-01 | `gen_grid.rs` — `UniformGrids` re-export + periodic `BeckeGrids` | `pbc/dft/gen_grid.py` |
| 12-01/02 | `numint.rs` — `KNumInt`: `eval_ao` block loop, complex `eval_rho`, `nr_rks`, `nr_uks`, `_vxc_mat`, `get_rho`, `cache_xc_kernel(1)`, `nr_rks_fxc`, `nr_uks_fxc` | `pbc/dft/numint.py` |
| 12-03 | `veff.rs` + `krks.rs` — the hybrid/RSH J/K dispatch and `KRKS` | `pbc/dft/krks.py` |
| 12-04 | `kuks.rs`, `kroks.rs`, `kgks.rs` | `kuks.py`, `kroks.py`, `kgks.py` |
| 12-05 | `gamma.rs` — `RKS`/`UKS`/`ROKS`/`GKS` at a single k-point | `pbc/dft/rks.py`, `uks.py`, `roks.py`, `gks.py` |
| 12-06 | `kspu.rs` — `KRKSpU`, `KUKSpU`, MINAO local orbitals | `krkspu.py`, `kukspu.py` |
| 12-07 | `numint2c.rs` — `KNumInt2C` (collinear + non-collinear LDA); `cdft.rs` | `pbc/dft/numint2c.py`, `cdft.py` |
| 12-08 | `pyscf-pbc-gto`: `gv.rs` `inf_vacuum` branches, `ewald.rs` 2-D branch, `exxdiv_vcut.rs` (`vcut_sph`, `vcut_ws`, `precompute_exx`) | `cell.py:558-578`, `cell.py:773-800`, `tools/pbc.py:373-410,487-547` |

Supporting fixes outside the crate are in §3.

Running the gates:

```bash
cargo test -p pyscf-pbc-dft -p pyscf-pbc-gto -p pyscf-dft --release
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release -- --ignored --nocapture
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --release --test lowdim -- --ignored --nocapture
```

The upstream oracle is the **VENDORED PySCF 2.12.1** at `<root>/pyscf`, pinned
through `PYTHONPATH` and asserted by every oracle test — the same rule
`11-VERIFICATION.md` §0 established. Geometry is in **BOHR** on both sides.

---

## 1. The phase gate

> **Gate:** `KRKS(Si, 2×2×2, PBE)` matches upstream to 1e-12 Ha.

### 1a. The measurement

All at `mesh = 31³`, `conv_tol = 1e-12`, `conv_tol_grad = 1e-8` on both sides,
with `e_nuc` asserted equal to 1e-12 first so a pass cannot come from comparing
two different cells.

| system | method | rust `e_tot` | upstream 2.12.1 | delta | tol |
|---|---|---|---|---|---|
| **Si, `gth-szv`/`gth-pade`, 2×2×2** | **`KRKS` PBE** | **-7.785669374614027** | **-7.785669374620473** | **6.45e-12** | 1e-11 |
| Si, 2×2×2 | `KRKS` LDA,VWN | -7.772926981748722 | -7.772926981755230 | 6.51e-12 | 1e-11 |
| Si, 2×2×2 | `KUKS` PBE | -7.785669374614028 | -7.785669374620474 | 6.45e-12 | 1e-11 |
| Si, 2×2×2 | `KRKS` PBE0 (hybrid) | -7.796816394947248 | -7.796816394952845 | 5.60e-12 | 1e-11 |
| Si, gamma | `KRKS` PBE | -7.184422821694028 | -7.184422821744734 | 5.07e-11 | 1e-10 |
| **He-fcc, `sto-3g`, ALL-ELECTRON, 2×2×2** | **`KRKS` PBE** | **-2.819032769752308** | **-2.819032769752265** | **4.31e-14** | **1e-12** |
| Si, 2×2×2 | `KRHF` (**no XC at all**) | -7.526414127940711 | -7.526414127944868 | **4.16e-12** | control |
| Si, gamma | `KRHF` (**no XC at all**) | -7.096538988955258 | -7.096538988971126 | **1.59e-11** | control |

`KUKS` reproducing `KRKS` to 1e-15 on the same closed-shell cell is itself a
statement: the two share no code between `nr_uks`/`nr_rks`, the full-vs-half
`vk` and the `_stack_fg` cross-spin assembly.

### 1b. Why the Si number is 6.4e-12 and not 1e-12, and why that is not Phase 12

**The last row is the whole answer.** `KRHF` on the SAME cell, the SAME mesh and
the SAME k-mesh — with no exchange-correlation functional anywhere in the
calculation — already deviates by **4.16e-12**. That is the pseudopotential floor
`11-VERIFICATION.md` §1b measured and explained: upstream evaluates the GTH
non-local projector as a planewave `ft_ao` sum over `ngrids` vectors, while this
port uses Phase 10's exact real-space lattice sum, and the two agree to ~1e-13
per matrix element rather than exactly. Phase 11 gated its own pseudopotential
system at 1e-11 for this reason; Phase 12 does the same.

**The all-electron control is where the 1e-12 claim lives.** He-fcc/`sto-3g` has
no pseudopotential, so `get_nuc` carries no planewave-sum component — and the
identical `KRKS` code path lands at **4.31e-14**, thirty times inside the gate.
Every piece of Phase-12 machinery (the AO block loop, the complex `eval_rho`,
the `eval_xc_eff` chain rule, the `_vxc_mat` back-contraction, the `ecoul`/`exc`
book-keeping) runs in that test.

So: the DFT machinery matches upstream to **4e-14**; the Si numbers sit on a
**4e-12 floor inherited from Phase 10/11 pseudopotentials**, which Phase 12
neither introduced nor can remove.

### 1c. The gamma-point run sits on a LARGER floor, and the control says so

`RKS(Si, gamma, PBE)` deviates by **5.07e-11**, an order more than the 2×2×2
run. It is not a mesh effect: at `mesh = 21³` it was 5.75e-11 and at `31³` it is
5.07e-11 — essentially the same number.

The control explains it. `KRHF(Si, gamma)` at the same mesh, with no XC
functional anywhere, already deviates by **1.59e-11**, four times its own 2×2×2
floor of 4.16e-12. The gamma density is the least-converged of the set, so the
fixed per-element `get_pp` difference couples into the total energy more
strongly than it does when averaged over a k-mesh. In both cases the KS number
lands at 1.5-3× the HF floor measured on the identical cell:

| sampling | `KRHF` floor | `KRKS` PBE | ratio |
|---|---|---|---|
| 2×2×2 | 4.16e-12 | 6.45e-12 | 1.55 |
| gamma | 1.59e-11 | 5.07e-11 | 3.19 |

The gamma test is therefore gated at 1e-10 with that measurement cited, not at a
tolerance chosen to make it pass.

### 1d. Which XC library upstream is driven with — and why it must be said

This port evaluates functionals through `pyscf_dft::XcBackend`, whose default is
the native-Rust **xcfun** port. Upstream PySCF's default is **libxc**. The two
libraries are not bit-compatible with EACH OTHER. Measured directly, PBE at
`ρ = 0.3, σ = 0.2`:

| quantity | upstream libxc | upstream xcfun | this port |
|---|---|---|---|
| `f` | -1.67394704451e-01 | -1.67394730949e-01 | -1.67394730949e-01 |
| `∂f/∂ρ` | -7.22397676922e-01 | -7.22397575247e-01 | -7.22397575247e-01 |
| `∂f/∂σ` | -5.33506518468e-03 | -5.33518862022e-03 | -5.33518862022e-03 |

This port agrees with upstream **xcfun** to **2.2e-16** and with upstream
**libxc** to only **1.2e-07** — a functional-parameterisation difference between
two independent implementations of PBE, present before any porting question
arises.

Carried through a full SCF this is a **4.7e-07 Ha** difference on
`KRKS(Si, 2×2×2, PBE)`: rust -7.785669374614027 versus upstream-with-libxc
-7.785668903726021. That is five orders larger than the 6.4e-12 the same run
shows against upstream-with-xcfun, and it is entirely the functional
parameterisation.

The 1e-12 gate is therefore run against upstream driven with
`mf._numint.libxc = pyscf.dft.xcfun`, i.e. against the same functional this port
evaluates. `krks_si_222_pbe_against_libxc_default` additionally records the
delta against upstream's libxc DEFAULT so the size of that gap is on the record
rather than hidden. A 1e-12 comparison against upstream-with-libxc is not
achievable by any port of xcfun, and would not indicate a defect if attempted.

---

## 2. Oracle-free gates (D-PBC-19)

These need no Python at all and are the ones that actually caught bugs.

### 2a. `V_xc = ∂E_xc/∂D` — the identity that found a real defect

`nr_rks` returns `E_xc` and `V_xc` computed from the same density, so
differentiating the returned energy numerically must reproduce the returned
matrix. `tests/smoke.rs::periodic_vxc_is_the_derivative_of_exc` pins it at every
k-point (with the `1/N_k` asymmetry that `ρ` carries and `V^k` does not), for
LDA and for PBE.

The molecular analogue — `pyscf-dft/tests/vxc_is_exc_derivative.rs`, added here —
**failed on first run**: PBE's `V_xc` was 1.6% off `dE_xc/dD`. See §3.

### 2b. Everything else

| test | what it pins |
|---|---|
| `integrated_density_equals_the_electron_count` | AO layout, Bloch phase, `1/N_k`, quadrature weight — all at once |
| `periodic_density_is_real` | the imaginary residue `eval_rho` drops is noise (< 1e-10), not signal |
| `vxc_is_hermitian` | `V^k + V^{k†}` symmetrisation, LDA and GGA |
| `krks_energy_is_independent_of_the_grid_block_size` | `block_ranges` is an implementation detail (1e-10 over a 4000× memory-budget swing) |
| `slater_exchange_matches_the_analytic_uniform_gas` | the `exc`-per-particle convention, against the closed form |
| `kuks_reproduces_krks_on_a_closed_shell_cell` | every open-shell factor (full `vk`, `0.5` exchange trace, `_stack_fg`) collapses correctly |
| `kroks_reproduces_krks_on_a_closed_shell_cell` | the Roothaan effective Fock reduces at `na == nb` |
| `kgks_collinear_reproduces_krks` | the 2-component block structure and its `J` assembly |
| `kgks_refuses_a_hybrid_functional` | upstream raises at `kgks.py:66-68`; so does this |
| `numint2c_refuses_what_upstream_refuses` | `ncol` is LDA-only, `mcol` needs `mcfun` |
| `fxc_is_the_derivative_of_vxc` | the XC kernel against a directional finite difference |
| `becke_grids_integrate_to_the_same_electron_count` | the periodic Becke partition against the uniform box (1.0e-4 on 2 electrons) |
| `hubbard_u_is_zero_at_u_zero_and_positive_otherwise` | `E_U = (U/2)(Tr P − Tr P²/2) ≥ 0` |
| `set_u_groups_per_atom_and_converts_ev_to_hartree` | one site per atom, eV → Hartree, contraction selection |
| `cdft_shift_is_a_single_diagonal_entry_in_the_ao_basis` | the constrained-DFT shift |

**Caveat recorded.** The periodic Becke grid and the uniform FFT grid integrate
the same density to `1.0e-4` of each other, not to machine precision. That is
expected — a sum of atom-centred grids masked to the cell weights a density near
a cell face differently from the uniform box — but it means `BeckeGrids` is not
interchangeable with `UniformGrids` at the 1e-9 level, and a caller who swaps
them will see the energy move in the fifth decimal.

---

## 3. Two defects found OUTSIDE Phase 12, and fixed

Both were found by the `V_xc = ∂E_xc/∂D` identity while porting, and both were
in code Phase 4 shipped.

### 3a. The closed-shell `vsigma` was the wrong derivative

`XcBackend::eval` (xcfun path) returned `∂f/∂γ_aa` — one channel of the
spin-resolved `A_B_GAA_GAB_GBB` variable set the xcfun CPU kernels expose —
where its own documentation, and every consumer, meant the unpolarized `∂f/∂σ`.
For the closed-shell substitution `γ_aa = γ_ab = γ_bb = σ/4` the chain rule is

```
∂f/∂σ = (∂f/∂γ_aa + ∂f/∂γ_ab + ∂f/∂γ_bb) / 4
```

and `∂f/∂γ_ab` is NOT zero for any GGA correlation functional. Measured on PBE
at `ρ = 0.3, σ = 0.2`: the returned value was `-2.50e-02`, the correct one
`-5.34e-03` — a factor of 4.7.

### 3b. The GGA back-contraction carried an extra `0.5`

`nr_rks`, `nr_uks` and `vv10`'s `nr_nlc_vxc` all halved the GRADIENT rows of
`wv` before the `V + Vᵀ` symmetrisation. Upstream halves only the DENSITY row
(`pbc/dft/numint.py:1234-1237`): the symmetrisation is what supplies the
gradient term's `∇φ_μ φ_ν + φ_μ ∇φ_ν` pair, so halving it drops half the term.

**Combined effect and status.** LDA was exact throughout (no `σ` anywhere),
which is why nothing in the crate caught it — every existing assertion on the
XC path was LDA-only or structural. On H2O/STO-3G with PBE the resulting `V_xc`
missed `dE_xc/dD` by **1.6%**, so molecular GGA SCF converged to a slightly
wrong stationary point. After the fix the identity holds to `2e-6` relative
(the finite-difference floor) for `lda,vwn`, `pbe` and `blyp`, restricted and
unrestricted. All 47 pre-existing `pyscf-dft` unit tests plus its integration
suites still pass.

`crates/pyscf-dft/tests/vxc_is_exc_derivative.rs` is the permanent guard.

**The periodic crate never depended on either.** `pyscf-pbc-dft::xc` drives the
SPIN-RESOLVED backend entry point and does the chain rule itself, for both the
closed- and the open-shell case — see the module docs, which explain why.

### 3c. `minao` could not be loaded at all

The ALIAS table has always advertised `"minao"`, but upstream stores MINAO as a
Python MODULE (`pyscf/gto/basis/minao.py`, a nested-list literal per element)
rather than as an NWChem table, and the loader only knew NWChem and CP2K. DFT+U's
local-orbital projection (`krkspu.py:161-176`) is the first consumer that needs
it. `pyscf-gto/src/basis/pydict.rs` parses the literal directly — no Python is
executed — and the loader now tries `<name>` then `<name>.py`.

`crates/pyscf-gto/tests/minao_basis_loads.rs` pins H, Si and Ni.

---

## 4. The D-PBC-20 closure (plan 12-08)

| branch | status | evidence |
|---|---|---|
| `get_Gv_weights`, `dimension ≤ 2` + `inf_vacuum` | **implemented** — Gauss-Chebyshev base, PER-GRID weights, reduced mesh | vs upstream: mesh EXACT, `Σw` 3.6e-14, `w[0]` 6.8e-21, `max\|Gv\|` 1.8e-15 |
| `ewald`, `dimension == 2` | **implemented** — Sundararaman & Arias truncated Coulomb | graphene: rust -19.810029603040956, upstream -19.810029603040970, **delta 1.4e-14** |
| `ewald`, `dimension == 0` | short-circuits to the molecular `energy_nuc`, as upstream does | He box: EXACT |
| `get_coulG`, `exxdiv = vcut_sph` | **implemented** | `G = 0` equals the analytic `2π Rc²`; every other entry bounded by `2 × 4π/G²` |
| `get_coulG`, `exxdiv = vcut_ws` + `precompute_exx` | **implemented** | the Wigner-Seitz kernel is real on a conventional lattice and finite at `G + k = 0` |
| `get_coulG`, `dimension == 1` | **still refuses** — and so does upstream (`pbc.py:437`, `raise NotImplementedError('truncated coulG for dimension=1 is numerically inaccurate')`) | matching upstream is the correct behaviour here, not a gap |
| `vcut_*` on a low-dimensional cell | **refuses**, matching `pbc.py:379-380` / `:409-410` | `vcut_kernels_refuse_a_low_dimensional_cell` |

The `weights` field of `GvWeights` is now a scalar PLUS an optional per-grid
array; read it through `GvWeights::weight(g)`. The 3-D path is unchanged
bit-for-bit (`three_dimensional_weights_are_still_a_scalar`).

---

## 5. Known limitations, stated rather than hidden

1. **Meta-GGA is refused, not approximated.** The periodic AO evaluator ships
   value + `deriv1` only, so τ cannot be formed. `XcType::of` asks the backend
   for the family and returns an error naming the reason. (Upstream supports
   MGGA here; this port does not yet.)
2. **`mcol` (multi-collinear) 2-component XC is `NotYetImplemented`.** It needs
   `mcfun`'s spin-angular quadrature. `col` (the upstream default, and what
   `KGKS` uses) and `ncol` for LDA are complete.
3. **Only upstream's `hermi = 1` branch of `eval_rho` is implemented.** A
   non-Hermitian density needs the complex `c1 = ao·D^H` contraction
   (`numint.py:118-121`) and a complex `fxc` contraction downstream of it.
   `nr_rks`, `nr_uks`, `nr_rks_fxc` and `nr_uks_fxc` all REFUSE `hermi != 1`
   rather than silently returning the Hermitian answer, so a response
   calculation that needs it fails loudly.
4. **`fxc` is formed by central differences of the ANALYTIC first derivatives**
   with respect to the total-density variables, not by a second analytic
   derivative: the backend's order-2 output is only available in the
   spin-resolved variable set, whose chain rule back to `(ρ, σ)` needs the
   individual `γ_ab` cross-terms this crate cannot read. Accuracy is ~1e-9
   relative, which response calculations need but bit-parity work would not.
5. **Multigrid (`MultiGridNumInt`) is not ported.** PBC-MASTER-PLAN lists it in
   the Phase-12 crate description but not in any of the nine plans; every driver
   routes through the standard `KNumInt`.
6. **NLC / VV10 on a periodic grid is not wired.** `ks.do_nlc()` has no
   analogue; a functional requesting it is simply evaluated without the
   non-local term.
7. **No PyO3 bindings.** D-PBC-14 puts every periodic binding in Phase 20 plan
   20-05; `python/pyscf/pbc/__init__.py` remains the import-path overlay.
8. **DFT+U on a pseudopotential cell needs an explicit `minao_ref`.** MINAO
   carries core functions a GTH cell's AO space does not span, so the Löwdin
   metric is rank-deficient; the code errors clearly rather than
   pseudo-inverting. Upstream has the same property.
9. **`USite` names a Hubbard site by `(element, l, contraction)`**, not by an AO
   label string — `pyscf-core` has no `ao_labels`. For a minimal reference basis
   the contraction index is the same ordering upstream's `'Si 3p'` selects.

---

## 6. Deferral tests that plan 12-08 turned into value tests

Four tests in `pyscf-pbc-gto` existed to assert that a D-PBC-20 branch REFUSED.
Closing the branches made each of them assert the opposite; all four were
rewritten to compare against a value rather than deleted:

| test | was | is now |
|---|---|---|
| `ewald.rs::ewald_defers_the_dimension_2_branch_to_phase_12` | `NotYetImplemented{phase:12}` | `ewald_dimension_2_matches_the_recorded_upstream_target` — against `EWALD_REFERENCES["graphene"].ewald`, now `Some(-44.57202102404764)` |
| `ewald.rs::angstrom_reference_systems_match_upstream_within_the_unit_gap` | skipped graphene | gates all FIVE §9.2 systems |
| `ewald.rs::pseudised_ewald_matches_the_recorded_upstream_targets` | skipped `dimension != 3` | gates all five, including graphene's `-19.80978712179894` |
| `gv.rs::inf_vacuum_gv_weights_are_deferred_to_phase_12` | `NotYetImplemented{phase:12}` | `inf_vacuum_gv_weights_use_the_non_uniform_base` — mesh reduction, per-grid weights, `get_SI` on the reduced mesh |
| `coulg.rs::vcut_branches_defer_to_phase_12` | `NotYetImplemented{phase:12}` | `vcut_branches_produce_a_finite_kernel` — both kernels finite at `G + k = 0` |

The graphene targets were recorded during Phase 9 SPECIFICALLY as plan 12-08's
targets (`ewald_reference.rs` carried the upstream value in a comment beside
`ewald: None`), so these are pre-committed references, not numbers fitted after
the fact.

---

## 7. Determinism

Every grid-axis reduction in the periodic `NumInt` (`nelec`, `excsum`, and each
`V^k[μν]` element) goes through `pyscf_algebra::oracle_sum`, so the result is
independent of thread count — the FOUND-06 discipline Phases 1-11 already pay
for. The AO evaluation reaches the device through
`pyscf_pbc_gto::eval_ao_kpts` (K-07/K-08); nothing in `pyscf-pbc-dft` names a
`cubecl` symbol, so the ALG-06 wall holds.
