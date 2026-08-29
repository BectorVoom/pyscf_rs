# Phase 13 Verification — `ft_ao` + AFTDF

**Status:** implementation complete for plans 13-01 … 13-05 and 13-07; plan 13-06
(`fft_ao2mo` / `aft_ao2mo`) is **partially shipped** — `get_eri` and
`get_ao_pairs_G` for both builders, but not `general` / `get_mo_pairs_G` /
`ao2mo_7d`. See §5.
**Gates:** 1 **MET** (three variants), 2 **MET** (as a `(rcut, mesh)` ladder), 3 **MET for `get_nuc`/`vj`/`vk`**, near-met for `get_pp` — §1.
**Oracle:** vendored PySCF **2.12.1** at `<root>/pyscf`, `PYTHONPATH` pinned, the
version asserted in every oracle test.
**Reference system:** diamond, fcc `a₀ = 6.74064` Bohr, `gth-szv` / `gth-pade`,
8 AOs — plus He-fcc / `sto-3g` (all-electron) wherever the pseudopotential path
would hide a branch.

---

## 1. The gates, and what each one actually measures

Every tolerance below was measured on upstream **before** any Rust existed
(`measurements/README.md`), which is why none of them is the round number the
roadmap started with.

### Gate 1 — `ft_aopair[G=0] == periodic overlap`

The roadmap asked for 1e-10 against `pbc_intor("int1e_ovlp")`. **That is not
achievable, and the limiting factor is the reference, not the kernel.**
`ft_ao.estimate_rcut` is *looser* than `cell.rcut` (20.420 vs 21.319 Bohr), so
upstream's own `ft_aopair[G=0]` misses `int1e_ovlp` by 1.554e-9. Scaling
`estimate_rcut` on upstream converges the FT sum by ×1.5 — ×2.0 is identical to
four digits — and the residual still sits at 1.472e-10, because by then it is
`pbc_intor`'s own truncation.

So Gate 1 is three measurements, not one:

| | `ft_aopair` `rcut` | reference | target | **measured** |
|---|---|---|---|---|
| **1a** | `estimate_rcut` | `pbc_intor` | ≤ 2e-9 | **PASS** (upstream itself: 1.554e-9) |
| **1b** | 1.5 × `cell.rcut` | `pbc_intor` | ≤ 2e-10 | floor 1.472e-10, as predicted |
| **1c** | 1.5 × `cell.rcut` | `intor_cross_with_images`, **same `Ls`** | ≤ 1e-13 | **PASS** on diamond, He-fcc, and all 8 k-points of a 2×2×2 mesh |

**1c is the real gate on the McMurchie–Davidson algebra** — the only variant in
which both sides are converged over one identical image list, so nothing but the
recursion, the contraction or the cart→sph transform can move it.

`estimate_rcut` itself reproduces upstream element-for-element:
`[20.113957729325875, 20.420183850079926, …]`, and the `_RangeSeparatedCell`
split behind it — `[16.123457658142510, 20.113957729325875, 20.420183850079926]`
per shell — matches to 1e-9.

### Gate 2 — AFTDF `KRHF` == FFTDF `KRHF`

The roadmap said 1e-13. **Unreachable, and a monotone-in-mesh restatement would
fail a correct implementation.** Measured on upstream, diamond 2×2×2:

| mesh | `dvj` | `dvk` (`exxdiv='ewald'`) | `dvk` (`exxdiv=None`) | `|ΔE|` |
|---|---|---|---|---|
| 15 | 3.827e-7 | 1.112e-6 | — | 2.630e-5 |
| 21 | 2.365e-10 | 1.804e-9 | 1.690e-9 | 2.536e-8 |
| 31 | 1.996e-11 | 6.487e-10 | 2.653e-11 | **2.607e-11** |
| 41 | 1.996e-11 | 6.485e-10 | — | **2.607e-11** |

Mesh 31 and mesh 41 are **bit-identical in both energies, all 14 digits**. A
second, independent plateau appears in `rcut` at fixed mesh 31 — ×1.5 and ×2.0
agree to four digits — so there are **two floors**, AFTDF's screening and
FFTDF's aliasing, and lowering one alone stalls against the other.

Gate 2 is therefore a `(rcut, mesh)` ladder pinned to the recorded floor, not a
sweep. **The port reproduces both the shape and the rate.** `KRHF` on
diamond at gamma, both builders, `conv_tol = 1e-11`:

| mesh | `E_FFTDF` | `E_AFTDF` | `|ΔE|` | upstream at 2×2×2 |
|---|---|---|---|---|
| 15 | −10.13727190400028 | −10.13717881553468 | 9.309e-5 | 2.630e-5 |
| 21 | −10.13717271405267 | −10.13717266339423 | 5.066e-8 | 2.536e-8 |
| 27 | −10.13717266000622 | −10.13717265976846 | **2.378e-10** | — |

Same ~1000× per 6 mesh points as upstream, magnitudes within 2–3.6× (the k-sets
differ — gamma here, 2×2×2 upstream — so the levels are not directly comparable;
the convergence LAW is what Gate 2 asserts). `get_pp` vs FFTDF falls
2.339e-3 → 2.355e-5 → **6.035e-9** over meshes 11/15/21, and `KRHF` converges on
both builders for the all-electron He cell too.

**Two traps recorded.** With `exxdiv='ewald'` — the SCF default — the R-15 G=0
asymmetry is ~96% of `dvk` at mesh 31 (6.2e-10 of 6.487e-10; `exxdiv=None` drops
it 25×). And the ENERGY floors with `dvj` (1.996e-11), not with the `ewald`
`dvk` (6.487e-10): the G=0 term is a near-uniform shift that largely cancels in
`Tr(D·vk)`. **Matrices and energies need different tolerances**; one number
across both would either mask a real matrix defect or fail a correct energy.

### Gate 3 — vs upstream 2.12.1

| quantity | plan target | **measured** | |
|---|---|---|---|
| `AFTDF.get_nuc` | 1e-11 | **2.755e-12** | **MET** |
| `AFTDF.get_jk` `vj` | 1e-11 | **3.733e-12** | **MET** |
| `AFTDF.get_jk` `vk` | 1e-11 | **2.116e-12** | **MET** |
| `AFTDF.get_eri` | 1e-11 | **4.172e-12** | **MET** |
| `AFTDF.get_pp`, like-for-like `part2` | 1e-11 | **3.982e-11** | near |
| `AFTDF.get_pp`, as upstream computes it | 1e-11 | 1.806e-9 | see below |
| `ft_aopair` | — | **5.121e-10** | screening |

**Four of the five quantities beat the plan's 1e-11 outright**, which by itself
disproves the obvious hypothesis — that `ft_aopair`'s 5.121e-10 screening residual
propagates into everything. It does not. `get_nuc`, `vj`, `vk` and `get_eri` all run through
the same AO-pair transform and all land at 2–4e-12.

**`get_pp` is the exception, and the cause is upstream, not this port.**
`aft.get_pp` builds its part 2 with `_IntPPBuilder.get_pp_loc_part2()` — a
range-separated 3-centre builder — while `pp_int.get_pp_loc_part2` is the
reference implementation that Phase 10 ported and that `fft.get_pp` agrees with.
**Those two upstream routes disagree with each other by 1.7933e-9** on this cell,
measured directly:

```
max |_IntPPBuilder.get_pp_loc_part2() − pp_int.get_pp_loc_part2()| = 1.7933e-9
```

That is 99.3% of this port's 1.8056e-9 `get_pp` deviation. Substituting the
`pp_int` route into upstream's own `AFTDF.get_pp` and re-comparing collapses the
deviation to **3.982e-11** — a 45× improvement, and the residual is then ordinary
accumulated roundoff over the G-sum, not a screening artefact.

So the honest statement is: **this port's `get_pp` is consistent with upstream's
`pp_int` reference route to 3.98e-11**, and the remaining 1.79e-9 is a choice
between two upstream implementations that do not agree. Matching
`aft.get_pp` bit-for-bit means porting `_IntPPBuilder`, which is a Phase-14
3-centre builder (`incore.aux_e2` / `_Int3cBuilder`, plan 14-01) — not a Phase-13
deliverable. The `pp_int_part2` row is asserted in the test suite so the
attribution cannot silently rot.

**`ft_aopair`'s own 5.121e-10** vs upstream is separate and is screening. Three of
upstream's screens were ported to get there — `strip_basis`, `get_ovlp_mask` over
the `_RangeSeparatedCell` per-primitive grouping, and libcint's `PTR_EXPCUTOFF` —
taking it 1.553e-9 → 5.733e-10 → 5.121e-10. What remains is upstream's
`ExtendedMole` supermole construction, which D-PBC-21 declines to port. Given
that `get_nuc`/`vj`/`vk` are at 2–4e-12 despite it, this residual is bounded and
not load-bearing.

---

## 2. Evidence that the Gate-3 residual is truncation, not algebra

All oracle-free:

| check | result |
|---|---|
| Gate 1c, matching image list, 2 cells × 8 k-points | **1e-13** |
| `get_pp` anti-Hermitian residue, converged `rcut` | **2.665e-15** (vs 5.133e-11 at upstream `rcut`) |
| screens self-consistent over image lists at 20.4 / 32.0 / 42.6 Bohr | **bit-identical** |
| `ft_aopair` vs a dense-grid numerical FT | < 1e-6 |
| `s`-`s` closed form | < 1e-14 |
| `ft[μν,G] == conj(ft[μν,−G])` | < 1e-13 |
| `get_pp` vs FFTDF, mesh 11 → 21 | 2.34e-3 → 6.03e-9 |

The Hermiticity pair is the sharpest of these. Upstream's `strip_basis` screens
on the KET shell alone, so it is not symmetric in `(μ,ν)`; `ft_ao.py:749-753`
tightens `precision` by 1e-2 for exactly this reason ("errors around the required
precision [are] found when checking hermitian symmetry"). At `cell.precision =
1e-8` that target is 1e-10 and the measured residue is 5.133e-11 — on target. At
a converged `rcut` it collapses to 2.665e-15. A Hermiticity failure that survived
a converged `rcut` would be an algebra bug; this one does not.

---

## 3. What shipped

| plan | content | tests |
|---|---|---|
| **13-01** | K-15 `ft_aopair` cubecl kernel + host McMurchie–Davidson recursion + `estimate_rcut` + the three upstream screens | `pyscf-pbc-df --test ft_ao` **9/9** |
| **13-02** | single-centre `ft_ao`, `_fake_nuc` | `--test ft_ao_single` **4/4** |
| **13-03** | `FtKernel` (build/eval split), G-blocking, `q` shift, k-resolution | folded into the above |
| **13-04** | `AFTDF`: `ft_loop`, `weighted_coulG`, `get_pp_loc_part1`, `get_nuc`, `get_pp` | `--test aftdf` |
| **13-05** | `aft_jk`: `get_j_kpts`, `get_k_kpts`, `get_jk` | `--test aftdf` |
| **13-07** | `Box<dyn PeriodicDf>` across all 8 drivers + `veff::get_jk` + `get_hcore` | `pyscf-pbc-scf --test df_swap` |
| **13-06** (partial) | `get_eri` + `get_ao_pairs_G` for AFTDF and FFTDF, through ONE shared contraction | `--test pbc_ao2mo` **3/3** |

`get_eri` results: 8-fold permutational symmetry at gamma holds to **1.966e-12**
(the sequential G-sum's roundoff floor — the oracle bounds total numerical noise
at 4.172e-12 against an upstream value that is symmetric by construction);
AFTDF vs FFTDF converges **4.250e-5 → 5.855e-7 → 5.427e-10** over meshes
11/15/21; and the oracle agrees to **4.172e-12**.

`boxing_the_builder_is_bit_identical` passes: `Krhf::new` and an explicitly
constructed `Fftdf` give **bit-identical** energies, so D-PBC-22 moved no number.

**And the FFTDF oracle suite confirms it against upstream, not just against
itself:** `cargo test -p pyscf-pbc-df --test fftdf -- --include-ignored` is
**11/11**, including the four oracle tests Phase 11 marks `#[ignore]` —
`get_pp`, `get_hcore` and `jk` on diamond 2×2×2, and `get_nuc` on He-fcc. A
plain `cargo test` reports those as *ignored*, not run; the refactor would have
looked green without them.

### Verification totals

| | |
|---|---|
| `pyscf-kernels` | 26 passed, 0 failed (7 binaries) |
| `pyscf-pbc-df` | 27 passed, 0 failed (5 binaries) + 11/11 FFTDF with `--include-ignored` |
| `pyscf-pbc-scf --test df_swap` | 4 passed, 0 failed |
| `pyscf-pbc-scf` / `pyscf-pbc-dft` test binaries | compile clean after D-PBC-22 |
| `cargo build --workspace` | 0 errors |
| `check-dependency-wall` | PASS — cubecl-* containment intact (ALG-06) |
| `check-orphan-modules` | PASS — 279 source files, all reachable |
| `check-no-fma` | PASS — no FMA mnemonics in `release-oracle` asm (FOUND-05) |

Bit-reproducibility (D-PBC-17): every new contraction — `get_pp_loc_part1`,
`get_j_kpts`, `get_k_kpts`, `contract_eri` — is a sequential `+=` over a fixed
index order with no `rayon` anywhere in the new code, so it is thread-count
independent by construction rather than by `oracle_sum` wrapping.

### Four defects the tests caught

1. **The per-record screen was `cell.precision·1e-2`.** That threshold derives the
   image list; applying it again as an absolute per-primitive-pair cutoff dropped
   ~1e-10 terms that accumulated over `nimgs × nprim²` records — 1.66e-7 on
   diamond's `p`-`p` block, with every angular-off-diagonal element still exact to
   1e-16, which is the signature of a screen rather than an algebra bug.
2. **`estimate_rcut`'s `cs` is the libcint contraction coefficient, not
   `gto_norm`.** `aft.estimate_ke_cutoff` overwrites `cs` with `gto_norm`;
   `ft_ao.estimate_rcut` does not. Conflating them moved the radius 21.186 vs
   20.420 Bohr.
3. **`Γ(1.5)` returned 1.** The half-integer reduction loop stopped at `> 1.5`
   instead of `> 1.0`, making `_fake_nuc` short by exactly `√π/2` — caught by the
   identity that `_fake_nuc` + `ft_ao` must reproduce `get_gth_vlocG_part1 × SI`.
4. **Phase-10 outputs are F-order.** `get_pp_loc_part2` and `get_pp_nl` needed
   `forder_to_c`; adding them raw transposed the non-local block and broke
   Hermiticity.

### One process finding

Plan 13-07's Task 1 says "map the blast radius before touching anything", and I
did — but only over `with_df.<member>` USAGE, which found exactly `kpts`, `cell`
and `mesh` (all trait methods, so the refactor looked clean). That query missed
every `from_df(` CONSTRUCTION site, and the signature change broke three test
files and one example that only surfaced at `cargo test --no-run`, after the
libraries were already green. **A type change needs both queries: who reads the
field, and who builds one.** Recorded because the next such refactor —
`_IntPPBuilder` in Phase 14 — has the same shape.

### Two performance corrections, both driven by measurement

- `FtKernel` was going to be "consolidated away" into `ft_loop` as unnecessary.
  It is not: the record table is `O(nimgs·nprim²·npairs)` McMurchie–Davidson
  recursions and does **not** depend on `G`, so rebuilding it per G-block made one
  `get_pp` at mesh 15 take minutes. Plan 13-03 was right and the consolidation
  note was wrong.
- `get_k_kpts` built one kernel per `(ki, kj)` pair, but the table depends on the
  ket k-point only — never on `q`. Building `nkpts` instead of `nkpts²` is an 8×
  saving on a 2×2×2 mesh and the difference between an SCF that finishes and one
  that does not.

---

## 4. Deviations from the plan, recorded

| plan said | shipped | why |
|---|---|---|
| kernel generic over `F: Float` | concrete `f64` | `cube_math::double` IS the f64 libm and a `#[cube]` body generic over `F` cannot call it. Every sibling PBC kernel that does scalar math (`struct_factor`, `ewald`, `eval_gto`) is concrete `f64` for the same reason; `cube-math` exposes `single` for f32, so the f32 seam is a second entry point, not a type parameter. |
| tests in `pyscf-kernels/tests/pbc_ft_aopair.rs` | `pyscf-pbc-df/tests/ft_ao.rs` | the kernel entry point takes pre-built flat tables; exercising it means building them, which is the `pyscf-pbc-df` driver. A test in `pyscf-kernels` could only check the kernel against a re-implementation of the driver. |
| `ft_ao` in `crates/pyscf-gto/src/ft_ao.rs` | `pyscf-pbc-df/src/ft_ao/single.rs` | it must reuse the McMurchie–Davidson recursion, which lives in `pyscf-pbc-df`. The alternative was a second recursion, which 13-02's own must-haves forbid. |
| 13-07 STEP 5: `mf.with_df = AFTDF(...)` from Python | not done — **not doable** | there is NO periodic PyO3 surface: `pbc` does not appear anywhere in `crates/pyscf-py/`. D-PBC-14 puts the whole `pyscf.pbc.*` binding surface in **Phase 20**, so STEP 5 was premised on something that does not exist yet. A plan defect, not a skipped step. |
| `aft_jk` via `kk_adapted_iter` + `swap_2e` | every ordered `(ki,kj)` pair, "case 1" only | the same sum with no symmetry bookkeeping to get wrong. `kk_adapted_iter`/`group_by_conj_pairs` were therefore NOT ported — they are still missing from `pyscf-pbc-lib` and Phase 14 will need them. |
| Gate 1 ≤ 1e-10, Gate 2 = 1e-13, Gate 3 = 1e-11 | 1a/1b/1c, a `(rcut,mesh)` ladder, 5e-9 | see §1 — all three originals were unmeasured. |

---

## 5. Tuning accuracy — measured 2026-08-29

Asked to "optimise precision", the first job was finding out what actually
limits it. Two candidates: summation roundoff over the G-sum, and lattice-sum
screening. **One measurement separates them.**

The gamma-point ERI is exactly 8-fold symmetric, so any asymmetry is pure error.
Measured across meshes at the default precision, the residue is **1.9659e-12 at
every mesh — bit-identical from 1 331 to 19 683 G-vectors.** Roundoff grows with
the number of terms; this does not. It is screening, not summation.

Confirmed from the other side: at a converged `rcut` the residue drops to
2.9143e-16 at 1 331 G and 2.9230e-16 at 9 261 — a 0.3% rise over 7× more terms,
extrapolating to ~3.0e-16 at the default mesh 47. **The G-sums are well
conditioned, so routing them through `oracle_sum` would buy nothing measurable.**
That work was considered and declined on evidence rather than skipped.

### The knob is `cell.precision`, and it is cheaper than a radius multiplier

`estimate_rcut` targets `cell.precision · 1e-2` and all three screens derive
their cutoffs from `cell.precision`, so tightening it moves the radius AND the
screens together. `RcutChoice::Scaled` only inflates the radius.

| `precision` | `rcut` | images | ERI residue | `get_pp` anti-Hermitian |
|---|---|---|---|---|
| 1e-8 (default) | 20.420 | 675 | 1.966e-12 | 5.131e-11 |
| 1e-10 | 22.297 | 887 (1.31×) | 1.497e-14 | 5.000e-13 |
| **1e-12** | **24.020** | **1055 (1.56×)** | **3.842e-16** | **4.647e-15** |
| `Scaled(1.5)` | 31.979 | 2315 (3.43×) | 2.914e-16 | 2.665e-15 |

`precision = 1e-12` reaches the f64 floor for **2.2× fewer lattice images** than
`Scaled(1.5)` — an 11 000× accuracy gain on `get_pp`'s Hermiticity for a 1.56×
image count. (Wall-clock is not a clean cost proxy here: these runs shared a
machine with other builds, and the runtime is dominated by the `ngrids × nao²`
G-contraction rather than the lattice sum.)

### Set it at BUILD time

`cell.rcut` is cached during `Cell::build`. A post-hoc `cell.precision = p`
tightens only the call-time estimators (this phase's), leaving `pbc_intor`,
Ewald and `eval_gto` on the original target. Built with the tighter value, both
sides of the `G = 0` identity converge — including the `pbc_intor` truncation
that WAS Gate 1b's floor:

| `precision` | `cell.rcut` | `\|ft[G=0] − int1e_ovlp\|` |
|---|---|---|
| 1e-8 (default) | 21.319 | 1.189e-9 |
| 1e-10 | 23.193 | 1.142e-11 |
| **1e-12** | **24.910** | **8.416e-14** — 14 000× better |

### The default does not change, and here is why

Converging the sum moves the result AWAY from upstream's truncated value, so
tightening `precision` by default would break Gate 3 — and `AFTDF` is *defined*
by upstream's screening. The default stays at `cell.precision = 1e-8` with
`RcutChoice::Upstream`; the accuracy path is documented, tested and one field
away. Both directions are now pinned by tests
(`tightening_cell_precision_improves_the_eri`,
`build_time_precision_converges_both_sides_of_gate1`) so neither can regress
silently.

---

## 6. Carry-overs

1. **Plan 13-06 is partial.** `get_eri` and `get_ao_pairs_G` ship for both
   builders, sharing one contraction so the cross-builder test measures
   `ft_aopair` rather than the transform. **`general`, `get_mo_pairs_G` and
   `ao2mo_7d` do NOT ship.** `ao2mo_7d` is deliberately left out rather than
   guessed: its `[nk,nk,nk,nmo,nmo,nmo,nmo]` index order is a contract with
   Phase 15's `KMP2` and Phase 16's `KCCSD`, and it should be defined against a
   real consumer instead of inherited from a Phase-13 guess. Phase 15 is
   blocked on it either way.
2. **Gate 3 is met for `get_nuc`/`vj`/`vk` and near-met for `get_pp`.** Two
   separate residuals remain, and neither is Phase-13 work: (a) `ft_aopair`'s
   5.121e-10 needs upstream's `ExtendedMole` supermole, and (b) `get_pp`'s
   1.79e-9 needs `_IntPPBuilder`, a 3-centre builder. Both land naturally in
   Phase 14 — `gdf_builder` needs the first and `incore.aux_e2` IS the second.
   **Upstream's own `pp_int` and `_IntPPBuilder` part-2 routes disagree by
   1.79e-9**, which is worth reporting upstream independently of this port.
3. **`kk_adapted_iter` / `group_by_conj_pairs`** remain unported (§4).
4. **AFTDF band k-points** (`get_j_for_bands` / `get_k_for_bands`) return
   `NotYetImplemented { phase: 14 }`, as do the four alternative `_update_vk*`
   variants.
5. **The BvK bucket contraction** (D-PBC-21) is still deferred; `ft_aopair` runs
   one launch per k-point.
