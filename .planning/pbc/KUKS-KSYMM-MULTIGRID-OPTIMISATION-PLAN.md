# KUKS + k-symmetry + multigrid — Memory & Speed Optimisation Plan

**Created:** 2026-09-02
**Target:** the three periodic-DFT surfaces Phase 17 left in the tree:
`Kuks` (`pbc.dft.KUKS`), the k-symmetric adapters `KsymAdaptedKrks` /
`KsymAdaptedKuks` / `KsymAdaptedKrhf`, and the two multigrid drivers
`MultiGridNumInt` (v1) / `MultiGridNumInt2` (v2).
**Successor to:** [`KUKS-OPTIMISATION-PLAN.md`](./KUKS-OPTIMISATION-PLAN.md)
(U-04, U-05 and U-06 step 6 still open) and
[`KRKS-OPTIMISATION-PLAN.md`](./KRKS-OPTIMISATION-PLAN.md) (W-03/W-04/W-06/W-09
still open). Neither is superseded; this plan inherits their gates and their
rules and only adds items those plans could not see because the ksymm and
multigrid code did not exist when they were written.
**Status:** **IN EXECUTION.** See
[`KUKS-KSYMM-MULTIGRID-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-EXECUTION-SUMMARY.md)
for the session record. As of 2026-09-02: **P-00, P-01, P-02, P-03, S-00,
S-01 (step 1), S-02, U-09 (step 2), U-10, M-00, M-02, M-03, M-04 (steps 1-2)
LANDED**; **S-03, S-04, M-01 NOT STARTED**; **S-05, M-05 deferred as planned**.
Two items were closed by MEASUREMENT rather than by implementation — U-09
step 1 (16 KiB, not worth a public-signature refactor) and D-PBC-26 point 1
(not an identity; §2.2.3's derivation, now with a test). Every number below is
evidence-tagged per RULE V.
**Successor:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md)
(2026-09-03) re-schedules the open items (S-03, S-04, M-01, M-04 step 3,
S-02 step 4) under their original numbers, adds the AO-evaluation track
(A-00…A-03) this plan recorded as out of scope, and the kernel-level
multigrid items M-06…M-10. This file is not superseded; its rules and gates
are inherited there.
**Audience:** an execution agent that follows instructions literally and does
NOT infer.

---

## 0. HOW TO EXECUTE THIS PLAN

Inherits every standing rule of [`PBC-MASTER-PLAN.md`](./PBC-MASTER-PLAN.md) §0,
`AGENTS.md`, `KRKS-OPTIMISATION-PLAN.md` §0 and `KUKS-OPTIMISATION-PLAN.md` §0:

* **RULE 4** — tests in separate files, never `mod tests` in a production
  source file.
* **RULE 5** — cubecl: read the manual **before** writing a kernel; kernels
  are `<F: Float + CubeElement>` except where the file header documents the
  `exp`-only `f64` exception; on ANY cubecl build error read
  `cubecl_error_guideline.md` before touching code.
* **RULE 6** — ALG-06: no `cubecl-*` dependency outside `pyscf-algebra` /
  `pyscf-kernels`; `cargo run -p xtask --bin check-dependency-wall` after
  every item that touches `Cargo.toml`.
* **RULE O** — measure, change ONE thing, re-measure. Every item ends with
  a re-profile against the baseline its section names.
* **RULE U** — no KUKS item is validated on a closed-shell cell
  (`dm_a == dm_b` is an exact fixed point; the unrestricted path degenerates).
* **RULE V** — every quantity is `MEASURED (source)`, `MODELLED` or
  `UNVERIFIED`; promote only by doing the work.
* **D-PBC-17** — every reduction that reaches an energy, a density matrix or
  a convergence test goes through `oracle_sum`/`oracle_dot`, and is
  bit-identical under `RAYON_NUM_THREADS=1` and `=8`.

Three rules are new to this plan.

* **RULE K — an IBZ-vs-full-BZ comparison is only meaningful when the full-BZ
  solution is star-symmetric (D-17-08-03).** Every ksymm speed item is
  validated against the port's OWN reference route inside one process (two
  routes, one density, 1e-13), never by comparing two SCFs. A ksymm gate that
  compares converged energies must assert `check_mo_occ_symmetry` on the
  full-BZ side first, or it is measuring physics, not code.
* **RULE M — a multigrid speed claim is a ratio against the reference
  `KNumInt` + `Fftdf` route at the SAME mesh, measured in the same process.**
  17-01 measured upstream's own multigrid SLOWER than its reference route
  (0.18x-0.49x), and 17-12 measured this port's v2 at 0.02x. "Multigrid is
  faster" is not a premise anywhere in this plan.
* **RULE S — a symmetry-derived saving is opt-in and gate-scored when it
  changes the arithmetic.** The literal upstream route (unfold, run the
  full-BZ code, fold) stays as the reference and the default until the
  opt-in route has BOTH a 1e-13 equivalence test and a measured speed ratio
  above 1.0 on the CPU backend (D-PBC-26 point 6, the `zgemm_dense`
  precedent).

---

## 1. Scope and the gates

### 1.1 In scope

| surface | code | crate |
|---|---|---|
| KUKS driver | `kuks.rs:199-284` `veff_from_parts`, `:376-398` `energy_elec` | `pyscf-pbc-dft` |
| open-shell quadrature | `numint.rs:629-731` `nr_uks`, `:1176-1263` `eval_rho_one`, `:1107-1172` `vxc_mat_one` | `pyscf-pbc-dft` |
| ksymm KRKS/KUKS | `krks_ksymm.rs:155-251` `KsymAdaptedKrks::get_veff_tagged`, `:570-660` `KsymAdaptedKuks::get_veff_tagged`, `:220-246` / `:648-666` `weighted_trace*` | `pyscf-pbc-dft` |
| ksymm KRHF J/K routes | `khf_ksymm.rs:277-337` `veff_reference` / `veff_fast` | `pyscf-pbc-scf` |
| the IBZ⇄BZ transforms | `kpts.rs:1163-1265` `transform_dm` / `transform_1e_operator` / `sandwich_unfold`, `:1657` `symmetrize_density` | `pyscf-pbc-symm` |
| ksymm numint plumbing | `numint.rs:274-393` `unfold_kdms` / `unfold_mos` / `unfold_dms`, `:419-450` the AO cache | `pyscf-pbc-dft` |
| multigrid v1 | `multigrid/numint.rs:151-266` `nr_rks`, `colloc.rs:53-160` `collocate_level`, `:217-276` `level_rho` / `level_pass2` | `pyscf-pbc-dft` |
| multigrid v2 | `multigrid/pair.rs:494-700` `build_pair_level_table`, `:733-957` `grid_blocks` / `block_slots` / `pairlevel_rho` / `pairlevel_pass2`, `:1067-1163` `nr_rks` | `pyscf-pbc-dft` |
| multigrid kernels | `multigrid_pair.rs:592-676` `launch_rho` / `launch_integrate`, `multigrid_collocate.rs:207-240` | `pyscf-kernels` |
| the FFTDF K build under `kpts_band` | `fft_jk.rs:330-470` `get_k_kpts_opts` | `pyscf-pbc-df` |

### 1.2 Out of scope (non-goals)

* Everything the KRKS plan still owns and has not landed — **W-03/W-04**
  (GEMM + device residency for `fft_jk`), **W-06** (GEMM for `numint`),
  **W-09** (AO screening). Where an item here depends on one, it says so and
  waits. In particular **U-05 stays sequenced after W-06** exactly as the
  KUKS plan states.
* Changing any gated energy without a re-baselined gate. Items that change
  the arithmetic say so in their heading and ship opt-in.
* Porting `grid_collocate.c` / `grid_integrate.c` line by line
  (17-12 judged it out of budget; nothing here re-opens that).
* k-point multigrid. Both multigrid drivers are gamma-only by 17-11/17-12's
  stated scope; a k-point Bloch-phase collocation is Phase-18-or-later work
  and is not an "optimisation" of anything that exists.
* `Kukspu` (U-08) and the `KsymAdaptedKuhf` adapter (17-07 Task 5) — the
  first is a completeness gap, the second is a port item, neither is speed.

### 1.3 GATE A — the closed-shell accuracy gate must not regress

`crates/pyscf-pbc-dft/tests/gate.rs`, current tolerances, unchanged.
MEASURED 2026-09-02 (`KUKS-EXECUTION-SUMMARY.md` §Verification): 7/7 PASS,
`KUKS Si 2×2×2 PBE` at 6.446e-12 against 1e-11.

```bash
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
```

### 1.4 GATE U — the open-shell gate must not regress

`crates/pyscf-pbc-dft/tests/gate_openshell.rs`. MEASURED 2026-09-02: five rows
green as written, four `Li/sto-3g` rows re-gated to `TOL_LI = 5e-11` on a
measured `get_nuc` quadrature floor (the no-XC `KUHF` control is the WORST row
at 1.494e-11). **The re-gated rows have NOT been re-run since the tolerance
edit** — see §3 item P-00.

### 1.5 GATE B — determinism (D-PBC-17)

Bit-identical energies under `RAYON_NUM_THREADS=1` and `=8`, for every driver
this plan touches. Last MEASURED on the six `gate.rs` energies
(`SUMMARY.md:118-127`); NOT yet measured on any ksymm driver or on either
multigrid driver's `nr_rks` (only `eval_rho_g` is covered —
`multigrid2.rs::eval_rho_g_is_bit_identical_across_thread_counts_v2`). §3
item P-01 closes that.

### 1.6 GATE C — the ksymm port-vs-port gate

`crates/pyscf-pbc-dft/tests/krks_ksymm.rs::krks_ibz_energy_matches_full_bz`.
MEASURED (17-08-SUMMARY): 3.109e-14 / 2.842e-14 on `si [2,2,2]` FFTDF, both
`use_ao_symmetry` branches. The GDF row FAILS at 1.432e-06 and is recorded,
not absorbed; the KUKS row is `#[ignore]`d on the D-17-08-03 precondition.
This plan does not touch the GDF failure — it is a fidelity item, not speed.

### 1.7 GATE E — multigrid vs reference numint

`tests/multigrid.rs` (v1) and `tests/multigrid2.rs` (v2), at the quadrature
floors 17-11/17-12 measured: v1 `∫ρ − Tr(DS)` ≤ 2.1e-11, v2 `get_j` vs FFTDF
≤ 6.8e-8 (si) / 1.24e-8 (diamond). Every multigrid item here is scored
against these; none may move them.

### 1.8 GATE S — the new speed-and-memory ledger (this plan builds it)

One table, kept in `KUKS-KSYMM-MULTIGRID-EXECUTION-SUMMARY.md` when execution
starts, with one row per item: `(cell, k-mesh, mesh, xc, driver)`, wall time
of the timed stage BEFORE and AFTER, peak RSS BEFORE and AFTER, the gate
residual BEFORE and AFTER, and whether the change is bit-exact. A row with a
blank cell reads as "never measured" — the failure mode 17-13 Task 2 names.

---

## 2. Where the time and the memory go

### 2.0 READ THIS FIRST — what is and is not measured

| quantity | status |
|---|---|
| KUKS/KRKS wall-time multiplier on `si 2×2×2 mesh 31` | **MEASURED 2026-09-02, §2.1.0a** — K build ×1.03 (hybrid), J build ×1.9-2.2, `nr_uks`/`nr_rks` ×1.3-1.75; all three are small against the one-off AO evaluation |
| ksymm `get_veff` wall time vs the non-symmetric driver | **STILL UNMEASURED** — S-00 shipped the harness (`krks_profile ksymm`) on 2026-09-02, but the machine ran at load average 28-33 all session and RULE O forbids quoting a ratio off that |
| ksymm peak memory | **STILL UNMEASURED** — same reason; the harness now reports `VmHWM` |
| multigrid v1 / v2 vs reference | MEASURED (17-12): v2 `get_j` 0.023x / 0.028x of the reference route, v1 1.5x / 1.2x FASTER than the reference on `get_j` alone at `mesh 25³`; 17-01 measured upstream's own v1/v2 at 0.18x-0.49x on a full SCF |
| multigrid peak memory | MEASURED (17-12): v2 density evaluation 0.46 GiB after the streaming fix; v1 UNMEASURED |

#### 2.1.0 The KUKS baseline (U-01), filled by this session if the machine stayed idle

The instrument: `cargo run -p pyscf-bench --release --bin krks_profile -- jk
--driver kuks --cell si --nk 2,2,2 --mesh 31,31,31 --xc <pbe|pbe0>`. Load
average at the start of this session was **1.7 on 16 cores** (MEASURED,
`uptime`), i.e. RULE O's precondition that the previous session lacked.

> The numbers are in §2.1.0a, directly below, with the raw JSON paths.

#### 2.1.0a The KUKS baseline — MEASURED 2026-09-02

`si` `gth-szv`/`gth-pade`, `nao = 8`, `[2,2,2]` (`nkpts = 8`), `mesh = 31³`
(`ngrids = 29 791`), release build, CPU backend. Raw reports:
`.planning/pbc/baselines/2026-09-02-kuks-si222-mesh31-{pbe,pbe0}.json`.
Load average at launch: **3.6** (pbe0) / **8.5** (pbe, still decaying from
the harness build) on 16 cores — the pbe0 run is the clean one; the pbe run's
absolute times carry some contention but its RATIOS are taken within one
process on identical data and are the quantity U-01 asks for.

| stage (warm unless stated) | `pbe` (pure) | `pbe0` (hybrid) |
|---|---|---|
| `get_j_kpts`, nset = 1 | 29.3 ms | 19.3 ms |
| `get_j_kpts`, nset = 2 | 55.7 ms (**×1.90**) | 42.5 ms (**×2.20**) |
| `get_k_kpts`, nset = 1 | — | 6 685 ms |
| `get_k_kpts`, nset = 2 | — | 6 913 ms (**×1.034**) |
| `nr_rks` | 39.0 ms | 82.6 ms |
| `nr_uks` | 68.4 ms (**×1.75**) | 107.1 ms (**×1.30**) |
| `nr_rks` COLD (first call, AO evaluation included) | **6 358 ms** | 6 037 ms |
| `get_hcore` (one-off) | 2 369 ms | 2 043 ms |
| `get_ovlp` (one-off) | 195 ms | 191 ms |
| full `Kuks::kernel()` | 10 148 ms | 52 074 ms |

**What this settles, and it re-orders the KUKS plan's speed items.**

1. **The hybrid multiplier is 1.03.** The KUKS plan's §2.1.1 structural claim
   ("the expensive half of `get_k_kpts` is spin-independent") is now
   MEASURED: the doubled contractions are ~3 % of the K build. A hybrid KUKS
   costs a hybrid KRKS plus 3 %. There is nothing KUKS-specific left to win
   in `get_k_kpts`; every remaining K-build item is a KRKS item (W-03/W-04,
   W-08).
2. **U-04 and U-05 are worth ~30 ms and ~30-40 ms per cycle respectively on
   this cell.** Against a 6.7 s K build (hybrid) that is < 1 %; against the
   pure-functional SCF (10.1 s, of which 6.4 s is the one-off AO evaluation
   and 2.4 s `get_hcore`) it is also < 1 %. Both items keep their place in
   the KUKS plan's sequencing but **must not be scheduled ahead of anything
   in §3 here on speed grounds**. U-05's real value is the memory shape
   (U-10), which is why U-10 is in this plan and U-05 is not.
3. **The pure-functional SCF is dominated by the COLD AO evaluation** —
   `eval_ao_kpts` on the full mesh at 8 k-points takes 6.0-6.4 s against a
   39-83 ms warm quadrature. That cost is paid once per SCF (the AO cache
   holds it) and it scales as `nkpts · nao · ngrids`. This is the number
   §2.2.4's `N_ibz / N` saving (S-03) applies to under symmetry, and it is
   why S-03 is the headline pure-functional item rather than any
   contraction. The AO evaluator itself (`pyscf-pbc-gto::eval_ao_kpts`) is
   outside this plan's scope and belongs in the KRKS plan as a new W- item;
   recorded here so it is not lost.
4. `get_hcore` at 2.0-2.4 s is the second one-off. Under symmetry it is
   already built at `kpts_ibz` only (`krks_ksymm.rs:269-271`), so the ksymm
   adapters already take that saving.

RULE O status: this is the KUKS baseline the previous session could not
take. Every U- item in this plan re-profiles against these two JSON files
with `--compare`.

### 2.1 KUKS (non-symmetric) — what is left after U-00…U-07

VERIFIED by reading the source this session; each row carries its line.

| fact | evidence | consequence |
|---|---|---|
| `veff_from_parts` still clones both channels into `sets` — `[vec![dms[0].clone()], vec![dms[1].clone()]]` | `kuks.rs:214` | U-06 step 2 was "redirected" to `unfold_kdms`'s bigger clone and this one survived. `2 · nkpts · nao²` complex per `get_veff`, per cycle. Bit-exact to remove |
| `Kuks::energy_elec`'s `e1` is a naive `e1 += trace_ab(..)` over `2·nkpts` terms | `kuks.rs:388-393` | U-03 ordered `krdm::energy_elec` (the KUHF copy) and `veff.rs`'s traces, and did NOT reach this fold. Same defect class U-03 closed, one file over. For `nkpts ≤ 64` this is `≤ 128` terms — small, but it is ON the total energy |
| `nr_uks` allocates per block: `rho_a`, `rho_b`, `dena`, `denb`, `ta`, `tb`, plus `weighted(..)` twice; `eval_rho_one` allocates `c0_re/c0_im` (`ngrids·nao` each) and `acc_re/acc_im` per component, per k, per spin; `vxc_mat_one` allocates `aow_re/aow_im` (`ngrids·nao`) per k per spin and `terms_re/terms_im` (`ngrids`) per output row per worker | `numint.rs:679-716`, `:1202-1250`, `:1124-1170` | At `si 2×2×2 mesh 31` (`ngrids = 29 791`, `nao = 8`): `c0` alone is 3.8 MiB × 8 k × 2 spins = **61 MiB of allocate-and-zero per block per cycle**, twice what KRKS pays. This is U-06 step 6 ("interior-mutable scratch on `KNumInt`"), left open there and still open |
| the AO table is shared between the two spins (one `eval_ao` per block) | `numint.rs:683-688` | correct; the doubling is contraction-only, as the KUKS plan's §2.1.1 says |
| `get_k_kpts` transforms are spin-independent; only `contract_vr_aodm` / `accumulate_vk` double | `fft_jk.rs:401-470` | unchanged since the KUKS plan; the hybrid multiplier is `1 + (contraction share)`, which U-01 measures |
| `get_j_kpts` runs the whole `accumulate_rho → fft → coulG → ifft` chain once PER SET | `fft_jk.rs:87-116` | U-04 territory, unchanged |
| the `KNumInt` AO cache holds `comp · nkpts · ngrids · nao · 16` bytes under `0.25 · max_memory` | `numint.rs:437-446` | 122 MiB for GGA at the reference cell; grows linearly in `nkpts` and `nao`. Not KUKS-specific, but it is the dominant resident allocation of every DFT driver and the ksymm section below is about shrinking it |

### 2.2 k-point symmetry — where the IBZ saving is and is not being taken

#### 2.2.1 The DFT ksymm adapters unfold the density TWICE per cycle (VERIFIED)

`KsymAdaptedKrks::get_veff_tagged`:

1. `self.ni.nr_rks(cell, grids, xc, dms, 1, Some(&band))` — `nr_rks` calls
   `unfold_kdms` (`numint.rs:566`) → `transform_dm` over all `nkpts`.
2. `let dm_bz = self.ni.unfold_kdms(cell, dms, nao)?` (`krks_ksymm.rs:182`)
   → the SAME `transform_dm` again, for the J/K call.

`KsymAdaptedKuks::get_veff_tagged` does it for both channels: `nr_uks`
unfolds `dms[0]` and `dms[1]` (`numint.rs:640-643`), then `:597` unfolds both
again. **Four `transform_dm` calls per cycle where two are needed.**

Each `transform_dm` is `nkpts` sandwiches `R·M·Rᴴ` (`nao³` complex each,
rayon over `k`), plus TWO full format conversions through the
`CTensor ⇄ Vec<Complex64>` seam (`khf_ksymm.rs:158-170`, and the same idiom
inlined at `numint.rs:329-352`), plus `get_rotation_mat` per `(k, op)`
(`symmetry.rs:970-980`), which is rebuilt on every call and never cached
across the SCF. The rotation matrices depend only on `(cell, k, op)`.

17-08 measured `unfold_kdms` to be a **bit-exact no-op on full-BZ input**
(`krks_ksymm.rs::unfold_is_a_bit_exact_no_op_on_full_bz_input`). So unfolding
once in `get_veff_tagged` and handing the full-BZ stack to `nr_rks`/`nr_uks`
is bit-exact by an existing test. This is **item S-01**.

#### 2.2.2 The `weighted_trace*` folds are naive and on the energy path (VERIFIED)

`krks_ksymm.rs:220-246` (`weighted_trace`) and `:648-666`
(`weighted_trace_uks`): `t += d.re*v.re - d.im*v.im` over `nao²`, then
`acc += w[k] * t` over `nkpts_ibz`, both plain running sums. They feed
`ecoul`, the hybrid `exc` correction AND `energy_elec`'s `e1` for every ksymm
KS driver — exactly the pattern D-PBC-17 forbids and U-03 closed for the
non-symmetric drivers. `KsymAdaptedKuks::energy_elec` additionally
materialises `vec![h1e.to_vec(), h1e.to_vec()]` (`:855`) and
`get_veff_tagged` materialises `vec![jtot.clone(), jtot.clone()]` (`:626`) —
the two clones U-06 deleted from `kuks.rs`, still present in the ksymm twin.
**Item P-02.**

#### 2.2.3 D-PBC-26's "IBZ-only `get_jk`" is not an identity — MODELLED, with the derivation

17-CONTEXT §8 rules that `get_jk` should be called at `kpts_ibz` only and the
result unfolded with `transform_1e_operator`, targeting the 40x (GDF) / 223x
(FFTDF) ratios `speed_get_jk.py` measured. `khf_ksymm.rs:312-337` implements
that literally as `veff_fast`, and 17-07 records it as **NOT validated**.

It cannot validate at 1e-13, for a reason that does not depend on the DF route:

* **J.** `get_j_kpts` forms `ρ(r) = (1/N) Σ_{k∈list} Σ_ij D_k,ij φ*_k,i φ_k,j`
  over the k-list it is handed. Over the IBZ list that is
  `Σ_{k∈IBZ} ρ_k / N_ibz`. The true density is
  `Σ_{k∈IBZ} w_k · ⟨ρ_k⟩_star`, where `⟨·⟩_star` is the average over the star,
  and `ρ_k(r)` for an IBZ point is NOT invariant under the point group
  (`ρ_{Rk}(r) = ρ_k(R⁻¹r)`). The two agree only when every star has size 1.
  Applying `transform_1e_operator` to `vj` afterwards rotates a potential built
  from the wrong density; it does not repair it.
* **K.** `vk(k1) = Σ_{k2 ∈ BZ} K(k1, k2)[D_k2]`. Restricting `k2` to the IBZ
  drops `N − N_ibz` of the terms. Using equivariance to restore them,
  `Σ_{k2∈BZ} K(k1,k2)[D_k2] = Σ_{k2∈IBZ} Σ_{R∈star(k2)} R·K(R⁻¹k1, k2)[D_k2]·Rᴴ`,
  needs `K` evaluated at all `R⁻¹k1`, which for `k1` ranging over the IBZ is
  the whole BZ. **The pair count is `N · N_ibz` either way.** That is exactly
  the count the DFT adapters ALREADY get by passing `kpts_band = kpts_ibz`
  (`krks_ksymm.rs:163`, and `get_k_kpts_opts`'s `k1 in 0..nband` loop at
  `fft_jk.rs:403`).

What `speed_get_jk.py` measured (26.27 s vs 0.12 s) is a full-BZ `get_jk`
against an IBZ-only `get_jk` that computes a **different quantity**. The
realistic bound for exact exchange under symmetry is `N / N_ibz`
(8x on `si [4,4,4]`), not 40x or 223x, and it is a bound the K pair loop's
band route already attains. The plan therefore:

* records this as an **erratum against D-PBC-26 points 1 and 3** (item S-02
  writes it into 17-CONTEXT and the master plan, the way KRKS §9 records its
  errata);
* keeps `veff_fast` only long enough to run its 1e-13 test ONCE and record the
  predicted failure (RULE V: a MODELLED claim is promoted by measuring it);
* replaces it with the band route, which IS bit-exact against the reference
  route for GDF (17-08 measured `max|dvj| = max|dvk| = 0` on
  `gdf_band_route_matches_the_direct_route`) and is expected bit-exact for
  FFTDF because the `k1` loop is outer-independent — to be asserted, not
  assumed.

**Consequence for `KsymAdaptedKrhf` specifically:** its `veff_reference`
calls `get_jk` with `kpts_band: None` (`khf_ksymm.rs:283`) and then
`fold_to_ibz`, i.e. it builds `vk` at all `N` output points and throws
`N − N_ibz` of them away. The HF adapter pays `N²` pairs where the DFT
adapter pays `N · N_ibz`. **Item S-02.**

#### 2.2.4 The XC quadrature under symmetry is still full-BZ-costed (VERIFIED, 17-08 §"The cost consequence")

`nr_rks` / `nr_uks` under `KSet::Ibz` evaluate `eval_ao` at all `N` k-points,
`eval_rho_one` at all `N`, and only the `accumulate_vxc` half at `N_ibz`
(`band`). Memory follows: the AO cache holds `N` tables, the largest resident
allocation in §2.1's last row.

The port already ships the machinery to do better and it is upstream's own:
`KPoints::symmetrize_density(rho_k, ibz_k_idx, mesh)` (`kpts.rs:1657`,
17-05 Task 4, gated) turns one IBZ point's real-space density into its
star-average by grid-index permutation (`rotated_grid_index`, with
`ft_offsets` for the non-symmorphic translations). Then

```text
ρ(r) = Σ_{k∈IBZ} w_k · symmetrize_density(ρ_k)(r)
```

is the full-BZ density from `N_ibz` AO tables and `N_ibz` `eval_rho_one`
calls. This is on a UNIFORM mesh only (the permutation is exact there; Becke
grids are not closed under the group — 17-08 removed exactly such a refusal).

This CHANGES THE RESULT: the star-average sums `|star|` rotated copies where
the full-BZ path sums `|star|` independently evaluated densities; the
rounding differs at the last bits. RULE S: opt-in, gate-scored, and it needs
its own equivalence test against the full-BZ path at 1e-11 (the floor
17-08 measured for `unfolded_ibz_density_equals_full_bz_density`), not 1e-13.
**Item S-03.**

**Expected gain (MODELLED):** `eval_ao` + `eval_rho` are `N/N_ibz` cheaper
and the AO cache is `N/N_ibz` smaller — 8/3 on `si [2,2,2]`, 8 on `[4,4,4]`,
28 on `[16,16,16]`. On a PURE functional this is most of `get_veff`; on a
hybrid it is dwarfed by K. S-00 measures which case the user has.

#### 2.2.5 The DF layer's AO cache is full-BZ under symmetry (VERIFIED)

`Fftdf::ao_kpts` caches `nao · ngrids · 16` bytes per k-point, keyed by the
k-list (`fftdf.rs:185-208`). The K loop needs `ao2` at every `k2 ∈ BZ`
(`fft_jk.rs:387`), so this table cannot shrink without an on-the-fly
`ao_{Rk}(r) = ao_k(R⁻¹r) · D(R)` construction — a grid permutation plus the
17-03 `Dmat` rotation. That is real work with a real bit-parity cost and no
measurement behind it yet. **Item S-05, DEFERRED**, with the trigger stated.

### 2.3 Multigrid — where v1 and v2 spend time and memory

#### 2.3.1 Neither driver is reachable from an SCF, and neither has `nr_uks` (VERIFIED)

`Krks.ni` is a concrete `KNumInt` (`krks.rs:69`); there is no seam through
which `MultiGridNumInt` / `MultiGridNumInt2` can be selected
(17-11 and 17-12 both list this as a carry-over). Both drivers expose
`nr_rks(cell, xc, dm: &[f64])` — real, gamma, one density — and **no
`nr_uks`**, while upstream has `multigrid.py:1166` `nr_uks`. So "KUKS +
multigrid" does not exist in the tree today. **Item M-00** builds the
`nr_uks` port (a completeness item this plan carries because every later
multigrid speed item must be validated on the open-shell path too, per
RULE U) and **M-01** the numint seam.

#### 2.3.2 v1 collocates every level TWICE per call and rebuilds its tasks every call (VERIFIED)

`MultiGridNumInt::nr_rks` (`multigrid/numint.rs:151`):

* `build_tasks` (decontraction + `multi_grids_tasks_for_ke_cut`) — per call,
  depends only on the cell.
* `rho_g_from_levels` → `collocate_level` per level (`:278`), materialising
  `values: (n_slots × ngrids)` f64.
* `pass2_from_full_vg` → `collocate_level` per level AGAIN (`:311`), the same
  values.

`collocate_level` also rebuilds `PeriodicGrids::uniform` and every pshell's
`get_lattice_ls` per call (`colloc.rs:59`, `:113`).

`level_rho` allocates `buf: Vec<f64>` of `terms.len()` **per grid point**
(`colloc.rs:230`) — `ngrids` heap allocations per level per call; upstream's
per-point work has no allocation at all. `level_pass2` allocates a
`ngrids`-long `buf` per `(ci, cj)` entry (`:261`) — `nao_p²` allocations of
`ngrids` each — and sweeps the FULL grid for every entry with no radius
screening.

Memory (MODELLED from the layout): `values` is `n_slots · ngrids · 8` B per
level. `si gth-szv` has 16 pshells; with s+p Cartesian slots that is
`~40 · 42 875 · 8 ≈ 14 MiB` at `35³` — trivial. At 100 AOs and `65³` it is
`~300 · 275k · 8 ≈ 660 MiB`, held once for `rho` and once for `pass2`. v1
does not stream; upstream streams per rcut sub-mesh.

#### 2.3.3 v2 rebuilds its geometry per call and per direction, and launches per block (VERIFIED)

`MultiGridNumInt2::nr_rks` (`pair.rs:1067`):

* `build_tasks` → `build_pair_level_tables` per call: for EVERY pshell pair,
  `get_lattice_ls` (`:559`) and the full binomial-shift / image enumeration
  (`:562-690`). Depends only on the cell. Rebuilt every SCF cycle.
* `pairlevel_rho` and `pairlevel_pass2` each call `grid_blocks(lv)` and
  `block_slots(lv, block)` (`:882-883`, `:935-936`) — pure geometry
  (17-12's own words: "the partition depends only on the mesh") — so the
  block partition and the per-block reach lists are computed **twice per
  level per cycle** and never cached.
* Per block: `block_table` rebuilds the coordinate slice and slot sub-table
  (`:798-848`), then `launch_rho` / `launch_integrate` **upload seven buffers
  and read back one per launch** (`multigrid_pair.rs:597-631`). With
  `BLOCK_EDGE = 5` on a `25³` mesh that is `5³ = 125` launches per level per
  direction, each re-uploading the block's coordinates that `pairlevel_pass2`
  will upload again. 17-12 already identified "per-launch buffer copies" as
  the dominant cost of the first streamed version (130 s → 7-9 s per
  density) and left "batched launches" as its carry-over #3.
* 35-63 % of the dense `(image × monomial × point)` product is still
  evaluated (17-12, MEASURED) because a `5³` block is coarse against a
  per-Gaussian sub-mesh.

This is `11_launch_overhead_and_transfers.md` §2 ("hoist invariant uploads
out of loops"), §3 ("batch read-backs") and §5 ("collapse per-item launches
into one") applied verbatim. **Items M-02, M-03.**

#### 2.3.4 Both multigrid `nr_rks` bodies reduce energies with `.sum()` (VERIFIED)

`ecoul` (`numint.rs:172-176`, `pair.rs:1090-1094`), `nelec` (`:181`, `:1100`)
and `exc` (`:203-207`, `:1123`) are naive `Iterator::sum::<f64>()` over
`ngrids` — the largest naive reductions on any energy path in the tree
(`42 875` terms at `35³`). D-PBC-17 applies; `oracle_sum` is one call away.
**Item P-03.**

#### 2.3.5 What 17-12 measured about where v2's time goes

| stage | share | source |
|---|---|---|
| per-launch upload / read-back | dominant in the first streamed version | 17-12 §"The OOM" |
| dense product still evaluated inside blocks | 35-63 % of the dense count | `pair_level_tables_stream_under_budget` |
| every image re-derives its monomials from one `exp` | unquantified | 17-12 carry-over #3 |

v2 at 0.02x of the reference route means that even a 10x win leaves it
slower than `KNumInt` + `Fftdf` on the reference cells. RULE M: v2 items are
scheduled for `isinstance` fidelity (Phase 18 needs the class) and for the
memory shape, not on the promise of a speed win at this scale.

---

## 3. Work items

Prefix key: **P-** precision/determinism (bit-exact or ordered), **S-**
k-symmetry, **U-** KUKS (numbering continues the KUKS plan), **M-**
multigrid. Each item states FILES, WHY, STEPS, BIT-PARITY, TEST, and what it
is measured against.

### P-00 — Re-run GATE U and GATE B as the KUKS execution summary left them (**do this first; 10 minutes**)

**FILES** none.

**WHY** `KUKS-EXECUTION-SUMMARY.md` §Verification: the four `Li/sto-3g` rows
were re-gated to `5e-11` by arithmetic over energies already produced and
"NOT re-run since the edit"; GATE B was argued, not re-measured, after U-03.
Nothing in this plan may start from an inferred green.

**STEPS**

1. `PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate_openshell -- --ignored --nocapture`
   — expect 9/9 (the Li rows at ≤ 1.5e-11 against 5e-11).
2. `RAYON_NUM_THREADS=1` and `=8` runs of `--test gate --test gate_openshell`
   with `--nocapture`; diff the printed energies byte-for-byte.
3. Record both in `KUKS-EXECUTION-SUMMARY.md` under a dated "P-00" heading.

**BIT-PARITY** n/a. **TEST** the existing files.

---

### P-01 — Determinism coverage for the ksymm drivers and both multigrid `nr_rks`

**FILES** `crates/pyscf-pbc-dft/tests/ksymm_threads.rs` (new),
`crates/pyscf-pbc-dft/tests/multigrid_threads.rs` (new).

**WHY** §1.5. GATE B has never been run on `KsymAdaptedKrks`/`Kuks` or on
either multigrid `nr_rks`; §2.2.2 and §2.3.4 show naive folds on both, so a
thread-dependence is plausible, not hypothetical. Every later item in this
plan is scored against GATE B, so the gate must exist before the items land.

**STEPS**

1. Copy `tests/numint_threads.rs`'s shape: run the target once under
   `RAYON_NUM_THREADS=1` and once under `=8` **in separate processes**
   (rayon's global pool is set once), compare `e_tot` and the returned
   stacks with `to_bits()`.
2. Targets: `KsymAdaptedKrks::kernel` and `KsymAdaptedKuks::kernel` on
   `si [2,2,2]` LDA (the `kuks_ibz_runs_and_stays_symmetric` fixture);
   `MultiGridNumInt::nr_rks` and `MultiGridNumInt2::nr_rks` on
   `small_silicon()` from `multigrid2.rs`.
3. If a target is NOT bit-identical, that is a finding for P-02/P-03, not a
   reason to widen the assertion. Record it and proceed to the P- item.

**BIT-PARITY** n/a (a test). **TEST** the two new files.

---

### P-02 — Ordered reductions and clone removal in the ksymm adapters (**bit-exact on the gated cells, argued as U-03 was**)

**FILES** `crates/pyscf-pbc-dft/src/krks_ksymm.rs`,
`crates/pyscf-pbc-dft/src/veff.rs`.

**WHY** §2.2.2.

**STEPS**

1. Add `veff::weighted_trace_dm_v(dms, v, weights, nao) -> f64` and
   `weighted_trace_dm_v_shared(dms, v_shared, weights, nao)`: per-`(set, k)`
   partial `w[k] · trace_ab(d, v).0` (already ordered), then `oracle_sum` over
   the partials — the exact shape `trace_dm_v` / `trace_dm_v_shared` have,
   with a weight vector. Doc comment cites D-PBC-17 and `krks_ksymm.py:76-81`.
2. Replace `weighted_trace` and `weighted_trace_uks` with calls to them.
   `KsymAdaptedKuks::energy_elec` passes `h1e` as the SHARED stack (no
   `vec![h1e.to_vec(), h1e.to_vec()]`); `get_veff_tagged` passes `jtot` as the
   shared stack (no `vec![jtot.clone(), jtot.clone()]`).
3. In `KsymAdaptedKuks::get_veff_tagged`, take the two `vmat` stacks out of
   the owned `NrKUksResult` (`swap_remove`, as `kuks.rs:220-224` does) instead
   of `nr.vmat[0][0].clone()` / `[1][0].clone()`.
4. Do NOT touch `sandwich_unfold` or the transforms here (S-01).

**BIT-PARITY** For `nao ≤ 11` and `nkpts_ibz ≤ 128`, `oracle_sum`'s base case
is the same left-to-right fold the loops did — bit-identical on every
reference cell (the U-03 argument, re-asserted by the P-01 test). For larger
cells the tree engages and the last bits move in the direction of a smaller
error bound.

**TEST** `crates/pyscf-pbc-dft/tests/ksymm_trace_precision.rs` (new; the
`veff_trace_precision.rs` pattern): ordered vs naive at `nao ∈ {8, 26}`,
`nkpts_ibz ∈ {3, 10}`, with the `nao = 8` row asserted bit-identical. Then
GATE C unchanged (3.1e-14 / 2.8e-14), P-01 green.

---

### P-03 — `oracle_sum` on the multigrid energy path (**bit-parity: NO, documented**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/numint.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`.

**WHY** §2.3.4.

**STEPS**

1. `ecoul`, `nelec`, `exc` in both `nr_rks` bodies: materialise the term
   vector (they already are `map`s over `0..ngrids`) and reduce with
   `oracle_sum`. Three call sites per file.
2. `MultiGridNumInt::get_j` / `MultiGridNumInt2::get_j` have no energy
   reduction; leave them.

**BIT-PARITY** **NO** — `ngrids = 42 875 > PAIRWISE_CHUNK`, so the tree
engages and `exc`/`ecoul` move at the ~1e-15 relative level. GATE E's floors
are 1e-11 (v1) and 1e-7 (v2); the move is invisible there but it is a move,
so the item says so.

**TEST** GATE E unchanged to its printed residuals; P-01's multigrid rows
green (this is what makes them green if P-01 found a thread dependence).

---

### S-00 — The ksymm profiling mode (**the speed prerequisite for every S- item**)

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs`.

**WHY** §2.0: no ksymm number has ever been measured. RULE O forbids landing
S-01…S-03 without a baseline.

**STEPS**

1. `--driver {krks,kuks,krks-ksymm,kuks-ksymm}`. The ksymm drivers take
   `--kmesh-type {mp,gamma-centred}` and build `KPoints` through
   `pyscf_pbc_symm::make_kpts` with `space_group_symmetry = true,
   time_reversal_symmetry = false` (17-07's D-17-07-01 blocks the
   time-reversal + `use_ao_symmetry` combination; say so in `--help`).
2. Time, cold and warm, on the SAME converged density: `unfold_kdms` alone
   (count the calls per `get_veff` — the instrument for S-01), `nr_rks` /
   `nr_uks`, `get_jk` at `kpts_band = ibz`, and the full `kernel()`; report
   `nkpts`, `nkpts_ibz`, `weights_ibz`, and peak RSS via
   `/proc/self/status` `VmHWM` before and after `kernel()`.
3. Report `ksymm_over_full_<stage>` ratios beside `nset2_over_nset1_*`; the
   `--compare` diff covers the new fields.
4. Baseline cells: `si [2,2,2]` (`N_ibz = 3`), `si [4,4,4]` (`N_ibz = 8`),
   both `pbe` and `pbe0`, `mesh 31`. Record in the GATE S ledger.

**DONE** when a table exists with a MEASURED ratio for each stage on an idle
machine (load average printed in the report, as `uptime` prints it).

---

### S-01 — Unfold the IBZ density ONCE per cycle, and cache the rotation matrices (**bit-exact**)

**FILES** `crates/pyscf-pbc-dft/src/krks_ksymm.rs`,
`crates/pyscf-pbc-dft/src/numint.rs`, `crates/pyscf-pbc-symm/src/kpts.rs`,
`crates/pyscf-pbc-symm/src/symmetry.rs`.

**WHY** §2.2.1.

**STEPS**

1. In `KsymAdaptedKrks::get_veff_tagged`: compute `dm_bz` FIRST and pass it to
   `nr_rks`. `unfold_kdms` on full-BZ input is the bit-exact no-op 17-08
   tested, so `nr_rks`'s own unfold becomes free. Same in
   `KsymAdaptedKuks::get_veff_tagged` for both channels (build `sets` from the
   unfolded stacks — and while there, borrow rather than clone; see U-09).
2. `sandwich_unfold`: hoist `get_rotation_mat(cell, k_scaled, nao, op, dmats)`
   into a `OnceLock<Vec<Option<Vec<Complex64>>>>` on `KPoints`, keyed by the
   full-BZ index `k` (the `(k, op)` pair is fixed per `k`). Same pattern as
   `addition_table` / `inverse_table` (`kpts.rs:380-383`). The matrices are
   `nao²` complex per non-identity k — `N · nao² · 16` B, smaller than one AO
   table.
3. Replace the two `CTensor ⇄ Vec<Complex64>` conversions per call with a
   `sandwich_unfold_planar` that reads and writes planar `re/im` directly. The
   arithmetic (`cmatmul`, `kpts.rs:1417`) is unchanged; only the copy goes.
4. Count `transform_dm` calls per `get_veff` in the S-00 instrument: must go
   `2 → 1` (KRKS) and `4 → 2` (KUKS).

**BIT-PARITY** **EXACT.** Step 1 by 17-08's test; step 2 caches a value that
was recomputed identically; step 3 removes copies. If any number moves,
something else changed.

**TEST** `krks_ksymm.rs`'s three unfold tests unchanged; GATE C unchanged;
P-01 green; `kpts_transform.rs` unchanged. Add
`crates/pyscf-pbc-symm/tests/kpts_transform_cache.rs`: cached vs uncached
`transform_dm` bit-identical on `si [2,2,2]` and `[3,3,3]`.

**EXPECTED GAIN** MODELLED: the unfold is `O(N · nao³)` against a quadrature
of `O(N · nao² · ngrids)`, so this is a small wall-time item on the
reference cells and a real one on `nao ≳ 50`; the allocation removal is what
S-00 will actually see. Scheduled first among S- items because it is
bit-exact and it simplifies S-03.

---

### S-02 — Retire `veff_fast`; route `KsymAdaptedKrhf` through `kpts_band = kpts_ibz`; record the D-PBC-26 erratum

**FILES** `crates/pyscf-pbc-scf/src/khf_ksymm.rs`,
`crates/pyscf-pbc-scf/tests/khf_ksymm.rs`,
`.planning/phases/17-ksymm-multigrid/17-CONTEXT.md` §8,
`.planning/pbc/PBC-MASTER-PLAN.md` (the D-PBC-26 entry).

**WHY** §2.2.3.

**STEPS**

1. **Measure the prediction first.** Write 17-07's missing Task 6 test as
   specified there — `veff_fast` vs `veff_reference` at 1e-13, `si [2,2,2]`,
   FFTDF — and run it ONCE. §2.2.3 predicts a failure of order the star
   asymmetry of `ρ_k` (MODELLED: `≥ 1e-3` in `vj` on `si [2,2,2]`, where the
   stars are `[1, 3, 4]`). Record the number. If it PASSES at 1e-13 this
   section's derivation is wrong; stop and re-derive before step 2.
2. Replace `veff_fast`'s body with the band route: `get_jk(&[dm_bz],
   kpts_bz, JkOpts { kpts_band: Some(&kpts_ibz), .. })` — the unfold of the
   density stays (it is what makes the K sum run over the full BZ), the
   output is produced only at the IBZ, and `fold_to_ibz` disappears. Rename
   `JkRoute::Fast` to `JkRoute::Band`. Keep `Reference` as the default until
   step 4.
3. Equivalence test, `Band` vs `Reference`, at 1e-13 **and** at `to_bits()`
   equality: FFTDF and GDF, `si [2,2,2]` and `[1,1,3]`. GDF is expected
   bit-exact (17-08 measured 0 on the DFT side); FFTDF is expected bit-exact
   because `get_k_kpts_opts`'s `k1` loop is independent per `k1` and
   `ewald_exxdiv_for_g0` is applied per band point — assert it, do not argue
   it. If FFTDF is NOT bit-exact, find out which line differs before shipping.
4. Speed: S-00's `get_jk` row, `Band` vs `Reference`, on `si [4,4,4]`
   (`N/N_ibz = 8`). Expected MODELLED ratio for the K-dominated case: close to
   8x for FFTDF (`N·N_ibz` vs `N²` pairs), less for GDF. Flip the default to
   `Band` only when the ratio is measured `> 1.0`, per D-PBC-26 point 6.
5. Write the erratum: D-PBC-26 points 1 and 3 assumed an IBZ-only `get_jk`
   computes the same `vj`/`vk`; §2.2.3 here shows it does not, the measured
   40x/223x compared unlike quantities, and the attainable bound is `N/N_ibz`
   via `kpts_band`, which the DFT adapters already use. Point 4's "KRKS/KUKS
   inherit the route" is therefore already true and needs no code.

**BIT-PARITY** Step 2 is expected EXACT against the reference route; scored
by step 3.

**TEST** `khf_ksymm.rs` gains `band_route_matches_reference_route_bit_exact`
(FFTDF + GDF) and `ibz_only_get_jk_is_not_an_identity` (step 1's recorded
number, asserted `> 1e-6` so nobody re-adopts the route on a loose gate).

---

### S-03 — **Opt-in, changes results:** IBZ-costed XC quadrature via `symmetrize_density`

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`,
`crates/pyscf-pbc-dft/src/krks_ksymm.rs`, `crates/pyscf-pbc-symm/src/kpts.rs`.

**WHY** §2.2.4 — the largest ksymm speed AND memory lever for pure
functionals, and the only item that shrinks the numint AO cache.

**SEQUENCE AFTER S-00 and S-01.** S-00 tells you whether the run is
quadrature- or K-dominated; on a `pbe0` run this item is worth ~nothing.

**STEPS**

1. `KNumInt` gains `rho_route: RhoRoute { Unfold, Symmetrize }`, default
   `Unfold` (the current path). `PYSCF_PBC_KSYMM_RHO=symmetrize` flips it, the
   way `PYSCF_PBC_KK_SYMMETRY` gates W-08.
2. Under `Symmetrize`, `nr_rks` / `nr_uks` take the IBZ-length `dms` as is,
   evaluate `ao2 = eval_ao(cell, chunk, kpts_ibz, ty)` (so the cache holds
   `N_ibz` tables), and form per IBZ point `k`: `ρ_k = eval_rho_one(ao2_k,
   dm_k)`, then `ρ += w_k · symmetrize_density(ρ_k, k, mesh)` for
   component 0. **Gradient components need the rotated vector:** for GGA the
   three `∇ρ` rows transform as `R·∇ρ(R⁻¹r)`, so `symmetrize_density` needs a
   vector variant (`symmetrize_density_vec`, applying the Cartesian rotation
   from `ops[iop]` to the three rows after the index permutation). Port the
   rotation from `symmetry.py`'s `_get_rotation_mat`'s Cartesian block; do not
   invent a convention — the `Dmat` for `l = 1` is already that rotation.
3. **Grid blocking.** `symmetrize_density` permutes indices across the WHOLE
   mesh, so a grid block cannot be symmetrised on its own. Under `Symmetrize`
   the block loop must cover the full mesh for `eval_rho` (one block, or
   accumulate the per-`k` real-space `ρ_k` over blocks first, then
   symmetrise once). Say which in the code; the memory cost is one `ngrids`
   real vector per IBZ point per component, far below the AO table it
   replaces.
4. `accumulate_vxc` is unchanged — it already runs at `band = kpts_ibz`.
5. Refuse `Symmetrize` on a non-uniform grid (`PeriodicGrids::Becke`) with a
   typed error, and on any `KPoints` whose `ft_offsets` fail (non-symmorphic
   translation not commensurate with the mesh — `kpts.rs:1594` already
   refuses).

**BIT-PARITY** **NO** — the star-average sums rotated copies of one
evaluation where the reference path sums `|star|` separate evaluations; the
last bits differ. **Opt-in, default off.**

**TEST** `crates/pyscf-pbc-dft/tests/ksymm_symmetrize_rho.rs` (new):
`Symmetrize` vs `Unfold` `ρ(r)` at **1e-11** on `si [2,2,2]` and `[3,3,3]`
with the tight fixture (`si_precision(1e-10)`, the 17-08 floor), LDA and GGA;
`nelec` to 1e-12; converged `e_tot` to 1e-11 — and, per RULE K, this LAST
comparison is within ONE driver under two routes, not two SCFs of different
symmetry constraint. GATE C re-baselined **separately** for the flag-on run,
recorded, not inherited.

**EXPECTED GAIN** MODELLED: `eval_ao` + `eval_rho` wall time and AO-cache
bytes both `× N_ibz / N` — 3/8 at `[2,2,2]`, 1/8 at `[4,4,4]`. S-00 measures.

---

### S-04 — Cache the pair-independent J/K operands per IBZ output point (**bit-exact, small**)

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`.

**WHY** With `kpts_band = kpts_ibz` the K loop is `k2 ∈ BZ (outer) × k1 ∈
IBZ (inner)`; `ao_dms` is per `k2` (correct) but `coulg_and_expmikr` is
looked up per pair (already cached by W-01) and `vr_dm` is hoisted (U-06
step 5). What is NOT shared is the `build_rho1` result across the two spins
in the `nset` loop — it already is (`:409`, outside `for i in 0..nset`).
**Audit only:** verify by reading that nothing pair-invariant remains inside
the `k1` loop under the band route, and record the answer. If something is
found, hoist it with a bit-exactness argument; if nothing, close the item
with the audit table.

**BIT-PARITY** EXACT or n/a.

---

### S-05 — **DEFERRED:** on-the-fly `ao_{Rk}` from IBZ AO tables in the K loop

**WHY** §2.2.5. The DF AO cache is the one full-BZ allocation S-03 cannot
touch.

**DEFER UNTIL** S-00 shows the `Fftdf` AO cache is the peak-RSS driver on a
case the user actually runs (it is `N · nao · ngrids · 16` B; at `si [4,4,4]
mesh 31` that is 244 MiB — real, but below the numint GGA cache S-03
removes). The construction (`ao_k(R⁻¹r) · D(R)` by grid permutation and the
17-03 `Dmat`) is a new numerical route with its own gate; it must not start
on a guess.

---

### U-09 — Delete the surviving `nset = 2` clones and the `e1` fold in `kuks.rs` (**bit-exact**)

**FILES** `crates/pyscf-pbc-dft/src/kuks.rs`, `crates/pyscf-pbc-dft/src/numint.rs`.

**WHY** §2.1 rows 1-2.

**STEPS**

1. `nr_uks` takes `dms: &[KDms; 2]` by value of two `KDms`; change its
   signature to `dms: [&KMats; 2]` per set — or simplest, keep the type and
   make `veff_from_parts` build `sets` from `Cow::Borrowed` slices. The
   cheapest faithful change: `nr_uks(cell, grids, xc, &[&dms[0], &dms[1]],
   ..)` with `nr_uks` accepting `&[&KMats]` per channel (one set is all the
   SCF ever passes; the multi-set shape is response-only and has no caller —
   `grep nr_uks crates/` before removing it, and keep it if one exists).
2. `energy_elec`'s `e1`: collect the `2·nkpts` `trace_ab` partials and
   `oracle_sum` them — the `krdm::energy_elec` shape U-03 built.
3. `KsymAdaptedKuks::get_veff_tagged` gets the same `sets` fix (it is the
   twin; S-01 step 1 touches the same lines — coordinate so one edit lands).

**BIT-PARITY** EXACT for step 1 and 3; step 2 is bit-identical for
`2·nkpts ≤ 128` (every reference cell) by the U-03 argument.

**TEST** GATE A and GATE U unchanged; `numint_blocking_uks.rs` unchanged.

---

### U-10 — Reusable per-block scratch in `nr_uks` / `eval_rho_one` / `vxc_mat_one` (**bit-exact**; U-06 step 6, U-05 step 2)

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`.

**WHY** §2.1 row 3 — 61 MiB of allocate-and-zero per block per cycle on the
reference cell, `2×` KRKS. This is the item U-06 left open because "interior-
mutable scratch on `KNumInt` whose aliasing story is not free". The story is
simpler than that sentence: the scratch is per CALL, not per `KNumInt`.

**STEPS**

1. Introduce `struct RhoScratch { c0_re, c0_im: Vec<f64>, acc_re, acc_im:
   Vec<f64> }` and `struct VxcScratch { aow_re, aow_im: Vec<f64> }`, owned by
   `nr_rks` / `nr_uks` for the duration of the block loop, sized once at
   `ngrids_max · nao`, passed `&mut` into `eval_rho_one` / `vxc_mat_one`.
   `13_memory_preallocation.md` §"Methods & Approach" is the pattern, applied
   host-side.
2. `eval_rho_one` zero-fills `c0` with `fill(0.0)` instead of allocating —
   the same "overwrite, don't allocate" argument U-06 step 5 used for
   `vr_dm`; `acc_re/acc_im` likewise per component.
3. `vxc_mat_one`'s per-worker `terms_re/terms_im` stay per-worker (rayon
   needs them disjoint) but come from a `par_chunks_mut` over a
   pre-allocated `nao · ngrids` scratch rather than `vec!` per row.
4. The `dena/denb/ta/tb` vectors in `nr_uks` become four reused buffers.
5. **Do NOT fuse the two spins yet** — that is U-05 and it waits for W-06.

**BIT-PARITY** **EXACT.** Every replaced allocation was zero-filled and then
fully overwritten or accumulated from zero; `fill(0.0)` gives the same
starting state. Assert with `numint_blocking_uks.rs` (bit-identity on the
default partition).

**TEST** `numint_blocking.rs`, `numint_blocking_uks.rs`, `numint_threads.rs`
unchanged; GATE A/U unchanged. S-00/U-01 re-profile: `warm_nr_uks_ms` and
peak RSS, before/after.

---

### U-04 / U-05 — unchanged from the KUKS plan

Not restated. §2.1.0a now exists and it **demotes both**: measured at
~30 ms (U-04) and ~30-40 ms (U-05) per cycle on the reference cell, under 1 %
of either a hybrid or a pure SCF. U-04 still waits on the KUKS plan's §8 Q5
(second `get_jk` cost on GDF/RSDF/MDF); U-05 still waits on W-06. Neither
may be scheduled ahead of a §3 item on speed grounds.

---

### M-00 — `nr_uks` for both multigrid drivers (**port item; prerequisite for RULE U on every M- item**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/numint.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`,
`crates/pyscf-pbc-dft/tests/multigrid_uks.rs` (new).

**WHY** §2.3.1. Upstream `multigrid.py:1166-1270` `nr_uks`: `rhoG` for
`dm_a` and `dm_b` (two collocations), `vG` from the SUM, `eval_xc_eff_uks`
on the pair, two `pass2` calls. Without it, "KUKS + multigrid" is a phrase,
not a code path, and no later multigrid item can be validated open-shell.

**STEPS**

1. Refactor both `nr_rks` bodies so the shared middle (`coulG`, `ecoul`,
   the GGA G-space gradient, the XC call, the `wv[0] -= i·G·wv[1:4]` fold,
   the `+ vG` fold) is one function taking `&[CTensor]` of `nset` densities,
   as upstream's `nr_rks`/`nr_uks` share `_eval_rhoG`/`_get_gga_pass2`.
   Bit-identity of the refactored `nr_rks` is asserted before `nr_uks` is
   added on top.
2. Port `nr_uks` line by line (RULE 2): `ecoul` from the spin-summed `rhoG`,
   `exc` from `Σ_s ρ_s · ε`, two `pass2` calls, one per spin `wv`.
3. Open-shell fixture: `li_atom_spin1()` from `tests/common` (all-electron,
   GATE U's fixture) at gamma; gate `nr_uks` against `KNumInt::nr_uks` on the
   same mesh at the v1 floor (1e-11 on `nelec`, `exc`) and the v2 floor
   (1e-6 / 1e-7), the way `gate_e_nr_rks_lda_vs_reference_v2` does.

**BIT-PARITY** the `nr_rks` refactor is EXACT (asserted); `nr_uks` is new.

---

### M-01 — The numint seam: make `MultiGridNumInt{,2}` selectable from `Krks` / `Kuks` at gamma

**FILES** `crates/pyscf-pbc-dft/src/krks.rs`, `kuks.rs`, `numint.rs`,
`crates/pyscf-pbc-dft/tests/multigrid_scf.rs` (new).

**WHY** §2.3.1, 17-11/17-12 carry-over #1. Also the only way RULE M's
"converged SCF ratio" (17-01's 0.18x-0.49x row) can be measured for this
port instead of `get_j` alone.

**STEPS**

1. `enum KsNumInt { Grid(KNumInt), MultiGrid(MultiGridNumInt),
   MultiGrid2(MultiGridNumInt2) }` on `Krks`/`Kuks`; `Krks::multigrid_numint()`
   mirrors `krks.py:284-290`. The multigrid arms are **gamma-only and
   `with_j`-fused** (`xc_with_j`, `kuks_ksymm.py:44-47`'s branch), so
   `veff_from_parts` under a multigrid arm skips `get_jk`'s J and takes
   `ecoul` from the multigrid result, exactly as upstream's `j_in_xc` does.
   Refuse `nkpts > 1`, hybrids (K still needs the DF; upstream allows it —
   port that branch only if a test needs it) and non-uniform grids with
   typed errors.
2. The SCF gate: converged `KRKS(LDA,VWN)` and `KUKS(LDA,VWN)` on
   `small_silicon()` / `li_atom_spin1()`, multigrid arm vs `Grid` arm, at the
   v1/v2 floors — and the wall time of both, printed in the same table
   (RULE M; this is 17-13 Task 2's blank row).

**BIT-PARITY** the `Grid` arm is EXACT (it is the existing code behind an
enum); the multigrid arms are new routes.

---

### M-02 — Cache the multigrid geometry across SCF cycles (**bit-exact**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/numint.rs`, `colloc.rs`,
`pair.rs`.

**WHY** §2.3.2, §2.3.3 — `build_tasks`, `collocate_level`,
`build_pair_level_tables`, `grid_blocks`, `block_slots` are functions of the
cell and the mesh alone and are rebuilt per call, and (v1) per direction.

**STEPS**

1. `MultiGridNumInt` and `MultiGridNumInt2` become stateful:
   `OnceLock<Prepared>` holding `(Decontracted, levels/tables)`, keyed by a
   hash of `(cell.a, cell.mesh, cell.precision, atom coords, basis)` — the
   same idea as `KNumInt`'s `AoKey`. A different cell drops the cache
   (`reset()`, mirroring `KNumInt::reset`).
2. v1: `collocate_level` once per level per `nr_rks`, shared by
   `rho_g_from_levels` and `pass2_from_full_vg` (they are called back to back
   with nothing between them that changes the level). Whether the `values`
   table is also kept ACROSS cycles is a memory decision: keep it only while
   `n_slots · ngrids · 8 · nlevels < 0.25 · max_memory` (the numint rule),
   else recollocate per cycle. Print the decision at `tracing::debug!`.
3. v2: cache `grid_blocks(lv)` and `block_slots(lv, block)` on the
   `PairLevelTable` (`Vec<GridBlock>`, `Vec<Vec<u32>>`) at build time; both
   directions read them. The per-block `block_table` coordinate slice is also
   geometry — precompute `coords` per block once.
4. `get_lattice_ls` per pshell / per pair: memoise by `rcut` within one
   build (the pair loop calls it `npairs` times with `npairs ≫ distinct
   rcut`).

**BIT-PARITY** **EXACT** — every cached value was recomputed identically.
The v2 slot ORDER is unchanged because the tables are the same objects.

**TEST** GATE E unchanged to printed residuals; `eval_rho_g_is_bit_identical_
across_thread_counts_v2` unchanged; new `multigrid_cache.rs`: second call's
output `to_bits()`-equal to the first's, and to an uncached build.

**EXPECTED GAIN** MODELLED, and RULE M says do not guess it: the S-00-style
instrument (M-01's SCF gate prints per-stage times) measures the share of
`build_tasks` + geometry per cycle. 17-12's "7-9 s per density evaluation"
includes the build; the split is unknown.

---

### M-03 — One launch per level: batch the v2 blocks (**bit-exact by construction; verify**)

**FILES** `crates/pyscf-kernels/src/multigrid_pair.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`,
`crates/pyscf-kernels/tests/multigrid_pair.rs`.

**WHY** §2.3.3; `11_launch_overhead_and_transfers.md` §5 "Collapse per-item
launches into one", §2 "Hoist invariant uploads out of loops", §3 "Batch
read-backs". 125 launches per level per direction, each uploading seven
buffers, on a CPU backend where every launch is a full round trip
(memory: `pyscf-algebra-cpu-is-default-backend`).

**READ FIRST** (RULE 5): `INDEX.md`, `11_launch_overhead_and_transfers.md`,
`10_grid_stride_occupancy.md` §2-3 (the per-lane work is now uneven across
blocks), `Cubecl_conditionals.md` (no per-lane `if` on the block id — use a
block-offset table), `13_memory_preallocation.md`.

**STEPS**

1. Kernel side: `collocate_pairs_rho_batched` takes the CONCATENATED block
   tables plus `block_point0: Array<u32>` (prefix over points) and
   `block_slot0: Array<u32>` (prefix over kernel slots), one lane per grid
   point of the concatenation; a lane finds its block by a comptime-unrolled
   or binary search over `block_point0` (a `u32` array of ~125 entries —
   `01_loop_unrolling.md` if unrolled, else a plain loop; measure both, RULE
   O) and then runs today's per-point loop over
   `block_slot0[b]..block_slot0[b+1]`. The in-kernel sum order per point is
   UNCHANGED (same slots, same table order), so the result is bit-identical
   to the per-block launch by construction — assert it.
2. `collocate_pairs_integrate_batched` likewise, one lane per (block, kernel
   slot) pair via the same prefix tables; per-block partials are summed on
   the host in block order exactly as `pairlevel_pass2` does now (`kint[sel[j]]
   += v` in `grid_blocks` order) so the fixed-order property (D-PBC-17 shape)
   survives.
3. Upload the concatenated coordinates ONCE per level per cycle (they are
   M-02's cached geometry) and reuse the handle for `rho` and `integrate`;
   per cycle only `slot_coef` (density-dependent) and the weight field
   change. `11_launch_overhead_and_transfers.md` §2.
4. Keep the per-block path behind the existing functions; the batched path is
   selected by `pairlevel_rho`/`pairlevel_pass2` when the concatenated table
   fits the 256 MiB launch budget `pair_level_tables_stream_under_budget`
   asserts (largest observed 42 MiB per block; the concatenation is the sum —
   measure it, and fall back to per-block streaming above the budget).
5. **FMA:** VERIFIED this session — `xtask/src/bin/check_no_fma.rs:81-104`
   scans `pyscf-algebra`, `pyscf-core`, `pyscf-ccsd`, `pyscf-pbc-gto`,
   `pyscf-pbc-df`, `pyscf-pbc-tools`; **`pyscf-kernels` is NOT scanned**, and
   `pyscf-pbc-dft` is excluded on a documented rustc segfault under
   `codegen-units = 1`. Add `("pyscf-kernels", "pyscf_kernels")` with this
   item (the multigrid kernels reach `ecoul`/`exc` directly), per KUKS §1.5's
   standing obligation; if the same `libxc_rs` build crash blocks it, record
   that at the list the way the `pyscf-pbc-dft` entry does.

**BIT-PARITY** **EXACT** against the per-block route — asserted at
`to_bits()` in the kernel test, not argued.

**TEST** `pyscf-kernels/tests/multigrid_pair.rs`: batched vs per-block
`to_bits()`-identical on the existing fixtures; `multigrid2.rs` GATE E
unchanged; thread-count bit-identity unchanged. Speed row in the GATE S
ledger: `pairlevel_rho` + `pairlevel_pass2` wall time per level before/after,
and `get_j` v2 vs reference (17-12's 0.023x / 0.028x is the baseline).

**EXPECTED GAIN** MODELLED: 17-12 attributed the first streamed version's
130 s → 7-9 s to buffer copies; a further ~125x reduction in launch count
plausibly moves v2 from 0.02x toward 0.1-0.3x of the reference. **Not a
promise; RULE M.**

---

### M-04 — v1 host loops: no per-point allocation, radius-screened `pass2` (**bit-exact for `level_rho`; `pass2` screening changes results — flagged**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/colloc.rs`.

**WHY** §2.3.2. v1 is the multigrid route that is actually FASTER than the
reference on `get_j` (17-12: 1.5x / 1.2x) and exact to 1e-11, so it is the
one worth making cheap.

**STEPS**

1. `level_rho`: `par_chunks_mut(RHO_CHUNK)` over grid points with ONE `buf`
   per chunk, `oracle_sum` per point unchanged (the term list is fixed, so
   this is the `eval_rho_one` W-06 idiom). Bit-exact: same terms, same
   order, same reducer.
2. `level_pass2`: one `buf` per worker (rayon over `(ci, cj)` rows), not per
   entry. Bit-exact.
3. **Opt-in, changes results:** skip grid points outside `rcut_i + rcut_j`
   of the pair's centres in `level_pass2` (upstream's sub-mesh does exactly
   this — `multigrid.py`'s `_get_j_pass2` restricts to the task's `rcut`
   sub-block). The dropped terms are below `precision · EXTRA_PREC` by
   construction of `rcut`, but their omission moves the sum. Flag
   `PYSCF_PBC_MULTIGRID_PASS2_SCREEN`, default off; gate-scored against
   GATE E's v1 floor.

**BIT-PARITY** steps 1-2 EXACT; step 3 NO, opt-in.

**TEST** `tests/multigrid.rs::int_rho_matches_tr_dm_s` unchanged; new
`multigrid_pass2_screen.rs` for step 3 with its own recorded residual.

---

### M-05 — **DEFERRED:** per-Gaussian sub-mesh and Hermite recursion in v2

17-12 carry-over #3. Only after M-02 and M-03 have been measured, and only if
Phase 18 needs v2 for more than `isinstance` (RULE M). The trigger is a
measured `get_j` ratio still below 0.3x after M-03.

---

## 4. Sequencing

```text
P-00 ──► P-01 ──► P-02 ──► P-03
                    │
S-00 ───────────────┼──► S-01 ──► S-02 ──► S-03 (opt-in)   S-04 (audit)   S-05 (deferred)
                    │
U-09 ──► U-10 ──────┘        (U-04 / U-05 per the KUKS plan: after §2.1.0a, W-06)

M-00 ──► M-01 ──► M-02 ──► M-03 ──► M-04        M-05 (deferred)
```

* **P-00 lands alone and first** — it is a re-run, not a change, and every
  later item's "unchanged" claim is measured against it.
* **P-01 before any P-02/P-03/S-/M- item** — the determinism gate must exist
  before items that could break it land.
* **Bit-exact items land in pairs at most**, each with its own GATE A/U/C/E
  print; two bit-exact items cannot hide each other's movement because
  neither may move anything.
* **S-03 and M-04 step 3 land ALONE**, behind their flags, with a separate
  re-baselined gate print each — the W-08 discipline.
* **The three tracks (P/S, U, M) are independent** and may be executed by
  different sessions; the only cross-track edge is S-01 step 1 ↔ U-09 step 3
  (same lines in `KsymAdaptedKuks::get_veff_tagged`).

---

## 5. Verification protocol — run after EVERY work item

```bash
# 1. GATE A — closed-shell accuracy (oracle)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
# 2. GATE U — open-shell accuracy (oracle)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate_openshell -- --ignored --nocapture
# 3. GATE B — thread-count bit-identity (D-PBC-17), incl. P-01's new files
for n in 1 8; do RAYON_NUM_THREADS=$n cargo test -p pyscf-pbc-dft --release \
  --test numint_threads --test ksymm_threads --test multigrid_threads -- --nocapture; done
# 4. GATE C — ksymm port-vs-port (FFTDF rows; the GDF row is #[ignore]d and known-failing)
cargo test -p pyscf-pbc-dft --release --test krks_ksymm -- --test-threads=1
cargo test -p pyscf-pbc-scf --release --test khf_ksymm -- --test-threads=1
# 5. GATE E — multigrid
cargo test -p pyscf-kernels --release --test multigrid_pair
cargo test -p pyscf-pbc-dft --release --test multigrid --test multigrid2 -- --test-threads=1
# 6. The lints
cargo run -p xtask --bin check-dependency-wall
cargo run -p xtask --bin check-orphan-modules
cargo run -p xtask --bin check-no-fma
# 7. Everything downstream of the touched crates
cargo test --release -p pyscf-pbc-symm -p pyscf-pbc-scf -p pyscf-pbc-dft -p pyscf-pbc-df -p pyscf-kernels -p pyscf-bench
# 8. Re-profile — ONE variable changed since the last run; machine idle (print `uptime`)
cargo run -p pyscf-bench --release --bin krks_profile -- jk --driver <..> --cell si --nk 2,2,2 --mesh 31,31,31 --xc <..> --json after.json --compare before.json
```

Record the GATE S ledger row (§1.8) before moving to the next item. A row
without a peak-RSS number is incomplete for any item whose heading says
"memory".

---

## 6. Risks

| risk | mitigation |
|---|---|
| S-02 step 1 PASSES (the derivation in §2.2.3 is wrong) | stop; the erratum is not written; re-derive with the measured `vj` difference in hand before touching `veff_fast` |
| FFTDF band route is NOT bit-exact against the full route (S-02 step 3) | find the line (`ewald_exxdiv_for_g0` ordering, or `real_out` differing between band and full) before shipping; do not relax to 1e-13 silently |
| S-03's GGA rotation convention is wrong | the 1e-11 `ρ` and `∇ρ` comparison against the `Unfold` route catches a wrong `Dmat` sign immediately on `si` (non-symmorphic, stars `[1,3,4]`); do not test on `he_fcc` alone (symmorphic, Γ-heavy) |
| M-03's batched kernel changes the per-point sum order | asserted `to_bits()`-identical to the per-block launch; a mismatch is a bug in the prefix tables, not a tolerance problem |
| M-02 keeps a `values` table that no longer fits | the `0.25 · max_memory` rule with a `tracing::debug!` of the decision; `pair_level_tables_stream_under_budget` extended to v1 |
| the machine is contended when a ratio is measured | RULE O — `uptime` is printed in every report; a load average above ~4 on 16 cores invalidates the row |
| cubecl build error while touching `multigrid_pair.rs` | AGENTS.md §4 — read `cubecl_error_guideline.md` first, no blind fixes |
| a ksymm energy gate is compared across symmetry constraints | RULE K — assert `check_mo_occ_symmetry` on the full-BZ side or compare two routes in one driver |

---

## 7. CubeCL manual sections this plan depends on

All under `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`. Read
before the item that names them (RULE 5).

| section | item |
|---|---|
| `INDEX.md` | every M- item touching `pyscf-kernels` |
| `11_launch_overhead_and_transfers.md` §2 hoist uploads, §3 batch read-backs, §5 collapse per-item launches, §6 re-attribute after every change | M-03; §6 is RULE O's source |
| `13_memory_preallocation.md` | U-10 (host-side application), M-02, M-03 step 3 |
| `10_grid_stride_occupancy.md` §2-3 | M-03 (uneven per-lane work across blocks) |
| `Cubecl_conditionals.md`, `plane_alignment.md` | M-03 (no per-lane branch on the block id) |
| `01_loop_unrolling.md` | M-03's block lookup, if unrolled |
| `03_kernel_fusion.md` §4 | U-05 (inherited from the KUKS plan), not started here |
| `16_profiling_and_bottleneck_identification.md` §3-5 | S-00 / M-01's per-stage attribution |
| `cubecl_error_guideline.md` (`../cubecl_error_guideline.md`) | any build failure |

---

## 8. Open questions

| # | question | who answers |
|---|---|---|
| Q1 | ~~What is the KUKS/KRKS multiplier on an idle machine, pure and hybrid?~~ **ANSWERED** — K ×1.03, J ×1.9-2.2, quadrature ×1.3-1.75 (§2.1.0a) | closed |
| Q2 | What fraction of a ksymm `get_veff` is the unfold, on `si [4,4,4]`? | S-00 |
| Q3 | Is the FFTDF band route bit-exact against the full route? (GDF: MEASURED yes) | S-02 step 3 |
| Q4 | Does the `Symmetrize` rho route reach 1e-11 against `Unfold` with the tight fixture, GGA included? | S-03 test |
| Q5 | Share of `build_tasks` + geometry in a v1/v2 cycle | M-01's instrument, before M-02 |
| Q6 | Does the batched v2 launch fit the 256 MiB budget on the reference cells? | M-03 step 4 |
| Q7 | ~~Is `pyscf-kernels` in `check-no-fma`'s `SCAN_TARGETS`?~~ **ANSWERED: no** (`check_no_fma.rs:81-104`); M-03 step 5 adds it | closed |
| Q8 | Which multigrid driver does Phase 18 actually need beyond `isinstance`? | Phase 18's context; gates M-05 |
