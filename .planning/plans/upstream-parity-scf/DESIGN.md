---
title: Upstream parity of the periodic SCF — non-local PP route, level shift, final diagonalisation
status: draft
updated_at: 2026-10-02
evidence: .yta_bundle/upcheck/ktao3_cpu/upstream_check.json (KTaO3, PySCF 2.12.1)
---

> **BACKGROUND ONLY — do not execute from this file.** The executable plan is
> `README.md` + `tasks/T*.md`. Where this file and a task file disagree, the
> task file wins. Four things were simplified after this was written:
> K-PP1 is a dedicated kernel (no K-15 reuse); K-PP4 runs on the host from a
> 5 MB read-back; the level-shift products use the existing `zgemm_dense`
> (no handle-level wrapper); there is no GEMM alternative for the fold.

# PLAN — three differences between the port and stock PySCF

## 0. What was measured (2026-10-02)

The YTa7O19 recipe was run on KTaO3 (5 atoms, `gth-szv-molopt-sr` →
`gth-dzvp-molopt-sr`, `gth-pbe`, `ke_cutoff` 60 Ha → mesh 29³, 3×3×3 k-mesh,
PBE; gauss 0.03 / damp 0.7 / shift 0.3 / DIIS 8,16 → gauss 0.002 / shift 0.1)
through `yta7o19_bands` (CPU backend, batching forced) and through upstream
PySCF 2.12.1 (`tools/kaggle-t4/upstream_check.py`).

| quantity | port vs stock PySCF | port vs PySCF with D1–D3 removed |
|---|---|---|
| `e_tot`, final stage | 1.88e-3 Ha | 5.4e-10 Ha |
| band energies, max | 3.5e-3 Ha | 4.9e-6 Ha (SCF tolerance 1e-6) |
| band gap | 3.3e-3 eV | 1.7e-5 eV |
| cycles `pre` / `s1` / `s2` | 12/16, 13/22, 3/6 | 12/13, 13/14, 3/3 |
| stage reported converged (`pre`, `s1`, `s2`) | port T,T,T — PySCF **F,F,T** | T,T,T both |

At one density: overlap 9e-15, `E_J + E_xc` 1e-13, and `H_core` against
`T + V_loc(grid) + pp_int.get_pp_nl` 4e-13. Everything that is not D1–D3
already agrees.

| id | difference | port | upstream | effect |
|---|---|---|---|---|
| D1 | non-local pseudopotential under FFTDF | real space, analytic (`pyscf_pbc_gto::pseudo::get_pp_nl`) — `fftdf.rs:902 vpp_add_nonlocal`, documented deviation in the module header | reciprocal space on the FFT mesh — `pbc/df/fft.py:114-176` | changes the answer at a low cutoff: 3.9e-4 Ha per O 2s element at 60 Ha; zero at a converged mesh |
| D2 | level shift | `kscf.rs:54 level_shift` uses `S − ½·S·D·S` for every driver | restricted: FULL density (`khf.py:157`) → occupied −shift, virtual +shift; unrestricted: per-spin density (`kuhf.py:99-101`); ROHF: `dm_sf·½` (`krohf.py:80`) | convergence path; with smearing, also the occupations along it |
| D3 | final step | `kscf.rs:209-220` stops at the converged cycle | `scf/hf.py:211-232`: re-diagonalise the bare Fock, new occupations/density/energy, re-test convergence at 10×/3× | reported `e_tot`, `mo_energy` (port's are level-shifted), `mo_occ`, `dm`, `converged` |

Not in this plan (recorded so it is not lost): the MINAO guess differs
(init E −134.971 port vs −135.380 upstream on the SZV cell). Path only. T0
captures the evidence; the fix is its own item.

## 1. Constraints

- **C1 — a T4 must still run it in batches.** 15 GB device memory, ~31 GB
  host, 12 h sessions. Nothing new may hold a `(ngrids, nao)` table for the
  whole grid: YTa7O19 DZVP is 249 615 × 910 × 16 B = 3.6 GB **per k-point**.
  Every new stage takes an MB budget from the environment, like
  `PYSCF_PBC_NUMINT_AO_BUDGET_MB`, and must give the same matrix (to
  rounding) at any budget.
- **C2 — checkpoints survive.** A run started before this change resumes, or
  is refused with a reason; it is never silently mixed (`fingerprint.json`,
  stage identity tokens).
- **C3 — repo rules.** Tests in separate files (AGENTS §2). Every new CubeCL
  kernel is generic over the float type (`F: Float`, launched as `f64`) and
  written after reading the CubeCL manual (AGENTS §3); a build failure follows
  the error guide (AGENTS §4). One named-target cargo call per test batch
  (libxc rebuild cost).
- **C5 — the new work runs ON THE DEVICE.** The G-space pseudopotential and
  the level-shift products are CubeCL kernels on resident buffers (§T3, §T1);
  the host sees only `nao × nao` answers. The same kernels run on the CPU
  runtime (the only backend available locally — the ROCm iGPU has no f64) and
  on CUDA (T4 / RTX PRO 6000, with the patched `cubecl-cpp` F64).
- **C4 — oracle discipline.** Each fix is gated against live PySCF 2.12.1 on
  an odd k-mesh with complex k-points, at a low AND at the default cutoff.

## 2. Order

```
T0 baseline + gates-first      (no behaviour change)
T1 D2 level shift              (smallest, isolated)
T2 D3 final diagonalisation    (driver + pipeline checkpoint contract)
T3 D1 G-space non-local PP     (new batched stage; largest)
T4 pipeline + ktrun: knobs, fingerprint, migration, recipe
T5 validation: KTaO3 local CPU → KTaO3 on a T4 → YTa7O19 re-run decision
```

T1 and T2 touch the same function (`kscf::kernel`), so they are serial. T3 is
independent of both and can be developed in parallel; T4 needs all three.

---

## T0 — Baseline and red gates

Goal: every later task turns a failing upstream gate green, and nothing else
moves unnoticed.

1. **Freeze the baseline.** Keep `target/release/examples/yta7o19_bands`
   (built 2026-10-02 15:33) as the BEFORE binary and
   `.yta_bundle/upcheck/ktao3_cpu/` as the BEFORE result.
2. **Inventory what will move.** One run, results saved to
   `.planning/plans/upstream-parity-scf/baseline.txt`:
   `cargo test --release -p pyscf-pbc-scf -p pyscf-pbc-dft -p pyscf-pbc-df`
   with the test targets named, plus `python/pyscf/tests/test_pbc_scf.py`.
   List every test that passes a non-zero `level_shift`
   (`krks_dzvp_small_cell.rs`, `test_pbc_scf.py`, the `pyscf-py` bindings)
   and every test that pins an SCF energy or `mo_energy` to a literal.
3. **Three red oracle tests** (ignored unless `PYSCF_ORACLE_VENV` is set, the
   existing pattern of `scf_ovlp_smearing_oracle.rs`), all on the KTaO3 SZV
   cell, 3×3×1 so they run in about a minute:
   - `crates/pyscf-pbc-scf/tests/level_shift_oracle.rs` — one Fock matrix,
     one density: `level_shift` output against upstream `khf.get_fock`,
     `kuhf.get_fock`, `krohf.get_fock`, `kghf` (element-wise, 1e-12). Red
     today for KRHF, KUHF, KGHF; green for KROHF.
     The KUHF/KGHF/KROHF verdicts are from reading the code, not measured —
     this test is what confirms them.
   - `crates/pyscf-pbc-dft/tests/kscf_conv_check_oracle.rs` — KRKS with
     shift 0.3 and gauss 0.03 from a fixed `dm0` exported by upstream:
     `e_tot`, `mo_energy`, `mo_occ`, `converged` against stock upstream
     (`conv_check=True`). Red today.
   - `crates/pyscf-pbc-df/tests/fftdf_pp_gspace_oracle.rs` — `Fftdf::get_pp`
     at `ke_cutoff` 60 against `fft.FFTDF.get_pp` (1e-11), at Γ and at a
     complex k-point. Red today at 3.9e-4. The existing `tests/fftdf.rs`
     gate (mesh ≥ 31, where upstream is converged) stays as it is.
4. **MINAO evidence.** Add the initial density's trace per atom to the
   `init E` log line of the example (no behaviour change) and record port vs
   upstream on KTaO3. No fix here.

Done when: the three tests exist, fail for the stated reason, and
`baseline.txt` is committed.

---

## T1 — D2: level shift

**Upstream, exactly.** `mol_hf.level_shift(s, d, f, factor) = f + (s − s·d·s)·factor`
(`scf/hf.py:794`). What differs per driver is the `d` handed in:

| driver | `d` | occupied | virtual |
|---|---|---|---|
| KRHF / KRKS (`khf.py:157`) | `dm_kpts[k]` (occupation 2) | −shift | +shift |
| KUHF / KUKS (`kuhf.py:99-101`) | `dm_kpts[σ][k]` (occupation 1) | 0 | +shift |
| KGHF / KGKS (inherits `khf.get_fock`) | `dm_kpts[k]` (occupation 1) | 0 | +shift |
| KROHF (`krohf.py:80`) | `(dmα+dmβ)[k]·½` | 0 (doubly), +½ (singly) | +shift |

The restricted row is an upstream quirk (the molecular RHF passes `dm·½`,
`scf/hf.py:1133`); the port follows upstream, quirk included.

**Change.**
1. `khooks.rs`: new defaulted hook
   `fn level_shift_dms(&self, dms: &KDms) -> KDms { dms.clone() }` — the
   density `level_shift` receives. Default = the density channels themselves
   (KRHF, KRKS, KUHF, KUKS, KGHF, KGKS, and the k-symmetry driver).
2. `krohf.rs`: override → `(dmα+dmβ)·½`, one channel (its Fock is the single
   Roothaan matrix).
3. `kscf.rs:54`: `level_shift` becomes `F += factor·(S − S·D·S)` — the `0.5`
   goes. Call site (`kscf.rs:178`) uses `hooks.level_shift_dms(&dm)` instead
   of `hooks.diis_dms(&dm)`.
4. Fix the doc comment on `level_shift` (it cites `khf.py:155-157` but
   describes the molecular RHF form).
5. Optional, same task if cheap: `(shift_a, shift_b)` for KUHF
   (`kuhf.py:80-83`). `KScfConfig::level_shift` stays `f64`; a second field
   `level_shift_beta: Option<f64>` only if the Python binding needs it.

Every implementor of `KOverrideHooks` must be checked for an override need:
`krhf.rs`, `kuhf.rs`, `krohf.rs`, `kghf.rs`, `khf_ksymm.rs`, `krks.rs`,
`kuks.rs`, `kgks.rs`, and the Python-side hook wrappers in
`pyscf-py/src/pbc/scf.rs`.

**On-device kernel (K-LS).** Today `kscf.rs::mm` is a scalar host triple loop:
`S·D` and `(S·D)·S` are two `nao³` complex products per k-point per cycle —
for YTa7O19 DZVP, 9 × 2 × 910³ ≈ 1.4e10 complex multiply-adds on one host
thread, every cycle. Replace it:

1. `crates/pyscf-kernels/src/pbc/level_shift.rs` (new):
   `level_shift_apply_kernel<F: Float>(f_re, f_im, s_re, s_im, sds_re, sds_im,
   factor, n)` — one lane per matrix element, `f += factor·(s − sds)` on
   planar buffers. Closed under `Float`, no transcendental calls.
2. The two products use `pyscf_algebra::gemm::launch_gemm_on_handles` on
   RESIDENT planes (four real GEMMs per complex product, in D-PBC-03's order,
   as `zgemm_dense` does — but without `zgemm_dense`'s upload/read-back per
   call). `launch_gemm_on_handles` is `pub(crate)`; it gets a public
   handle-level wrapper `zgemm_on_handles` in `pyscf-algebra/src/zgemm.rs`.
3. Transfers (manual §11, "hoist invariant uploads", "batch read-backs"):
   `S_k` is uploaded once per SCF and stays resident; `D_k` and `F_k` go up
   once per cycle; the shifted `F` comes back in ONE batched read for all
   k-points (the eigensolver is host-side).
4. Selection: device route when `nao ≥ 128`, host loop below (launch overhead
   dominates tiny matrices). The threshold is set from a measurement on the
   CPU runtime and on a T4, not guessed — record both.
5. Test `crates/pyscf-kernels/tests/level_shift_device.rs`: device vs the
   host loop, 1e-13, at Γ (real) and a complex k-point, `nao` 27 and 79.

First measure: the `scf_get_fock` span of the YTa7O19 log already contains
this cost; quote it before and after.

**What moves.** Only runs with `level_shift ≠ 0`. For a restricted run the old
behaviour at shift `s` is the new behaviour at `s/2` minus `(s/2)·S` — same
orbitals, same occupations, every orbital energy lower by `s/2`. That
identity is the regression test:
`crates/pyscf-pbc-scf/tests/level_shift_half_identity.rs` — new code at
`s/2` reproduces the BEFORE binary's cycle energies at `s` (1e-10) on the
small TaO cell.

**Gates.** `level_shift_oracle.rs` green for all four drivers.
`krks_dzvp_small_cell.rs` still converges (retune its shift to half if its
cycle count is asserted).

**Risk.** Convergence of level-shifted restricted runs changes: on KTaO3
upstream semantics took 16 and 22 cycles where the port took 12 and 13. T4
handles the recipe (T4.3).

---

## T2 — D3: the final diagonalisation (`conv_check`)

**Upstream, exactly** (`scf/hf.py:211-232`), after the loop and only if it
converged and `conv_check` is on:

```
eps, C  = eig(fock, s1e)          # fock = BARE Fock of the latest density
occ     = get_occ(eps, C)         # smearing re-assigns occupations here
dm_last, dm = dm, make_rdm1(C, occ)
vhf     = get_veff(dm, dm_last, vhf)
last_e, e_tot = e_tot, energy_tot(dm, h1e, vhf)
fock    = get_fock(h1e, s1e, vhf, dm)            # bare
norm_gorb = |get_grad(C, occ, fock)|
converged = |e_tot − last_e| < 10·conv_tol  OR  norm_gorb < 3·conv_tol_grad
```

`mf.cycles` is NOT incremented by this step.

**Change — driver (`kscf.rs`).**
1. `KScfConfig::conv_check: bool`, default `true` (`types.rs`; upstream
   default `scf_hf_SCF_conv_check = True`).
2. After the loop, when `converged && cfg.conv_check`: the block above with
   the existing hooks (`get_fock`, `eig`, `get_occ`, `make_rdm1`, `get_veff`,
   `energy_elec`, `get_grad`). Note the test is **OR**, with the loosened
   thresholds.
3. `e_free` / `e_zero` are taken after this block, so `hooks.free_energy()`
   reflects the final occupations (it already reads the last `get_occ`).
4. `max_cycle = 0` branch unchanged.

**Device work in the final step.** No new kernel: the block is one more
`get_veff` (the existing device J and XC kernels, already grid-blocked by
`PYSCF_PBC_NUMINT_AO_BUDGET_MB` / `PYSCF_PBC_FFTJK_AO_BUDGET_MB`), one
host-side `eig`, and one `get_grad` (`fock_times_columns`). On a T4 it costs
one SCF cycle (55 min for YTa7O19 DZVP), so it must be checkpointed like a
cycle (below) and it must run before `release_device_pool()` drops the AO
tables the cycle loop built.

**Change — cycle hook and resume.** Today the hook fires once per cycle with
`converged`, and the pipeline treats `converged = true` as "stage finished,
this is its density". With `conv_check` the density changes after that.
1. `CycleState` gains `final_state: bool`.
2. In the loop the hook reports the converging cycle with `converged: false`
   whenever `cfg.conv_check` is on. A session killed right there resumes with
   one more ordinary cycle — safe.
3. After the block the hook fires once more: same `cycle`,
   `final_state: true`, the final `dm`, `e_tot`, and `converged` = the
   re-tested value; `fock` = the bare Fock just built.
4. `kscf_resume.rs` gets a case: kill between the converging cycle and the
   final state, resume, and the result equals the uninterrupted run
   (bit-identical while DIIS has not started, 1e-10 after).

**Change — results.** `KScfResult::mo_energy` are now un-shifted, so the
pipeline's `homo_ha` / `lumo_ha` stop reporting the level-shifted virtuals
(the YTa7O19 `scf_stage1.lumo_ha = 0.7227` artefact). Check every consumer
that reads `mo_energy` / `mo_occ` / `dm` after a level-shifted SCF:
`pyscf-pbc-cc` (`kccsd*.rs`, `rccsd.rs`, `uccsd.rs`, `gccsd.rs` carry their
own `level_shift` for the CC equations — unrelated, but they take SCF
orbital energies as input), `pyscf-pbc-grad`, `tdscf`, `stability.rs`,
`newton_ah.rs`, and `pyscf-py/src/pbc/scf.rs` (expose `conv_check` as an
attribute, default `True`).

**What moves.** Every converged periodic SCF: `e_tot` by O(conv_tol), and
`converged` can flip to `false` (KTaO3: stock PySCF reports `pre` and `s1`
unconverged with shift 0.3 + gauss 0.03, because re-diagonalising without the
shift re-smears the occupations). Literal-pinned energies in the tests from
T0.2 are re-derived from upstream, not from the new output.

**Gates.** `kscf_conv_check_oracle.rs` green. The existing upstream gates
(Al smearing 1e-7, Si KRHF, Y/Ta bands) must not get worse — they compare
against upstream runs that always had the extra step, so they should tighten.

---

## T3 — D1: the reciprocal-space non-local pseudopotential, in batches

**Upstream, exactly** (`pbc/df/fft.py:114-176`), per k-point:

```
Gk    = Gv + kpt                              (ngrids, 3)
aokG  = ft_ao(cell, Gv, kpt) / sqrt(vol)      (ngrids, nao)   analytic FT of the AOs
for atom a, channel l with nl > 0, i < nl:
    pYlm[i, m, g] = GTOval_l(Gk; exp ½·rl², coeff rl^(l+1.5)·π^1.25)[g, m] · qli(|Gk|·rl, l, i)
    SPG[i, m, g]  = conj(SI[a, g]) · pYlm[i, m, g]
B_a[i, m, p]      = Σ_g SPG[i, m, g] · aokG[g, p]
vppnl[p, q]      += Σ_{i,j,m} conj(B_a[i, m, p]) · hl[i, j] · B_a[j, m, q]
vppnl /= vol;   Γ: keep the real part
```

Pieces that already exist: `ft_ao::single::ft_ao_kpt` (analytic FT at
`Gv + kpt`, no lattice sum — the same call upstream makes), `get_gv`,
`get_si`, the `hl` blocks in `pyscf_pbc_gto::pseudo`. Missing: `_qli`
(`pp.py:150-200`, closed-form polynomials) and the one-shell real solid
harmonic Gaussian at `Gk`.

**Design: on the device, in G-blocks (C1, C5).** The sum over `g` is a plain
accumulation. The grid is walked in blocks; the block tables live on the
device; only the `nao × nao` result is read back.

```
P = Σ_atoms Σ_l nl·(2l+1)             # YTa7O19: 2·18 + 14·19 + 38·1 = 340
resident for the whole call:  Gv (ngrids, 3) · SI (natm, ngrids) · shell and projector tables
resident per k-point:         B[P, nao] (5 MB) · vpp_k[nao, nao]
for G-block [g0, g1) of nb points:
    K-PP1  A[nao, nb]  = ft_ao(Gv[g0:g1] + k) / sqrt(vol)        AO-major planes
    K-PP2  S[P, nb]    = conj(SI[a]) · pYlm · qli                projector planes
    K-PP3  B[p, q]    += Σ_g S[p, g] · A[q, g]                   fused fold, accumulate
K-PP4      vpp_k      += B^H · (h ∘ B) / vol                     Γ: imaginary plane zeroed
read back vpp_k                                                  the only transfer
```

Block size from `PYSCF_PBC_PP_NL_BUDGET_MB` (default 1024):
`nb = budget / (16·(nao + P))`. YTa7O19 DZVP at 1 GB: `nb ≈ 52 000`, five
blocks per k-point; device peak = budget + SI (216 MB) + 20 MB, independent
of the grid. A T4 (15 GB) runs it next to the AO cache, and the 9 + 66
k-point band job uses the same tables.

**The four kernels** (`crates/pyscf-kernels/src/pbc/pp_gspace.rs`, new; all
`#[cube(launch_unchecked)]`, generic `F: Float`, planar re/im buffers —
D-PBC-02):

- **K-PP1 — AO transform block.** Upstream's `ft_ao` IS `ft_aopair` against a
  ghost `s` shell of exponent 0 (`gto/ft_ao.py:175-180`). First choice:
  drive the existing K-15 kernel (`pbc/ft_aopair.rs`) with that ghost partner
  and no lattice images, writing straight into the AO-major block — no new
  arithmetic to validate. To verify before coding: that
  `FtAopairTables` can express a zero-exponent partner and a single image. If
  it cannot, a dedicated kernel: one lane per `(cartesian AO, g)`, looping the
  shell's primitives — `c·(π/α)^{3/2}·e^{−|Gk|²/4α}·e^{−iGk·R}` times the
  Hermite polynomial in `−iGk`, coefficients tabulated on the host; cart→sph
  by a per-shell matrix in a second pass.
  K-15 is concrete `f64` (it calls `cube_math::double` for bit-exact
  `exp`/`sincos`); reusing it keeps that documented exception. A dedicated
  kernel is generic with `F::exp` / `F::cos` / `F::sin` (the target here is
  1e-11 against upstream, not bit-exactness).
- **K-PP2 — projector block.** One lane per `(projector row, g)`. Host table
  per row: atom, `l`, `m`, `i`, `rl`, and the real-solid-harmonic polynomial
  coefficients for `l ≤ 3`. The lane forms `|Gk|`, the polynomial,
  `rl^(l+1.5)·π^1.25·exp(−½·rl²·|Gk|²)`, `qli(|Gk|·rl, l, i)` (closed-form
  polynomials, selected with `#[comptime]`-free integer branches — `l`, `i`
  are data), and multiplies by `conj(SI[a, g])` read from the resident K-02
  table (`pbc/struct_factor.rs`). No local arrays (CPU-runtime stack rule).
- **K-PP3 — fused fold.** One lane owns one `(p, q)` element of `B` and walks
  `g` upward through the block, starting from the value already in `B`
  (`accumulate = 1`) — BAND-08's shape (`pbc/band_vmat.rs`), including its
  GPU split-lane variant and the register-blocked fold. Because one
  accumulator per element sums `g` in ascending order across ALL blocks, the
  result is **bit-identical at every block size**; that is the C1 gate.
  AO-major `A[q·nb + g]` and `S[p·nb + g]` keep each lane's reads contiguous
  (point-major planes cost 1.8× on the K-14f reduction).
  Alternative to benchmark, same operands: four real
  `launch_gemm_on_handles` per block. Faster on dense shapes, but its
  summation order depends on the block size. Decision rule: take the GEMM
  route only if it is more than 2× faster on a T4 AND the oracle gate still
  passes at two different budgets.
- **K-PP4 — Hermitian sandwich.** `h ∘ B` (block-diagonal `hl` per atom and
  channel, a `P × nao` kernel with a per-row table of `(row offset, nl,
  hl[i][j])`), then `B^H · (hB)` through `zgemm_on_handles` (§T1 K-LS item 2)
  and an add into the resident `vpp_k` that the local part
  (`pbc/local_vmat.rs`, K-14f) already produced on the device. `get_pp` then
  reads `vpp_k` back once.

**CubeCL rules these kernels follow** (manual: `Cubecl_generics.md`,
`11_launch_overhead_and_transfers.md`, `03_kernel_fusion.md`,
`13_memory_preallocation.md`; repo: `pyscf-algebra/src/launch.rs`):

- launch through `launch_1d` / `reduction_lanes` — never a hand-picked 256
  cube (pathological on the CPU runtime);
- no per-lane `Array::new` locals; if one becomes necessary, launch through
  `launch_1d_chunked` with its byte count (CPU-runtime stack grows per
  iteration);
- no `while { if … else … }` in a cube fn (aborts the CPU runtime's MLIR
  pass) — use `for … in range_stepped` and guarded `if`s;
- the `i < lanes` guard on every lane (launches round up to whole cubes);
- `Gv`, `SI` and the tables are uploaded once per `get_pp` call, block
  buffers are allocated once and reused (pool pre-allocation), one read-back
  per k-point, and `memory_cleanup()` when the call ends;
- every `unsafe` handle construction sits in the launcher with a `SAFETY:`
  note, after host-side validation of every length.

**Change.**
1. `crates/pyscf-kernels/src/pbc/pp_gspace.rs` (new) — K-PP2, K-PP3, K-PP4
   and, if needed, the dedicated K-PP1. Launchers follow `struct_factor.rs`:
   a `launch_*_on_handles<R: Runtime>` core plus a `dispatch_backend` entry.
2. `crates/pyscf-pbc-gto/src/pseudo/projg.rs` (new): host builders of the
   projector tables and a HOST `qli` used only to build test references.
3. `crates/pyscf-pbc-df/src/fftdf_pp_nl.rs` (new):
   `get_pp_nl_gspace(df, kpts)` — table upload, the block loop, the budget.
4. `Fftdf` gains `pp_nonlocal: PpNonlocal { Reciprocal, RealSpace }`.
   Default **`Reciprocal`** (upstream). `PYSCF_PBC_FFTDF_PP_NL=realspace`
   selects the old route for anyone who wants the cutoff-independent
   operator.
5. `fftdf.rs:902 vpp_add_nonlocal` takes `&Fftdf` and dispatches.
   `get_hcore_nonlocal` (BAND-03, called from `krks.rs:391`) must use the
   same route as `get_hcore`, or the bands are computed with a different
   Hamiltonian than the SCF. It currently takes only `cell`; it gets the
   `df`.
6. `PeriodicDf` already has `get_pp`; no new trait method is needed if step
   5 stays inside `fftdf.rs`. If one is added, `pyscf-py`'s `SharedDf` MUST
   forward it — a defaulted trait method silently returns the default from
   Python.
7. AFTDF, GDF, RSDF and MDF keep `get_pp_nl`: upstream's `aft.py` route is
   analytic too. Multigrid: read upstream's `multigrid` `get_pp` before
   touching it — not checked for this plan. Gradients keep
   `vppnl_nuc_grad` (upstream `pbc/grad/krhf.py:69` is analytic even under
   FFTDF). Check the finite-difference gradient/stress gates: with an
   analytic gradient and a G-space energy they agree only at a converged
   mesh, which is upstream's behaviour too — any FD gate that runs at a
   coarse mesh needs `PpNonlocal::RealSpace` pinned, with a comment.
8. Module header of `fftdf.rs` and `tests/fftdf.rs`: the "deliberate
   deviation" text becomes the description of the opt-in route.

**Kernel tests** (separate files, CPU runtime locally, CUDA in V3):
- `crates/pyscf-kernels/tests/pp_gspace_projector.rs` — K-PP2 against
  upstream `fakemol.eval_gto('GTOval', Gk) · _qli` values exported by the
  oracle, every `(l, i)` the GTH tables use (K, Y, Ta, O), 1e-13.
- `crates/pyscf-kernels/tests/pp_gspace_ft_ao.rs` — K-PP1 against the host
  `ft_ao_kpt` and against upstream `ft_ao.ft_ao`, s through f shells, Γ and a
  complex k-point, 1e-12.
- `crates/pyscf-kernels/tests/pp_gspace_fold.rs` — K-PP3: one block vs many
  blocks **`to_bits()`-identical**; vs a host complex dot, 1e-12; the split
  variant vs the plain one.
- `crates/pyscf-pbc-df/tests/fftdf_pp_nl_blocked.rs` — the assembled
  `get_pp`: budget 1 MB (ragged last block) vs one block, bit-identical; peak
  device bytes (`client.memory_usage()`) do not grow with the grid.

**Cost — measured, not assumed.** Per k-point: one transform of
`ngrids × nao` values and `P × nao × ngrids` ≈ 7.7e10 complex multiply-adds
(YTa7O19 DZVP). The real-space route cost 1021 s for the SZV `pre_h1e` on a
T4. Record, with a warmed backend, for KTaO3 and for YTa7O19 SZV
(`YTA_STOP_AFTER=1e`): CPU runtime locally, T4 in V4. Report the split
K-PP1 / K-PP2 / K-PP3 / K-PP4 from one same-binary run (lazy launches blur
host spans — attribute on the total, one variable at a time).

**Gates.** `fftdf_pp_gspace_oracle.rs` green at 60 Ha and at the default
mesh, Γ and complex k; the four kernel tests above; Hermiticity and Γ-real
checks from `tests/fftdf.rs` re-run on the new route; `YTA_REQUIRE_BACKEND`
= `cuda` in V3 so a silent CPU fallback cannot pass as the GPU result.

---

## T4 — Pipeline, driver, and the T4 recipe

1. **Fingerprint.** `fingerprint.json` gains `"pp_nonlocal"`. A checkpoint
   without the key was computed with the real-space route: a run that now
   asks for `reciprocal` is refused (exit 6) with the message "the saved
   one-electron matrices use the real-space non-local pseudopotential; start
   a new run or set PYSCF_PBC_FFTDF_PP_NL=realspace". Never mixed.
2. **Stage identity.** `Stage::identity` gains `conv_check` and a
   `level_shift_convention: 2` marker. Migration, so an in-flight run is not
   thrown away:
   - saved identity without the marker, same controls otherwise → the stage
     is resumed with the shift **halved automatically** for restricted runs
     (the exact equivalent, T1) and a one-line notice; the marker is written
     on the next cycle.
   - a stage saved as converged without `conv_check` → kept as converged only
     if the stage runs with `conv_check` off; otherwise it restarts from its
     saved density (one or two cycles plus the final step).
3. **Recipe.** `YTA_CONV_CHECK` = `final` (default: on for the last SCF stage
   only), `all`, or `off`. `pre` and `s1` are intermediate densities; with
   shift + wide smearing upstream itself calls them unconverged after the
   final step, and `YTA_REQUIRE_CONVERGED=1` must not abort on that.
   `config.example.json`: `YTA_LEVEL_SHIFT` 0.3 → 0.15 and
   `YTA_REFINE_LEVEL_SHIFT` 0.1 → 0.05 — the same Fock matrices as today,
   so the measured convergence (YTa7O19: 17 cycles) is kept.
   README: one paragraph on the convention change.
4. **Batch knob.** `PYSCF_PBC_PP_NL_BUDGET_MB` in `config.example.json`
   (T4: 1024) and in the README's memory table.
5. **Runner.** `tools/kaggle-t4/build_runner.sh` rebuild (podman, no GPU
   needed locally); new runner dataset slug and SHA through
   `ktrun.py publish-runner` (Kaggle mounts stale versions of a reused slug).
6. **`upstream_check.py`.** The `like` arm loses its three patches one by
   one as T1–T3 land; when all are in, `like` and `stock` are the same run
   and the arm is deleted. `tests/test_ktrun.py` is unaffected.

---

## T5 — Validation

| step | where | what must hold |
|---|---|---|
| V1 | local CPU | the three oracle tests green; `baseline.txt` suite re-run, every changed test explained |
| V2 | local CPU | KTaO3 pipeline vs **stock** PySCF: `e_tot` < 1e-7 Ha, bands < 1e-5 Ha, gap < 1e-4 eV, same `converged` flags, cycle counts within ±1 (MINAO guess still differs) |
| V3 | **Kaggle T4**, `ktrun.py`, KTaO3 via `YTA_CELL`, budgets forced small (`PP_NL` 16 MB, `NUMINT_AO` 256 MB, `FFTJK_AO` 256 MB, image cache 1 MB) | finishes in one session; `upstream_check.py` on the pulled checkpoint meets the V2 limits — this is the first CUDA-vs-PySCF comparison of the pipeline |
| V4 | Kaggle T4, YTa7O19 SZV, `YTA_STOP_AFTER=1e`, `YTA_REQUIRE_BACKEND=cuda` | `h1e` stage completes within the 15 GB device / 31 GB host limits; per-kernel time (K-PP1..4) and peak device memory recorded against the 1021 s real-space figure; a second run at `PYSCF_PBC_PP_NL_BUDGET_MB=64` gives a bit-identical `h1e.bin` |
| V5 | decision | re-run YTa7O19 DZVP with the reciprocal route, or keep the published 3.116 eV as the real-space-route result. Expected shift from KTaO3: about 0.6 mHa per oxygen (≈ 24 mHa in `e_tot`) and a few meV in the gap — an extrapolation, to be replaced by the V5 number if the re-run is done |

V3 and V4 need about half an hour of T4 quota together (estimate). V5 costs
what the original run did: two T4 sessions (55 min per DZVP cycle, 17 cycles,
plus `pre` and the bands), or a few hours on an RTX PRO 6000 (5.5 min per
cycle). It cannot resume from the existing checkpoint — `h1e.bin` changes.

## 3. Decisions needed from the user

1. **D1 default.** Reciprocal (upstream-identical, cutoff-dependent) as the
   default, real space as the opt-in — recommended, since the port's contract
   is parity. The alternative keeps today's default and adds reciprocal as
   the opt-in.
2. **D2 scope.** Follow upstream's restricted-case quirk exactly
   (recommended), or keep the port's form and document it.
3. **V5.** Re-run YTa7O19 after the fixes or not.

## 4. Risks

- Existing users of `level_shift` see different convergence (mitigated: the
  half-shift identity, the config change, the migration in T4.2).
- `converged = false` after the final step where it was `true` (upstream
  behaviour; the pipeline default `YTA_CONV_CHECK=final` contains it).
- The reciprocal route may be slower than the real-space one on small cells
  and faster on large ones — unknown until T3's measurement.
- CUDA kernels cannot be executed locally (no CUDA device; the ROCm iGPU has
  no f64). They are developed and tested on the CPU runtime and first run on
  a GPU in V3 — a CUDA-only codegen failure costs a runner rebuild and a
  Kaggle round trip. Keep K-PP1..4 free of constructs that have failed
  before (above) and run V3 before anything depends on them.
- K-15 reuse for K-PP1 is unverified (ghost partner, single image).
- D1 changes `H_core` for every FFTDF pseudopotential run below a converged
  mesh; gradient and stress finite-difference gates may need the real-space
  route pinned (T3.6).
