# 17-14 — k-point-resolved multigrid, k-symmetry, and the on-device forward scatter — SUMMARY

**Closes `17-VERIFICATION.md` §10.12**, the phase's one multigrid carry-over:

> `pp.rs`'s IBZ path is not yet connected to 17-05's `KPoints` (which now
> exists); multigrid v1/v2 are gamma-only and not yet selectable as an SCF
> `numint`.

Written 2026-09-09, after Phase 17 closed on 2026-09-07. Three work items,
labelled K-01/K-02/K-03 to keep them separable in a later bisect:

| id | what |
|---|---|
| **K-01** | the k-point-resolved multigrid (`multigrid::kpts`), `kpts_band` support, and the k-general `pp` delegation |
| **K-02** | `KsymAdaptedKrks`/`KsymAdaptedKuks` able to run their SCF on it |
| **K-03** | the forward scatter moved onto the device |

## The finding that shaped K-01

**The k-point generalisation needed no kernel change at all.**

A fused pair term in `multigrid::pair` is one primitive pair `(p at A,
q at B + L)` at one lattice image `L`, and every Cartesian slot feeding that
term shares `L` — because `build_pair_level_table` resets its
monomial→term map on every `L`. The collocation kernel only ever sees
`term_coef[t]`, a real per term, and hands back `I[t]`, a real grid integral
per term. Neither knows what a k-point is. So:

* forward (upstream's `_dm_translation`):
  `D_R[μν, L] = Σ_k w_k D_k[μν] e^{-i k·L}` replaces the gamma point's
  `D[μν]` in the slot contraction;
* reverse: `V_k[μν] = Σ_L I[μν, L] e^{+i k·L}` scatters the same integrals
  with the conjugate phase.

The whole of K-01 is therefore two host-side contractions plus one new field
on `PairLevelTable` (`term_img`, the term's lattice image, and the
deduplicated `images` list). **No new kernel, no new launch, and the
collocation cost does not grow with `nkpts` at all** — it is over IMAGES.
That last point is the structural reason multigrid is the right engine for a
dense k-mesh, and also the reason k-point symmetry buys multigrid far less
than it buys the reference `KNumInt`, whose AO evaluation IS per k.

`kpts_band` fell out of the same fact: `I[t]` knows nothing about k, so
evaluating the potential at a different k-list from the density is one more
phase table. That is the whole of what K-02 needed.

## The trap, and the gate that exists because of it

**The forward direction cannot detect a Bloch-phase sign error.** The pair
list runs over every ordered `(pi, pj)` and its image list is closed under
`L → -L`, so `(μ, ν, L)` and `(ν, μ, -L)` are both present and contribute
complex conjugates whose sum is the same real number under EITHER sign
convention. A density-only gate — `∫ρ`, an energy, a trace — passes with the
potential silently returning `V_{-k}` in `V_k`'s slot.

So the sign gate is `vj_matches_fftdf_per_k`: **per k-point, complex,
against FFTDF**, which has no multigrid in it.

And that gate had a second trap of its own. Its first version used Γ-centred
`[1,1,2]` / `[2,2,2]` meshes on diamond and passed — with
`|Im vj|max ≈ 1e-20`. Those meshes sample nothing but time-reversal
invariant momenta, at which every one-electron matrix is REAL, so the sign
was unobservable and the gate was vacuous. It now runs on `[1,1,3]` /
`[2,2,3]` (`k = ±1/3` along an axis, not TRIMs) and **asserts the
precondition** that the reference's imaginary part is above `1e-4` before
comparing anything.

## The gates, measured

Cell: `common::diamond()`, mesh pinned `[15,15,15]` unless stated.
`tests/multigrid_kpts.rs`, `tests/krks_ksymm_multigrid.rs`,
`tests/multigrid_device_scatter.rs`.

| gate | claim | measured |
|---|---|---|
| `gamma_is_bit_identical_to_the_gamma_only_route` | the k-general route at one Γ point reproduces the gamma-only route it generalises | **`to_bits()` equal** on `nelec`, `exc`, `ecoul` and every `veff` element, for `lda,vwn` and `pbe,pbe`; `Im veff ≡ 0.0` |
| `gamma_get_j_is_bit_identical_to_the_gamma_only_route` | the same for the Coulomb matrix alone | **`to_bits()` equal** |
| `nelec_matches_the_k_resolved_trace` | `∫ρ = Σ_k w_k Re Tr(D_k S_k)` | `8.152e-7` (`[1,1,2]`), `2.354e-7` (`[2,2,2]`) — the collocation's own quadrature error at mesh 15, nothing else |
| `vj_matches_fftdf_per_k` | per-k complex `vj` vs FFTDF's, on a non-TRIM mesh | worst `1.884e-8` (`[1,1,3]`), `1.832e-8` (`[2,2,3]`) — with the precondition holding at `|Im vj|max = 4.403e-2` / `4.030e-2`, against the `~1e-20` that made the first version vacuous |
| `veff_is_hermitian_per_k` | `V_k = V_k^†` | `4.996e-16` |
| `band_kpts_subset_matches_the_full_evaluation` | `kpts_band` honoured positionally, out of order | **`to_bits()` equal** to the full evaluation's rows |
| `ibz_energy_matches_full_bz_on_multigrid` | K-02, IBZ vs full BZ | **`6.8747e-7`**, against a grid CONTROL of **`6.8737e-7`** on the identical fixture — see below |
| `multigrid_and_grid_agree_at_the_multigrid_floor` | Gate E, k-symmetric | **`1.3067e-9` Ha** (`e_grid -7.772927281262`, `e_multigrid -7.772927279955`), si, 3 IBZ of 8 BZ |
| `device_scatter_is_bit_identical_*` | K-03, both routes in one process | **`to_bits()` equal** for RKS Γ (LDA and PBE), RKS k-resolved, and UKS |

## Gate D — against UPSTREAM PySCF 2.12.1

Everything above is **Gate C** (port vs port). Gate D is the separate
question, and it lives in its own file (`tests/multigrid_kpts_oracle.rs`) so
the distinction cannot quietly erode. `E_tot` of a converged k-point KRKS,
same cell, same pinned mesh `[15,15,15]`, `lda,vwn`, diamond:

| k-mesh | upstream FFTDF | this port, grid | this port, MULTIGRID k |
|---|---|---|---|
| `[1,1,2]` | `-10.756804832489` | `1.323e-5` | **`1.289e-5`** |
| `[2,2,2]` | `-11.240948604144` | `2.381e-5` | **`2.374e-5`** |
| `[1,1,3]` | `-10.817702580812` | `1.738e-5` | **`1.715e-5`** |

Worst `|dE|` **2.374e-5 Ha**, four orders inside the `1e-1` gate.

**The control is what makes this readable.** The port's own GRID numint sits
`2.381e-5` from the same upstream number on the same fixture, and the
multigrid sits `2.374e-5` — indistinguishable, and the multigrid is
marginally CLOSER. So the residual is the port-vs-upstream baseline, not
something the k-point multigrid path introduces. Both named floors account
for it: xcfun vs upstream's libxc for `lda,vwn` (~5e-7) and the pinned coarse
mesh (~1.35e-5 on `KRHF`, the dominant term here).

### Finding: upstream's own multigrid REFUSES k-points

`pyscf.pbc.dft.multigrid.MultiGridNumInt2` on any of the three k-meshes:

```
NotImplementedError: MultiGridNumInt2 only supports Gamma-point calculations.
```

So there is **no upstream counterpart to compare this path against** —
"matches PySCF" can only mean "matches PySCF's reference FFTDF route", which
is what the table measures. This port's multigrid is k-point-general where
PySCF 2.12.1's is not; that is a capability beyond upstream, not a port of
one, and it is recorded here so a later reader does not go looking for the
upstream number that would make the comparison tighter. It does not exist.

The test asserts the refusal is REPORTED rather than skipped, so if a future
PySCF gains k-point multigrid, the log says so on the next run.

### The symmetry gate's control, and why it is a control

`ibz_energy_matches_full_bz_on_multigrid` first asserted `|dE| < 1e-9`
outright and measured **6.875e-7**. That looked like a symmetry defect and
was not one. The multigrid IBZ run agrees with the *grid* IBZ run to
**1.307e-9**, so the outlier was the full-BZ side — and it is the outlier for
BOTH quadratures. Measuring the reference grid's own IBZ-vs-full-BZ gap on
the identical fixture gives **6.8737e-7** against the multigrid's
**6.8747e-7**: the same number to four significant figures.

The cause is the PINNED COARSE MESH this file uses to keep the two sides
comparable — the same effect already recorded for `KRHF` (a pinned mesh 15
moves the energy by 1.35e-5 where the default mesh gives 4.8e-11), and
`krks_ksymm.rs`'s own `krks_ibz_energy_matches_full_bz` avoids it by not
pinning the mesh at all.

A mesh artefact shared by both quadratures is not a defect in the one under
test, and loosening the tolerance to hide it would have thrown away the
gate's ability to catch a real one. The assertion is therefore relative to
the measured control (`SYMMETRY_GAP_RATIO`), and the control is re-measured
on every run rather than recorded once here.

## K-03 — the forward scatter, on the device

Before K-03 the batched forward route read `npoints · 8` B back **per chunk**
and ran a host loop `rho[point_global[p]] = out[p]` over every padded point
of every chunk of every level, on every SCF cycle. K-03 keeps the level's
density on the device across its chunks (`PairOutScratch::mesh`), scatters
into it with `mg_scatter_kernel`, and reads `ngrids · 8` B back ONCE per
level.

**No atomics, and none are needed.** `grid_blocks`' partition gives each real
grid index exactly one owning block, so the scatter is a plain STORE, not an
accumulate — there is no summation whose order could change. That is what
makes the claim bit-identity rather than agreement, and
`tests/multigrid_device_scatter.rs` compares both routes IN ONE PROCESS
through the `PYSCF_MG_PAIR_DEVICE_SCATTER` seam (M-03's `use_batch` seam
applied to the other direction).

### The speed measurement, and it is not a win here

**Measured, and the honest answer on this machine is "no change":**

```
K-03 A/B on cpu (mesh [15,15,15], 3 reps after a warm-up):
  device scatter ON  : 2.2812 s / nr_rks
  device scatter OFF : 2.2884 s / nr_rks
  ratio (off/on)     : 1.003x
```

`1.003x` is noise. **K-03 buys nothing on the CPU backend**, and that is the
expected result rather than a disappointment: on the CPU runtime "device
memory" IS host memory, so the read-backs K-03 removes were already memcpys
and the host scatter loop it deletes is a small fraction of a `nr_rks` that
is dominated by the collocation's exponentials. What K-03 removes that a
discrete GPU would feel — one PCIe transfer per level instead of one per
chunk — cannot be measured here at all, because this machine's ROCm iGPU has
no f64 and `PYSCF_BACKEND=rocm` silently falls back to CPU.

So the row in the performance ledger reads: **structurally better, measured
flat on the only backend available, unmeasured where it should matter.** It
is recorded that way rather than as a speedup, per D-PBC-26 point 6 and the
`zgemm_dense` precedent — a route that is structurally better is not
automatically faster, and this one is a good example of it. The change is
kept because it is bit-exact and strictly removes work, not because a
measurement here supports it.

## What did NOT ship, and why

* **`MultiGridNumInt` (v1) stays gamma-only.** It collocates one real density
  from one real density matrix and has no per-image term identity to hang a
  Bloch phase on. v2 is the shipped fast path and the one Phase 18's
  gradients assert on (`PBC-MASTER-PLAN.md §8.10`), so it is the one that was
  generalised. `KsNumInt::require_multigrid_inputs` still refuses k-points
  and `kpts_band` for v1, by name.
* **No multigrid analogue of S-03's symmetrised quadrature.** S-03 exists
  because the grid route's cost is per k-point. Multigrid's is not, so there
  is nothing for it to save; the k-symmetric driver's existing S-01 unfold
  feeds it a full-BZ density and it takes its saving in the IBZ-length
  `kpts_band` return. Stated rather than left as a silent omission.
* **`multigrid::pp::get_nuc_kpts` / `get_pp_kpts` delegate to AFTDF**, as
  their gamma twins already did, at whatever k-list the caller passes. An
  IBZ-restricted caller passes `kpts.kpts_ibz` and unfolds with
  `transform_1e_operator` itself — per D-PBC-26 rule 5 the unfold belongs in
  `pyscf-pbc-dft`/`pyscf-pbc-scf`, never in the DF layer. These are not on
  the SCF path (`get_hcore` goes through `with_df`); they close §10.12's
  first clause.

## Files

| file | change |
|---|---|
| `crates/pyscf-pbc-dft/src/multigrid/kpts.rs` | NEW — the k-point layer: `PhaseTable`, `KDmP`, `pairlevel_rho_kpts`, `pairlevel_pass2_kpts`, `nr_rks_kpts` / `nr_uks_kpts` / `get_j_kpts` / `eval_rho_g_kpts` |
| `crates/pyscf-pbc-dft/src/multigrid/pair.rs` | `PairLevelTable::{term_img, images}`; the term-level seams `pairlevel_rho_from_terms` / `pairlevel_integrals`; K-03's device-scatter branch and its `PYSCF_MG_PAIR_DEVICE_SCATTER` seam |
| `crates/pyscf-pbc-dft/src/multigrid/pp.rs` | `get_nuc_kpts`, `get_pp_kpts` |
| `crates/pyscf-pbc-dft/src/numint.rs` | `KsNumInt` MultiGrid2 arms take k-points and `kpts_band`; `unfold_kdms_sym` / `unfold_dms_sym` split out of `KNumInt` so a multigrid `ni` needs no `KPoints` |
| `crates/pyscf-pbc-dft/src/krks_ksymm.rs` | `KsymAdaptedKrks::ni` / `KsymAdaptedKuks::ni` are `KsNumInt`; both `get_veff_tagged` bodies take the fused-Coulomb branch when the quadrature returns one |
| `crates/pyscf-kernels/src/multigrid_pair.rs` | `PairSlotBatch::point_global`; `PairOutScratch::{mesh, mesh_b, zero_mesh, read_mesh}`; `mg_scatter_kernel`; `PairSlotBatchDevice::{rho_into_mesh, rho2_into_mesh}` |
| `crates/pyscf-pbc-dft/tests/multigrid_kpts.rs` | NEW — 6 gates |
| `crates/pyscf-pbc-dft/tests/krks_ksymm_multigrid.rs` | NEW — 2 gates |
| `crates/pyscf-pbc-dft/tests/multigrid_device_scatter.rs` | NEW — 3 bit-parity gates + the A/B wall clock |

AGENTS.md §2 held: no `mod tests` in any `src/*.rs`. ALG-06 held:
`pyscf-pbc-dft` names no cubecl crate; the one new kernel lives in
`pyscf-kernels`.

## Carry-overs

1. **A production-mesh symmetry measurement.** Every number above is at a
   pinned coarse mesh, which is what the control exists to account for. The
   unpinned comparison `krks_ksymm.rs` makes for the grid route has no
   multigrid twin yet.
2. **The A/B wall clock on a GPU backend.** This machine's ROCm iGPU has no
   f64, so `PYSCF_BACKEND=rocm` silently falls back to CPU and K-03's
   transfer saving cannot be measured where it is largest.
3. **The reverse direction's `v_p` scatter is still on the host** —
   `O(nslots · nkpts)` per level. It is far smaller than the collocation, but
   it is the remaining host-side term that grows with `nkpts`.
4. **`KsymAdaptedKuks` on multigrid has no energy gate of its own.** The
   restricted twin has one; the unrestricted fixture needs a system whose
   full-BZ solution is symmetric in both spin channels (D-17-08-03), which is
   §10.9's open item.
