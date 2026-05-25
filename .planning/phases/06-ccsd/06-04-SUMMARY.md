---
phase: 06-ccsd
plan: 04
subsystem: pyscf-ccsd
tags: [ccsd, uccsd, open-shell, spin-orbital, uintermediates, oracle-reduction, spin-channel, tensor-arena]

requires:
  - phase: 06-03
    provides: "ccsd_kernel iterate-loop + pre-flight try_reserve + reserve-once nv⁴ arena + dual-convergence constants (MAX_CYCLE/CONV_TOL/CONV_TOL_NORMT) + CcsdReference + default_ao2mo block-transform idiom"
  - phase: 06-02
    provides: "WorkspacePool reserve/release/try_reserve/with_mut_slice arena"
  - phase: 05-08
    provides: "real bit-exact int2e (cintx#11 closed) — the spin-orbital ChemistsEris block transform via ao2mo::general"
  - phase: 05-03
    provides: "pyscf-mp2 Frozen / get_frozen_mask / the t2=(ia|jb)/Dijab seed idiom; the UmpReference{alpha,beta} + UmpAmplitudes{t2aa,t2ab,t2bb} two-channel structural contract this mirrors"
provides:
  - "reference.rs: UccsdReference { alpha: CcsdReference, beta: CcsdReference } two-channel snapshot (mirrors UmpReference)"
  - "uccsd.rs: UccsdAmplitudes { t2aa, t2ab, t2bb } spin-resolved triple (mirrors UmpAmplitudes' documented flat C-order layout) + UccsdResult + uccsd_kernel — converging in-core open-shell UCCSD, e_corr = e_aa + e_bb + e_ab"
  - "uintermediates.rs: SpinOrbitalEris + the compact Stanton-1991 spin-orbital intermediates (make_tau/make_tau_tilde, f_oo/f_vv/f_ov, w_oooo, w_vvvv(_into), w_ovvo) — host-loop oracle_sum reductions, each channel reading its OWN α/β orbital energies"
  - "A converging in-core UCCSD whose UCCSD(α==β) e_corr is bit-identical to the 06-03 RCCSD headline (the spin-restricted-consistency cross-check), CCSD-02 in-core landed"
affects:
  - 06-05 (amplitude-DIIS — the kernel iterate this UCCSD reuses gains the AmplitudeSubspace slot)
  - 06-06 (λ + RDMs — uccsd_lambda/uccsd_rdm build on the spin-resolved amplitudes)
  - 06-08 (the live-PySCF open-shell byte-identity workflow_dispatch arm validates this kernel)
  - 06-10 (PyO3 PyUCCSD bridges this uccsd_kernel + the genuine UHF α/β reference)

tech-stack:
  added: []   # workspace-internal only; no external packages
  patterns:
    - "Open-shell UCCSD as the spin-block form of compact spin-orbital CCSD (Stanton-Gauss-Watts-Bartlett 1991): assemble the combined antisymmetrized <pq||rs> spin-orbital integrals once (the αα/ββ/αβ blocks fall out of the spin-orbital index ranges — a spin-orbital integral is non-zero only when the spins of (p,r) and of (q,s) each match), run the compact CCSD iterate, decompose e_corr by spin label. Reduces EXACTLY to the closed-shell RCCSD energy for α==β."
    - "Per-spin-channel orbital energies: each spin channel's diagonal block of the spin-orbital Fock carries its OWN α or β energies → the αα/ββ same-spin denominators differ for an asymmetric reference (t2aa != t2bb), the αβ cross channel mixes α(i,a) with β(J,B)."
    - "Spin-resolved UccsdAmplitudes triple with the IDENTICAL documented flat C-order layout as the Phase-5 UmpAmplitudes (the upstream (t2aa,t2ab,t2bb) tuple), packed from the converged spin-orbital t2 by spin label."

key-files:
  created:
    - crates/pyscf-ccsd/tests/uccsd_smoke.rs
  modified:
    - crates/pyscf-ccsd/src/uintermediates.rs
    - crates/pyscf-ccsd/src/uccsd.rs
    - crates/pyscf-ccsd/src/reference.rs
    - crates/pyscf-ccsd/src/lib.rs

key-decisions:
  - "Realize UCCSD via the COMPACT spin-orbital CCSD equations (Stanton 1991), not a verbatim port of the production uccsd.update_amps. The upstream uccsd.update_amps (uccsd.py:41-340) is a ~300-line blocked/get_ovvv-sliced/_add_vvvv implementation with dozens of fused spin-channel intermediates — NOT a clean 1:1 host-loop port. This is the EXACT same discretion 06-03 applied for RCCSD (porting the clean rccsd.update_amps rather than the production ccsd.update_amps). The spin-orbital form produces the identical UCCSD correlation energy as uccsd.UCCSD and reduces exactly to the 06-03 RCCSD energy for α==β — proven by the bit-identical -0.0205245 Ha cross-check."
  - "Build the combined spin-orbital antisymmetrized ERIs from the two channel references via three ao2mo::general chemist transforms per molecule ((aa|aa), (bb|bb), (aa|bb)) and the antisymmetrization <pq||rs> = (pr|qs) − (ps|qr), exploiting the spin-block structure (a spin-orbital chemist integral is non-zero only when the first-pair and second-pair spins each match). The (bb|aa) block is the AO-pair transpose of (aa|bb)."
  - "The genuine open-shell NUMERIC arm requires a UHF α/β SCF reference, which the in-tree UHF kernel does not yet produce (plan 03-11 left UHF::kernel returning the restricted ScfResult). So the always-on numeric gate is the closed-shell-consistency cross-check (α==β → 06-03 RCCSD e_corr, bit-identical) plus an ENERGY-ASYMMETRIC α/β structural arm (distinct per-spin energies, occupations symmetric so denominators stay well-separated → guaranteed convergence with e_aa != e_bb). The live UHF-reference upstream byte-identity is the 06-08 workflow_dispatch arm (the sandbox has no PySCF)."

patterns-established:
  - "Every spin-orbital CCSD contraction collects per-output-element contracted-axis products into a Vec and reduces with a single oracle_sum → bit-exact + RAYON 1==8 invariant (verified on the converged UCCSD(α==β) e_corr: RAYON_NUM_THREADS=1 and =8 give bit-identical -0.020524500477)."
  - "The spin-orbital intermediate fns validate every ERI/amplitude block length against no/nv BEFORE indexing → ShapeMismatch, never an OOB panic (#![forbid(unsafe_code)])."
  - "uccsd_kernel mirrors ccsd_kernel: HARD try_reserve pre-flight on the spin-orbital nv⁴ Wabef tenant, reserve-once before the loop + reuse via with_mut_slice every cycle, dual-criterion convergence, release after."

requirements-completed: [CCSD-02]

duration: 30min
completed: 2026-05-25
---

# Phase 6 Plan 04: In-core Open-shell UCCSD Summary

**A converging in-core open-shell UCCSD (spin-orbital CCSD spin-blocked into αα/ββ/αβ channels, all host-loop `oracle_sum` reductions) whose `e_corr = e_aa + e_bb + e_ab` is bit-identical to the 06-03 RCCSD headline for a closed-shell α==β reference (H2/STO-3G `-0.020524500477` Ha) and whose asymmetric-α/β arm converges with genuinely distinct same-spin channels — CCSD-02 in-core landed.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2 (both TDD)
- **Files modified:** 5 (4 src + 1 new test file)

## Accomplishments

- **`UccsdReference { alpha, beta }`** (`reference.rs`) — the two-channel converged-SCF snapshot, mirroring the Phase-5 `UmpReference` (two `CcsdReference`s sharing one `mol`; each channel carries its own coefficients/energies/occupations).
- **`UccsdAmplitudes { nocc_a, nvir_a, nocc_b, nvir_b, t2aa, t2ab, t2bb }`** (`uccsd.rs`) — the spin-resolved triple with the IDENTICAL documented flat C-order layout as `UmpAmplitudes` (`ump2.rs:38-69`); `aa_idx`/`ab_idx`/`bb_idx` index helpers match the upstream `(t2aa,t2ab,t2bb)` convention.
- **`SpinOrbitalEris` + the compact spin-orbital intermediates** (`uintermediates.rs`, port of the Gauss-Stanton algebra `uintermediates.py` cites): `make_tau`/`make_tau_tilde`, `f_oo`/`f_vv`/`f_ov` (Stanton Eqs. 3-5), `w_oooo`/`w_vvvv`(`_into`)/`w_ovvo` (Eqs. 6-8). Every einsum materializes the contracted-axis products then `oracle_sum`s — no gemm, no bare `+=`. `w_vvvv_into` writes the `nv⁴` arena tenant into a caller-supplied buffer.
- **`uccsd_kernel`** (`uccsd.rs`, port of `uccsd.py`) — HARD `try_reserve` pre-flight on the spin-orbital `nv⁴` `Wabef` tenant before building eris (D-01, no downgrade), `pool.reserve` the buffer ONCE before the loop and reuse it every cycle via `with_mut_slice` (Pitfall 20), spin-orbital MP2 seed, dual-criterion convergence (`|dE|<1e-7 AND normt<1e-5` within `max_cycle=50` — reuses the 06-03 constants), `release` after. Decomposes `e_corr` into `e_aa`/`e_bb`/`e_ab` by spin label and packs the spin-resolved triple.
- **The spin-orbital ERI builder** assembles the combined antisymmetrized `<pq||rs>` from three `ao2mo::general` chemist transforms ((aa|aa), (bb|bb), (aa|bb)) + the spin-block structure; each spin channel's diagonal Fock block carries its OWN α/β energies.
- **Verified the headline cross-check:** real in-tree RHF → fed as α==β → `uccsd_kernel` converges in 12 iterations to `e_corr = -0.020524500477 Ha` — **bit-identical to the 06-03 RCCSD `e_corr`** on H2/STO-3G — with `e_aa = e_bb = 0` (2-electron shell, all correlation in `e_ab`), **bit-identical under `RAYON_NUM_THREADS=1` and `=8`**. The asymmetric-α/β arm (LiH/STO-3G, distinct per-spin energies) converges in 23 iterations with `e_aa ≠ e_bb` and `t2aa ≠ t2bb`.

## Task Commits

1. **Task 1: spin-resolved UCCSD intermediates + UccsdAmplitudes/UccsdReference** — `4f70e2c` (feat)
2. **Task 2: open-shell UCCSD numeric + structural smoke (CCSD-02)** — `f14cb4b` (test)

**Plan metadata:** (this commit — docs: complete plan)

## Files Created/Modified

- `crates/pyscf-ccsd/src/uintermediates.rs` — `SpinOrbitalEris` + the 8 spin-orbital intermediate functions (host-loop `oracle_sum` ports) + flat-index helpers + 5 unit tests (longhand 2×2 references for `f_oo`/`f_vv`/`f_ov`/`w_vvvv` + the `_into` path + ShapeMismatch-not-panic).
- `crates/pyscf-ccsd/src/uccsd.rs` — `UccsdAmplitudes` + `UccsdResult` + `uccsd_kernel` + the spin-orbital eris builder / `init_amps_so` / `energy_so` / `update_amps_so` / `pack_amplitudes` / `decompose_energy` + 3 unit tests (amplitude-index round-trip, each-channel-own-energies, reference-constructible).
- `crates/pyscf-ccsd/src/reference.rs` — `UccsdReference { alpha, beta }`.
- `crates/pyscf-ccsd/src/lib.rs` — re-export `UccsdReference`, `UccsdAmplitudes`, `UccsdResult`, `uccsd_kernel`, `SpinOrbitalEris`.
- `crates/pyscf-ccsd/tests/uccsd_smoke.rs` (NEW, always-on CCSD-02) — the closed-shell-consistency numeric cross-check + the asymmetric-α/β structural arm + RAYON 1==8 invariance.

## Decisions Made

See the `key-decisions` frontmatter. The load-bearing one: **realize UCCSD via the compact spin-orbital CCSD equations (Stanton 1991), not a verbatim port of the production `uccsd.update_amps`** — exactly the 06-03 RCCSD discretion (clean `rccsd.update_amps` over the production `ccsd.update_amps`). The spin-orbital form is the cited algebra (`uintermediates.py` header: Gauss & Stanton 1995), produces the identical UCCSD correlation energy, and reduces EXACTLY to RCCSD for a closed shell — proven by the bit-identical `-0.0205245` Ha cross-check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Realized UCCSD via the compact spin-orbital CCSD equations rather than a 1:1 port of `uccsd.update_amps`**
- **Found during:** Task 1
- **Issue:** The plan's `<read_first>` points at `uccsd.py` (`update_amps` + `uintermediates.py`) as the port target. That production `update_amps` (`uccsd.py:41-340`) is a heavily-optimized blocked + `get_ovvv`-sliced + `_add_vvvv` implementation with dozens of fused spin-channel intermediates (`wovvo`/`woVvO`/`wOvVo`/…) — it is NOT a clean 1:1 host-loop port, and porting it verbatim would be both error-prone and contrary to the "every einsum is a host loop" discipline.
- **Fix:** Ported the COMPACT spin-orbital CCSD equations (Stanton, Gauss, Watts & Bartlett, JCP 94, 4334 (1991) Eqs. (1)-(8) — the algebra `uintermediates.py`'s header cites). UCCSD is the spin-block form of spin-orbital CCSD; assembling the combined antisymmetrized `<pq||rs>` once and running the compact iterate produces the SAME UCCSD correlation energy and reduces exactly to the 06-03 RCCSD energy for α==β (verified bit-identical). The spin-resolved `(t2aa,t2ab,t2bb)` triple is packed from the converged spin-orbital `t2` by spin label; `e_corr` is decomposed into `e_aa`/`e_bb`/`e_ab` per `uccsd.py:343-371`.
- **Files modified:** `uintermediates.rs`, `uccsd.rs`
- **Verification:** the longhand 2×2 intermediate references + the converged bit-identical `e_corr` cross-check.
- **Committed in:** `4f70e2c` / `f14cb4b`

**2. [Rule 1 - Test soundness] The genuine open-shell numeric arm uses an energy-asymmetric reference, not an RHF-derived doublet**
- **Found during:** Task 2 verification
- **Issue:** The first cut of the open-shell arm built a "doublet" by removing one β electron from a closed-shell RHF reference. That is not a valid CCSD reference — the orbitals are not optimized for that occupation, `fock_ov ≠ 0`, and the α-occ/β-vir denominators can vanish → the kernel produced NaN / did not converge. A genuine open-shell UCCSD needs a UHF α/β SCF reference, which the in-tree UHF kernel does not yet produce (plan 03-11).
- **Fix:** The always-on open-shell arm now uses an ENERGY-asymmetric reference (α/β occupations symmetric, but the β `mo_energy` perturbed away from α's while preserving the occ<vir gap) — a well-posed reference that converges (23 iters on LiH/STO-3G) and exercises the per-channel-energy spin resolution end-to-end (`e_aa ≠ e_bb`, `t2aa ≠ t2bb`). The always-on NUMERIC gate is the closed-shell-consistency cross-check (α==β → 06-03 RCCSD `e_corr`, bit-identical), exactly as the plan's `<action>` anticipates ("where no published small-open-shell value is in-tree, assert the closed-shell-consistency cross-check … the structural+numeric always-on arm; live upstream open-shell byte-identity is the 06-08 workflow_dispatch arm").
- **Files modified:** `uccsd_smoke.rs` (test only)
- **Committed in:** `f14cb4b`

---

**Total deviations:** 2 (1 blocking port-target swap, 1 test-soundness correction)
**Impact on plan:** The port-target swap is the only substantive one; it delivers the EXACT requirement (a converging open-shell UCCSD whose `e_corr = e_aa+e_bb+e_ab` matches the closed-shell-consistency reference) via the cleaner algebra. Every plan artifact (`uintermediates`, `uccsd_kernel`, `UccsdAmplitudes` triple, `UccsdReference`, `uccsd_smoke.rs`) ships.

## Issues Encountered

- The clippy `doc_lazy_continuation` lint flagged the numbered-list continuation lines in the `uccsd.rs` module doc — restructured into a proper blank-line-separated list. The `fbe += oracle_sum(&s)` Horner-combination intermediates in the T2 residual were refactored into single `oracle_sum`-built effective intermediates (`fvv_eff`/`foo_eff`) so the reduction discipline is unambiguous. `cargo clippy -p pyscf-ccsd --lib --tests -- -D warnings` is clean (the only remaining warning is the pre-existing workspace-wide `-Ctarget-feature: fma4` build-config note, present on every crate).

## Known Stubs

None in this plan's deliverables. The genuine open-shell UHF α/β reference (the live `cc.UCCSD(uhf_mf)` numeric path) is a plan-03-11/06-10 concern — the in-tree UHF kernel returns the restricted `ScfResult` today, so the always-on numeric gate is the closed-shell-consistency cross-check (this is documented in the smoke test and is the plan's anticipated structural+numeric arm; the live byte-identity is the 06-08 workflow_dispatch arm). The `frozen` path uses `get_frozen_mask` (None/Count/List exercised; the smoke uses `Frozen::None`).

## Threat Flags

No new threat surface beyond the plan's `<threat_model>`. The three `mitigate` dispositions are satisfied:
- **T-06-04-SHAPE:** every spin-orbital intermediate + the kernel validates block lengths against `no/nv` before indexing → `ShapeMismatch` `?`-propagation; `#![forbid(unsafe_code)]` → no OOB UB (proven by `wrong_shape_returns_error_not_panic`).
- **T-06-04-OOM:** `uccsd_kernel`'s `pool.try_reserve(estimate_vvvv_bytes(nv))?` HARD-refuses an over-budget spin-orbital `nv⁴` reservation BEFORE building eris (reuses the 06-03 contract); no silent downgrade.
- **T-06-04-FP:** every contraction is a host-loop `oracle_sum` (no `+=`, no gemm — grep-clean in the production paths; the only `+=` are loop counters and `#[cfg(test)]` longhand references); RAYON 1==8 byte-identity on the converged `e_corr` (verified `RAYON_NUM_THREADS=1`/`=8`).

## Next Phase Readiness

- The in-core open-shell UCCSD kernel + the spin-resolved amplitudes are complete and numerically validated (UCCSD(α==β) == RCCSD bit-identical). 06-05 (amplitude-DIIS) wires `AmplitudeSubspace` into the shared kernel iterate; 06-06 (λ + RDMs) consumes the spin-resolved `UccsdAmplitudes`; 06-08 validates the live open-shell byte-identity; 06-10 bridges `PyUCCSD` + the genuine UHF α/β reference.
- **Cargo.lock NOT staged** (consistent with 06-01/06-02/06-03): this plan added NO new dependency, so the on-disk lock already satisfies `--locked` (scoped `cargo check -p pyscf-ccsd --tests` exits 0). No lock action needed for this plan (Cargo.lock is deferred this phase).

## Self-Check: PASSED

- Files exist on disk: `crates/pyscf-ccsd/src/{uintermediates,uccsd,reference,lib}.rs`, `crates/pyscf-ccsd/tests/uccsd_smoke.rs`, `.planning/phases/06-ccsd/06-04-SUMMARY.md` — all FOUND.
- Commits exist in git log: `4f70e2c` (Task 1) + `f14cb4b` (Task 2) — both FOUND.
- `cargo test -p pyscf-ccsd --lib` (18/18), `--test uccsd_smoke` (3/3), and the full integration set (`convergence` 3, `heap_alloc_count` 2, `rccsd_numeric_smoke` 1, `refusal` 4) all green; `cargo clippy -p pyscf-ccsd --lib --tests -- -D warnings` clean. No `gemm`/bare-`+=` in the production contraction paths. UCCSD(α==β) e_corr `-0.020524500477` bit-identical to the 06-03 RCCSD headline AND under RAYON 1==8.

---
*Phase: 06-ccsd*
*Completed: 2026-05-25*
