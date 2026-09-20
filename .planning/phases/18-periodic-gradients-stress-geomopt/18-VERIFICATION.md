# Phase 18 verification rollup (18-15) — 2026-09-20

**Status: IMPLEMENTED, NOT CLOSED.** Every plan shipped code or measurements;
six gates are RUN/FAIL with measured numbers and defect IDs below, and three
gate families are NOT RUN with reasons. No gate was loosened to make it pass;
no tolerance below differs from `18-CONTEXT §2.3` / `measurements/README.md`
as restated by 18-21. Code navigation used the CodeGraph MCP server.

## 1. Plan disposition (all 21)

| plan | content | verdict |
|---|---|---|
| 18-01 | MEASURE gradient floors | DONE (pre-existing; `measurements/README.md` + `anchors.out` + `gradient-sweep.out`) |
| 18-16 | MEASURE strain floors / Parseval / Gate E | DONE (pre-existing; `gate-a-tiers.md` stable-ref green at unchanged bounds, `parseval.md`, `gate-e-gamma.md`) |
| 18-17 | MEASURE screening / sizings / fusion | DONE (pre-existing; `sizings.md` — default True, multiplicity 2, DO-NOT-FUSE) |
| 18-02 | grad crate surface + `verify_fd` + strain FD | DONE — `verify_fd` 4/4, `contract` 1/1, `tagged_dm` 2/2 green (this session, post svml fix) |
| 18-18 | aoslice 4-tuple + tagged density + image weights | DONE (pre-existing; used by 18-08/18-12 gates this session) |
| 18-03 | PP + Ewald nuclear gradients | DONE (pre-existing `5e6dd65`; `ewald_nuc_grad` 2/2 green this session) |
| 18-09 | multigrid-v2 gradient entry points | DONE (pre-existing; gamma suites' structural tests green — §3) |
| 18-11 | strain-AO kernel in `pyscf-kernels` | DONE (pre-existing; A1 strain-AO tests green inside `rks_stress` 27 — §3) |
| 18-04 | FFTDF gradient JK + route refusals | DONE (pre-existing; refusal re-verified by 18-14 `gdf_route_is_a_named_refusal` this session) |
| 18-10 | gamma `rhf`/`uhf` + `rks`/`uks` aliases | SHIPPED, Gate E RUN/FAIL — **F-18-10-01** |
| 18-12 | `rks_stress` core + closed form | SHIPPED — Gate A 25/27 assertions green, 2 RUN/FAIL (**F-18-12-01**); Gate D LDA green |
| 18-05 | `krhf` assembly | SHIPPED — 10/10 unit green; Gate B gamma green, 1×1×2 RUN/FAIL (**F-18-19-01**) |
| 18-20 | `rks_stress` terms + Gates A/D | SHIPPED — see 18-12 row |
| 18-19 | KRHF surface + scanner + Gate B/C | SHIPPED — Gate C green; Gate B split (see F-18-19-01) |
| 18-13 | `uks`/`krks`/`kuks` stress | SHIPPED — `uks` 9/9, `krks` 21/21, `kuks` 10/10, hubbard 1/1 green |
| 18-06 | `kuhf` | SHIPPED — Gate C + 4 unit green (release); Gate B open-shell RUN/FAIL (**F-18-06-01**) |
| 18-14 | geomopt over native BFGS+RFO | SHIPPED this session — 4/4 machinery green; convergence gate RUN/FAIL (**F-18-14-01**); `ROADMAP:464` lattice half restated |
| 18-07 | `krks`/`kuks` | SHIPPED — `krks` 13/13 (debug), `kuks` 10/10 (release) green |
| 18-08 | `krkspu`/`kukspu` | SHIPPED this session — 6/6 component/wiring green; full-SCF+U B/C NOT RUN (no driver — carryover C-18-08-01) |
| 18-21 | gate restatement | DONE this session — five gates with floors/margins in `measurements/README.md`, `18-CONTEXT §2.3`, `ROADMAP:464`, `PBC-MASTER-PLAN §7`/`§2.4`/`§8.10` |

## 2. Gates A–E per body

**Gate A (strain components vs own FD; A1 1e-9 / A2 2e-9 / A3 1e-8).**
`rks_stress`: 27/29 green — all A1 strain-AO/grid-response/lattice/vxc,
A2 `get_j`/`get_nuc`, A3 `get_pp`, block-budget, symmetry/determinism tests
PASS. RUN/FAIL: `closed_form_ovlp_matches_fd_oracle_at_1e9` (lif (0,0)
3.741e-9) and `closed_form_kin_matches_fd_oracle_at_1e9` (lif (0,0)
1.082e-9) — **F-18-12-01**. `krks_stress` 21/21, `uks_stress` 9/9,
`kuks_stress` 10/10, `kuks_stress_hubbard` 1/1 green.

**Gate B (analytic vs `verify_fd`; 1e-6, 5e-6 DFT+U).**
PASS: `krhf_verify_fd` 1/1; `krks` DFT-step gates (in 13/13);
`krhf_surface` gamma 4.037e-7; `kuks` release 10/10.
RUN/FAIL: `krhf_surface` 1×1×2 — 3.323e-6 > 1e-6 (**F-18-19-01**);
`kuhf_open_shell_passes_verify_fd` (HeH gamma) — 1.282e-4, bit-identical in
debug and release (**F-18-06-01**).
NOT RUN: `krkspu`/`kukspu` full-gradient B (needs SCF+U energies).

**Gate C (vs upstream `lib.fp(g)` at upstream decimals).**
PASS: `krhf_surface::gate_c_fingerprint_matches_upstream_to_6dp`;
`krks_{lda,gga,hybrid}_matches_upstream_fingerprint` (in 13/13);
`kuhf_closed_shell_matches_upstream_fingerprint` (release);
`kuks` fingerprints (release 10/10).
NOT RUN: `krkspu`/`kukspu` full-SCF+U fingerprints (no DFT+U SCF driver —
`KrkspU`/`Kukspu` are `veff` wrappers only; carryover C-18-08-01). The 18-08
wiring tests prove `kernel = base + U rows` exactly (≤1e-12) so the missing
half is the SCF, not the derivative.

**Gate D (stress vs FD of `E(ε)`; 1e-6 Ha/Bohr³, `h = 1e-3`, `/vol`).**
PASS: `gate_d_lda_stress_matches_scf_fd_over_vol`. GGA/MGGA end-to-end
correctly refuse without deriv-2/tau XC support (upstream refuses NLC/MGGA
on the gradient half and the port mirrors it; refusal tests green).

**Gate E (gamma multigrid-v2 bodies; 1e-8, never Gate B).**
RUN/FAIL — **F-18-10-01**: `gamma_rhf` 8/10 green but
`fd_gate_hf_on_gamma_path` 5.898e-1 and `fd_gate_lda_on_gamma_path`
4.084e-1 vs 1e-8; `gamma_uhf` 5/8 green but
`closed_shell_limit_reproduces_rhf` 5.013e-2,
`veff_stays_spin_resolved_with_upstream_sign`,
`fd_gate_hf_on_gamma_path` 3.387e-1 fail. Residuals O(0.1–0.5 Ha/Bohr) are
a structural analytic-gradient defect on the gamma path, not tolerance
noise. Structural gamma tests (refusals, aliases, determinism, screening
default) all green.

**18-08 component gates (this session).** Task 1 C1-vs-FD 3.267e-7 at 6
decimals — reproduces upstream's own 3.266e-7 to 4 digits. Task 2 U-vs-E_U-FD
2.435e-10 (restricted) / green (unrestricted). Closed-shell limit
restricted-vs-unrestricted EXACT (0.0). Wiring exact (≤1e-12). No substrate
duplication (`grep "fn set_u\|fn make_minao_lo" crates/pyscf-pbc-grad/`
empty). ALG-06 wall green.

**18-14 gates (this session).** `optimizer_entry_points`,
`gdf_route_is_a_named_refusal`, `input_cell_is_unchanged`,
`displaced_diamond_relaxes_stretch` (energy −6.3e-4 over 8 cycles,
displaced atom moves back, lattice/mesh untouched) green. Convergence gate
RUN/FAIL — **F-18-14-01**. No `fn rfo`/`struct Rfo`/lattice/strain
variation in the crate (verified by grep).

## 3. Findings (all with numbers; none loosened)

* **F-18-10-01** (18-10 Gate E): gamma analytic gradient wrong at O(0.1–0.5).
  Numbers above. The v2 entry points, refusals and aliases are fine — the
  defect is in the gradient body downstream of them.
* **F-18-19-01** (18-19 Gate B): KRHF 1×1×2 FD residual 3.323e-6 > 1e-6
  while gamma passes at 4.037e-7. k-point-dependent; analytic
  `[[−0.04194196647266641, …]]` vs fd `[[−0.041942137585238015, …]]`.
* **F-18-06-01** (18-06 Gate B): KUHF open-shell (HeH gamma) 1.282e-4,
  deterministic across debug/release. Closed-shell KUHF (incl. Gate C)
  green — the defect is in the open-shell spin threading or its FD oracle.
* **F-18-12-01** (18-12 Gate A1): lif closed-form-vs-FD 3.741e-9 (ovlp),
  1.082e-9 (kin) vs 1e-9. The closed form is exact by construction;
  suspicion is FD-oracle-side screening/truncation noise (D-PBC-30 clause 2
  bounds it by `cell.precision`), but attribution needs a tightened oracle —
  recorded, not absorbed.
* **F-18-14-01** (18-14 convergence): 2-atom diamond stretch-only
  optimization stalls with the stretch converged (`|gint|` < 1e-13) and
  Cartesian `max|de|` = 4.6e-2. The transverse PBC restoring forces are
  faithful (port and upstream agree to 7e-9 at the stall geometry) but lie
  outside the molecular redundant-internal span; mesh 13→31 does not move
  the floor (mesh-converged physics, not artifact). A faithful geomeTRIC
  port stalls identically — transverse-capable periodic internals are
  post-v2.0 scope alongside the lattice DOF.
* External blockers (not Phase-18 defects): lapack_rs `zheevd`
  divide-and-conquer panics (`ptr.rs:45`, `view offset 528 beyond length 1`)
  for nao ≥ 16, blocking every SCF gate on cells bigger than 2-atom
  (sibling repo, out of scope); no DFT+U SCF driver exists
  (carryover C-18-08-01).
* Incident: a repo-wide rustfmt churn (newer-rustfmt import reflow) was
  found in the working tree mid-session and fully reverted — the final diff
  touches only the files in §5.

## 4. NOT RUN (never inference)

* `krkspu`/`kukspu` full-SCF+U Gates B/C (C-18-08-01).
* 18-16's port-reference-cell re-measurements (`gate-a-tiers.md` notes them
  as remaining for 18-11/12/13).
* `kuhf`/`kuks` debug-mode full suites (release evidence stands; debug
  timed out at Ring 15 min per target — cost, not signal).
* Upstream geomeTRIC comparison for F-18-14-01 (`geometric` not installed
  in `.venv`; the stall analysis is first-principles + port-vs-upstream
  gradient agreement instead).

## 5. Diff inventory (this session)

* New: `pyscf-pbc-grad/{src/krkspu.rs, src/kukspu.rs, tests/krkspu.rs,
  tests/kukspu.rs}`, `pyscf-pbc-geomopt/{src/geometric_solver.rs,
  tests/optimize.rs}`.
* Modified: `pyscf-gto/src/svml_pow.rs` (1-line debug_assert fix —
  bit-pattern `as f64` → `f64::from_bits`, which un-breaks every debug
  test), `pyscf-pbc-grad/{src/lib.rs, src/gradients.rs}` (module wires +
  the `'ase'` arm corrected to `NotYetImplemented { phase: 20 }` per 18-14
  Task 3), `pyscf-pbc-grad/src/verify_fd.rs` (`with_coords` public for the
  optimizer seam), `pyscf-pbc-geomopt/{Cargo.toml, src/error.rs, src/lib.rs}`,
  and the four 18-21 documents (`18-CONTEXT.md §2.3`,
  `measurements/README.md`, `ROADMAP.md:464`, `PBC-MASTER-PLAN.md
  §7/§2.4/§8.10`).
* `FEATURES` moves nothing this phase (per-gate rule: k-point gradients and
  stress bodies with green gates are already listed at the family level;
  gamma gradients, DFT+U gradients and geomopt stay unlisted until their
  gates pass).
