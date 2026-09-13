# Phase 19 verification — periodic response + relativistic (CLOSED 2026-09-13)

Nineteen plans, all shipped with summaries (19-01…19-19). Rollup command
(all green 2026-09-13):

```bash
cargo test -p pyscf-pbc-tdscf -p pyscf-pbc-gw -p pyscf-pbc-adc -p pyscf-pbc-x2c -p pyscf-pbc-eph
# 24 suites, 0 failed (2 pre-existing #[ignore]d live arms + ea 1 ignored stay ignored)
```

`grep -n '1e-15' .planning/ROADMAP.md` on the Phase-19 line returns only the
`~~1e-15 eV~~` strikethrough — the convention for superseded numbers (struck
through, never deleted, Phase-16 precedent). No ACTIVE 1e-15 gate remains;
no tolerance introduced anywhere in this phase is tighter than one ulp of the
quantity it bounds. `FEATURES` untouched (vendored upstream capability list,
unchanged since the initial commit — not a port-status ledger).

## Gate matrix (one row per gate per method)

| gate | method(s) | measured | tolerance (19-01) | verdict | test file:line |
|---|---|---|---|---|---|
| A1 | RHF TDA singlet/triplet, TDHF singlet (gamma, live Diamond fixture) | pass | 1e-5 (asserted unit) | MET | `pyscf-pbc-tdscf/tests/rhf.rs` (6/6) |
| A1 | KRHF TDA/TDHF per k-shift, sorted (live (2,1,1) fixture) | pass | 1e-5 | MET | `pyscf-pbc-tdscf/tests/krhf.rs` (7/7) |
| A1 | UHF/KUHF TDA/TDHF | 7/7 green | 1e-5 | MET (1 live arm NOT RUN → carryover) | `pyscf-pbc-tdscf/tests/uhf.rs` |
| A2 | RKS/UKS/KRKS (+PBE0 hybrid) | pass | 1e-8 Ha | MET | `pyscf-pbc-tdscf/tests/krks.rs` (8/8) |
| B | periodic CPHF enters `pyscf_grad::cphf::solve`; no second `pub fn solve` | structural pass | pass/fail | MET | 19-03 + `check-single-cphf` gate |
| B-adjacent | `newton_ah` 6 green; `stability` 7/7 | pass | — | MET (1 newton live arm NOT RUN → carryover) | `pyscf-pbc-scf` |
| C-AC | KRGWAC QP (live diamond/GDF fixture, `nw=100` pinned) | pass at 4dp | 1e-4 | MET | `pyscf-pbc-gw/tests/krgw_ac.rs` (6/6) |
| C-CD | KRGWCD closed-form root 1e-9; split recorded; route-blind refused | 1e-9 vs 1e-4 floor | 1e-4 per route | MET (analytic; live-vs-upstream NOT RUN → carryover) | `pyscf-pbc-gw/tests/krgw_cd.rs` (8/8) |
| C-UAC | KUGWAC closed-shell U==R (bit + 1e-12); open-shell vs bisection inside 1e-4 | inside 1e-4 | 1e-4 per route | MET (analytic; live NOT RUN → carryover) | `pyscf-pbc-gw/tests/kugw_ac.rs` (4/4) |
| C-slow | supercell σ identity 1e-12; `nk=1` bit-identities; slow-vs-AC inside 1e-4 | inside 1e-4 | 1e-4 per route | MET (live NOT RUN → carryover) | `pyscf-pbc-gw/tests/kgw_slow.rs` (4/4) |
| D | ADC IP/EA incore + DFADC, each vs own upstream number | pass at 4dp | 1e-4 | MET | `pyscf-pbc-adc/tests/gate_d.rs` (5/5); `ip` 4/4, `ea` 3/3+1 ignored |
| D-adjacent | `kadc_base` 7/7 incl. rollup-fixed sign gate (see below) | pass | — | MET | `pyscf-pbc-adc/tests/kadc_base.rs` |
| E | sfx2c1e/x2c1e transform on upstream `(t,v,w,s)` blocks | < 1e-8 | 1e-8 Ha | MET | `pyscf-pbc-x2c/tests/gate_e.rs` (8/8) |
| (own floor) | `eph_fd` linearity/quadratic-rate/refusals | pass | own floor | MET | `pyscf-pbc-eph/tests/eph_fd.rs` (5/5) |

## Defect found AT rollup (closed, not carried)

- `kadc_base::t2_first_order_matches_defining_equation` FAILED when the
  rollup gate ran (bit patterns differed by exactly 2⁶³ = a pure sign flip).
  Root cause: the TEST used positive-gap denominators while upstream
  (`kadc_rhf_amplitudes.py:101-108`, `_get_epq fac=[1,-1]`) and the
  implementation use negative gaps. Test corrected to the upstream
  convention; implementation untouched (already pinned by `tests/ip.rs`
  live-block comparison). `kadc_base` 7/7 after fix. The defect predates the
  rollup (19-14's summary claims 7/7 — the tree drifted after it was written).

## NOT RUN (carryover ledger, not passes)

- Gate C live-vs-upstream arms for CD / unrestricted-AC / slow (analytic arms
  MET; live GDF fixtures don't exist except 19-10's AC one).
- Ignored live arms: `newton_ah` (1), `uhf` (1), `ea` (1) — all need
  `PYSCF_ORACLE_VENV`.
- All rows: `.planning/carryovers/19-gw-live-oracle.md`.

## No `NotYetImplemented { phase: 19 }` remains unaccounted

`grep -rn "phase: 19" crates/pyscf-pbc-{gw,tdscf,adc,x2c,eph}/src` returns only
documented seams (DF `W`/`Lpq` builds needing a live GDF object; two-pole FIT
needing an optimizer; SO blocks; MPI) — each refused loudly at its boundary,
none silently bypassed.

## Rollup edits

- `19-VERIFICATION.md` (this file); `19-01-SUMMARY.md` written (was missing —
  measurements README existed without it).
- `19-11/19-12/19-13-SUMMARY.md` (this session): CD 8/8, UAC 4/4, slow 4/4.
- `ROADMAP.md` Phase-19 row: close-out appended; checkbox stays `[ ]` — Gate C
  live arms for three of four GW routes are NOT RUN, and the box ticks only
  when all gates are met.
- `STATE.md`: position notes Phase-19 closure; counters untouched (owned by
  20-01 reconciliation).
