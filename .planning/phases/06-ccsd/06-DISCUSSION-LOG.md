# Phase 6: CCSD - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-24
**Phase:** 6-ccsd
**Areas discussed:** Tensor-arena & memory model, ERI-mode scope/sequencing, Λ + RDM depth, Numeric oracle corpus

---

## Tensor-arena & memory model (CCSD-11)

| Option | Description | Selected |
|--------|-------------|----------|
| Opaque spillable tensor + hard refuse | t1/t2/Wabef are opaque Tensor handles (Vec OR HDF5-backed) from day one so spill is a storage-backend swap; PYSCF_MAX_MEMORY pre-flight HARD-REFUSES an over-budget in-core job, user opts into DF/direct explicitly | ✓ |
| Opaque tensor + auto-downgrade | Same opaque handle, but over-budget AUTO-switches to spill/DF with a warning instead of refusing | |
| Pre-alloc-once Vec + spill in DF only | Keep Amplitudes{t2: Vec<f64>}, allocate Wabef-class buffers once via WorkspacePool; HDF5 spill only in the DF-CCSD path | |

**User's choice:** Opaque spillable tensor + hard refuse
**Notes:** Most faithful reading of CCSD-11 "from day one, not retrofitted" — spill becomes a backend swap behind the handle. Hard refusal makes the memory-vs-accuracy tradeoff explicit (user opts into DF/`direct`). Drives the upgrade of the `pyscf-core::Amplitudes` Vec fields to opaque `Tensor` handles and the fill of `WorkspacePool` (D-01, D-08).

---

## ERI-mode scope & sequencing (CCSD-07 / CCSD-08)

| Option | Description | Selected |
|--------|-------------|----------|
| In-core headline, direct+DF as sequenced plans | In-core RCCSD/UCCSD is the un-gated numeric headline (int2e real post-05-08); AO-direct + DF-CCSD+spill land as explicit follow-on plans/waves — Phase-5 MVP sequencing | ✓ |
| All three co-equal first-class | Plan AO-direct/in-core/DF-CCSD as parallel first-class deliverables, no headline primacy | |
| In-core + DF only; AO-direct minimal/structural | In-core + DF-CCSD are the real numeric deliverables; mycc.direct=True is a thin structural path | |

**User's choice:** In-core headline, direct+DF as sequenced plans
**Notes:** All three ERI modes ship this phase, ordered not co-equal. In-core proves out first on the now-real `int2e` (cintx#11 closed in-tree by 05-08), then AO-direct, then DF-CCSD+HDF5-spill (D-02).

---

## Λ-equations & RDM depth (CCSD-05 / CCSD-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Full numeric λ + RDMs, ao_repr deferred | solve_lambda + make_rdm1/2 numeric in MO basis this phase; make_rdm2(ao_repr=True) deferred to Phase-7 (the Phase-5 MP2 precedent) | |
| Full numeric λ + RDMs incl. ao_repr | Everything numeric this phase including the nmo^4 AO back-transform for make_rdm2(ao_repr=True); no Phase-7 carry-over | ✓ |
| λ structural/forward-facing, RDMs full | make_rdm1/2 full numeric; solve_lambda structural until Phase-7 gradients consume it | |

**User's choice:** Full numeric λ + RDMs incl. ao_repr
**Notes:** Deliberate departure from Phase-5's ao_repr deferral. CCSD RDMs are the heaviest tensor-arena tenant (best CCSD-11 stress test), and Phase-7 CCSD gradients want a complete, validated λ + RDM surface (D-03).

---

## Numeric oracle corpus (CCSD-01 / CCSD-11)

| Option | Description | Selected |
|--------|-------------|----------|
| Tiered: small in-tree, caffeine CI/human-verify | Always-on in-tree bit-exact on small systems (H2O/cc-pVDZ ± water-dimer); caffeine/cc-pVDZ + benzene-dimer DF-CCSD spill on a CI/human-verify arm (02-10/05-08 precedent) | ✓ |
| Push caffeine in-tree (memory-bounded) | Make caffeine/cc-pVDZ (with spill) an always-on in-tree gate | |
| Synthetic-ERI in-tree + real small + caffeine CI | Three tiers: synthetic-ERI roundtrip + real small bit-exact + caffeine/DF-spill CI | |

**User's choice:** Tiered: small in-tree, caffeine CI/human-verify
**Notes:** Keeps `cargo test` fast and green; caffeine (Wabef ≈ multi-GB) + the DF-CCSD spill proof run as workflow_dispatch/human-verify, honoring the "don't freeze the test run" user-memory constraint (D-04).

---

## Claude's Discretion

The user confirmed (by not objecting) these precedent-resolved items, left to researcher/planner within the locked decisions:

- **Port targets** — `ccsd.CCSD` (in-core RHF), `uccsd.UCCSD`, `dfccsd.RCCSD`/`dfuccsd.UCCSD` (DF); NOT the spin-orbital `rccsd.RCCSD` (D-05).
- **Amplitude-DIIS** — reuse `pyscf-diis` via a new `AmplitudeSubspace: DiisStorable`, default `diis_space=6`, re-validates Pitfall 9 (D-06).
- **HDF5 spill** — reuse the `pyscf-chkfile` re-exported `hdf5` alias, no new `hdf5-metno` dep (D-07).
- **Spillable `Tensor` location** — `pyscf-runtime` next to `WorkspacePool`; `pyscf-core::Amplitudes` consumes it (D-08).
- **PyO3 bridge** — `PyRCCSD`/`PyUCCSD` + `CcsdOverrideHooks` (`ao2mo`/`update_amps`/`make_rdm1`/`make_rdm2`/`energy`) + `mf.CCSD()`/`mf.density_fit().CCSD()` factory + `as_scanner` (D-09).
- **Intermediates / update_amps / init-amps / convergence defaults / T1-D1-D2 diagnostics / frozen-core** — mirror upstream (`rintermediates.py`/`uintermediates.py`/`ccsd.py`); reuse Phase-5 frozen helpers verbatim.

## Deferred Ideas

- CCSD(T) perturbative triples (v1.x P1).
- EOM-CC excited states (separate milestone).
- GCCSD / GHF-reference CC, FNO-CCSD (v1.x).
- QCISD / BCCD / CCD (no v1 REQ).
- Fused cubecl CCSD kernel (Phase 8).
- CCSD analytical gradients (Phase 7 GRAD-06 — consumes the λ produced here).
- Higher-order CC (CC3/CCSDT/CCSDTQ) — out of v1.
