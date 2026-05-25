# Phase 5: MP2 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-23
**Phase:** 5-mp2
**Areas discussed:** AO→MO transform crate, Phase-5 ERI scope, DF-MP2 + cintx gap, MP2 PyO3 surface

---

## AO→MO transform crate

### Where should the AO→MO transformation live (must be reusable by CCSD)?

| Option | Description | Selected |
|--------|-------------|----------|
| New pyscf-ao2mo crate | Dedicated workspace member (19→20), mirrors upstream pyscf/ao2mo/ + own-crate-per-shared-concern (D-05/D-10); CCSD imports directly, no CCSD→MP2 dep | ✓ |
| Inside pyscf-mp2 | Keep in pyscf-mp2; CCSD imports from there — creates a backwards CCSD→MP2 crate edge | |
| Inside pyscf-algebra/kernels | Generic primitive in the algebra layer — mixes chemistry-specific transform into the backend-agnostic surface | |

**User's choice:** New pyscf-ao2mo crate
**Notes:** Keeps the Phase-6 dependency DAG clean; consistent with upstream keeping ao2mo separate from mp/.

### What public surface should pyscf-ao2mo expose?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror upstream ao2mo | general transform + incore.full + MP2 (ov|ov) helper; CCSD reuses general/full day one | ✓ |
| MP2-only ovov now | Only the transform MP2 needs; generalize when CCSD lands — risks the retrofit the ROADMAP wording prevents | |
| General typed API (not upstream-shaped) | Fresh Rust API; cleaner ergonomics but diverges from sibling-fidelity + oracle mapping | |

**User's choice:** Mirror upstream ao2mo
**Notes:** Sibling-crate fidelity hard preference; gives CCSD the general/full kernel with no retrofit.

### How should the AO→MO contraction be computed in v1?

| Option | Description | Selected |
|--------|-------------|----------|
| Algebra gemm chain, defer fusion | gemm chains through pyscf-algebra, no new cubecl kernel (D-07/D-10 precedent); fused kernel → Phase 8 | ✓ |
| Fused cubecl ao2mo kernel now | Dedicated cubecl transform kernel from day one — front-loads optimization Phase 8 owns | |

**User's choice:** Algebra gemm chain, defer fusion

---

## Phase-5 ERI scope

### What ERI-storage scope should Phase 5 ship?

| Option | Description | Selected |
|--------|-------------|----------|
| In-core + DF only | In-core AO→MO + DF-MP2; defer outcore/semi-incore HDF5 spill to Phase 6 (CCSD-08/11), matches D-11 | ✓ |
| Add outcore/semi-incore now | Port HDF5-spilling AO→MO in Phase 5 — pulls Phase 6's tensor-arena/spill forward, contradicts D-11 | |

**User's choice:** In-core + DF only

### How should Phase-5 MP2 treat PYSCF_MAX_MEMORY?

| Option | Description | Selected |
|--------|-------------|----------|
| Log-only, no enforcement | Log at kernel entry (Phase 3/4 convention); refusal/spill is Phase 6 CCSD-11 | ✓ |
| Preflight refusal | Refuse over-budget in-core MP2 — duplicates the enforcement CCSD-11 owns | |

**User's choice:** Log-only, no enforcement

---

## DF-MP2 + cintx gap

### How should Phase 5 handle DF-MP2 numeric parity given the open cintx int3c2e_sph gap?

| Option | Description | Selected |
|--------|-------------|----------|
| Follow the DF-HF precedent | Land DF-MP2 structural code now; CI-gate the bit-exact oracle behind the cintx int3c2e_sph merge. In-core RMP2/UMP2 (int2e, bit-exact) is the un-gated headline | ✓ |
| Block phase on cintx merge | Hard phase-completion gate — couples the timeline to an external cross-repo dependency open since Phase 2 | |
| Defer DF-MP2 entirely to a gap-closure plan | Don't write DF-MP2 now; whole feature lands once cintx merges — re-does SCF-reference plumbing later | |

**User's choice:** Follow the DF-HF precedent

### Which upstream DF-MP2 implementation is the bit-exact fidelity target?

| Option | Description | Selected |
|--------|-------------|----------|
| dfmp2.py DFRMP2 | Conventional path; what mp.DFMP2(mf) returns; subclasses RMP2; primary oracle (was the recommendation) | |
| dfmp2_native.py DFRMP2 | Native RI-MP2 fast path; distinct code path, not the default factory | |
| Both | Port conventional AND native paths | ✓ |

**User's choice:** Both
**Notes:** Deliberate scope expansion (over the recommended conventional-only). Conventional dfmp2.py/dfump2.py is the default `mp.DFMP2` factory target + primary oracle reference and subclasses RMP2; native dfmp2_native.py/dfump2_native.py is the additional fast path via its own module. Both DF paths need 3-center integrals, so both numeric oracles gate behind the same cintx int3c2e_sph merge. Planner sequences native as a follow-on, optionally behind a status marker.

---

## MP2 PyO3 surface

### How should the Rust MP2 ingest the converged SCF reference?

| Option | Description | Selected |
|--------|-------------|----------|
| Eager snapshot in pyscf-py | Extract mo_coeff/energy/occ/e_hf in pyscf-py, pass plain Rust arrays to a pyo3-free pyscf-mp2 kernel (D-01); as_scanner re-runs + re-snapshots | ✓ |
| Hold a live Python mf handle | pyscf-mp2 keeps Py<PyAny>, pulls lazily — forces a pyo3 dep into pyscf-mp2, breaks D-01 | |

**User's choice:** Eager snapshot in pyscf-py

### What subclass-override surface should MP2 expose?

| Option | Description | Selected |
|--------|-------------|----------|
| Same trait-callback bridge, focused hooks | Mp2OverrideHooks trait (pyo3-free) bridged via slf.call_method1; hooks = ao2mo + make_rdm1 + make_rdm2 + energy | ✓ |
| Full upstream hook parity | Bridge every overrideable method incl. helpers — large surface for little subclass benefit | |
| No override dispatch (plain pyclass) | Simplest; breaks the drop-in contract for subclasses overriding ao2mo/make_rdm1 | |

**User's choice:** Same trait-callback bridge, focused hooks

---

## Claude's Discretion

- Frozen-core semantics (MP2-03): mirror upstream `mp2.py` (`frozen=int/list/'auto'/window`, `get_frozen_mask`/`_mo_without_core`/`_mo_splitter`); confirm the `'auto'` core table source.
- SCS-MP2 (MP2-06): mirror upstream SS/OS energy split with `emp2_ss_factor`/`emp2_os_factor`.
- `make_rdm1`/`make_rdm2` surface (MP2-05): mirror upstream incl. `ao_repr`/`with_frozen` flags.
- MP2-08 helper export call site: match `cc/ccsd.py:35` import semantics; contract test mimics it.
- `with_t2` / amplitude retention default; `mp.MP2()` cross-module factory dispatch; DF auxbasis defaults; `canonicalize_signs` reuse from the reference; phase MVP wave sequencing.

## Deferred Ideas

- Fused cubecl AO→MO kernel — Phase 8 (only if profiling demands).
- Outcore/semi-incore HDF5-spilling AO→MO — Phase 6 (CCSD-08/11).
- PYSCF_MAX_MEMORY budget-aware refusal/spill — Phase 6 CCSD-11.
- GMP2 / DFGMP2 (GHF-reference MP2) — no v1 REQ maps to them.
- MP2-F12 — out of v1.
- FNO-MP2 / `make_fno` — not in MP2-01..08.
- `_iterative_kernel` (non-canonical/Brueckner MP2) — likely out of v1 scope.
- cintx `int3c2e_sph` gap-closure (cintx#11) — cross-repo; unblocks the DF-MP2 oracle when merged.
