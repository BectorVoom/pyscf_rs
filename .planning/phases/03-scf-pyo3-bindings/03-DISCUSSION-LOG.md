# Phase 3: SCF + PyO3 bindings - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `03-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 3 — SCF + PyO3 bindings
**Areas discussed:** PyO3 subclass-override dispatch, HDF5 chkfile crate choice, DIIS home & Phase 6 reuse, Density-fitting (DF-HF) home

---

## Gray-area selection

| Option | Description | Selected |
|--------|-------------|----------|
| PyO3 subclass-override dispatch shape | How Rust calls back into Python overrides every cycle; the heart of Pitfall 7 / BIND-07 / SCF-08 | ✓ |
| HDF5 chkfile crate choice | hdf5-metno vs hdf5 vs hdf5-sys; SCF-10 + ORACLE-08 + Pitfall 11; round-trip oracle | ✓ |
| DIIS implementation home & Phase 6 reuse model | Where C-DIIS lives + reuse seam for CCSD-04 amplitude DIIS | ✓ |
| Density-fitting (DF-HF) crate home | DF machinery for SCF-07 + Phase 4/5/6 DF-* reuse | ✓ |

**User's choice:** All four areas.

---

## Area 1: PyO3 subclass-override dispatch shape

### Q1: Dispatch strategy for the 10 overrideable SCF hooks?

| Option | Description | Selected |
|--------|-------------|----------|
| Trait-callback bridge | pyscf-scf declares pub `OverrideHooks` trait (zero pyo3 dep); pyscf-py impls via `slf.call_method1`. Python MRO does subclass dispatch; #[pymethods] default forwards to Rust default. Pitfall 7 immune by construction. Matches PyO3 0.28 trait-bounds.md verbatim. | ✓ |
| Direct `call_method1` in pyscf-scf | pyscf-scf depends on pyo3 directly; holds `Py<PyAny>` of self; SCF loop calls `self.model.bind(py).call_method1(...)` inline. Simpler but breaks the convention that method crates are pyo3-free. | |
| Detect overrides upfront + Rust fast path | At kernel() entry, scan `slf.get_type().__mro__`; cache which hooks are overridden; skip call_method1 for non-overridden hooks. Faster for the 95% case but subtle correctness — if user mutates class mid-kernel it breaks. | |
| Per-hook policy | Cheap hooks always Rust fast path; expensive hooks always via call_method1. Two code paths. | |

**User's choice:** Trait-callback bridge.
**Rationale captured in:** D-01.

### Q2: pyscf-scf Rust-only public API exposure?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — pub trait + pub kernel | `OverrideHooks` pub; `pyscf_scf::RHF::kernel<H: OverrideHooks>(mol, hooks)` generic. Rust-only callers work without pyo3. Aligns with DIST-01. | ✓ |
| No — pyscf-scf internal | `OverrideHooks` pub(crate) or behind feature flag; pyscf-scf has no Rust-API promise. Simpler short-term but DIST-01 crates.io publication becomes thin re-export. | |
| Pub trait, but kernel takes `Py<PyAny>` | Compromise that defeats the trait-bridge purpose. | |

**User's choice:** Yes — pub trait + pub kernel.
**Rationale captured in:** D-02.

### Q3: GIL release granularity (`Python::detach`)?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-hook detach | Each heavy hook body wraps compute in `Python::detach`. Override call sites stay GIL-attached. Phase 4 (XC eval) and Phase 6 (CCSD doubles update) inherit the seam. python3.13t CI probes. | ✓ |
| Coarse kernel-level detach | One `Python::detach` for the whole loop; every override call site uses `Python::attach`. Larger surface; broader deadlock risk. | |
| Defer GIL seam to Phase 8 | Phase 3 ships with GIL held; Phase 8 retrofit. Risk: Phase 4-6 inherit broken seam. | |
| Per-iteration detach | One detach per iteration body; doesn't compose well with the trait-callback bridge. | |

**User's choice:** Per-hook detach.
**Rationale captured in:** D-03.

### Q4: NumPy ↔ Rust type conversion at PyO3 boundary (BIND-04, Pitfall 5)?

| Option | Description | Selected |
|--------|-------------|----------|
| Type-specific converters in pyscf-py | `to_density(arr) -> Density`, etc. — each runs `is_standard_layout()` + `to_owned()` before constructing pyscf-core type. Output helpers always C-contig (or F per pyscf/gto/moleintor.py per-name from Phase 2). Greppable in CI; BIND-04 stride-fuzz test asserts `a`, `a.T`, `a[::2]`, `a[:, 1:5]` identical. | ✓ |
| Generic `as_contig<T, D>` helper | Single generic helper at every entry point; type conversion separate concern. More flexible, more verbose. | |
| Hybrid (type-specific + generic) | Type-specific for canonical state, generic for ad-hoc args. Two patterns to teach. | |
| Proc-macro `#[contiguify]` | Most ergonomic at use site; adds proc-macro build cost (conflicts with no-compile-freeze preference). | |

**User's choice:** Type-specific converters in pyscf-py.
**Rationale captured in:** D-04.

**Pacing check after Area 1:** "Next area (HDF5 chkfile)" — user moved on.

---

## Area 2: HDF5 chkfile crate choice

### Q1: Rust HDF5 binding crate?

| Option | Description | Selected |
|--------|-------------|----------|
| hdf5-metno | Maintained metno fork. Static linking via `hdf5-metno-sys/static` satisfies DIST-05. STATE.md already names it. | ✓ |
| hdf5 (original aldanor) | Last release Aug 2022; slow maintenance; only safe if vendored+patched. | |
| hdf5-sys + thin hand-rolled wrapper | Maximum control; 200-500 LoC of unsafe ffi; Phase 8 retrofit risk on edge cases. | |
| Defer to oracle-via-pytest (no Rust HDF5) | Chkfile delegated to Python via PyO3; breaks DIST-01 Rust-API expectation. | |

**User's choice:** hdf5-metno.
**Rationale captured in:** D-05.

### Q2: Chkfile code home + cross-phase reuse?

| Option | Description | Selected |
|--------|-------------|----------|
| New `pyscf-chkfile` crate + per-method schema modules | 16th workspace member. Owns sole hdf5-metno dep. `Checkpointable` trait; per-method schemas in each method crate's chkfile.rs. Workspace 15→16. | ✓ |
| Inside `pyscf-runtime` | Adds HDF5 dep to runtime crate (consumed by all); bloats compile time; conflicts with no-compile-freeze. | |
| Inside `pyscf-scf` only | Phase 4-7 reach in; geomopt depends on scf just for HDF5 — backwards coupling. | |
| Inside `pyscf-core` | Violates FOUND-02 (zero compute/heavy deps in pyscf-core). | |

**User's choice:** New `pyscf-chkfile` crate + per-method schema modules.
**Rationale captured in:** D-06.

### Q3: Round-trip oracle test driver direction (ORACLE-08)?

| Option | Description | Selected |
|--------|-------------|----------|
| Rust-driven via pyscf-oracle | `oracle_check!("chkfile_roundtrip", fixture)` macro in pyscf-oracle (pyo3 already in dev-deps); spawns Python via Python::attach for both directions. Consistent with ROADMAP success criterion 6. | ✓ |
| Python-driven via pytest | pytest fixtures call both upstream and pyscf-rs; reuses Phase 2 pytest scaffolding; Rust-side test coverage relies on Python runner. | |
| Both | Redundant; double maintenance. | |

**User's choice:** Rust-driven via pyscf-oracle.
**Rationale captured in:** D-07.

**Pacing check after Area 2:** "Next area (DIIS home)" — user moved on.

---

## Area 3: DIIS implementation home & Phase 6 reuse model

### Q1: DIIS crate home?

| Option | Description | Selected |
|--------|-------------|----------|
| New `pyscf-diis` crate | 17th workspace member. Generic over `DiisStorable` trait. Depends only on pyscf-algebra. SCF + CCSD consume. Workspace 16→17. | ✓ |
| Inside `pyscf-algebra` as `oracle_diis` primitive | Avoids new crate; pyscf-algebra grows from primitives to iterative-method building blocks. Role drifts. | |
| Inside `pyscf-scf` only | Phase 6 copies; duplication; Pitfall 9 risk surface doubles. | |
| Inside `pyscf-core` with trait + per-crate impls | Trait surface out-of-band from any impl; awkward. | |

**User's choice:** New `pyscf-diis` crate.
**Rationale captured in:** D-08.

### Q2: Storage abstraction for Fock (Phase 3) vs (T1, T2) tuple (Phase 6)?

| Option | Description | Selected |
|--------|-------------|----------|
| Generic `DiisStorable` trait | `trait DiisStorable { as_flat, from_flat, dot }`. pyscf-scf::FockSubspace impls for Fock; pyscf-ccsd::AmpsSubspace impls for (T1,T2). Trait object-safe. | ✓ |
| Concrete `DiisStack<Vec<f64>>` | Caller flattens; layout knowledge pushed to callers; copying overhead risk. | |
| Two separate impls (no shared code) | Defeats the reason for the crate. | |

**User's choice:** Generic `DiisStorable` trait.
**Rationale captured in:** D-09.

**Pacing check after Area 3:** "Last area (DF-HF home)" — user moved on.

---

## Area 4: Density-fitting (DF-HF) crate home

### Q1: DF machinery home?

| Option | Description | Selected |
|--------|-------------|----------|
| New `pyscf-df` crate | 18th workspace member. Mirrors upstream pyscf/df/ exactly. SCF/DFT/MP2/CCSD consume uniform `DfIntegrals`. Workspace 15→18 across Phase 3. | ✓ |
| In `pyscf-gto` | Auxbasis already in gto; 3-center is another intor; widens pyscf-gto's role; drifts from upstream gto vs df separation. | |
| Hybrid: wrapper in pyscf-scf + heavy kernel in pyscf-kernels | Mirror of eval_gto split; splits DF concept across two crates. | |
| In `pyscf-scf` only | Phase 4/5/6 reach into scf; worst long-term coupling (DFT depends on SCF for DF, conceptually backwards). | |

**User's choice:** New `pyscf-df` crate.
**Rationale captured in:** D-10.

### Q2: B-integral storage strategy?

| Option | Description | Selected |
|--------|-------------|----------|
| In-memory only Phase 3; Phase 6 adds HDF5 spill | Sufficient for Phase 3 test corpus; aligns with CCSD-08 + CCSD-11 explicit Phase 6 placement. No premature optimization. | ✓ |
| HDF5 spill from day one in Phase 3 | Future-proof but more Phase 3 work; spec drift risk designing for Phase 6 needs without Phase 6 in front of us. | |
| PYSCF_MAX_MEMORY-aware auto-switching | Most polished UX; same timing question hidden behind runtime check; most complexity. | |

**User's choice:** In-memory only in Phase 3; Phase 6 adds HDF5 spill.
**Rationale captured in:** D-11.

**Final check after Area 4:** "Ready for CONTEXT.md" — user advanced to write phase.

---

## Claude's Discretion (items the user explicitly delegated)

- `init_guess` scope across plans (all 5 modes vs MVP-first)
- `canonicalize_signs` location (pyscf-core::lib per REQUIREMENTS.md SCF-13 confirmation)
- `python/pyscf/__init__.py` overlay mechanism (maturin `python-source` likely)
- `GILOnceCell` migration sites (preventive lint; no existing `lazy_static!` to migrate)
- abi3-py310 wheel skeleton scope split between Phase 3 and Phase 8
- Panic → Python exception conversion shape (BIND-09)
- `mf.as_scanner()` shape (mirrors upstream)
- Cross-module dispatch helper stubs (`to_uks`/`to_rks` return NotYetImplemented{phase:4})
- 30-attribute SCF floor enumeration
- Test corpus tiering (PR-CI vs nightly)
- DF 3-center kernel home (no new cubecl kernel needed; cintx + algebra GEMM suffice for Phase 3)
- DIIS B-matrix linear solver (host-faer via pyscf-algebra::solve_linear)

## Deferred Ideas

(See CONTEXT.md `<deferred>` section — 14 items captured.)
