# Phase 2: GTO - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `02-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 02-gto
**Areas discussed:** Basis file packaging, Mole ↔ cintx bridge, eval_gto kernel home, ECP scope

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Basis file packaging | Where the 207 built-in basis files (and ALIAS table) physically live for the Rust crate | ✓ |
| Mole ↔ cintx bridge | How _atm/_bas/_env (flat) are produced from cintx_core::BasisSet (typed), satisfying both GTO-04 byte-identity and GTO-11 no-parallel-structure | ✓ |
| eval_gto kernel home | Where eval_gto (6 variants, hot path for Phase 4 DFT grids) lives | ✓ |
| ECP scope in Phase 2 | How GTO-05 (int1e_ecp bit-exact) is satisfied given cintx has no ECP yet | ✓ |

---

## Basis File Packaging

### Source-of-truth (Q1)

| Option | Description | Selected |
|--------|-------------|----------|
| Freeze at build time from upstream | build.rs walks pyscf/gto/basis/ at compile time and bakes the files + ALIAS table into a generated Rust module (include_bytes! / const arrays) | |
| Freeze via build.rs that parses to Rust LUT | (REJECTED before presenting — violates "don't freeze compile" preference) | |
| Read live from upstream tree at runtime | Resolve path to pyscf/gto/basis/ at runtime via env var with repo-relative fallback; lazy parse on first use behind OnceLock | ✓ |
| include_bytes! the upstream files (no parsing at compile time) | Embed raw bytes into binary; parse lazily at runtime | |
| Vendor copy in pyscf-gto/data/, read from disk at runtime | Copy 207 files into crates/pyscf-gto/data/, read at runtime, parse lazily | |

**User's choice:** Read live from upstream pyscf/gto/basis/ at runtime
**Notes:** User feedback "You should not freeze compile" eliminated build.rs codegen options. Saved as persistent feedback memory `feedback_no_compile_freeze.md`.

### Path resolver (Q2)

| Option | Description | Selected |
|--------|-------------|----------|
| PYSCF_BASIS_PATH env var, then repo-relative fallback | Priority chain: env var → walk-up from CARGO_MANIFEST_DIR/current_exe() → error with named env var | ✓ |
| Ask the upstream Python pyscf install | Shell out to `python -c 'import pyscf; print(...)'` once and cache | |
| Repo-relative only, no env var | Walk up from current_exe() looking for upstream pyscf/ tree | |

**User's choice:** PYSCF_BASIS_PATH env var, then repo-relative fallback
**Notes:** Mirrors the PYSCF_BACKEND env-var pattern locked in Phase 1 D-07.

---

## Mole ↔ cintx Bridge

### Projection timing (Q3)

| Option | Description | Selected |
|--------|-------------|----------|
| Eager on mol.build() | mol.build() constructs cintx_core::BasisSet once, then derives flat _atm/_bas/_env/ao_loc_nr/nao_nr arrays into Mole fields | ✓ |
| Lazy on first access via OnceLock | mol.build() only constructs cintx BasisSet; getters lazily project on first call, memoize via OnceLock | |
| Push projection into cintx-compat | Propose adding a typed→raw outbound API (RawAtmView::from_basis(&BasisSet)) to cintx-compat | |

**User's choice:** Eager on mol.build()
**Notes:** Single source of truth = cintx_core::BasisSet (typed, Arc-shared, satisfies GTO-11 zero-copy). Flat arrays are derived caches. Byte-identity via direct assert_eq!. Reuse cintx-compat::raw slot constants — no duplicate libcint layout.

---

## eval_gto Kernel Home

### Crate placement (Q4)

| Option | Description | Selected |
|--------|-------------|----------|
| pyscf-kernels (separate kernels crate) | pyscf-kernels owns the cubecl-driven eval_gto; pyscf-gto wraps; mirrors cintx-cubecl/xcfun-kernels split | ✓ |
| pyscf-gto (in-crate) | Both wrapper and kernel in pyscf-gto, depends on pyscf-algebra for compute | |
| Defer — stub in Phase 2, implement in Phase 4 DFT plan | Phase 2 ships eval_gto as NotYetImplemented; Phase 4 DFT plan implements | |

**User's choice:** pyscf-kernels (separate kernels crate)
**Notes:** Sibling-crate fidelity (cintx-cubecl/cintx-rs, xcfun-kernels/xcfun-rs). Phase 4 DFT imports pyscf-kernels directly. pyscf-gto stays cubecl-free per algebra wall.

---

## ECP Scope in Phase 2

### Resolution strategy (Q5)

| Option | Description | Selected |
|--------|-------------|----------|
| Loading-only in Phase 2; amend GTO-05 to defer evaluation | Parse ECP files, populate _ecpbas, intor('int1e_ecp') returns NotYetImplemented; amend GTO-05 in REQUIREMENTS.md | |
| Push int1e_ecp upstream into cintx | Propose adding ECP integral support to cintx; Phase 2 gets EcpEngine surface from cintx | ✓ |
| Implement int1e_ecp in pyscf-kernels | Port cint1e_ecp (Type-1/Type-2 projectors) into pyscf-kernels as a domain kernel | |
| Drop GTO-05 from Phase 2 entirely | Move GTO-05 to v1.x | |

**User's choice:** Push int1e_ecp upstream into cintx
**Notes:** ECP belongs in cintx (the libcint-replacement crate); separate workstream lands int1e_ecp Type-1+Type-2 projectors there.

### Sequencing (Q6)

| Option | Description | Selected |
|--------|-------------|----------|
| Parallel: pyscf-gto ships ECP loading + EcpEngine trait shim now; wire cintx ECP when it lands | EcpEngine trait declared; stub returns EcpEngineNotAvailable; Phase 2 gap-closure plan wires when cintx ECP merges | ✓ |
| Block Phase 2 on cintx ECP | Wait until cintx ECP is merged + a release SHA is available before starting Phase 2 | |
| Insert a Phase 2.1 (ECP) between Phase 2 and Phase 3 | Move GTO-05 to a new 2.1 phase | |

**User's choice:** Parallel
**Notes:** Phase 2 ships ahead; gap-closure plan analogous to Phase 1's 01-08-PLAN.md wires cintx ECP later.

### Trait shape (Q7)

| Option | Description | Selected |
|--------|-------------|----------|
| Separate EcpEngine trait, uniform user-facing API | EcpEngine in pyscf-core; cintx implements; pyscf-gto's intor(name) routes int1e_ecp* names internally; user keeps writing mol.intor('int1e_ecp'); Phase 7 grad hooks _ipnuc variants | ✓ |
| Extend IntegralEngine; name-string dispatch only | No new trait; intor(name) gains 'int1e_ecp' as recognised name | |
| Unified trait, ECP-aware method on IntegralEngine | Add fn supports_ecp + fn intor_with_ecp to IntegralEngine | |

**User's choice:** Separate EcpEngine trait, uniform user-facing API
**Notes:** ECP-specific contract is type-safe; cintx implements EcpEngine on its own type; non-ECP cintx versions don't need to stub anything.

---

## Claude's Discretion (Areas Deferred to Planner)

The following implementation details were noted as discretion-level during discussion:
- ALIAS table porting strategy (hand-port to Rust static OnceLock<HashMap>)
- USER_BASIS_DIR / USER_BASIS_ALIAS override semantics
- Parser-dispatch shape (parse_nwchem vs parse_cp2k routing)
- mol.cart flag handling (spherical vs cartesian AO counting in projection)
- Arc<BasisSet> vs by-value ownership inside Mole
- mol.set_geom_() cache invalidation strategy
- 6 GTOval variants priority for Phase 4 DFT
- Output layout (F-order vs C-order — match upstream per-name)
- Grid-batching strategy for cubecl dispatch
- Atom-input callable form (defer to Phase 3 with NotYetImplemented)
- mol.dumps() / loads() format (semantic round-trip + oracle interop, not byte-identical)
- Test corpus tiering (small PR-CI + nightly Phase 8 sweep)

---

## Deferred Ideas

- int1e_ecp evaluation kernel inside cintx (separate cintx workstream + Phase 2 gap-closure plan)
- Atom-input callable form (Phase 3 — needs PyO3)
- Wheel packaging of pyscf/gto/basis/ (Phase 8 DIST-02)
- mol.dumps() byte-identity to upstream's JSON (explicitly out of scope)
- Phase 4 DFT GTOval variant priority drilling
- USER_BASIS_DIR/USER_BASIS_ALIAS Python config integration (Phase 3 PyO3)
- Per-basis nightly sweep (Phase 8 ORACLE-06)
- libxc_rs re-enable (Phase 4 DFT-03)

---

## Out-of-Workflow Side Quest

During the discussion, the user requested commenting out `libxc_rs` from the workspace because pulling it into the dep graph triggers a ~6h native compile (per persistent memory `feedback_libxc_compile_time.md`). Two changes landed:
- `Cargo.toml:94` — `libxc_rs = { path = "../libxc_rs" }` patch entry commented out with a note pointing to Phase 4 (DFT) as the re-enable trigger.
- `.github/workflows/nightly-cross-crate.yml:40` — `cargo update -p` list updated to skip `libxc_rs` (note left for re-enable path).

Verification: `cargo build --workspace` (CPU-only, default features) completed in 21.19s with exit 0. Cargo.lock confirms `libxc_rs` is `[[patch.unused]]` — no crate depends on it.

This work is logged here for audit; it does not affect Phase 2 decisions but is referenced in `02-CONTEXT.md` <specifics>.
