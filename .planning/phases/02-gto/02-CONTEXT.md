# Phase 2: GTO - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

A user constructs a molecule with any of upstream PySCF's 5 atom-input forms × 11 basis-input forms, runs any 1e/2e integral upstream supports for in-scope methods (HF/DFT/MP2/CCSD/grad), and gets `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-for-byte parity with upstream on the test corpus.

**In scope:** GTO-01..11 (11 REQ-IDs). `pyscf.M(...)` factory + `gto.Mole` class accepting all 5 atom-input forms (string, list-of-tuples, list-of-lists, file path, geom callable — see Deferred for the callable form), all 11 basis-input forms, runtime resolution of the 207 built-in basis files in upstream `pyscf/gto/basis/`, ECP loading + `EcpEngine` trait shim (evaluation closed via gap-closure plan when cintx ECP lands), `mol.intor(name)` thin wrapper over cintx, `eval_gto` (6 variants: `GTOval`, `GTOval_sph`, `GTOval_deriv1`, `GTOval_deriv2`, `GTOval_ip`, `GTOval_ig`) as kernel in `pyscf-kernels` + user wrapper in `pyscf-gto`, the ≥30-attribute floor (`atom`, `basis`, `charge`, `spin`, `nelectron`, `natm`, `nbas`, `nao_nr`, `ao_loc_nr`, `_atm`, `_bas`, `_env`, …), `mol.dumps()`/`gto.Mole.loads()` JSON round-trip, `mol.copy()` deep-copy, `mol.set_geom_(new_atom)` in-place mutation, zero-copy re-export of `cintx_core::BasisSet` (GTO-11).

**Out of scope:** PyO3 wiring (Phase 3 BIND-02), `int1e_ecp` evaluation kernel inside cintx (separate cintx workstream, then a Phase 2 gap-closure plan wires through), Python `import pyscf` shim (Phase 3), maturin wheel packaging of `pyscf/gto/basis/` (Phase 8 DIST-02), the geom-callable atom-input form (deferred to Phase 3 with `NotYetImplemented`).

</domain>

<decisions>
## Implementation Decisions

### Basis file packaging (Area 1)
- **D-01:** Read built-in basis files live from upstream `pyscf/gto/basis/` at runtime. No `build.rs` codegen, no `include_bytes!` snapshot — parsing is lazy at first use, behind a `OnceLock<HashMap<String, ParsedBasis>>` cache. Constraint: chosen because of the project-wide "don't freeze compile" preference (heavy build-time parsing rejected).
- **D-02:** Path resolver uses a priority chain: (1) `PYSCF_BASIS_PATH` env var if set, (2) repo-relative walk-up from `CARGO_MANIFEST_DIR` / `current_exe()` looking for `../../pyscf/gto/basis/`, (3) error with a message naming the env var. Mirrors the `PYSCF_BACKEND` env-var pattern locked in Phase 1 D-07.

### Mole ↔ cintx bridge (Area 2)
- **D-03:** `Mole.build()` eagerly projects `cintx_core::BasisSet` (typed, `Arc`-shared, single source of truth — satisfies GTO-11 zero-copy) into derived flat arrays cached on `Mole`: `_atm: Vec<i32>`, `_bas: Vec<i32>`, `_env: Vec<f64>`, `ao_loc_nr: Vec<i32>`, `nao_nr: usize`. Eager (not lazy) so byte-identity assertions against upstream are direct `assert_eq!` checks, and `build()` semantics match upstream PySCF's "materialize everything" contract. Reuse `cintx-compat::raw` slot constants (`ATM_SLOTS=6`, `BAS_SLOTS=8`, `PTR_ENV_START=20`) — do **not** duplicate libcint slot layout knowledge.

### eval_gto kernel home (Area 3)
- **D-04:** The `eval_gto` cubecl kernel lives in `pyscf-kernels`; `pyscf-gto::eval_gto(mol, name, coords)` is the user-facing wrapper that builds the shell list and dispatches via `pyscf-algebra` (algebra-wall friendly — `pyscf-gto` never imports `cubecl-*`). Mirrors the cintx-cubecl / xcfun-kernels split established in Phase 1 D-04. Phase 4 DFT imports `pyscf-kernels` directly for grid loops (not via `pyscf-gto`).

### ECP scope (Area 4)
- **D-05:** `int1e_ecp` (and ECP gradient variants for Phase 7 GRAD-07) belong in cintx, not in pyscf-rs. cintx is the libcint replacement; the integral engine seam is the right place for ECP.
- **D-06:** Sequencing is **parallel**, not blocking. Phase 2 ships:
  1. ECP basis-file parser in `pyscf-gto` (parses NWChem-format ECP blocks from `.dat` files like LANL2DZ, SBKJC).
  2. `EcpEngine` trait declared in `pyscf-core` (D-07).
  3. Stub impl that returns `EcpEngineNotAvailable` until cintx ECP lands.

  A separate cintx workstream (out of pyscf-rs scope) ships `cint1e_ecp` Type-1 + Type-2 projectors. When that lands, a Phase 2 gap-closure plan (analogous to Phase 1's `01-08-PLAN.md`) bumps the cintx pin + wires the cintx-side `EcpEngine` impl. GTO-05 effectively splits into "loading shipped in Phase 2" + "evaluation closed by gap-closure plan."
- **D-07:** `EcpEngine` is a separate trait in `pyscf-core` (not an extension to `IntegralEngine`). Surface includes `ecp_int1e(&self, mol, basis) -> Result<...>` plus `_ipnuc` variants for Phase 7 ECP gradients. The user-facing API stays uniform — `mol.intor("int1e_ecp")` — because pyscf-gto's `intor` dispatcher routes `int1e_ecp*` names to `EcpEngine` internally. Cintx implements both `IntegralEngine` and `EcpEngine` on the same engine type; non-ECP cintx versions simply don't impl `EcpEngine`.

### Claude's Discretion

The following are not user-decided — researcher / planner picks the implementation:

- **ALIAS table porting** — hand-port upstream's `pyscf/gto/basis/__init__.py::ALIAS` dict to a Rust `static OnceLock<HashMap<&'static str, &'static str>>` (~hundreds of entries). Source-of-truth lives in the Python module; oracle test catches drift.
- **`USER_BASIS_DIR` / `USER_BASIS_ALIAS` override semantics** — upstream supports user-supplied basis dirs and alias overrides via `pyscf.__config__`. Planner picks whether to surface these as `PYSCF_USER_BASIS_DIR` env vars, a programmatic API on `Mole`, or both.
- **Parser-dispatch shape** — upstream routes through `parse_nwchem`, `parse_nwchem_ecp`, `parse_cp2k`, `parse_cp2k_pp`. Planner picks whether to mirror as 4 separate Rust modules, one dispatcher with format-detect, or a single parser with format-aware tokens.
- **`mol.cart` flag handling** — spherical (default) vs cartesian AO counting in the projection: `(2l+1)` vs `(l+1)(l+2)/2` per shell. Both modes must work. Planner picks how to thread the flag through `_bas` `kappa` slot.
- **`Arc<BasisSet>` vs by-value ownership of cintx structure inside Mole** — `Arc` is the obvious answer given GTO-11 zero-copy intent, but planner confirms.
- **`mol.set_geom_(new_atom)` cache invalidation** — when geometry mutates, `_atm` rows must update but `_bas`/basis structure is preserved. Planner picks a granular invalidation strategy.
- **6 GTOval variants priority for Phase 4 DFT** — `GTOval_sph` and `GTOval_sph_deriv1` are the hot paths for DFT grid integration; `GTOval_ig`/`GTOval_ip` are gradient-side. Planner can ship in priority order if Phase 2 size is tight.
- **Output layout (F-order vs C-order)** — match upstream PySCF's per-integral convention (most are F-order). Planner consults `pyscf/gto/moleintor.py` per-name for the truth and documents in code.
- **Grid-batching strategy for cubecl dispatch** — eval_gto over 10000 grid points × 100 AOs is one big kernel launch vs chunked. Planner consults `docs/manual/Cubecl/` and the cubecl-matmul precedent.
- **Atom-input "callable" form (5th form)** — deferred to Phase 3 (needs PyO3); Phase 2 stub returns `NotYetImplemented { phase: 3, what: "atom callable form (GTO-01.5)" }`.
- **`mol.dumps()` / `gto.Mole.loads()` format** — semantic round-trip + oracle interop tested (PySCF writes → pyscf-rs reads + reverse). Not byte-identical to upstream's JSON string formatting (fragile, GTO-09 doesn't require it). Planner picks `serde_json` vs hand-rolled writer.
- **Test corpus tiering** — small (H2O/cc-pVDZ + benzene/6-31G* + water-trimer) for PR-CI, full 5×11 atom×basis grid for nightly. Phase 8 already has a per-basis sweep slot (ORACLE-06).

### Folded Todos

None — the cross-reference scan found 0 pending todos for Phase 2.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — vision, core value, key decisions table
- `.planning/REQUIREMENTS.md` — Phase 2 owns GTO-01..11 (11 REQs)
- `.planning/ROADMAP.md` §"Phase 2: GTO" — goal, dependencies, success criteria (5 items)
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns" — algebra-responsibility wall, bit-exact-with-PySCF, panic policy, scope-creep lint inherited from Phase 1
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" — Phase 2 owns Pitfall 8 (loop / F-order layout), Pitfall 17 (off-by-one basis indexing), Pitfall 18 (Boys-function accuracy — delegated to cintx)
- `.planning/STATE.md` §"Blockers/Concerns" — cubecl 0.10.0 lockstep, cintx ECP coordination (new for Phase 2)
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01..15 carried forward (workspace layout, AlgebraClient shape, cubecl pin, env-var pattern, sibling-crate sourcing)

### Upstream PySCF (this repo, the reference implementation + oracle)
- `pyscf/gto/mole.py` (4383 lines) — `M()` factory at L106, `format_atom` L320, `format_basis` L418, `make_atm_env` L961, `make_bas_env` L984, `make_env` L1029, `make_ecp_env` L1107, `tot_electrons` L1162, `copy` L1188, `dumps` L1251, `loads` L1293, `nao_nr` L1378, `nao_cart` L1386. The reference for every shape decision in `Mole`.
- `pyscf/gto/basis/__init__.py` — `ALIAS` dict (source-of-truth for D-01 hand-port), `load(basis, symb)`, `parse(...)` entry points. ~hundreds of ALIAS entries.
- `pyscf/gto/moleintor.py` (889 lines) — `intor` dispatcher reference for D-04/D-07 wrapper shape; per-name F-order vs C-order convention.
- `pyscf/gto/eval_gto.py` (248 lines) — eval_gto reference for the 6 variants in D-04.
- `pyscf/gto/ecp.py` (186 lines) — ECP loading reference for the parser in D-06.
- `pyscf/gto/basis/` — 207 `.dat` files resolved at runtime per D-02.

### cintx workspace (sibling repo, the integral engine)
- `~/Documents/workspace/cintx/crates/cintx-core/src/basis.rs` — `BasisSet { atoms: Arc<[Atom]>, shells: Arc<[Arc<Shell>]>, meta: BasisMeta }` — typed structure pyscf-rs re-exports zero-copy per GTO-11 + D-03.
- `~/Documents/workspace/cintx/crates/cintx-core/src/atom.rs` — `Atom`, `NuclearModel` types.
- `~/Documents/workspace/cintx/crates/cintx-core/src/shell.rs` — `Shell`, `ShellTuple` types.
- `~/Documents/workspace/cintx/crates/cintx-compat/src/raw.rs` — libcint slot constants (`ATM_SLOTS=6`, `BAS_SLOTS=8`, `PTR_ENV_START=20`, `CHARGE_OF`, `PTR_COORD`, `NUC_MOD_OF`, `ATOM_OF`, `ANG_OF`, `NPRIM_OF`, `NCTR_OF`, `KAPPA_OF`, `PTR_EXP`, `PTR_COEFF`). Reuse for the D-03 projection — do not duplicate.
- `~/Documents/workspace/cintx/crates/cintx-rs/src/api.rs` — `SessionRequest`, `SessionQuery`, `IntegralTensor` — the user-facing cintx API the `mol.intor(...)` wrapper consumes.
- `~/Documents/workspace/cintx/crates/cintx-compat/src/raw.rs` `RawApiId` enum — the catalogue of integral names cintx supports (`int1e_ovlp_sph`, `int2e_sph`, `int3c2e_ip1_sph`, …). Reference for which `intor` names route through cintx.

### pyscf-rs codebase (this repo, Phase 1 stubs)
- `crates/pyscf-core/src/mole.rs` — Phase 1 skeleton (`atom_coords`, `atom_charges`, `charge`, `spin`, `nelectron`); Phase 2 fills the ≥30-attribute floor.
- `crates/pyscf-core/src/basis_set.rs` — Phase 1 placeholder; Phase 2 replaces with re-export of `cintx_core::BasisSet`.
- `crates/pyscf-core/src/traits.rs` — Phase 1 declares `IntegralEngine`; Phase 2 declares the new `EcpEngine` trait per D-07.
- `crates/pyscf-kernels/` — Phase 1 stub; Phase 2 adds the eval_gto cubecl kernel per D-04.
- `crates/pyscf-gto/` — Phase 1 stub (Cargo.toml only); Phase 2 fills.
- `crates/pyscf-algebra/` — Phase 1-shipped surface; pyscf-kernels' eval_gto launches via `pyscf-algebra` per the algebra wall.

### cubecl reference docs (this repo)
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern; basis for the eval_gto kernel launch shape.
- `docs/manual/Cubecl/cubecl_macro_fanout_manual.md` — `#[cube]` element-wise pattern; basis for per-grid-point AO evaluation.
- `docs/manual/Cubecl/Cubecl_vector.md` — vectorisation primitives.

### External
- libcint `cint_bas.h` slot definitions — Phase 2 should NOT need to consult this directly because cintx-compat re-exports the relevant constants. Listed only as the ultimate authority if a discrepancy arises.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`cintx_core::BasisSet`** — typed Arc-backed basis structure. pyscf-rs re-exports zero-copy (GTO-11). All shell/atom logic lives there; pyscf-gto only adds the input-form parsers + flat-array projection.
- **`cintx_compat::raw` slot constants** (`ATM_SLOTS`, `BAS_SLOTS`, `PTR_ENV_START`, etc.) — reuse for the D-03 projection; do not duplicate libcint layout.
- **`cintx_rs::api`** — `SessionRequest`/`SessionQuery`/`IntegralTensor` is the user-facing cintx API; `mol.intor(...)` wrapper translates cintx output to upstream-PySCF-shaped arrays.
- **Phase 1 `pyscf-algebra` surface** (`gemm`, `reduce_sum`, `axpy`, `dot`, `oracle_*`) — eval_gto kernel launches via this layer (algebra wall).
- **Phase 1 `pyscf-core::Mole`/`BasisSet`/`IntegralEngine` stubs** — Phase 2 fills bodies, doesn't redesign shapes.
- **Upstream `pyscf/gto/mole.py` reference algorithms** — `format_atom`, `make_env`, `make_atm_env`, `make_bas_env` are the canonical algorithms to port (Apache-2.0, port-friendly).
- **Upstream `pyscf/gto/basis/` 207 `.dat` files** — read live at runtime per D-01/D-02; no copy in pyscf-rs.

### Established Patterns
- **Algebra wall** (Phase 1 D-04..06) — `pyscf-gto` and `pyscf-kernels` import `pyscf-algebra` only. No direct `cubecl-*` import. Enforced by the algebra dependency-wall lint.
- **Sibling-crate fidelity** — kernel/facade split mirrors cintx-cubecl/cintx-rs and xcfun-kernels/xcfun-rs. eval_gto kernel in `pyscf-kernels`, user wrapper in `pyscf-gto` (D-04) follows this.
- **Env-var resolver pattern** — `PYSCF_BACKEND` resolver in Phase 1 D-07 establishes the priority-chain shape; D-02's `PYSCF_BASIS_PATH` reuses it.
- **Bit-exact-with-upstream under `release-oracle`** — Phase 2 success criteria 1, 2, 3 all assert byte-identity against upstream PySCF on the test corpus.

### Integration Points
- **`pyscf-core::traits.rs`** — Phase 2 adds `EcpEngine` trait per D-07; existing `IntegralEngine` impl-by-cintx happens here too.
- **`crates/pyscf-gto/Cargo.toml`** — Phase 2 adds deps on `pyscf-core`, `pyscf-algebra`, `pyscf-kernels`, `cintx-core` (registry, via `[patch.crates-io]` to local path), `cintx-rs` (for the integral session API). No `cubecl-*` direct dep (algebra wall).
- **`crates/pyscf-kernels/Cargo.toml`** — Phase 2 adds dep on `pyscf-algebra` + `pyscf-core::BasisSet` re-export. No `cintx-*` dep (cintx is for integrals; eval_gto is independent).
- **Upstream `pyscf/gto/basis/`** — accessed at runtime via `PYSCF_BASIS_PATH` (D-02). pyscf-rs does not vendor or move these files.
- **`.cargo/config.toml`** — no changes; `[patch.crates-io]` Cargo.toml entries (cintx, xcfun_rs) cover sibling-crate sourcing. libxc_rs patch is commented out (not needed until Phase 4).

</code_context>

<specifics>
## Specific Ideas

- **"Don't freeze compile" is a project-wide preference** — confirmed during the basis-packaging discussion. No heavy `build.rs` codegen, no parse-N-files macros. Drives D-01 directly. See `~/.claude/projects/-home-user-Documents-workspace-pyscf-rs/memory/feedback_no_compile_freeze.md`.
- **libxc_rs is disabled in `[patch.crates-io]` and CI nightly cross-crate update** — re-enable only when Phase 4 (DFT) needs it. Edit recorded at `Cargo.toml:94` (commented entry with note) and `.github/workflows/nightly-cross-crate.yml:40` (cargo update -p list excludes libxc_rs).
- **cintx ECP coordination is a new cross-crate dependency** — Phase 2 ships ahead of cintx ECP per D-06. The follow-up gap-closure plan that wires cintx ECP through is analogous to Phase 1's `01-08-PLAN.md` (cintx clean-SHA repin). Planner should explicitly include this gap-closure plan in the Phase 2 plan list with a `Pending cintx ECP merge` status marker.
- **Sibling-crate fidelity is a hard preference** (carried from Phase 1) — kernel/facade split for eval_gto follows cintx-cubecl/cintx-rs and xcfun-kernels/xcfun-rs verbatim. Deviation requires explicit justification.
- **Method crates must NEVER touch cubecl types directly** (carried from Phase 1) — pyscf-gto and pyscf-kernels both go through pyscf-algebra. The dependency-wall lint enforces this at build time.
- **GTO-05 ECP loading + EcpEngine trait shim ships in Phase 2; int1e_ecp evaluation closes via gap-closure** (D-06). This split is intentional; do not retroactively claim GTO-05 fully shipped until the gap-closure plan completes.

</specifics>

<deferred>
## Deferred Ideas

- **`int1e_ecp` evaluation kernel inside cintx** — separate cintx workstream (Type-1 + Type-2 projectors); pyscf-rs gap-closure plan wires when available (D-06).
- **Atom-input "callable" (5th form)** — needs PyO3 (Phase 3 BIND-02). Phase 2 returns `NotYetImplemented { phase: 3, what: "atom callable form" }`.
- **Wheel packaging of `pyscf/gto/basis/`** — D-01 reads at runtime, so the maturin wheel must bundle these files (~MB). Phase 8 DIST-02 owns the wheel-content manifest.
- **`mol.dumps()` byte-identical to upstream's JSON string** — explicitly out of scope; semantic round-trip + oracle interop is the contract (Claude's discretion).
- **Phase 4 DFT GTOval variant priority** — `GTOval_sph` + `GTOval_sph_deriv1` are the hot paths; ranking can defer to Phase 4 plan if Phase 2 size is tight.
- **`USER_BASIS_DIR`/`USER_BASIS_ALIAS` config-dict overrides** — Phase 2 may stub; Phase 3 PyO3 integration is where users actually set these.
- **Per-basis nightly sweep** — already owned by Phase 8 ORACLE-06; Phase 2 only commits to the small PR-CI corpus.
- **libxc_rs re-enable** — Phase 4 (DFT-03 routes through libxc_rs).

### Reviewed Todos (not folded)
None — todo cross-reference scan returned 0 matches for Phase 2.

</deferred>

---

*Phase: 02-gto*
*Context gathered: 2026-05-10*
