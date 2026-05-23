---
phase: 02-gto
verified: 2026-05-24T10:00:00Z
status: human_needed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 11/11
  gaps_closed:
    - "GTO-02/GTO-03 general-contraction parsing: NWChem .dat parser now emits N contractions for N coefficient columns (plan 02-11 — e9fa626)"
    - "projection.rs coefficient layout mismatch: row-major flatten now matches cintx 1e/2e kernel contract (e9fa626)"
    - "minao heavy-atom caveat: H2O Tr(dm·S) 7.9 → 9.86 after parser fix (b6a9898)"
    - "DI-02-11-CINTX-NCTR, DI-02-11-CINTX-NCTR-HIGHL, DI-02-11-ECP-NCTR all RESOLVED in cintx (commit 9af2164, sibling repo)"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "eval_gto for l >= 1 shells (p, d, f, ...) and derivative variants (deriv1/deriv2/ip/ig)"
    addressed_in: "Phase 4 DFT (l >= 1 evaluation) + Phase 7 grad (ip/ig)"
    evidence: "02-VALIDATION.md GTO-07 row: l=0 green; l >= 1 pending Phase 4 DFT extension. deriv1/deriv2 NotYetImplemented{phase:4}; ip/ig NotYetImplemented{phase:7} in crates/pyscf-gto/src/eval_gto.rs:120-138."
  - truth: "Atom-input form 5 (Python callable)"
    addressed_in: "Phase 3 BIND-02"
    evidence: "format_atom.rs returns NotYetImplemented{phase:3} for AtomInput::Callable. Test callable_form_returns_not_yet_implemented_phase_3 asserts the error variant."
  - truth: "Full >=184-file builtin basis sweep (GTO-03 saturation coverage)"
    addressed_in: "Phase 8 ORACLE-06"
    evidence: "Representative subset green; full sweep behind #[ignore] in tests/builtin_basis_sweep.rs::full_alias_sweep_proves_loader_path_robust."
  - truth: "Arity >=3 intor families (int2e_sph 4-center, int3c2e, int4c1e, ...)"
    addressed_in: "Phase 3+ (cintx safe-API arity >= 3 surface)"
    evidence: "intor.rs xfail-tracked for arity > 2. Plans 05-08/05-09 closed the int2e/int3c2e gap; GTO-06 arity-2 is SATISFIED."
  - truth: "cintx nuclear attraction for li+lj > 3 (DI-02-11-CINTX-NUC-HIGHL)"
    addressed_in: "Future cintx work (rys_root3+ implementation)"
    evidence: "deferred-items.md DI-02-11-CINTX-NUC-HIGHL: affects non-DF SCF on d/f orbital bases only; does not affect minao/overlap or any current pyscf-rs numeric path."
human_verification:
  - test: "Upstream byte-identity for mol.intor('int1e_ecp') on Cu/LANL2DZ"
    expected: "pyscf-rs int1e_ecp matrix agrees with upstream PySCF to atol=1e-10 on Cu/LANL2DZ"
    why_human: "tests/oracle/test_ecp_int1e.py requires numpy + upstream pyscf venv (tests/oracle/requirements.txt). Oracle venv unavailable in default sandbox. cintx pins atol=1e-12 vs vendored PySCF nr_ecp in cintx-oracle/tests/safe_api_ecp_parity.rs (indirect byte-identity). To run: install requirements.txt, then pytest tests/oracle/test_ecp_int1e.py::test_cu_lanl2dz_int1e_ecp_byte_equal -v"
  - test: "Upstream byte-identity for _atm/_bas/_env/ao_loc_nr/nao_nr on H2O/cc-pVDZ, benzene/6-31G*, water-trimer/STO-3G"
    expected: "tests/oracle/test_byte_identity.py exits 0 — 15 byte-equal assertions (3 fixtures x 5 arrays)"
    why_human: "Requires upstream pyscf venv. Exists as tests/oracle/test_byte_identity.py."
  - test: "mol.intor() arity-2 parity vs upstream (7 names) + Pitfall 8 F-order layout check"
    expected: "tests/oracle/test_intor_oracle.py exits 0 — 7 arity-2 names green at atol=1e-10"
    why_human: "Requires upstream pyscf venv. Exists as tests/oracle/test_intor_oracle.py."
---

# Phase 02 (GTO): Verification Report — Re-verification (plan 02-11 gap closure)

**Phase Goal:** Gaussian-type-orbital infrastructure — Mole construction, basis-set loading (including general contractions), the Mole-cintx bridge, mol.intor() dispatch, eval_gto, ECP loading and evaluation, JSON round-trip.

**Verified:** 2026-05-24
**Status:** human_needed (all automated checks PASS; 3 oracle pytest items need upstream pyscf venv; unchanged from prior re-verification)
**Re-verification:** Yes — after plan 02-11 (general-contraction parser fix + cintx coefficient-layout fix + minao heavy-atom caveat closure)

---

## Executive Summary

Plan 02-11 closed the final open correctness gap in Phase 2: general-contraction support in the NWChem `.dat` parser. The root cause was a single-column truncation bug (`CurrentShell::Single` pushed only `cols[1]` and discarded `cols[2..N]`). The fix emits N contractions for an `exp + N`-column primitive block. A companion bug was also fixed: `projection.rs` was flattening the cintx `Shell` coefficients column-major while the cintx 1e/2e kernel reads them row-major (`coefficients[prim*nctr+ctr]`); for nctr=1 the layouts coincide (masked by the truncating parser), but for nctr>1 the scrambling produced wrong, asymmetric overlaps.

Downstream closure: three cintx-side issues exposed once bases load their real contraction counts (DI-02-11-CINTX-NCTR, DI-02-11-CINTX-NCTR-HIGHL, DI-02-11-ECP-NCTR) were all resolved in cintx (commit 9af2164, branch fix/general-contraction-nctr-1e) via the path-dep. All three deferred-item entries now read RESOLVED.

The minao heavy-atom caveat from plan 03-13 is closed: H2O `Tr(dm·S)` recovered from the truncated 7.9 to the correct 9.86. The plan's stated target of `Tr(dm·S) == nelec` is documented as a plan-premise error (not a code gap) — upstream minao is intentionally unnormalised (the normalization line is commented out), which is why the byte-matched H2 docstring dm itself traces to 1.976, not 2.0. The test correctly pins 9.86 (tight bound) plus `> 9.5`.

Gate result (from 02-11 SUMMARY Addendum): cintx 173 lib tests + pyscf-rs gto+scf+df+mp2 = 280 tests (0 failures) + pyscf-dft 47 lib tests. clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS; 0 libxc.

Pre-existing unrelated failure: pyscf-dft `cam_b3lyp_h2o_rsh::rsh_get_veff_dispatches_into_range_coulomb_branch` — a Phase 04 RSH stale test that expects the int2e gap which Phase 05 (05-08) closed; it now returns Ok and panics on the unwrap. This is orthogonal to Phase 2's 1e/ECP work (`two_electron.rs` was untouched by 02-11) and is NOT attributed to Phase 2.

---

## 1. Re-verification: Gaps from Previous Verification

| Previous State | Plan 02-11 Fix | Status |
|---------------|----------------|--------|
| GTO-02/GTO-03 general-contraction bug: ANO O S-block loaded nctr=1 (7 of 8 columns dropped) | `nwchem.rs` `CurrentShell::Single` now accumulates all `cols[1..]` columns; ragged blocks rejected with a Parse error; SharedSP path untouched | CLOSED (commit e9fa626) |
| projection.rs column-major flatten scrambled nctr>1 coefficient blocks | Row-major flatten `coeffs_flat[prim*nctr+ctr]` to match cintx kernel contract | CLOSED (commit e9fa626) |
| minao H2O `Tr(dm·S)` ≈ 7.9 (truncated ANO O → only 1 s-contraction for O) | Correct ANO O loads nctr=8 S-contractions → H2O Tr 9.86 | CLOSED (commit b6a9898) |
| DI-02-11-CINTX-NCTR: cintx 1e kernel summed all (ci,cj) pairs into slot (0,0) | cintx 6b14d48 rewrites kernel to accumulate one Cart block per (ci,cj) pair | CLOSED (cintx commit 6b14d48 via path-dep) |
| DI-02-11-CINTX-NCTR-HIGHL: cross-l blocks transposed for li≠lj, both>0 | cintx 9af2164 changes contraction functions to emit column-major `out[ket*nci+bra]` | CLOSED (cintx commit 9af2164 via path-dep) |
| DI-02-11-ECP-NCTR: `launch_ecp` OOB panic + column-major read for ECP nctr>1 | cintx 9af2164 nctr-aware sizing + per-contraction cart→sph scatter + `coeffs_col_major()` | CLOSED (cintx commit 9af2164 via path-dep) |

Regression check: no previously-passing tests regressed. Segmented bases (sto-3g, 6-31g, 6-31g*, cc-pvdz H) are byte-identical (nctr=1 row-major and column-major layouts coincide). Confirmed by `general_contraction.rs` regression pins (tests `sto3g_segmented_unchanged`, `six_31g_segmented_unchanged`, `six_31gs_segmented_unchanged`).

---

## 2. Plan 02-11 Must-Have Check

| # | Must-Have (from 02-11-PLAN.md frontmatter) | Status | Evidence |
|---|--------------------------------------------|--------|----------|
| 1 | NWChem `.dat` parser supports general contractions: a primitive block with `exp + N` columns loads as N separate contractions sharing exponents | VERIFIED | `nwchem.rs` lines 183-204: `n_ctr = cols.len() - 1`; `coeffs.resize(n_ctr, Vec::new())`; per-column push in loop; ragged rejection at line 189-200. Commit e9fa626. |
| 2 | ROOT CAUSE confirmed: `CurrentShell::Single` was pushing only `cols[1]` — diagnosed and fixed | VERIFIED | The prior bug (`exps.push(cols[0]); coeffs.push(cols[1])`) is replaced by the multi-column accumulator. The fix is substantive, not a stub. Commit e9fa626 diff shows 46-line change to nwchem.rs. |
| 3 | ANO O S-block loads nctr=8; ANO H gains real contractions; segmented bases (sto-3g/6-31g/6-31g*/cc-pvdz) UNCHANGED | VERIFIED | `general_contraction.rs` test `ano_o_s_block_has_eight_contractions`: asserts `s_shell.coeffs.len() == 8`. Test `ano_h_gains_its_real_contractions`: asserts `[6s,4p,3d,1f]` contraction pattern. Segmented regression tests `sto3g_segmented_unchanged`, `six_31g_segmented_unchanged`, `six_31gs_segmented_unchanged` pin nctr=1 for all shells. cc-pVDZ O latent bug (nctr=2 S-block) also captured in `ccpvdz_o_general_contraction_nctr2_pinned`. |
| 4 | minao for heavier atoms normalizes: H2O `Tr(dm·S)` recovers from ≈7.9; minao H2 byte-match still holds. DEVIATION: plan said `== nelec`; correct anchor is ≈9.86 (intentional unnormalization) | VERIFIED WITH JUSTIFIED DEVIATION — see note below | `init_guess_minao.rs` `minao_h2o_heavy_atom_normalizes_after_general_contraction_fix`: asserts `tr > 9.5` + `(tr - 9.86).abs() < 0.05`. H2 byte-match retained in `minao_h2_byte_matches_upstream_docstring`. The 9.86 value is the correct post-fix heavy-atom projection; `== nelec` would be physically wrong. |
| 5 | cintx evaluates generally-contracted (nctr>1) shells: int1e_ovlp on ANO mol is finite/symmetric/PSD-diagonal | VERIFIED | `general_contraction.rs` tests `ccpvdz_general_contraction_overlap_unit_diagonal` (exact unit diagonal, symmetry to 1e-12 for all 14 AOs) and `ano_general_contraction_overlap_finite_unit_diagonal` (unit diagonal across all l; l≤2 sub-block symmetric to 1e-9 for 49 AOs). Both tests read real cintx output — not stubs. |
| 6 | No new crate dep; libxc NEVER compiled; reductions stay oracle_sum/oracle_dot | VERIFIED | 02-11 SUMMARY key-decisions: `tech-stack: added: []`. Gate confirms 0 libxc in the full test run. |

**Deviation note — must-have #4 (`Tr(dm·S) == nelec`):** The plan's stated target was based on the incorrect premise that the ANO truncation was the only source of the 7.9 deficit. When the fix lands and minao runs correctly, upstream behavior reveals that `init_guess_by_minao` is intentionally unnormalised — the line `dm *= nelec/(dm·s).sum()` is commented out in `pyscf/scf/hf.py`. The byte-matched H2 docstring density itself traces to 1.976/2.0, confirming this is upstream behavior, not a pyscf-rs gap. The test anchors on the now-correct heavy-atom projection (9.86) which decisively exceeds the old truncated value (7.9) and is tight enough to catch regressions. This is a plan-premise correction, not a gap.

---

## 3. Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pyscf.M(...)` and 4 of 5 atom-input forms produce a Mole whose `_atm`, `_bas`, `_env`, `ao_loc_nr`, `nao_nr` match upstream byte-for-byte (GTO-01, GTO-04) | VERIFIED (automated) + HUMAN (oracle pytest) | mole_construction.rs 9 passed. byte_identity oracle: tests/oracle/test_byte_identity.py (3 fixtures x 5 arrays). Form 5 (Callable) NotYetImplemented{phase:3} — deferred per ROADMAP. |
| 2 | All built-in basis files resolve; ECP loads and `int1e_ecp` matches upstream under release-oracle; general contractions load correctly (GTO-02, GTO-03, GTO-05) | VERIFIED (automated) + HUMAN (oracle pytest) | general_contraction.rs 9/9 pass (ANO O nctr=8, segmented regression, cc-pVDZ overlap exact, ANO overlap unit-diagonal). ecp_int1e_oracle.rs 2/2 pass. |
| 3 | `mol.intor(name)` dispatches all in-scope integrals to cintx with correct F-order layout (GTO-06) | VERIFIED (arity-2 automated) + HUMAN (oracle pytest) | intor_smoke.rs 8 passed. test_intor_oracle.py (oracle venv). layout_table.rs 23 entries. |
| 4 | `eval_gto` for GTOval, GTOval_sph works for l=0 (GTO-07 partial) | VERIFIED (l=0) + DEFERRED (l>=1 Phase 4) | eval_gto_smoke.rs 8 passed. l>=1 xfail-tracked. Phase 4 plan 04-03 closed l>=1. |
| 5 | >=30 attribute floor; dumps/loads round-trip; copy/set_geom_ (GTO-08, GTO-09, GTO-10) | VERIFIED | attribute_floor.rs, dumps_loads.rs, mole_copy.rs, set_geom.rs — all passed. |

**Score: 5/5 ROADMAP Success Criteria verified (automated checks pass; 3 oracle pytest items require upstream pyscf venv).**

---

## 4. Per-Requirement Mapping (GTO-01..11)

| REQ-ID | Description | Status | Evidence |
|--------|-------------|--------|----------|
| GTO-01 | `pyscf.M(...)` + 5 atom-input forms | VERIFIED (4 shipped; 5th Callable NotYetImplemented{phase:3}) | mole_construction.rs 9 passed. REQUIREMENTS.md [x] confirmed. |
| GTO-02 | `mol.basis = ...` accepts all 11 input forms; general-contraction correctness fixed in 02-11 | VERIFIED | basis_input_forms.rs 9 passed. general_contraction.rs 9/9 pass. REQUIREMENTS.md annotation updated with 02-11 parser fix reference. |
| GTO-03 | All 207 built-in basis files resolve; gto.parse() handles Gaussian-94/NWChem; coefficient layout correct | VERIFIED (representative subset + general-contraction correctness) + DEFERRED (full sweep Phase 8 ORACLE-06) | builtin_basis_sweep.rs representative subset green. general_contraction.rs tests cover the generally-contracted path (ANO, cc-pVDZ O). REQUIREMENTS.md annotation updated with row-major layout fix reference. |
| GTO-04 | `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-identical to upstream | VERIFIED (automated structure) + HUMAN (oracle pytest) | make_env.rs ports pyscf/gto/mole.py. test_byte_identity.py (oracle venv). |
| GTO-05 | ECP loading + `int1e_ecp` evaluation match upstream under release-oracle; ECP nctr>1 now correct | VERIFIED (in-tree gate) + HUMAN (upstream byte-identity) | ecp_int1e_oracle.rs 2/2 passed. DI-02-11-ECP-NCTR RESOLVED via cintx 9af2164 (ECP eval was green against truncated LANL2DZ; now green against correct nctr=2 basis). test_ecp_int1e.py venv-gated. |
| GTO-06 | `mol.intor(name, ...)` thin wrapper over cintx, all arity-2 in-scope families | VERIFIED (arity-2) + DEFERRED (arity>=3 Phase 3+) | intor.rs 441 LoC. intor_smoke.rs 8 passed. test_intor_oracle.py (oracle venv). |
| GTO-07 | `eval_gto(mol, eval_name, coords, ...)` for 6 variants | VERIFIED (l=0) + DEFERRED (l>=1 Phase 4; deriv/ip/ig Phase 4/7) | eval_gto.rs + pyscf-kernels cubecl kernel. eval_gto_smoke.rs 8 passed. |
| GTO-08 | Mole exposes >=30 attribute floor | VERIFIED | pyscf-core/src/mole.rs 30 pub fields + 7 methods. attribute_floor.rs 1 passed. |
| GTO-09 | `mol.dumps()`/`Mole::loads()` JSON round-trip | VERIFIED | dumps_loads.rs 3 passed. test_json_interop.py (oracle venv). |
| GTO-10 | `mol.copy()` deep-copy + `mol.set_geom_(new_atom)` in-place | VERIFIED | mole_copy.rs 2 passed (Arc identity preserved). set_geom.rs 5 passed. |
| GTO-11 | Zero-copy re-export of `cintx_core::BasisSet` | VERIFIED | pyscf-core/src/basis_set.rs: `pub use cintx_core::BasisSet`. cintx_zerocopy.rs 6 passed (Arc::ptr_eq across clone, set_geom_, repeat calls). |

**Score: 11/11 requirement IDs satisfied or explicitly deferred with roadmap coverage.**

Cross-reference check: All 11 IDs (GTO-01..GTO-11) appear in the plans' frontmatter `requirements` fields across plans 02-01..02-11 and are accounted for in REQUIREMENTS.md (each carries [x] and a Phase 2 source annotation). No orphaned requirement IDs found for Phase 2.

---

## 5. Required Artifacts (Plan 02-11 additions)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-gto/src/basis/nwchem.rs` | General-contraction: N coefficient columns → N contractions; ragged rejection | VERIFIED (substantive) | 319 LoC. Lines 183-204 implement the multi-column accumulator. Line 189 rejects ragged blocks. The SharedSP path (lines 207-226) is untouched. Commit e9fa626 changed 46 lines. No stub patterns. |
| `crates/pyscf-gto/src/projection.rs` | Row-major coefficient flatten for cintx Shell (`coefficients[prim*nctr+ctr]`) | VERIFIED (substantive) | Line 290-294: `coeffs_flat[prim * nctr + ctr] = c` (explicitly comments the kernel's row-major requirement). Commit e9fa626 changed 21 lines in projection.rs. |
| `crates/pyscf-gto/tests/general_contraction.rs` | ANO general-contraction + segmented regression + cintx evaluation correctness | VERIFIED (substantive, wired) | 292 LoC. 9 `#[test]` functions covering: (a) ANO O nctr=8, ANO H contractions; (b) sto-3g/6-31g/6-31g*/cc-pvdz regression pins, nao_nr pins; (c) cc-pVDZ unit-diagonal overlap (exact) + ANO unit-diagonal/l≤2 symmetry. All assertions are numeric, none are stubs. |
| `crates/pyscf-scf/tests/init_guess_minao.rs` | H2O minao test renamed + retightened; H2 byte-match retained | VERIFIED (substantive, wired) | 167 LoC. 3 tests: `minao_h2_byte_matches_upstream_docstring` (1e-6 tolerance), `minao_dm_symmetric_and_traces_to_nelec` (H2 sanity), `minao_h2o_heavy_atom_normalizes_after_general_contraction_fix` (H2O: `tr > 9.5` + `(tr - 9.86).abs() < 0.05` + RHF convergence). Commit b6a9898. |
| `.planning/REQUIREMENTS.md` | GTO-02/GTO-03 general-contraction note; SCF-05 heavy-atom caveat RESOLVED | VERIFIED | GTO-02 line: "GENERAL-CONTRACTION correctness fixed in 02-11: …". GTO-03 line: "02-11 also fixed the cintx Shell coefficient layout…". SCF-05: "HEAVY-ATOM CAVEAT RESOLVED via 02-11". |
| `.planning/phases/02-gto/deferred-items.md` | DI-02-11-CINTX-NCTR RESOLVED; DI-02-11-CINTX-NCTR-HIGHL RESOLVED; DI-02-11-ECP-NCTR RESOLVED; DI-02-11-CINTX-NUC-HIGHL TRACKED | VERIFIED | All four items present with correct disposition headers. The three RESOLVED items cite the cintx commits. DI-02-11-CINTX-NUC-HIGHL is correctly marked as a cross-repo pre-existing tracked gap (not a v1 Phase 2 blocker). |

---

## 6. Key Link Verification (Plan 02-11 additions)

| From | To | Via | Status |
|------|----|----|--------|
| `nwchem.rs CurrentShell::Single` | `ShellSpec.coeffs` (N contraction vectors) | `coeffs.resize(n_ctr, Vec::new())` + per-column push | WIRED — nwchem.rs:184-204 |
| `projection.rs build_atoms_and_shells_with_base` | `CintxShell.coefficients` row-major layout | `coeffs_flat[prim * nctr + ctr] = c` | WIRED — projection.rs:290-294 |
| `general_contraction.rs` tests | `pyscf_gto::basis::load_basis` → `ParsedBasis.shells[i].coeffs.len()` | direct call to production loader | WIRED — uses the same `load_basis` path as production code |
| `general_contraction.rs` overlap tests | `pyscf_gto::intor` → `cintx` `int1e_ovlp_sph` | `intor(&mol, "int1e_ovlp_sph")` | WIRED — exercises full path: parser → projection → cintx kernel |
| `init_guess_minao.rs` H2O test | `pyscf_scf::init_guess::default_get_init_guess` → `InitGuessMode::Minao` | direct call | WIRED — calls through the full minao stack including intor_cross + NRSRHF_CONFIGURATION |
| All pre-existing wiring links from 02-10 verification | (Unchanged) | Spot-checked regression | WIRED — no pre-existing wiring was modified by 02-11 |

---

## 7. Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `general_contraction.rs::ccpvdz_general_contraction_overlap_unit_diagonal` | `s` (nao×nao overlap) | `intor(&mol, "int1e_ovlp_sph")` → cintx `one_electron.rs` kernel with row-major nctr=2 coefficients | YES — asserts exact unit diagonal `(d - 1.0).abs() < 1e-9` for all 14 AOs | FLOWING |
| `general_contraction.rs::ano_general_contraction_overlap_finite_unit_diagonal` | `s` (nao×nao overlap) | `intor(&mol, "int1e_ovlp_sph")` → cintx with ANO O nctr=8 S-block | YES — asserts unit diagonal `(d - 1.0).abs() < 1e-6` for all AOs + l≤2 symmetry | FLOWING |
| `init_guess_minao.rs::minao_h2o_heavy_atom_normalizes_after_general_contraction_fix` | `dm` (nao×nao density) then `tr` (trace) | `default_get_init_guess` → intor_cross with correct ANO O contractions | YES — `tr ≈ 9.86`, tight bound `(tr - 9.86).abs() < 0.05`; also runs RHF kernel to convergence | FLOWING |
| All pre-existing data flows from 02-10 | (Unchanged) | (Unchanged) | YES | FLOWING |

---

## 8. Behavioral Spot-Checks

Tests were confirmed passing by the orchestrator's gate run (read-only verification per CRITICAL_CONSTRAINTS). The SUMMARY Addendum records the gate result as authoritative.

| Behavior | Evidence | Status |
|----------|----------|--------|
| general_contraction.rs 9/9 (ANO O S nctr=8; ANO H; segmented regression; cc-pVDZ unit-diagonal; ANO unit-diagonal/l≤2 symmetry) | 02-11 SUMMARY Addendum: "cintx 173 lib + pyscf-rs gto+scf+df+mp2 = 280 tests (0 failures)" | PASS |
| init_guess_minao.rs 3/3 (H2 byte-match + H2O Tr≈9.86 + RHF convergence) | 02-11 SUMMARY Addendum: 280 tests, 0 failures | PASS |
| Full pyscf-gto suite (regression — all pre-02-11 tests unchanged) | 02-11 SUMMARY Addendum: 280 tests, 0 failures | PASS |
| pyscf-dft 47 lib tests | 02-11 SUMMARY Addendum (with one pre-existing RSH stale test noted) | PASS |
| clippy -D warnings | 02-11 SUMMARY Addendum: PASS | PASS |
| cargo fmt | 02-11 SUMMARY Addendum: PASS | PASS |
| check-no-fma | 02-11 SUMMARY Addendum: PASS | PASS |
| check-dependency-wall | 02-11 SUMMARY Addendum: PASS | PASS |
| 0 libxc in scope | 02-11 SUMMARY Addendum: PASS | PASS |

**Pre-existing unrelated failure (out-of-scope):** pyscf-dft `cam_b3lyp_h2o_rsh::rsh_get_veff_dispatches_into_range_coulomb_branch` — Phase 04 RSH stale test. Its `two_electron.rs` was untouched by 02-11. Not attributed to Phase 2.

---

## 9. Anti-Pattern Scan (Plan 02-11 modified files)

No TBD, FIXME, or XXX markers found in any of the five files modified by plan 02-11 (`nwchem.rs`, `projection.rs`, `general_contraction.rs`, `init_guess_minao.rs`, `REQUIREMENTS.md`).

The `NotYetImplemented { phase, what }` returns in other pyscf-gto files (cp2k.rs, eval_gto.rs, format_atom.rs) are intentional per-phase deferrals referencing explicit phase numbers — not untracked debt. Unchanged from the 02-10 verification.

Advisories carried forward from 02-10 (WR-01, WR-02, WR-03) are unchanged:

- **WR-01** (WARNING): `int1e_ecp_ipnuc`/`int1e_ecp_iprinv` silently route to the scalar operator — Phase 7 GRAD-07 scope, not Phase 2.
- **WR-02** (WARNING): `unwrap_or(0)` in `ecp_engine_cintx.rs` on shell offset/count — impossible None path, maintainability concern.
- **WR-03** (WARNING): Stale comment in `intor_smoke.rs` — documentation inconsistency.

None is a blocker for Phase 2's goal.

---

## 10. Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| GTO-01 | 02-02, 02-09 | 5 atom-input forms | SATISFIED (4 shipped; 5th NotYetImplemented{phase:3}) | mole_construction.rs |
| GTO-02 | 02-03, 02-09, 02-11 | 11 basis-input forms; general-contraction correctness | SATISFIED | basis_input_forms.rs; general_contraction.rs |
| GTO-03 | 02-03, 02-09, 02-11 | All built-in basis files; gto.parse(); coefficient layout correct | SATISFIED (representative subset; Phase 8 full sweep) | builtin_basis_sweep.rs; general_contraction.rs |
| GTO-04 | 02-04, 02-09 | `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-identical | SATISFIED (automated) + HUMAN (oracle venv) | test_byte_identity.py |
| GTO-05 | 02-07, 02-10 | ECP loading + `int1e_ecp` evaluation; ECP nctr>1 correct | SATISFIED (in-tree gate) + HUMAN (oracle venv) | ecp_int1e_oracle.rs 2/2; DI-02-11-ECP-NCTR RESOLVED |
| GTO-06 | 02-05, 02-09 | mol.intor() cintx dispatcher (arity-2) | SATISFIED (arity-2) + HUMAN (oracle venv) | test_intor_oracle.py |
| GTO-07 | 02-06, 02-09 | eval_gto 6 variants | SATISFIED (l=0) + DEFERRED (l>=1 Phase 4) | eval_gto_smoke.rs |
| GTO-08 | 02-02, 02-09 | >=30 attribute floor | SATISFIED | attribute_floor.rs |
| GTO-09 | 02-08, 02-09 | dumps/loads round-trip | SATISFIED | dumps_loads.rs |
| GTO-10 | 02-08, 02-09 | copy/set_geom_ | SATISFIED | mole_copy.rs, set_geom.rs |
| GTO-11 | 02-04, 02-08 | Zero-copy BasisSet re-export | SATISFIED | cintx_zerocopy.rs |

All 11 requirement IDs from the plan frontmatter (`requirements: ["GTO-02", "GTO-03"]` in 02-11; all 11 across plans 02-01..02-11) are accounted for. No orphaned Phase 2 requirement IDs in REQUIREMENTS.md.

---

## 11. Human Verification Required

### 1. ECP byte-identity vs upstream PySCF (GTO-05)

**Test:** Install `tests/oracle/requirements.txt` (numpy + pyscf), then run `pytest tests/oracle/test_ecp_int1e.py::test_cu_lanl2dz_int1e_ecp_byte_equal -v`
**Expected:** Cu/LANL2DZ `int1e_ecp` matrix matches upstream `mol.intor('int1e_ecp')` to atol=1e-10. Note: 02-11 fixed the LANL2DZ basis loading (Cu S-block nctr=2 was being truncated); the oracle test now runs against the correct basis.
**Why human:** Oracle venv (numpy + upstream pyscf) unavailable in default sandbox. cintx pins atol=1e-12 vs vendored PySCF nr_ecp in cintx-oracle/tests/safe_api_ecp_parity.rs (indirect assurance).

### 2. Internal array byte-identity vs upstream PySCF (GTO-04)

**Test:** With oracle venv, run `pytest tests/oracle/test_byte_identity.py -v`
**Expected:** 15 byte-equal assertions (3 fixtures x 5 arrays: H2O/cc-pVDZ, benzene/6-31G*, water-trimer/STO-3G) all pass. Note: cc-pVDZ now loads the correct O S-block (nctr=2); the byte-identity test exercises the corrected basis.
**Why human:** Requires upstream pyscf venv.

### 3. mol.intor() arity-2 parity + F-order layout check (GTO-06)

**Test:** With oracle venv, run `pytest tests/oracle/test_intor_oracle.py -v`
**Expected:** 7 arity-2 names green at atol=1e-10; test_int1e_ipovlp_sph_layout confirms (3, nao, nao) shape
**Why human:** Requires upstream pyscf venv.

---

## 12. Gaps Summary

No actionable gaps. All 11 GTO-* requirements are VERIFIED (automated checks) or have explicit roadmap-documented deferrals. Plan 02-11 closed the last open correctness gap (general-contraction parsing + cintx coefficient layout). The three WR-* warnings from 02-REVIEW.md are carried forward unchanged for Phase 7 handoff (WR-01) and general maintainability (WR-02, WR-03).

The `Tr(dm·S) == nelec` plan-must-have deviation is correctly classified as a plan-premise correction: the implementation is right; the plan's stated target was wrong. The H2 byte-match + H2O trace recovery are the correct anchors.

The three human_verification items are oracle pytest files requiring the upstream pyscf venv — unchanged from the 02-10 re-verification. They do not represent new gaps.

---

_Verified: 2026-05-24_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — after plan 02-11 gap-closure (general-contraction parser + cintx coefficient layout + minao heavy-atom caveat)_
