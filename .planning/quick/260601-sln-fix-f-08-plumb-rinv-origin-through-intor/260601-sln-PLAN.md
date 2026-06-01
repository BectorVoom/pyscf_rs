---
phase: quick-260601-sln
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-gto/src/intor.rs
  - crates/pyscf-gto/src/lib.rs
  - crates/pyscf-gto/tests/grad_intor_smoke.rs
  - crates/pyscf-grad/src/rhf.rs
autonomous: true
requirements: [F-08]
quick_id: 260601-sln

must_haves:
  truths:
    - "A public pyscf-gto entry point evaluates `int1e_iprinv` with a caller-supplied rinv origin (Bohr), threading it into ExecutionOptions instead of the validator-failing default None."
    - "A `with_rinv_at_nucleus`-style convenience resolves the origin from `mol.atom_coord(atm_id)` and evaluates `int1e_iprinv` there."
    - "`int1e_iprinv` now EVALUATES (finite component-leading [3,nao,nao]) when an origin is supplied — no longer routes to a cintx-availability / validator error."
    - "pyscf-grad `hcore_deriv` calls the per-atom origin entry point with `mol.atom_coord(atm_id)`, replacing the origin-less `intor(mol, \"int1e_iprinv\")` call."
    - "An in-tree physics oracle proves the per-atom iprinv is correct via the translational-invariance identity int1e_ipnuc == Σ_atoms (-Z_atom)·int1e_iprinv|rinv@atom, with NO live PySCF / libxc."
  artifacts:
    - path: "crates/pyscf-gto/src/intor.rs"
      provides: "intor_with_rinv_origin + intor_with_rinv_at_nucleus public entry points"
      contains: "rinv_orig: Some"
    - path: "crates/pyscf-gto/src/lib.rs"
      provides: "re-export of the new entry points"
      contains: "intor_with_rinv"
    - path: "crates/pyscf-grad/src/rhf.rs"
      provides: "hcore_deriv wired to per-atom rinv origin"
      contains: "intor_with_rinv_at_nucleus"
    - path: "crates/pyscf-gto/tests/grad_intor_smoke.rs"
      provides: "flipped non-ECP int1e_iprinv test (evaluates) + translational-invariance oracle"
      contains: "int1e_ipnuc"
  key_links:
    - from: "crates/pyscf-grad/src/rhf.rs::hcore_deriv"
      to: "crates/pyscf-gto/src/intor.rs::intor_with_rinv_at_nucleus"
      via: "per-atom origin call mol.atom_coord(atm_id)"
      pattern: "intor_with_rinv_at_nucleus"
    - from: "crates/pyscf-gto/src/intor.rs"
      to: "cintx_runtime::ExecutionOptions.rinv_orig"
      via: "Some(origin) instead of Default None"
      pattern: "rinv_orig: Some"
---

<objective>
Plumb a caller-supplied `rinv_origin` through pyscf-gto's `intor` dispatcher for the
`int1e_iprinv` family, and wire pyscf-grad's `hcore_deriv` to call it with each
nucleus's coordinate. This is the SINGLE remaining actionable audit-fix piece of
F-08 (the audit names it verbatim: "the remaining integral task: plumb
`rinv_origin` through `intor` for the `hcore` `iprinv` arm").

Today plain `intor(mol, "int1e_iprinv")` passes `ExecutionOptions::default()`
(`rinv_orig: None`). cintx's validator REQUIRES `rinv_orig: Some(..)` for any
operator whose name contains `"iprinv"` and returns `InvalidEnvParam{PTR_RINV_ORIG}`
otherwise — so the origin-less call fails. This plan closes exactly that gap by
mirroring upstream `pyscf/gto/mole.py::with_rinv_origin` / `with_rinv_at_nucleus`
and the in-tree F-05 `ecp_int1e_iprinv` precedent (which already sets
`ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() }`).

OUT OF SCOPE (explicitly deferred to waves 07-03..07-08, do NOT touch): the F-08
hook *bodies* — `get_veff` 2e-response over `int2e_ip1`, `hcore_generator`
de-assembly into nuclear forces, `get_ovlp` energy-weighted overlap-derivative
assembly. The `rhf_verify_fd_numeric` ignored test STAYS ignored (it needs the
full multi-wave FD assembly).

Purpose: close the last integral-availability gap blocking the F-08 grad waves;
make the nuclear-attraction Hellmann-Feynman shift physically correct per-atom.
Output: two new public entry points + a wired `hcore_deriv` + an in-tree physics
oracle proving correctness without live PySCF.
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

# The audit this fix closes — read the F-08 rows + "F-08 update (follow-up pass 2)"
# note + the F-05 resolution (pass 7) which is the direct precedent pattern.
@.planning/AUDIT-FIX-2026-06-01.md

# The dispatcher to extend. Every evaluate_arity* currently passes
# ExecutionOptions::default(). The new entry point is a focused variant of
# evaluate_arity2 that sets rinv_orig: Some(origin).
@crates/pyscf-gto/src/intor.rs

# The F-05 precedent (lines ~455-632): CintxEcpEngine::ecp_int1e_iprinv sets
# `ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() }`,
# resolves the op via `Resolver::descriptor_by_symbol(symbol).id` (no OperatorId
# const exists for iprinv), and stitches the component-leading [3,nao,nao] buffer.
# Mirror its options construction (incl. the #[rustfmt::skip] one-line constructor
# so a grep gate sees the LIVE `rinv_orig: Some` token, not a comment).
@crates/pyscf-gto/src/ecp_engine_cintx.rs

# The consumer to wire (hcore_deriv, lines ~259-323): currently calls
# `pyscf_gto::intor(mol, "int1e_iprinv")` WITHOUT an origin. Replace that one call.
@crates/pyscf-grad/src/rhf.rs

<interfaces>
<!-- Contracts the executor needs. Extracted from the codebase — do NOT re-explore. -->

cintx_runtime::ExecutionOptions has field `rinv_orig: Option<[f64;3]>`
(units: Bohr). `..Default::default()` gives rinv_orig: None.

cintx validator (cintx-runtime/src/validator.rs::validate_rinv_orig_env_params):
  - name contains "iprinv" + rinv_orig == None  → Err InvalidEnvParam{PTR_RINV_ORIG}
  - name contains "iprinv" + rinv_orig == Some(..) → Ok (test validate_rinv_orig_accepts_some)

Resolver (cintx_ops::resolver::Resolver):
  descriptor_by_symbol("int1e_iprinv_sph") -> Ok(OperatorDescriptor) ; .id is the
  OperatorId, .entry.arity == 2. (NO OperatorId const for iprinv — resolve by symbol,
  exactly like F-05 and the existing dispatcher do.) `int1e_iprinv` is a REAL
  AllCint1e operator family in cintx — distinct from the cintx-MISSING
  `int1e_ecp_iprinv`. It evaluates today once an origin is supplied.

pyscf_core::Mole::atom_coord(i: usize) -> [f64; 3]   // Bohr; the rinv origin source

pyscf_gto::intor(mol, name) -> Result<IntorOutput, PyscfRsError>
  IntorOutput { values: Vec<f64>, shape: Vec<usize>, layout: IntorLayout }
  For int1e_iprinv / int1e_ipnuc: layout == ComponentLeadingFOrder{components:3},
  shape == [3, nao, nao], element (comp,i,j) at comp*nao*nao + i + j*nao (F-order inner).

layout_table: "int1e_iprinv_sph" AND "int1e_ipnuc_sph" both registered,
ComponentLeadingFOrder{components:3}. So int1e_ipnuc evaluates today (the oracle anchor).

evaluate_arity2(descriptor, operator, representation, basis, nbas, nao, layout,
  intor_name) is the existing per-pair stitch loop. It constructs the request with
  `ExecutionOptions::default()` at intor.rs:285. The new entry point reuses the same
  stitch machinery but passes `ExecutionOptions { rinv_orig: Some(origin), .. }`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: pyscf-gto entry points — intor_with_rinv_origin + intor_with_rinv_at_nucleus, with the translational-invariance oracle</name>
  <files>crates/pyscf-gto/src/intor.rs, crates/pyscf-gto/src/lib.rs, crates/pyscf-gto/tests/grad_intor_smoke.rs</files>
  <behavior>
    - int1e_iprinv WITH an origin evaluates: intor_with_rinv_origin(mol, "int1e_iprinv", origin)
      returns Ok with layout ComponentLeadingFOrder{components:3}, shape [3,nao,nao],
      all-finite values. (Replaces/supersedes any stale assertion that int1e_iprinv
      → cintx-availability / validator error.)
    - intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id) == intor_with_rinv_origin(
      mol, "int1e_iprinv", mol.atom_coord(atm_id)) elementwise.
    - PHYSICS ORACLE (translational-invariance identity, the rigorous in-tree anchor):
      for a real molecule (use the existing he_sto3g or a 2-atom helper; pick one with
      ≥1 nucleus and nontrivial nao), Σ_atoms (-Z_atom) · int1e_iprinv|rinv@atom_coord(atom)
      equals int1e_ipnuc (already available via plain intor) elementwise to a tight atol
      (≤ 1e-10). This is exactly the relation hcore_deriv exploits and is checkable
      against the already-available int1e_ipnuc — NO live PySCF, NO libxc.
    - A non-iprinv name passed to intor_with_rinv_origin (e.g. "int1e_ovlp") errors
      cleanly (the entry point is scoped to the iprinv family — the origin is only
      meaningful/validated for iprinv).
  </behavior>
  <action>
    Add `pub fn intor_with_rinv_origin(mol: &Mole, name: &str, rinv_origin: [f64; 3]) -> Result<IntorOutput, PyscfRsError>` to intor.rs (F-08, quick 260601-sln). Mirror upstream `pyscf/gto/mole.py::with_rinv_origin`. Implementation: built-check (same as `intor`); `add_suffix(name, mol.cart)`; reject any name whose suffix-stripped core is NOT in the iprinv family (the core must equal "int1e_iprinv" — the validator only accepts/requires the origin for iprinv-named ops; route a non-iprinv name to a clean InvalidMolecule error naming the function). Look up the layout via `layout_table::lookup` (must be ComponentLeadingFOrder{components:3}). Resolve representation from the suffix (spinor → NotYetImplemented{phase:3}, mirroring `intor`). Resolve the OperatorId via `Resolver::descriptor_by_symbol(&full_name).id` (NO OperatorId const exists). Get the typed basis via `mol.cintx_basis()`. Then run the SAME per-shell-pair stitch loop as `evaluate_arity2`, but construct the request with the origin-bearing options. Build the options ONCE before the loop, on a single line guarded by `#[rustfmt::skip]` so a grep gate sees the LIVE token (copy the F-05 idiom verbatim): `let opts = ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() };`, then pass `opts.clone()` into each `SessionRequest::new(...)`. To avoid duplicating the ~90-line stitch body, REFACTOR `evaluate_arity2` to take an `ExecutionOptions` parameter (the existing `intor` arity-2 call site passes `ExecutionOptions::default()`; the new entry point passes the origin-bearing opts) — this keeps stitch logic single-source. Do NOT change the arity-4 / arity-3 paths.

    Add `pub fn intor_with_rinv_at_nucleus(mol: &Mole, name: &str, atm_id: usize) -> Result<IntorOutput, PyscfRsError>` mirroring `pyscf/grad/rhf.py:121-143` `with_rinv_at_nucleus(atm_id)`: bounds-check atm_id < mol.natm (clean InvalidMolecule error otherwise), then delegate to `intor_with_rinv_origin(mol, name, mol.atom_coord(atm_id))`.

    Re-export both from lib.rs: extend the existing `pub use intor::{...}` line to include `intor_with_rinv_origin, intor_with_rinv_at_nucleus`.

    In grad_intor_smoke.rs: (a) ADD a test `int1e_iprinv_with_origin_evaluates_component_leading` asserting the entry point returns a finite [3,nao,nao] buffer on a real mol. (b) ADD the translational-invariance oracle test `int1e_iprinv_sum_over_nuclei_equals_ipnuc` per the behavior block. (c) Update the stale module-level doc comment (intor.rs / grad_intor_smoke.rs header lines ~10-11) that lists `int1e_ip{...,rinv}` among the "MISSING from every cintx branch" families — `int1e_iprinv` now evaluates WITH an origin; correct the prose like F-05 corrected its 3 stale comments. If grad_intor_smoke.rs (or intor_smoke.rs) carries an active assertion that `int1e_iprinv` errors, FLIP it to the evaluates-with-origin assertion (do NOT leave a stale negative assertion that now contradicts behavior). Do NOT touch the `int1e_ecp_iprinv` scalar-path WR-01 rejection in ecp_engine_stub.rs (that correctly rejects iprinv as a gradient-name on the scalar path — unrelated).

    No fenced code in this action — follow the F-05 idioms named above. Use `oracle_sum` / the project's pairwise-sum helper for the Σ-over-atoms accumulation in the oracle test ONLY if it is already importable in the test crate; otherwise a plain f64 fold at atol 1e-10 is acceptable for the test (the production hcore_deriv already uses oracle_sum).
  </action>
  <verify>
    <automated>cargo +nightly test -p pyscf-gto --locked --test grad_intor_smoke 2>&1 | tee log/260601-sln-task1-gto-test.log | tail -30</automated>
  </verify>
  <done>
    intor_with_rinv_origin + intor_with_rinv_at_nucleus exist and are re-exported from
    pyscf-gto; `int1e_iprinv` evaluates with an origin (finite [3,nao,nao]); the
    translational-invariance oracle (Σ_atoms (-Z)·iprinv|@atom == int1e_ipnuc) passes
    at atol ≤ 1e-10; non-iprinv names error cleanly; grad_intor_smoke green; any stale
    "iprinv missing/errors" assertion flipped. Full cargo output saved under log/.
  </done>
</task>

<task type="auto">
  <name>Task 2: wire pyscf-grad hcore_deriv to the per-atom rinv origin (with_rinv_at_nucleus)</name>
  <files>crates/pyscf-grad/src/rhf.rs</files>
  <action>
    In `hcore_deriv` (rhf.rs ~259-323; F-08, quick 260601-sln), replace the origin-less
    `let iprinv = pyscf_gto::intor(mol, "int1e_iprinv")?;` (line ~287) with
    `let iprinv = pyscf_gto::intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id)?;`
    — implementing `with_rinv_at_nucleus(atm_id)` from `pyscf/grad/rhf.py:121-143`.
    The `assert_component_leading(&iprinv, nao, "int1e_iprinv")?` check and all
    downstream math (vrinv = -Z·iprinv, the `h1` block add, the symmetrisation) are
    UNCHANGED — only the integral source gains the per-atom origin.

    Update the now-stale doc comment on `hcore_deriv` (lines ~259-267) that says
    `int1e_iprinv + with_rinv_at_nucleus are MISSING from cintx → clean availability
    error` — they are no longer missing; the function now evaluates the real per-atom
    Hellmann-Feynman shift. Correct the prose to match the wired behavior.

    Do NOT touch `get_veff` (int2e_ip1 2e-response), `get_ovlp`, `hcore_generator`,
    or `grad_elec` assembly — those stay deferred to waves 07-03..07-08 (F-08 bodies).
    The `rhf_verify_fd_numeric` ignored test in rhf_verify_fd.rs STAYS ignored (it
    needs the full multi-wave FD assembly, not just the iprinv integral) — leave its
    `#[ignore]` intact; do NOT flip it.
  </action>
  <verify>
    <automated>cargo +nightly test -p pyscf-gto -p pyscf-grad --locked 2>&1 | tee log/260601-sln-task2-grad-test.log | tail -40</automated>
  </verify>
  <done>
    hcore_deriv calls intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id); the
    origin-less call is gone; doc comment corrected; `cargo +nightly test -p pyscf-gto
    -p pyscf-grad --locked` passes with 0 failures (rhf_verify_fd_numeric remains
    correctly ignored). Full cargo output saved under log/.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| pyscf-grad → pyscf-gto intor | caller-supplied rinv origin (atm_id → coord) crosses into the cintx validator |
| pyscf-gto → cintx ExecutionOptions | rinv_orig: Some(..) must satisfy the iprinv validator (None → InvalidEnvParam) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-sln-01 | Tampering | intor_with_rinv_origin accepting a non-iprinv name | mitigate | reject any name whose core ≠ "int1e_iprinv" with a clean InvalidMolecule error (origin only valid for iprinv) |
| T-sln-02 | Denial of Service | atm_id out of range in intor_with_rinv_at_nucleus | mitigate | bounds-check atm_id < mol.natm before atom_coord; clean error, never panic/OOB |
| T-sln-03 | Information Disclosure | silent wrong physics (origin not actually threaded) | mitigate | translational-invariance oracle (Σ_atoms (-Z)·iprinv|@atom == int1e_ipnuc) at atol ≤1e-10 proves the origin is live and correct |
| T-sln-SC | Tampering | npm/pip/cargo installs | accept | NO new dependencies added; pure in-tree wiring over existing cintx_runtime::ExecutionOptions — no install task, no package-legitimacy gate needed |
</threat_model>

<verification>
Gating verification (in-tree cargo, NO live PySCF, NO libxc):
- `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` → 0 failures. Both crates
  EXCLUDE libxc; confirm once with `cargo tree -p pyscf-grad | grep -c libxc` (expect 0)
  and `cargo tree -p pyscf-gto | grep -c libxc` (expect 0) before relying on the gate.
  Per CLAUDE.md, NEVER run a cargo command that pulls libxc_rs into the dep graph.
- The translational-invariance oracle (Task 1) is the physics anchor: int1e_ipnuc ==
  Σ_atoms (-Z_atom)·int1e_iprinv|rinv@atom, atol ≤ 1e-10, against the already-available
  int1e_ipnuc — no external oracle.
- REAL CI fmt gate (avoid the edition-mismatch false positive F-05's verifier hit):
  `git ls-files '*.rs' | xargs rustfmt --edition 2024 --check` over the touched files.
- clippy on touched crates: `cargo +nightly clippy -p pyscf-gto -p pyscf-grad --locked`.
- ALWAYS save full cargo output under log/ BEFORE investigating any failure (CLAUDE.md).

OPTIONAL / nice-to-have (NOT gating): a live-PySCF byte-identity check via the
.upstream-venv (pyscf 2.12.1 + maturin) two-venv .npy cross-compare. A maturin rebuild
is heavy — treat as optional; the gating verification is the in-tree cargo test above.
</verification>

<success_criteria>
- intor_with_rinv_origin + intor_with_rinv_at_nucleus public + re-exported from pyscf-gto.
- `int1e_iprinv` evaluates to a finite component-leading [3,nao,nao] buffer when an
  origin is supplied (was: validator/availability error with default None).
- Translational-invariance oracle passes (Σ_atoms (-Z)·iprinv|@atom == int1e_ipnuc, ≤1e-10).
- hcore_deriv wired to intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id); the
  origin-less intor call removed.
- `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` → 0 failures;
  rhf_verify_fd_numeric STAYS ignored (full FD assembly is F-08 / 07-03..07-08).
- Stale "iprinv missing/errors" comments + assertions corrected/flipped; the
  ecp_engine_stub.rs WR-01 scalar-path rejection RETAINED (unrelated).
- fmt (--edition 2024) + clippy clean on touched crates.
- Atomic commits per task, each tagged F-08 + quick id 260601-sln.
- ROADMAP.md NOT touched.
</success_criteria>

<output>
Create `.planning/quick/260601-sln-fix-f-08-plumb-rinv-origin-through-intor/260601-sln-01-SUMMARY.md` when done.
</output>
