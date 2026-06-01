---
phase: quick-260601-rhc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-core/src/traits.rs
  - crates/pyscf-gto/src/ecp_engine_cintx.rs
  - crates/pyscf-grad/src/ecp.rs
  - crates/pyscf-gto/tests/grad_intor_smoke.rs
  - crates/pyscf-gto/tests/ecp_engine_stub.rs
  - crates/pyscf-grad/tests/ecp_verify_fd.rs
autonomous: true
requirements: [F-05, GRAD-07]

must_haves:
  truths:
    - "CintxEcpEngine evaluates int1e_ecp_iprinv / ECPscalar_iprinv for an ECP-bearing molecule, returning a finite per-atom [3, nao, nao] buffer (no longer a hardcoded availability error)."
    - "For the single-ECP-atom Cu/LANL2DZ fixture, iprinv at the Cu nucleus equals ECPscalar_ipnuc — an in-tree SELF-CONSISTENCY check against the cintx ipnuc kernel (single-ECP-atom degeneracy: with exactly one ECP nucleus the per-atom rinv selection coincides with all-slot accumulation). This is a structural smoke, NOT an external oracle; cintx's own ecp_iprinv_parity.rs holds the byte-identity vs upstream nr_ecp_deriv."
    - "An iprinv rinv-origin matching no atom yields an all-zero [3, nao, nao] buffer (never a panic, never a wrong-atom selection)."
    - "pyscf-grad::hcore_deriv_ecp(mol, atm_id) returns the real per-atom buffer for an ECP-bearing atom instead of the hardcoded cintx-availability error."
    - "Spinor ECP iprinv still fails closed (NotYetImplemented / UnsupportedApi); the legacy libcint-FFI resolver path is untouched."
  artifacts:
    - path: "crates/pyscf-core/src/traits.rs"
      provides: "EcpEngine::ecp_int1e_iprinv default trait method (per-atom rinv origin)"
      contains: "fn ecp_int1e_iprinv"
    - path: "crates/pyscf-gto/src/ecp_engine_cintx.rs"
      provides: "CintxEcpEngine::ecp_int1e_iprinv un-gated impl resolving INT1E_ECP_IPRINV_{CART,SPH} via Resolver::descriptor_by_symbol(..).id + ExecutionOptions.rinv_orig"
      contains: "ecp_int1e_iprinv"
    - path: "crates/pyscf-grad/src/ecp.rs"
      provides: "hcore_deriv_ecp wired to ecp_int1e_iprinv with layout normalisation (mirrors get_hcore_ecp)"
      contains: "ecp_int1e_iprinv"
  key_links:
    - from: "crates/pyscf-grad/src/ecp.rs"
      to: "pyscf_gto::CintxEcpEngine::ecp_int1e_iprinv"
      via: "engine.ecp_int1e_iprinv(mol, name, rinv_origin)"
      pattern: "ecp_int1e_iprinv"
    - from: "crates/pyscf-gto/src/ecp_engine_cintx.rs"
      to: "cintx ecp_iprinv kernel"
      via: "ExecutionOptions { rinv_orig: Some(origin), .. } + Resolver::descriptor_by_symbol(\"int1e_ecp_iprinv_sph\").id"
      pattern: "rinv_orig"
---

<objective>
Un-gate the `int1e_ecp_iprinv` / `ECPscalar_iprinv` ECP-gradient integral now that
cintx workstream 21-07 ships native `ecp_iprinv` (commits dc9c0fc, 84a5b77,
30ed06c/5b24142). The audit's claim that iprinv is "MISSING from every cintx branch,
no scheduled workstream" (AUDIT-FIX-2026-06-01 §F-05 follow-up pass 3) is STALE.

This closes F-05: the per-atom `hcore_deriv` ECP-gradient term
(`pyscf/grad/rhf.py:139-140` — `vrinv += mol.intor('ECPscalar_iprinv', comp=3)` under
`with_rinv_at_nucleus(atm_id)`).

Purpose: deliver the per-atom ECP force integral so the downstream analytic
ECP-gradient assembly (F-08, OUT OF SCOPE here) has a real integral to consume,
instead of the hardcoded availability error currently in `hcore_deriv_ecp`.

Output:
- A new `EcpEngine::ecp_int1e_iprinv` trait method (per-atom rinv origin), defaulting
  to `EcpEngineNotAvailable` so non-cintx impls/stubs stay valid.
- `CintxEcpEngine::ecp_int1e_iprinv` un-gated: resolves `INT1E_ECP_IPRINV_{CART,SPH}`,
  sets `ExecutionOptions.rinv_orig` to the target nucleus, stitches a per-atom
  `[3, nao, nao]` buffer.
- `pyscf-grad::hcore_deriv_ecp` wired to the real per-atom buffer (layout-normalised
  to the RHF component-leading F-order, mirroring `get_hcore_ecp`).
- Tests updated: two stale tests that assert the OLD gated behavior (grad_intor_smoke,
  ecp_verify_fd) are FLIPPED to assert the un-gated behavior, anchored on the in-tree
  single-nucleus self-consistency check (`iprinv@Cu == ipnuc` for the single-ECP-atom
  Cu/LANL2DZ fixture). One test (ecp_engine_stub scalar-path) gets a DOC-COMMENT-ONLY
  update — its `InvalidMolecule` assertion stays correct post-F-05 (WR-01 invariant).
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/AUDIT-FIX-2026-06-01.md
@CLAUDE.md

# The two production files this plan edits (ipnuc is the WORKING template for iprinv):
@crates/pyscf-gto/src/ecp_engine_cintx.rs
@crates/pyscf-grad/src/ecp.rs

<interfaces>
<!-- GROUND-TRUTHED. The executor uses these directly — no codebase/cintx exploration. -->

EcpEngine trait (crates/pyscf-core/src/traits.rs:82-99) — current shape:
  pub trait EcpEngine: Send + Sync {
      fn ecp_int1e(&self, mol: &Mole, name: &str) -> Result<Density, PyscfRsError>;
      fn ecp_int1e_ipnuc(&self, mol: &Mole, name: &str) -> Result<Density, PyscfRsError> { ... default: NotYetImplemented{phase:7} }
  }

Density (crates/pyscf-core/src/density.rs:8-25):
  pub struct Density { pub nao: usize, pub data: Vec<f64>, .. }
  pub fn from_flat(nao: usize, data: Vec<f64>) -> Self

Mole accessors (crates/pyscf-core/src/mole.rs):
  pub natm: usize
  pub nao_nr: usize
  pub fn atom_coord(&self, i: usize) -> [f64; 3]   // Bohr (internal storage is ALWAYS Bohr)

PyscfRsError (crates/pyscf-core/src/error.rs):
  ::EcpEngineNotAvailable
  ::Core(CoreError::InvalidMolecule(String))
  ::NotYetImplemented { phase: u8, what: &'static str }

cintx OperatorId resolution for iprinv (CRITICAL — cintx_core::OperatorId has NO
INT1E_ECP_IPRINV const; only ipnuc=28/29 exist as core consts). Resolve the iprinv
OperatorId via the manifest, EXACTLY as grad_intor_smoke.rs:88-90 already does for
int3c2e_ip1_sph:
  use cintx_ops::resolver::Resolver;
  let operator = Resolver::descriptor_by_symbol("int1e_ecp_iprinv_sph")  // or _cart
      .map_err(|e| ... InvalidMolecule(format!("...{e}")))?
      .id;                                    // descriptor.id : cintx_core::OperatorId
This `operator` is then passed to SessionRequest::new unchanged. The manifest
(cintx-ops/src/generated/api_manifest.rs) carries `operator_name: "ecp_iprinv"`,
`symbol_name: "int1e_ecp_iprinv_{cart,sph}"`, arity 2, component_rank "3", canonical_family "ecp".

cintx rinv-origin plumbing (ALL CONFIRMED present on cintx HEAD 55bf984):
  cintx_runtime::ExecutionOptions {
      pub rinv_orig: Option<[f64; 3]>,   // env[4..6]; validated by validate_rinv_orig_env_params
      ..                                  // derives Default
  }
  // The validator REJECTS rinv_orig: None for any "ecp_iprinv" operator (InvalidEnvParam{PTR_RINV_ORIG}).
  // The native cubecl launch_ecp branches operator_name == "ecp_iprinv" → selects ONLY the
  // ECP slot whose coord_bohr matches rinv_orig within IPRINV_ORIGIN_MATCH_TOL = 1e-10.
  // Origin matching NO atom → zero-filled output (cintx test: iprinv_origin_matching_no_atom_selects_nothing).
  // Spinor iprinv fails closed with UnsupportedApi (cintx side).
  // NOTE: the LEGACY libcint-FFI resolver (cintx-ops/src/resolver.rs:328) returns None for
  // int1e_ecp_iprinv — that is a DIFFERENT path (no cint* macro wrapper). The native
  // SessionRequest→query_workspace→evaluate→launch_ecp path used here DOES support it. Do
  // NOT route through the legacy path; do NOT "fix" the legacy None.

WORKING ipnuc template (ecp_engine_cintx.rs:258-423): builds the ECP-augmented BasisSet,
loops shell pairs, SessionRequest::new(operator, representation, basis_ref, shells, opts),
query_workspace().evaluate(), stitches a COMPONENT-LEADING [3,nao,nao] buffer respecting
outcome.tensor.component_axis_leading. The iprinv path is IDENTICAL except:
  (a) operator = the iprinv OperatorId (resolved via Resolver, above), NOT ipnuc,
  (b) ExecutionOptions carries rinv_orig = Some(target_atom_coord_bohr) instead of default(),
  (c) it takes the target atom's coordinate as an extra arg.

cintx-proven IDENTITY anchor (cintx-oracle/tests/ecp_iprinv_parity.rs:21-30, vendor
byte-identity at atol=1e-12): for the single-ECP-atom Cu/LANL2DZ fixture (ECP on Cu,
atom index 0 ONLY) the per-nucleus selection degenerates — `iprinv@Cu == ipnuc` because
there is exactly one ECP nucleus. NOTE ON WHAT THIS PROVES: the IN-TREE comparison
written in Task 3 is a SELF-CONSISTENCY / structural smoke (cintx iprinv kernel vs cintx
ipnuc kernel, same engine, same stitch) — if cintx returned the same wrong value for
both, the in-tree check would still pass. The math (single-ECP-atom degeneracy) is valid,
but the EXTERNAL byte-identity vs upstream PySCF nr_ecp_deriv is owned by cintx's own
ecp_iprinv_parity.rs, NOT by this in-tree test. Use the in-tree compare as a cheap
deterministic structural anchor (no live PySCF run needed); do not advertise it as the
external oracle.

PySCF semantics (pyscf/grad/rhf.py:135-141, CONFIRMED): hcore_deriv(atm_id) does
`with mol.with_rinv_at_nucleus(atm_id): vrinv += mol.intor('ECPscalar_iprinv', comp=3)`
ONLY when `atm_id in ecp_atoms`. So the rinv origin IS the nucleus coordinate of atm_id.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add EcpEngine::ecp_int1e_iprinv trait method + un-gate CintxEcpEngine::ecp_int1e_iprinv</name>
  <files>crates/pyscf-core/src/traits.rs, crates/pyscf-gto/src/ecp_engine_cintx.rs</files>
  <action>
TRAIT (crates/pyscf-core/src/traits.rs, the EcpEngine trait at ~82-99): add a new default
method (minimal-surface — a dedicated method, NOT an overload of ecp_int1e_ipnuc, because
iprinv carries per-atom rinv-origin semantics distinct from ipnuc's all-slot accumulation):

  fn ecp_int1e_iprinv(&self, mol: &Mole, name: &str, rinv_origin: [f64; 3]) -> Result<Density, PyscfRsError>

Default body returns Err(PyscfRsError::EcpEngineNotAvailable) (so the stub impl
EcpEngineNotAvailable and any other impl stay valid without change — verify the stub at
crates/pyscf-gto/src/ecp_engine_stub.rs needs no edit because it inherits the default).
Doc-comment it: per-atom ECP force term ECPscalar_iprinv (pyscf/grad/rhf.py:139-140),
rinv_origin = the nucleus coordinate in Bohr; cintx 21-07 ships it (F-05).

ENGINE (crates/pyscf-gto/src/ecp_engine_cintx.rs): implement ecp_int1e_iprinv on
CintxEcpEngine by COPYING the ecp_int1e_ipnuc body (258-423) and changing exactly three
things, plus removing the stale gate:
  1. Signature gains `rinv_origin: [f64; 3]`.
  2. The suffix-stripped core guard accepts int1e_ecp_iprinv / ECPscalar_iprinv (instead
     of the ipnuc names). Keep the spinor rep arm returning NotYetImplemented{phase:3,
     what:"spinor ECP gradient integrals (out of v1 scope)"} (spinor iprinv fails closed).
  3. Operator resolution: cintx_core::OperatorId has NO iprinv const, so resolve via the
     manifest (the grad_intor_smoke.rs:88-90 pattern):
       use cintx_ops::resolver::Resolver;
       let symbol = match representation { Cart => "int1e_ecp_iprinv_cart", Spheric => "int1e_ecp_iprinv_sph", Spinor => return NotYetImplemented{phase:3,..} };
       let operator = Resolver::descriptor_by_symbol(symbol).map_err(|e| Core(InvalidMolecule(format!("cintx iprinv operator '{symbol}' not in manifest for '{name}': {e}"))))?.id;
  4. ExecutionOptions: build `ExecutionOptions { rinv_orig: Some(rinv_origin), ..ExecutionOptions::default() }`
     and pass it to BOTH the query (it threads through) — replace the two
     `ExecutionOptions::default()` call sites in the shell-pair loop. The cintx validator
     requires Some(origin) for ecp_iprinv; passing None would error InvalidEnvParam.
  5. The stitch is byte-identical to ipnuc (component-leading [3,nao,nao], honoring
     outcome.tensor.component_axis_leading) — DO NOT change it. Reductions: the stitch is a
     copy (out[o] = block[b]), no accumulation, so no oracle_sum needed there (matches ipnuc).
Also UPDATE the now-stale module doc-comment (lines 49-53) and the ecp_int1e_ipnuc
doc-comment D-02 paragraph (254-257) and the inline gate comment (267-282): iprinv is NO
LONGER missing — it routes through the new ecp_int1e_iprinv method (cintx 21-07, F-05). The
ipnuc method's gate (275-282) STAYS (ipnuc must still reject iprinv names — iprinv now has
its own method), but reword its message to point at ecp_int1e_iprinv rather than claiming
iprinv is missing from cintx.
Do NOT place any fenced code in this action — the signature + identifiers above are the contract.
  </action>
  <verify>
    <automated>cargo +nightly build -p pyscf-gto --locked 2>&1 | tee log/rhc-t1-build.log | tail -20; grep -c 'ExecutionOptions { rinv_orig: Some' crates/pyscf-gto/src/ecp_engine_cintx.rs; cargo +nightly test -p pyscf-gto --locked -q ecp_iprinv_evaluates_real_per_atom_buffer 2>&1 | tail -8</automated>
  </verify>
  <done>EcpEngine::ecp_int1e_iprinv exists with a default returning EcpEngineNotAvailable; CintxEcpEngine::ecp_int1e_iprinv compiles, resolves INT1E_ECP_IPRINV_{CART,SPH} via Resolver::descriptor_by_symbol(..).id, sets ExecutionOptions.rinv_orig = Some(rinv_origin), spinor arm still errors, ipnuc method's gate still rejects iprinv names but no longer claims iprinv is missing. The `grep -c 'ExecutionOptions { rinv_orig: Some'` count matches ONLY live constructor code (not comments), and the ecp_iprinv_evaluates_real_per_atom_buffer smoke test (created in Task 3, run here once it exists) exercises the real runtime path. pyscf-gto builds clean.</done>
</task>

<task type="auto">
  <name>Task 2: Wire pyscf-grad::hcore_deriv_ecp to the real per-atom iprinv buffer</name>
  <files>crates/pyscf-grad/src/ecp.rs</files>
  <action>
Rewrite hcore_deriv_ecp (currently 120-143, the hardcoded availability error). Keep the
atm_id >= natm range guard. Then, mirroring get_hcore_ecp (63-105):
  1. Resolve the rinv origin: let rinv_origin = mol.atom_coord(atm_id); (Bohr — Mole storage
     is always Bohr, matching cintx's coordinate-match selector).
  2. Switch the CALL SITE from the OLD `engine.ecp_int1e_ipnuc(mol, "ECPscalar_iprinv")`
     to the NEW `engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", rinv_origin)`. The OLD
     ipnuc call MUST be removed — leaving it would compile but route through the wrong
     (all-slot, no-rinv) kernel. The live call must read `engine.ecp_int1e_iprinv(...)`.
  3. Match the result EXACTLY like get_hcore_ecp:
       Ok(density) => validate density.nao == nao && density.data.len() == NCOMP*nao*nao
         (else InvalidMolecule with the element-count message), then NORMALISE the engine's
         component-INNER layout (data[comp + p*3 + q*3*nao]) to the RHF component-leading
         F-order (out[comp*nao*nao + p + q*nao]) — copy the exact normalisation loop from
         get_hcore_ecp:86-95. Return Ok(out).
       Err(EcpEngineNotAvailable) => for hcore_deriv, a non-ECP atom selected (or a no-ECP
         molecule) means the iprinv term contributes nothing → Ok(vec![0.0; NCOMP*nao*nao]).
         (Matches PySCF: vrinv += ECPscalar_iprinv ONLY when atm_id in ecp_atoms; a non-ECP
         atom is a zero contribution, not an error. Note the cintx kernel ALSO returns a
         zero buffer when the origin matches no ECP atom — so a real non-ECP atm_id on an
         ECP molecule yields all-zeros via the Ok path too; both routes give the same zero.)
       Err(e) => Err(e) (propagate any other error — malformed ECP, workspace failure).
  4. Update the doc-comment (107-119) + the module doc-comment (1-37, the hcore_deriv
     bullet at 17-20) + the layout-normalisation note: iprinv is NOW cintx-READY (F-05 /
     cintx 21-07), no longer "MISSING from cintx". Note that consuming this per-atom buffer
     into total nuclear forces is F-08 (still out of scope — this returns the integral only).
Bit-exact discipline (CLAUDE.md / ecp.rs Pitfall): the normalisation is a pure copy
(out[..] = density.data[..]), no accumulation — so no oracle_sum is required (identical to
get_hcore_ecp). Do NOT introduce a bare += anywhere.
  </action>
  <verify>
    <automated>cargo +nightly build -p pyscf-grad --locked 2>&1 | tee log/rhc-t2-build.log | tail -20; grep -n 'engine.ecp_int1e_iprinv' crates/pyscf-grad/src/ecp.rs; ! grep -n 'engine.ecp_int1e_ipnuc' crates/pyscf-grad/src/ecp.rs</automated>
  </verify>
  <done>hcore_deriv_ecp's LIVE call site is `engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", mol.atom_coord(atm_id))` (the grep -n shows it; the OLD `engine.ecp_int1e_ipnuc` call site is GONE — the negated grep passes). It normalises the buffer to RHF component-leading F-order, returns Ok zero-buffer on EcpEngineNotAvailable, propagates other errors, and no longer contains the hardcoded "MISSING from every cintx branch" error. pyscf-grad builds clean. (Runtime exercise of this method through hcore_deriv_ecp is asserted by the Task 3 Cu nonzero-buffer test.)</done>
</task>

<task type="auto">
  <name>Task 3: Update the three stale tests — flip the two gated-behavior assertions (smoke + FD), doc-only on the scalar-path test</name>
  <files>crates/pyscf-gto/tests/grad_intor_smoke.rs, crates/pyscf-gto/tests/ecp_engine_stub.rs, crates/pyscf-grad/tests/ecp_verify_fd.rs</files>
  <action>
Two of the three tests assert the OLD gated behavior (iprinv errors) — those assertions are
now WRONG and get FLIPPED. The third (scalar-path stub) keeps its assertion and only gets a
doc-comment refresh. Add a clear note in each touched test: "F-05 / cintx 21-07: iprinv
un-gated; prior assertion (iprinv → cintx-availability error) is superseded." (sub-tests A
and C only — NOT B).

(A) grad_intor_smoke.rs:203-225 (ecp_iprinv_is_clean_cintx_availability_error_not_phase_7):
    FLIP. REPLACE with a test that calls CintxEcpEngine::ecp_int1e_iprinv on cu_lanl2dz at
    the Cu nucleus coord and asserts: Ok; density.data.len() == 3*nao*nao; all finite; at
    least one |v| > 1e-18 (real values, not a zero-fill stub). Use mol.atom_coord(0) as the
    origin. Name it EXACTLY ecp_iprinv_evaluates_real_per_atom_buffer (Task 1's verify smoke
    runs this test by name).
    ADD a second test ecp_iprinv_at_cu_equals_ipnuc_single_nucleus: call BOTH
    ecp_int1e_ipnuc(mol, "ECPscalar_ipnuc") and ecp_int1e_iprinv(mol, "ECPscalar_iprinv",
    mol.atom_coord(0)) on cu_lanl2dz, assert the two [3,nao,nao] buffers are equal within
    atol 1e-12. DOC-COMMENT THIS HONESTLY: it is a SELF-CONSISTENCY / structural smoke
    against the cintx ipnuc kernel (cintx iprinv vs cintx ipnuc, same engine + same stitch),
    NOT an external oracle — if cintx returned the same wrong value for both this would still
    pass. The single-ECP-atom degeneracy (exactly one ECP nucleus ⇒ per-atom rinv selection
    coincides with all-slot accumulation) is the math that makes the compare meaningful; the
    EXTERNAL byte-identity vs upstream nr_ecp_deriv lives in cintx's own ecp_iprinv_parity.rs,
    not here. Compare element-by-element.
    ADD a third test ecp_iprinv_origin_matching_no_atom_is_all_zeros: call
    ecp_int1e_iprinv(mol, "ECPscalar_iprinv", [100.0, 100.0, 100.0]) on cu_lanl2dz, assert
    Ok with all elements == 0.0 (the kernel's no-match → zero-fill path).

(B) ecp_engine_stub.rs:52-81 (int1e_ecp_iprinv_via_scalar_intor_is_clean_cintx_availability_error):
    DOC-COMMENT-ONLY. The assertion is UNCHANGED — do NOT remove it. The scalar
    intor()/ecp_int1e path STILL correctly rejects derivative names: `int1e_ecp_iprinv` is a
    GRADIENT name and must never resolve to the SCALAR operator, so the InvalidMolecule
    rejection on the scalar `ecp_int1e` path is CORRECT post-F-05 (the WR-01 invariant). Keep
    both assertions (NOT NotYetImplemented{phase:7}; IS InvalidMolecule) exactly as-is. ONLY
    rewrite the doc-comment: iprinv is no longer "MISSING from every cintx branch" — the
    DEDICATED ecp_int1e_iprinv method (F-05 / cintx 21-07) now serves it; the scalar path
    rejects it because iprinv is a gradient name, NOT because cintx lacks the family. (This
    test is the WR-01 guard — do NOT weaken it.)

(C) ecp_verify_fd.rs:106-122 (ecp_iprinv_per_atom_term_routes_to_the_gated_arm):
    FLIP. REPLACE with ecp_iprinv_per_atom_term_returns_real_buffer: call
    hcore_deriv_ecp(&mol, 0) on cu_lanl2dz, assert Ok; len == 3*nao*nao; all finite; at least
    one nonzero (the Cu atom IS the ECP atom — this is the runtime exercise of Task 2's new
    call site through hcore_deriv_ecp). ADD a check that hcore_deriv_ecp on he_sto3g (no ECP,
    an INDEPENDENT anchor) at atm_id 0 returns Ok all-zeros (non-ECP atom → zero
    contribution). Update the helper is_clean_cintx_availability_error usage as needed (it may
    become unused — if so, remove it AND its #[allow]/import to keep clippy clean, or leave it
    only if still referenced).
    Leave ecp_verify_fd_numeric (the #[ignore]'d end-to-end FD, 132-153) AS-IS: its #[ignore]
    reason still cites the F-08 base assembly (int2e_ip1 + int1e_ip*), which is out of scope —
    but UPDATE the reason string to drop "+ ECPscalar_iprinv" from the missing list (iprinv
    is now landed; only the base grad-intor families remain).
Build/clippy must stay clean (remove any now-dead imports/helpers).
  </action>
  <verify>
    <automated>cargo +nightly test -p pyscf-gto -p pyscf-grad --locked 2>&1 | tee log/rhc-t3-test.log | tail -40; cargo +nightly clippy -p pyscf-gto -p pyscf-grad --locked 2>&1 | tee log/rhc-t3-clippy.log | grep -E 'warning|error' | head; cargo +nightly fmt --check 2>&1 | tail -5</automated>
  </verify>
  <done>grad_intor_smoke has the evaluate/self-consistency-identity/no-match-zero tests (the identity test doc-commented as a self-consistency smoke, NOT an external oracle); ecp_engine_stub's scalar-path test has an updated doc-comment with its InvalidMolecule assertion RETAINED (WR-01); ecp_verify_fd asserts hcore_deriv_ecp returns a real per-atom buffer (nonzero on Cu, all-zero on He as an independent anchor). pyscf-gto + pyscf-grad tests pass, clippy clean, fmt clean. The single-nucleus self-consistency check (iprinv@Cu == ipnuc, atol 1e-12) holds.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| pyscf-grad → pyscf-gto engine | atm_id-derived rinv origin crosses into the cintx ECP launcher |
| pyscf-gto → cintx safe API | OperatorId + ExecutionOptions.rinv_orig cross into kernel launch |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-rhc-01 | Tampering | rinv origin from mol.atom_coord(atm_id) | mitigate | atm_id range-guarded (< natm) in hcore_deriv_ecp BEFORE atom_coord; cintx validator rejects rinv_orig:None for ecp_iprinv (InvalidEnvParam{PTR_RINV_ORIG}) so a missing origin can never read garbage. |
| T-rhc-02 | Information disclosure | wrong-atom ECP slot selection | mitigate | cintx selects ONLY the slot whose coord matches rinv_orig within 1e-10; no-match → zero output (cintx test iprinv_origin_matching_no_atom_selects_nothing). Test C asserts the no-match-zeros path. |
| T-rhc-03 | Spoofing | iprinv mis-resolving to the scalar/ipnuc operator | mitigate | dedicated ecp_int1e_iprinv method with its own suffix-stripped-core guard; scalar ecp_int1e + ipnuc paths retain their derivative-name rejection (WR-01 invariant — the ecp_engine_stub scalar-path assertion is RETAINED, Task 3-B). |
| T-rhc-SC | Tampering | npm/pip/cargo installs | accept | No new dependencies — uses existing cintx-{core,compat,rs,ops,runtime} path deps already in pyscf-gto/Cargo.toml; no install step. |
</threat_model>

<verification>
- `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` passes (NOT a full workspace
  build — these dep graphs exclude libxc per CLAUDE.md / MEMORY; no ~6h compile).
- Single-nucleus self-consistency holds: iprinv@Cu (Cu/LANL2DZ) == ECPscalar_ipnuc within
  atol 1e-12 (in-tree cintx-vs-cintx structural smoke; external byte-identity is cintx's
  ecp_iprinv_parity.rs).
- No-match origin yields all-zeros; non-ECP molecule (He) hcore_deriv_ecp yields all-zeros
  (independent anchor).
- Task 2 call-site check: `grep -n 'engine.ecp_int1e_iprinv'` shows the live call; the old
  `engine.ecp_int1e_ipnuc` call site is gone (negated grep passes).
- `cargo +nightly clippy -p pyscf-gto -p pyscf-grad --locked` is warning-clean on new code.
- `cargo +nightly fmt --check` (edition 2024) clean.
- Full cargo output saved under log/ before any build-issue investigation (CLAUDE.md).
- Grep gate hygiene: Task 1 counts ONLY live constructor code (`grep -c 'ExecutionOptions {
  rinv_orig: Some'`, not a comment-leaking unfiltered token) and additionally runs the named
  smoke test so a count match cannot mask a failed edit.
</verification>

<success_criteria>
- F-05 closed: int1e_ecp_iprinv / ECPscalar_iprinv evaluates through CintxEcpEngine and
  pyscf-grad::hcore_deriv_ecp returns the real per-atom [3, nao, nao] buffer.
- The stale "MISSING from every cintx branch" gate + the two flipped tests are gone; replaced
  by un-gated assertions anchored on the in-tree single-nucleus self-consistency check. The
  scalar-path WR-01 guard (ecp_engine_stub) keeps its rejection assertion (doc-comment only).
- Spinor iprinv still fails closed; the legacy libcint-FFI resolver path is untouched.
- Out of scope (NOT attempted): F-08 end-to-end analytic ECP-gradient FD assembly; spinor
  ECP iprinv numerics.
</success_criteria>

<output>
Create `.planning/quick/260601-rhc-fix-f-05-un-gate-int1e-ecp-iprinv-ecp-gr/260601-rhc-SUMMARY.md` when done.
</output>
