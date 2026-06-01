---
phase: quick-260601-nfb
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-mp2/src/mp2.rs
  - crates/pyscf-mp2/src/lib.rs
  - crates/pyscf-mp2/tests/ump2_cross_spin.rs
  - crates/pyscf-py/src/mp.rs
  - crates/pyscf-py/tests/ump2_open_shell_oracle.py
autonomous: true
requirements: [F-06]
must_haves:
  truths:
    - "A cross-spin (o_α v_α | o_β v_β) ao2mo wrapper exists in pyscf-mp2 and returns the αβ block in the kernel's C-order [nocc_a,nvir_a,nocc_b,nvir_b] layout, NOT the raw F-order ao2mo::general output"
    - "PyUMP2.kernel returns a real opposite-spin αβ energy (no NotYetImplemented{plan:4}) for a UHF reference"
    - "PyMp2Scanner unrestricted path returns a real UMP2 total energy at a new geometry (no NotYetImplemented{plan:4})"
    - "The unrestricted DF MP2 scanner / kernel routes through dfump2_kernel (not dfrmp2_kernel) and returns a real cross-spin DF-UMP2 energy"
    - "Cross-spin layout correctness is PROVEN by open-shell byte-identity vs live PySCF on a real doublet radical (UMP2 + DFUMP2), NOT merely by the closed-shell UMP2==RMP2 invariant"
  artifacts:
    - path: "crates/pyscf-mp2/src/mp2.rs"
      provides: "cross_spin_ao2mo wrapper: builds co_a/cv_a/co_b/cv_b, calls ao2mo::general, transposes F-order -> C-order [nocc_a,nvir_a,nocc_b,nvir_b]"
      contains: "fn cross_spin_ao2mo"
    - path: "crates/pyscf-mp2/tests/ump2_cross_spin.rs"
      provides: "closed-shell UMP2==RMP2 guard (necessary, not sufficient) + F->C layout unit assertion"
      min_lines: 40
    - path: "crates/pyscf-py/tests/ump2_open_shell_oracle.py"
      provides: "two-venv live-PySCF byte-identity check on an open-shell doublet (UMP2 + DFUMP2)"
      min_lines: 40
  key_links:
    - from: "crates/pyscf-py/src/mp.rs"
      to: "pyscf_mp2::cross_spin_ao2mo"
      via: "eris_ab construction in PyUMP2.kernel and PyMp2Scanner Unrestricted arm"
      pattern: "cross_spin_ao2mo"
    - from: "crates/pyscf-py/src/mp.rs"
      to: "pyscf_mp2::dfump2_kernel"
      via: "unrestricted DF scanner/kernel dispatch"
      pattern: "dfump2_kernel"
    - from: "crates/pyscf-mp2/src/mp2.rs::cross_spin_ao2mo"
      to: "pyscf_ao2mo::general"
      via: "[&co_a,&cv_a,&co_b,&cv_b] then explicit F-order->C-order repack"
      pattern: "general\\(.*&co_a"
---

<objective>
Close F-06: deliver the cross-spin `(o_α v_α | o_β v_β)` ao2mo wrapper with a correct
F-order→C-order layout transpose, un-gate the two PyO3 UMP2 αβ sites, wire `dfump2_kernel`
into the unrestricted PyO3 scanner, and PROVE cross-spin correctness via open-shell
byte-identity vs live PySCF (UMP2 + DFUMP2).

Purpose: The opposite-spin (αβ) term is the dominant MP2 correlation contribution. Today both
PyO3 UMP2 sites refuse it with `NotYetImplemented{plan:4}` (safe, but UMP2 is unusable from
Python); the unrestricted DF scanner silently calls `dfrmp2_kernel` instead of the existing
`dfump2_kernel`. The kernel (`opposite_spin_channel`) and the DF cross-spin assembler already
exist and are byte-correct — the only missing piece for conventional UMP2 is a layout-correct
αβ ao2mo input.

Output: `pyscf_mp2::cross_spin_ao2mo`; un-gated `PyUMP2.kernel` + `PyMp2Scanner` (Unrestricted
+ DensityFitted-unrestricted) arms; in-tree layout/closed-shell guard test; live-PySCF
open-shell byte-identity oracle.
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/AUDIT-FIX-2026-06-01.md
@CLAUDE.md
@docs/rust_crate_test_guideline.md

<critical_constraints>
- NEVER trigger a libxc_rs build (~6h, forbidden). VERIFIED SAFE: `cargo tree -p pyscf-py`
  shows ZERO libxc rows, and `cargo tree -p pyscf-mp2` likewise. `maturin develop` on pyscf-py
  does NOT pull libxc. Before any heavy cargo invocation, save full output to `log/`.
- Scope guard: touch ONLY pyscf-mp2 / pyscf-ao2mo / pyscf-py. Do NOT modify the kernel
  (`opposite_spin_channel`), `ao2mo::general`, or `dfump2_kernel` — they are byte-correct;
  this plan only builds correct INPUTS and wires them.
- Read `docs/rust_crate_test_guideline.md` before writing any test.
- If maturin/PyO3 usage details are unclear, fetch current docs via the `ctx7` CLI
  (resolve library then fetch docs) — do NOT rely on memory.
</critical_constraints>

<interfaces>
<!-- Ground-truthed contracts. Use directly — no codebase exploration needed. -->

ao2mo::general (crates/pyscf-ao2mo/src/incore.rs:39) — returns FLAT F-ORDER [n0,n1,n2,n3]:
  general(eri_ao: &[f64], nao: usize, mo_coeffs: [&MOCoefficients; 4]) -> Result<Vec<f64>, Ao2moError>
  element (i,j,k,l) at  i + j*n0 + k*n0*n1 + l*n0*n1*n2
  ni = mo_coeffs[i].nmo ; each block is column-major [nao, nmo].

ump2_kernel αβ input (ChemistsEris.ovov) is read in C-ORDER (crates/pyscf-mp2/src/ump2.rs:274):
  g(i,a,J,B) = ovov[ ((i*nvir_a + a)*nocc_b + J)*nvir_b + B ]
  expected shape [nocc_a, nvir_a, nocc_b, nvir_b].

THE TRANSPOSE (the delicate part): general([&co_a,&cv_a,&co_b,&cv_b]) gives F-order
  [n0=nocc_a, n1=nvir_a, n2=nocc_b, n3=nvir_b]:  F(i,a,J,B) at i + a*nocc_a + J*nocc_a*nvir_a + B*nocc_a*nvir_a*nocc_b.
  Same axis ORDER (i,a,J,B), opposite contiguity. Repack to C-order:
  C(i,a,J,B) at ((i*nvir_a+a)*nocc_b+J)*nvir_b+B.  This is a pure F→C re-stride of a 4-D array.
  ⚠ The RMP2 path (default_ao2mo, mp2.rs:137) skips this repack and still works ONLY because the
  αα palindromic shape (n0=n2, n1=n3) + (ia|jb)=(jb|ia) symmetry absorb F↔C. That symmetry does
  NOT hold cross-spin (nocc_a≠nocc_b, distinct α/β orbitals) — so the repack is MANDATORY here.

default_ao2mo (mp2.rs:137) — proven αα/ββ same-spin builder; mirror its mask/mo_subset idiom:
  let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;
  let co = mo_subset(refr, &mask, true)?;  let cv = mo_subset(refr, &mask, false)?;
  (reference_elements(refr) builds the &elements arg; mp2.rs:60.)

UmpReference (ump2.rs:80): { pub alpha: Mp2Reference, pub beta: Mp2Reference }
ChemistsEris (hooks.rs:23): { ovov: Vec<f64>, nocc: usize, nvir: usize }  (nocc/nvir = α side)

PyO3 gate sites to un-gate (crates/pyscf-py/src/mp.rs):
  :552-553  PyUMP2.kernel        eris_ab = Err(NotYetImplemented{plan:4})
  :815-816  PyMp2Scanner::__call__ Unrestricted arm  ab = Err(NotYetImplemented{plan:4})
  :827-837  PyMp2Scanner::__call__ DensityFitted arm  currently calls dfrmp2_kernel even for UHF
            (NOTE: the Unrestricted DF path must route dfump2_kernel — confirm how DF-unrestricted
             is dispatched; PyDFMP2 vs PyDFUMP2 / the scanner `kind`. dfump2_kernel exists at
             crates/pyscf-mp2/src/dfmp2.rs:386 and already returns the correct C-order αβ block.)

Live oracle: .upstream-venv (pyscf 2.12.1). maturin nightly is pip-installable into that venv.
  VERIFIED: maturin develop on pyscf-py does NOT trigger libxc.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: cross_spin_ao2mo wrapper (F-order -> C-order αβ block) + in-tree layout/closed-shell guard</name>
  <files>crates/pyscf-mp2/src/mp2.rs, crates/pyscf-mp2/src/lib.rs, crates/pyscf-mp2/tests/ump2_cross_spin.rs</files>
  <behavior>
    - Test A (F->C unit): construct a tiny synthetic eri/coeff case with nocc_a≠nocc_b and
      nvir_a≠nvir_b (e.g. 1/2 occ, 2/1 vir) so the αα palindromic symmetry CANNOT mask a
      transpose bug. Assert the returned ovov satisfies, for every (i,a,J,B),
      ovov[((i*nvir_a+a)*nocc_b+J)*nvir_b+B] == general_F_order[i + a*nocc_a + J*nocc_a*nvir_a + B*nocc_a*nvir_a*nocc_b].
      This pins the repack independent of physics.
    - Test B (closed-shell guard, NECESSARY-not-sufficient): feed a closed-shell reference as an
      unrestricted pair (α==β); ump2_kernel using cross_spin_ao2mo for αβ must reproduce
      rmp2_kernel e_corr to <1e-10. Add an in-code comment that this guard does NOT certify the
      cross-spin layout (a symmetric-equivalent transpose bug is invisible when α==β) — Task 3 is
      the real acceptance gate.
  </behavior>
  <action>Add `pub fn cross_spin_ao2mo(alpha: &Mp2Reference, beta: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError>` to crates/pyscf-mp2/src/mp2.rs, exported from lib.rs. Mirror default_ao2mo's mask/mo_subset idiom (reference_elements + frozen_mask + mo_subset) to build co_a/cv_a from `alpha` and co_b/cv_b from `beta`. Compute `eri = pyscf_gto::intor(&alpha.mol, "int2e")?` (single shared molecule; α and β share geometry/basis). Call `pyscf_ao2mo::general(&eri.values, nao, [&co_a, &cv_a, &co_b, &cv_b])` — this yields F-order [nocc_a,nvir_a,nocc_b,nvir_b]. Then EXPLICITLY repack into C-order: allocate `ovov = vec![0.0; nocc_a*nvir_a*nocc_b*nvir_b]` and copy `ovov[((i*nvir_a+a)*nocc_b+J)*nvir_b+B] = f_order[i + a*nocc_a + J*nocc_a*nvir_a + B*nocc_a*nvir_a*nocc_b]` over all (i,a,J,B). Return `ChemistsEris { ovov, nocc: nocc_a, nvir: nvir_a }`. The repack is the F→C transpose called out in the audit doc — do NOT skip it (only the αα palindrome lets default_ao2mo skip it; cross-spin breaks that symmetry). Document this with a doc-comment citing the layout contract. Per F-06: never substitute zeros; `?`-propagate intor/ao2mo errors. Write the test file per docs/rust_crate_test_guideline.md.</action>
  <verify>
    <automated>cargo +nightly test -p pyscf-mp2 --test ump2_cross_spin --locked 2>&1 | tee log/t1-cross-spin-test.log | tail -20</automated>
  </verify>
  <done>cross_spin_ao2mo is exported from pyscf_mp2; Test A (F->C repack) and Test B (closed-shell UMP2==RMP2 <1e-10) both pass; clippy clean on new code; pyscf-mp2 dep graph confirmed libxc-free.</done>
</task>

<task type="auto">
  <name>Task 2: Un-gate the two PyO3 UMP2 αβ sites + wire dfump2_kernel into the unrestricted DF path</name>
  <files>crates/pyscf-py/src/mp.rs</files>
  <action>Replace the `NotYetImplemented{plan:4}` αβ gate at PyUMP2.kernel (mp.rs:552-553) with `let eris_ab = pyscf_mp2::cross_spin_ao2mo(&refr.alpha, &refr.beta, &frozen).map_err(pyscf_to_py)?;` then call the existing ump2_kernel unchanged. Replace the identical gate in the PyMp2Scanner::__call__ Unrestricted arm (mp.rs:815-816) the same way (inside the existing `py.detach` closure, building `ab = cross_spin_ao2mo(&refr.alpha, &refr.beta, &self.frozen)?`). For the DF unrestricted path: trace how an unrestricted DF MP2 reaches the scanner (the DensityFitted arm at mp.rs:827-837 currently always calls dfrmp2_kernel). Route the unrestricted-DF case through `pyscf_mp2::dfump2_kernel(&ump_refr, &self.frozen, &df, false)` — which already returns the correct C-order αβ block via assemble_cross_spin. If DF-unrestricted is dispatched by a separate scanner `kind` or a distinct PyDFUMP2 class, wire dfump2_kernel at that site; if the `kind` enum lacks an unrestricted-DF variant, add it (do NOT misroute UHF-DF through the RHF dfrmp2_kernel). Remove/replace the now-stale gating doc-comments. Do NOT modify ump2_kernel or dfump2_kernel themselves. Save the build log to log/.</action>
  <verify>
    <automated>cargo +nightly build -p pyscf-py --locked 2>&1 | tee log/t2-pyscf-py-build.log | tail -15 && ! grep -rn "NotYetImplemented { plan: 4 }\|NotYetImplemented{plan:4}\|NotYetImplemented { plan : 4 }" crates/pyscf-py/src/mp.rs</automated>
  </verify>
  <done>Both UMP2 αβ gate sites call cross_spin_ao2mo; the unrestricted DF path calls dfump2_kernel (not dfrmp2_kernel); pyscf-py builds clean with no remaining `plan:4` NotYetImplemented in mp.rs; build log saved under log/; libxc not pulled.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Open-shell byte-identity vs live PySCF (UMP2 + DFUMP2) — THE acceptance gate</name>
  <files>crates/pyscf-py/tests/ump2_open_shell_oracle.py</files>
  <what-built>A two-venv `.npy` cross-compare proving the cross-spin αβ layout is numerically correct on a REAL open-shell doublet (not just the α==β closed-shell guard, which is necessary-but-NOT-sufficient — a symmetric-equivalent transpose bug survives it).</what-built>
  <action>Before checkpoint, AUTOMATE everything possible:
  1. Install maturin into .upstream-venv: `.upstream-venv/bin/pip install maturin` (nightly toolchain). If usage is unclear, fetch maturin docs via `ctx7`.
  2. `.upstream-venv/bin/maturin develop -m crates/pyscf-py/Cargo.toml` (or the workspace pyo3 member) — VERIFIED libxc-free, no ~6h build. Save output to log/t3-maturin-develop.log.
  3. Write crates/pyscf-py/tests/ump2_open_shell_oracle.py per docs/rust_crate_test_guideline.md: pick a real doublet radical (e.g. OH radical or CH3, spin=1, a small basis like sto-3g or 6-31g where cintx ordering is known-good). Run UPSTREAM pyscf UHF→UMP2 and UHF→DFUMP2, dump reference e_corr/e_tot to .npy. Run the pyscf-rs bridge (UMP2().kernel and the DF unrestricted path) on the SAME molecule. Assert |e_corr_rs - e_corr_upstream| <= 1e-9 for BOTH conventional UMP2 and DFUMP2.
  4. Run it: `.upstream-venv/bin/python crates/pyscf-py/tests/ump2_open_shell_oracle.py 2>&1 | tee log/t3-oracle.log`.
  Report the measured deltas. The closed-shell guard from Task 1 is NOT a substitute for this.</action>
  <how-to-verify>
    1. Confirm log/t3-maturin-develop.log shows a successful build with NO libxc rows.
    2. Confirm log/t3-oracle.log reports both UMP2 and DFUMP2 deltas <= 1e-9 vs upstream pyscf 2.12.1 on the chosen open-shell doublet.
    3. If either delta exceeds 1e-9 → the F→C transpose (or DF cross-spin wiring) is wrong; do NOT approve — return to Task 1/2.
  </how-to-verify>
  <resume-signal>Type "approved" once both open-shell UMP2 and DFUMP2 match live PySCF to <=1e-9, or describe the delta to debug.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python → pyscf-py (PyO3) | User-supplied UHF reference + frozen mask cross into Rust; shapes must be validated, not assumed |
| pyscf-mp2 → ao2mo/intor | F-order ao2mo output consumed as C-order kernel input — silent layout mismatch is the primary correctness hazard |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-nfb-01 | Tampering (data integrity) | cross_spin_ao2mo F→C repack | mitigate | Explicit index repack (not a memcpy); Task 1 Test A pins the index map with nocc_a≠nocc_b≠nvir; Task 3 open-shell live-PySCF byte-identity is the real gate |
| T-nfb-02 | Information disclosure (wrong energy) | PyO3 αβ un-gating | mitigate | Never reuse the αα block as αβ; route only through cross_spin_ao2mo / dfump2_kernel; `?`-propagate, never substitute zeros |
| T-nfb-03 | Denial of service (panic/OOB) | ChemistsEris shape on αβ block | mitigate | ump2_kernel::check_block already validates [nocc_a,nvir_a,nocc_b,nvir_b]; cross_spin_ao2mo sets nocc/nvir from α side; ao2mo::general validates nao spans |
| T-nfb-SC | Tampering | pip install maturin into .upstream-venv | mitigate | maturin is a well-known PyPI build tool (verify on pypi.org/project/maturin if version unfamiliar); installed only into the isolated upstream venv; pyscf-py dep graph confirmed libxc-free so no forbidden build |
</threat_model>

<verification>
- `cargo +nightly test -p pyscf-mp2 --test ump2_cross_spin --locked` → F→C repack + closed-shell guard pass.
- `cargo +nightly build -p pyscf-py --locked` → clean; no `plan:4` NotYetImplemented remains in mp.rs.
- `cargo +nightly clippy -p pyscf-mp2 -p pyscf-py` → no findings on new code.
- Live oracle (Task 3): open-shell UMP2 AND DFUMP2 within 1e-9 of upstream pyscf 2.12.1.
- `cargo tree -p pyscf-mp2` and `-p pyscf-py` → 0 libxc rows (no forbidden build).
- All heavy cargo/maturin output saved under log/.
</verification>

<success_criteria>
- cross_spin_ao2mo ships in pyscf-mp2 with the explicit F-order→C-order repack and is exported.
- Both PyO3 UMP2 αβ sites return real opposite-spin energies; the unrestricted DF path runs dfump2_kernel.
- Closed-shell UMP2==RMP2 guard passes (necessary).
- Open-shell byte-identity vs live PySCF passes for BOTH UMP2 and DFUMP2 (sufficient — the acceptance gate).
- No libxc build triggered; scope confined to pyscf-mp2 / pyscf-ao2mo / pyscf-py.
</success_criteria>

<output>
Create `.planning/quick/260601-nfb-cross-spin-o-a-v-a-o-b-v-b-ao2mo-wrapper/260601-nfb-SUMMARY.md` when done.
</output>
