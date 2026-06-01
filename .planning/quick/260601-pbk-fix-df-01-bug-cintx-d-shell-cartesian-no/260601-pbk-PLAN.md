---
phase: quick-260601-pbk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: false
requirements: [DF-01]
---

<objective>
Fix DF-01: the cintx int2c2e d-shell normalization bug that makes the DF metric
(P|Q) wrong (dz²/dx²−y² come out half), degrading rs DF-MP2/DFUMP2 by ~1e-3.
User chose "attempt the cintx kernel fix now" (the fix lives in the external
cintx crate `cintx-cubecl/src/kernels/center_2c2e.rs`; int2e/cc-pVDZ confirmed
unaffected, so it is safe to touch the 2c2e path).
</objective>

<task type="checkpoint:human-verify">
  <name>Locate + fix the cintx 2c2e d-shell defect (host + #[cube] device), validate byte-identity</name>
  <action>Pinpoint the wrong factor in the 2c2e Rys g-tensor / cartesian assembly by
  comparing cintx's cartesian int2c2e d-block to upstream's int2c2e_cart; fix BOTH
  host (fill_g_tensor_2c2e) and device (center_2c2e_kernel) paths; validate
  int2c2e (P|P) d-block, DF-RMP2, DFUMP2 vs upstream + regression (int2e/cc-pVDZ,
  conventional UMP2, existing cintx 2c2e host==device tests).</action>
  <done>rs DF-RMP2/DFUMP2 byte-match upstream; no regression in int2e/cc-pVDZ or
  the conventional MP2 path; cintx 2c2e tests green.</done>
</task>
