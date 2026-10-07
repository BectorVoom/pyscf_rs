---
title: PLAN-CHECK — bitexact-gpu (self-review, UNVERIFIED)
status: UNVERIFIED
reason: "No agent named `plan-checker`/`gsd-plan-checker` is installed in this environment (available subagents: explore, general). The review below is a self-review against the skill's checker criteria using codegraph + local tools — it is NOT an independent PASS."
passes: 1 (self)
updated_at: 2026-09-19T00:00:00Z
---

# PLAN-CHECK — self-review (UNVERIFIED, not an independent PASS)

## Verdict: CONDITIONAL-GO (self-assessed) — ready for Wave 1 only

Checked against the six checker criteria:

1. **Spec coverage** — GO. Every spec S1–S8 maps to ≥1 task (§Coverage);
   every acceptance behavior A1–A5 has a Red task. No spec without a task,
   no task without a spec.
2. **Dependent malfunction risk** — GO with note. T9 (the only production
   math change) requires unmodified oracle suites to pass and pins
   `PAIRWISE_CHUNK`; blast radius (algebra→kernels→method crates) is named
   in SPEC §7. Risk: fixed-lane cost on T4 — explicitly measured in T9
   before merge (R2).
3. **Latent bugs / edge cases** — ISSUES (non-blocking, recorded):
   - (a) S1 harness depends on `kernel(dm0)` existing for PBC-KRHF
     bindings; verified for molecular `scf.rs:319` [VERIFIED: CODEGRAPH]
     but PBC-side `dm0` support is [INFERRED] — T1 Red will expose it;
     fallback: feed dm via `make_rdm1` + `max_cycle=0` if missing.
   - (b) YN small-gap k-points (~6.5e-4 Ha, prior session) can make SCF
     legs noisy — mitigated by frozen-dm design (R4), but T1 acceptance
     should assert digest equality of the frozen fixture first.
   - (c) T4/ROCm/shader-f64-WGPU runner availability is operationally open
     (S7) — Wave 1 deliberately needs no hardware.
4. **Prerequisites / order / parallelization** — GO. T9 blocked on T5;
   T10 on T5–T8; T11 last. T1∥T2∥T3∥T4 verified conflict-free (separate
   files: bench harness vs algebra tests vs workflow file).
5. **Dependency assumptions vs evidence** — GO. All load-bearing claims
   re-verified this session (reduce.rs partial geometry, krks one-shot
   bands, ORACLE-07 wording/pending status, `kernel(dm0)` hook);
   research-report claims labeled [R], prior-session YN numbers labeled
   [UNVERIFIED] (not in repo).
6. **Validation sufficiency** — GO. Per-task commands named; CPU
   bit-exact suites + no-fma + wall lints guard regressions; 1-ulp floor
   rule (measurements §3) restated as invariant so no 0-ulp GPU gate can
   slip in.

## Required revisions before implementation

- None blocking Wave 1. T1 implementer: add fixture-digest assertion (3b).
- Before Wave 2: confirm hardware access; else S2/S4/S5 stay open and S7
  remains scaffolding (plan already encodes this).

## Revision history

- Pass 1 (self, 2026-09-19): CONDITIONAL-GO as above. No independent pass
  claimed. Next: independent `plan-checker` PASS when such an agent exists,
  or human approval of SPEC §2 non-goals (notably: no 0-ulp CUDA gate).
- Amendment A1 (2026-09-19, user directive — xcfun excluded, KRHF vehicle):
  S5 rewritten (XC-substrate trace → libcint-vs-cintx integral parity,
  CPU-only, Wave 1); S1/T1 re-vehicled to KRHF + `get_bands`; T7 rewritten;
  waves rebalanced (Wave 2 = T5/T6/T8 only). Delta re-review (self):
  no new production-code risk (T7 is census-only); S5 acceptance no longer
  needs T4; coverage S5→T7 preserved; verdict stands at CONDITIONAL-GO for
  Wave 1. Research Q3 (XC trace) void.
