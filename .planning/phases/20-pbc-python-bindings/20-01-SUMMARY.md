# 20-01 SUMMARY — reconcile the gates (docs only)

**Shipped:** 2026-09-13. Documentation only: no Rust, Python or other code file
was created or edited. Every number written was copied from a named source file
and line; nothing was re-run.

## What changed

| file | change |
|---|---|
| `.planning/phases/20-pbc-python-bindings/measurements/README.md` | **new.** The v2.0 floor table (§1, 13 plan rows plus sub-rows, each with file:line provenance and MET / NOT MET / NOT RUN status), the irreducible gaps with the reason no implementation reaches them (§2), bitwise equality recorded as unattainable plus the same-implementation-A/B-only `to_bits()` rule (§3), the restated Phase-20 gate with `identity-gate-pre-20-09.out` as the evidence that it discriminates (§4), and Phase 18/19 gates as their own phases restated them (§5). `identity-gate-pre-20-09.out` left untouched. |
| `.planning/ROADMAP.md` (rows 459-466) | **13** `[ ]` kept, header corrected ("13-06 PARTIALLY shipped"), reason and `ao2mo_7d`-closed-by-14-05 note added. **14** `[x]`, "PLANNED / not yet implemented" replaced by the measured gates 1/1b/1c/2/3/4, **Gate 1b PARTIAL kept visible**. **15** `[x]`, gate → `2e-6` per DF route, MET, with residuals. **16** `[ ]` kept (verification says PARTIAL), gate → `1e-7` per DF route, **G2 NOT RUN kept visible**, EOM-EE claim corrected from `16-VERIFICATION §6.3`. **17** `[x]`, the five measured gates, **GDF Gate C 1.432e-06 NOT MET and Gate D NOT RUN kept visible**. **18** marked IN PROGRESS/PAUSED per `18-IMPLEMENTATION-CHECKPOINT.md`, the old gate struck and restated from `18-CONTEXT.md §2.1-§2.3`. **19** untouched. **20** gate restated as the identity contract + the two named upstream example scripts, citing `python/pyscf/tests/test_pbc_identity_gate.py` and 20-18. |
| `.planning/pbc/PBC-MASTER-PLAN.md` | §7 row 20: gate struck as vacuous and restated identically to ROADMAP, with the 18-plan scope note (`mpicc` dropped). §7 row 17: a 20-01 note that `1.703e-11` is NOT Gate C (it is the `eig`-route identity, `17-VERIFICATION.md:137-141`); Gate C FFTDF `KRHF` is 8.793e-14. No text deleted. |
| `.planning/pbc/PBC-DRIVER-INVENTORY.md` | header block: the `St` column is UNMAINTAINED (167 rows `[ ]`, 0 `[x]`, 0 `[~]`); the document is a scope map only; status authorities named. Checkboxes not refreshed. |
| `.planning/STATE.md` | front-matter counters reconciled with the counting rule written as YAML comments (v2.0 scope, dirs 09-20): `total_phases 12`, `completed_phases 7`, `total_plans 119`, `completed_plans 68` (incl. this file), `percent 57`; `last_updated`/`last_activity` → 2026-09-13. "Current focus" → Phase 20 in progress; the Phase-19 paragraph kept as "Previous focus". No other prose changed. |

### Which phases were ticked, and why

| phase | `*-VERIFICATION.md` says | box |
|---|---|---|
| 13 | "implementation complete for plans 13-01 … 13-05 and 13-07; plan 13-06 partially shipped" (`:3-6`) — not a phase closure; 0 SUMMARY files | `[ ]` |
| 14 | "four of five gates MET at close; Gate 3 … now MET" (`:6-7`); Gate 1b PARTIAL (`:19`) | `[x]` (1b visible) |
| 15 | "Verdict: CLOSED" (`:7`); the NOT MET row re-measured MET (`:42`) | `[x]` |
| 16 | "Status: PARTIAL" (`:3`); G2 NOT RUN (`:63`) | `[ ]` |
| 17 | "CLOSED" (STATE + verification §12); two gates NOT MET/NOT RUN (`:801-806`) | `[x]` (both visible) |

## Verification (commands from the plan, actual output)

1. `grep -c '^- \[ \] \*\*Phase 1[3-7]' .planning/ROADMAP.md` → **`2`** (plan expects `0`).
   Lines 459 (Phase 13) and 462 (Phase 16). **Deliberate** — see Deviation 1.
2. `grep -n '1e-14\|1e-15' .planning/ROADMAP.md` → **2 lines** in the Phase 9-20 block:
   `458` (Phase 12: "A 1e-15 Ha gate was considered and is not reachable") and
   `465` (Phase 19: `~~…1e-15 eV.~~` strikethrough). Rows 13-18 and 20, the ones
   this plan owns, carry none. Neither remaining match is an active gate — see Deviation 2.
3. `measurements/README.md` contains `1.432e-06` → `grep -c` = **1**; `NOT MET` → `grep -c` = **2**. PASS.
4. `grep -n 'unmodified upstream' .planning/ROADMAP.md` → **line 466** (Phase 20), and that line contains `_native.pbc` (`grep -c _native.pbc` on the match = 1). PASS.
5. `git diff --stat -- crates/` → **NOT empty**: `29 files changed, 2071 insertions(+), 144 deletions(-)`
   (24 files at the start of this plan). See Deviation 3 — none of it is 20-01's.

## Deviations

1. **Phases 13 and 16 stay `[ ]`** (plan Task 4 said tick all five). Per the
   execution brief, a box is ticked only where the phase's own verification says
   the phase is closed/complete: `13-VERIFICATION` claims plan-level
   implementation with 13-06 partial and has zero SUMMARY files, and
   `16-VERIFICATION` opens "Status: PARTIAL" with G2 NOT RUN. Both rows now say
   why. Verification 1 therefore returns 2, not 0.
2. **Two `1e-15` matches remain in the Phase 9-20 block**, both in rows this plan
   does not own: Phase 12's "considered and not reachable" note (historical, it
   is the 221-ulp argument itself) and Phase 19's struck-through gate, which the
   brief says to leave as is (and which `19-VERIFICATION.md` states is the
   struck-through convention). In the rows this plan rewrote, superseded gates
   are struck through with a pointer to the verification that quotes them
   verbatim instead of repeating the literal number.
3. **`git diff --stat -- crates/` cannot be empty on this tree.** Phase 18's work
   is uncommitted in the same crates (20-EXECUTION-NOTES §1, D-20-A), and other
   Phase-20 plans ran concurrently (the count grew from 24 to 29 files during
   this plan: `pyscf-pbc-scf/src/rsjk.rs`, `pyscf-py/{Cargo.toml,src/lib.rs,src/numpy_io.rs}`,
   `pyscf-pbc-gto/tests/cintx_moment_weighted_available.rs`). This plan made no
   edit under `crates/` or `python/`; its only writes were the five `.planning/`
   files above plus this summary.
4. **Phase 18 is IN PROGRESS/PAUSED, not "planned-only"** (plan Task 4): 21 PLAN
   files (not 15), 0 SUMMARY, `18-IMPLEMENTATION-CHECKPOINT.md`. **Phase 19 is
   CLOSED, not nonexistent**; its row was already rewritten 2026-09-13 and was
   not touched (20-EXECUTION-NOTES §1).
5. **Plan table corrections** (recorded in the README): the plan's "ksymm Gate
   C, FFTDF 1.703e-11" is not Gate C (Gate C FFTDF is 8.793e-14 `KRHF` /
   3.109e-14 `KRKS`); `krhf_bands_oracle.rs` asserts `< 1e-9` and does not
   record 1.68e-11 — that number's source is `20-CONTEXT.md:150`;
   `band_kpoints.rs:211` does record ~1.394e-9 (asserted `< 2e-9`); the CC route
   split `9.22e-04` is in `16 measurements/README.md §4` (`:165`), not §3.
   Every other plan-quoted number was found at its cited location.
6. **Concurrent writer on `measurements/README.md`.** While this plan was
   running, the README was rewritten on disk by another process (20:18:14) with
   the same plan's content in a fuller layout (adds §5, Phases 18-19). It was
   checked against the sources, found consistent with the numbers above, and
   kept rather than overwritten. Its §1 note says `PBC-MASTER-PLAN §7` row 17's
   mislabel "is fixed there by a note" — that note is this plan's edit.
7. **Phase-20 gate names two upstream scripts** — `examples/pbc/20-k_points_scf.py`
   and `examples/pbc/22-k_points_mp2.py`, taken from `20-18-PLAN.md:52-53,74`
   (plan Task 5 asked for "a named upstream script").
