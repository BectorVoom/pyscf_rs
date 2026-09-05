# 16-01 — MEASURE the floor. COMPLETE 2026-09-06.

**No Rust was written.** `git diff --stat crates/` for this plan is empty; the
Rust in the same commit belongs to 16-02 and 16-03, which are wave-0 siblings
with no dependency on this one.

**Deliverable:** `.planning/phases/16-periodic-cc-ci/measurements/README.md`,
seven committed scripts and their outputs. That file is the authority every
later Phase-16 gate cites; this summary records only what a reader needs to
know without opening it.

## The gate, restated in four documents together (Task 7)

`ROADMAP.md` (Phase-16 line), `PBC-MASTER-PLAN §7` row 16, `§8.8`'s D-PBC-29
entry, and `16-CONTEXT §2` now agree, and the old numbers — `1e-14` and
`1e-8` — are **struck through, not deleted**. Eleven gates, each naming its DF
route, fixture, mesh and `cell.precision`. The headline: **`KRCCSD e_corr` vs
upstream at `1e-7`, stated separately per DF route.**

## Five findings that change the plan set

1. **The DF split is `9.22e-4 Ha`, not `4.5e-6`.** Upstream's plane-wave pair
   (FFTDF, MDF) self-agrees to `2.59e-7` and its Gaussian pair (GDF, RSDF) to
   `6.82e-9`, but the two PAIRS are `9.22e-4` apart on diamond `gth-szv`
   `[1,1,2]` — three orders worse than the standing memory
   `rsdf-gdf-disagree-on-diamond` records at SCF level, because a `3.3e-3`
   mean-field difference propagates. A single "matches upstream" gate would be
   untestable.
2. **`symm_map` is `2.10×`, not `~4×`** (`README §7`). 176 orbit
   representatives for 512 triples at 2×2×2; `vvvv` is built by `ao2mo_7d` in
   BOTH paths so it saves nothing. **D-PBC-29 clause 3 is amended** and
   `16-REVIEW.md §3`'s own "report it if materially below 4" clause is
   discharged. The clause still stands — `symm_map` ships from the first
   version — at `~2×`.
3. **The symmetry loop is NOT bit-identical to an all-triples loop.**
   Upstream's own two paths differ by up to `1.32e-7`. **16-05 test 5's
   bit-identity requirement is replaced by `1e-6`.**
4. **Two plan tolerances were tighter than upstream's own agreement**, and are
   loosened by measurement rather than by negotiation: 16-07 test 2's `1e-10`
   for `KGCCSD == KRCCSD` (upstream's own gap `4.95e-9` → **1e-8**) and 16-08
   test 2's `1e-11` for spin-orbital-vs-RHF (T) (`2.86e-10` → **1e-9**).
5. **Three upstream anchor sets are excluded from every gate.**
   `test_krccsd.py::test_frozen_n3` FAILS on the vendored 2.12.1 tree
   (`ehf_bench` off by `1.56e-6`, asserted at 6 decimals) — the fourth
   instance of `15-VERIFICATION §7`'s standing caveat. Every `cu_metallic`
   anchor (`:338`, `:356`, `:359-366`) sits in a test upstream itself disabled
   with `@unittest.skip('Results not match')` at `:403`; run anyway, all eight
   diverge by `2e-2 … 7e-2`.

## The three numbers a later plan will reach for first

* **`conv_tol = 1e-9`, `conv_tol_normt = 1e-7`** — the plateau. Tightening to
  `1e-11` costs 46× the wall clock and moves `e_corr` by under `1e-9`.
* **The tier window is `(130.7, 264.9) MB` at diamond `gth-szv` 2×2×2** and is
  **EMPTY at `[1,1,2]`**, so 16-05 test 4 must cross the boundary at 2×2×2.
* **(T) fast-vs-slow is `2.95e-13` relative** — the one place a Phase-16 gate
  can be tight, because it is the same input through the same formula twice.

## Not reached

`gth-dzvp` at any mesh (byte counts in `README §6` are derived, not run);
`si`/`lif`/`graphene`; the `cell.precision` ladder of Task 2 (one `[2,2,2]` SCF
at the default `[47,47,47]` mesh exceeds the session budget), so 17-01 Gate B's
"the floor is integral-screening-limited" finding is carried into 16-14
UNVERIFIED for this phase; `_ERIS` wall clock at `nkpts = 27`/`64` (only the
orbit counts were taken there). All four are listed in `README §9`.
