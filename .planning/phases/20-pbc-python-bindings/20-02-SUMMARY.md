# 20-02 SUMMARY: every ignored PBC gate measured, tiered by runtime, and the tier written into its ignore reason

**Shipped:** 2026-09-14. All six tasks are complete. The first executor ran the build
and run pipeline (P1–P3, 2026-09-13 21:19 → 2026-09-14 09:43) and was cut off by a
session limit before writing any deliverable. The finalising executor did three
things: it re-enumerated the gates, re-ran the gates whose outcome later fixes could
move (P4, 14:48–14:57), encoded the tiers, and wrote this summary.

No commits were made and no git operations were run (D-20-A). No `src/` file was
edited.

Deliverable: [`measurements/pbc-oracle-tiers.md`](measurements/pbc-oracle-tiers.md).
Raw material is in `target/p20-02-logs/`.

## Task 1: floating upstream pins (already done; landed in `5e6dd65`)

- `tests/oracle/requirements.txt:19` → `pyscf==2.12.1`. Its comment (`:14-18`) names
  `fft_jk.get_k_kpts` / `exxdiv='ewald'` folding into `get_coulG` and the ~1e-5 shift in K.
  At `0f8b58d` the line was `pyscf>=2.6` (`:13`).
- `.github/workflows/ci.yml` audit. At `0f8b58d`, lines `:552`, `:648` and `:875` read
  `pip install "numpy>=1.26" "pyscf>=2.5"`, which are the `*-oracle-upstream-manual`
  jobs. Today all three read `"pyscf==2.12.1"`. `:297`, `:339` and `:341` were already
  `==2.12.1` and were not touched. The plan's line numbers `:284/:326/:328/:542` had
  drifted by +13/+10.
- `grep -rn 'pyscf[>~]=' tests/oracle/requirements.txt .github/` → no hits.

## Task 2: enumeration (the count moved, and the table says so)

`grep -rn '#\[ignore' crates/pyscf-pbc-*/tests/*.rs`, run at **2026-09-14 14:46:01**,
returns **177 hits**. That is **157 `#[ignore]` attributes plus 20 doc-comment
mentions**; all 20 are listed in tiers §6.

- The plan's count was 161 (2026-09-12).
- The count was 170 hits / 150 attributes at kickoff.
- `gdf_omega.rs` (+4, 20-05) and `gdf_ksymm_bisect.rs` (+3, 20-04) account for the change.
- `rsjk.rs` and all Phase-18 `pyscf-pbc-grad` tests have 0 ignores.
- `gates-v3.json` (14:46) has exactly the same key set as the pipeline's `gates.json` (09:22).

**No gate went unmeasured.**

## Task 3: measurement

Build: one `cargo test --release -p <crate> --test … --no-run -j 6` per crate in
`target/gate` with `LTO=false`; every build exited 0.

Run: one gate per process, `--ignored --exact <name> --test-threads=1`,
`PYSCF_ORACLE_VENV=1`, under `systemd-run … MemoryMax=16G`, `/usr/bin/time`, and
`timeout 600`.

PASS is counted only when the log contains `test result: ok. 1 passed`. No run printed
a skip line.

Four binary generations were used (P1 pre-D-20-E, P2 +coulG fix, P3 +20-05,
P4 +20-04 XC-grid fix). The `runs` column in the tiers table records each gate's
history.

**Re-runs on current binaries (P4).** These were the dft `gate.rs` (7) and
`gate_openshell.rs` (9) oracle gates, plus `krks_ksymm::kuks_ibz_energy_matches_full_bz`,
the one FAIL on a route the 20-04 fix touched.

| gate | before | after (P4) |
|---|---|---|
| KRKS Si PBE (floor row 3) | 6.450e-12 | **6.453e-12** PASS (tol 1e-11) |
| KRHF Si pseudopotential floor | P1 **FAIL 2.884e-1** (pre-D-20-E) → P2 4.158e-12 | **4.159e-12** PASS |
| KRKS Si PBE0 | P1 **FAIL 7.369e-2** → P2 5.588e-12 | **5.590e-12** PASS |
| KRKS He AE PBE (row 3a) | 8.482e-14 | **8.482e-14** PASS (tol 1e-12) |
| other 12 `gate`/`gate_openshell` | all PASS | all PASS; every change is in the last printed digit |
| `kuks_ibz_energy_matches_full_bz` | FAIL (precondition) | **FAIL**, same precondition (fixture symmetry-broken, max\|Δdm\| 1.194) |

The `df_ao2mo` FAILs (GDF route, touched by 20-05) had already been re-run on P3,
which built at 09:22 from the 09:05 sources; they still fail. The `df`/`dft`/`kernels`
sources changed since P3/P4 only by rustfmt churn.

Gates on routes touched by 20-05 that PASSED on P2 were re-run by 20-05 itself and
still pass; they were not re-run here. TIMEOUT rows were not re-run, per D-20-D.

## Task 4: tiers (by runtime alone)

| | T1 < 30 s | T2 30 s–5 min | T3 > 5 min / TIMEOUT@600 | total |
|---|---:|---:|---:|---:|
| gates | **81** | **55** | **21** (16 TIMEOUT, outcome not observed) | **157** |

Outcomes: **132 PASS, 9 FAIL, 16 TIMEOUT.**

Concurrency noise is real. The 17 P4 re-runs ran 1.1–7.7× faster (median ≈2.9×) than the same
gates on P1/P2, and 10 gates changed tier: 9 moved T2 → T1, and the KRHF floor went from
443.6 s to 185.5 s, T3 → T2. **66 rows** fall within 2× of a tier edge and are flagged ⚑.

## Task 5: tiers encoded in the ignore reasons

`encode_tiers.py --apply` prefixed or corrected **152** attribute lines across **51**
files. The other 5 already carried the measured tier: 3 in `gdf_omega` as T2, and 2 in
`gdf_ksymm_bisect` as T3.

Two pre-set tiers were corrected against measurement:

- `gdf_omega::gdf_sr_matches_upstream`: T2 → **T1** (27.44 s, ⚑)
- `gdf_ksymm_bisect::bisect_fixture_and_arm_grids`: T3 → **T1** (0.00 s)

Only the text inside `#[ignore = "…"]` changed. Each script edit asserted that the
target line still held an ignore attribute. The skip contract (`oracle_python()` →
`None` → skip) is untouched.

`rustfmt --edition 2024 --check` passes on all 51 touched files. A rebuild of
`pyscf-pbc-df --test exclude_dd_block --test gdf_mo_k --test gdf_omega` (the two
backslash-continued reason strings) succeeded, and `--list --ignored` still lists the
gate.

## Task 6: failures for follow-up (no tolerance loosened)

**Real oracle regressions: `pyscf-pbc-df` `df_ao2mo`, GDF route.**

| test | measured | gate / prior | note |
|---|---|---|---|
| `get_eri_matches_upstream_on_he_fcc` (T1) | **2.3298e-10** | < 1e-11; Phase 14 measured **1.667e-12** | Same value before and after 20-05. The contraction is exact (`get_eri_is_bit_exact…` 3.85e-37), so the gap is in `cderi`. Bisect owed. |
| `ao2mo_7d_matches_upstream_on_he_fcc` (T1) | **8.2146e-11** | < 1e-11; Phase 14 measured **1.984e-12** | Likely the same cause. |
| `get_eri_matches_upstream_on_diamond_gamma` (T2, 146.8 s) | **2.3500e-7** (pre-20-05 2.3735e-7) | < 1e-11; first completed run (Phase 14 left it owed) | |

**Unwired arms (machinery missing, named in the reason string):**

- `adc ea::gate_d_ip_spec_factors`: 9.358e-1 against 5e-4
- `tdscf uhf::live_bigbox_matches_molecular`
- `scf newton_ah::live_newton_matches_upstream_energy`

**Not defects:**

- Two gto `emit_ao_bits` children fail standalone because they need the parent's
  environment variable.
- `kuks_ibz_energy_matches_full_bz` is a fixture precondition failure.

## Deviations

- **Count.** 157 attributes / 177 grep hits instead of 161; the tree moved (Task 2).
- **Target dir.** `target/gate` (EXECUTION-NOTES §2) was used, not the plan's
  `~/.cargo-target-gate`.
- **Runtime budget.** D-20-D's 600 s kill means 16 T3 rows have no observed outcome.
  For two of them, 20-04 / 20-04-FIX observed an outcome outside this budget (tiers §5d).
- **Gates not re-verified after D-20-E.** P1 gates that passed before D-20-E (`gto`,
  most of `mp`, `ci`, `tools`) were not re-run on post-fix binaries. The risk is noted
  in tiers §2.

## Verification (run 2026-09-14 ~15:00)

```text
$ grep -n 'pyscf' tests/oracle/requirements.txt
19:pyscf==2.12.1
$ grep -n 'pyscf>=\|pyscf~=' tests/oracle/requirements.txt .github/workflows/*.yml | wc -l
0
$ grep -rc '#\[ignore' crates/pyscf-pbc-*/tests/*.rs | awk -F: '{s+=$2} END {print s}'
177            # = 157 attributes (157 table rows) + 20 comment mentions (tiers §6)
$ grep -c '^| [0-9]* | `pbc-' measurements/pbc-oracle-tiers.md
157
$ grep '^| [0-9]* | `pbc-' measurements/pbc-oracle-tiers.md | grep -vc '\*\*T[123]\*\*'
0              # every row carries a tier; every non-TIMEOUT row has measured seconds
$ grep -c 'T1:' crates/pyscf-pbc-*/tests/*.rs | awk -F: '{s+=$2} END {print s}'
81
$ grep -rn '^\s*#\[ignore = "T[123]: ' crates/pyscf-pbc-*/tests/*.rs | wc -l
157            # T1 81 / T2 55 / T3 21
$ git diff --stat -- crates/*/src/ | tail -1
 56 files changed, 1887 insertions(+), 375 deletions(-)
```

The last check is **not empty**, and none of those changes come from 20-02. They are
other sessions' concurrent work: Phase 18, rustfmt churn, and the `pyscf-py` agents.
The count grew from 54 to 56 during this session, while this plan edited only
`crates/pyscf-pbc-*/tests/*.rs` attribute lines. The plan's "empty" criterion cannot
hold in a shared working tree.
