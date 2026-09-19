# `df_ao2mo` oracle gates vs upstream: bisect

Measured 2026-09-14 (15:00-16:00 JST). The three failures are listed in
`pbc-oracle-tiers.md` §5a.

## Verdict

**This is not a cderi regression, and no port code moved.** The test harness
compared two different upstream routes.

Commit **`a423c0e`** (2026-08-30, "port _RSGDFBuilder and _RSMDFBuilder —
Phase 14 Gate 3 MET", plan 14-07 Task 7d) flipped the port's default. The hunk
is in `crates/pyscf-pbc-df/src/gdf/mod.rs`, `Gdf::new`:

```rust
-            prefer_ccdf: true,
+            // Task 7d — upstream's default. See the field's docs.
+            prefer_ccdf: false,
```

The oracle in `crates/pyscf-pbc-df/tests/df_ao2mo.rs` still runs
`mydf._prefer_ccdf = True`, which is `_CCGDFBuilder` plus the
`exclude_dd_block` / `direct_scf_tol` / `estimate_rcut` patches. The Rust side
called `Gdf::new(..)` without pinning the route. From `a423c0e` onwards the
port therefore builds **RSGDF** `cderi`, and the gate compares it with upstream
**CCGDF**. Upstream's own two routes disagree by 5.222e-10 (He-fcc 2×2×2 KRHF
energy) and 4.502e-06 (diamond gamma KRHF energy)
(`14-gdf-mdf-rsdf-rsjk/measurements/ccdf.py`, `14-VERIFICATION.md:42,243`).
2.33e-10 and 2.35e-7 are the size of that route gap.

20-02 was right that the contraction is exact. The attribution device was also
blind to this, because it builds its stub `mydf` from the port's own `cderi`
whichever route produced it.

Other tests in the same crate already pin the route next to a CC oracle, for
example `tests/band_kpoints.rs:243` (`df.prefer_ccdf = true` with
`mydf._prefer_ccdf = True`). `df_ao2mo.rs` was the one that was missed.

## Bisect

Method: sibling sandboxes under `/home/user/Documents/workspace/gdfbis-*/`.
Each holds a `pyscf_rs` worktree, and where needed a `cintx` worktree. The
other path deps (`cube-math`, `rmath`, `libxc_rs`, `xcfun_rs`) are symlinks,
so `../cintx` and `../../../cintx` resolve inside the sandbox. Each sandbox had
its own `CARGO_TARGET_DIR`, warm-started with `cp -a --reflink=always
target/gate`. `cintx` is a separate repository, so it had to be controlled
separately.

Both steps 1 and 2 were needed. `cintx` gained 40 commits after Phase 14,
including "make every integral family bit-identical to libcint 6.1.3", so it
was a live suspect.

| step | pyscf_rs | cintx | `get_eri` He-fcc | `ao2mo_7d` He-fcc | exit |
|---|---|---|---|---|---|
| 0 | main working tree (HEAD `5e6dd65` + uncommitted) | main `33c7031` | **2.329758608254906e-10** | 8.214651181503996e-11 | 101 |
| 1 | `7b65198` (Phase 14 close) | `e623562` (contemporaneous) | **1.6673884495332914e-12** | 1.9839130338539235e-12 | 0 |
| 2 | `7b65198` | HEAD `33c7031` | 1.6667223157185163e-12 | 1.983940789429539e-12 | 0 |
| 3a | `a61a4a9` (§11 GDF reopen) | HEAD | 2.329753057139783e-10 | 8.214626201485942e-11 | 101 |
| 3b | `a423c0e` | HEAD | 2.329753057139783e-10 | 8.214631752601065e-11 | 101 |
| 4 | `789530b` (direct parent of `a423c0e`) | HEAD | 1.6667223157185163e-12 | 1.983940789429539e-12 | 0 |
| 5 | `a423c0e` + test-only `df.prefer_ccdf = true` | HEAD | **1.666666804567285e-12** | 1.9839685450051547e-12 | 0 |

What each step shows:

- **Step 1** reproduces `14-VERIFICATION.md:86-87` to every printed digit. The
  oracle (`.venv` native libs installed 2026-08-20, vendored `pyscf/` 2.12.1
  unchanged since then) has not moved.
- **Step 2** shows that 13 days of `cintx` moves the number by 7e-16. `cintx`
  is exonerated.
- **Steps 3b and 4** make `a423c0e` the first bad commit. `789530b` is its
  direct parent (`git log 789530b..a423c0e` is that one commit).
- **Step 5** is the A/B at the bad commit. Pinning the route in the TEST alone
  restores Phase 14's number. No `src/` change is needed.

Also cleared:

- The uncommitted `gdf/cderi_store.rs` hunk is rustfmt-only: one call re-wrapped
  and one closure joined, with no token changes.
- The 20-05 edits (`incore/int3c.rs`, `rsdf_builder/j2c.rs`) are cleared by
  step 3b, which is 16 days before them.

Commands (steps 1-5; `SB` is the sandbox):

```sh
git -C pyscf_rs worktree add --detach /home/user/Documents/workspace/$SB/pyscf_rs <commit>
git -C cintx    worktree add --detach /home/user/Documents/workspace/$SB/cintx    <commit>   # step 1 only; else symlink
cp -a --reflink=always pyscf_rs/target/gate pyscf_rs/target/gate-bis-<x>
cd $SB/pyscf_rs && PYSCF_ORACLE_VENV=/home/user/Documents/workspace/pyscf_rs/.venv \
  CARGO_TARGET_DIR=/home/user/Documents/workspace/pyscf_rs/target/gate-bis-<x> CARGO_PROFILE_RELEASE_LTO=false \
  cargo test --release -p pyscf-pbc-df --test df_ao2mo -j 6 -- --ignored --nocapture --test-threads 1 he_fcc
```

Logs: `target/gdfao2mo-bisect/step{0..7}-*.log`, with runner
`target/gdfao2mo-bisect/run.sh`. All worktrees, sandboxes and `gate-bis-*`
target dirs were removed afterwards, and `git worktree list` is back to the
main tree only in both repositories.

## Fix (applied, test-only)

File: `crates/pyscf-pbc-df/tests/df_ao2mo.rs`. It was last touched at 14:54:59
by the bulk T-tier re-labelling, and nothing was editing it after that. The
change adds `df.prefer_ccdf = true;` with a provenance comment to all four
oracle tests that pair with the CC oracle:

- `get_eri_matches_upstream_on_he_fcc`
- `ao2mo_7d_matches_upstream_on_he_fcc`
- `get_eri_matches_upstream_on_diamond_gamma`
- `get_eri_is_bit_exact_with_upstream_over_the_same_cderi` (so the device
  attributes the same `cderi` as the gates above it)

No gate was loosened and no `src/` file was touched.

Re-measure on the main tree:

```sh
PYSCF_ORACLE_VENV=1 CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false \
  cargo test --release -p pyscf-pbc-df --test df_ao2mo -j 6 -- --ignored --nocapture --test-threads 1 \
  upstream_on_he_fcc get_eri_is_bit_exact
```

| test | after | gate | Phase 14 |
|---|---|---|---|
| `get_eri_matches_upstream_on_he_fcc` (screens equalised) | **1.666666804567285e-12** PASS | 1e-11 | 1.667e-12 |
| same, upstream default screen | 2.7502031207937705e-9 | < 1e-8 | 2.750e-9 |
| `ao2mo_7d_matches_upstream_on_he_fcc` | **1.9839130338539235e-12** PASS | 1e-11 | 1.984e-12 |
| `get_eri_is_bit_exact_with_upstream_over_the_same_cderi` | 1.1102230246251565e-16 PASS | round-off | 1.110e-16 |

`test result: ok. 3 passed`, 10.94 s, exit=0 (`step6-main-fixed-he.log`).

## Diamond gamma

**The 1e-11 gate has never been met, and there is no earlier number to regress
from.**

- `14-VERIFICATION.md:124-137` (§3, Gate 1b PARTIAL) says the test "is written,
  `#[ignore]`d on cost … its number is owed". The CC `make_j3c` was killed after
  22 min real time in Phase 14.
- 20-02's 2.3735e-7 / 2.3500e-7 were therefore the first completed runs. Like
  He-fcc, they compared port RSGDF with upstream CCGDF. That is why they ran in
  83-147 s: the RS real-space sum is cheap.

With the route pinned (`step7-main-fixed-diamond.log`):

```sh
PYSCF_ORACLE_VENV=1 CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false \
  /usr/bin/time cargo test --release -p pyscf-pbc-df --test df_ao2mo -j 6 -- --ignored --nocapture \
  --exact get_eri_matches_upstream_on_diamond_gamma
```

| | value |
|---|---|
| port CCGDF vs upstream CCGDF (`exclude_dd_block=False`, screens at 1e-14, flattened `estimate_rcut`) | **3.8426711743144715e-8** FAIL vs 1e-11 |
| wall time | **2793 s** (46.5 min, ~12 cores), 240 MB RSS. Its tag says `T2`; by D-20-D it is **T3** |
| exit | 101 |

So the route pin removes most of the gap (2.35e-7 → 3.84e-8, 6.1× smaller).
What remains is **not attributed**, and the gate was not loosened. Provenance
for what it could be:

- **Not D-PBC-23.** The dd-block (1.835e-8 at 2×2×2, 2.900e-8 at gamma, in the
  energy; `14-VERIFICATION.md:490`) is switched off on the upstream side, so it
  is matched by construction.
- **Diamond's metric is near-singular.** `eig_min = 3.17e-11`, below
  `linear_dep_threshold`, and it is still decomposed by Cholesky
  (`14-VERIFICATION.md:377-387`). Phase 14 recorded that system's own identity
  check `V j2c Vᴴ = I` at **3.094e-08** and called it "a conditioning floor".
  The residual here is the same order. A 1e-12-level difference in `j3c`/`j2c`
  amplified through `j2c⁻¹` would produce it. That is a hypothesis, not a
  measurement.
- **D-PBC-21.** The `ft_aopair` screening residual (5.121e-10,
  `14-VERIFICATION.md:492`) is the other named carry-over on this path.

Owed, in this order:

1. Run the attribution device on diamond: upstream `get_eri` over the port's
   `cderi`. This confirms the contraction.
2. Compare the raw fused `j3c` and `j2c` elementwise against upstream
   (`gdf_builder.rs::cderi_fingerprint_matches_upstream_diamond`, T3). If both
   are at ~1e-12 while `cderi` is at ~1e-8, the residual is the conditioning
   of the gate, and the right fix is to gate diamond at the `j3c`/`j2c` level
   rather than on `get_eri`. That is a gate redesign for its owner, not a
   loosening done here.

Until then, `get_eri_matches_upstream_on_diamond_gamma` stays a named T3
failure.

## Follow-ups (not done here)

- Every other `Gdf::new` / `Mdf::new` next to a `_prefer_ccdf = True` oracle
  deserves the same audit. `band_kpoints.rs` is already pinned. A mechanical
  check: grep for `_prefer_ccdf = True` in test oracles, then assert that the
  Rust fixture in the same test sets `prefer_ccdf = true`.
- `pbc-oracle-tiers.md` §5a rows 1-2 can be closed with this file as the
  reference.
