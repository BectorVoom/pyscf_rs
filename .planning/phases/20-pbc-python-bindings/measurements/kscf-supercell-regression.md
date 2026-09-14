# `kscf.rs::supercell_equivalence_holds` regression — bisect

Measured 2026-09-13/14. Diamond GTH (`tests/common/mod.rs::diamond`), KRHF
`[2,1,1]` at mesh 15 vs gamma supercell `2x1x1` at mesh `[30,15,15]`, bound 1e-8.

## Verdict

**(b) — introduced by the UNCOMMITTED Phase-18 working-tree change** (plan 18-04,
D-PBC-31 clause 9). Not present at HEAD `0f8b58d`; not Phase 20.

**The k-point side moved; the supercell side is bit-identical** at every step
(`-10.531064341613`, same as `11-VERIFICATION.md` §1a).

Responsible hunk: `crates/pyscf-pbc-df/src/fftdf.rs`, `Fftdf::coulg_and_expmikr`.
It now keys the `coulG` cache on
`kdiff_index(&self.cell, dk)` (`crates/pyscf-pbc-df/src/ao_cache.rs`, `dk`
wrapped into `[0,1)` fractional) instead of the raw `dk` bits:

```rust
let class_key: CoulgKey = (
    kdiff_index(&self.cell, dk),   // <- the regression
    omega.map(f64::to_bits),
    exxdiv,
);
```

The claim in the doc comment that "`coulG` is genuinely class-invariant" is
false **per grid index**. `get_coulg` returns `coulG[g] = f(|wrap(G_g + dk)|^2)`.
For `dk' = dk + G0`, `coulG(dk')[g] = coulG(dk)[g + G0]`. That is a permutation
of the grid, not the same array. On `[2,1,1]` the pair loop sees `dk = +b1/2`
and `dk = -b1/2`. Both fall into class `0.5`, so whichever comes second gets
the first one's `coulG` with the wrong index order. It is then paired with its
own raw-keyed `expmikr`. The result is a wrong exchange matrix and a k-point
energy 0.184 Ha too high. Gamma (`dk = 0` only) never hits the collision, so
the supercell cannot move.

## Steps

| step | tree | k-point `e_tot` | supercell/2 | delta | exit |
|---|---|---|---|---|---|
| 0 | main working tree (Phase 18 + 20 uncommitted) | -10.347315387196 | -10.531064341613 | 1.837e-1 | 101 |
| 1 | worktree @ HEAD `0f8b58d` | -10.531064341456 | -10.531064341613 | 1.5650591933535907e-10 | 0 |
| 2 | HEAD + `pyscf-pbc-df/src/{fftdf,fft_jk,traits,lib,ao_cache,fft_jk_grad}.rs` from main tree | -10.347315387196 | -10.531064341613 | 1.837489544168438e-1 | 101 |
| 3 | step 2 with ONLY `kdiff_index(&self.cell, dk)` -> raw `dk.to_bits() as i64` in the key | -10.531064341456 | -10.531064341613 | 1.5650591933535907e-10 | 0 |

Step 2 is bit-identical to the working tree, and step 3 is bit-identical to
HEAD. Together they isolate the single key expression. No `pyscf-pbc-gto` /
`pyscf-gto` / Phase-20 file is needed to reproduce the failure or to clear it.

## Commands

```sh
# step 0 (main tree)
CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false \
  cargo test --release -p pyscf-pbc-scf --test kscf -j 4 -- --exact supercell_equivalence_holds --nocapture
# steps 1-3
git worktree add /home/user/Documents/workspace/pyscf_rs-head-wt 0f8b58d
cp -a --reflink=always target/gate target/gate-head   # warm start, btrfs
(cd ../pyscf_rs-head-wt && CARGO_TARGET_DIR=/home/user/Documents/workspace/pyscf_rs/target/gate-head \
  CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p pyscf-pbc-scf --test kscf -j 4 \
  -- --exact supercell_equivalence_holds --nocapture)
git worktree remove --force /home/user/Documents/workspace/pyscf_rs-head-wt
```

Logs: `target/kscf-wt.log`, `target/kscf-head.log`, `target/kscf-stepA.log`
(step 2), `target/kscf-stepB.log` (step 3).

## Fix direction (not applied — the file is Phase 18's, not Phase 20's)

- Key `coulG` on the raw `dk` bits again, as HEAD does. This is bit-exact to HEAD.
- The alternative is to keep the class key and permute the cached array by
  the `G0` offset. That needs care at the wrap boundary, where the
  `equal2boundary` ties are resolved per representative.
- Either way, `crates/pyscf-pbc-df/tests/fft_jk_grad.rs::coulg_build_counter_reads_nkpts`
  asserts `nkpts` builds (not `nkpts^2`) and will have to change. Its premise
  is the same false invariance.
- Also re-check whether the 18-04 gradient gates passed while this was wrong
  (the gradient K path uses the same cache). A `[2,1,1]`-style mesh with
  `±b/2` pairs, compared with the raw-key result, would expose it.
