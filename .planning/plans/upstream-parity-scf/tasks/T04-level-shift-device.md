# T04 — Level shift: run the two matrix products on the device

**Goal.** `S·D` and `(S·D)·S` are computed by the device GEMM
(`pyscf_algebra::zgemm_dense`) instead of the scalar host loop `mm`.
For YTa7O19 (nao 910, 9 k-points) the host loop is about 1.4e10 complex
multiply-adds per SCF cycle on one thread.

## Read first

- `crates/pyscf-pbc-scf/src/addons.rs` lines 277–283 — how this crate
  already gets a client and calls `zgemm_dense`. Copy that pattern.
- `crates/pyscf-algebra/src/zgemm.rs` lines 36–50 — `zgemm_dense(client, a,
  b, m, k, n)`: `a` is `m × k`, `b` is `k × n`, row-major, result `m × n`.
- Manual: `11_launch_overhead_and_transfers.md` §2 (do not re-create the
  client inside the loop).

## Do

1. In `crates/pyscf-pbc-scf/src/kscf.rs`, change `level_shift` to return
   `Result<(), PyscfRsError>`:
   - get the client ONCE at the top of the function, as `addons.rs:277-279`
     does (map the error with the same `PyscfRsError::Core(CoreError::
     InvalidMolecule(format!(…)))` shape this file already uses);
   - replace `let sd = mm(&s1e[k], &dms[s][k], nao);` by
     `let sd = pyscf_algebra::zgemm_dense(&client, &s1e[k], &dms[s][k], nao, nao, nao)` + error mapping;
   - same for `sds`;
   - end with `Ok(())`.
2. At the call site add `?`.
3. If `mm` is now unused in `kscf.rs`, delete it. If something else uses it,
   leave it.
4. In `crates/pyscf-pbc-scf/tests/level_shift_formula.rs` the two calls to
   `level_shift(...)` now return a `Result`: append `.expect("level_shift")`.

## Verify

1. TEST `-p pyscf-pbc-scf --test level_shift_formula` → `2 passed`
   (tolerances unchanged: 1e-14 and 1e-13).
2. TEST `-p pyscf-pbc-scf --test kscf --test kscf_resume` → same as baseline.

## If it fails

- Tolerance 1e-13 exceeded by less than 1e-11: the device GEMM sums in a
  different order. Raise ONLY that test's tolerance to `1e-12` and write the
  measured value in a comment. If it is worse than 1e-11, STOP and report.
- `kscf_resume` (bit-identical resume) fails: STOP and report. Do not
  loosen it.
