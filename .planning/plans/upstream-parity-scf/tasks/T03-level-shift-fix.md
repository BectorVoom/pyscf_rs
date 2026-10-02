# T03 — Level shift: fix

**Goal.** `level_shift` follows upstream, and each SCF type hands it the
density upstream hands it.

| SCF type | density given to `level_shift` | upstream line |
|---|---|---|
| KRHF, KRKS, KGHF, KGKS, k-symmetry drivers | the density as it is | `pbc/scf/khf.py:157` |
| KUHF, KUKS | each spin density as it is | `pbc/scf/kuhf.py:99-101` |
| KROHF, KROKS | `(D_alpha + D_beta) · ½`, one matrix per k-point | `pbc/scf/krohf.py:80` |

## Read first

- `crates/pyscf-pbc-scf/src/khooks.rs` lines 75–90 (`diis_dms`).
- `crates/pyscf-pbc-scf/src/krohf.rs` lines 305–330 (`diis_dms` override).
- `crates/pyscf-pbc-scf/src/kscf.rs` lines 49–66 and 160–182.

## Do

1. `crates/pyscf-pbc-scf/src/kscf.rs`, function `level_shift`:
   replace `0.5 * sds.re[i]` by `sds.re[i]` and `0.5 * sds.im[i]` by
   `sds.im[i]`. Replace its doc comment by:

```rust
/// `mol_hf.level_shift(s, d, f, factor)` — `scf/hf.py:794-795`:
/// `F' = F + (S − S·D·S)·factor`. Which `D` each driver passes is
/// [`KOverrideHooks::level_shift_dms`].
```
2. `crates/pyscf-pbc-scf/src/khooks.rs`, directly after the `diis_dms`
   method of the trait, add:

```rust
/// The density handed to the level shift. Default: the density channels
/// themselves (`khf.py:157`, `kuhf.py:99-101`). ROHF overrides it with
/// `(dma + dmb)·½` (`krohf.py:80`).
fn level_shift_dms(&self, dms: &KDms) -> KDms {
    dms.clone()
}
```
3. `crates/pyscf-pbc-scf/src/kscf.rs` line ~179: change
   `level_shift(&mut fock, &s1e, &hooks.diis_dms(&dm), cfg.level_shift, nao);`
   to
   `level_shift(&mut fock, &s1e, &hooks.level_shift_dms(&dm), cfg.level_shift, nao);`
4. `crates/pyscf-pbc-scf/src/krohf.rs`, directly after its `diis_dms`
   override, add:

```rust
fn level_shift_dms(&self, dms: &KDms) -> KDms {
    // krohf.py:80 — dm_sf * 0.5
    let mut sf = self.diis_dms(dms);
    for m in sf[0].iter_mut() {
        for v in m.re.iter_mut() {
            *v *= 0.5;
        }
        for v in m.im.iter_mut() {
            *v *= 0.5;
        }
    }
    sf
}
```
5. **Forwarding wrappers.** A wrapper that forwards `diis_dms` to an inner
   driver must forward the new method too, or it silently uses the default.
   In each place below there is a method `fn diis_dms`. Add a copy of it
   directly underneath with BOTH occurrences of `diis_dms` replaced by
   `level_shift_dms`:

   | file | line of `fn diis_dms` |
   |---|---|
   | `crates/pyscf-pbc-dft/src/kroks.rs` | 174 |
   | `crates/pyscf-py/src/pbc/scf.rs` | 510 |
   | `crates/pyscf-py/src/pbc/dft.rs` | 588, 891, 974 |
   | `crates/pyscf-py/src/pbc/kbridge.rs` | 264 |
   | `crates/pyscf-pbc-dft/tests/kukspu.rs` | 192 |

   Then run `grep -rn "fn diis_dms" crates/ --include=*.rs` and
   `grep -rn "fn level_shift_dms" crates/ --include=*.rs`. Every file in the
   first list must be in the second.
## Verify

1. TEST `-p pyscf-pbc-scf --test level_shift_formula` → `2 passed`.
2. TEST `-p pyscf-pbc-scf --test kscf --test kscf_resume --test grad_fc_hoist`
   → same pass/fail as `baseline.txt`.
3. TEST `-p pyscf-pbc-dft --test krks_dzvp_small_cell` → passes.
4. CHECK `-p pyscf-py` → compiles.

## If it fails

- `krks_dzvp_small_cell` no longer converges within its cycle limit: in that
  test file halve the `level_shift` value it passes (e.g. `0.3` → `0.15`).
  This is exact: for a restricted run the new formula at half the shift gives
  the same orbitals as the old formula at the full shift. Add a one-line
  comment saying so. Do not change anything else in the test.
- CHECK of `pyscf-py` takes over an hour: that is the libxc rebuild; let it
  finish.
