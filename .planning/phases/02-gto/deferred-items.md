# Phase 02 (gto) — Deferred / Surfaced Items

## DI-02-11-CINTX-NCTR — cintx mis-evaluates general-contraction (nctr>1) shells

**Surfaced by:** plan 02-11 (general-contraction parser fix), 2026-05-24.
**Disposition:** cross-repo gap (cintx), mirrors the cintx#11 pattern. BLOCKS the
02-11 success criteria (minao H2O `Tr(dm·S)==nelec`, ANO overlap PSD).

### What was found

The 02-11 parser fix to `crates/pyscf-gto/src/basis/nwchem.rs` is correct: a
primitive block with `exp + N` coefficient columns now loads as `N` contractions
(ANO O S-block nctr=8, ANO H S-block nctr=6, cc-pVDZ O first S-block nctr=2 — all
verified via `load_basis` in `tests/general_contraction.rs`). The downstream
`ParsedBasis` → `projection::build_atoms_and_shells` → cintx `BasisSet` layer is
ALSO correct: `meta().ao_count(shell)` reports `nctr*(2l+1)`, offsets are right,
`total_ao` matches `nao_nr` (ANO H = 40, cc-pVDZ O = 14).

The gap is in **cintx's spherical one-electron kernel evaluation** of nctr>1
shells. Direct probe of `SessionRequest::evaluate` per shell pair:

| shell pair | nctr | extents | comp_leading | nnz / total | symptom |
|---|---|---|---|---|---|
| cc-pVDZ O P (l=1) | 1 | [3,3] | true | 3/9, diag=[1,1,1] | CORRECT |
| cc-pVDZ O S (l=0) | 2 | [2,2] | true | 1/4, diag=[2.28, 0] | BROKEN |
| ANO H S (l=0) | 6 | [6,6] | true | 1/36, diag=[2885, 0,…] | BROKEN |
| ANO H P (l=1) | 4 | [12,12] | true | 3/144 (all in col 0) | BROKEN |

For nctr>1 the block is mostly zero and the single non-zero entry is unnormalized
(2885, 1735), so the resulting AO overlap matrix is **non-symmetric** and **not
PSD** (`S[6,10]=0` vs `S[10,6]=1735`).

### Root cause (cintx)

`cintx/crates/cintx-cubecl/src/kernels/one_electron.rs` (~line 497, 536-547):

```rust
let mut cart_buf = vec![0.0_f64; nci * ncj];   // sized for ONE contraction only
...
for ci in 0..n_ctr_i {
    let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
    for cj in 0..n_ctr_j {
        let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
        let weight = coeff_i * coeff_j;
        for k in 0..prim_buf.len() {
            cart_buf[k] += weight * prim_buf[k];   // NO ci/cj offset — all
                                                   // contractions summed on
                                                   // top of contraction (0,0)
        }
    }
}
```

`cart_buf` has no contraction dimension; every `(ci,cj)` pair accumulates into the
same `nci*ncj` slot. It must be `[nci*ncj * n_ctr_i*n_ctr_j]` with the write
indexed by `(ci,cj)`, then cart→sph applied per contraction block. (The arity-4
`int2e` parity tests pass because they were exercised with nctr=1 bases.)

### Impact on pyscf-rs

- **Cannot land the 02-11 parser fix alone.** It exposes this cintx bug end-to-end:
  the 03-13 minao H2 docstring byte-match (a GOLD-STANDARD test) FLIPS from
  `dm[0]=0.94758917` to `dm[0]=553` because ANO H now loads nctr=6 and the
  `intor_cross` overlap is garbage. The previously-green byte-match was implicitly
  relying on the parser BUG (ANO truncated to nctr=1, which cintx evaluates fine).
- **cc-pVDZ silently corrupted.** cc-pVDZ O goes 13→14 AOs (the 14 is correct), but
  the new S-contraction's integrals are wrong, so every cc-pVDZ DF/SCF/MP2 numeric
  becomes wrong while structural-only tests (finite/non-zero/shape) still pass.

### Required cintx work (cintx#11-style cross-repo task)

Fix `one_electron.rs` (overlap/kinetic/nuclear) to write a per-contraction-pair
Cartesian buffer and apply cart→sph per contraction block; extend the safe-API
arity-2 parity oracle to a generally-contracted basis (e.g. cc-pVDZ O / ANO).
Until then the 02-11 minao caveat (SCF-05) and ANO/cc-pVDZ numeric correctness
remain blocked at the cintx integral layer.
