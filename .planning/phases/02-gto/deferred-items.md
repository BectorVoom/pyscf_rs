# Phase 02 (gto) — Deferred / Surfaced Items

## DI-02-11-CINTX-NCTR — cintx mis-evaluates general-contraction (nctr>1) shells — RESOLVED

**Surfaced by:** plan 02-11 (general-contraction parser fix), 2026-05-24.
**Resolved by:** cintx branch `fix/general-contraction-nctr-1e` (commit `6b14d48`,
`fix(1e): evaluate general-contraction (nctr>1) shells correctly`) + plan 02-11's
pyscf-rs-side coefficient-layout fix, 2026-05-24.
**Disposition:** RESOLVED. No longer blocks 02-11.

### What was found (original gap)

The pre-`6b14d48` cintx 1e kernel (`one_electron.rs`) summed EVERY `(ci,cj)`
contraction pair into the single `(0,0)` Cartesian slot (the `cart_buf` had no
contraction dimension), so for nctr>1 shells the block was mostly zero and the one
non-zero entry was unnormalized — producing non-symmetric, non-PSD overlaps.

### How it was resolved

1. **cintx `6b14d48`** rewrote the kernel to accumulate ONE Cartesian block per
   `(ci,cj)` contraction pair, then cart→sph each block and scatter it into the
   contraction-major AO grid (`ctr*(2l+1) + m`). The pyscf-rs path-deps point at
   the cintx working tree, so `cargo test -p pyscf-gto` recompiles cintx-cubecl and
   picks up the fix (Cargo.lock unchanged) — the cintx#11 / 05-08 cross-repo
   precedent.
2. **pyscf-rs 02-11** found that `projection.rs` flattened the cintx `Shell`
   coefficients COLUMN-major (`[ctr][prim]`) while the cintx kernel reads them
   ROW-major (`coefficients[pi*n_ctr + ci]`). For nctr=1 the two layouts coincide
   (the bug was invisible while the parser truncated to 1 column); for nctr>1 they
   scrambled. Fixed `projection.rs` to interleave row-major to match cintx.

### Verification (always-on, in-tree)

- cc-pVDZ O (general contraction nctr=2 S-block, l≤2): `int1e_ovlp_sph` finite,
  symmetric, UNIT diagonal; the 2×2 S-block byte-matches the independent
  normalised Gram `[[1, -0.214], [-0.214, 1]]`.
- ANO O (full stack S nctr=8 … G nctr=2): overlap finite with UNIT diagonal across
  ALL l; l≤2 sub-block symmetric.
- minao H2 docstring byte-match (03-13) STILL holds (`dm[0]=0.94758917`).
- minao H2O `Tr(dm·S)` recovered 7.9 → 9.86 (the correct heavy-atom projection).

(`crates/pyscf-gto/tests/general_contraction.rs`,
`crates/pyscf-scf/tests/init_guess_minao.rs`.)

---

## DI-02-11-CINTX-NCTR-HIGHL — cintx l≥3 cart→sph asymmetry for nctr>1 cross-blocks

**Surfaced by:** plan 02-11, 2026-05-24.
**Disposition:** cross-repo gap (cintx), low priority — NOT a v1 blocker.

### What was found

After the 02-11 coefficient-layout fix, the cintx 1e kernel
(`one_electron.rs`@`6b14d48`) evaluates general-contraction (nctr>1) overlaps
correctly for **l ≤ 2** (proven exact by cc-pVDZ and the ANO s/p/d sub-block:
symmetric, unit diagonal). For nctr>1 it introduces a SYMMETRY violation in the
**(p,f) and (d,g) cross-blocks** (|Δ| up to ~6 on the ANO O overlap at l≥3),
i.e. the per-contraction-pair cart→sph scatter is incorrect when one shell has
l ≥ 3 AND nctr > 1. The diagonal stays correct (each AO still normalises to 1);
only the high-l cross-block off-diagonal is asymmetric.

### Impact on pyscf-rs — NONE for current deliverables

- `init_guess_by_minao` is UNAFFECTED: the occupation walk assigns occ=0 to every
  l≥3 (f,g) ANO contraction, so those columns are filtered out of the density. The
  H2O minao trace (9.86) and the H2 byte-match exercise only l≤1 occupied ANO
  orbitals. The l≤2 sub-block is symmetric, so the occupied projection is correct.
- No current pyscf-rs numeric path consumes a generally-contracted l≥3 overlap.

### Required cintx work (when an l≥3 nctr>1 numeric path lands)

Audit the cart→sph staging scatter in `one_electron.rs` for l≥3 with nctr>1 (the
`Representation::Spheric` branch's `staging[ii + jj*di_sph]` index vs the
`cart_to_sph_1e` output ordering); extend the cintx safe-API arity-2 parity oracle
to an ANO-style basis with f/g general contractions. Tracked here until a pyscf-rs
phase (e.g. correlation on ANO-RCC) needs it.
