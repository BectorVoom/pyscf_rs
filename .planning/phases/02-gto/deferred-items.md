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

## DI-02-11-CINTX-NCTR-HIGHL — cross-l 1e overlap/kinetic/nuclear asymmetry — RESOLVED

**Surfaced by:** plan 02-11, 2026-05-24.
**Resolved by:** cintx branch `fix/general-contraction-nctr-1e` (commit `9af2164`,
`fix(1e,ecp): general contraction for ECP + cross-l overlap layout`), 2026-05-24.
**Disposition:** RESOLVED.

### Root cause (corrected from the original framing)

Originally suspected as an l≥3-and-nctr>1 cart→sph scatter bug. The true root cause
is broader and independent of nctr: `contract_overlap/kinetic/nuclear` emitted
ROW-major `out[bra*ncj + ket]`, but `cart_to_sph_1e` (and the pyscf-rs block stitch)
read COLUMN-major bra-fastest `out[ket*nci + bra]`. The two coincide only when a
shell is l=0 (a vector) or li==lj with a symmetric block, so EVERY cross-l block
with both li,lj>0 (p-d, p-f, d-g) was silently transposed/scrambled — at any nctr.
It was invisible because the cintx byte-identity oracle is feature-gated (does not
run in the sandbox) and the in-tree analytic tests are s-s only.

### How it was resolved

`9af2164` changed the three contraction functions to emit column-major
`out[ket*nci+bra]` (matching `cart_to_sph_1e`'s documented input and the pyscf-rs
stitch). Single-contraction s-s/s-p/s-d/p-s are byte-unchanged (vectors/symmetric).
The `one_electron.rs` Cart launch branch reads/writes column-major to match.

### Verification (always-on, cintx in-tree)

- `test_cross_l_overlap_is_symmetric`: ovlp + kin (+ nuc for p-d) cross-l transpose
  symmetry for p-d, p-f, d-g, single contraction.
- `test_general_contraction_high_l_cross_block_is_symmetric`: the executor's exact
  case — generally-contracted d(nctr=2) × f(nctr=2) full-block transpose symmetry.
- pyscf-rs downstream (gto+scf+df+mp2): 280 tests pass, no regression.

---

## DI-02-11-ECP-NCTR — cintx int1e_ecp mis-evaluated general-contraction shells — RESOLVED

**Surfaced by:** plan 02-11 regression sweep (Cu/LANL2DZ S-block is nctr=2), 2026-05-24.
**Resolved by:** cintx `9af2164`. **Disposition:** RESOLVED.

The nwchem parser fix made LANL2DZ load its real contractions (nctr>1), exposing two
bugs in `ecp.rs`: (1) `launch_ecp` sized `gctr`/`needed` as ncart*ncart (no nctr) →
index-out-of-bounds panic, since `ecp_type1/2_cart` write the full nctr block; fixed
with nctr-aware sizing + per-contraction cart→sph scatter. (2) The kernel read
`Shell.coefficients` column-major while the canonical layout (all other kernels) is
row-major; fixed with a `coeffs_col_major()` transpose-once helper (identity at nctr=1).
The 02-10 `ecp_int1e_oracle` had been passing against a TRUNCATED (wrong) LANL2DZ; it now
passes (symmetric + finite) against the correct nctr=2 basis.

---

## DI-02-11-CINTX-NUC-HIGHL — cintx nuclear attraction limited to li+lj ≤ 3 (≤2 Rys roots)

**Surfaced by:** plan 02-11 cross-l test development, 2026-05-24.
**Disposition:** cross-repo gap (cintx), pre-existing — NOT a v1 02 blocker; tracked.

### What was found

`contract_nuclear` (`one_electron.rs`) computes `nrys_roots = (li+lj)/2 + 1` but only
implements 1 or 2 Rys roots (`rys_root1_host` / `rys_root2_host`). For li+lj > 3
(nrys_roots ≥ 3 — e.g. d-d, p-f, d-g) the per-root loop indexes the 2-element root
array out of bounds and panics. Overlap and kinetic use the analytic g-tensor and are
unaffected at any l; this is specific to nuclear attraction (`int1e_nuc`).

### Impact on pyscf-rs

- minao / ANO overlap path: UNAFFECTED (overlap only).
- Becomes relevant when a non-DF SCF builds the nuclear-attraction `hcore` over an
  orbital basis with high angular momentum (any d-d block already has li+lj=4). The
  default SCF path in-tree uses DF or low-l bases; full non-DF SCF on d/f bases needs
  `rys_root3`/`rys_root4`+ in cintx. Tracked here.

### Required cintx work

Implement higher-order Rys roots (3+) in `math/rys.rs` and generalize the
`if nrys_roots == 1 {..} else {rys_root2}` dispatch in `contract_nuclear`.
