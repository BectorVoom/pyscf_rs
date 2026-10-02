# T09 — Projector tables and values on the host

**Goal.** Host (plain Rust) code that produces upstream's projector table
`S[row, g]`, and an oracle test for it and for the AO transform. No kernel
yet. This is the reference every later kernel is tested against.

## The mathematics (from `pyscf/pbc/df/fft.py:128-150`)

For every atom `a` that has a pseudopotential, every channel `l` with
`nproj > 0` (radius `rl`), every `i < nproj`, every `m < 2l+1` there is one
**row**. Row order: atom, then `l`, then `i`, then `m`.

With `Gk = Gv[g] + kpt`, `alpha = ½·rl²`, `x² = |Gk|²·rl²`:

```
S[row, g] = exp(+i · Gv[g]·R_a)                    // conj(SI): uses Gv, NOT Gk
          · rl^(l+1.5) · π^1.25 · exp(−alpha·|Gk|²)
          · cfac(l) · Σ_c T_l[m, c] · Gk_x^ix · Gk_y^iy · Gk_z^iz
          · qli(x², l, i)
```

- `cfac(l)` = `pyscf_kernels::common_fac_sp(l)`
- `T_l` = `pyscf_kernels::cart2sph_l_matrix(l)`, row-major `(2l+1, ncart)`
- `(ix, iy, iz)` = `pyscf_kernels::cart_powers(l)[c]`
- `qli(x², l, i) = c0 + c1·x² + c2·x⁴` with (`pyscf/pbc/gto/pseudo/pp.py:150-185`):

| l | i | prefactor | c0 | c1 | c2 |
|---|---|---|---|---|---|
| 0 | 0 | `4·√2` | 1 | 0 | 0 |
| 0 | 1 | `8·√(2/15)` | 3 | −1 | 0 |
| 0 | 2 | `(16/3)·√(2/105)` | 15 | −10 | 1 |
| 1 | 0 | `8·√(1/3)` | 1 | 0 | 0 |
| 1 | 1 | `16·√(1/105)` | 5 | −1 | 0 |
| 1 | 2 | `(32/3)·√(1/1155)` | 35 | −14 | 1 |
| 2 | 0 | `8·√(2/15)` | 1 | 0 | 0 |
| 2 | 1 | `(16/3)·√(2/105)` | 7 | −1 | 0 |
| 2 | 2 | `(32/3)·√(2/15015)` | 63 | −18 | 1 |
| 3 | 0 | `16·√(1/105)` | 1 | 0 | 0 |
| 3 | 1 | `(32/3)·√(1/1155)` | 9 | −1 | 0 |
| 3 | 2 | `(64/45)·√(1/1001)` | 99 | −22 | 1 |

The stored coefficients are `prefactor·c0`, `prefactor·c1`, `prefactor·c2`.

## Read first

- `pyscf/pbc/df/fft.py` lines 114–176 (upstream, the thing being ported).
- `crates/pyscf-pbc-df/src/ft_ao/single.rs` lines 46–157 (`ft_ao_mol`: how
  this crate uses `cart_powers`, `common_fac_sp`, `cart2sph_l_matrix`).
- `crates/pyscf-core/src/mole.rs` lines 159–183 (`GthPseudo`,
  `GthProjector { r, nproj, h }`). Use `GthProjector.h` — the RAW matrix.
  Do NOT use `HlBlock` from `pseudo/vnl.rs` (that one is rescaled for the
  analytic route).

## Do

1. Create `crates/pyscf-pbc-df/src/pp_gspace.rs` and add `pub mod pp_gspace;`
   to `crates/pyscf-pbc-df/src/lib.rs` (next to `pub mod fftdf;`).
2. In it, write:

```rust
//! The non-local GTH pseudopotential in reciprocal space —
//! `pyscf/pbc/df/fft.py:114-176`.

use pyscf_kernels::{cart_powers, cart2sph_l_matrix, common_fac_sp};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// One projector function — one row of `S[row, g]`.
#[derive(Debug, Clone)]
pub struct ProjRow {
    pub atom: usize,
    pub l: usize,
    pub i: usize,
    pub m: usize,
    pub rl: f64,
}

/// One `(atom, l)` channel. Its rows are `row0 + i·(2l+1) + m`.
#[derive(Debug, Clone)]
pub struct ProjChannel {
    pub l: usize,
    pub nproj: usize,
    pub row0: usize,
    /// RAW `nproj × nproj` coupling matrix, row-major (`GthProjector::h`).
    pub h: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ProjTables {
    pub rows: Vec<ProjRow>,
    pub channels: Vec<ProjChannel>,
}

/// `[c0, c1, c2]` with `qli = c0 + c1·x² + c2·x⁴` (`pp.py:150-185`).
pub fn qli_coeffs(l: usize, i: usize) -> Option<[f64; 3]> { /* the table above */ }

/// Rows in the order atom, l, i, m (`fft.py:133-150`).
pub fn proj_tables(cell: &Cell) -> Result<ProjTables, PbcDfError> { /* see below */ }

/// `S[row, g]`, planar, row-major `(nrow, gv.len())`: index `row * n + g`.
pub fn proj_values_host(
    cell: &Cell,
    t: &ProjTables,
    gv: &[[f64; 3]],
    kpt: [f64; 3],
) -> Result<(Vec<f64>, Vec<f64>), PbcDfError> { /* the formula above */ }
```

   `proj_tables`: loop `ia in 0..cell.mol.natm`; symbol
   `&cell.mol._atom[ia].0`; `cell.pseudo.as_ref().and_then(|p| p.get(sym))`;
   skip the atom when `None`; for `(l, proj)` in
   `pseudo.projectors.iter().enumerate()` with `proj.nproj > 0`: push one
   `ProjChannel` and `nproj·(2l+1)` rows. If `qli_coeffs(l, i)` is `None`
   return `Err(PbcDfError::Backend(format!("pp_gspace: no qli for l={l} i={i}")))`.

   `proj_values_host`: atom positions from `cell.mol.atom_coords()` (Bohr).
   Use `rayon` over rows only if this crate already depends on it
   (`grep rayon crates/pyscf-pbc-df/Cargo.toml`); otherwise a plain loop.
3. Append to `crates/pyscf-pbc-df/tests/pp_gspace_oracle.rs` (the file from
   T08) two tests. `MESH_SMALL` is a tiny mesh so the tables are small:

```rust
const MESH_SMALL: [usize; 3] = [5, 5, 5];

/// Compare a planar `(re, im)` pair with the oracle payload.
fn planar_dev(re: &[f64], im: &[f64], want: &serde_json::Value) -> f64 {
    let t = pyscf_algebra::CTensor::from_planes(re.to_vec(), im.to_vec());
    max_dev(&[t], want)
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn projector_table_matches_upstream() {
    let cell = ktao3(BASIS);
    let kidx = 4; // a complex k-point of the 3x3x1 mesh
    let Some(want) = oracle(&cell, BASIS, MESH_SMALL, "proj", kidx) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH_SMALL)).expect("Gv");
    let t = pyscf_pbc_df::pp_gspace::proj_tables(&cell).expect("tables");
    // K: 2 + 2*3 = 8, Ta: 3 + 2*3 + 2*5 = 19, 3 O: 1 each
    assert_eq!(t.rows.len(), 8 + 19 + 3, "projector row count");
    let (re, im) = pyscf_pbc_df::pp_gspace::proj_values_host(&cell, &t, &gv, kpts[kidx]).expect("values");
    let w = planar_dev(&re, &im, &want);
    println!("projector table max|delta| vs upstream = {w:e}");
    assert!(w < 1e-12, "projector table deviates by {w:e}");
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn host_ft_ao_matches_upstream_including_f_shells() {
    for basis in [BASIS, "gth-dzvp-molopt-sr"] {
        let cell = ktao3(basis);
        let kidx = 4;
        let Some(want) = oracle(&cell, basis, MESH_SMALL, "aokG", kidx) else {
            eprintln!("SKIP: {GATE} is not set");
            return;
        };
        let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
        let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH_SMALL)).expect("Gv");
        let (re, im) = pyscf_pbc_df::ft_ao::ft_ao_kpt(&cell.mol, &gv, kpts[kidx]).expect("ft_ao");
        let w = planar_dev(&re, &im, &want);
        println!("{basis}: ft_ao max|delta| vs upstream = {w:e}");
        assert!(w < 1e-11, "{basis}: ft_ao deviates by {w:e}");
    }
}
```

## Verify

ORACLE `-p pyscf-pbc-df --test pp_gspace_oracle`:
- `projector_table_matches_upstream` passes (deviation below 1e-12);
- `host_ft_ao_matches_upstream_including_f_shells` passes;
- `get_pp_matches_upstream_on_ktao3_at_a_coarse_mesh` still FAILS (T10 fixes it).

## If it fails

- Projector deviation is large (≥ 1e-3) only on rows with `l = 1` or
  `l = 2`: the order of `m` or the factor `cfac(l)` differs. Print, for the
  first failing row, your value and upstream's at `g = 1`; check the ratio.
  A constant ratio → a missing/extra factor. Values that match a different
  `m` → the `m` order; use the order `cart2sph_l_matrix` gives and do not
  re-sort.
- Projector deviation large on ALL rows by a phase: you used `Gk` in the
  `exp(+i·…)` factor. It must be `Gv`.
- `host_ft_ao…` fails: do NOT edit `ft_ao/single.rs`. STOP and report the
  basis and the deviation — T10 and T13 depend on it.
- `get_gv` / `ft_ao_kpt` path not found: check the `pub use` lines in
  `crates/pyscf-pbc-gto/src/lib.rs` and `crates/pyscf-pbc-df/src/ft_ao/mod.rs:94`.
