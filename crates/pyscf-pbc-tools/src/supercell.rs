//! Supercell construction — the geometry-only core of `super_cell` and
//! `cell_plus_imgs` (`pyscf/pbc/tools/pbc.py:678-786`).
//!
//! As with [`crate::lattice`], everything that needs a `Cell` lives in
//! `pyscf_pbc_gto::supercell`; what is here is the translation-vector and
//! lattice arithmetic, which is pure geometry.

/// `supcell.a = np.einsum('i,ij->ij', n, a)` — scale lattice vector `i` by
/// `n[i]` (`pbc.py:715` and `:741`).
pub fn scale_lattice(a: &[[f64; 3]; 3], n: &[usize; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = n[i] as f64 * a[i][j];
        }
    }
    out
}

/// Translation vectors of an `ncopy[0] x ncopy[1] x ncopy[2]` supercell —
/// `pbc.py:706-713`.
///
/// Images run in the `+` direction only, unlike
/// [`cell_plus_imgs_translations`]. With `wrap_around`, indices from
/// `(n + 1) / 2` upward are shifted by `-n`, which centres the original cell on
/// the supercell — the same convention as `cell.make_kpts(wrap_around=True)`.
pub fn super_cell_translations(
    a: &[[f64; 3]; 3],
    ncopy: &[usize; 3],
    wrap_around: bool,
) -> Vec<[f64; 3]> {
    // xs = np.arange(ncopy[0]); if wrap_around: xs[(ncopy[0]+1)//2:] -= ncopy[0]
    let axis = |n: usize| -> Vec<i64> {
        (0..n as i64)
            .map(|x| {
                if wrap_around && x >= n.div_ceil(2) as i64 {
                    x - n as i64
                } else {
                    x
                }
            })
            .collect()
    };
    let (xs, ys, zs) = (axis(ncopy[0]), axis(ncopy[1]), axis(ncopy[2]));
    cartesian_prod_dot(a, &xs, &ys, &zs)
}

/// Translation vectors of a `+/- nimgs` supercell — `pbc.py:729-733`.
pub fn cell_plus_imgs_translations(a: &[[f64; 3]; 3], nimgs: &[usize; 3]) -> Vec<[f64; 3]> {
    let axis = |n: usize| -> Vec<i64> { (-(n as i64)..=(n as i64)).collect() };
    let (xs, ys, zs) = (axis(nimgs[0]), axis(nimgs[1]), axis(nimgs[2]));
    cartesian_prod_dot(a, &xs, &ys, &zs)
}

/// `Ls = lib.cartesian_prod((xs, ys, zs)).dot(a)` — the last index varies
/// fastest, matching `lib.cartesian_prod`.
fn cartesian_prod_dot(a: &[[f64; 3]; 3], xs: &[i64], ys: &[i64], zs: &[i64]) -> Vec<[f64; 3]> {
    let mut ls = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &x in xs {
        for &y in ys {
            for &z in zs {
                let t = [x as f64, y as f64, z as f64];
                let mut l = [0.0_f64; 3];
                for (j, lj) in l.iter_mut().enumerate() {
                    *lj = t[0] * a[0][j] + t[1] * a[1][j] + t[2] * a[2][j];
                }
                ls.push(l);
            }
        }
    }
    ls
}

/// `coords = Ls.reshape(-1,1,3) + cell.atom_coords()` flattened to `(-1, 3)` —
/// the image-major atom ordering of `_build_supcell_` (`pbc.py:757-759`).
///
/// Image `i` contributes atoms `i * natm .. (i + 1) * natm`, in the cell's own
/// atom order.
pub fn image_atom_coords(ls: &[[f64; 3]], atom_coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(ls.len() * atom_coords.len());
    for l in ls {
        for r in atom_coords {
            out.push([r[0] + l[0], r[1] + l[1], r[2] + l[2]]);
        }
    }
    out
}
