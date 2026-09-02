//! `pyscf/pbc/dft/multigrid/utils.py` (70 l) — `_take_4d`/`_takebak_4d`
//! (plan 17-12, Task 3).
//!
//! Generic index-window extraction/insertion on a flat row-major 4D array
//! `(n0,n1,n2,n3)`. Ported for completeness (Task 3's file list names this
//! module) and unit-tested directly against the definition below;
//! `crate::multigrid::numint`'s `window_index_map` /
//! `insert_gspace_window` / `extract_gspace_window` remain the ones
//! actually used on multigrid's hot path — they specialise to "axis 0 full,
//! axes 1-3 an `fftfreq` window" and precompute a flat index map once per
//! call rather than re-resolving four axis index lists on every element, so
//! this module is the literal general-purpose upstream primitive, not a
//! second competing implementation of the same call site. `_take_5d`/
//! `_takebak_5d` (`utils.py:54-70`) are one-line wrappers that flatten the
//! leading two axes and delegate to the 4D form — not ported separately, as
//! no call site in this plan's scope needs a 5th axis (`multigrid_pair.py`
//! only ever calls the 4D form).

/// One axis's index selection: [`AxisIndex::Full`] = the whole axis
/// (upstream's `s is None` branch, `numpy.arange`); [`AxisIndex::Explicit`]
/// = an explicit index list, upstream-style negative indices resolved
/// against the axis length.
#[derive(Debug, Clone)]
pub enum AxisIndex {
    Full,
    Explicit(Vec<i64>),
}

fn resolve(ax: &AxisIndex, dim: usize) -> Vec<usize> {
    match ax {
        AxisIndex::Full => (0..dim).collect(),
        AxisIndex::Explicit(idx) => idx
            .iter()
            .map(|&i| {
                if i < 0 {
                    (dim as i64 + i) as usize
                } else {
                    i as usize
                }
            })
            .collect(),
    }
}

/// `_take_4d(a, indices)` — `utils.py:21-33`. `a` is a flat row-major
/// `(n0,n1,n2,n3)` array; `indices` selects each axis. Returns the selected
/// flat row-major array plus its shape.
pub fn take_4d(a: &[f64], shape: [usize; 4], indices: [&AxisIndex; 4]) -> (Vec<f64>, [usize; 4]) {
    let idx0 = resolve(indices[0], shape[0]);
    let idx1 = resolve(indices[1], shape[1]);
    let idx2 = resolve(indices[2], shape[2]);
    let idx3 = resolve(indices[3], shape[3]);
    let out_shape = [idx0.len(), idx1.len(), idx2.len(), idx3.len()];
    let mut out = vec![0.0f64; out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]];
    for (o0, &i0) in idx0.iter().enumerate() {
        for (o1, &i1) in idx1.iter().enumerate() {
            for (o2, &i2) in idx2.iter().enumerate() {
                for (o3, &i3) in idx3.iter().enumerate() {
                    let src = ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3;
                    let dst = ((o0 * out_shape[1] + o1) * out_shape[2] + o2) * out_shape[3] + o3;
                    out[dst] = a[src];
                }
            }
        }
    }
    (out, out_shape)
}

/// `_takebak_4d(out, a, indices)` — `utils.py:35-52`. Scatters `a` (whose
/// shape must match `indices`' resolved axis lengths) into `out` (shape
/// `out_shape`), **adding** at the selected positions — upstream's
/// `lib.takebak_2d` adds rather than overwrites, and the multigrid call
/// sites (`_eval_rhoG`'s per-level `_takebak_4d` into the shared `rhoG`
/// buffer) rely on that to combine multiple grid levels into one array.
///
/// # Panics
/// If `a`'s length does not match the resolved index lengths' product.
pub fn takebak_4d(
    out: &mut [f64],
    out_shape: [usize; 4],
    a: &[f64],
    a_shape: [usize; 4],
    indices: [&AxisIndex; 4],
) {
    let idx0 = resolve(indices[0], out_shape[0]);
    let idx1 = resolve(indices[1], out_shape[1]);
    let idx2 = resolve(indices[2], out_shape[2]);
    let idx3 = resolve(indices[3], out_shape[3]);
    assert_eq!(idx0.len(), a_shape[0], "_takebak_4d: axis 0 length mismatch");
    assert_eq!(idx1.len(), a_shape[1], "_takebak_4d: axis 1 length mismatch");
    assert_eq!(idx2.len(), a_shape[2], "_takebak_4d: axis 2 length mismatch");
    assert_eq!(idx3.len(), a_shape[3], "_takebak_4d: axis 3 length mismatch");
    assert_eq!(
        a.len(),
        a_shape[0] * a_shape[1] * a_shape[2] * a_shape[3],
        "_takebak_4d: `a` length does not match `a_shape`"
    );
    for (o0, &i0) in idx0.iter().enumerate() {
        for (o1, &i1) in idx1.iter().enumerate() {
            for (o2, &i2) in idx2.iter().enumerate() {
                for (o3, &i3) in idx3.iter().enumerate() {
                    let src = ((o0 * a_shape[1] + o1) * a_shape[2] + o2) * a_shape[3] + o3;
                    let dst = ((i0 * out_shape[1] + i1) * out_shape[2] + i2) * out_shape[3] + i3;
                    out[dst] += a[src];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M-02 — the geometry cache key.
//
// `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md` §2.3.2/§2.3.3:
// both multigrid drivers rebuild their entire geometry — decontraction, task
// list, per-level collocation tables, grid-block partition, per-block reach
// lists — on EVERY call, and the v1 driver does it twice per call (once for
// the density, once for `pass2`). None of it depends on the density, so all of
// it is recomputed identically every SCF cycle.
//
// The drivers take `&Cell` per call rather than owning one, so a cache needs a
// key. This is that key.
// ---------------------------------------------------------------------------

/// A fingerprint of everything the multigrid geometry reads out of a [`Cell`].
///
/// # What it must cover, and why each entry is here
///
/// `build_pshells` reads `mol._bas`, `mol._env` (which is where the exponents,
/// the contraction coefficients AND the atom coordinates live), `mol.cart`,
/// `cell.precision`, `cell.rcut`, `cell.vol()` and `omega(cell)`;
/// `multi_grids_tasks_for_ke_cut` and `pair_grid_levels` additionally read
/// `cell.lattice_vectors()`, `cell.mesh` and `cell.ke_cutoff`. Hashing all of
/// them means a cache hit implies identical geometry.
///
/// **Bit patterns, not values** — `to_bits()` on every float, so `-0.0` and
/// `0.0` are distinct keys and no rounding can collide two different cells.
///
/// This is a HASH, so a collision is possible in principle. It is used the way
/// `KNumInt`'s AO cache uses `coord_hash`: as a fast identity check on a cell
/// the caller is re-passing, not as a security boundary.
pub fn cell_fingerprint(cell: &pyscf_pbc_gto::Cell) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();

    for row in cell.lattice_vectors() {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    cell.mesh.hash(&mut h);
    cell.precision.to_bits().hash(&mut h);
    cell.rcut.to_bits().hash(&mut h);
    match cell.ke_cutoff {
        Some(k) => (1u8, k.to_bits()).hash(&mut h),
        None => 0u8.hash(&mut h),
    }
    cell.dimension.hash(&mut h);

    let mol = &cell.mol;
    mol.cart.hash(&mut h);
    mol.nbas.hash(&mut h);
    mol.nao_nr.hash(&mut h);
    mol._bas.hash(&mut h);
    // `_env` carries the atom coordinates, the primitive exponents and the
    // contraction coefficients — every float `build_pshells` reads.
    for v in &mol._env {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}
