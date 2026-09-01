//! Multigrid v2 G-space helpers — `_backend_c.py`'s `gradient_gs` and
//! `get_gga_vrho_gs` (plan 17-12, Task 3).
//!
//! Both are documented, in upstream's OWN source, by an exact `einsum`
//! equivalent — this file's tests (`crates/pyscf-kernels/tests/multigrid_pair.rs`)
//! assert exactly that equivalence, no upstream oracle needed:
//!
//! * `gradient_gs(f_gs, Gv)` ≡ `einsum('np,px->nxp', f_gs, 1j*Gv)`
//!   (`_backend_c.py:27-31`).
//! * `get_gga_vrho_gs(v, v1, Gv, weight, ngrid, fac=2.)` ≡
//!   `v -= fac * 1j * einsum('px,xp->p', Gv, v1); v *= weight`
//!   (`_backend_c.py:46-49`).
//!
//! # Why plain host functions, not `#[cube(launch_unchecked)]` kernels
//!
//! Both reduce to a HANDFUL of real multiply-adds per output element (no
//! transcendentals, no reduction beyond `x/y/z` — three terms). D-PBC-03's
//! rule ("don't write a bespoke kernel where existing primitives — or, as
//! here, a few scalar multiply-adds — suffice") applies directly: adding a
//! `#[cube(launch_unchecked)]` kernel here would cost a device
//! upload/download round-trip for less arithmetic than the round-trip
//! itself, with no measured win on this workspace's default CPU backend
//! (`pyscf-algebra` defaults to CPU — see the standing note in
//! `/home/user/.claude/…/MEMORY.md`: "cube barriers and fixed 256-wide
//! cubes are pathological there"). They still live in `pyscf-kernels`
//! (ALG-06 / D-PBC-25 corollary: `pyscf-pbc-dft` may not know `cubecl`), so
//! the crate-layering rule is satisfied without a device launch this
//! workload does not need. `zhadamard_kernel` (K-04) is the converse
//! precedent: THAT op earned a kernel because it batches over `nao²`-scale
//! matrices in a GEMM-adjacent hot loop; a `(nset, 3, ngrids)` gradient
//! transform on a handful of reciprocal-space fields per SCF cycle does not.

use pyscf_algebra::CTensor;

/// `gradient_gs(f_gs, Gv)` — `_backend_c.py:27-41`.
///
/// `f_gs` is `(nset, ngrids)` (flattened row-major: `f_gs.re[n*ngrids+g]`).
/// `Gv` is `(ngrids, 3)` row-major, Bohr⁻¹. Returns `(nset, 3, ngrids)`
/// row-major: `out[(n*3+x)*ngrids+g] = i·Gv[g,x]·f_gs[n,g]`.
///
/// # Panics
/// If `f_gs.len()` is not a multiple of `ngrids`, or `Gv.len() != 3*ngrids`.
pub fn gradient_gs(f_gs: &CTensor, gv: &[f64], ngrids: usize) -> CTensor {
    assert_eq!(gv.len(), 3 * ngrids, "gradient_gs: Gv must be (ngrids,3)");
    assert!(
        f_gs.re.len().is_multiple_of(ngrids),
        "gradient_gs: f_gs.len() must be a multiple of ngrids"
    );
    let nset = f_gs.re.len() / ngrids;
    let mut out = CTensor::zeros(nset * 3 * ngrids);
    for n in 0..nset {
        for g in 0..ngrids {
            let fr = f_gs.re[n * ngrids + g];
            let fi = f_gs.im[n * ngrids + g];
            for x in 0..3 {
                let gx = gv[g * 3 + x];
                // i * Gv[g,x] * (fr + i*fi) = -Gv*fi + i*Gv*fr
                let idx = (n * 3 + x) * ngrids + g;
                out.re[idx] = -gx * fi;
                out.im[idx] = gx * fr;
            }
        }
    }
    out
}

/// `get_gga_vrho_gs(v, v1, Gv, weight, ngrid, fac=2.)` — `_backend_c.py:43-59`.
///
/// `v` is `(ngrids,)`, updated **in place**: `v -= fac * i * Σ_x Gv[:,x]·v1[x,:]`,
/// then `v *= weight`. `v1` is `(3, ngrids)` row-major.
///
/// # Panics
/// If `v.len() != ngrids`, `v1.len() != 3*ngrids`, or `Gv.len() != 3*ngrids`.
pub fn get_gga_vrho_gs(v: &mut CTensor, v1: &CTensor, gv: &[f64], weight: f64, ngrids: usize) {
    get_gga_vrho_gs_fac(v, v1, gv, weight, ngrids, 2.0)
}

/// [`get_gga_vrho_gs`] with an explicit `fac` (upstream default `2.`).
pub fn get_gga_vrho_gs_fac(
    v: &mut CTensor,
    v1: &CTensor,
    gv: &[f64],
    weight: f64,
    ngrids: usize,
    fac: f64,
) {
    assert_eq!(v.re.len(), ngrids, "get_gga_vrho_gs: v must be (ngrids,)");
    assert_eq!(v1.re.len(), 3 * ngrids, "get_gga_vrho_gs: v1 must be (3,ngrids)");
    assert_eq!(gv.len(), 3 * ngrids, "get_gga_vrho_gs: Gv must be (ngrids,3)");
    for g in 0..ngrids {
        let mut dot_re = 0.0f64;
        let mut dot_im = 0.0f64;
        for x in 0..3 {
            let gx = gv[g * 3 + x];
            dot_re += gx * v1.re[x * ngrids + g];
            dot_im += gx * v1.im[x * ngrids + g];
        }
        // fac * i * (dot_re + i*dot_im) = -fac*dot_im + i*fac*dot_re
        let sub_re = -fac * dot_im;
        let sub_im = fac * dot_re;
        let new_re = (v.re[g] - sub_re) * weight;
        let new_im = (v.im[g] - sub_im) * weight;
        v.re[g] = new_re;
        v.im[g] = new_im;
    }
}
