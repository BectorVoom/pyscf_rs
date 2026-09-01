//! Multigrid **pair-fused** real-space collocation —
//! `pyscf/pbc/dft/multigrid/multigrid_pair.py` (plan 17-12, Task 2).
//!
//! Upstream's v2 (`MultiGridNumInt2`) reaches the C library through twelve
//! entry points (`_backend_c.py`); this file ports the two that do the
//! actual floating-point work, `grid_collocate_drv` and `grid_integrate_drv`
//! (`pyscf/lib/dft/grid_collocate.c`, `grid_integrate.c`, 689 + 1648 lines of
//! hand-optimised, cache-blocked Hermite-Gaussian recursion). **Porting that
//! C line-by-line was judged out of this plan's time budget** — it is
//! `libdft`'s single largest body of PBC C code and its recursions are
//! written for a submesh/cache-blocking scheme this port does not replicate
//! (see `crate::multigrid::pair`'s module doc). What ships instead is a
//! mathematically faithful, independently-testable reformulation of the
//! SAME physical quantity:
//!
//! v1 (`multigrid.py`, already shipped, `crate::multigrid_collocate`)
//! evaluates each AO **separately** on the grid and multiplies pairs of AO
//! values together host-side (`level_rho`'s `Term{slot_i, slot_j, coeff}`).
//! v2 instead fuses a primitive PAIR `(p ∈ shell i, q ∈ shell j, image L)`
//! through the **Gaussian product theorem** into ONE combined Gaussian
//! centred at
//!
//! ```text
//! P = (alpha_p·A + alpha_q·(B+L)) / (alpha_p+alpha_q),  eta = alpha_p+alpha_q
//! K = exp(-alpha_p·alpha_q/eta · |A-(B+L)|²)
//! ```
//!
//! and re-expands the Cartesian product `(x-Ax)^a (x-Bx-Lx)^b` in powers of
//! `(x-Px)` via the standard binomial shift
//! (`f_k = Σ_{m+n=k} C(a,m)C(b,n) (Px-Ax)^{a-m} (Px-Bx-Lx)^{b-n}`, applied
//! separably per Cartesian axis — see `crate::multigrid::pair::binom_shift`).
//! Every `(pair, image, cart-component-pair, monomial term)` becomes one
//! **slot**: a Cartesian power triple `(k1,k2,k3)`, a scalar geometric
//! coefficient (`K · ctr_p · ctr_q · fx[k1]·fy[k2]·fz[k3]`, with NO density
//! or weight folded in), and the `(ci, cj)` decontracted-AO indices it
//! belongs to. This is the exact fused-product analogue of v1's
//! per-primitive monomial evaluation — the kernel below is the SAME
//! elementwise Gaussian-evaluation primitive
//! (`(r-P)^{k1,k2,k3}·exp(-eta|r-P|²)`) v1's `collocate_kernel` already
//! uses, generalised so the scalar coefficient varies PER SLOT rather than
//! per pshell (v1's shared pshells never needed that: an atom-centred
//! primitive's periodic images all carry the identical coefficient, but a
//! fused pair's images do not — each image's `(P-A, P-(B+L))` displacement,
//! and hence its shift coefficients, is image-specific).
//!
//! `grid_collocate_drv` (density forward) and `grid_integrate_drv` (its
//! adjoint) both reduce, on the host, to a WEIGHTED SUM over this kernel's
//! per-slot grid values — `crate::multigrid::pair::{pairlevel_rho,
//! pairlevel_pass2}` do the weighting, exactly mirroring the v1 host/kernel
//! split (`crates/pyscf-pbc-dft/src/numint.rs`'s `eval_rho_one`/
//! `vxc_mat_one` idiom). See that module's doc for the adjoint-identity test
//! this reformulation is gated by — it needs no upstream oracle.
//!
//! # `F: Float` and the one documented exception
//!
//! Same exception `multigrid_collocate.rs` and `pbc/ft_aopair.rs` already
//! document: the only transcendental this kernel calls is
//! `exp(-eta·r²)` (`cube_math::double::exp`), which has no generic-`F` seam.
//! Every other operation (integer powers via repeated multiplication) is
//! closed under `Float` already.
//!
//! # No cube barriers, no fixed-width cubes
//!
//! Launched via [`pyscf_algebra::launch::launch_1d`], sized from the device,
//! same as every sibling PBC kernel — see `multigrid_collocate.rs`'s module
//! doc for the CPU-runtime rationale (AGENTS.md §3 / RULE 5, Task 0).

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

/// Flat device tables for one [`collocate_pairs`] launch — one grid level's
/// worth of fused-pair "instances" (one primitive pair at one periodic
/// image), each contributing one or more Cartesian-monomial slots.
#[derive(Debug, Clone, Default)]
pub struct PairSlotTable {
    /// `(ngrids, 3)` real-space grid coordinates, row-major, Bohr.
    pub coords: Vec<f64>,
    /// Per slot, 3 entries: the monomial powers `(k1,k2,k3)` relative to the
    /// instance's own combined centre `P`.
    pub slot_pow: Vec<u32>,
    /// Per slot: the GEOMETRIC coefficient (`K · ctr_p · ctr_q ·
    /// fx[k1]·fy[k2]·fz[k3]`) — no density-matrix or grid-weight factor.
    pub slot_coef: Vec<f64>,
    /// Per slot: which instance it belongs to.
    pub slot_instance: Vec<u32>,
    /// Per instance: the combined exponent `eta = alpha_p + alpha_q`.
    pub instance_alpha: Vec<f64>,
    /// Per instance, 3 entries: the combined centre `P`.
    pub instance_center: Vec<f64>,
}

/// `i = slot*ngrids + g`. `out[slot*ngrids+g] = slot_coef[slot] · (r-P)^pow ·
/// exp(-eta·|r-P|²)` — ONE image per instance, unlike v1's kernel (no
/// per-instance record list: each periodic image is already its own
/// instance, because its shift coefficients differ from every other image's,
/// see the module doc).
#[cube(launch_unchecked)]
fn collocate_pair_kernel(
    coords: &Array<f64>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    slot_instance: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    nslots: usize,
    ngrids: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < nslots * ngrids {
        let slot = idx / ngrids;
        let g = idx % ngrids;

        let x = coords[g * 3];
        let y = coords[g * 3 + 1];
        let z = coords[g * 3 + 2];

        let ix = slot_pow[slot * 3];
        let iy = slot_pow[slot * 3 + 1];
        let iz = slot_pow[slot * 3 + 2];

        let inst = slot_instance[slot] as usize;
        let eta = instance_alpha[inst];
        let coef = slot_coef[slot];

        let dx = x - instance_center[inst * 3];
        let dy = y - instance_center[inst * 3 + 1];
        let dz = z - instance_center[inst * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;

        let mut poly = 1.0;
        let mut i = 0u32;
        while i < ix {
            poly *= dx;
            i += 1;
        }
        i = 0u32;
        while i < iy {
            poly *= dy;
            i += 1;
        }
        i = 0u32;
        while i < iz {
            poly *= dz;
            i += 1;
        }

        let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);
        out[idx] = coef * poly * e;
    }
}

fn work_per_thread(_t: &PairSlotTable) -> usize {
    // One slot·grid-point evaluation costs a handful of multiplies plus one
    // `exp` — the same per-lane cost `multigrid_collocate.rs` uses; there is
    // no per-instance record loop here (every image is its own instance).
    50
}

fn launch_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    h: &[Handle],
    out: &Handle,
    t: &PairSlotTable,
    nslots: usize,
    ngrids: usize,
) {
    let (count, dim) = launch_1d(client, nslots * ngrids, work_per_thread(t));
    unsafe {
        collocate_pair_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), t.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_coef.len()),
            ArrayArg::from_raw_parts(h[3].clone(), t.slot_instance.len()),
            ArrayArg::from_raw_parts(h[4].clone(), t.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.instance_center.len()),
            ArrayArg::from_raw_parts(out.clone(), nslots * ngrids),
            nslots,
            ngrids,
        );
    }
}

fn upload_u32<R: Runtime>(client: &ComputeClient<R>, data: &[u32]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(data))
}

fn launch<R: Runtime>(t: &PairSlotTable, client: &ComputeClient<R>) -> Vec<f64> {
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    let n_out = ngrids * nslots;

    let zeros = vec![0.0f64; n_out];
    let out_h = upload::<R, f64>(client, &zeros);
    drop(zeros);

    let h = vec![
        upload::<R, f64>(client, &t.coords),
        upload_u32::<R>(client, &t.slot_pow),
        upload::<R, f64>(client, &t.slot_coef),
        upload_u32::<R>(client, &t.slot_instance),
        upload::<R, f64>(client, &t.instance_alpha),
        upload::<R, f64>(client, &t.instance_center),
    ];
    launch_on_handles::<R>(client, &h, &out_h, t, nslots, ngrids);
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

/// `grid_collocate_drv` / `grid_integrate_drv`'s shared elementwise
/// primitive: evaluate every slot's fused-pair Gaussian monomial at every
/// grid point. The caller (`crate::multigrid::pair`) does the (dm- or
/// weight-)weighted contraction on the host, exactly as v1's
/// `multigrid_collocate::collocate` + `crate::multigrid::colloc` split does.
///
/// Returns a dense row-major `(n_slots, ngrids)` buffer.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if the per-slot / per-instance tables
/// disagree in length. An empty table or an empty grid returns an empty
/// vector without launching.
pub fn collocate_pairs(
    client: &AlgebraClient,
    t: &PairSlotTable,
) -> Result<Vec<f64>, AlgebraError> {
    let shape = |what: &str, actual: String| AlgebraError::ShapeMismatch {
        expected: what.to_string(),
        actual,
    };
    if !t.coords.len().is_multiple_of(3) {
        return Err(shape(
            "coords length a multiple of 3",
            format!("{}", t.coords.len()),
        ));
    }
    if !t.slot_pow.len().is_multiple_of(3) {
        return Err(shape(
            "slot_pow length a multiple of 3",
            format!("{}", t.slot_pow.len()),
        ));
    }
    let nslots = t.slot_pow.len() / 3;
    if t.slot_coef.len() != nslots || t.slot_instance.len() != nslots {
        return Err(shape(
            "slot_coef.len() == slot_instance.len() == nslots",
            format!(
                "nslots {nslots}, slot_coef {}, slot_instance {}",
                t.slot_coef.len(),
                t.slot_instance.len()
            ),
        ));
    }
    if !t.instance_center.len().is_multiple_of(3) {
        return Err(shape(
            "instance_center length a multiple of 3",
            format!("{}", t.instance_center.len()),
        ));
    }
    let ninst = t.instance_center.len() / 3;
    if t.instance_alpha.len() != ninst {
        return Err(shape(
            "instance_alpha.len() == ninstances",
            format!("ninstances {ninst}, instance_alpha {}", t.instance_alpha.len()),
        ));
    }
    if let Some(bad) = t.slot_instance.iter().find(|&&p| (p as usize) >= ninst) {
        return Err(shape(
            &format!("every slot_instance < ninstances = {ninst}"),
            format!("{bad}"),
        ));
    }
    let ngrids = t.coords.len() / 3;
    if nslots == 0 || ngrids == 0 {
        return Ok(Vec::new());
    }
    let out = dispatch_backend!(client, c, Rt, launch::<Rt>(t, c));
    Ok(out)
}
