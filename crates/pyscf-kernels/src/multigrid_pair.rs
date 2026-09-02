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

/// The same evaluation as [`collocate_pair_kernel`], one lane per
/// `(instance, grid point)` instead of per `(slot, grid point)`: the
/// exponential — the only transcendental, and the dominant cost — is
/// computed ONCE per lane and shared by every slot of that instance
/// (`inst_slot0[inst]..inst_slot0[inst+1]`, which requires slots grouped
/// by instance in ascending order; [`collocate_pairs`] checks that and
/// falls back to the per-slot kernel otherwise). Same operation order per
/// output element (`coef · poly · e`, `poly` by repeated multiplication),
/// so the two kernels are bit-identical — gated by
/// `crates/pyscf-kernels/tests/multigrid_pair.rs`.
#[cube(launch_unchecked)]
fn collocate_pair_grouped_kernel(
    coords: &Array<f64>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    inst_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
    ngrids: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < ninst * ngrids {
        let inst = idx / ngrids;
        let g = idx % ngrids;

        let x = coords[g * 3];
        let y = coords[g * 3 + 1];
        let z = coords[g * 3 + 2];

        let eta = instance_alpha[inst];
        let dx = x - instance_center[inst * 3];
        let dy = y - instance_center[inst * 3 + 1];
        let dz = z - instance_center[inst * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);

        let s0 = inst_slot0[inst] as usize;
        let s1 = inst_slot0[inst + 1] as usize;
        for slot in s0..s1 {
            let ix = slot_pow[slot * 3];
            let iy = slot_pow[slot * 3 + 1];
            let iz = slot_pow[slot * 3 + 2];
            let coef = slot_coef[slot];

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
            out[slot * ngrids + g] = coef * poly * e;
        }
    }
}

fn launch_grouped_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    h: &[Handle],
    out: &Handle,
    t: &PairSlotTable,
    inst_slot0: &[u32],
    nslots: usize,
    ngrids: usize,
) {
    let ninst = inst_slot0.len() - 1;
    // Per lane: one `exp` plus `nslots/ninst` polynomial slots.
    let per_lane = 50 + 10 * nslots.div_ceil(ninst.max(1));
    let (count, dim) = launch_1d(client, ninst * ngrids, per_lane);
    unsafe {
        collocate_pair_grouped_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), t.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_coef.len()),
            ArrayArg::from_raw_parts(h[3].clone(), inst_slot0.len()),
            ArrayArg::from_raw_parts(h[4].clone(), t.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.instance_center.len()),
            ArrayArg::from_raw_parts(out.clone(), nslots * ngrids),
            ninst,
            ngrids,
        );
    }
}

fn launch_grouped<R: Runtime>(
    t: &PairSlotTable,
    inst_slot0: &[u32],
    client: &ComputeClient<R>,
) -> Vec<f64> {
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
        upload_u32::<R>(client, inst_slot0),
        upload::<R, f64>(client, &t.instance_alpha),
        upload::<R, f64>(client, &t.instance_center),
    ];
    launch_grouped_on_handles::<R>(client, &h, &out_h, t, inst_slot0, nslots, ngrids);
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

/// `inst_slot0` (length `ninst + 1`) when `slot_instance` is grouped —
/// non-decreasing — so instance `i` owns slots `inst_slot0[i]..
/// inst_slot0[i+1]` (possibly none); `None` otherwise.
fn grouped_slot_ranges(slot_instance: &[u32], ninst: usize) -> Option<Vec<u32>> {
    if slot_instance.windows(2).any(|w| w[1] < w[0]) {
        return None;
    }
    let mut inst_slot0 = vec![0u32; ninst + 1];
    let mut s = 0usize;
    for (i, next) in inst_slot0.iter_mut().enumerate().skip(1) {
        while s < slot_instance.len() && (slot_instance[s] as usize) < i {
            s += 1;
        }
        *next = s as u32;
    }
    Some(inst_slot0)
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
    validate(t)?;
    let nslots = t.slot_pow.len() / 3;
    let ninst = t.instance_center.len() / 3;
    let ngrids = t.coords.len() / 3;
    if nslots == 0 || ngrids == 0 {
        return Ok(Vec::new());
    }
    let out = match grouped_slot_ranges(&t.slot_instance, ninst) {
        Some(inst_slot0) => {
            dispatch_backend!(client, c, Rt, launch_grouped::<Rt>(t, &inst_slot0, c))
        }
        None => dispatch_backend!(client, c, Rt, launch::<Rt>(t, c)),
    };
    Ok(out)
}

/// The shape checks both entry points share.
fn validate(t: &PairSlotTable) -> Result<(), AlgebraError> {
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
            format!(
                "ninstances {ninst}, instance_alpha {}",
                t.instance_alpha.len()
            ),
        ));
    }
    if let Some(bad) = t.slot_instance.iter().find(|&&p| (p as usize) >= ninst) {
        return Err(shape(
            &format!("every slot_instance < ninstances = {ninst}"),
            format!("{bad}"),
        ));
    }
    Ok(())
}

/// [`collocate_pairs`] forced onto the per-slot kernel (one `exp` per
/// `(slot, grid point)`), regardless of grouping — the reference the
/// grouped kernel is gated against. Same validation as [`collocate_pairs`].
///
/// # Errors
/// As [`collocate_pairs`].
pub fn collocate_pairs_per_slot(
    client: &AlgebraClient,
    t: &PairSlotTable,
) -> Result<Vec<f64>, AlgebraError> {
    validate(t)?;
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    if nslots == 0 || ngrids == 0 {
        return Ok(Vec::new());
    }
    let out = dispatch_backend!(client, c, Rt, launch::<Rt>(t, c));
    Ok(out)
}

/// `grid_collocate_drv` with the slot reduction INSIDE the kernel: one lane
/// per grid point, `out[g] = Σ_inst Σ_{slot∈inst} coef·(r-P)^pow·exp(-eta
/// |r-P|²)`, instances and slots visited in table order. Requires slots
/// grouped by instance (`inst_slot0`). The per-lane sum is strictly
/// sequential in a fixed order, so the result is bit-identical under any
/// launch geometry or host thread count (D-PBC-17) — it is not
/// `oracle_sum`'s pairwise tree, and `pyscf-pbc-dft`'s v2 driver documents
/// that trade (its previous host-side reduction needed every `(slot ×
/// point)` value materialised, ~100 GiB per level on the Gate-E cells).
#[cube(launch_unchecked)]
fn collocate_pairs_rho_kernel(
    coords: &Array<f64>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    inst_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
    ngrids: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ngrids {
        let x = coords[g * 3];
        let y = coords[g * 3 + 1];
        let z = coords[g * 3 + 2];
        let mut acc = 0.0;
        for inst in 0..ninst {
            let eta = instance_alpha[inst];
            let dx = x - instance_center[inst * 3];
            let dy = y - instance_center[inst * 3 + 1];
            let dz = z - instance_center[inst * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);
            let s0 = inst_slot0[inst] as usize;
            let s1 = inst_slot0[inst + 1] as usize;
            for slot in s0..s1 {
                let ix = slot_pow[slot * 3];
                let iy = slot_pow[slot * 3 + 1];
                let iz = slot_pow[slot * 3 + 2];
                let coef = slot_coef[slot];
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
                acc += coef * poly * e;
            }
        }
        out[g] = acc;
    }
}

/// `grid_integrate_drv` with the grid reduction INSIDE the kernel: one
/// lane per instance, `out[slot] = Σ_g weight[g]·coef·(r_g-P)^pow·exp(-eta
/// |r_g-P|²)` for every slot of that instance, grid points visited in
/// table order (one `exp` per `(instance, point)`, shared by the
/// instance's slots). Same determinism argument as
/// [`collocate_pairs_rho_kernel`]; each slot is written by exactly one
/// lane.
#[cube(launch_unchecked)]
fn collocate_pairs_integrate_kernel(
    coords: &Array<f64>,
    weight: &Array<f64>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    inst_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
    ngrids: usize,
) {
    let inst = ABSOLUTE_POS;
    if inst < ninst {
        let eta = instance_alpha[inst];
        let cx = instance_center[inst * 3];
        let cy = instance_center[inst * 3 + 1];
        let cz = instance_center[inst * 3 + 2];
        let s0 = inst_slot0[inst] as usize;
        let s1 = inst_slot0[inst + 1] as usize;
        for g in 0..ngrids {
            let dx = coords[g * 3] - cx;
            let dy = coords[g * 3 + 1] - cy;
            let dz = coords[g * 3 + 2] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);
            let we = weight[g] * e;
            for slot in s0..s1 {
                let ix = slot_pow[slot * 3];
                let iy = slot_pow[slot * 3 + 1];
                let iz = slot_pow[slot * 3 + 2];
                let coef = slot_coef[slot];
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
                out[slot] = out[slot] + coef * poly * we;
            }
        }
    }
}

fn launch_rho<R: Runtime>(
    t: &PairSlotTable,
    inst_slot0: &[u32],
    client: &ComputeClient<R>,
) -> Vec<f64> {
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    let ninst = inst_slot0.len() - 1;
    let zeros = vec![0.0f64; ngrids];
    let out_h = upload::<R, f64>(client, &zeros);
    let h = [
        upload::<R, f64>(client, &t.coords),
        upload_u32::<R>(client, &t.slot_pow),
        upload::<R, f64>(client, &t.slot_coef),
        upload_u32::<R>(client, inst_slot0),
        upload::<R, f64>(client, &t.instance_alpha),
        upload::<R, f64>(client, &t.instance_center),
    ];
    // Per lane: every instance (one `exp` each) and every slot.
    let per_lane = 50 * ninst.max(1) + 10 * nslots;
    let (count, dim) = launch_1d(client, ngrids, per_lane);
    unsafe {
        collocate_pairs_rho_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), t.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_coef.len()),
            ArrayArg::from_raw_parts(h[3].clone(), inst_slot0.len()),
            ArrayArg::from_raw_parts(h[4].clone(), t.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.instance_center.len()),
            ArrayArg::from_raw_parts(out_h.clone(), ngrids),
            ninst,
            ngrids,
        );
    }
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

fn launch_integrate<R: Runtime>(
    t: &PairSlotTable,
    weight: &[f64],
    inst_slot0: &[u32],
    client: &ComputeClient<R>,
) -> Vec<f64> {
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    let ninst = inst_slot0.len() - 1;
    let zeros = vec![0.0f64; nslots];
    let out_h = upload::<R, f64>(client, &zeros);
    let h = [
        upload::<R, f64>(client, &t.coords),
        upload::<R, f64>(client, weight),
        upload_u32::<R>(client, &t.slot_pow),
        upload::<R, f64>(client, &t.slot_coef),
        upload_u32::<R>(client, inst_slot0),
        upload::<R, f64>(client, &t.instance_alpha),
        upload::<R, f64>(client, &t.instance_center),
    ];
    // Per lane: every grid point (one `exp` each) times the instance's slots.
    let per_lane = ngrids * (50 + 10 * nslots.div_ceil(ninst.max(1)));
    let (count, dim) = launch_1d(client, ninst, per_lane);
    unsafe {
        collocate_pairs_integrate_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), t.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), weight.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[3].clone(), t.slot_coef.len()),
            ArrayArg::from_raw_parts(h[4].clone(), inst_slot0.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[6].clone(), t.instance_center.len()),
            ArrayArg::from_raw_parts(out_h.clone(), nslots),
            ninst,
            ngrids,
        );
    }
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

// ---------------------------------------------------------------------------
// M-03 — one launch per level per direction, instead of one per spatial block.
//
// `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md` §2.3.3. The v2
// host driver streams each level's mesh in ~5^3 spatial blocks and issues one
// launch per block per direction: at `mesh = 25^3` that is 125 launches, each
// uploading seven buffers and reading one back, for every level of every
// density evaluation. 17-12 already attributed the first streamed version's
// 130 s -> 7-9 s to per-launch buffer copies and left "batched launches" as
// its carry-over #3.
//
// This is `11_launch_overhead_and_transfers.md` §5 ("Collapse Per-Item
// Launches into One") applied verbatim, with §3's batched read-back: the
// blocks' tables are CONCATENATED and one launch covers all of them, the grid
// selecting the block.
//
// # How the block is selected, and why not a search
//
// §5's worked example pairs a single launch with an offset table
// (`slot_off[f]..slot_off[f+1]`). The same idea here needs the INVERSE map as
// well — a lane knows its own index and must find its block — and the manual's
// own conditionals guidance (`Cubecl_conditionals.md`, "avoid if expressions")
// argues against an in-kernel binary search over the offset table. So the
// inverse map is materialised host-side instead, as one `u32` per lane
// (`point_block`, `inst_block`): 62 KiB at `25^3`, against a per-lane search
// with a data-dependent trip count and the branch divergence
// `plane_alignment.md` warns about. Precomputing it is both simpler and
// strictly less work per lane.
//
// # Bit-parity: EXACT against the per-block route
//
// Every lane runs the identical inner loops over the identical slot list in
// the identical order; only the launch geometry changes. The per-point sum is
// still sequential in table order, each output is still written by exactly one
// lane, and the host still folds the integrate results in block-major,
// within-block-table order. Asserted at `to_bits()` in
// `tests/multigrid_pair.rs`, not argued.
// ---------------------------------------------------------------------------

/// Every spatial block of ONE grid level, concatenated into a single set of
/// device tables — the M-03 batch.
///
/// Everything here except [`Self::slot_coef`] is pure geometry and can be
/// built once per cell (which is what `pyscf-pbc-dft`'s `PairLevelTable`
/// does); `slot_coef` carries the density (forward) or ones (reverse) and
/// changes per call.
#[derive(Debug, Clone, Default)]
pub struct PairSlotBatch {
    /// `(npoints, 3)` grid coordinates, concatenated in block order.
    pub coords: Vec<f64>,
    /// Per concatenated point: which block owns it. The inverse of
    /// [`Self::block_point0`], materialised so no lane has to search.
    pub point_block: Vec<u32>,
    /// `block_point0[b]..block_point0[b+1]` — block `b`'s points.
    /// `nblocks + 1` entries.
    pub block_point0: Vec<u32>,
    /// `block_inst0[b]..block_inst0[b+1]` — block `b`'s instances.
    /// `nblocks + 1` entries.
    pub block_inst0: Vec<u32>,
    /// Per concatenated instance: which block owns it.
    pub inst_block: Vec<u32>,
    /// Per concatenated instance: `eta = alpha_p + alpha_q`.
    pub instance_alpha: Vec<f64>,
    /// Per concatenated instance, 3 entries: the combined centre `P`.
    pub instance_center: Vec<f64>,
    /// `inst_slot0[i]..inst_slot0[i+1]` — instance `i`'s slots.
    /// `n_instances + 1` entries.
    pub inst_slot0: Vec<u32>,
    /// Per concatenated slot, 3 entries: the monomial powers.
    pub slot_pow: Vec<u32>,
    /// Per concatenated slot: the scalar coefficient. **The only field that
    /// varies per call.**
    pub slot_coef: Vec<f64>,
}

impl PairSlotBatch {
    /// Concatenated grid points.
    pub fn npoints(&self) -> usize {
        self.point_block.len()
    }
    /// Concatenated instances.
    pub fn ninstances(&self) -> usize {
        self.instance_alpha.len()
    }
    /// Concatenated slots.
    pub fn nslots(&self) -> usize {
        self.slot_coef.len()
    }
    /// Blocks.
    pub fn nblocks(&self) -> usize {
        self.block_point0.len().saturating_sub(1)
    }
}

/// [`collocate_pairs_rho_kernel`], batched over every block of a level —
/// M-03. One lane per concatenated grid point; the lane's block selects which
/// instances it sums, in the same order the per-block launch used.
#[cube(launch_unchecked)]
fn collocate_pairs_rho_batched_kernel(
    coords: &Array<f64>,
    point_block: &Array<u32>,
    block_inst0: &Array<u32>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    inst_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    npoints: usize,
) {
    let p = ABSOLUTE_POS;
    if p < npoints {
        let x = coords[p * 3];
        let y = coords[p * 3 + 1];
        let z = coords[p * 3 + 2];
        let b = point_block[p] as usize;
        let i0 = block_inst0[b] as usize;
        let i1 = block_inst0[b + 1] as usize;
        let mut acc = 0.0;
        for inst in i0..i1 {
            let eta = instance_alpha[inst];
            let dx = x - instance_center[inst * 3];
            let dy = y - instance_center[inst * 3 + 1];
            let dz = z - instance_center[inst * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);
            let s0 = inst_slot0[inst] as usize;
            let s1 = inst_slot0[inst + 1] as usize;
            for slot in s0..s1 {
                let ix = slot_pow[slot * 3];
                let iy = slot_pow[slot * 3 + 1];
                let iz = slot_pow[slot * 3 + 2];
                let coef = slot_coef[slot];
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
                acc += coef * poly * e;
            }
        }
        out[p] = acc;
    }
}

/// [`collocate_pairs_integrate_kernel`], batched over every block of a level —
/// M-03. One lane per concatenated instance; the lane's block selects which
/// grid points it integrates over, in the same order the per-block launch
/// used.
#[cube(launch_unchecked)]
fn collocate_pairs_integrate_batched_kernel(
    coords: &Array<f64>,
    weight: &Array<f64>,
    inst_block: &Array<u32>,
    block_point0: &Array<u32>,
    slot_pow: &Array<u32>,
    slot_coef: &Array<f64>,
    inst_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
) {
    let inst = ABSOLUTE_POS;
    if inst < ninst {
        let eta = instance_alpha[inst];
        let cx = instance_center[inst * 3];
        let cy = instance_center[inst * 3 + 1];
        let cz = instance_center[inst * 3 + 2];
        let s0 = inst_slot0[inst] as usize;
        let s1 = inst_slot0[inst + 1] as usize;
        let b = inst_block[inst] as usize;
        let g0 = block_point0[b] as usize;
        let g1 = block_point0[b + 1] as usize;
        for g in g0..g1 {
            let dx = coords[g * 3] - cx;
            let dy = coords[g * 3 + 1] - cy;
            let dz = coords[g * 3 + 2] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let e = cube_math::double::exp::exp(0.0 - eta * r2, cube_math::MathConfig::EXACT);
            let we = weight[g] * e;
            for slot in s0..s1 {
                let ix = slot_pow[slot * 3];
                let iy = slot_pow[slot * 3 + 1];
                let iz = slot_pow[slot * 3 + 2];
                let coef = slot_coef[slot];
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
                out[slot] = out[slot] + coef * poly * we;
            }
        }
    }
}

fn launch_rho_batched<R: Runtime>(b: &PairSlotBatch, client: &ComputeClient<R>) -> Vec<f64> {
    let npoints = b.npoints();
    let out_h = upload::<R, f64>(client, &vec![0.0f64; npoints]);
    let h = [
        upload::<R, f64>(client, &b.coords),
        upload::<R, f64>(client, &b.slot_coef),
        upload::<R, f64>(client, &b.instance_alpha),
        upload::<R, f64>(client, &b.instance_center),
    ];
    let hu = [
        upload_u32::<R>(client, &b.point_block),
        upload_u32::<R>(client, &b.block_inst0),
        upload_u32::<R>(client, &b.slot_pow),
        upload_u32::<R>(client, &b.inst_slot0),
    ];
    // Per lane: its block's instances (one `exp` each) and their slots. The
    // average is the total divided by the block count, which is what
    // `launch_1d` needs to size the CPU cube.
    let nblocks = b.nblocks().max(1);
    let per_lane = 50 * (b.ninstances() / nblocks).max(1) + 10 * (b.nslots() / nblocks).max(1);
    let (count, dim) = launch_1d(client, npoints, per_lane);
    unsafe {
        collocate_pairs_rho_batched_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), b.coords.len()),
            ArrayArg::from_raw_parts(hu[0].clone(), b.point_block.len()),
            ArrayArg::from_raw_parts(hu[1].clone(), b.block_inst0.len()),
            ArrayArg::from_raw_parts(hu[2].clone(), b.slot_pow.len()),
            ArrayArg::from_raw_parts(h[1].clone(), b.slot_coef.len()),
            ArrayArg::from_raw_parts(hu[3].clone(), b.inst_slot0.len()),
            ArrayArg::from_raw_parts(h[2].clone(), b.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[3].clone(), b.instance_center.len()),
            ArrayArg::from_raw_parts(out_h.clone(), npoints),
            npoints,
        );
    }
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

fn launch_integrate_batched<R: Runtime>(
    b: &PairSlotBatch,
    weight: &[f64],
    client: &ComputeClient<R>,
) -> Vec<f64> {
    let ninst = b.ninstances();
    let nslots = b.nslots();
    let out_h = upload::<R, f64>(client, &vec![0.0f64; nslots]);
    let h = [
        upload::<R, f64>(client, &b.coords),
        upload::<R, f64>(client, weight),
        upload::<R, f64>(client, &b.slot_coef),
        upload::<R, f64>(client, &b.instance_alpha),
        upload::<R, f64>(client, &b.instance_center),
    ];
    let hu = [
        upload_u32::<R>(client, &b.inst_block),
        upload_u32::<R>(client, &b.block_point0),
        upload_u32::<R>(client, &b.slot_pow),
        upload_u32::<R>(client, &b.inst_slot0),
    ];
    let nblocks = b.nblocks().max(1);
    let per_lane = 50 * (b.npoints() / nblocks).max(1);
    let (count, dim) = launch_1d(client, ninst, per_lane);
    unsafe {
        collocate_pairs_integrate_batched_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), b.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), weight.len()),
            ArrayArg::from_raw_parts(hu[0].clone(), b.inst_block.len()),
            ArrayArg::from_raw_parts(hu[1].clone(), b.block_point0.len()),
            ArrayArg::from_raw_parts(hu[2].clone(), b.slot_pow.len()),
            ArrayArg::from_raw_parts(h[2].clone(), b.slot_coef.len()),
            ArrayArg::from_raw_parts(hu[3].clone(), b.inst_slot0.len()),
            ArrayArg::from_raw_parts(h[3].clone(), b.instance_alpha.len()),
            ArrayArg::from_raw_parts(h[4].clone(), b.instance_center.len()),
            ArrayArg::from_raw_parts(out_h.clone(), nslots),
            ninst,
        );
    }
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

/// The M-03 batched forward direction: `rho` at every concatenated grid point
/// of one level, in ONE launch.
///
/// Returns one value per entry of [`PairSlotBatch::coords`], in the batch's
/// own (block-major) order; the caller scatters them back to mesh order.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] on an inconsistent batch. An empty batch
/// returns an empty vector.
pub fn collocate_pairs_rho_batched(
    client: &AlgebraClient,
    b: &PairSlotBatch,
) -> Result<Vec<f64>, AlgebraError> {
    validate_batch(b)?;
    if b.npoints() == 0 {
        return Ok(Vec::new());
    }
    if b.nslots() == 0 {
        return Ok(vec![0.0; b.npoints()]);
    }
    Ok(dispatch_backend!(
        client,
        c,
        Rt,
        launch_rho_batched::<Rt>(b, c)
    ))
}

/// The M-03 batched reverse direction: every concatenated slot's weighted grid
/// integral for one level, in ONE launch.
///
/// `weight` is indexed by CONCATENATED point, i.e. it must be permuted into
/// the batch's block-major order by the caller — the same order
/// [`PairSlotBatch::coords`] is in.
///
/// # Errors
/// As [`collocate_pairs_rho_batched`], plus a length mismatch on `weight`.
pub fn collocate_pairs_integrate_batched(
    client: &AlgebraClient,
    b: &PairSlotBatch,
    weight: &[f64],
) -> Result<Vec<f64>, AlgebraError> {
    validate_batch(b)?;
    if weight.len() != b.npoints() {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("weight.len() == npoints = {}", b.npoints()),
            actual: format!("{}", weight.len()),
        });
    }
    if b.nslots() == 0 {
        return Ok(Vec::new());
    }
    if b.npoints() == 0 {
        return Ok(vec![0.0; b.nslots()]);
    }
    Ok(dispatch_backend!(
        client,
        c,
        Rt,
        launch_integrate_batched::<Rt>(b, weight, c)
    ))
}

/// Shape checks for a [`PairSlotBatch`] — the batched analogue of
/// [`validate`], and the same posture: an inconsistent table is a caller bug
/// and is refused rather than indexed past.
fn validate_batch(b: &PairSlotBatch) -> Result<(), AlgebraError> {
    let shape = |expected: String, actual: String| AlgebraError::ShapeMismatch { expected, actual };
    if b.coords.len() != b.point_block.len() * 3 {
        return Err(shape(
            format!("coords.len() == 3 * npoints = {}", b.point_block.len() * 3),
            format!("{}", b.coords.len()),
        ));
    }
    if b.block_point0.len() != b.block_inst0.len() {
        return Err(shape(
            "block_point0 and block_inst0 to have the same length (nblocks + 1)".to_string(),
            format!("{} vs {}", b.block_point0.len(), b.block_inst0.len()),
        ));
    }
    if b.instance_center.len() != b.instance_alpha.len() * 3 {
        return Err(shape(
            format!(
                "instance_center.len() == 3 * ninstances = {}",
                b.instance_alpha.len() * 3
            ),
            format!("{}", b.instance_center.len()),
        ));
    }
    if b.inst_block.len() != b.instance_alpha.len() {
        return Err(shape(
            format!("inst_block.len() == ninstances = {}", b.instance_alpha.len()),
            format!("{}", b.inst_block.len()),
        ));
    }
    if b.inst_slot0.len() != b.instance_alpha.len() + 1 {
        return Err(shape(
            format!(
                "inst_slot0.len() == ninstances + 1 = {}",
                b.instance_alpha.len() + 1
            ),
            format!("{}", b.inst_slot0.len()),
        ));
    }
    if b.slot_pow.len() != b.slot_coef.len() * 3 {
        return Err(shape(
            format!("slot_pow.len() == 3 * nslots = {}", b.slot_coef.len() * 3),
            format!("{}", b.slot_pow.len()),
        ));
    }
    Ok(())
}

fn grouped_or_err(t: &PairSlotTable) -> Result<Vec<u32>, AlgebraError> {
    let ninst = t.instance_center.len() / 3;
    grouped_slot_ranges(&t.slot_instance, ninst).ok_or_else(|| AlgebraError::ShapeMismatch {
        expected: "slot_instance grouped by instance (non-decreasing)".to_string(),
        actual: "ungrouped slot_instance".to_string(),
    })
}

/// `grid_collocate_drv`'s reduction fused into the kernel — `rho[g] =
/// Σ_slot coef_slot · slot(r_g)`, one value per grid point, no `(slot ×
/// point)` buffer. See [`collocate_pairs_rho_kernel`] for the ordering
/// contract. Requires an instance-grouped table.
///
/// # Errors
/// As [`collocate_pairs`], plus [`AlgebraError::ShapeMismatch`] for an
/// ungrouped table. An empty table or grid returns zeros / an empty vector.
pub fn collocate_pairs_rho(
    client: &AlgebraClient,
    t: &PairSlotTable,
) -> Result<Vec<f64>, AlgebraError> {
    validate(t)?;
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    if ngrids == 0 {
        return Ok(Vec::new());
    }
    if nslots == 0 {
        return Ok(vec![0.0; ngrids]);
    }
    let inst_slot0 = grouped_or_err(t)?;
    let out = dispatch_backend!(client, c, Rt, launch_rho::<Rt>(t, &inst_slot0, c));
    Ok(out)
}

/// `grid_integrate_drv`'s reduction fused into the kernel — `I[slot] =
/// Σ_g weight[g] · coef_slot · slot(r_g)`, one value per slot, no `(slot ×
/// point)` buffer. See [`collocate_pairs_integrate_kernel`] for the
/// ordering contract. Requires an instance-grouped table and
/// `weight.len() == ngrids`.
///
/// # Errors
/// As [`collocate_pairs_rho`], plus a length mismatch on `weight`.
pub fn collocate_pairs_integrate(
    client: &AlgebraClient,
    t: &PairSlotTable,
    weight: &[f64],
) -> Result<Vec<f64>, AlgebraError> {
    validate(t)?;
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    if weight.len() != ngrids {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("weight.len() == ngrids = {ngrids}"),
            actual: format!("{}", weight.len()),
        });
    }
    if nslots == 0 {
        return Ok(Vec::new());
    }
    if ngrids == 0 {
        return Ok(vec![0.0; nslots]);
    }
    let inst_slot0 = grouped_or_err(t)?;
    let out = dispatch_backend!(
        client,
        c,
        Rt,
        launch_integrate::<Rt>(t, weight, &inst_slot0, c)
    );
    Ok(out)
}
