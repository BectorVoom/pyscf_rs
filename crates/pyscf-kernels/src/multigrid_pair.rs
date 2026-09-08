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
use cubecl::stream_id::StreamId;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};
use pyscf_runtime::BackendKind;

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
    /// Per instance: the square of its own cutoff radius — M-21, the
    /// per-point screen. A grid point farther than this from `P` receives
    /// nothing from the instance (the fused Gaussian is below the screening
    /// threshold there by construction of the radius); `f64::INFINITY`
    /// disables the screen for that instance.
    pub instance_radius2: Vec<f64>,
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
    if t.instance_alpha.len() != ninst || t.instance_radius2.len() != ninst {
        return Err(shape(
            "instance_alpha.len() == instance_radius2.len() == ninstances",
            format!(
                "ninstances {ninst}, instance_alpha {}, instance_radius2 {}",
                t.instance_alpha.len(),
                t.instance_radius2.len()
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
    instance_radius2: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
    ngrids: usize,
    #[comptime] screen: bool,
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
            // M-21: outside the instance's own radius the point gets nothing.
            let mut keep = true;
            if comptime!(screen) {
                keep = r2 <= instance_radius2[inst];
            }
            if keep {
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
    instance_radius2: &Array<f64>,
    out: &mut Array<f64>,
    ninst: usize,
    ngrids: usize,
    #[comptime] screen: bool,
) {
    let inst = ABSOLUTE_POS;
    if inst < ninst {
        let eta = instance_alpha[inst];
        let cx = instance_center[inst * 3];
        let cy = instance_center[inst * 3 + 1];
        let cz = instance_center[inst * 3 + 2];
        let rad2 = instance_radius2[inst];
        let s0 = inst_slot0[inst] as usize;
        let s1 = inst_slot0[inst + 1] as usize;
        for g in 0..ngrids {
            let dx = coords[g * 3] - cx;
            let dy = coords[g * 3 + 1] - cy;
            let dz = coords[g * 3 + 2] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let mut keep = true;
            if comptime!(screen) {
                keep = r2 <= rad2;
            }
            if keep {
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
        upload::<R, f64>(client, &t.instance_radius2),
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
            ArrayArg::from_raw_parts(h[6].clone(), t.instance_radius2.len()),
            ArrayArg::from_raw_parts(out_h.clone(), ngrids),
            ninst,
            ngrids,
            point_screen_enabled(),
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
        upload::<R, f64>(client, &t.instance_radius2),
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
            ArrayArg::from_raw_parts(h[7].clone(), t.instance_radius2.len()),
            ArrayArg::from_raw_parts(out_h.clone(), nslots),
            ninst,
            ngrids,
            point_screen_enabled(),
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
// # M-15 (session 6) — term SETS instead of a per-concatenated-slot table
//
// A kernel instance is one wrap image of one fused pair `(p, q, L)`, and
// every image of that pair carries the SAME monomial-and-term sequence
// (`build_pair_level_table` pushes `terms_here` once per image). The M-12
// layout still spent one `u32` per CONCATENATED slot (`slot_global`, 311 MB
// at `25³` level 3) to reach a per-kernel-slot power and coefficient through
// two dependent gathers. The chunk now carries the level's term sets —
// `set_off`/`set_pow` (a few thousand entries) — and one set index per
// distinct instance; the slot loop of both kernels walks `set_pow[so..so+n]`
// and the per-call `set_coef[so..so+n]` SEQUENTIALLY (`07_memory_coalescing.md`
// §4: put the inner loop on the unit stride). The values read are the same
// values, in the same order, so the arithmetic is unchanged.
//
// # M-16 (session 6) — the reverse fold on the device
//
// The reverse direction used to read every concatenated slot's integral back
// (`nslots · 8` B, 622 MB at level 3) and fold it into the level's
// kernel-slot integrals on the host. `mg_fold_kernel` now does that fold on
// the device against a level-resident `kint` (`nkslots · 8` B): one lane per
// `(distinct instance, monomial)`, walking that instance's occurrences of the
// chunk IN OCCURRENCE ORDER and adding onto the running `kint` value. Per
// kernel slot the sequence of additions — occurrences in block-major order,
// chunk after chunk — is exactly the host fold's, so `kint` is bit-identical
// and only `nkslots · 8` B ever crosses back.
//
// # M-17 (session 6) — the forward kernel over `Vector<f64, N>` points
//
// One lane evaluates N ADJACENT grid points of one block (blocks are padded
// to a multiple of [`POINT_PAD`] points, so a vector never straddles two).
// Every operation is elementwise — the instance loop, the `exp` (M-20: one
// `cube_math::exp_vec` call on the whole vector, bit-identical per element
// to the scalar `exp`), the power products and the accumulate — so each point sees the
// scalar kernel's operations in the scalar kernel's order (`06_vectorization.md`,
// `Cubecl_dynamic_vectorization.md`). Pad points are evaluated and discarded.
//
// # Bit-parity: EXACT against the per-block route
//
// Every lane runs the identical inner loops over the identical slot list in
// the identical order; only the launch geometry and where the operands live
// change. The per-point sum is still sequential in table order, each output is
// still written by exactly one lane, and the fold still visits a kernel slot's
// occurrences in block-major order. Asserted at `to_bits()` in
// `tests/multigrid_batch.rs`, not argued.
// ---------------------------------------------------------------------------

/// Points per block are padded to a multiple of this, so that every vector
/// width [`pair_line_size`] can return divides every block's point range.
pub const POINT_PAD: usize = 8;

/// Sentinel in a batch's `point_global` map for a pad point — evaluated by
/// the forward kernel, never scattered back.
pub const PAD_POINT: u32 = u32::MAX;

/// Every spatial block of ONE grid level, concatenated into a single set of
/// device tables — the M-03 batch, in the M-15 term-set layout.
///
/// Everything here is geometry, built once per cell (which is what
/// `pyscf-pbc-dft`'s `PairLevelTable` does). The per-call inputs are the
/// forward coefficients (one per SET slot, see [`Self::set_pow`]) and the
/// reverse weights (one per padded point).
#[derive(Debug, Clone, Default)]
pub struct PairSlotBatch {
    /// Grid coordinates in structure-of-arrays form, concatenated in block
    /// order, each block padded to a multiple of [`POINT_PAD`] points (the
    /// pads repeat the block's last real point).
    pub coords_x: Vec<f64>,
    pub coords_y: Vec<f64>,
    pub coords_z: Vec<f64>,
    /// Per padded point: which block owns it.
    pub point_block: Vec<u32>,
    /// `block_point0[b]..block_point0[b+1]` — block `b`'s PADDED point range.
    /// `nblocks + 1` entries.
    pub block_point0: Vec<u32>,
    /// Per block: one past its last REAL point (the reverse kernel's bound).
    pub block_point_end: Vec<u32>,
    /// `block_inst0[b]..block_inst0[b+1]` — block `b`'s instance occurrences.
    /// `nblocks + 1` entries.
    pub block_inst0: Vec<u32>,
    /// Per occurrence: which block owns it.
    pub inst_block: Vec<u32>,
    /// Per occurrence: its row in the distinct-instance tables (M-13).
    pub inst_ref: Vec<u32>,
    /// `occ_slot0[o]..occ_slot0[o+1]` — occurrence `o`'s reverse outputs.
    /// `n_occurrences + 1` entries.
    pub occ_slot0: Vec<u32>,
    /// Per DISTINCT instance of this chunk: `eta = alpha_p + alpha_q`.
    pub instance_alpha: Vec<f64>,
    /// Per DISTINCT instance, 3 entries: the combined centre `P`.
    pub instance_center: Vec<f64>,
    /// Per DISTINCT instance: the square of its cutoff radius (M-21; see
    /// [`PairSlotTable::instance_radius2`]).
    pub instance_radius2: Vec<f64>,
    /// Per DISTINCT instance: its term set (M-15).
    pub instance_set: Vec<u32>,
    /// Per DISTINCT instance: the level's first kernel slot of that instance
    /// — where its set's monomials land in `kint` (M-16).
    pub instance_kslot0: Vec<u32>,
    /// `uocc_off[u]..uocc_off[u+1]` into [`Self::uocc`] — distinct instance
    /// `u`'s occurrences, in increasing occurrence order (M-16).
    pub uocc_off: Vec<u32>,
    pub uocc: Vec<u32>,
    /// `set_off[s]..set_off[s+1]` — term set `s`'s slots (M-15).
    pub set_off: Vec<u32>,
    /// Per set slot, packed as `ix | iy << 8 | iz << 16`.
    pub set_pow: Vec<u32>,
    /// Kernel slots of the level this batch was cut from — `kint`'s length.
    pub nkslots: usize,
}

impl PairSlotBatch {
    /// Concatenated (padded) grid points.
    pub fn npoints(&self) -> usize {
        self.point_block.len()
    }
    /// Concatenated instance occurrences (one per `(block, instance)`).
    pub fn ninstances(&self) -> usize {
        self.inst_block.len()
    }
    /// Distinct instances referenced by this chunk — M-13.
    pub fn nuinstances(&self) -> usize {
        self.instance_alpha.len()
    }
    /// Concatenated reverse-output slots.
    pub fn nslots(&self) -> usize {
        self.occ_slot0.last().map_or(0, |&e| e as usize)
    }
    /// Kernel slots of the level this batch was cut from.
    pub fn nkslots(&self) -> usize {
        self.nkslots
    }
    /// Set slots — the length of a per-call forward coefficient vector.
    pub fn nsetslots(&self) -> usize {
        self.set_pow.len()
    }
    /// Blocks.
    pub fn nblocks(&self) -> usize {
        self.block_point0.len().saturating_sub(1)
    }
    /// Resident bytes of this geometry on a device (M-15 ledger).
    pub fn geometry_bytes(&self) -> u64 {
        let f = self.coords_x.len() * 3 + self.instance_alpha.len() * 5;
        let u = self.point_block.len()
            + self.block_point0.len()
            + self.block_point_end.len()
            + self.block_inst0.len()
            + self.inst_block.len()
            + self.inst_ref.len()
            + self.occ_slot0.len()
            + self.instance_set.len()
            + self.instance_kslot0.len()
            + self.uocc_off.len()
            + self.uocc.len()
            + self.set_off.len()
            + self.set_pow.len();
        (f * 8 + u * 4) as u64
    }
}

/// Which `exp` the pair kernels evaluate — an INSTRUMENT, read from
/// `PYSCF_MG_PAIR_EXP` (`exact` (default) | `fast` | `none`). Anything but
/// `exact` changes results and exists only to attribute the kernels' time
/// (the K-08 lesson: a stage claim needs a kill-switch arm, not a span).
fn exp_mode_from_env() -> u32 {
    match std::env::var("PYSCF_MG_PAIR_EXP")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "fast" => 1,
        "none" => 2,
        _ => 0,
    }
}

/// M-21: whether the pair kernels apply the per-point instance-radius
/// screen — OPT-IN, `PYSCF_MG_PAIR_POINT_SCREEN=<factor>` (see
/// `pyscf_pbc_dft::multigrid::pair::point_screen_factor`); unset or `0`
/// is off. A point farther from an instance's centre than the instance's
/// per-point radius (`instance_radius2`) contributes nothing from it. That
/// drops terms below the (tightened) threshold, so the two arms are NOT
/// bit-identical; `tests/multigrid_batch.rs` bounds the difference and
/// asserts that, within one arm, batched / streamed / every vector width
/// agree bit for bit. Off by default because it MEASURED as a loss on the
/// CPU runtime at every factor that keeps the density inside the gate
/// (session 6 §3.3: the per-`(point, instance)` test and branch cost more
/// than the skipped work), and as 2.7e-5 electrons at factor 1.
fn point_screen_enabled() -> bool {
    std::env::var("PYSCF_MG_PAIR_POINT_SCREEN").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// `exp(x)` under [`exp_mode_from_env`]'s three arms.
#[cube]
fn mg_exp(x: f64, #[comptime] mode: u32) -> f64 {
    if comptime!(mode == 1) {
        cube_math::double::exp::exp(x, cube_math::MathConfig::FAST)
    } else if comptime!(mode == 2) {
        1.0 + x * 0.0
    } else {
        cube_math::double::exp::exp(x, cube_math::MathConfig::EXACT)
    }
}

/// [`mg_exp`] on N elements at once — M-20: `cube_math`'s `exp_vec`, which
/// is the scalar schedule on the whole vector and bit-identical to `exp` per
/// element (`cube-math/tests/vector.rs`, which now covers the whole exp/log
/// family), so the kernels below keep their per-point bits while the N
/// exponentials share one chain.
#[cube]
fn mg_exp_vec<N: Size>(x: Vector<f64, N>, #[comptime] mode: u32) -> Vector<f64, N> {
    if comptime!(mode == 1) {
        cube_math::double::exp::exp_vec::<N>(x, cube_math::MathConfig::FAST)
    } else if comptime!(mode == 2) {
        Vector::<f64, N>::new(1.0) + x * Vector::<f64, N>::new(0.0)
    } else {
        cube_math::double::exp::exp_vec::<N>(x, cube_math::MathConfig::EXACT)
    }
}

/// The vector width the pair kernels run at on `client`: the widest the
/// device likes for `f64` that divides [`POINT_PAD`], or the
/// `PYSCF_MG_PAIR_LINE` pin (which must divide [`POINT_PAD`] too).
pub fn pair_line_size<R: Runtime>(client: &ComputeClient<R>) -> usize {
    pyscf_algebra::launch::pinned_line_size::<R, f64>(client, POINT_PAD, "PYSCF_MG_PAIR_LINE")
}

/// M-19: the vector reverse kernel's group table for width `line` — per
/// group its first occurrence; groups tile each block's occurrence range in
/// order, the last one of a block ragged.
fn reverse_groups(batch: &PairSlotBatch, line: usize) -> Vec<u32> {
    let mut grp = Vec::new();
    for w in batch.block_inst0.windows(2) {
        let (i0, i1) = (w[0] as usize, w[1] as usize);
        let mut o = i0;
        while o < i1 {
            grp.push(o as u32);
            o += line;
        }
    }
    grp
}

/// Which reverse kernel runs: the M-19 vector-over-occurrences one (default)
/// or the scalar one — `PYSCF_MG_PAIR_REVERSE=scalar|vector`. Both are
/// bit-exact against the per-block route; the switch is the A/B arm.
fn reverse_vector_enabled() -> bool {
    !std::env::var("PYSCF_MG_PAIR_REVERSE").is_ok_and(|v| v.eq_ignore_ascii_case("scalar"))
}

/// M-06: device-resident invariant geometry for one batch chunk.
///
/// Handles remain private to keep CubeCL behind the kernels dependency wall.
/// Only the per-set-slot coefficients (forward) and the weights (reverse,
/// `npoints · 8` B) are uploaded per call; outputs are allocated once and
/// every launched lane overwrites its complete logical output.
pub struct PairSlotBatchDevice {
    backend: BackendKind,
    /// The cubecl stream the handles below were allocated on.
    ///
    /// **Why this field exists.** CubeCL 0.10 partitions its memory pools per
    /// [`StreamId`], and a `StreamId` IS the OS thread id (`cubecl-common`
    /// `stream_id.rs`: "The value representing the thread id", behind a
    /// `thread_local!`). `SlicedPool::find` resolves a binding by indexing
    /// `self.pages[descriptor.page()]` of *the current stream's* pool, so a
    /// `Handle` is only resolvable on the stream that allocated it.
    ///
    /// This struct is cached for the lifetime of an SCF (M-06) while the
    /// drivers above it run under changing rayon pools, so the allocating
    /// thread is routinely gone by the next call. Re-resolving a handle on a
    /// fresh stream found an empty pool and panicked inside CubeCL with
    /// `index out of bounds: the len is 0 but the index is 0`
    /// (`sliced_pool.rs:47`), which surfaced as the GATE B failures
    /// `v2_get_j`/`v2_nr_rks`/`eval_rho_g_..._v2`.
    ///
    /// Every launch and read below is therefore pinned back onto this stream
    /// with [`StreamId::executes`]. Pinning moves no arithmetic: the kernel,
    /// its lanes and their order are unchanged, so bit-parity is untouched.
    stream: StreamId,
    coords_x: Handle,
    coords_y: Handle,
    coords_z: Handle,
    point_block: Handle,
    block_point0: Handle,
    block_point_end: Handle,
    block_inst0: Handle,
    inst_block: Handle,
    inst_ref: Handle,
    occ_slot0: Handle,
    instance_alpha: Handle,
    instance_center: Handle,
    instance_radius2: Handle,
    instance_set: Handle,
    instance_kslot0: Handle,
    uocc_off: Handle,
    uocc: Handle,
    set_off: Handle,
    set_pow: Handle,
    out_rho: Handle,
    out_rho_b: std::sync::OnceLock<Handle>,
    out_integrate: Handle,
    out_integrate_b: std::sync::OnceLock<Handle>,
    /// M-16: the level's running kernel-slot integrals (shared by every chunk
    /// of the level when built through [`Self::new_shared`]).
    kint: Handle,
    kint_b: Handle,
    /// M-19: the vector reverse kernel's groups — per group, its first
    /// occurrence; a group is `line` consecutive occurrences of one block
    /// (the last group of a block may be ragged).
    grp_occ0: Handle,
    ngroups: usize,
    /// The vector width this chunk's group table was cut for.
    line: usize,
    /// M-14: the reals the output buffers actually hold. Equal to
    /// `npoints` / `nslots` when the chunk owns its buffers; the level's
    /// maxima when it borrows the shared [`PairOutScratch`].
    out_rho_cap: usize,
    npoints: usize,
    ninstances: usize,
    nslots: usize,
    nuinstances: usize,
    nsets: usize,
    nsetslots: usize,
    nkslots: usize,
    nblocks: usize,
}

/// M-14: ONE set of output buffers shared by every chunk of a level, plus
/// (M-16) the level's resident kernel-slot integrals.
///
/// A chunk's forward output is `npoints · 8` B and its reverse output
/// `nslots · 8` B, and each chunk used to own both for the life of the SCF —
/// at `25³` level 3, thirteen chunks × 48 MB of reverse output = 622 MB
/// resident that is only ever live one chunk at a time (the chunks run
/// sequentially and every launch reads its output back before the next
/// launch). The level allocates the largest chunk's buffers once; each chunk
/// writes its `npoints` / `nslots` prefix. Bit-exact: the same lanes write
/// the same values to the same logical positions; only the allocation behind
/// them changes.
///
/// Allocated on ONE stream and handed to [`PairSlotBatchDevice::new_shared`],
/// which allocates the chunk's geometry on that same stream
/// (`StreamId::executes`), so every later launch resolves every handle on the
/// stream that owns it — the session-3 pool lesson, kept.
pub struct PairOutScratch {
    backend: BackendKind,
    stream: StreamId,
    rho: Handle,
    integrate: Handle,
    kint: Handle,
    kint_b: Handle,
    max_points: usize,
    max_slots: usize,
    nkslots: usize,
}

impl core::fmt::Debug for PairOutScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairOutScratch")
            .field("max_points", &self.max_points)
            .field("max_slots", &self.max_slots)
            .field("nkslots", &self.nkslots)
            .finish_non_exhaustive()
    }
}

impl PairOutScratch {
    /// Buffers for the largest chunk of a level, plus the level's `kint`, on
    /// the current stream.
    pub fn new(
        client: &AlgebraClient,
        max_points: usize,
        max_slots: usize,
        nkslots: usize,
    ) -> Self {
        let stream = StreamId::current();
        dispatch_backend!(client, c, Rt, {
            Self {
                backend: client.kind(),
                stream,
                rho: c.empty(max_points.max(1) * core::mem::size_of::<f64>()),
                integrate: c.empty(max_slots.max(1) * core::mem::size_of::<f64>()),
                kint: c.empty(nkslots.max(1) * core::mem::size_of::<f64>()),
                kint_b: c.empty(nkslots.max(1) * core::mem::size_of::<f64>()),
                max_points,
                max_slots,
                nkslots,
            }
        })
    }

    fn check_backend(&self, client: &AlgebraClient) -> Result<(), AlgebraError> {
        if self.backend != client.kind() {
            return Err(AlgebraError::BackendMismatch {
                op: "PairOutScratch",
                expected: self.backend.name(),
                actual: client.kind().name(),
            });
        }
        Ok(())
    }

    /// Zero the level's running kernel-slot integrals — the start of a
    /// reverse sweep (`channels` = 1 or 2).
    ///
    /// # Errors
    /// A client of a different backend.
    pub fn zero_kint(&self, client: &AlgebraClient, channels: usize) -> Result<(), AlgebraError> {
        self.check_backend(client)?;
        self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                launch_zero::<Rt>(c, &self.kint, self.nkslots);
                if channels > 1 {
                    launch_zero::<Rt>(c, &self.kint_b, self.nkslots);
                }
            })
        });
        Ok(())
    }

    /// Read the level's kernel-slot integrals back — the end of a reverse
    /// sweep. One `nkslots · 8` B transfer per channel.
    ///
    /// # Errors
    /// A client of a different backend.
    pub fn read_kint(
        &self,
        client: &AlgebraClient,
        channels: usize,
    ) -> Result<Vec<Vec<f64>>, AlgebraError> {
        self.check_backend(client)?;
        Ok(self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let mut handles = vec![self.kint.clone()];
                if channels > 1 {
                    handles.push(self.kint_b.clone());
                }
                c.read(handles)
                    .into_iter()
                    .map(|b| bytemuck::cast_slice::<u8, f64>(&b)[..self.nkslots].to_vec())
                    .collect::<Vec<_>>()
            })
        }))
    }
}

impl core::fmt::Debug for PairSlotBatchDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairSlotBatchDevice")
            .field("backend", &self.backend)
            .field("stream", &self.stream)
            .field("npoints", &self.npoints)
            .field("ninstances", &self.ninstances)
            .field("nslots", &self.nslots)
            .field("nblocks", &self.nblocks)
            .finish_non_exhaustive()
    }
}

impl PairSlotBatchDevice {
    /// A chunk owning its own output buffers and `kint` (tests and the
    /// un-shared entry points).
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] on an inconsistent batch.
    pub fn new(client: &AlgebraClient, batch: &PairSlotBatch) -> Result<Self, AlgebraError> {
        validate_batch(batch)?;
        // Captured BEFORE the uploads below, which allocate on exactly this
        // stream; every later call is replayed onto it.
        let stream = StreamId::current();
        Self::upload(client, batch, stream, None)
    }

    /// M-14: like [`Self::new`], but the chunk borrows `scratch`'s output
    /// buffers and `kint`, and allocates its geometry on `scratch`'s stream,
    /// so the whole level lives on one stream. Refuses a chunk larger than
    /// the scratch.
    ///
    /// # Errors
    /// As [`Self::new`], plus a chunk that exceeds the scratch's capacity.
    pub fn new_shared(
        client: &AlgebraClient,
        batch: &PairSlotBatch,
        scratch: &PairOutScratch,
    ) -> Result<Self, AlgebraError> {
        validate_batch(batch)?;
        if batch.npoints() > scratch.max_points
            || batch.nslots() > scratch.max_slots
            || batch.nkslots() != scratch.nkslots
        {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!(
                    "chunk within the scratch: npoints <= {}, nslots <= {}, nkslots == {}",
                    scratch.max_points, scratch.max_slots, scratch.nkslots
                ),
                actual: format!(
                    "{} / {} / {}",
                    batch.npoints(),
                    batch.nslots(),
                    batch.nkslots()
                ),
            });
        }
        scratch
            .stream
            .executes(|| Self::upload(client, batch, scratch.stream, Some(scratch)))
    }

    fn upload(
        client: &AlgebraClient,
        batch: &PairSlotBatch,
        stream: StreamId,
        scratch: Option<&PairOutScratch>,
    ) -> Result<Self, AlgebraError> {
        Ok(dispatch_backend!(client, c, Rt, {
            let line = pair_line_size::<Rt>(c);
            let grp_occ0 = reverse_groups(batch, line);
            Self {
                backend: client.kind(),
                stream,
                ngroups: grp_occ0.len(),
                grp_occ0: upload_u32::<Rt>(c, &grp_occ0),
                line,
                coords_x: upload::<Rt, f64>(c, &batch.coords_x),
                coords_y: upload::<Rt, f64>(c, &batch.coords_y),
                coords_z: upload::<Rt, f64>(c, &batch.coords_z),
                point_block: upload_u32::<Rt>(c, &batch.point_block),
                block_point0: upload_u32::<Rt>(c, &batch.block_point0),
                block_point_end: upload_u32::<Rt>(c, &batch.block_point_end),
                block_inst0: upload_u32::<Rt>(c, &batch.block_inst0),
                inst_block: upload_u32::<Rt>(c, &batch.inst_block),
                inst_ref: upload_u32::<Rt>(c, &batch.inst_ref),
                occ_slot0: upload_u32::<Rt>(c, &batch.occ_slot0),
                instance_alpha: upload::<Rt, f64>(c, &batch.instance_alpha),
                instance_center: upload::<Rt, f64>(c, &batch.instance_center),
                instance_radius2: upload::<Rt, f64>(c, &batch.instance_radius2),
                instance_set: upload_u32::<Rt>(c, &batch.instance_set),
                instance_kslot0: upload_u32::<Rt>(c, &batch.instance_kslot0),
                uocc_off: upload_u32::<Rt>(c, &batch.uocc_off),
                uocc: upload_u32::<Rt>(c, &batch.uocc),
                set_off: upload_u32::<Rt>(c, &batch.set_off),
                set_pow: upload_u32::<Rt>(c, &batch.set_pow),
                out_rho: match scratch {
                    Some(sc) => sc.rho.clone(),
                    None => c.empty(batch.npoints().max(1) * core::mem::size_of::<f64>()),
                },
                out_rho_b: std::sync::OnceLock::new(),
                out_integrate: match scratch {
                    Some(sc) => sc.integrate.clone(),
                    None => c.empty(batch.nslots().max(1) * core::mem::size_of::<f64>()),
                },
                out_integrate_b: std::sync::OnceLock::new(),
                kint: match scratch {
                    Some(sc) => sc.kint.clone(),
                    None => c.empty(batch.nkslots().max(1) * core::mem::size_of::<f64>()),
                },
                kint_b: match scratch {
                    Some(sc) => sc.kint_b.clone(),
                    None => c.empty(batch.nkslots().max(1) * core::mem::size_of::<f64>()),
                },
                out_rho_cap: scratch.map_or(batch.npoints(), |sc| sc.max_points),
                npoints: batch.npoints(),
                ninstances: batch.ninstances(),
                nslots: batch.nslots(),
                nuinstances: batch.nuinstances(),
                nsets: batch.set_off.len().saturating_sub(1),
                nsetslots: batch.nsetslots(),
                nkslots: batch.nkslots(),
                nblocks: batch.nblocks(),
            }
        }))
    }

    fn check_backend(&self, client: &AlgebraClient) -> Result<(), AlgebraError> {
        if self.backend != client.kind() {
            return Err(AlgebraError::BackendMismatch {
                op: "PairSlotBatchDevice",
                expected: self.backend.name(),
                actual: client.kind().name(),
            });
        }
        Ok(())
    }

    /// The forward direction: `set_coef` — one coefficient per SET slot of
    /// the level (`nsetslots`). Returns one value per PADDED point.
    ///
    /// # Errors
    /// A client of a different backend, or a wrong coefficient length.
    pub fn rho(&self, client: &AlgebraClient, set_coef: &[f64]) -> Result<Vec<f64>, AlgebraError> {
        self.check_backend(client)?;
        if set_coef.len() != self.nsetslots {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("set_coef.len() == nsetslots = {}", self.nsetslots),
                actual: set_coef.len().to_string(),
            });
        }
        Ok(self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let coef = upload::<Rt, f64>(c, set_coef);
                launch_rho_resident::<Rt>(self, &coef, &self.out_rho, c);
                let bytes = c.read(vec![prefix(&self.out_rho, self.out_rho_cap, self.npoints)]);
                bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
            })
        }))
    }

    /// Collocate alpha and beta densities in one geometry traversal.
    ///
    /// # Errors
    /// As [`Self::rho`].
    pub fn rho2(
        &self,
        client: &AlgebraClient,
        set_coef: [&[f64]; 2],
    ) -> Result<[Vec<f64>; 2], AlgebraError> {
        self.check_backend(client)?;
        if set_coef.iter().any(|c| c.len() != self.nsetslots) {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("both set_coef lengths == nsetslots = {}", self.nsetslots),
                actual: format!("{}/{}", set_coef[0].len(), set_coef[1].len()),
            });
        }
        Ok(self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let ca = upload::<Rt, f64>(c, set_coef[0]);
                let cb = upload::<Rt, f64>(c, set_coef[1]);
                let out_b = self
                    .out_rho_b
                    .get_or_init(|| c.empty(self.npoints.max(1) * core::mem::size_of::<f64>()));
                launch_rho2_resident::<Rt>(self, &ca, &cb, &self.out_rho, out_b, c);
                let bytes = c.read(vec![
                    prefix(&self.out_rho, self.out_rho_cap, self.npoints),
                    out_b.clone(),
                ]);
                [
                    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
                    bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
                ]
            })
        }))
    }

    /// The reverse direction: integrate `weight` (one per PADDED point; pads
    /// are never read) against every occurrence's monomials and fold the
    /// result onto the resident `kint` (M-16). Nothing is read back; call
    /// [`PairOutScratch::read_kint`] (or [`Self::read_kint`]) after the last
    /// chunk. The driver zeroes `kint` before the first chunk.
    ///
    /// The reverse direction has no coefficient (M-12): the drivers always
    /// passed `1.0`, and `1.0 · x == x` exactly, so the multiply is gone.
    ///
    /// # Errors
    /// A client of a different backend, or a wrong weight length.
    pub fn integrate_into_kint(
        &self,
        client: &AlgebraClient,
        weight: &[f64],
    ) -> Result<(), AlgebraError> {
        self.check_backend(client)?;
        if weight.len() != self.npoints {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("weight.len() == npoints = {}", self.npoints),
                actual: weight.len().to_string(),
            });
        }
        self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let w = upload::<Rt, f64>(c, weight);
                launch_integrate_resident::<Rt>(self, &w, &self.out_integrate, c);
                launch_fold_resident::<Rt>(self, &self.out_integrate, &self.kint, c);
            })
        });
        Ok(())
    }

    /// Two-spin twin of [`Self::integrate_into_kint`]: spin 0 folds onto
    /// `kint`, spin 1 onto `kint_b`.
    ///
    /// # Errors
    /// As [`Self::integrate_into_kint`].
    pub fn integrate2_into_kint(
        &self,
        client: &AlgebraClient,
        weight: [&[f64]; 2],
    ) -> Result<(), AlgebraError> {
        self.check_backend(client)?;
        if weight.iter().any(|w| w.len() != self.npoints) {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("both weight lengths == npoints = {}", self.npoints),
                actual: format!("{}/{}", weight[0].len(), weight[1].len()),
            });
        }
        self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let wa = upload::<Rt, f64>(c, weight[0]);
                let wb = upload::<Rt, f64>(c, weight[1]);
                let out_b = self
                    .out_integrate_b
                    .get_or_init(|| c.empty(self.nslots.max(1) * core::mem::size_of::<f64>()));
                launch_integrate2_resident::<Rt>(self, &wa, &wb, &self.out_integrate, out_b, c);
                launch_fold_resident::<Rt>(self, &self.out_integrate, &self.kint, c);
                launch_fold_resident::<Rt>(self, out_b, &self.kint_b, c);
            })
        });
        Ok(())
    }

    /// Zero this chunk's `kint` (own or shared) — the un-shared twin of
    /// [`PairOutScratch::zero_kint`].
    ///
    /// # Errors
    /// A client of a different backend.
    pub fn zero_kint(&self, client: &AlgebraClient, channels: usize) -> Result<(), AlgebraError> {
        self.check_backend(client)?;
        self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                launch_zero::<Rt>(c, &self.kint, self.nkslots);
                if channels > 1 {
                    launch_zero::<Rt>(c, &self.kint_b, self.nkslots);
                }
            })
        });
        Ok(())
    }

    /// Read this chunk's `kint` (own or shared) — the un-shared twin of
    /// [`PairOutScratch::read_kint`].
    ///
    /// # Errors
    /// A client of a different backend.
    pub fn read_kint(
        &self,
        client: &AlgebraClient,
        channels: usize,
    ) -> Result<Vec<Vec<f64>>, AlgebraError> {
        self.check_backend(client)?;
        Ok(self.stream.executes(|| {
            dispatch_backend!(client, c, Rt, {
                let mut handles = vec![self.kint.clone()];
                if channels > 1 {
                    handles.push(self.kint_b.clone());
                }
                c.read(handles)
                    .into_iter()
                    .map(|b| bytemuck::cast_slice::<u8, f64>(&b)[..self.nkslots].to_vec())
                    .collect::<Vec<_>>()
            })
        }))
    }
}

/// `out[i] = 0` — the resident `kint` reset (M-16).
#[cube(launch_unchecked)]
fn mg_zero_kernel(out: &mut Array<f64>, n: usize) {
    let i = ABSOLUTE_POS;
    if i < n {
        out[i] = 0.0;
    }
}

/// M-17: the forward direction over `Vector<f64, N>` points. One lane per N
/// adjacent (padded) points of one block; the lane's block selects which
/// instances it sums, in the same order the per-block launch used, and each
/// instance's term set supplies its monomials in table order.
#[cube(launch_unchecked)]
fn mg_rho_kernel<N: Size>(
    coords_x: &Array<Vector<f64, N>>,
    coords_y: &Array<Vector<f64, N>>,
    coords_z: &Array<Vector<f64, N>>,
    point_block: &Array<u32>,
    block_inst0: &Array<u32>,
    inst_ref: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    set_coef: &Array<f64>,
    out: &mut Array<Vector<f64, N>>,
    nlanes: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    let v = ABSOLUTE_POS;
    if v < nlanes {
        let x = coords_x[v];
        let y = coords_y[v];
        let z = coords_z[v];
        let b = point_block[v * N::value()] as usize;
        let i0 = block_inst0[b] as usize;
        let i1 = block_inst0[b + 1] as usize;
        let zero = Vector::<f64, N>::new(0.0);
        let one = Vector::<f64, N>::new(1.0);
        let mut acc = zero;
        for inst in i0..i1 {
            let u = inst_ref[inst] as usize;
            let eta = Vector::<f64, N>::new(instance_alpha[u]);
            let dx = x - Vector::<f64, N>::new(instance_center[u * 3]);
            let dy = y - Vector::<f64, N>::new(instance_center[u * 3 + 1]);
            let dz = z - Vector::<f64, N>::new(instance_center[u * 3 + 2]);
            let r2 = dx * dx + dy * dy + dz * dz;
            let arg = zero - eta * r2;
            // M-21: an element outside the instance's radius receives an
            // exact `±0.0` from it (its `e` is zeroed), which leaves its
            // accumulator unchanged — the scalar screen's "skip", width for
            // width; a vector wholly outside skips the instance entirely.
            let mut run = true;
            let mut e = zero;
            if comptime!(screen) {
                let rad2 = instance_radius2[u];
                let mut inside: u32 = 0u32;
                #[unroll]
                for j in 0..N::value() {
                    if r2[j] <= rad2 {
                        inside += 1u32;
                    }
                }
                run = inside > 0u32;
                if run {
                    e = mg_exp_vec::<N>(arg, exp_mode);
                    #[unroll]
                    for j in 0..N::value() {
                        if r2[j] > rad2 {
                            e[j] = 0.0;
                        }
                    }
                }
            } else {
                e = mg_exp_vec::<N>(arg, exp_mode);
            }
            if run {
                let s = instance_set[u] as usize;
                let so0 = set_off[s] as usize;
                let so1 = set_off[s + 1] as usize;
                for sl in so0..so1 {
                    let packed = set_pow[sl];
                    let ix = packed & 255;
                    let iy = (packed >> 8) & 255;
                    let iz = (packed >> 16) & 255;
                    let coef = Vector::<f64, N>::new(set_coef[sl]);
                    let mut poly = one;
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
        }
        out[v] = acc;
    }
}

/// Two-spin twin of [`mg_rho_kernel`]: one geometry traversal, two
/// coefficient vectors, two outputs — each channel's arithmetic is the
/// single-channel kernel's.
#[cube(launch_unchecked)]
fn mg_rho2_kernel<N: Size>(
    coords_x: &Array<Vector<f64, N>>,
    coords_y: &Array<Vector<f64, N>>,
    coords_z: &Array<Vector<f64, N>>,
    point_block: &Array<u32>,
    block_inst0: &Array<u32>,
    inst_ref: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    set_coef_a: &Array<f64>,
    set_coef_b: &Array<f64>,
    out_a: &mut Array<Vector<f64, N>>,
    out_b: &mut Array<Vector<f64, N>>,
    nlanes: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    let v = ABSOLUTE_POS;
    if v < nlanes {
        let x = coords_x[v];
        let y = coords_y[v];
        let z = coords_z[v];
        let b = point_block[v * N::value()] as usize;
        let i0 = block_inst0[b] as usize;
        let i1 = block_inst0[b + 1] as usize;
        let zero = Vector::<f64, N>::new(0.0);
        let one = Vector::<f64, N>::new(1.0);
        let mut acc_a = zero;
        let mut acc_b = zero;
        for inst in i0..i1 {
            let u = inst_ref[inst] as usize;
            let eta = Vector::<f64, N>::new(instance_alpha[u]);
            let dx = x - Vector::<f64, N>::new(instance_center[u * 3]);
            let dy = y - Vector::<f64, N>::new(instance_center[u * 3 + 1]);
            let dz = z - Vector::<f64, N>::new(instance_center[u * 3 + 2]);
            let r2 = dx * dx + dy * dy + dz * dz;
            let arg = zero - eta * r2;
            // M-21: an element outside the instance's radius receives an
            // exact `±0.0` from it (its `e` is zeroed), which leaves its
            // accumulator unchanged — the scalar screen's "skip", width for
            // width; a vector wholly outside skips the instance entirely.
            let mut run = true;
            let mut e = zero;
            if comptime!(screen) {
                let rad2 = instance_radius2[u];
                let mut inside: u32 = 0u32;
                #[unroll]
                for j in 0..N::value() {
                    if r2[j] <= rad2 {
                        inside += 1u32;
                    }
                }
                run = inside > 0u32;
                if run {
                    e = mg_exp_vec::<N>(arg, exp_mode);
                    #[unroll]
                    for j in 0..N::value() {
                        if r2[j] > rad2 {
                            e[j] = 0.0;
                        }
                    }
                }
            } else {
                e = mg_exp_vec::<N>(arg, exp_mode);
            }
            if run {
                let s = instance_set[u] as usize;
                let so0 = set_off[s] as usize;
                let so1 = set_off[s + 1] as usize;
                for sl in so0..so1 {
                    let packed = set_pow[sl];
                    let ix = packed & 255;
                    let iy = (packed >> 8) & 255;
                    let iz = (packed >> 16) & 255;
                    let mut poly = one;
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
                    acc_a += Vector::<f64, N>::new(set_coef_a[sl]) * poly * e;
                    acc_b += Vector::<f64, N>::new(set_coef_b[sl]) * poly * e;
                }
            }
        }
        out_a[v] = acc_a;
        out_b[v] = acc_b;
    }
}

/// The reverse direction (M-08 register accumulators, M-15 term sets). One
/// lane per instance occurrence; the lane's block selects which grid points
/// it integrates over, in the same order the per-block launch used.
#[cube(launch_unchecked)]
fn mg_integrate_kernel(
    coords_x: &Array<f64>,
    coords_y: &Array<f64>,
    coords_z: &Array<f64>,
    weight: &Array<f64>,
    inst_block: &Array<u32>,
    block_point0: &Array<u32>,
    block_point_end: &Array<u32>,
    inst_ref: &Array<u32>,
    occ_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    out: &mut Array<f64>,
    nocc: usize,
    lane0: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    // `lane0`: chunked on the CPU runtime — `acc` is stack per iteration
    // there (`launch_1d_chunked`).
    let occ = ABSOLUTE_POS + lane0;
    if occ < nocc {
        let u = inst_ref[occ] as usize;
        let eta = instance_alpha[u];
        let cx = instance_center[u * 3];
        let cy = instance_center[u * 3 + 1];
        let cz = instance_center[u * 3 + 2];
        let rad2 = instance_radius2[u];
        let s = instance_set[u] as usize;
        let so0 = set_off[s] as usize;
        let nsl = set_off[s + 1] as usize - so0;
        let b = inst_block[occ] as usize;
        let g0 = block_point0[b] as usize;
        let g1 = block_point_end[b] as usize;
        let mut acc = Array::<f64>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl {
            acc[local] = 0.0;
        }
        for g in g0..g1 {
            let dx = coords_x[g] - cx;
            let dy = coords_y[g] - cy;
            let dz = coords_z[g] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let mut keep = true;
            if comptime!(screen) {
                keep = r2 <= rad2;
            }
            if keep {
                let e = mg_exp(0.0 - eta * r2, exp_mode);
                let we = weight[g] * e;
                for local in 0..nsl {
                    let packed = set_pow[so0 + local];
                    let ix = packed & 255;
                    let iy = (packed >> 8) & 255;
                    let iz = (packed >> 16) & 255;
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
                    // M-12: the driver's coefficient was always `1.0` here, and
                    // `1.0 * x == x` exactly, so dropping it is bit-exact.
                    acc[local] += poly * we;
                }
            }
        }
        let o0 = occ_slot0[occ] as usize;
        for local in 0..nsl {
            out[o0 + local] = acc[local];
        }
    }
}

/// Two-spin twin of [`mg_integrate_kernel`].
#[cube(launch_unchecked)]
fn mg_integrate2_kernel(
    coords_x: &Array<f64>,
    coords_y: &Array<f64>,
    coords_z: &Array<f64>,
    weight_a: &Array<f64>,
    weight_b: &Array<f64>,
    inst_block: &Array<u32>,
    block_point0: &Array<u32>,
    block_point_end: &Array<u32>,
    inst_ref: &Array<u32>,
    occ_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    out_a: &mut Array<f64>,
    out_b: &mut Array<f64>,
    nocc: usize,
    lane0: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    let occ = ABSOLUTE_POS + lane0;
    if occ < nocc {
        let u = inst_ref[occ] as usize;
        let eta = instance_alpha[u];
        let cx = instance_center[u * 3];
        let cy = instance_center[u * 3 + 1];
        let cz = instance_center[u * 3 + 2];
        let rad2 = instance_radius2[u];
        let s = instance_set[u] as usize;
        let so0 = set_off[s] as usize;
        let nsl = set_off[s + 1] as usize - so0;
        let b = inst_block[occ] as usize;
        let g0 = block_point0[b] as usize;
        let g1 = block_point_end[b] as usize;
        let mut acc_a = Array::<f64>::new(MAX_SLOTS_PER_INSTANCE);
        let mut acc_b = Array::<f64>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl {
            acc_a[local] = 0.0;
            acc_b[local] = 0.0;
        }
        for g in g0..g1 {
            let dx = coords_x[g] - cx;
            let dy = coords_y[g] - cy;
            let dz = coords_z[g] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let mut keep = true;
            if comptime!(screen) {
                keep = r2 <= rad2;
            }
            if keep {
                let e = mg_exp(0.0 - eta * r2, exp_mode);
                let we_a = weight_a[g] * e;
                let we_b = weight_b[g] * e;
                for local in 0..nsl {
                    let packed = set_pow[so0 + local];
                    let ix = packed & 255;
                    let iy = (packed >> 8) & 255;
                    let iz = (packed >> 16) & 255;
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
                    acc_a[local] += poly * we_a;
                    acc_b[local] += poly * we_b;
                }
            }
        }
        let o0 = occ_slot0[occ] as usize;
        for local in 0..nsl {
            out_a[o0 + local] = acc_a[local];
            out_b[o0 + local] = acc_b[local];
        }
    }
}

/// M-19: the reverse direction over `Vector<f64, N>` OCCURRENCES. One lane
/// per group of N consecutive occurrences of one block (a ragged tail
/// duplicates the block's last occurrence into its unused elements, whose
/// results are never stored). Per grid point the lane forms N displacements,
/// N scalar `exp`s (one per element, the bit-exact `cube_math` one) and, per
/// monomial slot, N predicated power products:
///
/// ```text
/// poly *= dx · m + (1 − m),  m ∈ {0, 1}
/// ```
///
/// is exactly `poly *= dx` when `m = 1` (`dx · 1 = dx`, `dx + 0 = dx`) and
/// exactly `poly` when `m = 0` (`dx · 0 = ±0`, `±0 + 1 = 1`, `poly · 1 =
/// poly`), so a lane whose slot has power `p` performs the scalar kernel's
/// `p` multiplications by `dx` in the scalar kernel's order — the one
/// difference, a `-0.0` where the scalar path had `+0.0 · dx` with `dx =
/// -0.0`, is a sign of zero that no accumulator can observe (an accumulator
/// starting at `+0.0` never becomes `-0.0`). Powers are at most 2 in a
/// batched set (`validate_batch`), so two predicated steps per axis suffice.
#[cube(launch_unchecked)]
fn mg_integrate_vec_kernel<N: Size>(
    coords_x: &Array<f64>,
    coords_y: &Array<f64>,
    coords_z: &Array<f64>,
    weight: &Array<f64>,
    grp_occ0: &Array<u32>,
    inst_block: &Array<u32>,
    block_point0: &Array<u32>,
    block_point_end: &Array<u32>,
    block_inst0: &Array<u32>,
    inst_ref: &Array<u32>,
    occ_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    out: &mut Array<f64>,
    ngroups: usize,
    lane0: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    let grp = ABSOLUTE_POS + lane0;
    if grp < ngroups {
        let o0 = grp_occ0[grp] as usize;
        let b = inst_block[o0] as usize;
        let i1 = block_inst0[b + 1] as usize;
        let g0 = block_point0[b] as usize;
        let g1 = block_point_end[b] as usize;
        let zero = Vector::<f64, N>::new(0.0);
        let one = Vector::<f64, N>::new(1.0);
        let mut eta = Vector::<f64, N>::empty();
        let mut cx = Vector::<f64, N>::empty();
        let mut cy = Vector::<f64, N>::empty();
        let mut cz = Vector::<f64, N>::empty();
        let mut rad2 = Vector::<f64, N>::empty();
        // Per slot, per element: the packed powers (0 = "no slot", whose
        // products are all `× 1` and whose result is never stored).
        let mut pw = Array::<Vector<u32, N>>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..MAX_SLOTS_PER_INSTANCE {
            pw[local] = Vector::<u32, N>::new(0u32);
        }
        let mut nsl_max = 0usize;
        #[unroll]
        for j in 0..N::value() {
            let mut occ = o0 + j;
            if occ >= i1 {
                occ = i1 - 1;
            }
            let u = inst_ref[occ] as usize;
            eta[j] = instance_alpha[u];
            cx[j] = instance_center[u * 3];
            cy[j] = instance_center[u * 3 + 1];
            cz[j] = instance_center[u * 3 + 2];
            rad2[j] = instance_radius2[u];
            let s = instance_set[u] as usize;
            let so0 = set_off[s] as usize;
            let nsl = set_off[s + 1] as usize - so0;
            if nsl > nsl_max {
                nsl_max = nsl;
            }
            for local in 0..nsl {
                let mut v = pw[local];
                v[j] = set_pow[so0 + local];
                pw[local] = v;
            }
        }
        // Hoisted masks: per slot and axis, the two predication steps.
        let m255 = Vector::<u32, N>::new(255u32);
        let m1 = Vector::<u32, N>::new(1u32);
        let sh1 = Vector::<u32, N>::new(1u32);
        let sh8 = Vector::<u32, N>::new(8u32);
        let sh16 = Vector::<u32, N>::new(16u32);
        let mut mask = Array::<Vector<f64, N>>::new(6 * MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl_max {
            let packed = pw[local];
            let ix = packed & m255;
            let iy = (packed >> sh8) & m255;
            let iz = (packed >> sh16) & m255;
            mask[local * 6] = Vector::<f64, N>::cast_from((ix & m1) | (ix >> sh1));
            mask[local * 6 + 1] = Vector::<f64, N>::cast_from(ix >> sh1);
            mask[local * 6 + 2] = Vector::<f64, N>::cast_from((iy & m1) | (iy >> sh1));
            mask[local * 6 + 3] = Vector::<f64, N>::cast_from(iy >> sh1);
            mask[local * 6 + 4] = Vector::<f64, N>::cast_from((iz & m1) | (iz >> sh1));
            mask[local * 6 + 5] = Vector::<f64, N>::cast_from(iz >> sh1);
        }
        let mut acc = Array::<Vector<f64, N>>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl_max {
            acc[local] = zero;
        }
        for g in g0..g1 {
            let dx = Vector::<f64, N>::new(coords_x[g]) - cx;
            let dy = Vector::<f64, N>::new(coords_y[g]) - cy;
            let dz = Vector::<f64, N>::new(coords_z[g]) - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let arg = zero - eta * r2;
            // M-21: per element (occurrence) the point is inside its radius
            // or receives an exact `±0.0`; a point outside every lane's
            // radius is skipped. The group's occurrences are different
            // instances (usually images of one pair, far apart), so at any
            // point most elements are outside: one SCALAR exp per inside
            // element beats a vector exp over all of them.
            let mut run = true;
            let mut e = zero;
            if comptime!(screen) {
                let mut inside: u32 = 0u32;
                #[unroll]
                for j in 0..N::value() {
                    if r2[j] <= rad2[j] {
                        inside += 1u32;
                        e[j] = mg_exp(arg[j], exp_mode);
                    }
                }
                run = inside > 0u32;
            } else {
                e = mg_exp_vec::<N>(arg, exp_mode);
            }
            if run {
                let we = Vector::<f64, N>::new(weight[g]) * e;
                for local in 0..nsl_max {
                    let mut poly = one;
                    let m = mask[local * 6];
                    poly *= dx * m + (one - m);
                    let m = mask[local * 6 + 1];
                    poly *= dx * m + (one - m);
                    let m = mask[local * 6 + 2];
                    poly *= dy * m + (one - m);
                    let m = mask[local * 6 + 3];
                    poly *= dy * m + (one - m);
                    let m = mask[local * 6 + 4];
                    poly *= dz * m + (one - m);
                    let m = mask[local * 6 + 5];
                    poly *= dz * m + (one - m);
                    acc[local] += poly * we;
                }
            }
        }
        #[unroll]
        for j in 0..N::value() {
            let occ = o0 + j;
            if occ < i1 {
                let u = inst_ref[occ] as usize;
                let s = instance_set[u] as usize;
                let nsl = set_off[s + 1] as usize - set_off[s] as usize;
                let oo = occ_slot0[occ] as usize;
                for local in 0..nsl {
                    let a = acc[local];
                    out[oo + local] = a[j];
                }
            }
        }
    }
}

/// Two-spin twin of [`mg_integrate_vec_kernel`].
#[cube(launch_unchecked)]
fn mg_integrate2_vec_kernel<N: Size>(
    coords_x: &Array<f64>,
    coords_y: &Array<f64>,
    coords_z: &Array<f64>,
    weight_a: &Array<f64>,
    weight_b: &Array<f64>,
    grp_occ0: &Array<u32>,
    inst_block: &Array<u32>,
    block_point0: &Array<u32>,
    block_point_end: &Array<u32>,
    block_inst0: &Array<u32>,
    inst_ref: &Array<u32>,
    occ_slot0: &Array<u32>,
    instance_alpha: &Array<f64>,
    instance_center: &Array<f64>,
    instance_radius2: &Array<f64>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    set_pow: &Array<u32>,
    out_a: &mut Array<f64>,
    out_b: &mut Array<f64>,
    ngroups: usize,
    lane0: usize,
    #[comptime] exp_mode: u32,
    #[comptime] screen: bool,
) {
    let grp = ABSOLUTE_POS + lane0;
    if grp < ngroups {
        let o0 = grp_occ0[grp] as usize;
        let b = inst_block[o0] as usize;
        let i1 = block_inst0[b + 1] as usize;
        let g0 = block_point0[b] as usize;
        let g1 = block_point_end[b] as usize;
        let zero = Vector::<f64, N>::new(0.0);
        let one = Vector::<f64, N>::new(1.0);
        let mut eta = Vector::<f64, N>::empty();
        let mut cx = Vector::<f64, N>::empty();
        let mut cy = Vector::<f64, N>::empty();
        let mut cz = Vector::<f64, N>::empty();
        let mut rad2 = Vector::<f64, N>::empty();
        let mut pw = Array::<Vector<u32, N>>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..MAX_SLOTS_PER_INSTANCE {
            pw[local] = Vector::<u32, N>::new(0u32);
        }
        let mut nsl_max = 0usize;
        #[unroll]
        for j in 0..N::value() {
            let mut occ = o0 + j;
            if occ >= i1 {
                occ = i1 - 1;
            }
            let u = inst_ref[occ] as usize;
            eta[j] = instance_alpha[u];
            cx[j] = instance_center[u * 3];
            cy[j] = instance_center[u * 3 + 1];
            cz[j] = instance_center[u * 3 + 2];
            rad2[j] = instance_radius2[u];
            let s = instance_set[u] as usize;
            let so0 = set_off[s] as usize;
            let nsl = set_off[s + 1] as usize - so0;
            if nsl > nsl_max {
                nsl_max = nsl;
            }
            for local in 0..nsl {
                let mut v = pw[local];
                v[j] = set_pow[so0 + local];
                pw[local] = v;
            }
        }
        let m255 = Vector::<u32, N>::new(255u32);
        let m1 = Vector::<u32, N>::new(1u32);
        let sh1 = Vector::<u32, N>::new(1u32);
        let sh8 = Vector::<u32, N>::new(8u32);
        let sh16 = Vector::<u32, N>::new(16u32);
        let mut mask = Array::<Vector<f64, N>>::new(6 * MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl_max {
            let packed = pw[local];
            let ix = packed & m255;
            let iy = (packed >> sh8) & m255;
            let iz = (packed >> sh16) & m255;
            mask[local * 6] = Vector::<f64, N>::cast_from((ix & m1) | (ix >> sh1));
            mask[local * 6 + 1] = Vector::<f64, N>::cast_from(ix >> sh1);
            mask[local * 6 + 2] = Vector::<f64, N>::cast_from((iy & m1) | (iy >> sh1));
            mask[local * 6 + 3] = Vector::<f64, N>::cast_from(iy >> sh1);
            mask[local * 6 + 4] = Vector::<f64, N>::cast_from((iz & m1) | (iz >> sh1));
            mask[local * 6 + 5] = Vector::<f64, N>::cast_from(iz >> sh1);
        }
        let mut acc_a = Array::<Vector<f64, N>>::new(MAX_SLOTS_PER_INSTANCE);
        let mut acc_b = Array::<Vector<f64, N>>::new(MAX_SLOTS_PER_INSTANCE);
        for local in 0..nsl_max {
            acc_a[local] = zero;
            acc_b[local] = zero;
        }
        for g in g0..g1 {
            let dx = Vector::<f64, N>::new(coords_x[g]) - cx;
            let dy = Vector::<f64, N>::new(coords_y[g]) - cy;
            let dz = Vector::<f64, N>::new(coords_z[g]) - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let arg = zero - eta * r2;
            // M-21: per element (occurrence) the point is inside its radius
            // or receives an exact `±0.0`; a point outside every lane's
            // radius is skipped. The group's occurrences are different
            // instances (usually images of one pair, far apart), so at any
            // point most elements are outside: one SCALAR exp per inside
            // element beats a vector exp over all of them.
            let mut run = true;
            let mut e = zero;
            if comptime!(screen) {
                let mut inside: u32 = 0u32;
                #[unroll]
                for j in 0..N::value() {
                    if r2[j] <= rad2[j] {
                        inside += 1u32;
                        e[j] = mg_exp(arg[j], exp_mode);
                    }
                }
                run = inside > 0u32;
            } else {
                e = mg_exp_vec::<N>(arg, exp_mode);
            }
            if run {
                let we_a = Vector::<f64, N>::new(weight_a[g]) * e;
                let we_b = Vector::<f64, N>::new(weight_b[g]) * e;
                for local in 0..nsl_max {
                    let mut poly = one;
                    let m = mask[local * 6];
                    poly *= dx * m + (one - m);
                    let m = mask[local * 6 + 1];
                    poly *= dx * m + (one - m);
                    let m = mask[local * 6 + 2];
                    poly *= dy * m + (one - m);
                    let m = mask[local * 6 + 3];
                    poly *= dy * m + (one - m);
                    let m = mask[local * 6 + 4];
                    poly *= dz * m + (one - m);
                    let m = mask[local * 6 + 5];
                    poly *= dz * m + (one - m);
                    acc_a[local] += poly * we_a;
                    acc_b[local] += poly * we_b;
                }
            }
        }
        #[unroll]
        for j in 0..N::value() {
            let occ = o0 + j;
            if occ < i1 {
                let u = inst_ref[occ] as usize;
                let s = instance_set[u] as usize;
                let nsl = set_off[s + 1] as usize - set_off[s] as usize;
                let oo = occ_slot0[occ] as usize;
                for local in 0..nsl {
                    let a = acc_a[local];
                    let bb = acc_b[local];
                    out_a[oo + local] = a[j];
                    out_b[oo + local] = bb[j];
                }
            }
        }
    }
}

/// M-16: fold one chunk's reverse outputs onto the level's running `kint`.
/// One lane per `(distinct instance, monomial slot)`, `MAX_SLOTS_PER_INSTANCE`
/// lanes per instance (the surplus idle): `kint[k] = (kint[k] + out[occ_1,
/// j]) + out[occ_2, j] + …` over the instance's occurrences in increasing
/// occurrence order — the host fold's sequence, addition for addition.
#[cube(launch_unchecked)]
fn mg_fold_kernel(
    uocc_off: &Array<u32>,
    uocc: &Array<u32>,
    occ_slot0: &Array<u32>,
    instance_kslot0: &Array<u32>,
    instance_set: &Array<u32>,
    set_off: &Array<u32>,
    out: &Array<f64>,
    kint: &mut Array<f64>,
    nlanes: usize,
) {
    let lane = ABSOLUTE_POS;
    if lane < nlanes {
        let u = lane / MAX_SLOTS_PER_INSTANCE;
        let j = lane % MAX_SLOTS_PER_INSTANCE;
        let s = instance_set[u] as usize;
        let nsl = set_off[s + 1] as usize - set_off[s] as usize;
        if j < nsl {
            let k = instance_kslot0[u] as usize + j;
            let mut acc = kint[k];
            for oi in uocc_off[u] as usize..uocc_off[u + 1] as usize {
                let occ = uocc[oi] as usize;
                acc += out[occ_slot0[occ] as usize + j];
            }
            kint[k] = acc;
        }
    }
}

/// M-14: the chunk's prefix of a (possibly larger, shared) output buffer.
fn prefix(h: &Handle, cap: usize, len: usize) -> Handle {
    if cap > len {
        h.clone()
            .offset_end(((cap - len) * core::mem::size_of::<f64>()) as u64)
    } else {
        h.clone()
    }
}

fn launch_zero<R: Runtime>(client: &ComputeClient<R>, h: &Handle, n: usize) {
    if n == 0 {
        return;
    }
    let (count, dim) = launch_1d(client, n, 1);
    unsafe {
        mg_zero_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h.clone(), n),
            n,
        );
    }
}

/// Per-lane work model of the forward kernels, for [`launch_1d`].
fn forward_work_per_lane(d: &PairSlotBatchDevice, line: usize) -> usize {
    let nblocks = d.nblocks.max(1);
    line * (50 * (d.ninstances / nblocks).max(1) + 10 * (d.nslots / nblocks).max(1))
}

fn launch_rho_resident<R: Runtime>(
    d: &PairSlotBatchDevice,
    coef: &Handle,
    out: &Handle,
    client: &ComputeClient<R>,
) {
    if d.npoints == 0 {
        return;
    }
    let line = d.line;
    let nlanes = d.npoints / line;
    let (count, dim) = launch_1d(client, nlanes, forward_work_per_lane(d, line));
    let exp_mode = exp_mode_from_env();
    unsafe {
        mg_rho_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            line,
            ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.point_block.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.block_inst0.clone(), d.nblocks + 1),
            ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
            ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
            ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
            ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
            ArrayArg::from_raw_parts(coef.clone(), d.nsetslots),
            ArrayArg::from_raw_parts(out.clone(), d.npoints),
            nlanes,
            exp_mode,
            point_screen_enabled(),
        );
    }
}

fn launch_rho2_resident<R: Runtime>(
    d: &PairSlotBatchDevice,
    coef_a: &Handle,
    coef_b: &Handle,
    out_a: &Handle,
    out_b: &Handle,
    client: &ComputeClient<R>,
) {
    if d.npoints == 0 {
        return;
    }
    let line = d.line;
    let nlanes = d.npoints / line;
    let (count, dim) = launch_1d(client, nlanes, forward_work_per_lane(d, line));
    let exp_mode = exp_mode_from_env();
    unsafe {
        mg_rho2_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            line,
            ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.point_block.clone(), d.npoints),
            ArrayArg::from_raw_parts(d.block_inst0.clone(), d.nblocks + 1),
            ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
            ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
            ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
            ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
            ArrayArg::from_raw_parts(coef_a.clone(), d.nsetslots),
            ArrayArg::from_raw_parts(coef_b.clone(), d.nsetslots),
            ArrayArg::from_raw_parts(out_a.clone(), d.npoints),
            ArrayArg::from_raw_parts(out_b.clone(), d.npoints),
            nlanes,
            exp_mode,
            point_screen_enabled(),
        );
    }
}

/// Bytes of per-lane local arrays in the reverse kernels (one accumulator
/// array per channel) — what `launch_1d_chunked` budgets on the CPU runtime.
const REVERSE_LOCAL_BYTES: usize = MAX_SLOTS_PER_INSTANCE * core::mem::size_of::<f64>();

/// Bytes of per-lane local arrays in the vector reverse kernels: the packed
/// powers, the hoisted masks and the accumulator(s), per element.
fn reverse_vec_local_bytes(line: usize, channels: usize) -> usize {
    MAX_SLOTS_PER_INSTANCE * line * (4 + 6 * 8 + channels * 8) + 8 * line * 8
}

fn launch_integrate_resident<R: Runtime>(
    d: &PairSlotBatchDevice,
    weight: &Handle,
    out: &Handle,
    client: &ComputeClient<R>,
) {
    if d.ninstances == 0 {
        return;
    }
    let per_lane = 50 * (d.npoints / d.nblocks.max(1)).max(1);
    let exp_mode = exp_mode_from_env();
    if reverse_vector_enabled() {
        for chunk in pyscf_algebra::launch::launch_1d_chunked(
            client,
            d.ngroups,
            per_lane * d.line,
            reverse_vec_local_bytes(d.line, 1),
        ) {
            unsafe {
                mg_integrate_vec_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(chunk.count_x, 1, 1),
                    chunk.dim,
                    d.line,
                    ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
                    ArrayArg::from_raw_parts(weight.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.grp_occ0.clone(), d.ngroups),
                    ArrayArg::from_raw_parts(d.inst_block.clone(), d.ninstances),
                    ArrayArg::from_raw_parts(d.block_point0.clone(), d.nblocks + 1),
                    ArrayArg::from_raw_parts(d.block_point_end.clone(), d.nblocks),
                    ArrayArg::from_raw_parts(d.block_inst0.clone(), d.nblocks + 1),
                    ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
                    ArrayArg::from_raw_parts(d.occ_slot0.clone(), d.ninstances + 1),
                    ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
                    ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
                    ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
                    ArrayArg::from_raw_parts(out.clone(), d.nslots),
                    d.ngroups,
                    chunk.lane0,
                    exp_mode,
                    point_screen_enabled(),
                );
            }
        }
        return;
    }
    for chunk in pyscf_algebra::launch::launch_1d_chunked(
        client,
        d.ninstances,
        per_lane,
        REVERSE_LOCAL_BYTES,
    ) {
        unsafe {
            mg_integrate_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(chunk.count_x, 1, 1),
                chunk.dim,
                ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
                ArrayArg::from_raw_parts(weight.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.inst_block.clone(), d.ninstances),
                ArrayArg::from_raw_parts(d.block_point0.clone(), d.nblocks + 1),
                ArrayArg::from_raw_parts(d.block_point_end.clone(), d.nblocks),
                ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
                ArrayArg::from_raw_parts(d.occ_slot0.clone(), d.ninstances + 1),
                ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
                ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
                ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
                ArrayArg::from_raw_parts(out.clone(), d.nslots),
                d.ninstances,
                chunk.lane0,
                exp_mode,
                point_screen_enabled(),
            );
        }
    }
}

fn launch_integrate2_resident<R: Runtime>(
    d: &PairSlotBatchDevice,
    weight_a: &Handle,
    weight_b: &Handle,
    out_a: &Handle,
    out_b: &Handle,
    client: &ComputeClient<R>,
) {
    if d.ninstances == 0 {
        return;
    }
    let per_lane = 50 * (d.npoints / d.nblocks.max(1)).max(1);
    let exp_mode = exp_mode_from_env();
    if reverse_vector_enabled() {
        for chunk in pyscf_algebra::launch::launch_1d_chunked(
            client,
            d.ngroups,
            per_lane * d.line,
            reverse_vec_local_bytes(d.line, 2),
        ) {
            unsafe {
                mg_integrate2_vec_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(chunk.count_x, 1, 1),
                    chunk.dim,
                    d.line,
                    ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
                    ArrayArg::from_raw_parts(weight_a.clone(), d.npoints),
                    ArrayArg::from_raw_parts(weight_b.clone(), d.npoints),
                    ArrayArg::from_raw_parts(d.grp_occ0.clone(), d.ngroups),
                    ArrayArg::from_raw_parts(d.inst_block.clone(), d.ninstances),
                    ArrayArg::from_raw_parts(d.block_point0.clone(), d.nblocks + 1),
                    ArrayArg::from_raw_parts(d.block_point_end.clone(), d.nblocks),
                    ArrayArg::from_raw_parts(d.block_inst0.clone(), d.nblocks + 1),
                    ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
                    ArrayArg::from_raw_parts(d.occ_slot0.clone(), d.ninstances + 1),
                    ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
                    ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
                    ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
                    ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
                    ArrayArg::from_raw_parts(out_a.clone(), d.nslots),
                    ArrayArg::from_raw_parts(out_b.clone(), d.nslots),
                    d.ngroups,
                    chunk.lane0,
                    exp_mode,
                    point_screen_enabled(),
                );
            }
        }
        return;
    }
    for chunk in pyscf_algebra::launch::launch_1d_chunked(
        client,
        d.ninstances,
        per_lane,
        2 * REVERSE_LOCAL_BYTES,
    ) {
        unsafe {
            mg_integrate2_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(chunk.count_x, 1, 1),
                chunk.dim,
                ArrayArg::from_raw_parts(d.coords_x.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.coords_y.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.coords_z.clone(), d.npoints),
                ArrayArg::from_raw_parts(weight_a.clone(), d.npoints),
                ArrayArg::from_raw_parts(weight_b.clone(), d.npoints),
                ArrayArg::from_raw_parts(d.inst_block.clone(), d.ninstances),
                ArrayArg::from_raw_parts(d.block_point0.clone(), d.nblocks + 1),
                ArrayArg::from_raw_parts(d.block_point_end.clone(), d.nblocks),
                ArrayArg::from_raw_parts(d.inst_ref.clone(), d.ninstances),
                ArrayArg::from_raw_parts(d.occ_slot0.clone(), d.ninstances + 1),
                ArrayArg::from_raw_parts(d.instance_alpha.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.instance_center.clone(), d.nuinstances * 3),
                ArrayArg::from_raw_parts(d.instance_radius2.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
                ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
                ArrayArg::from_raw_parts(d.set_pow.clone(), d.nsetslots),
                ArrayArg::from_raw_parts(out_a.clone(), d.nslots),
                ArrayArg::from_raw_parts(out_b.clone(), d.nslots),
                d.ninstances,
                chunk.lane0,
                exp_mode,
                point_screen_enabled(),
            );
        }
    }
}

fn launch_fold_resident<R: Runtime>(
    d: &PairSlotBatchDevice,
    out: &Handle,
    kint: &Handle,
    client: &ComputeClient<R>,
) {
    if d.nuinstances == 0 {
        return;
    }
    let nlanes = d.nuinstances * MAX_SLOTS_PER_INSTANCE;
    let per_lane = 2 * (d.ninstances / d.nuinstances.max(1)).max(1);
    let (count, dim) = launch_1d(client, nlanes, per_lane);
    unsafe {
        mg_fold_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(d.uocc_off.clone(), d.nuinstances + 1),
            ArrayArg::from_raw_parts(d.uocc.clone(), d.ninstances),
            ArrayArg::from_raw_parts(d.occ_slot0.clone(), d.ninstances + 1),
            ArrayArg::from_raw_parts(d.instance_kslot0.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.instance_set.clone(), d.nuinstances),
            ArrayArg::from_raw_parts(d.set_off.clone(), d.nsets + 1),
            ArrayArg::from_raw_parts(out.clone(), d.nslots),
            ArrayArg::from_raw_parts(kint.clone(), d.nkslots),
            nlanes,
        );
    }
}

/// The M-03 batched forward direction: `rho` at every concatenated (padded)
/// grid point of one chunk, in ONE launch, through a throw-away resident
/// upload. `set_coef` has one entry per SET slot.
///
/// Returns one value per entry of [`PairSlotBatch::coords_x`], in the batch's
/// own (block-major, padded) order; the caller scatters the real points back
/// to mesh order.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] on an inconsistent batch. An empty batch
/// returns an empty vector.
pub fn collocate_pairs_rho_batched(
    client: &AlgebraClient,
    b: &PairSlotBatch,
    set_coef: &[f64],
) -> Result<Vec<f64>, AlgebraError> {
    if b.npoints() == 0 {
        validate_batch(b)?;
        return Ok(Vec::new());
    }
    PairSlotBatchDevice::new(client, b)?.rho(client, set_coef)
}

/// The M-03 batched reverse direction for ONE chunk: the level's kernel-slot
/// integrals (`nkslots` entries) from this chunk alone, i.e. `kint` folded
/// from zero. `weight` is indexed by CONCATENATED (padded) point.
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
    let d = PairSlotBatchDevice::new(client, b)?;
    d.zero_kint(client, 1)?;
    if b.npoints() > 0 {
        d.integrate_into_kint(client, weight)?;
    }
    Ok(d.read_kint(client, 1)?.swap_remove(0))
}

/// The most slots one instance may own in a batched launch — M-08.
///
/// The reverse batched kernels accumulate an instance's slot integrals in a
/// fixed-width register array indexed by the slot's position in its term set,
/// so an instance carrying more slots than this cannot be batched at all.
/// Hosts must partition to respect it (see the pair-level table builder's own
/// guard); [`validate_batch`] is the last line of defence, not the first.
pub const MAX_SLOTS_PER_INSTANCE: usize = 10;

/// Shape checks for a [`PairSlotBatch`] — the batched analogue of
/// [`validate`], and the same posture: an inconsistent table is a caller bug
/// and is refused rather than indexed past.
fn validate_batch(b: &PairSlotBatch) -> Result<(), AlgebraError> {
    let shape = |expected: String, actual: String| AlgebraError::ShapeMismatch { expected, actual };
    let npoints = b.point_block.len();
    if b.coords_x.len() != npoints || b.coords_y.len() != npoints || b.coords_z.len() != npoints {
        return Err(shape(
            format!("each coordinate plane has npoints = {npoints}"),
            format!(
                "{}/{}/{}",
                b.coords_x.len(),
                b.coords_y.len(),
                b.coords_z.len()
            ),
        ));
    }
    if !npoints.is_multiple_of(POINT_PAD) {
        return Err(shape(
            format!("npoints a multiple of POINT_PAD = {POINT_PAD}"),
            npoints.to_string(),
        ));
    }
    let nblocks = b.block_point0.len().saturating_sub(1);
    if b.block_inst0.len() != nblocks + 1 || b.block_point_end.len() != nblocks {
        return Err(shape(
            "block_point0 (nblocks + 1), block_inst0 (nblocks + 1) and block_point_end (nblocks)"
                .to_string(),
            format!(
                "{} / {} / {}",
                b.block_point0.len(),
                b.block_inst0.len(),
                b.block_point_end.len()
            ),
        ));
    }
    for bi in 0..nblocks {
        let p0 = b.block_point0[bi] as usize;
        let p1 = b.block_point0[bi + 1] as usize;
        let pe = b.block_point_end[bi] as usize;
        if !(p0 <= pe && pe <= p1) || !p0.is_multiple_of(POINT_PAD) || p1 > npoints {
            return Err(shape(
                "every block's point range padded to POINT_PAD with its real end inside"
                    .to_string(),
                format!("block {bi}: {p0}..{pe}..{p1}"),
            ));
        }
    }
    let nocc = b.inst_block.len();
    if b.inst_ref.len() != nocc || b.occ_slot0.len() != nocc + 1 {
        return Err(shape(
            format!("inst_ref (nocc = {nocc}) and occ_slot0 (nocc + 1)"),
            format!("{} / {}", b.inst_ref.len(), b.occ_slot0.len()),
        ));
    }
    if b.block_inst0.last().is_some_and(|&e| e as usize != nocc) {
        return Err(shape(
            format!("block_inst0 to end at nocc = {nocc}"),
            format!("{:?}", b.block_inst0.last()),
        ));
    }
    let nuinst = b.instance_alpha.len();
    if b.instance_center.len() != nuinst * 3
        || b.instance_radius2.len() != nuinst
        || b.instance_set.len() != nuinst
        || b.instance_kslot0.len() != nuinst
        || b.uocc_off.len() != nuinst + 1
    {
        return Err(shape(
            format!("per-distinct tables sized nuinst = {nuinst} (centre 3x, uocc_off + 1)"),
            format!(
                "{} / {} / {} / {} / {}",
                b.instance_center.len(),
                b.instance_radius2.len(),
                b.instance_set.len(),
                b.instance_kslot0.len(),
                b.uocc_off.len()
            ),
        ));
    }
    if b.uocc.len() != nocc || b.uocc_off.last().is_some_and(|&e| e as usize != nocc) {
        return Err(shape(
            format!("uocc to list every occurrence once (nocc = {nocc})"),
            format!("{} / {:?}", b.uocc.len(), b.uocc_off.last()),
        ));
    }
    if b.inst_ref.iter().any(|&u| u as usize >= nuinst)
        || b.uocc.iter().any(|&o| o as usize >= nocc)
    {
        return Err(shape(
            "every inst_ref < nuinst and every uocc < nocc".to_string(),
            "an out-of-range index".to_string(),
        ));
    }
    let nsets = b.set_off.len().saturating_sub(1);
    if b.set_off.first().is_some_and(|&s| s != 0)
        || b.set_off
            .last()
            .is_some_and(|&e| e as usize != b.set_pow.len())
        || b.set_off
            .windows(2)
            .any(|w| w[1] < w[0] || (w[1] - w[0]) as usize > MAX_SLOTS_PER_INSTANCE)
    {
        return Err(shape(
            format!(
                "set_off a prefix over set_pow with at most {MAX_SLOTS_PER_INSTANCE} slots per set"
            ),
            format!(
                "{:?}..{:?} over {}",
                b.set_off.first(),
                b.set_off.last(),
                b.set_pow.len()
            ),
        ));
    }
    if b.set_pow
        .iter()
        .any(|&p| p & 255 > 2 || (p >> 8) & 255 > 2 || (p >> 16) & 255 > 2)
    {
        return Err(shape(
            "every monomial power <= 2 in a batched set".to_string(),
            "a power above 2".to_string(),
        ));
    }
    if b.instance_set.iter().any(|&s| s as usize >= nsets) {
        return Err(shape(
            format!("every instance_set < nsets = {nsets}"),
            "an out-of-range set index".to_string(),
        ));
    }
    for (u, &s) in b.instance_set.iter().enumerate() {
        let nsl = (b.set_off[s as usize + 1] - b.set_off[s as usize]) as usize;
        let k0 = b.instance_kslot0[u] as usize;
        if k0 + nsl > b.nkslots {
            return Err(shape(
                format!("instance_kslot0 + set size <= nkslots = {}", b.nkslots),
                format!("distinct instance {u}: {k0} + {nsl}"),
            ));
        }
    }
    // Every occurrence's output span is its set's size, and the spans tile
    // `occ_slot0` exactly.
    if b.occ_slot0.first().is_some_and(|&s| s != 0) {
        return Err(shape(
            "occ_slot0 to start at 0".to_string(),
            format!("{:?}", b.occ_slot0.first()),
        ));
    }
    for occ in 0..nocc {
        let s = b.instance_set[b.inst_ref[occ] as usize] as usize;
        let nsl = b.set_off[s + 1] - b.set_off[s];
        if b.occ_slot0[occ + 1] - b.occ_slot0[occ] != nsl {
            return Err(shape(
                "occ_slot0 spans equal to the occurrence's set size".to_string(),
                format!("occurrence {occ}"),
            ));
        }
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
