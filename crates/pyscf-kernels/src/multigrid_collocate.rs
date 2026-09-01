//! Multigrid real-space collocation — `pyscf/pbc/dft/multigrid/multigrid.py`
//! (plan 17-11, Task 2). The two C entry points that file needs
//! (`libdft.NUMINT_fill`/`NUMINT_fill2c` at `multigrid.py:146`,
//! `libdft.NUMINT_rho_drv` at `:280`) both reduce to the SAME primitive
//! operation: evaluate a real-space Cartesian primitive Gaussian
//!
//! ```text
//! g(r; A, alpha, ix,iy,iz) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz · exp(-alpha·|r-A|²)
//! ```
//!
//! on a batch of grid points, summed over the periodic images of its centre
//! `A`. `NUMINT_fill2c` (density) then contracts these values against a
//! density matrix to get `rho(r)`; `NUMINT_rho_drv`'s adjoint (the multigrid
//! "pass2") contracts them against a weighted grid function to get an AO
//! matrix element. Both contractions are small dense linear algebra
//! (`n_pshell x n_pshell`, `n_pshell` in the tens for every reference system
//! in `.planning/phases/17-ksymm-multigrid/measurements/`) and are done on
//! the HOST in `pyscf-pbc-dft::multigrid` — exactly the split
//! `crates/pyscf-pbc-dft/src/numint.rs` already uses for the reference
//! quadrature (`eval_gto` kernel evaluates AO values; `eval_rho_one` /
//! `vxc_mat_one` contract on the host with plain rayon loops, not a second
//! kernel). This file is the ONE new kernel: the elementwise, embarrassingly
//! parallel Gaussian evaluation over `(cart slot, grid point)`.
//!
//! # Why per-PRIMITIVE ("pshell"), not per-shell
//!
//! Upstream's multigrid decontracts every contracted shell into its
//! individual primitives before assigning grid levels
//! (`h_cell.decontract_basis(to_cart=True, aggregate=True)`,
//! `multigrid.py:614-616`) — a contracted `gth-szv` shell's four primitives
//! (exponents spanning roughly two orders of magnitude, see
//! `_primitive_gto_cutoff`) can legitimately need four different mesh
//! resolutions. `pyscf_pbc_dft::multigrid::tasks` performs the same
//! decontraction and calls this kernel once per grid level with the
//! primitive ("pshell") records assigned to that level. The kernel itself
//! therefore never sees a contraction — every record is already a single
//! primitive with coefficient `coef = ctr_coeff · common_fac_sp(l)` baked in
//! host-side (`pyscf_kernels::common_fac_sp`, the same libcint `sp`-shell
//! convention `crates/pyscf-pbc-df/src/ft_ao/single.rs` already uses for the
//! reciprocal-space analogue).
//!
//! # `F: Float` and the one documented exception
//!
//! AGENTS.md §3 / RULE 5 requires every kernel here to be generic over the
//! device float. This one is NOT, for the same reason
//! `crates/pyscf-kernels/src/pbc/ft_aopair.rs` and the `exp` call sites in
//! `eval_gto.rs` are not: the only transcendental this kernel needs is
//! `exp(-alpha·r²)`, and `cube_math::double::exp` **is** the f64 libm — there
//! is no generic-`F` entry point to call. `cube_math::single` is the f32
//! seam; nothing here calls it because the CPU runtime (this workspace's
//! default backend, see `pyscf-algebra/src/launch.rs`) and the local ROCm
//! iGPU (no f64) together mean only the f64 path is exercised or measured.
//! Every OTHER operation in the kernel body (integer powers via repeated
//! multiplication, adds, the final scale) is closed under `Float` already —
//! the kernel is simply pinned to `f64` at the one call site that is not.
//!
//! # No cube barriers, no fixed-width cubes
//!
//! Per `AGENTS.md`/`17-11-PLAN.md` Task 0: this workspace's default backend
//! is CubeCL's CPU runtime, where a cube is a batch of OS threads and a
//! barrier is a real (slow) synchronisation, not a hardware no-op. This
//! kernel launches with [`pyscf_algebra::launch::launch_1d`], which sizes the
//! cube from the device rather than hard-coding 256, and has no
//! `sync_cube`/shared-memory staging at all — every lane is independent, so
//! there is nothing to synchronise.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError};

/// Flat device tables for one [`collocate`] launch — one grid level's worth of
/// primitive ("pshell") records.
#[derive(Debug, Clone, Default)]
pub struct PshellGridTable {
    /// `(ngrids, 3)` real-space grid coordinates, row-major, Bohr.
    pub coords: Vec<f64>,
    /// Per cartesian output slot, 3 entries: `ix, iy, iz`.
    pub slot_pow: Vec<u32>,
    /// Per cartesian output slot: which pshell it belongs to.
    pub slot_pshell: Vec<u32>,
    /// Per pshell: first record index (`>= 0`).
    pub pshell_rec0: Vec<u32>,
    /// Per pshell: record count (periodic images that survived screening).
    pub pshell_nrec: Vec<u32>,
    /// Per pshell: the primitive Gaussian exponent.
    pub pshell_alpha: Vec<f64>,
    /// Per pshell: `ctr_coeff * common_fac_sp(l)`, applied once per slot.
    pub pshell_coef: Vec<f64>,
    /// Per record, 3 entries: the image-shifted centre `A - L`.
    pub rec_center: Vec<f64>,
}

/// `i = slot*ngrids + g`, `g` the fast axis (adjacent threads read adjacent
/// grid points), matching `ft_aopair_kernel`'s layout convention.
///
/// `out[slot*ngrids + g] = coef · Σ_{images} (r-A_L)^pow · exp(-alpha·|r-A_L|²)`.
#[cube(launch_unchecked)]
fn collocate_kernel(
    coords: &Array<f64>,
    slot_pow: &Array<u32>,
    slot_pshell: &Array<u32>,
    pshell_rec0: &Array<u32>,
    pshell_nrec: &Array<u32>,
    pshell_alpha: &Array<f64>,
    pshell_coef: &Array<f64>,
    rec_center: &Array<f64>,
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

        let p = slot_pshell[slot] as usize;
        let alpha = pshell_alpha[p];
        let coef = pshell_coef[p];
        let r0 = pshell_rec0[p] as usize;
        let nrec = pshell_nrec[p] as usize;

        let mut acc = 0.0;
        for r in r0..(r0 + nrec) {
            let dx = x - rec_center[r * 3];
            let dy = y - rec_center[r * 3 + 1];
            let dz = z - rec_center[r * 3 + 2];
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

            let e = cube_math::double::exp::exp(0.0 - alpha * r2, cube_math::MathConfig::EXACT);
            acc += poly * e;
        }
        out[idx] = coef * acc;
    }
}

fn work_per_thread(t: &PshellGridTable) -> usize {
    let npshell = t.pshell_rec0.len().max(1);
    let nrec = t.rec_center.len() / 3;
    let avg_rec = (nrec / npshell).max(1);
    // One record costs a handful of multiplies plus one `exp`, worth roughly
    // fifty flops.
    avg_rec * 50
}

fn launch_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    h: &[Handle],
    out: &Handle,
    t: &PshellGridTable,
    nslots: usize,
    ngrids: usize,
) {
    let (count, dim) = launch_1d(client, nslots * ngrids, work_per_thread(t));
    unsafe {
        collocate_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(h[0].clone(), t.coords.len()),
            ArrayArg::from_raw_parts(h[1].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_pshell.len()),
            ArrayArg::from_raw_parts(h[3].clone(), t.pshell_rec0.len()),
            ArrayArg::from_raw_parts(h[4].clone(), t.pshell_nrec.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.pshell_alpha.len()),
            ArrayArg::from_raw_parts(h[6].clone(), t.pshell_coef.len()),
            ArrayArg::from_raw_parts(h[7].clone(), t.rec_center.len()),
            ArrayArg::from_raw_parts(out.clone(), nslots * ngrids),
            nslots,
            ngrids,
        );
    }
}

fn upload_u32<R: Runtime>(client: &ComputeClient<R>, data: &[u32]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(data))
}

fn launch<R: Runtime>(t: &PshellGridTable, client: &ComputeClient<R>) -> Vec<f64> {
    let ngrids = t.coords.len() / 3;
    let nslots = t.slot_pow.len() / 3;
    let n_out = ngrids * nslots;

    let zeros = vec![0.0f64; n_out];
    let out_h = upload::<R, f64>(client, &zeros);
    drop(zeros);

    let h = vec![
        upload::<R, f64>(client, &t.coords),
        upload_u32::<R>(client, &t.slot_pow),
        upload_u32::<R>(client, &t.slot_pshell),
        upload_u32::<R>(client, &t.pshell_rec0),
        upload_u32::<R>(client, &t.pshell_nrec),
        upload::<R, f64>(client, &t.pshell_alpha),
        upload::<R, f64>(client, &t.pshell_coef),
        upload::<R, f64>(client, &t.rec_center),
    ];
    launch_on_handles::<R>(client, &h, &out_h, t, nslots, ngrids);
    let bytes = client.read(vec![out_h]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

/// The multigrid collocation kernel's public entry point.
///
/// Returns a dense row-major `(n_cart_slots, ngrids)` buffer: slot `s`'s
/// value at every grid point. `n_cart_slots = t.slot_pow.len() / 3`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if the per-slot / per-pshell / per-record
/// tables disagree in length. An empty table or an empty grid returns an
/// empty vector without launching.
pub fn collocate(client: &AlgebraClient, t: &PshellGridTable) -> Result<Vec<f64>, AlgebraError> {
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
    if t.slot_pshell.len() != nslots {
        return Err(shape(
            "slot_pshell.len() == nslots",
            format!("nslots {nslots}, slot_pshell {}", t.slot_pshell.len()),
        ));
    }
    let npshell = t.pshell_rec0.len();
    if t.pshell_nrec.len() != npshell
        || t.pshell_alpha.len() != npshell
        || t.pshell_coef.len() != npshell
    {
        return Err(shape(
            "pshell_rec0 / pshell_nrec / pshell_alpha / pshell_coef all of length npshell",
            format!("npshell {npshell}"),
        ));
    }
    if let Some(bad) = t.slot_pshell.iter().find(|&&p| (p as usize) >= npshell) {
        return Err(shape(
            &format!("every slot_pshell < npshell = {npshell}"),
            format!("{bad}"),
        ));
    }
    if !t.rec_center.len().is_multiple_of(3) {
        return Err(shape(
            "rec_center length a multiple of 3",
            format!("{}", t.rec_center.len()),
        ));
    }
    let nrec = t.rec_center.len() / 3;
    let claimed: usize = t
        .pshell_nrec
        .iter()
        .map(|&n| n as usize)
        .sum::<usize>()
        .max(
            t.pshell_rec0
                .iter()
                .zip(&t.pshell_nrec)
                .map(|(&r0, &n)| r0 as usize + n as usize)
                .max()
                .unwrap_or(0),
        );
    if npshell > 0 && claimed > nrec {
        return Err(shape(
            "every pshell_rec0+pshell_nrec within rec_center",
            format!("nrec {nrec}, claimed up to {claimed}"),
        ));
    }
    let ngrids = t.coords.len() / 3;
    if nslots == 0 || ngrids == 0 {
        return Ok(Vec::new());
    }
    let out = dispatch_backend!(client, c, Rt, launch::<Rt>(t, c));
    Ok(out)
}
