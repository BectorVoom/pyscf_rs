//! Periodic AO evaluation — `cell.pbc_eval_gto` / `eval_ao_kpts`.
//!
//! Port of `pyscf/pbc/gto/eval_gto.py:32-192` (`eval_gto`, `_estimate_rcut`)
//! and `cell.py:2043-2053` (the `Cell.eval_gto` dispatch).
//!
//! # What it computes
//!
//! ```text
//! ao_k[k][g, mu] = Σ_L exp(i·k·L) · phi_mu(coords[g] − L)
//! ```
//!
//! i.e. the Bloch sum of the molecular AO. Upstream shifts the ATOM by `+L`
//! (`grid_ao.c`); evaluating the unshifted molecular AO at `coords − L` is the
//! same function of the same argument, and it lets this port reuse the existing
//! `pyscf_kernels::eval_gto` (2 564 lines of s/p/d + deriv1 kernels) verbatim.
//! Plan 10-04 is explicit that no new AO evaluator may be written here.
//!
//! # Conventions
//!
//! * Phase `exp(+i·k·L)`, `eval_gto.py:139` (`expLk = exp(1j·Ls·kptsᵀ)`),
//!   the same sign the 1-electron driver uses (K-07).
//! * A gamma k-point drops its imaginary plane, `eval_gto.py:157-158`.
//! * Output is F-order per component, `values[c*ngrids*nao + g + mu*ngrids]`,
//!   the SAME layout [`pyscf_gto::EvalGtoOutput`] uses (upstream transposes to
//!   `(ngrids, nao)` C-order at the end; the layout note in
//!   [`crate::pbc_intor`] applies here too).
//!
//! # Image list
//!
//! `eval_gto.py:137` uses a grid-edge-aware `get_lattice_Ls` that keeps images
//! able to reach the GRID BOX rather than another atom. This port instead calls
//! [`crate::lattice::get_lattice_ls`] with `discard = false`, whose raw
//! `cartesian_prod` box is a SUPERSET of upstream's mask: it can only add
//! numerically-negligible images, never drop a needed one, so the Bloch sum
//! stays converged for grid points anywhere in the cell (which
//! `bloch_periodicity_holds` pins).
//!
//! # AO screening (W-09, `.planning/pbc/KRKS-OPTIMISATION-PLAN.md`)
//!
//! The image list above is a bounding BOX, so most `(image, grid block)` pairs
//! are numerically zero: an image `L` in a far corner of the box is outside
//! every shell's `rcut` for every grid point. Evaluating them costs a full
//! `eval_gto` sweep over the whole grid and contributes nothing.
//!
//! [`screen_blocks`] therefore computes, per image, which `BLKSIZE`-sized grid
//! blocks any shell can reach — upstream's `non0tab` / `make_screen_index`
//! (`gto/eval_gto.py:155`) at the same block granularity, and against the same
//! per-shell `rcut` that [`estimate_rcut_for_eval`] already derives from
//! `cell.precision`. An image with no surviving block is skipped outright.
//!
//! **The screen decides which images and blocks are LAUNCHED at block
//! granularity.** W-09 stopped there, reasoning that a per-element skip is the
//! data-dependent branch `plane_alignment.md` warns about. Session 3 then
//! measured what the launched blocks still pay for (`krks_profile ao`, si
//! 2×2×2 mesh 31): the kept blocks cover 70 % of the grid on 454 launched
//! images, i.e. ~316 `(image, point)` evaluations per grid point, against the
//! ~50-75 a shell's cutoff sphere actually contains. A-04 (session 4) adds the
//! per-point test INSIDE the kernels — one compare per `(point, shell)` lane
//! against the same `rcut²`, on spatially adjacent lanes (so a plane diverges
//! only along the sphere's surface), in exchange for the shell's primitives
//! and angular work. `PYSCF_PBC_AO_POINT_SCREEN=0` restores block-only, and
//! the A/B is measured, not argued.
//!
//! **This DROPS TERMS, so it changes the result.** The dropped mass is bounded
//! by the same `precision` that sized the image list, and
//! `tests/eval_ao_screen.rs` pins that: screened vs unscreened agree to well
//! inside the KRKS gate, and the screen is convergent in `rcut`.
//! `PYSCF_PBC_AO_SCREEN=0` turns it off for bisection.

use crate::cell::Cell;
use crate::cutoff::{PgtoOp, extract_pgto_params};
use crate::pbc_intor::is_gamma;
use pyscf_algebra::{AlgebraClient, CTensor, select_backend};
use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
use pyscf_core::{CoreError, PyscfRsError};
use std::f64::consts::PI;

/// k-resolved AO values on a grid.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalAoKptsOutput {
    /// One planar-complex buffer per k-point, each `comp * ngrids * nao` long,
    /// F-order per component (see the module docs).
    pub kaos: Vec<CTensor>,
    /// Grid-point count.
    pub ngrids: usize,
    /// AO count.
    pub nao: usize,
    /// Component count — 1 for `GTOval_sph`, 4 for `GTOval_sph_deriv1`, …
    pub comp: usize,
    /// `true` for every k-point whose imaginary plane was dropped.
    pub gamma: Vec<bool>,
}

impl EvalAoKptsOutput {
    /// The AO block at k-point `k`.
    pub fn at(&self, k: usize) -> &CTensor {
        &self.kaos[k]
    }

    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.kaos.len()
    }

    /// `(re, im)` of component `c` of AO `mu` at grid point `g`, k-point `k`.
    pub fn element(&self, k: usize, c: usize, g: usize, mu: usize) -> (f64, f64) {
        let p = c * self.ngrids * self.nao + g + mu * self.ngrids;
        (self.kaos[k].re[p], self.kaos[k].im[p])
    }
}

/// `_estimate_rcut(cell, deriv)` — `eval_gto.py:171-192`.
///
/// One radius per shell: how far that shell's most diffuse primitive reaches
/// before falling under the grid-weighted precision. `deriv` is the number of
/// `ip` factors in the eval name (upstream counts the substring `'ip'`).
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when `rcut` has to be estimated and cannot.
pub fn estimate_rcut_for_eval(cell: &Cell, deriv: u32) -> Result<Vec<f64>, PyscfRsError> {
    let (es, cs) = extract_pgto_params(cell, PgtoOp::Min);
    let ls: Vec<f64> = (0..cell.mol.nbas)
        .map(|i| crate::cutoff::bas_angular(cell, i) as f64)
        .collect();

    let vol = cell.vol();
    let rcut = cell.try_rcut()?;
    // eval_gto.py:177-183 — the grid-weight penalty and the lattice-sum surface.
    let weight_penalty = vol;
    let rad = vol.powf(-1.0 / 3.0) * rcut + 1.0;
    let surface = 4.0 * PI * rad * rad;
    let precision = cell.precision / (weight_penalty * surface).max(1.0);

    let mut out = Vec::with_capacity(es.len());
    for ((e, c), l) in es.iter().zip(cs.iter()).zip(ls.iter()) {
        let norm_ang = ((2.0 * l + 1.0) / (4.0 * PI)).sqrt();
        let fac = 2.0 * PI / vol * c * norm_ang / e / precision;
        // Two fixed-point sweeps from r = cell.rcut, exactly as upstream.
        let mut r = rcut;
        for _ in 0..2 {
            let t = fac * r.powf(l + 1.0) * (2.0 * e * r).powi(deriv as i32) + 1.0;
            r = (t.ln() / e).sqrt();
        }
        out.push(r);
    }
    Ok(out)
}

/// Number of `ip` derivative factors in an eval name — upstream's
/// `eval_name.count('ip')` (`eval_gto.py:134`).
fn deriv_count(eval_name: &str) -> u32 {
    eval_name.matches("ip").count() as u32
}

/// `cell.pbc_eval_gto(eval_name, coords, kpts)` — `eval_gto.py:32-167`.
///
/// `eval_name` is a MOLECULAR name (`"GTOval_sph"`, `"GTOval_sph_deriv1"`, …);
/// the `"PBC"` prefix upstream prepends selects its periodic C driver and has no
/// analogue here. An empty `kpts` means the single gamma point.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] — unbuilt cell, or an eval name
///   [`pyscf_gto::eval_gto`] does not know.
/// * [`PyscfRsError::NotYetImplemented`] — an eval variant the molecular kernel
///   defers (deriv2, GIAO).
pub fn eval_ao_kpts(
    cell: &Cell,
    eval_name: &str,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
) -> Result<EvalAoKptsOutput, PyscfRsError> {
    if !cell.mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "eval_ao_kpts: the cell must be built first".into(),
        )));
    }
    let rcut = estimate_rcut_for_eval(cell, deriv_count(eval_name))?;
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    // `discard = false` — see the module docs on the image list.
    let ls = crate::lattice::get_lattice_ls(cell, Some(rmax), None, false)?;
    eval_ao_kpts_with_images(cell, eval_name, coords, kpts, &ls)
}

/// [`eval_ao_kpts`] against a caller-supplied image list.
///
/// Exposed for the same reason [`crate::pbc_intor::intor_cross_with_images`] is:
/// callers that evaluate several eval names over one grid should build `Ls`
/// once, and the `rcut`-convergence test needs to vary it deliberately.
///
/// # Errors
/// As [`eval_ao_kpts`].
pub fn eval_ao_kpts_with_images(
    cell: &Cell,
    eval_name: &str,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    ls: &[[f64; 3]],
) -> Result<EvalAoKptsOutput, PyscfRsError> {
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = kpts.len();
    let ngrids = coords.len();
    let nao = cell.mol.nao_nr;
    // One span per cold evaluation, so a driver-level profile can count how
    // many AO tables an SCF builds and at how many k-points each.
    let _call_span = tracing::info_span!(
        "pbc_eval_ao_kpts",
        nkpts = nkpts as u64,
        ngrids = ngrids as u64,
        nao = nao as u64,
        eval_name
    )
    .entered();

    let client = select_backend()
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "eval_ao_kpts: backend selection failed: {e}"
            )))
        })?
        .client;

    // K-07 — the same `exp(+i k·L)` table the 1-electron driver uses.
    let kflat: Vec<f64> = kpts.iter().flatten().copied().collect();
    let lflat: Vec<f64> = ls.iter().flatten().copied().collect();
    let (expkl_re, expkl_im) =
        pyscf_kernels::pbc::bloch_phase(&client, &kflat, &lflat).map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "eval_ao_kpts: K-07 bloch_phase failed: {e}"
            )))
        })?;
    let nimgs = ls.len();

    // The component count is whatever the molecular evaluator reports; a zero
    // grid still has to produce correctly-shaped (empty) output, so probe with
    // the real grid and accept the cost — `eval_gto` is called nimgs times
    // anyway.
    let mut n = 0usize;
    let mut comp = 1usize;
    // quick-260826-spd: the k-resolved accumulators live on the DEVICE for the
    // whole image loop. They used to be host `Vec`s handed to K-08 by value and
    // returned fresh, which meant both `(nkpts, n)` planes were uploaded and read
    // back once per lattice image — `4*nkpts*n` reals of round-trip traffic to
    // fold in `n` reals of new AO data, repeated `nimgs` times. Now only the AO
    // block and the `2*nkpts` phase factors cross per image, and the planes come
    // home once, after the loop.
    let mut acc: Option<pyscf_kernels::pbc::AoKAccumulator> = None;
    // A-02: both coordinate and index workspaces are reused for every image.
    // The device evaluator uploads/copies from these slices before returning,
    // so clearing them for the next image cannot alias a queued kernel input.
    let mut shifted_workspace = Vec::with_capacity(3 * ngrids);
    let mut index_workspace = Vec::with_capacity(ngrids);

    // W-09: the per-image block screen. Built once, outside the image loop,
    // because both the block boxes and the shell radii are image-independent.
    // `screen` is `None` when screening is off, and then the loop below is
    // exactly the pre-W-09 one.
    let screen: Option<(Vec<BlockBox>, Vec<[f64; 3]>, Vec<f64>)> = if ao_screen_enabled() {
        let rcut = estimate_rcut_for_eval(cell, deriv_count(eval_name))?;
        // Squared, so the per-(image, block, shell) test needs no sqrt.
        let rcut2: Vec<f64> = rcut.iter().map(|r| r * r).collect();
        let centres = shell_centres(cell);
        if centres.len() == rcut2.len() {
            Some((block_boxes(coords), centres, rcut2))
        } else {
            // `estimate_rcut_for_eval` returns one radius per SHELL; if that
            // ever stops matching `nbas` the screen would silently mis-pair
            // radii with centres, so refuse to screen rather than guess.
            tracing::warn!(
                shells = centres.len(),
                radii = rcut2.len(),
                "eval_ao_kpts: per-shell rcut count does not match nbas; \
                 W-09 AO screening disabled for this call"
            );
            None
        }
    } else {
        None
    };
    // A-04: the same per-shell radius, applied inside the kernel per grid
    // point. Only meaningful with the block screen on — the two switches
    // share one radius, and the per-point test alone would still launch every
    // image. `PYSCF_PBC_AO_POINT_SCREEN=0` pins the block-only behaviour.
    let point_rcut2: Option<&[f64]> = match &screen {
        Some((_, _, rcut2)) if ao_point_screen_enabled() => Some(rcut2.as_slice()),
        _ => None,
    };

    // K-09 (session 4): how many images share one accumulate launch. `1` is
    // the per-image path above (the kill switch, and the bit-identity
    // reference); more needs the block size up front, which the two device
    // eval names provide. Anything else stays per-image.
    let mut batch: Option<pyscf_kernels::pbc::AoImageBatch> = None;
    // A-05/A-06 (session 5): the image-invariant operands uploaded once, and
    // the AO evaluation itself launched once per `eval_batch` images straight
    // into the K-09 slots. `eval_batch == 0` is the per-image evaluation
    // (`eval_gto_device_target`, the pre-A-06 kernels) — the reference arm.
    let mut eval_ctx: Option<pyscf_kernels::EvalGtoDeviceContext> = None;
    let mut eval_batch = 0usize;
    let mut eval_coords: Vec<f64> = Vec::new();
    let mut eval_images: Vec<pyscf_kernels::EvalGtoImage> = Vec::new();
    let mut eval_first_slot = 0usize;
    if let Some(comp_expected) = comp_of_eval_name(eval_name) {
        let n_expected = comp_expected * ngrids * nao;
        let capacity = image_batch_capacity(n_expected);
        if capacity > 1 && n_expected > 0 && pyscf_kernels::eval_gto_device_capable(&cell.mol._bas)
        {
            n = n_expected;
            comp = comp_expected;
            batch = Some(pyscf_kernels::pbc::AoImageBatch::new(
                &client, capacity, n_expected, ngrids, nkpts,
            ));
            eval_batch = eval_batch_size(capacity);
            if eval_batch > 0 {
                eval_ctx = Some(pyscf_kernels::EvalGtoDeviceContext::new(
                    &client,
                    &cell.mol._atm,
                    &cell.mol._bas,
                    &cell.mol._env,
                    &cell.mol.ao_loc_nr,
                    nao,
                    point_rcut2,
                )?);
                eval_coords.reserve(3 * ngrids * eval_batch);
            }
        }
    }
    let deriv1 = comp == 4;

    // K-10 (session 5, user-authorised over plan 10-04): the fused path — no
    // AO block is ever written or read. Engages on the K-09-capable bases
    // except all-s at deriv 0 (the s-kernel's arithmetic is its own), sized
    // so `B · Q_max` fits the lane's value array. `PYSCF_PBC_AO_FUSE=0` pins
    // the K-09/A-06 path (the bit-identity reference for the gate).
    let mut fused: Option<FusedState> = None;
    if let (Some(ctx), true) = (eval_ctx.as_ref(), ao_fuse_enabled()) {
        let all_s = cell
            .mol
            ._bas
            .chunks_exact(BAS_SLOTS)
            .all(|row| row[pyscf_core::raw_layout::ANG_OF] == 0);
        let qmax = pyscf_kernels::fused_values_per_image(&cell.mol._bas, deriv1);
        if !(all_s && !deriv1) && qmax > 0 && qmax <= pyscf_kernels::FUSED_VALS_CAP {
            let cap = (pyscf_kernels::FUSED_VALS_CAP / qmax)
                .clamp(1, pyscf_kernels::AO_FUSED_BATCH_MAX)
                .min(fused_batch_override().unwrap_or(usize::MAX))
                .max(1);
            let mut flat = Vec::with_capacity(3 * ngrids);
            for axis in 0..3 {
                flat.extend(coords.iter().map(|r| r[axis]));
            }
            let grid = pyscf_kernels::AoGridDevice::new(&client, &flat, ngrids);
            let _ = ctx;
            fused = Some(FusedState {
                grid,
                cap,
                images: Vec::with_capacity(cap),
                pr: Vec::with_capacity(cap * nkpts),
                pi: Vec::with_capacity(cap * nkpts),
            });
        }
    }

    for (m, l) in ls.iter().enumerate() {
        // phi(r − L): shift the GRID, not the atoms — same function, and it
        // keeps the molecular evaluator and its `Mole` untouched.
        let keep = match &screen {
            None => None,
            Some((boxes, centres, rcut2)) => {
                match screen_one_image(boxes, centres, rcut2, *l) {
                    // No block of the grid is within any shell's rcut of this
                    // image: it contributes nothing anywhere. This is the
                    // skip that pays for the whole item.
                    None => continue,
                    // K-08b (session 3): every block kept means the "sub-grid"
                    // IS the grid, in grid order — so take the dense path
                    // (contiguous shift, the vectorised K-08) instead of
                    // gathering every point and scatter-accumulating it back.
                    // The dense kernel adds the same `phase_k · ao[p]` to the
                    // same `(k, p)`; `eval_ao_stages` asserts the identity.
                    Some(keep) if dense_full_images_enabled() && keep.iter().all(|&b| b) => None,
                    some => some,
                }
            }
        };

        if let (Some(f), Some(ctx)) = (fused.as_mut(), eval_ctx.as_ref()) {
            // K-10: stage the image's lattice vector, block flags and phases;
            // the kernel does the shift, the evaluation and the accumulate.
            f.images.push(pyscf_kernels::FusedImage {
                l: *l,
                keep_blocks: match &keep {
                    None => Vec::new(),
                    Some(k) => k.iter().map(|&b| u32::from(b)).collect(),
                },
            });
            f.pr.extend((0..nkpts).map(|k| expkl_re[k * nimgs + m]));
            f.pi.extend((0..nkpts).map(|k| expkl_im[k * nimgs + m]));
            if f.images.len() >= f.cap {
                flush_fused(&client, ctx, f, &mut acc, nkpts, n, deriv1, m)?;
            }
            continue;
        }

        // The shift/gather half: `shifted_workspace` holds the image's grid
        // (all of it, or the kept blocks) in F-order; `scatter_index` its
        // kept-point list when gathered.
        let scatter_index: Option<&[usize]> = match &keep {
            None => {
                let span = tracing::info_span!("pbc_eval_ao_shift_pack");
                let _entered = span.enter();
                shifted_workspace.clear();
                for axis in 0..3 {
                    shifted_workspace.extend(coords.iter().map(|r| r[axis] - l[axis]));
                }
                None
            }
            Some(keep) => {
                let span = tracing::info_span!("pbc_eval_ao_shift_pack");
                let _entered = span.enter();
                gather_kept(
                    coords,
                    keep,
                    *l,
                    &mut shifted_workspace,
                    &mut index_workspace,
                );
                Some(index_workspace.as_slice())
            }
        };
        let image_ngrids = scatter_index.map_or(ngrids, <[usize]>::len);

        // K-08 phases for this image — `exp(i·k·L)`, one per k-point.
        let pr: Vec<f64> = (0..nkpts).map(|k| expkl_re[k * nimgs + m]).collect();
        let pi: Vec<f64> = (0..nkpts).map(|k| expkl_im[k * nimgs + m]).collect();

        if let Some(batch) = batch.as_mut() {
            // K-09 (session 4): evaluate straight into the batch's next slot and
            // register the image; the accumulate happens once per full batch.
            // `n`/`comp` are known up front on this path (`comp_of_eval_name`).
            let len = comp * image_ngrids * nao;
            let shape = if comp == 1 {
                vec![image_ngrids, nao]
            } else {
                vec![comp, image_ngrids, nao]
            };
            if let Some(ctx) = eval_ctx.as_ref() {
                // A-06: stage this image's grid; the launch covers `eval_batch`
                // images at once (or the rest of the accumulate batch).
                if eval_images.is_empty() {
                    eval_first_slot = batch.len();
                }
                eval_coords.extend_from_slice(&shifted_workspace);
                eval_images.push(pyscf_kernels::EvalGtoImage { npts: image_ngrids });
                batch.push(scatter_index, &pr, &pi).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "eval_ao_kpts: K-09 batch push at image {m}: {e}"
                    )))
                })?;
                if eval_images.len() >= eval_batch || batch.is_full() {
                    let span = tracing::info_span!(
                        "pbc_eval_ao_eval_gto",
                        points = eval_images.iter().map(|i| i.npts as u64).sum::<u64>(),
                        images = eval_images.len() as u64
                    );
                    let _entered = span.enter();
                    pyscf_kernels::eval_gto_batch_into_image_batch(
                        &client,
                        ctx,
                        deriv1,
                        &eval_coords,
                        &eval_images,
                        eval_first_slot,
                        batch,
                    )?;
                    eval_coords.clear();
                    eval_images.clear();
                }
                if batch.is_full() {
                    flush_image_batch(&client, batch, &mut acc, nkpts, n, ngrids, nao, comp, m)?;
                }
                continue;
            }
            {
                let span =
                    tracing::info_span!("pbc_eval_ao_eval_gto", points = image_ngrids as u64);
                let _entered = span.enter();
                let target = batch.slot(len, shape).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "eval_ao_kpts: K-09 batch slot at image {m}: {e}"
                    )))
                })?;
                eval_gto_device_target(
                    &client,
                    cell,
                    eval_name,
                    &shifted_workspace,
                    image_ngrids,
                    point_rcut2,
                    &target,
                )?;
            }
            batch.push(scatter_index, &pr, &pi).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_ao_kpts: K-09 batch push at image {m}: {e}"
                )))
            })?;
            if batch.is_full() {
                flush_image_batch(&client, batch, &mut acc, nkpts, n, ngrids, nao, comp, m)?;
            }
            continue;
        }

        let ao_device = {
            // `points` — how many grid points this launch covers, so the A-00
            // instrument can report the launched-image count and the
            // kept-point total (the screen's actual yield).
            let span = tracing::info_span!("pbc_eval_ao_eval_gto", points = image_ngrids as u64);
            let _entered = span.enter();
            eval_gto_device(
                &client,
                cell,
                eval_name,
                &shifted_workspace,
                image_ngrids,
                point_rcut2,
            )?
        };

        // The evaluator reports its own layout — `[ngrids, nao]`, or
        // `[comp, ngrids, nao]` for the derivative variants. Taking `comp`
        // from the shape rather than dividing the buffer length means a block
        // that is short a component is an error here, not a silently
        // truncated AO array downstream.
        let image_comp = match ao_device.shape() {
            [g, a] if *g == image_ngrids && *a == nao => 1,
            [c, g, a] if *g == image_ngrids && *a == nao => *c,
            other => {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_ao_kpts: image {m} produced an AO block of shape {other:?}, expected \
                     [{image_ngrids}, {nao}] or [comp, {image_ngrids}, {nao}]",
                ))));
            }
        };
        let image_n = image_comp * ngrids * nao;
        if n == 0 {
            n = image_n;
            comp = image_comp;
        } else if image_comp != comp {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "eval_ao_kpts: image {m} produced {image_comp} AO components, an earlier image \
                 produced {comp}",
            ))));
        }
        if n == 0 {
            continue;
        }

        // Measurement arm ONLY (`PYSCF_PBC_AO_SKIP_K08=1`, session 4): drop
        // the image's AO block without accumulating it, so the cold pass can
        // be timed without K-08. The output is garbage (zeros).
        if skip_k08_for_measurement() {
            if acc.is_none() {
                acc = Some(pyscf_kernels::pbc::AoKAccumulator::zeros(&client, nkpts, n));
            }
            continue;
        }
        // K-08 — one launch per image, folding this image into every k at once,
        // in place on the device-resident accumulators.
        // Built on the first image that actually has AO values, so `n` is known;
        // `get_or_insert_with` keeps that lazy without an unreachable panic
        // branch (FOUND-07 — no `unwrap`/`expect` in production code).
        {
            let span = tracing::info_span!("pbc_eval_ao_k08_accumulate");
            let _entered = span.enter();
            let accumulator = acc.get_or_insert_with(|| {
                pyscf_kernels::pbc::AoKAccumulator::zeros(&client, nkpts, n)
            });
            let result = if let Some(index) = scatter_index {
                accumulator.accumulate_device_scatter(
                    &client, &ao_device, index, ngrids, nao, comp, &pr, &pi,
                )
            } else {
                accumulator.accumulate_device(&client, &ao_device, &pr, &pi)
            };
            result.map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_ao_kpts: K-08 accumulate failed at image {m}: {e}"
                )))
            })?;
        }
    }
    // K-10: the ragged last fused batch.
    if let (Some(f), Some(ctx)) = (fused.as_mut(), eval_ctx.as_ref()) {
        if !f.images.is_empty() {
            flush_fused(&client, ctx, f, &mut acc, nkpts, n, deriv1, nimgs)?;
        }
    }
    // K-09: the ragged last batch (A-06: its staged images evaluated first).
    if let Some(batch) = batch.as_mut() {
        if let (Some(ctx), false) = (eval_ctx.as_ref(), eval_images.is_empty()) {
            let span = tracing::info_span!(
                "pbc_eval_ao_eval_gto",
                points = eval_images.iter().map(|i| i.npts as u64).sum::<u64>(),
                images = eval_images.len() as u64
            );
            let _entered = span.enter();
            pyscf_kernels::eval_gto_batch_into_image_batch(
                &client,
                ctx,
                deriv1,
                &eval_coords,
                &eval_images,
                eval_first_slot,
                batch,
            )?;
            eval_coords.clear();
            eval_images.clear();
        }
        if !batch.is_empty() {
            flush_image_batch(&client, batch, &mut acc, nkpts, n, ngrids, nao, comp, nimgs)?;
        }
    }

    // W-09: every image may have been screened out (an empty basis, or a grid
    // nothing can reach). `n` is then still 0 and the split below yields the
    // correctly-shaped empty planes, exactly as an empty image list does.
    if n == 0 {
        comp = 1;
    }

    // One read-back for the whole lattice sum. An empty image list never built
    // an accumulator, and `n` is then 0, so the split below yields no planes.
    // K-10v: read back per k — a point-major (fused) accumulator is gathered
    // into per-k planes on the way, a k-major one is split.
    let planes = match acc {
        Some(a) => a.into_k_planes(&client),
        None => Vec::new(),
    };

    // One CTensor per k, dropping the imaginary plane at gamma
    // (eval_gto.py:157-158).
    let gamma: Vec<bool> = kpts.iter().map(is_gamma).collect();
    let mut kaos = Vec::with_capacity(nkpts);
    for (k, is_g) in gamma.iter().enumerate() {
        let (re, im) = match planes.get(k) {
            Some((re, im)) => (re.clone(), if *is_g { vec![0.0; n] } else { im.clone() }),
            None => (Vec::new(), Vec::new()),
        };
        kaos.push(CTensor::from_planes(re, im));
    }

    Ok(EvalAoKptsOutput {
        kaos,
        ngrids,
        nao,
        comp,
        gamma,
    })
}

/// The component count of the two eval names the device kernels serve —
/// what K-09 needs before the first image is evaluated. `None` for anything
/// else (the host fallback decides its own shape).
fn comp_of_eval_name(eval_name: &str) -> Option<usize> {
    match eval_name {
        "GTOval" | "GTOval_sph" => Some(1),
        "GTOval_sph_deriv1" => Some(4),
        _ => None,
    }
}

/// K-09's images per accumulate launch. `PYSCF_PBC_AO_IMAGE_BATCH=<n>` pins it
/// (`1` = one launch per image, the pre-K-09 path); unset, the batch buffer is
/// sized under [`AO_IMAGE_BATCH_BUDGET_BYTES`] and capped at
/// `AO_IMAGE_BATCH_MAX`.
fn image_batch_capacity(block_len: usize) -> usize {
    if let Some(v) = std::env::var("PYSCF_PBC_AO_IMAGE_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return v.clamp(1, pyscf_kernels::pbc::AO_IMAGE_BATCH_MAX);
    }
    let block_bytes = block_len.saturating_mul(core::mem::size_of::<f64>()).max(1);
    (AO_IMAGE_BATCH_BUDGET_BYTES / block_bytes).clamp(1, pyscf_kernels::pbc::AO_IMAGE_BATCH_MAX)
}

/// K-10's per-call staging: the resident unshifted grid and the images
/// collected for the next fused launch.
struct FusedState {
    grid: pyscf_kernels::AoGridDevice,
    cap: usize,
    images: Vec<pyscf_kernels::FusedImage>,
    pr: Vec<f64>,
    pi: Vec<f64>,
}

/// `PYSCF_PBC_AO_FUSE`, per call. `0`/`false`/`no`/`off` pins the K-09/A-06
/// path (the reference arm of `tests/eval_ao_image_batch.rs`); anything
/// else, including unset, takes the fused K-10 kernel.
fn ao_fuse_enabled() -> bool {
    !std::env::var("PYSCF_PBC_AO_FUSE").is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// `PYSCF_PBC_AO_FUSE_BATCH=<n>` caps the fused batch (measurement dial).
fn fused_batch_override() -> Option<usize> {
    std::env::var("PYSCF_PBC_AO_FUSE_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// K-10: one fused launch over the staged images, then reset the staging.
#[allow(clippy::too_many_arguments)]
fn flush_fused(
    client: &AlgebraClient,
    ctx: &pyscf_kernels::EvalGtoDeviceContext,
    f: &mut FusedState,
    acc: &mut Option<pyscf_kernels::pbc::AoKAccumulator>,
    nkpts: usize,
    n: usize,
    deriv1: bool,
    m: usize,
) -> Result<(), PyscfRsError> {
    let span = tracing::info_span!("pbc_eval_ao_k08_accumulate", images = f.images.len() as u64);
    let _entered = span.enter();
    // K-10v: point-major, so the kernel's k-loop is a vector.
    let accumulator = acc.get_or_insert_with(|| {
        pyscf_kernels::pbc::AoKAccumulator::zeros_point_major(client, nkpts, n)
    });
    if !skip_k08_for_measurement() {
        pyscf_kernels::eval_ao_k_fused_batch(
            client,
            ctx,
            &f.grid,
            deriv1,
            &f.images,
            &f.pr,
            &f.pi,
            accumulator,
            nkpts,
            SCREEN_BLKSIZE,
        )
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "eval_ao_kpts: K-10 fused launch failed at image {m}: {e}"
            )))
        })?;
    }
    f.images.clear();
    f.pr.clear();
    f.pi.clear();
    Ok(())
}

/// A-06's images per AO evaluation launch. `PYSCF_PBC_AO_EVAL_BATCH=<n>`
/// pins it: `0` is one launch per image through the pre-A-06 kernels (the
/// reference arm), `n >= 1` is the batched kernel over `min(n, capacity)`
/// images (so `1` isolates the hoisted uploads from the collapsed launches).
/// Unset: the whole accumulate batch in one launch.
fn eval_batch_size(capacity: usize) -> usize {
    match std::env::var("PYSCF_PBC_AO_EVAL_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(n) => n.min(capacity),
        None => capacity,
    }
}

/// The most device memory K-09 spends on staged AO blocks. 256 MiB — the same
/// per-launch budget the multigrid batches use (`BATCH_BUDGET_BYTES`); at
/// `si gth-dzvp deriv 1 mesh 31` a block is 25 MB, so ten images share a
/// launch, and at gth-szv sixteen (the cap).
pub const AO_IMAGE_BATCH_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// K-09: fold the registered images into the accumulator in one launch and
/// empty the batch. `m` names the image just registered, for the error.
#[allow(clippy::too_many_arguments)]
fn flush_image_batch(
    client: &AlgebraClient,
    batch: &mut pyscf_kernels::pbc::AoImageBatch,
    acc: &mut Option<pyscf_kernels::pbc::AoKAccumulator>,
    nkpts: usize,
    n: usize,
    ngrids: usize,
    nao: usize,
    comp: usize,
    m: usize,
) -> Result<(), PyscfRsError> {
    let span = tracing::info_span!("pbc_eval_ao_k08_accumulate", images = batch.len() as u64);
    let _entered = span.enter();
    let accumulator =
        acc.get_or_insert_with(|| pyscf_kernels::pbc::AoKAccumulator::zeros(client, nkpts, n));
    if !skip_k08_for_measurement() {
        accumulator
            .accumulate_batch(client, batch, ngrids, nao, comp)
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_ao_kpts: K-09 accumulate failed at image {m}: {e}"
                )))
            })?;
    }
    batch.clear();
    Ok(())
}

/// [`eval_gto_device`] writing into a caller-owned slot (K-09). Only the two
/// device eval names reach here (`comp_of_eval_name`).
#[allow(clippy::too_many_arguments)]
fn eval_gto_device_target(
    client: &AlgebraClient,
    cell: &Cell,
    eval_name: &str,
    flat: &[f64],
    ngrids: usize,
    rcut2: Option<&[f64]>,
    target: &pyscf_kernels::AoBlockDevice,
) -> Result<(), PyscfRsError> {
    match eval_name {
        "GTOval" | "GTOval_sph" => pyscf_kernels::eval_gto_sph_into_target(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
            rcut2,
            target,
        ),
        "GTOval_sph_deriv1" => pyscf_kernels::eval_gto_sph_deriv1_into_target(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
            rcut2,
            target,
        ),
        other => Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "eval_ao_kpts: K-09 has no device target path for {other:?}"
        )))),
    }
}

/// A-04 (session 4): `rcut2` is the per-shell squared cutoff the W-09 block
/// screen was built from, handed to the device kernels so they apply the SAME
/// radius per grid point; `None` is the unscreened kernel.
fn eval_gto_device(
    client: &AlgebraClient,
    cell: &Cell,
    eval_name: &str,
    flat: &[f64],
    ngrids: usize,
    rcut2: Option<&[f64]>,
) -> Result<pyscf_kernels::AoBlockDevice, PyscfRsError> {
    match eval_name {
        "GTOval" | "GTOval_sph" => pyscf_kernels::eval_gto_sph_into_screened(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
            true,
            rcut2,
        ),
        "GTOval_sph_deriv1" => pyscf_kernels::eval_gto_sph_deriv1_into_screened(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
            rcut2,
        ),
        _ => {
            let coords: Vec<[f64; 3]> = (0..ngrids)
                .map(|g| [flat[g], flat[ngrids + g], flat[2 * ngrids + g]])
                .collect();
            let host = pyscf_gto::eval_gto(&cell.mol, eval_name, &coords)?;
            Ok(pyscf_kernels::AoBlockDevice::from_values(
                client,
                &host.values,
                host.shape,
            ))
        }
    }
}

impl Cell {
    /// `cell.pbc_eval_gto(eval_name, coords, kpts)` — `cell.py:2040`.
    ///
    /// # Errors
    /// As [`eval_ao_kpts`].
    pub fn pbc_eval_gto(
        &self,
        eval_name: &str,
        coords: &[[f64; 3]],
        kpts: &[[f64; 3]],
    ) -> Result<EvalAoKptsOutput, PyscfRsError> {
        eval_ao_kpts(self, eval_name, coords, kpts)
    }
}

// ---------------------------------------------------------------------------
// W-09 — AO screening (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`)
// ---------------------------------------------------------------------------

/// Grid points per screening block — upstream's `BLKSIZE`
/// (`gto/eval_gto.py:26`). The screen decides one block at a time; see the
/// module docs on why never one element at a time.
pub const SCREEN_BLKSIZE: usize = 128;

/// `PYSCF_PBC_AO_SCREEN`, read once. `0`/`false`/`no`/`off` disables the W-09
/// screen; anything else, including unset, leaves it on.
///
/// Off is the pre-W-09 behaviour, kept as a bisection switch: the screen drops
/// terms, so it is the first thing to rule out when a periodic result moves.
fn ao_screen_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("PYSCF_PBC_AO_SCREEN").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
    })
}

/// `PYSCF_PBC_AO_POINT_SCREEN`, read once. `1`/`true`/`yes`/`on` also applies
/// the per-shell radius inside the AO kernels per grid point (A-04, session
/// 4); unset or anything else keeps the W-09 block screen only.
///
/// **Off by default — RULE S.** The item was written on the model that a kept
/// 128-point block still holds many points outside every shell's cutoff
/// sphere. Measured (session 4, `krks_profile ao`, si 2×2×2 mesh 31, same
/// binary): the cold pass moved by under 5 % on every row (1.82 → 1.77 s,
/// 5.03 → 5.16 s, 17.8 → 17.4 s), and a lane that skips ALL its arithmetic
/// costs the same as one that does it — the AO kernel is not where the pass's
/// time goes (K-09 is). It drops terms, so without a speed ratio above 1.0 it
/// does not ship on; `tests/eval_ao_point_screen.rs` keeps it gated at the
/// W-09 bound for whoever measures it on a GPU.
fn ao_point_screen_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("PYSCF_PBC_AO_POINT_SCREEN").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

fn skip_k08_for_measurement() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PYSCF_PBC_AO_SKIP_K08").is_ok_and(|v| v == "1"))
}

/// Axis-aligned bounding box of one grid block, in Bohr.
#[derive(Clone, Copy)]
struct BlockBox {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl BlockBox {
    /// Squared distance from `p` to the nearest point of the box — `0` when `p`
    /// is inside. The standard point-to-AABB test, and the reason the screen is
    /// `O(1)` per `(image, block, shell)` instead of `O(SCREEN_BLKSIZE)`.
    fn dist2(&self, p: [f64; 3]) -> f64 {
        let mut d2 = 0.0;
        for axis in 0..3 {
            let x = p[axis];
            let d = if x < self.lo[axis] {
                self.lo[axis] - x
            } else if x > self.hi[axis] {
                x - self.hi[axis]
            } else {
                0.0
            };
            d2 += d * d;
        }
        d2
    }
}

/// One bounding box per `SCREEN_BLKSIZE`-sized block of `coords`.
fn block_boxes(coords: &[[f64; 3]]) -> Vec<BlockBox> {
    coords
        .chunks(SCREEN_BLKSIZE)
        .map(|chunk| {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for c in chunk {
                for axis in 0..3 {
                    if c[axis] < lo[axis] {
                        lo[axis] = c[axis];
                    }
                    if c[axis] > hi[axis] {
                        hi[axis] = c[axis];
                    }
                }
            }
            BlockBox { lo, hi }
        })
        .collect()
}

/// The screen for ONE lattice image: which grid blocks any shell of the image
/// at `l` can reach.
///
/// `shell_centres` and `rcut2` are per-shell; `rcut2[s]` is `rcut[s]^2`, kept
/// squared so the test needs no square root. Returns `None` when no block
/// survives — the caller then skips the image entirely, which is where most of
/// the saving comes from.
/// `PYSCF_PBC_AO_DENSE_FULL=0` keeps the gather/scatter path even for images
/// whose every block is kept — the profiler's A/B switch for K-08b.
fn dense_full_images_enabled() -> bool {
    !std::env::var("PYSCF_PBC_AO_DENSE_FULL").is_ok_and(|v| v == "0")
}

fn screen_one_image(
    boxes: &[BlockBox],
    shell_centres: &[[f64; 3]],
    rcut2: &[f64],
    l: [f64; 3],
) -> Option<Vec<bool>> {
    let mut keep = vec![false; boxes.len()];
    let mut any = false;
    for (b, bx) in boxes.iter().enumerate() {
        for (s, centre) in shell_centres.iter().enumerate() {
            // The AO of shell `s` in image `l` is centred at `centre + l`;
            // equivalently the grid is shifted by `-l`, which is what
            // `eval_ao_kpts_with_images` actually does. Same distance either way.
            let p = [centre[0] + l[0], centre[1] + l[1], centre[2] + l[2]];
            if bx.dist2(p) <= rcut2[s] {
                keep[b] = true;
                any = true;
                break;
            }
        }
    }
    if any { Some(keep) } else { None }
}

/// Per-shell centres, one entry per basis function shell.
fn shell_centres(cell: &Cell) -> Vec<[f64; 3]> {
    (0..cell.mol.nbas)
        .map(|i| {
            let atom = cell.mol._bas[i * BAS_SLOTS + ATOM_OF].max(0) as usize;
            cell.mol.atom_coord(atom)
        })
        .collect()
}

/// Gather the coordinates of the kept blocks, and the flat grid index of each
/// gathered point.
fn gather_kept(
    coords: &[[f64; 3]],
    keep: &[bool],
    l: [f64; 3],
    shifted: &mut Vec<f64>,
    index: &mut Vec<usize>,
) {
    let n_kept: usize = keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(b, _)| {
            let start = b * SCREEN_BLKSIZE;
            (start + SCREEN_BLKSIZE).min(coords.len()) - start
        })
        .sum();
    index.clear();
    index.reserve(n_kept);
    for (b, k) in keep.iter().enumerate() {
        if !k {
            continue;
        }
        let start = b * SCREEN_BLKSIZE;
        let end = (start + SCREEN_BLKSIZE).min(coords.len());
        for g in 0..(end - start) {
            index.push(start + g);
        }
    }
    shifted.clear();
    shifted.reserve(3usize.saturating_mul(n_kept));
    for axis in 0..3 {
        shifted.extend(index.iter().map(|&g| coords[g][axis] - l[axis]));
    }
}
