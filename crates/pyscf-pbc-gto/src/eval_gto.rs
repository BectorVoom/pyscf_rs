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
//! **Block granularity, never per element.** A per-element skip is a data-
//! dependent branch in the inner loop, which is the branch divergence
//! `plane_alignment.md` warns about on any SIMT backend and a mispredict on
//! the CPU one.
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

        let (ao_device, scatter_index): (pyscf_kernels::AoBlockDevice, Option<&[usize]>) =
            match &keep {
                None => {
                    {
                        let span = tracing::info_span!("pbc_eval_ao_shift_pack");
                        let _entered = span.enter();
                        shifted_workspace.clear();
                        for axis in 0..3 {
                            shifted_workspace.extend(coords.iter().map(|r| r[axis] - l[axis]));
                        }
                    }
                    let ao = {
                        // `points` — how many grid points this launch covers, so
                        // the A-00 instrument can report the launched-image count
                        // and the kept-point total (the screen's actual yield).
                        let span =
                            tracing::info_span!("pbc_eval_ao_eval_gto", points = ngrids as u64);
                        let _entered = span.enter();
                        eval_gto_device(&client, cell, eval_name, &shifted_workspace, ngrids)?
                    };
                    (ao, None)
                }
                Some(keep) => {
                    {
                        let span = tracing::info_span!("pbc_eval_ao_shift_pack");
                        let _entered = span.enter();
                        gather_kept(
                            coords,
                            keep,
                            *l,
                            &mut shifted_workspace,
                            &mut index_workspace,
                        );
                    }
                    let ao = {
                        let span = tracing::info_span!(
                            "pbc_eval_ao_eval_gto",
                            points = index_workspace.len() as u64
                        );
                        let _entered = span.enter();
                        eval_gto_device(
                            &client,
                            cell,
                            eval_name,
                            &shifted_workspace,
                            index_workspace.len(),
                        )?
                    };
                    (ao, Some(index_workspace.as_slice()))
                }
            };

        let image_ngrids = scatter_index.map_or(ngrids, <[usize]>::len);
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

        // K-08 — one launch per image, folding this image into every k at once,
        // in place on the device-resident accumulators.
        let pr: Vec<f64> = (0..nkpts).map(|k| expkl_re[k * nimgs + m]).collect();
        let pi: Vec<f64> = (0..nkpts).map(|k| expkl_im[k * nimgs + m]).collect();
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

    // W-09: every image may have been screened out (an empty basis, or a grid
    // nothing can reach). `n` is then still 0 and the split below yields the
    // correctly-shaped empty planes, exactly as an empty image list does.
    if n == 0 {
        comp = 1;
    }

    // One read-back for the whole lattice sum. An empty image list never built
    // an accumulator, and `n` is then 0, so the split below yields no planes.
    let (out_re, out_im) = match acc {
        Some(a) => a.into_planes(&client),
        None => (Vec::new(), Vec::new()),
    };

    // Split the flat (nkpts, n) accumulators into one CTensor per k, dropping
    // the imaginary plane at gamma (eval_gto.py:157-158).
    let gamma: Vec<bool> = kpts.iter().map(is_gamma).collect();
    let mut kaos = Vec::with_capacity(nkpts);
    for (k, is_g) in gamma.iter().enumerate() {
        let re = out_re[k * n..(k + 1) * n].to_vec();
        let im = if *is_g {
            vec![0.0; n]
        } else {
            out_im[k * n..(k + 1) * n].to_vec()
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

fn eval_gto_device(
    client: &AlgebraClient,
    cell: &Cell,
    eval_name: &str,
    flat: &[f64],
    ngrids: usize,
) -> Result<pyscf_kernels::AoBlockDevice, PyscfRsError> {
    match eval_name {
        "GTOval" | "GTOval_sph" => pyscf_kernels::eval_gto_sph_into(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
            true,
        ),
        "GTOval_sph_deriv1" => pyscf_kernels::eval_gto_sph_deriv1_into(
            client,
            flat,
            ngrids,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.ao_loc_nr,
            cell.mol.nao_nr,
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
