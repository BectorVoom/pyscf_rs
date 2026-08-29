//! K-15 — the analytic Fourier transform of a Gaussian AO pair (PBC-MASTER-PLAN
//! §6, plan 13-01). **The largest new kernel in v2.0.**
//!
//! ```text
//! ft[μν, G] = Σ_{t,u,v} E_t^x·E_u^y·E_v^z · (−iG_x)^t (−iG_y)^u (−iG_z)^v
//!             · (π/p)^{3/2} · e^{−|G|²/4p} · e^{−iG·P}
//! ```
//!
//! The `E` coefficients come from the McMurchie–Davidson recursion, which runs on
//! the HOST (`pyscf_pbc_df::ft_ao::mcmurchie`) because it is independent of `G`
//! and there are orders of magnitude more G-vectors than primitive pairs. What
//! is left here is a pure polynomial-times-exponential evaluation: no recursion,
//! no allocation, one complex accumulator per thread and therefore **no atomics**.
//!
//! One thread owns one `(cartesian AO pair slot, G)` and loops over that shell
//! pair's `(lattice image × primitive pair)` records, so the lattice sum and the
//! contraction both close inside the thread. The Bloch phase `e^{ik·L}` is folded
//! into each record's prefactor on the host, which is what lets one launch serve
//! one k-point without a separate phase pass.
//!
//! Layout is PLANAR (D-PBC-02 / RULE 8): `out_re` / `out_im` are separate
//! buffers, never an interleaved `[re, im, …]` one. The output is dense
//! `(ngrids, ncart, ncart)` and **must be zeroed by the caller** — screened-out
//! shell pairs are simply never written.
//!
//! **Concrete `f64`, not generic over `F: Float`** — deliberately, and for the
//! same reason as every sibling PBC kernel that does scalar math
//! (`struct_factor.rs`, `ewald.rs`, and the `exp` call sites in `eval_gto.rs`):
//! `cube_math::double` IS the f64 libm, and a `#[cube]` body generic over `F`
//! cannot call it. `cube-math` exposes a separate `single` module for f32, so
//! the f32 seam is a second entry point rather than a type parameter. AGENTS.md
//! §3's generics rule is satisfied by kernels whose arithmetic is closed under
//! `Float`; this one is not, and pretending otherwise would mean hand-rolling
//! `exp`/`sincos` and losing the bit-exactness `MathConfig::EXACT` buys.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::dispatch_backend;
use pyscf_algebra::launch::launch_1d;
use pyscf_algebra::{AlgebraClient, AlgebraError};

/// Flat device tables describing one `ft_aopair` launch.
///
/// Built by `pyscf_pbc_df::ft_ao`; this struct is the wire format between the
/// two crates and every field is documented as the host writes it.
#[derive(Debug, Clone, Default)]
pub struct FtAopairTables {
    /// `(ngrids, 3)` G-vectors, already shifted by `q` (`Gv + q`).
    pub gv: Vec<f64>,
    /// Per slot: the shell-pair index it belongs to.
    pub slot_pair: Vec<u32>,
    /// Per slot: flat offset `i_cart * ncart + j_cart` into one G's output plane.
    pub slot_out: Vec<u32>,
    /// Per slot, 6 entries: `ix, iy, iz, jx, jy, jz` Cartesian powers.
    pub slot_pow: Vec<u32>,
    /// Per shell pair: first record index.
    pub pair_rec0: Vec<u32>,
    /// Per shell pair: record count (`images × primitive pairs` that survived).
    pub pair_nrec: Vec<u32>,
    /// Per shell pair: ket angular momentum `lj` (the `E`-index shape).
    pub pair_lj: Vec<u32>,
    /// Per shell pair: `li + lj`.
    pub pair_lij: Vec<u32>,
    /// Per record, 4 entries: `Px, Py, Pz, p = a + b`.
    pub rec_p: Vec<f64>,
    /// Per record, 2 entries: the complex prefactor — contraction coefficients ×
    /// `(π/p)^{3/2}` × `common_fac_sp(li)·common_fac_sp(lj)` × `e^{ik·L}`.
    pub rec_pref: Vec<f64>,
    /// Per record: base offset of its x-axis `E` table inside [`Self::etab`].
    pub rec_eoff: Vec<u32>,
    /// Per record: per-axis stride, so y is at `eoff + stride` and z at
    /// `eoff + 2·stride`.
    pub rec_estride: Vec<u32>,
    /// Every record's three `E` tables, concatenated.
    pub etab: Vec<f64>,
    /// Side length of one output plane (`ncart`), so a plane is `ncart²`.
    pub ncart: usize,
}

/// `i = slot·ngrids + g`, so `g` is the fast axis: adjacent threads read
/// adjacent `Gv` entries and write adjacent output words.
///
/// The `i < nslots·ngrids` guard is required — the launch rounds the thread
/// count up to a whole number of cubes.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ft_aopair_kernel(
    gv: &Array<f64>,
    slot_pair: &Array<u32>,
    slot_out: &Array<u32>,
    slot_pow: &Array<u32>,
    pair_rec0: &Array<u32>,
    pair_nrec: &Array<u32>,
    pair_lj: &Array<u32>,
    pair_lij: &Array<u32>,
    rec_p: &Array<f64>,
    rec_pref: &Array<f64>,
    rec_eoff: &Array<u32>,
    rec_estride: &Array<u32>,
    etab: &Array<f64>,
    out_re: &mut Array<f64>,
    out_im: &mut Array<f64>,
    nslots: usize,
    ngrids: usize,
    ncart2: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < nslots * ngrids {
        let e = idx / ngrids;
        let g = idx % ngrids;

        let gx = gv[g * 3];
        let gy = gv[g * 3 + 1];
        let gz = gv[g * 3 + 2];
        let g2 = gx * gx + gy * gy + gz * gz;

        let p = slot_pair[e] as usize;
        let lj = pair_lj[p] as usize;
        let lij = pair_lij[p] as usize;
        // E[i][j][t] lives at base + i*njt + j*nt + t (mcmurchie::e_index).
        let nt = lij + 1;
        let njt = (lj + 1) * nt;

        let ix = slot_pow[e * 6] as usize;
        let iy = slot_pow[e * 6 + 1] as usize;
        let iz = slot_pow[e * 6 + 2] as usize;
        let jx = slot_pow[e * 6 + 3] as usize;
        let jy = slot_pow[e * 6 + 4] as usize;
        let jz = slot_pow[e * 6 + 5] as usize;

        let r0 = pair_rec0[p] as usize;
        let nrec = pair_nrec[p] as usize;

        let mut acc_re = 0.0;
        let mut acc_im = 0.0;

        for r in r0..(r0 + nrec) {
            let px = rec_p[r * 4];
            let py = rec_p[r * 4 + 1];
            let pz = rec_p[r * 4 + 2];
            let pexp = rec_p[r * 4 + 3];

            let eb = rec_eoff[r] as usize;
            let es = rec_estride[r] as usize;
            let bx = eb + ix * njt + jx * nt;
            let by = eb + es + iy * njt + jy * nt;
            let bz = eb + 2 * es + iz * njt + jz * nt;

            // Σ_{t,u,v} E_t E_u E_v · G_x^t G_y^u G_z^v · (−i)^{t+u+v}.
            // The coefficient is real, so (−i)^n only chooses which accumulator
            // it lands in and with which sign: n%4 = 0,1,2,3 → +re, −im, −re, +im.
            let mut poly_re = 0.0;
            let mut poly_im = 0.0;
            let mut gxp = 1.0;
            for t in 0..(ix + jx + 1) {
                let et = etab[bx + t];
                let mut gyp = 1.0;
                for u in 0..(iy + jy + 1) {
                    let eu = etab[by + u];
                    let etu = et * eu * gxp * gyp;
                    let mut gzp = 1.0;
                    for v in 0..(iz + jz + 1) {
                        let w = etu * etab[bz + v] * gzp;
                        let n = (t + u + v) % 4;
                        if n == 0 {
                            poly_re += w;
                        } else if n == 1 {
                            poly_im -= w;
                        } else if n == 2 {
                            poly_re -= w;
                        } else {
                            poly_im += w;
                        }
                        gzp *= gz;
                    }
                    gyp *= gy;
                }
                gxp *= gx;
            }

            // e^{−|G|²/4p} · e^{−iG·P}
            let expo = cube_math::double::exp::exp(
                0.0 - g2 / (4.0 * pexp),
                cube_math::MathConfig::EXACT,
            );
            let theta = 0.0 - (gx * px + gy * py + gz * pz);
            let (sn, cs) = cube_math::double::trig::sincos(theta, cube_math::MathConfig::EXACT);

            // (poly) · (cs + i·sn) · expo · (pref_re + i·pref_im)
            let ar = (poly_re * cs - poly_im * sn) * expo;
            let ai = (poly_re * sn + poly_im * cs) * expo;
            let fr = rec_pref[r * 2];
            let fi = rec_pref[r * 2 + 1];
            acc_re += ar * fr - ai * fi;
            acc_im += ar * fi + ai * fr;
        }

        let o = g * ncart2 + slot_out[e] as usize;
        out_re[o] = acc_re;
        out_im[o] = acc_im;
    }
}

/// Rough element-operation count per thread, for [`launch_1d`]'s CPU sizing.
///
/// One record costs the `(t,u,v)` triple plus a transcendental pair; `nt³` is a
/// generous upper bound on the triple and `exp`+`sincos` are worth roughly a
/// hundred flops between them.
fn work_per_thread(t: &FtAopairTables) -> usize {
    let nrec = t.rec_eoff.len().max(1);
    let npair = t.pair_nrec.len().max(1);
    let avg_rec = nrec / npair;
    let nt = t.pair_lij.iter().copied().max().unwrap_or(0) as usize + 1;
    avg_rec.max(1) * (nt * nt * nt + 100)
}

#[allow(clippy::too_many_arguments)]
fn launch_on_handles<R: Runtime>(
    client: &ComputeClient<R>,
    h: &[Handle],
    out_re: &Handle,
    out_im: &Handle,
    t: &FtAopairTables,
    nslots: usize,
    ngrids: usize,
) {
    let ncart2 = t.ncart * t.ncart;
    let lanes = nslots * ngrids;
    let (count, dim) = launch_1d(client, lanes, work_per_thread(t));
    unsafe {
        ft_aopair_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            // SAFETY: every length below is the host Vec's length, and the
            // kernel guards `idx < nslots*ngrids`; `slot_out` was range-checked
            // against `ncart2` in `ft_aopair`.
            ArrayArg::from_raw_parts(h[0].clone(), t.gv.len()),
            ArrayArg::from_raw_parts(h[1].clone(), t.slot_pair.len()),
            ArrayArg::from_raw_parts(h[2].clone(), t.slot_out.len()),
            ArrayArg::from_raw_parts(h[3].clone(), t.slot_pow.len()),
            ArrayArg::from_raw_parts(h[4].clone(), t.pair_rec0.len()),
            ArrayArg::from_raw_parts(h[5].clone(), t.pair_nrec.len()),
            ArrayArg::from_raw_parts(h[6].clone(), t.pair_lj.len()),
            ArrayArg::from_raw_parts(h[7].clone(), t.pair_lij.len()),
            ArrayArg::from_raw_parts(h[8].clone(), t.rec_p.len()),
            ArrayArg::from_raw_parts(h[9].clone(), t.rec_pref.len()),
            ArrayArg::from_raw_parts(h[10].clone(), t.rec_eoff.len()),
            ArrayArg::from_raw_parts(h[11].clone(), t.rec_estride.len()),
            ArrayArg::from_raw_parts(h[12].clone(), t.etab.len()),
            ArrayArg::from_raw_parts(out_re.clone(), ngrids * ncart2),
            ArrayArg::from_raw_parts(out_im.clone(), ngrids * ncart2),
            nslots,
            ngrids,
            ncart2,
        );
    }
}

/// `pyscf_algebra::launch::upload` is bounded by `DeviceScalar`, which the index
/// tables' `u32` is not (it is not a device *float*). Same one-copy staging,
/// without widening that bound for a buffer the kernel only reads as an index.
fn upload_u32<R: Runtime>(client: &ComputeClient<R>, data: &[u32]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(data))
}

fn launch<R: Runtime>(t: &FtAopairTables, client: &ComputeClient<R>) -> (Vec<f64>, Vec<f64>) {
    use pyscf_algebra::launch::upload;
    let ngrids = t.gv.len() / 3;
    let nslots = t.slot_pair.len();
    let ncart2 = t.ncart * t.ncart;
    let n_out = ngrids * ncart2;

    // The output is DENSE and screened-out pairs are never written, so it must
    // start at zero — `client.empty` would leave whatever the allocator held.
    let zeros = vec![0.0f64; n_out];
    let re_h = upload::<R, f64>(client, &zeros);
    let im_h = upload::<R, f64>(client, &zeros);
    drop(zeros);

    let h = vec![
        upload::<R, f64>(client, &t.gv),
        upload_u32::<R>(client, &t.slot_pair),
        upload_u32::<R>(client, &t.slot_out),
        upload_u32::<R>(client, &t.slot_pow),
        upload_u32::<R>(client, &t.pair_rec0),
        upload_u32::<R>(client, &t.pair_nrec),
        upload_u32::<R>(client, &t.pair_lj),
        upload_u32::<R>(client, &t.pair_lij),
        upload::<R, f64>(client, &t.rec_p),
        upload::<R, f64>(client, &t.rec_pref),
        upload_u32::<R>(client, &t.rec_eoff),
        upload_u32::<R>(client, &t.rec_estride),
        upload::<R, f64>(client, &t.etab),
    ];
    launch_on_handles::<R>(client, &h, &re_h, &im_h, t, nslots, ngrids);
    let bytes = client.read(vec![re_h, im_h]);
    (
        bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec(),
        bytemuck::cast_slice::<u8, f64>(&bytes[1]).to_vec(),
    )
}

/// K-15 public entry point.
///
/// Returns the PLANAR `(re, im)` pair, each a dense row-major
/// `(ngrids, ncart, ncart)` buffer. Slots the caller did not emit stay zero.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] if any per-slot / per-pair / per-record table
/// is not the length the others imply, or if a `slot_out` would write outside a
/// `ncart × ncart` plane. An empty slot list or mesh returns two empty vectors
/// without launching.
pub fn ft_aopair(
    client: &AlgebraClient,
    t: &FtAopairTables,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
    let shape = |what: &str, actual: String| AlgebraError::ShapeMismatch {
        expected: what.to_string(),
        actual,
    };
    if !t.gv.len().is_multiple_of(3) {
        return Err(shape("gv length a multiple of 3", format!("{}", t.gv.len())));
    }
    let nslots = t.slot_pair.len();
    if t.slot_out.len() != nslots || t.slot_pow.len() != 6 * nslots {
        return Err(shape(
            "slot_out = nslots and slot_pow = 6*nslots",
            format!(
                "nslots {nslots}, slot_out {}, slot_pow {}",
                t.slot_out.len(),
                t.slot_pow.len()
            ),
        ));
    }
    let npair = t.pair_rec0.len();
    if t.pair_nrec.len() != npair || t.pair_lj.len() != npair || t.pair_lij.len() != npair {
        return Err(shape(
            "pair_rec0 / pair_nrec / pair_lj / pair_lij all of length npair",
            format!("npair {npair}"),
        ));
    }
    let nrec = t.rec_eoff.len();
    if t.rec_p.len() != 4 * nrec || t.rec_pref.len() != 2 * nrec || t.rec_estride.len() != nrec {
        return Err(shape(
            "rec_p = 4*nrec, rec_pref = 2*nrec, rec_estride = nrec",
            format!("nrec {nrec}"),
        ));
    }
    let ncart2 = t.ncart * t.ncart;
    if let Some(bad) = t.slot_out.iter().find(|&&o| (o as usize) >= ncart2) {
        return Err(shape(
            &format!("every slot_out < ncart² = {ncart2}"),
            format!("{bad}"),
        ));
    }
    if let Some(bad) = t.slot_pair.iter().find(|&&p| (p as usize) >= npair) {
        return Err(shape(
            &format!("every slot_pair < npair = {npair}"),
            format!("{bad}"),
        ));
    }
    let ngrids = t.gv.len() / 3;
    if nslots == 0 || ngrids == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let out = dispatch_backend!(client, c, Rt, launch::<Rt>(t, c));
    Ok(out)
}
