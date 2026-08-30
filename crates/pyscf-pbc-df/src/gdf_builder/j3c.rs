//! The 3-centre tensor and `cderi` — `gdf_builder.py:198-495` and
//! `rsdf_builder.py:830-1011` (plan 14-02, Task 6).
//!
//! # The pipeline, in upstream's order
//!
//! ```text
//! outcore_auxe2   real-space (mu nu | P) over the FUSED cell, fused on the
//!                 auxiliary index                      -> (nao_pair, naux)
//! gen_j3c_loader  transpose into rows [0, naux) of an (nauxc, ncol) buffer,
//!                 leaving the model-charge rows ZERO, and subtract the
//!                 background-charge term at the gamma difference
//! add_ft_j3c      fill the model-charge rows with the reciprocal-space part
//! solve_cderi     fuse (which subtracts those rows from the auxiliary ones)
//!                 and apply the metric
//! ```
//!
//! The second `fuse` is where the long-range half is added back: the
//! model-charge rows carry `-SUM_G coulG conj(FT_aux) FT_pair`, and fusing
//! subtracts them from the auxiliary rows.
//!
//! # Why the real-space part is finite at all
//!
//! Plan 14-01 measured the raw 3-centre lattice sum diverging with `rcut`
//! against a CHARGED auxiliary cell. Here it is evaluated against the FUSED
//! cell and immediately fused, and upstream's `fuse(j3c)` is bit-identical at
//! `rcut` x1.0/x1.5/x2.0. The compensating charge is what makes this
//! well-defined, and it is why [`crate::incore::aux_e2`]'s doc insists on the
//! fused argument.

use std::collections::HashMap;

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::aftdf::Aftdf;
use crate::error::PbcDfError;
use crate::ft_ao::single::ft_ao_kpt;
use crate::gdf_builder::fuse::{FusedCell, auxbar};
use crate::gdf_builder::j2c::{CdJ2c, J2cTag, weighted_coulg};
use crate::incore::int3c::KptPair;
use crate::incore::{Aosym, aux_e2};

/// The `cderi` tensor of one `(ki, kj)` pair — `(rank, nao_pair)` row-major.
#[derive(Debug, Clone)]
pub struct CderiBlock {
    /// `cderi[L, mu nu]`.
    pub data: CTensor,
    /// Number of fitting vectors (`naux` for the Cholesky route).
    pub rank: usize,
    /// Number of `(mu, nu)` pairs.
    pub nao_pair: usize,
    /// The NEGATIVE branch, for 2-D truncated Coulomb. Always `None` in 3-D.
    pub negative: Option<CTensor>,
}

/// The whole `cderi` store, keyed by `ki * nkpts + kj`.
#[derive(Debug, Clone, Default)]
pub struct Cderi {
    /// One block per computed k-pair.
    pub blocks: HashMap<usize, CderiBlock>,
    /// The k-points the store was built for.
    pub kpts: Vec<[f64; 3]>,
    /// `s1` or `s2`.
    pub aosym: Aosym,
}

impl Cderi {
    /// The block for `(ki, kj)`, if it was computed.
    pub fn get(&self, ki: usize, kj: usize) -> Option<&CderiBlock> {
        self.blocks.get(&(ki * self.kpts.len() + kj))
    }
    /// `get_naoaux()` — `df.py:568-611`.
    ///
    /// The rank of the **diagonal** `(0, 0)` block, i.e. of the `q = 0` group.
    ///
    /// **Not a global agreement check, and 14-06 is why.** Plan 14-03 made this
    /// raise when the blocks disagreed, on the reasoning that upstream "raises
    /// rather than silently truncating". Upstream does no such thing: it opens
    /// the file, takes `next(iter(...))` — one arbitrary block — and returns its
    /// leading dimension (`df.py:592-597`). And the ranks legitimately DO
    /// differ per k-difference on the eigen route: MDF on He-fcc 2x2x2 at mesh
    /// 15 drops 10 vectors for one group and 11 for another, and that is
    /// correct, because the auxiliary index is only comparable WITHIN a group.
    ///
    /// The one consumer that needs a number — `df_jk::get_j_kpts`'s `rho`
    /// accumulator — sums over the DIAGONAL pairs only, and every diagonal pair
    /// `(k, k)` has `q = 0`, so they all share one metric and one rank. That is
    /// the rank this returns.
    ///
    /// The consumers that contract ACROSS groups (`df_ao2mo`'s two-block
    /// branches) check the pair they actually use, where the mismatch is
    /// actionable.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when the store is empty.
    pub fn naoaux(&self) -> Result<usize, PbcDfError> {
        if let Some(b) = self.get(0, 0) {
            return Ok(b.rank);
        }
        let mut keys: Vec<usize> = self.blocks.keys().copied().collect();
        keys.sort_unstable();
        match keys.first().and_then(|k| self.blocks.get(k)) {
            Some(b) => Ok(b.rank),
            None => Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(
                    "get_naoaux: the cderi store is empty".into(),
                ),
            ))),
        }
    }
}

/// `(re, im, chg)` — the model-charge Fourier transform laid out
/// `[g * n_chg + c]`, plus the fused AO indices those columns belong to.
pub type WeightedFtAo = (Vec<f64>, Vec<f64>, Vec<usize>);

/// `weighted_ft_ao(kpt)` — `gdf_builder.py:359-390`, the `dimension == 3` branch.
///
/// The Fourier transform of the MODEL-CHARGE shells only, times the weighted
/// Coulomb kernel.
///
/// # Errors
/// Propagates the G-vector build and `get_coulG`.
pub fn weighted_ft_ao(
    cell: &Cell,
    fused: &FusedCell,
    kpt: [f64; 3],
    mesh: [usize; 3],
) -> Result<WeightedFtAo, PbcDfError> {
    let gv = pyscf_pbc_gto::gv::get_gv(&fused.fused.cell, Some(mesh))?;
    let ngrids = gv.len();
    let nauxc = fused.nauxc();
    let chg: Vec<usize> = (0..nauxc).filter(|q| !fused.aux_ao.contains(q)).collect();
    let (agr, agi) = ft_ao_kpt(&fused.fused.cell.mol, &gv, kpt)?;
    let coulg = weighted_coulg(cell, kpt, mesh)?;

    let nchg = chg.len();
    let mut re = vec![0.0_f64; ngrids * nchg];
    let mut im = vec![0.0_f64; ngrids * nchg];
    for g in 0..ngrids {
        let w = coulg[g];
        for (ci, &c) in chg.iter().enumerate() {
            re[g * nchg + ci] = agr[g * nauxc + c] * w;
            im[g * nchg + ci] = agi[g * nauxc + c] * w;
        }
    }
    Ok((re, im, chg))
}

/// `outcore_auxe2(...)` — `gdf_builder.py:198-357`, minus every `merge_dd`
/// branch and the `_outcore_dd_block` call (D-PBC-23).
///
/// Returns the real-space `(nao_pair, naux)` block per `(ki, kj)` pair, already
/// `fuse`d on the auxiliary index.
///
/// # Errors
/// Propagates [`aux_e2`].
pub fn outcore_auxe2(
    cell: &Cell,
    fused: &FusedCell,
    aosym: Aosym,
    kptij: &[KptPair],
    rcut: Option<f64>,
    omega: Option<f64>,
) -> Result<Vec<CTensor>, PbcDfError> {
    let nao_pair = aosym.nao_pair(cell.mol.nao_nr);
    let raw = aux_e2(cell, &fused.fused, aosym, kptij, rcut, omega)?;
    Ok(raw
        .iter()
        .map(|m| CTensor {
            re: fused.fuse_cols(&m.re, nao_pair),
            im: fused.fuse_cols(&m.im, nao_pair),
        })
        .collect())
}

/// Which of the two 3-centre schemes `make_j3c` is driving.
///
/// The pipeline — one real-space pass, one metric pass, one reciprocal-space
/// pass, one solve — is IDENTICAL for both, and upstream expresses that by
/// making `_CCMDFBuilder` a subclass of `_CCGDFBuilder` that overrides four
/// methods (`mdf.py:354-460`). This port expresses it as one driver with a
/// scheme tag, for the same reason: two copies of `make_j3c` would drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// `_CCGDFBuilder` — the compensating charge. The working buffer has
    /// `nauxc` rows; the MODEL-CHARGE rows carry the reciprocal-space half and
    /// `fuse` subtracts them from the auxiliary rows inside `solve_cderi`.
    CompensatedCharge,
    /// `_CCMDFBuilder` — mixed density fitting. The buffer is fused down to
    /// `naux` rows BEFORE the reciprocal-space pass (`mdf.py:420-428` truncates
    /// the loader's output to `[:naux]`), and the plane-wave residual is
    /// subtracted from every auxiliary row rather than from a separate set of
    /// model-charge rows. The metric is the plane-wave-projected one
    /// (`mdf.py:369-400`) and the solve is always the eigen route.
    Mixed,
}

/// The `(nrows, ncol)` buffer `add_ft_j3c` and `solve_cderi` operate on.
///
/// `nrows` is `nauxc` for [`Scheme::CompensatedCharge`] and `naux` for
/// [`Scheme::Mixed`].
struct J3cBuf {
    re: Vec<f64>,
    im: Vec<f64>,
    ncol: usize,
    nrows: usize,
    real_only: bool,
}

/// `gen_j3c_loader(...)` — `gdf_builder.py:391-448`, minus the
/// `exclude_dd_block` branch.
///
/// Transposes the real-space block into rows `[0, naux)` of an `(nauxc, ncol)`
/// buffer — the model-charge rows stay ZERO for `add_ft_j3c` to fill — and
/// applies the background-charge correction `vbar * S` at the gamma difference.
fn load_j3c(
    fused: &FusedCell,
    realspace: &CTensor,
    nao_pair: usize,
    vbar: Option<&(Vec<f64>, CTensor)>,
    real_only: bool,
    scheme: Scheme,
) -> J3cBuf {
    let naux = fused.naux();
    let nrows = match scheme {
        Scheme::CompensatedCharge => fused.nauxc(),
        // `mdf.py:420-428`: the loader's output is truncated to `[:naux]`, so
        // MDF never allocates model-charge rows at all.
        Scheme::Mixed => naux,
    };
    let aux_row = |a: usize| match scheme {
        Scheme::CompensatedCharge => fused.aux_ao[a],
        Scheme::Mixed => a,
    };
    let mut re = vec![0.0_f64; nrows * nao_pair];
    let mut im = vec![0.0_f64; nrows * nao_pair];
    for a in 0..naux {
        let row = aux_row(a) * nao_pair;
        for c in 0..nao_pair {
            re[row + c] = realspace.re[c * naux + a];
            if !real_only {
                im[row + c] = realspace.im[c * naux + a];
            }
        }
    }
    // `vmod = vbar[:,None] * ovlp[kj]; vR -= vmod.real; vI -= vmod.imag`.
    if let Some((vb, ovlp)) = vbar {
        for (a, &v) in vb.iter().enumerate().take(naux) {
            if v == 0.0 {
                continue;
            }
            let row = aux_row(a) * nao_pair;
            for c in 0..nao_pair {
                re[row + c] -= v * ovlp.re[c];
                if !real_only {
                    im[row + c] -= v * ovlp.im[c];
                }
            }
        }
    }
    J3cBuf {
        re,
        im,
        ncol: nao_pair,
        nrows,
        real_only,
    }
}

/// `add_ft_j3c(j3c, Gpq, Gaux, p0, p1)` — `gdf_builder.py:450-477`, the
/// `dimension == 3` branch.
///
/// Fills the MODEL-CHARGE rows with `-SUM_G conj(FT_chg coulG) FT_pair`. The
/// auxiliary rows are untouched in 3-D; `fuse` moves the result onto them.
#[allow(clippy::too_many_arguments)]
fn add_ft_j3c(
    buf: &mut J3cBuf,
    chg: &[usize],
    gauxr: &[f64],
    gauxi: &[f64],
    gpqr: &[f64],
    gpqi: &[f64],
    p0: usize,
    p1: usize,
) {
    let nchg = chg.len();
    let ncol = buf.ncol;
    for (gi, _g) in (p0..p1).enumerate() {
        let gb = gi * nchg;
        let pb = gi * ncol;
        for (ci, &c) in chg.iter().enumerate() {
            let (ar, ai) = (gauxr[gb + ci], gauxi[gb + ci]);
            let row = c * ncol;
            for col in 0..ncol {
                let (br, bi) = (gpqr[pb + col], gpqi[pb + col]);
                // j3cR -= GchgR·GpqR + GchgI·GpqI
                buf.re[row + col] -= ar * br + ai * bi;
                if !buf.real_only {
                    // j3cI -= GchgR·GpqI - GchgI·GpqR
                    buf.im[row + col] -= ar * bi - ai * br;
                }
            }
        }
    }
}

/// `solve_cderi(cd_j2c, j3cR, j3cI)` — `gdf_builder.py:479-495`.
///
/// `fuse` first — that is where the reciprocal-space half joins the real-space
/// one — then either a lower-triangular solve (Cholesky) or a dense multiply
/// (eigen).
///
/// # Errors
/// [`PbcDfError::Core`] on a singular triangular factor.
fn solve_cderi(
    fused: &FusedCell,
    cd: &CdJ2c,
    buf: &J3cBuf,
    scheme: Scheme,
) -> Result<CderiBlock, PbcDfError> {
    let naux = fused.naux();
    let ncol = buf.ncol;
    // `mdf.py:445-458` does NOT fuse: its buffer is already `naux` rows.
    let j3c = match scheme {
        Scheme::CompensatedCharge => {
            let (jr, ji) = fused.fuse_rows_complex(&buf.re, &buf.im, ncol);
            CTensor { re: jr, im: ji }
        }
        Scheme::Mixed => {
            debug_assert_eq!(buf.nrows, naux, "MDF's buffer is already naux rows");
            CTensor {
                re: buf.re.clone(),
                im: buf.im.clone(),
            }
        }
    };

    let data = match cd.tag {
        J2cTag::Cd => solve_triangular_lower(&cd.j2c, &j3c, naux, ncol)?,
        J2cTag::Eig => zmul(&cd.j2c, &j3c, cd.rank, naux, ncol),
    };
    let negative = cd
        .j2c_negative
        .as_ref()
        .map(|n| zmul(n, &j3c, n.re.len() / naux, naux, ncol));
    Ok(CderiBlock {
        data,
        rank: cd.rank,
        nao_pair: ncol,
        negative,
    })
}

/// `scipy.linalg.solve_triangular(L, b, lower=True)` for a complex lower `L`.
fn solve_triangular_lower(
    l: &CTensor,
    b: &CTensor,
    n: usize,
    ncol: usize,
) -> Result<CTensor, PbcDfError> {
    let mut xr = b.re.clone();
    let mut xi = b.im.clone();
    for i in 0..n {
        for k in 0..i {
            let (ar, ai) = (l.re[i * n + k], l.im[i * n + k]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            for c in 0..ncol {
                let (br, bi) = (xr[k * ncol + c], xi[k * ncol + c]);
                xr[i * ncol + c] -= ar * br - ai * bi;
                xi[i * ncol + c] -= ar * bi + ai * br;
            }
        }
        let d = l.re[i * n + i];
        if d == 0.0 || !d.is_finite() {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "solve_cderi: the Cholesky factor has a zero pivot at row {i}"
                )),
            )));
        }
        for c in 0..ncol {
            xr[i * ncol + c] /= d;
            xi[i * ncol + c] /= d;
        }
    }
    Ok(CTensor { re: xr, im: xi })
}

/// `(m, n) x (n, ncol)` complex multiply, row-major.
fn zmul(a: &CTensor, b: &CTensor, m: usize, n: usize, ncol: usize) -> CTensor {
    let mut re = vec![0.0_f64; m * ncol];
    let mut im = vec![0.0_f64; m * ncol];
    for i in 0..m {
        for k in 0..n {
            let (ar, ai) = (a.re[i * n + k], a.im[i * n + k]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            for c in 0..ncol {
                let (br, bi) = (b.re[k * ncol + c], b.im[k * ncol + c]);
                re[i * ncol + c] += ar * br - ai * bi;
                im[i * ncol + c] += ar * bi + ai * br;
            }
        }
    }
    CTensor { re, im }
}

/// `make_j3c(...)` — `rsdf_builder.py:889-1011`, minus the `dd` branches.
///
/// Groups the `(ki, kj)` pairs by their k-difference (each group shares one
/// metric), builds the real-space and reciprocal-space halves, and solves.
///
/// # Errors
/// Propagates every stage.
#[allow(clippy::too_many_arguments)]
pub fn make_j3c(
    cell: &Cell,
    fused: &FusedCell,
    kpts: &[[f64; 3]],
    aosym: Aosym,
    mesh: [usize; 3],
    j_only: bool,
    j2c_eig_always: bool,
    rcut: Option<f64>,
) -> Result<Cderi, PbcDfError> {
    make_j3c_scheme(
        cell,
        fused,
        kpts,
        aosym,
        mesh,
        j_only,
        j2c_eig_always,
        rcut,
        Scheme::CompensatedCharge,
    )
}

/// [`make_j3c`] with the scheme selected explicitly — the entry point
/// [`crate::mdf`] uses. See [`Scheme`].
///
/// # Errors
/// As [`make_j3c`].
#[allow(clippy::too_many_arguments)]
pub fn make_j3c_scheme(
    cell: &Cell,
    fused: &FusedCell,
    kpts: &[[f64; 3]],
    aosym: Aosym,
    mesh: [usize; 3],
    j_only: bool,
    j2c_eig_always: bool,
    rcut: Option<f64>,
    scheme: Scheme,
) -> Result<Cderi, PbcDfError> {
    let nkpts = kpts.len();
    let naux = fused.naux();
    let nao = cell.mol.nao_nr;
    let nao_pair = aosym.nao_pair(nao);

    let groups = kk_groups(cell, kpts, j_only)?;
    let aftdf = Aftdf::with_mesh(cell.clone(), kpts, mesh)?;

    // `vbar` — the background-charge term, gamma difference only.
    let vbar_full = fused.fuse_rows(&auxbar(&fused.fused.cell), 1);
    let ovlp = pyscf_pbc_gto::pbc_intor::pbc_intor(
        cell,
        "int1e_ovlp",
        kpts,
        pyscf_pbc_gto::pbc_intor::PbcIntorOpts {
            hermi: 1,
            ..Default::default()
        },
    )?;

    // --- ONE real-space pass over every k-pair, and ONE metric pass over every
    //     group k-point.
    //
    // Upstream's `make_j3c` calls `outcore_auxe2` once (`rsdf_builder.py:930`)
    // and `gen_uniq_kpts_groups` calls `get_2c2e(j2c_uniq_kpts)` once
    // (`:853`), both BEFORE the group loop. Doing either inside the loop is a
    // faithfulness bug as well as a performance one: both `aux_e2` and
    // `pbc_intor` fold each lattice image into EVERY requested k-point in a
    // single sweep, so `nkpts` separate calls cost `nkpts` times one call and
    // return the same numbers. Measured on He-fcc 2x2x2 (8 groups):
    // `get_2c2e` 5.47 s and `outcore_auxe2` 10.98 s PER GROUP became one pass
    // each.
    // --- the PASSES, which are not the same thing as the groups.
    //
    // `gen_uniq_kpts_groups` (`rsdf_builder.py:851-871`) yields TWO entries per
    // non-self-conjugate group: the group itself at `+kpt`, and its
    // time-reverse at `-kpt` with the pairs SWAPPED and the SAME decomposition
    // conjugated. Upstream's comment says why it does not simply decompose
    // `j2c[-k]` instead:
    //
    // > If self.mesh is not enough to converge compensated charge or SR-coulG,
    // > the conj symmetry between j2c[k] and j2c[k_conj] may not be strictly
    // > held. Decomposing j2c[k] and j2c[k_conj] may lead to different
    // > dimension in cderi tensor. […] By using the conj(j2c[k]) and
    // > -uniq_kpts[k] […] conj-symmetry in j2c is imposed.
    //
    // `kk_adapted_iter` is called with time-reversal symmetry ON, so it returns
    // only the representative of each conjugate pair — the second half is
    // synthesised here, exactly as upstream does.
    let passes: Vec<Pass> = groups
        .iter()
        .enumerate()
        .flat_map(|(gi, g)| {
            let mut v = vec![Pass {
                group: gi,
                kpt: g.kpt,
                ki: g.ki_idx.clone(),
                kj: g.kj_idx.clone(),
                conj: false,
            }];
            if !g.self_conj {
                v.push(Pass {
                    group: gi,
                    kpt: [-g.kpt[0], -g.kpt[1], -g.kpt[2]],
                    ki: g.kj_idx.clone(),
                    kj: g.ki_idx.clone(),
                    conj: true,
                });
            }
            v
        })
        .collect();

    let flat_pairs: Vec<KptPair> = passes
        .iter()
        .flat_map(|p| {
            p.ki.iter()
                .zip(p.kj.iter())
                .map(|(&i, &j)| KptPair {
                    ki: kpts[i],
                    kj: kpts[j],
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // Both schemes here are FULL-RANGE: the compensated-charge and mixed
    // routes make the lattice sum converge by neutralising the auxiliary
    // functions, not by splitting the kernel. Range-separated fitting
    // (`_RSGDFBuilder`, plan 14-07 sub-tasks 7b/7c) passes `Some(-omega)` to
    // both of these and adds the long-range plane-wave half separately; the ω
    // argument exists on both callees so that builder is a builder and not a
    // plumbing change.
    let realspace_all = outcore_auxe2(cell, fused, aosym, &flat_pairs, rcut, None)?;

    let uniq_kpts: Vec<[f64; 3]> = groups.iter().map(|g| g.kpt).collect();
    let j2c_all = match scheme {
        Scheme::CompensatedCharge => {
            crate::gdf_builder::j2c::get_2c2e(cell, fused, &uniq_kpts, None)?
        }
        // MDF's metric is the Gaussian one with the plane-wave projection
        // REMOVED — `mdf.py:369-400`, and on MDF's own (small) mesh, not the
        // tightened `j2c_mesh` the compensated route uses.
        Scheme::Mixed => crate::mdf::builder::get_2c2e(cell, fused, &uniq_kpts, mesh)?,
    };

    let mut out = Cderi {
        blocks: HashMap::new(),
        kpts: kpts.to_vec(),
        aosym,
    };

    // One decomposition per GROUP, reused by both of its passes.
    let mut cds: Vec<CdJ2c> = Vec::with_capacity(groups.len());
    for (gi, g) in groups.iter().enumerate() {
        // `if self_conj: j2c = np.asarray(j2c).real` — `rsdf_builder.py:866-868`.
        //
        // **This line is load-bearing and its absence is invisible on the
        // Cholesky route.** For a self-conjugate difference the metric is real
        // in exact arithmetic and carries a ~1e-16 imaginary part in practice.
        // A complex Hermitian eigensolver is then free to return each
        // eigenvector with an arbitrary phase `e^{i theta}` — and `cderi` is
        // contracted as `SUM_L c_L c_L` with NO conjugate (`df_ao2mo.py:74`,
        // `zdotNN`), so that phase survives as `e^{2 i theta}` instead of
        // cancelling. Cholesky has no such freedom (its factor is unique with a
        // positive real diagonal), which is why GDF never saw this and MDF —
        // whose `j2c_eig_always` is `True` — failed on it: measured 6.3e6 Ha on
        // He-fcc 2x2x2 before this line, and its own Gate 2 afterwards.
        let mut j2c = j2c_all[gi].clone();
        if g.self_conj {
            j2c.im.iter_mut().for_each(|v| *v = 0.0);
        }
        cds.push(crate::gdf_builder::j2c::decompose_j2c(
            &j2c,
            naux,
            j2c_eig_always,
        )?);
    }

    let mut base = 0usize;
    for g in &passes {
        let uniq_kpt = g.kpt;
        let cd = if g.conj {
            conj_j2c(&cds[g.group])
        } else {
            cds[g.group].clone()
        };

        // The reciprocal-space half, blocked over G exactly as `ft_loop` is.
        //
        // CC: only the MODEL-CHARGE columns, landing on the model-charge rows.
        // MDF: EVERY (fused) auxiliary column, landing on every auxiliary row —
        // `mdf.py:410-418`. That difference is the whole of mixed density
        // fitting: the plane waves correct the Gaussian fit itself rather than
        // just carrying the compensating charge's long-range tail.
        let (gauxr, gauxi, chg) = match scheme {
            Scheme::CompensatedCharge => weighted_ft_ao(cell, fused, uniq_kpt, mesh)?,
            Scheme::Mixed => crate::mdf::builder::weighted_ft_ao(cell, fused, uniq_kpt, mesh)?,
        };
        let kptjs: Vec<[f64; 3]> = g.kj.iter().map(|&j| kpts[j]).collect();

        let mut bufs: Vec<J3cBuf> = Vec::with_capacity(g.ki.len());
        for (t, (&ki, &kj)) in g.ki.iter().zip(g.kj.iter()).enumerate() {
            let real_only = pyscf_pbc_lib::kpts_helper::is_zero(&kpts[ki])
                && pyscf_pbc_lib::kpts_helper::is_zero(&kpts[kj]);
            let vb = if pyscf_pbc_lib::kpts_helper::is_zero(&uniq_kpt) && cell.dimension == 3 {
                let s = packed_ovlp(&ovlp.kmats[kj], nao, aosym);
                Some((vbar_full.clone(), s))
            } else {
                None
            };
            bufs.push(load_j3c(
                fused,
                &realspace_all[base + t],
                nao_pair,
                vb.as_ref(),
                real_only,
                scheme,
            ));
        }

        let nchg = chg.len();
        aftdf.ft_loop(mesh, uniq_kpt, &kptjs, |b| {
            for (t, buf) in bufs.iter_mut().enumerate() {
                let (gr, gi2) = pair_pack(&b.re[t], &b.im[t], b.p1 - b.p0, nao, aosym);
                let gb = b.p0 * nchg;
                add_ft_j3c(
                    buf,
                    &chg,
                    &gauxr[gb..gb + (b.p1 - b.p0) * nchg],
                    &gauxi[gb..gb + (b.p1 - b.p0) * nchg],
                    &gr,
                    &gi2,
                    0,
                    b.p1 - b.p0,
                );
            }
            Ok(())
        })?;

        for (t, buf) in bufs.iter().enumerate() {
            let block = solve_cderi(fused, &cd, buf, scheme)?;
            out.blocks.insert(g.ki[t] * nkpts + g.kj[t], block);
        }
        base += g.ki.len();
    }
    Ok(out)
}

/// One `(kpt, ki_idx, kj_idx, cd_j2c)` yield of `gen_uniq_kpts_groups`. A
/// non-self-conjugate group produces TWO of these — see the comment at the
/// construction site.
struct Pass {
    group: usize,
    kpt: [f64; 3],
    ki: Vec<usize>,
    kj: Vec<usize>,
    conj: bool,
}

/// `_conj_j2c(cd_j2c)` — `rsdf_builder.py:1394-1399`.
fn conj_j2c(cd: &CdJ2c) -> CdJ2c {
    let conj = |t: &CTensor| CTensor {
        re: t.re.clone(),
        im: t.im.iter().map(|v| -v).collect(),
    };
    CdJ2c {
        j2c: conj(&cd.j2c),
        rank: cd.rank,
        j2c_negative: cd.j2c_negative.as_ref().map(conj),
        tag: cd.tag,
    }
}

/// `gen_uniq_kpts_groups(j_only, ...)` — `rsdf_builder.py:830-887`.
///
/// `j_only` (or a single k-point) collapses to the diagonal pairs and ONE
/// metric at the gamma difference; otherwise the pairs are grouped by their
/// difference through [`pyscf_pbc_lib::kpts_helper::kk_adapted_iter`].
fn kk_groups(
    cell: &Cell,
    kpts: &[[f64; 3]],
    j_only: bool,
) -> Result<Vec<pyscf_pbc_lib::kpts_helper::KkGroup>, PbcDfError> {
    let nkpts = kpts.len();
    if j_only || nkpts == 1 {
        return Ok(vec![pyscf_pbc_lib::kpts_helper::KkGroup {
            kpt: [0.0; 3],
            ki_idx: (0..nkpts).collect(),
            kj_idx: (0..nkpts).collect(),
            self_conj: true,
        }]);
    }
    let mut dk_abs = Vec::with_capacity(nkpts * nkpts);
    for i in 0..nkpts {
        for j in 0..nkpts {
            dk_abs.push([
                kpts[j][0] - kpts[i][0],
                kpts[j][1] - kpts[i][1],
                kpts[j][2] - kpts[i][2],
            ]);
        }
    }
    let scaled = cell.get_scaled_kpts(&dk_abs);
    pyscf_pbc_lib::kpts_helper::kk_adapted_iter(nkpts, &scaled, &dk_abs, None, true).map_err(|()| {
        PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(
                "make_j3c: kk_adapted_iter refused its arguments".into(),
            ),
        ))
    })
}

/// `lib.pack_tril(s)` / `s.ravel()` on one F-order overlap matrix.
fn packed_ovlp(s: &CTensor, nao: usize, aosym: Aosym) -> CTensor {
    let n = aosym.nao_pair(nao);
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for mu in 0..nao {
        for nu in 0..nao {
            let Some(row) = pair_row(aosym, nao, mu, nu) else {
                continue;
            };
            // `pbc_intor` is F-order: element (mu, nu) is `mu + nu*nao`.
            re[row] = s.re[mu + nu * nao];
            im[row] = s.im[mu + nu * nao];
        }
    }
    CTensor { re, im }
}

/// Pack one `ft_aopair` G-block from `(nG, nao, nao)` into `(nG, nao_pair)`.
fn pair_pack(re: &[f64], im: &[f64], ng: usize, nao: usize, aosym: Aosym) -> (Vec<f64>, Vec<f64>) {
    let np = aosym.nao_pair(nao);
    if matches!(aosym, Aosym::S1) {
        return (re.to_vec(), im.to_vec());
    }
    let mut r = vec![0.0_f64; ng * np];
    let mut i = vec![0.0_f64; ng * np];
    for g in 0..ng {
        let src = g * nao * nao;
        let dst = g * np;
        for mu in 0..nao {
            for nu in 0..=mu {
                let row = mu * (mu + 1) / 2 + nu;
                r[dst + row] = re[src + mu * nao + nu];
                i[dst + row] = im[src + mu * nao + nu];
            }
        }
    }
    (r, i)
}

fn pair_row(aosym: Aosym, nao: usize, mu: usize, nu: usize) -> Option<usize> {
    match aosym {
        Aosym::S1 => Some(mu * nao + nu),
        Aosym::S2 => {
            if mu >= nu {
                Some(mu * (mu + 1) / 2 + nu)
            } else {
                None
            }
        }
    }
}
