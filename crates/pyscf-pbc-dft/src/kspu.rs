//! DFT+U with k-point sampling — plan 12-06.
//!
//! Ports `pyscf/pbc/dft/krkspu.py` (325 lines) and `kukspu.py` (301 lines):
//! a Hubbard `U` acting on projected LOCAL orbitals, added to `V_xc` and to the
//! total energy.
//!
//! ```text
//! P^k     = (S^k C_loc)^H D^k (S^k C_loc)                    (krkspu.py:107-109)
//! E_U    += w_k · (U/2) · [Tr P^k − Tr(P^k P^k)/2]           (krkspu.py:110)
//! V_hub  += (S^k C_loc) [(1 − P^k)·(U/2) + α·1] (S^k C_loc)^H (krkspu.py:112-118)
//! ```
//!
//! with `w_k = 1/N_k` for a full-BZ sampling. `α` is the linear-response
//! perturbation used to FIT `U`; when set it adds `α·Tr P^k` to the energy and
//! `α·1` to the local potential.
//!
//! # Local orbitals
//!
//! `C_ao_lo` defaults to Löwdin-orthogonalised MINAO projections
//! (`krkspu.py:161-176`):
//!
//! ```text
//! C_k = S_k^{-1} S12_k            (the least-squares MINAO projection)
//! C_k ← C_k (C_k^H S_k C_k)^{-1/2} (Löwdin orthogonalisation)
//! ```
//!
//! # Deviation from upstream: how a Hubbard site is named
//!
//! Upstream accepts `U_idx` as either explicit AO indices or a
//! `mol.search_ao_label` string such as `'Ni 3d'`. `pyscf-core` has no
//! `ao_labels`/`search_ao_label` surface, so [`USite`] names a site either by
//! explicit reference-basis indices or by `(element, angular momentum,
//! contraction)`, grouped one site per atom exactly as `_set_U` groups them
//! (`rkspu.py:152-155`). The contraction INDEX stands in for the principal
//! quantum number: a minimal reference basis orders its contractions core-first,
//! so upstream's `'Si 3p'` is this port's `contraction: Some(1)` (silicon's `2p`
//! being `Some(0)`), and `None` takes every contraction of that `l`.
//!
//! # A pseudopotential cell needs a different `minao_ref`
//!
//! MINAO carries the CORE functions. A GTH cell's AO space does not span them,
//! so `C^H S C` over the full MINAO set is rank-deficient and the Löwdin step
//! fails with a clear error rather than a silent pseudo-inverse. Upstream has
//! the same property — `reference_mol` has no pseudopotential-aware reduction —
//! and the fix is the same on both sides: set [`HubbardU::minao_ref`] to a
//! valence basis (`"gth-szv"`).

use pyscf_algebra::{CTensor, zeigh_gen, zsolve_linear};
use pyscf_core::PyscfRsError;
use pyscf_core::raw_layout::{ANG_OF, ATOM_OF, BAS_SLOTS};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::pbc_intor::{PbcIntorOpts, intor_cross};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use pyscf_pbc_scf::types::{KDms, KMats};

use crate::error::PbcDftError;
use crate::krks::{Krks, KsEnergyTags};
use crate::kuks::Kuks;
use crate::xc::err;

/// `HARTREE2EV` — `U_val` is given in eV upstream (`krkspu.py:166`).
pub const HARTREE2EV: f64 = 27.211386245988;

/// One Hubbard site.
#[derive(Debug, Clone)]
pub enum USite {
    /// Explicit orbital indices in the MINAO REFERENCE basis.
    Indices(Vec<usize>),
    /// Functions of angular momentum `l` on every atom whose element symbol is
    /// `element`, grouped one site per atom — upstream's
    /// `search_ao_label('Ni 3d')` behaviour (`rkspu.py:151-155`).
    Shell {
        /// Element symbol, e.g. `"Ni"`.
        element: String,
        /// Angular momentum (`2` for d).
        l: u32,
        /// Which CONTRACTION of that `l` to take, 0-based, or `None` for all of
        /// them.
        ///
        /// A reference basis may carry several contractions of one `l` — MINAO
        /// gives silicon both a `2p` and a `3p`. Upstream names one of them by
        /// its principal quantum number (`'Si 3p'`); `pyscf-core` has no AO
        /// labels, so the contraction is addressed by index, which for a
        /// minimal basis is the same ordering (core first, valence last).
        contraction: Option<usize>,
    },
}

/// The DFT+U configuration.
#[derive(Debug, Clone)]
pub struct HubbardU {
    /// One entry per requested site specification.
    pub sites: Vec<USite>,
    /// `U` in ELECTRON VOLTS, one per entry of [`HubbardU::sites`].
    pub u_val: Vec<f64>,
    /// The linear-response perturbation `α` in Hartree, one per site (or one
    /// shared value). Empty means no perturbation.
    pub alpha: Vec<f64>,
    /// The reference basis for the local orbitals. Upstream's `minao_ref`,
    /// default `"minao"`.
    pub minao_ref: String,
    /// Caller-supplied local orbitals `C_ao_lo[k]`, COLUMN-MAJOR
    /// `nao x nlo`. `None` builds the Löwdin MINAO set.
    pub c_ao_lo: Option<Vec<CTensor>>,
}

impl Default for HubbardU {
    fn default() -> Self {
        Self {
            sites: Vec::new(),
            u_val: Vec::new(),
            alpha: Vec::new(),
            minao_ref: "minao".to_string(),
            c_ao_lo: None,
        }
    }
}

/// The resolved sites: `(orbital indices in the LO basis, U in Hartree, α)`.
#[derive(Debug, Clone)]
pub struct ResolvedU {
    /// Orbital indices, one group per Hubbard site.
    pub indices: Vec<Vec<usize>>,
    /// `U` in HARTREE.
    pub u_val: Vec<f64>,
    /// `α` in Hartree; `None` when no perturbation is applied to that site.
    pub alpha: Vec<Option<f64>>,
}

/// `reference_mol(cell, minao)` — `lo/iao.py:123-136`, as a `Cell`.
///
/// # Errors
/// Propagates the cell build.
pub fn reference_cell(cell: &Cell, minao: &str) -> Result<Cell, PbcDftError> {
    let atoms: Vec<(String, [f64; 3])> = cell
        .mol
        ._atom
        .iter()
        .map(|(s, c)| (s.clone(), *c))
        .collect();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name(minao.to_string()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(cell.a),
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        precision: cell.precision,
        ..Default::default()
    })
    .map_err(PbcDftError::Core)
}

/// `_set_U(cell, pcell, U_idx, U_val)` — `rkspu.py:133-169`.
///
/// # Errors
/// [`PbcDftError`] when a site names an element the cell does not contain, or
/// when the site and value counts disagree.
pub fn set_u(pcell: &Cell, cfg: &HubbardU) -> Result<ResolvedU, PbcDftError> {
    if cfg.sites.len() != cfg.u_val.len() {
        return Err(err(format!(
            "DFT+U: {} site specifications for {} U values",
            cfg.sites.len(),
            cfg.u_val.len()
        )));
    }
    let mut indices: Vec<Vec<usize>> = Vec::new();
    let mut u_val: Vec<f64> = Vec::new();
    let mut alpha: Vec<Option<f64>> = Vec::new();

    let alpha_of = |i: usize| -> Option<f64> {
        if cfg.alpha.is_empty() {
            None
        } else if cfg.alpha.len() == 1 {
            Some(cfg.alpha[0])
        } else {
            cfg.alpha.get(i).copied()
        }
    };

    for (i, site) in cfg.sites.iter().enumerate() {
        match site {
            USite::Indices(idx) => {
                indices.push(idx.clone());
                u_val.push(cfg.u_val[i] / HARTREE2EV);
                alpha.push(alpha_of(i));
            }
            USite::Shell {
                element,
                l,
                contraction,
            } => {
                // One group per ATOM, exactly as `_groupby(lab_idx, atm_ids)`.
                let groups = shell_indices(pcell, element, *l, *contraction);
                if groups.is_empty() {
                    return Err(err(format!(
                        "DFT+U: no l = {l} shell on element '{element}' in the \
                         reference basis"
                    )));
                }
                for g in groups {
                    indices.push(g);
                    u_val.push(cfg.u_val[i] / HARTREE2EV);
                    alpha.push(alpha_of(i));
                }
            }
        }
    }
    Ok(ResolvedU {
        indices,
        u_val,
        alpha,
    })
}

/// AO indices of the requested `l`-shell on every atom of `element`, grouped
/// per atom. `contraction` selects one contraction within the shell (0-based);
/// `None` takes all of them.
fn shell_indices(
    cell: &Cell,
    element: &str,
    l: u32,
    contraction: Option<usize>,
) -> Vec<Vec<usize>> {
    let nbas = cell.mol.nbas;
    let mut per_atom: std::collections::BTreeMap<i32, Vec<usize>> =
        std::collections::BTreeMap::new();
    for ib in 0..nbas {
        let atom = cell.mol._bas[ib * BAS_SLOTS + ATOM_OF];
        let ang = cell.mol._bas[ib * BAS_SLOTS + ANG_OF] as u32;
        if ang != l {
            continue;
        }
        let sym = cell
            .mol
            ._atom
            .get(atom as usize)
            .map(|(s, _)| s.as_str())
            .unwrap_or("");
        if !sym.eq_ignore_ascii_case(element) {
            continue;
        }
        let p0 = cell.mol.ao_loc_nr[ib] as usize;
        let p1 = cell.mol.ao_loc_nr[ib + 1] as usize;
        // Within one shell the AO block is `[contraction][m]`, `2l+1` wide.
        let m = 2 * l as usize + 1;
        let range = match contraction {
            None => p0..p1,
            Some(c) => {
                let a = p0 + c * m;
                let b = a + m;
                if b > p1 {
                    continue;
                }
                a..b
            }
        };
        per_atom.entry(atom).or_default().extend(range);
    }
    per_atom.into_values().collect()
}

/// `_make_minao_lo(cell, minao_ref, kpts)` — `krkspu.py:161-176`.
///
/// Returns `C_ao_lo[k]`, COLUMN-MAJOR `nao x nlo`.
///
/// # Errors
/// Propagates the overlap integrals and the linear algebra.
// `k` indexes three parallel k-resolved stacks.
#[allow(clippy::needless_range_loop)]
pub fn make_minao_lo(
    cell: &Cell,
    pcell: &Cell,
    kpts: &[[f64; 3]],
) -> Result<Vec<CTensor>, PbcDftError> {
    let nao = cell.mol.nao_nr;
    let nlo = pcell.mol.nao_nr;
    let s1 = pyscf_pbc_gto::get_ovlp(cell, kpts)?;
    let s12 = intor_cross(
        "int1e_ovlp",
        cell,
        pcell,
        kpts,
        PbcIntorOpts {
            hermi: 0,
            ..PbcIntorOpts::default()
        },
    )?;

    let mut out = Vec::with_capacity(kpts.len());
    for k in 0..kpts.len() {
        // Phase-10 products are F-order; the algebra here is row-major.
        let sk = pyscf_pbc_df::zlinalg::forder_to_c(&s1[k], nao, nao);
        let s12k = pyscf_pbc_df::zlinalg::forder_to_c(&s12.kmats[k], nao, nlo);
        // C = S^-1 S12, solved column by column (`la.cho_solve`).
        let mut c = CTensor::zeros(nao * nlo);
        for j in 0..nlo {
            let mut rhs = CTensor::zeros(nao);
            for i in 0..nao {
                rhs.re[i] = s12k.re[i * nlo + j];
                rhs.im[i] = s12k.im[i * nlo + j];
            }
            let x = zsolve_linear(&sk, &rhs, nao).map_err(|e| {
                err(format!("DFT+U: the MINAO projection solve failed: {e}"))
            })?;
            for i in 0..nao {
                c.re[i * nlo + j] = x.re[i];
                c.im[i * nlo + j] = x.im[i];
            }
        }
        out.push(vec_lowdin(&c, &sk, nao, nlo)?);
    }
    Ok(out)
}

/// `lo.vec_lowdin(c, s)` — `lo/orth.py:43-48`: `c · (c^H s c)^{-1/2}`.
///
/// The inverse square root is taken through the Hermitian eigendecomposition
/// of `m = c^H s c`, which is what `lo.lowdin` does.
///
/// `c` is ROW-MAJOR `nao x nlo` here and the result is returned COLUMN-MAJOR
/// (`out[ao + lo*nao]`), which is the layout every consumer of MO-like
/// coefficients in this workspace uses.
///
/// # Errors
/// [`PbcDftError`] when the metric is singular.
fn vec_lowdin(
    c: &CTensor,
    s: &CTensor,
    nao: usize,
    nlo: usize,
) -> Result<CTensor, PbcDftError> {
    // m = c^H s c
    let mut m = CTensor::zeros(nlo * nlo);
    // sc[i, j] = Σ_t s[i, t] c[t, j]
    let mut sc = CTensor::zeros(nao * nlo);
    for i in 0..nao {
        for j in 0..nlo {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for t in 0..nao {
                let (ar, ai) = (s.re[i * nao + t], s.im[i * nao + t]);
                let (br, bi) = (c.re[t * nlo + j], c.im[t * nlo + j]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            sc.re[i * nlo + j] = re;
            sc.im[i * nlo + j] = im;
        }
    }
    for i in 0..nlo {
        for j in 0..nlo {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for t in 0..nao {
                let (ar, ai) = (c.re[t * nlo + i], -c.im[t * nlo + i]);
                let (br, bi) = (sc.re[t * nlo + j], sc.im[t * nlo + j]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            m.re[i * nlo + j] = re;
            m.im[i * nlo + j] = im;
        }
    }
    // m^{-1/2} through the Hermitian eigendecomposition (identity metric).
    let mut ident = CTensor::zeros(nlo * nlo);
    for i in 0..nlo {
        ident.re[i * nlo + i] = 1.0;
    }
    let (w, u) = zeigh_gen(&m, &ident, nlo)
        .map_err(|e| err(format!("DFT+U: the Löwdin metric eigensolve failed: {e}")))?;
    // x = U diag(w^{-1/2}) U^H, with `u` COLUMN-MAJOR.
    let mut x = CTensor::zeros(nlo * nlo);
    for (n, wn) in w.iter().enumerate() {
        if *wn <= 1e-14 {
            return Err(err(
                "DFT+U: the local-orbital metric is singular; check `minao_ref`"
                    .to_string(),
            ));
        }
        let inv = cube_math::double::pow::pow(*wn, -0.5, cube_math::MathConfig::EXACT);
        let base = n * nlo;
        for i in 0..nlo {
            let (ar, ai) = (u.re[base + i], u.im[base + i]);
            for j in 0..nlo {
                let (br, bi) = (u.re[base + j], -u.im[base + j]);
                x.re[i * nlo + j] += inv * (ar * br - ai * bi);
                x.im[i * nlo + j] += inv * (ar * bi + ai * br);
            }
        }
    }
    // out = c · x, returned COLUMN-MAJOR.
    let mut out = CTensor::zeros(nao * nlo);
    for i in 0..nao {
        for j in 0..nlo {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for t in 0..nlo {
                let (ar, ai) = (c.re[i * nlo + t], c.im[i * nlo + t]);
                let (br, bi) = (x.re[t * nlo + j], x.im[t * nlo + j]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            out.re[i + j * nao] = re;
            out.im[i + j * nao] = im;
        }
    }
    Ok(out)
}

/// `_add_Vhubbard(vxc, ks, dm, kpts)` — `krkspu.py:67-137`.
///
/// Adds the Hubbard potential to `vxc` IN PLACE and returns `E_U`.
///
/// `dms` carries one channel for KRKSpU and two for KUKSpU; the `U/2` factor
/// upstream applies is per CHANNEL, so an unrestricted density (whose channels
/// each hold one electron per orbital) gets the same expression with its own
/// `P^k`.
///
/// # Errors
/// Propagates the overlap integrals and the local-orbital construction.
pub fn add_vhubbard(
    vxc: &mut [KMats],
    cell: &Cell,
    kpts: &[[f64; 3]],
    dms: &KDms,
    cfg: &HubbardU,
) -> Result<f64, PbcDftError> {
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    let weight = 1.0 / nkpts as f64;

    let pcell = reference_cell(cell, &cfg.minao_ref)?;
    let resolved = set_u(&pcell, cfg)?;
    let owned_lo;
    let c_ao_lo: &[CTensor] = match cfg.c_ao_lo.as_deref() {
        Some(c) => c,
        None => {
            owned_lo = make_minao_lo(cell, &pcell, kpts)?;
            &owned_lo
        }
    };
    let nlo = pcell.mol.nao_nr;

    let ovlp = pyscf_pbc_gto::get_ovlp(cell, kpts)?;
    let s: Vec<CTensor> = ovlp
        .iter()
        .map(|m| pyscf_pbc_df::zlinalg::forder_to_c(m, nao, nao))
        .collect();

    let mut e_u = 0.0_f64;
    for (site, (&val, &alpha)) in resolved
        .indices
        .iter()
        .zip(resolved.u_val.iter().zip(resolved.alpha.iter()))
    {
        let m = site.len();
        for k in 0..nkpts {
            // SC[μ, a] = Σ_ν S[μ, ν] C_lo[ν, site[a]]
            let mut sc = CTensor::zeros(nao * m);
            for mu in 0..nao {
                for (a, &idx) in site.iter().enumerate() {
                    let mut re = 0.0_f64;
                    let mut im = 0.0_f64;
                    for nu in 0..nao {
                        let (ar, ai) = (s[k].re[mu * nao + nu], s[k].im[mu * nao + nu]);
                        let (br, bi) =
                            (c_ao_lo[k].re[nu + idx * nao], c_ao_lo[k].im[nu + idx * nao]);
                        re += ar * br - ai * bi;
                        im += ar * bi + ai * br;
                    }
                    sc.re[mu * m + a] = re;
                    sc.im[mu * m + a] = im;
                }
            }
            let _ = nlo;

            for (spin, dmset) in dms.iter().enumerate() {
                // P = SC^H D SC
                let p = triple(&sc, &dmset[k], nao, m);
                let tr = (0..m).map(|i| p.re[i * m + i]).sum::<f64>();
                let mut tr_pp = 0.0_f64;
                for i in 0..m {
                    for j in 0..m {
                        let (ar, ai) = (p.re[i * m + j], p.im[i * m + j]);
                        let (br, bi) = (p.re[j * m + i], p.im[j * m + i]);
                        tr_pp += ar * br - ai * bi;
                    }
                }
                e_u += weight * (val * 0.5) * (tr - tr_pp * 0.5);

                // vhub_loc = (I − P)·(U/2) [+ α·I]
                let mut vloc = CTensor::zeros(m * m);
                for i in 0..m {
                    for j in 0..m {
                        vloc.re[i * m + j] = -p.re[i * m + j] * (val * 0.5);
                        vloc.im[i * m + j] = -p.im[i * m + j] * (val * 0.5);
                    }
                    vloc.re[i * m + i] += val * 0.5;
                }
                if let Some(a) = alpha {
                    e_u += weight * a * tr;
                    for i in 0..m {
                        vloc.re[i * m + i] += a;
                    }
                }
                // vhub = SC · vloc · SC^H
                let vhub = expand(&sc, &vloc, nao, m);
                let nchan = vxc.len();
                let target = vxc
                    .get_mut(spin)
                    .and_then(|c| c.get_mut(k))
                    .ok_or_else(|| {
                        err(format!(
                            "DFT+U: density channel {spin} at k = {k} has no matching \
                             potential block ({nchan} channels)"
                        ))
                    })?;
                for i in 0..nao * nao {
                    target.re[i] += vhub.re[i];
                    target.im[i] += vhub.im[i];
                }
            }
        }
    }
    Ok(e_u)
}

/// `SC^H · D · SC` — an `m x m` block.
fn triple(sc: &CTensor, dm: &CTensor, nao: usize, m: usize) -> CTensor {
    // t[μ, a] = Σ_ν D[μ, ν] SC[ν, a]
    let mut t = CTensor::zeros(nao * m);
    for mu in 0..nao {
        for a in 0..m {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for nu in 0..nao {
                let (ar, ai) = (dm.re[mu * nao + nu], dm.im[mu * nao + nu]);
                let (br, bi) = (sc.re[nu * m + a], sc.im[nu * m + a]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            t.re[mu * m + a] = re;
            t.im[mu * m + a] = im;
        }
    }
    let mut p = CTensor::zeros(m * m);
    for a in 0..m {
        for b in 0..m {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                let (ar, ai) = (sc.re[mu * m + a], -sc.im[mu * m + a]);
                let (br, bi) = (t.re[mu * m + b], t.im[mu * m + b]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            p.re[a * m + b] = re;
            p.im[a * m + b] = im;
        }
    }
    p
}

/// `SC · V · SC^H` — back to the `nao x nao` AO basis.
fn expand(sc: &CTensor, v: &CTensor, nao: usize, m: usize) -> CTensor {
    // t[μ, b] = Σ_a SC[μ, a] V[a, b]
    let mut t = CTensor::zeros(nao * m);
    for mu in 0..nao {
        for b in 0..m {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for a in 0..m {
                let (ar, ai) = (sc.re[mu * m + a], sc.im[mu * m + a]);
                let (br, bi) = (v.re[a * m + b], v.im[a * m + b]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            t.re[mu * m + b] = re;
            t.im[mu * m + b] = im;
        }
    }
    let mut out = CTensor::zeros(nao * nao);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for b in 0..m {
                let (ar, ai) = (t.re[mu * m + b], t.im[mu * m + b]);
                let (br, bi) = (sc.re[nu * m + b], -sc.im[nu * m + b]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            out.re[mu * nao + nu] = re;
            out.im[mu * nao + nu] = im;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The two drivers
// ---------------------------------------------------------------------------

/// `KRKSpU` — restricted DFT+U (`krkspu.py:180-325`).
#[derive(Debug)]
pub struct Krkspu {
    /// The underlying KS object; owns the cell, the k-points and the grid.
    pub ks: Krks,
    /// The Hubbard configuration.
    pub u: HubbardU,
    e_u: std::cell::Cell<f64>,
}

impl Krkspu {
    /// Wrap a [`Krks`] with a Hubbard `U`.
    pub fn new(ks: Krks, u: HubbardU) -> Self {
        Self {
            ks,
            u,
            e_u: std::cell::Cell::new(0.0),
        }
    }

    /// `E_U` of the last `get_veff`.
    pub fn e_u(&self) -> f64 {
        self.e_u.get()
    }

    /// `get_veff` with the Hubbard term folded in — `krkspu.py:37-65`.
    ///
    /// # Errors
    /// Propagates the KS `get_veff` and the Hubbard build.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
    ) -> Result<(KDms, KsEnergyTags, f64), PbcDftError> {
        let (mut v, tags) = self.ks.get_veff_tagged(dms, None)?;
        let e_u = add_vhubbard(&mut v, self.ks.cell(), self.ks.kpts(), dms, &self.u)?;
        self.e_u.set(e_u);
        Ok((v, tags, e_u))
    }

    /// The DFT+U electronic energy — `krkspu.py:139-160`.
    ///
    /// # Errors
    /// Propagates the `get_veff`.
    pub fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
    ) -> Result<f64, PyscfRsError> {
        let (_, tags, e_u) = self
            .get_veff_tagged(dms)
            .map_err(crate::krks::unwrap_err)?;
        let nao = self.ks.cell().mol.nao_nr;
        let weight = 1.0 / h1e.len() as f64;
        let mut e1 = 0.0_f64;
        for (k, h) in h1e.iter().enumerate() {
            e1 += pyscf_pbc_scf::krdm::trace_ab(&dms[0][k], h, nao).0;
        }
        Ok(e1 * weight + tags.ecoul + tags.exc + e_u)
    }
}

/// `KUKSpU` — unrestricted DFT+U (`kukspu.py`).
#[derive(Debug)]
pub struct Kukspu {
    /// The underlying KS object.
    pub ks: Kuks,
    /// The Hubbard configuration.
    pub u: HubbardU,
    e_u: std::cell::Cell<f64>,
}

impl Kukspu {
    /// Wrap a [`Kuks`] with a Hubbard `U`.
    pub fn new(ks: Kuks, u: HubbardU) -> Self {
        Self {
            ks,
            u,
            e_u: std::cell::Cell::new(0.0),
        }
    }

    /// `E_U` of the last `get_veff`.
    pub fn e_u(&self) -> f64 {
        self.e_u.get()
    }

    /// `get_veff` with the Hubbard term folded in — `kukspu.py:37-63`.
    ///
    /// # Errors
    /// Propagates the KS `get_veff` and the Hubbard build.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
    ) -> Result<(KDms, KsEnergyTags, f64), PbcDftError> {
        let (mut v, tags) = self.ks.get_veff_tagged(dms, None)?;
        let e_u = add_vhubbard(&mut v, self.ks.cell(), self.ks.kpts(), dms, &self.u)?;
        self.e_u.set(e_u);
        Ok((v, tags, e_u))
    }
}
