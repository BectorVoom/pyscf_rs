//! The `eval_xc_eff` layer — plan 12-01.
//!
//! Ports `pyscf/dft/numint.py:LibXCMixin.eval_xc_eff` together with the
//! transformation it delegates to, `pyscf/dft/xc_deriv.py:transform_vxc`
//! (`:186-244`) and `transform_fxc` (`:246-...`). The periodic `NumInt`
//! consumes ONLY this shape, exactly as `pbc/dft/numint.py:nr_rks` does
//! (`numint.py:361`, `wv = weight * vxc`).
//!
//! # Why this module exists rather than reusing `pyscf_dft::NumInt::eval_xc`
//!
//! `XcOutput::vsigma` from the closed-shell path carries `∂f/∂γ_aa` — the
//! spin-resolved gradient derivative of the `A_B_GAA_GAB_GBB` variable set the
//! xcfun CPU kernels expose — NOT the unpolarized `∂f/∂σ` that upstream's
//! `vxc[1]` means. The two differ by the `γ_ab` channel:
//!
//! ```text
//! a = b = ρ/2,  γ_aa = γ_ab = γ_bb = σ/4
//! ∂f/∂σ = (∂f/∂γ_aa + ∂f/∂γ_ab + ∂f/∂γ_bb)/4 = (2·∂f/∂γ_aa + ∂f/∂γ_ab)/4
//! ```
//!
//! and `∂f/∂γ_ab` is nonzero for every GGA correlation functional. This module
//! therefore always drives the SPIN-RESOLVED backend entry point and does the
//! chain rule itself, for both the closed- and the open-shell case. That also
//! means one code path covers `nr_rks` and `nr_uks`.
//!
//! # The `eff` layout
//!
//! `wv` is indexed `[spin][var][grid]` with `nvar = 1` (LDA) or `4` (GGA):
//!
//! ```text
//! LDA  vp[0]   = ∂f/∂ρ
//! GGA  vp[0]   = ∂f/∂ρ
//!      vp[1:4] = 2 · ∂f/∂σ · ∇ρ                       (spin-unpolarized)
//!      vp[s,1:4] = Σ_t stack_fg[s,t] · ∇ρ_t           (spin-polarized), with
//!      stack_fg = [[2·f_aa, f_ab], [f_ab, 2·f_bb]]    (`xc_deriv.py:_stack_fg`)
//! ```
//!
//! Second derivatives (`fxc`) follow `transform_fxc`: the `(nvar, nvar, ngrids)`
//! (RKS) or `(2, nvar, 2, nvar, ngrids)` (UKS) kernel tensor.

use pyscf_dft::{DerivOrder, Family, XcBackend};

use crate::error::PbcDftError;

/// Functional family, in the two flavours the periodic grid loop supports.
///
/// Meta-GGA is deliberately absent: the periodic AO evaluator ships value +
/// `deriv1` only ([`pyscf_pbc_gto::eval_ao_kpts`] over `GTOval_sph_deriv1`), so
/// τ cannot be formed. A meta-GGA `xc_code` is rejected at
/// [`XcType::of`] rather than silently integrated as a GGA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcType {
    /// ρ only — `GTOval_sph`, `ao_deriv = 0`.
    Lda,
    /// ρ and ∇ρ — `GTOval_sph_deriv1`, `ao_deriv = 1`.
    Gga,
}

impl XcType {
    /// Number of density variables per spin channel: 1 for LDA, 4 for GGA.
    pub fn nvar(self) -> usize {
        match self {
            XcType::Lda => 1,
            XcType::Gga => 4,
        }
    }

    /// The AO derivative order the grid loop must evaluate.
    pub fn ao_deriv(self) -> u32 {
        match self {
            XcType::Lda => 0,
            XcType::Gga => 1,
        }
    }

    /// The `eval_gto` name producing those components.
    pub fn eval_gto_name(self) -> &'static str {
        match self {
            XcType::Lda => "GTOval_sph",
            XcType::Gga => "GTOval_sph_deriv1",
        }
    }

    /// Number of AO components (`(deriv+1)(deriv+2)(deriv+3)/6`).
    pub fn ncomp(self) -> usize {
        match self {
            XcType::Lda => 1,
            XcType::Gga => 4,
        }
    }

    /// `ni._xc_type(xc_code)` — `numint.py:LibXCMixin._xc_type`.
    ///
    /// Resolved in the xcfun namespace, which is the default backend of
    /// [`pyscf_dft::XcBackend`] and therefore the one this crate evaluates in.
    ///
    /// # Errors
    /// [`PbcDftError`] when the functional string does not parse, or when it
    /// resolves to a meta-GGA (see the type docs).
    pub fn of(xc_code: &str) -> Result<Self, PbcDftError> {
        let backend = XcBackend::default();
        // Parse and classify in the SAME namespace: the ids only mean something
        // to the backend that emitted them. Asking the backend rather than
        // consulting a table here keeps one source of truth.
        let spec = backend.parse(xc_code).map_err(dft_err)?;
        match backend.family(&spec).map_err(dft_err)? {
            Family::Lda => Ok(XcType::Lda),
            Family::Gga => Ok(XcType::Gga),
            Family::Mgga => Err(err(format!(
                "pbc NumInt: meta-GGA functional '{xc_code}' needs the kinetic-energy \
                 density, which the periodic AO evaluator does not produce (value + \
                 deriv1 only)"
            ))),
        }
    }
}

/// `(hyb, alpha, omega)` for `xc_code` — `numint.rsh_and_hybrid_coeff`.
///
/// # Errors
/// [`PbcDftError`] when the functional string does not parse.
pub fn rsh_and_hybrid_coeff(xc_code: &str) -> Result<(f64, f64, f64), PbcDftError> {
    // Delegated to the BACKEND. Reading `spec.hyb()` here would be wrong under
    // the libxc default: the libxc parser resolves `pbe0` to the single compound
    // id 406 and reports `hyb = 0`, with the 0.25 reachable only by asking the
    // library. That would make every hybrid look pure and drop the exact
    // exchange from `veff::get_jk` entirely.
    XcBackend::default()
        .rsh_and_hybrid_coeff(xc_code)
        .map_err(dft_err)
}

/// `ni.libxc.is_hybrid_xc(xc)` — a nonzero `hyb` OR a nonzero long-range
/// coefficient. Matches upstream's test, which is `hyb != 0 or (omega != 0 and
/// alpha != 0)` folded into `parse_xc`'s triple.
///
/// # Errors
/// As [`rsh_and_hybrid_coeff`].
pub fn is_hybrid_xc(xc_code: &str) -> Result<bool, PbcDftError> {
    let (omega, alpha, hyb) = rsh_and_hybrid_coeff(xc_code)?;
    Ok(hyb != 0.0 || (omega != 0.0 && alpha != 0.0))
}

/// The density block one spin channel of the grid loop produces.
///
/// `rho[0]` is ρ; for GGA `rho[1..4]` are `∂ρ/∂{x,y,z}`. Row `v` is
/// `ngrids` long, so the whole block is `nvar * ngrids` in `[var][grid]`
/// order — upstream's `(nvar, N)` C-order array.
#[derive(Debug, Clone, PartialEq)]
pub struct RhoEff {
    /// `nvar` (1 or 4).
    pub nvar: usize,
    /// Grid-point count.
    pub ngrids: usize,
    /// `data[v * ngrids + g]`.
    pub data: Vec<f64>,
}

impl RhoEff {
    /// An all-zero block.
    pub fn zeros(ty: XcType, ngrids: usize) -> Self {
        let nvar = ty.nvar();
        Self {
            nvar,
            ngrids,
            data: vec![0.0; nvar * ngrids],
        }
    }

    /// Row `v` of the block.
    pub fn row(&self, v: usize) -> &[f64] {
        &self.data[v * self.ngrids..(v + 1) * self.ngrids]
    }

    /// Mutable row `v`.
    pub fn row_mut(&mut self, v: usize) -> &mut [f64] {
        let n = self.ngrids;
        &mut self.data[v * n..(v + 1) * n]
    }

    /// `self += other`, element-wise. Both blocks must have the same shape.
    pub fn add_assign(&mut self, other: &RhoEff) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// `self *= s`.
    pub fn scale(&mut self, s: f64) {
        for a in self.data.iter_mut() {
            *a *= s;
        }
    }

    /// The `[p0, p1)` grid slice as a standalone block.
    pub fn slice(&self, p0: usize, p1: usize) -> RhoEff {
        let n = p1 - p0;
        let mut data = Vec::with_capacity(self.nvar * n);
        for v in 0..self.nvar {
            data.extend_from_slice(&self.row(v)[p0..p1]);
        }
        RhoEff {
            nvar: self.nvar,
            ngrids: n,
            data,
        }
    }

    /// Append `block`'s grid points to `self` (both `nvar`-compatible).
    pub fn append(&mut self, block: &RhoEff) {
        let mut data = Vec::with_capacity(self.data.len() + block.data.len());
        for v in 0..self.nvar {
            data.extend_from_slice(self.row(v));
            data.extend_from_slice(block.row(v));
        }
        self.ngrids += block.ngrids;
        self.data = data;
    }
}

/// `exc` plus the transformed first derivative — the `(exc, vxc_eff)` pair
/// `eval_xc_eff(..., deriv=1)` returns.
#[derive(Debug, Clone, Default)]
pub struct VxcEff {
    /// Per-grid XC energy DENSITY PER PARTICLE, i.e. upstream's `exc`
    /// (`f / ρ_total`). The grid loop multiplies it by `den = ρ·w`.
    pub exc: Vec<f64>,
    /// `[spin][var][grid]`, flattened `spin * nvar * ngrids + var * ngrids + g`.
    /// `nspin` is 1 for the closed-shell call and 2 for the open-shell one.
    pub vp: Vec<f64>,
    /// 1 or 2.
    pub nspin: usize,
    /// 1 or 4.
    pub nvar: usize,
    /// Grid-point count.
    pub ngrids: usize,
}

impl VxcEff {
    /// The `(spin, var)` row.
    pub fn row(&self, spin: usize, var: usize) -> &[f64] {
        let base = (spin * self.nvar + var) * self.ngrids;
        &self.vp[base..base + self.ngrids]
    }
}

/// The transformed SECOND derivative — `eval_xc_eff(..., deriv=2)[2]`.
///
/// Closed shell (`nspin == 1`): `[var][var][grid]`, upstream's `(nvar, nvar, N)`.
/// Open shell (`nspin == 2`): `[spin][var][spin][var][grid]`, upstream's
/// `(2, nvar, 2, nvar, N)`.
#[derive(Debug, Clone, Default)]
pub struct FxcEff {
    /// Flattened in the index order given above.
    pub data: Vec<f64>,
    /// 1 or 2.
    pub nspin: usize,
    /// 1 or 4.
    pub nvar: usize,
    /// Grid-point count.
    pub ngrids: usize,
}

impl FxcEff {
    /// `fxc[a, x, b, y, g]` for the open-shell layout (`a = b = 0` reproduces
    /// the closed-shell one).
    pub fn at(&self, a: usize, x: usize, b: usize, y: usize, g: usize) -> f64 {
        self.data[self.index(a, x, b, y) * self.ngrids + g]
    }

    fn index(&self, a: usize, x: usize, b: usize, y: usize) -> usize {
        if self.nspin == 1 {
            x * self.nvar + y
        } else {
            ((a * self.nvar + x) * 2 + b) * self.nvar + y
        }
    }

    /// The `[p0, p1)` grid slice.
    pub fn slice(&self, p0: usize, p1: usize) -> FxcEff {
        let n = p1 - p0;
        let blocks = self.data.len() / self.ngrids;
        let mut data = Vec::with_capacity(blocks * n);
        for b in 0..blocks {
            data.extend_from_slice(&self.data[b * self.ngrids + p0..b * self.ngrids + p1]);
        }
        FxcEff {
            data,
            nspin: self.nspin,
            nvar: self.nvar,
            ngrids: n,
        }
    }
}

/// Raw spin-resolved derivatives at one grid point, as the backend hands them
/// over. `f` is the energy density per unit VOLUME.
#[derive(Debug, Clone, Copy, Default)]
struct Raw1 {
    fa: f64,
    fb: f64,
    gaa: f64,
    gab: f64,
    gbb: f64,
}

/// `eval_xc_eff(xc_code, rho, deriv=1, spin=0)` — the CLOSED-SHELL branch.
///
/// `rho` is the total density block. Internally the functional is evaluated at
/// `ρ_a = ρ_b = ρ/2`, `γ_aa = γ_ab = γ_bb = σ/4` (the only variable set the
/// xcfun CPU kernels expose), and the chain rule back to the total-density
/// variables is
///
/// ```text
/// ∂f/∂ρ = ∂f/∂a                                (a = b, so the two halves add to one)
/// ∂f/∂σ = (∂f/∂γ_aa + ∂f/∂γ_ab + ∂f/∂γ_bb)/4
/// ```
///
/// # Errors
/// [`PbcDftError`] on a functional-name or backend failure.
pub fn eval_xc_eff_rks(xc_code: &str, rho: &RhoEff) -> Result<VxcEff, PbcDftError> {
    let ty = XcType::of(xc_code)?;
    let ngrids = rho.ngrids;
    let raw = eval_raw1_rks(xc_code, ty, rho)?;

    let mut exc = vec![0.0_f64; ngrids];
    let nvar = ty.nvar();
    let mut vp = vec![0.0_f64; nvar * ngrids];
    let (f, d) = raw;
    for g in 0..ngrids {
        // `exc` is per PARTICLE — upstream divides the energy density by ρ and
        // the grid loop multiplies by `den = ρ·w`. Guard the vacuum, where both
        // ρ and f are zero.
        let r = rho.row(0)[g];
        exc[g] = if r == 0.0 { 0.0 } else { f[g] / r };
        vp[g] = d[g].fa;
    }
    if ty == XcType::Gga {
        for g in 0..ngrids {
            let dfds = (d[g].gaa + d[g].gab + d[g].gbb) * 0.25;
            // `xc_deriv.py:239` — vp[1:4] = 2 * fg * rho[1:4]
            for c in 1..4 {
                vp[c * ngrids + g] = 2.0 * dfds * rho.row(c)[g];
            }
        }
    }
    Ok(VxcEff {
        exc,
        vp,
        nspin: 1,
        nvar,
        ngrids,
    })
}

/// `eval_xc_eff(xc_code, (rho_a, rho_b), deriv=1, spin=1)` — the OPEN-SHELL
/// branch (`xc_deriv.py:227-236`).
///
/// # Errors
/// [`PbcDftError`] on a functional-name or backend failure, or when the two
/// spin blocks disagree in shape.
pub fn eval_xc_eff_uks(
    xc_code: &str,
    rho_a: &RhoEff,
    rho_b: &RhoEff,
) -> Result<VxcEff, PbcDftError> {
    let ty = XcType::of(xc_code)?;
    if rho_a.ngrids != rho_b.ngrids || rho_a.nvar != rho_b.nvar {
        return Err(err(
            "pbc NumInt: alpha and beta density blocks differ in shape",
        ));
    }
    let ngrids = rho_a.ngrids;
    let nvar = ty.nvar();
    let (f, d) = eval_raw1_uks(xc_code, ty, rho_a, rho_b)?;

    let mut exc = vec![0.0_f64; ngrids];
    let mut vp = vec![0.0_f64; 2 * nvar * ngrids];
    for g in 0..ngrids {
        let rt = rho_a.row(0)[g] + rho_b.row(0)[g];
        exc[g] = if rt == 0.0 { 0.0 } else { f[g] / rt };
        vp[g] = d[g].fa;
        vp[nvar * ngrids + g] = d[g].fb;
    }
    if ty == XcType::Gga {
        // `_stack_fg`: [[2 f_aa, f_ab], [f_ab, 2 f_bb]] contracted with ∇ρ.
        for g in 0..ngrids {
            let (gaa, gab, gbb) = (d[g].gaa, d[g].gab, d[g].gbb);
            for c in 1..4 {
                let ga = rho_a.row(c)[g];
                let gb = rho_b.row(c)[g];
                vp[c * ngrids + g] = 2.0 * gaa * ga + gab * gb;
                vp[(nvar + c) * ngrids + g] = gab * ga + 2.0 * gbb * gb;
            }
        }
    }
    Ok(VxcEff {
        exc,
        vp,
        nspin: 2,
        nvar,
        ngrids,
    })
}

/// `eval_xc_eff(..., deriv=2, spin=0)[2]` — the closed-shell XC KERNEL.
///
/// Second derivatives are formed by central differences of the ANALYTIC first
/// derivatives with respect to the total-density variables `(ρ, σ)`. The
/// backend exposes order-2 output only in the spin-resolved variable set, whose
/// chain rule back to `(ρ, σ)` for a symmetric split is a 5x5 contraction that
/// would still need the `γ_ab` cross-terms this crate cannot read individually;
/// differentiating the (exact) first derivatives keeps ONE code path and is
/// accurate to `~1e-9` relative, which is what a response calculation needs.
///
/// The step is scaled to each variable's magnitude so the kernel behaves at the
/// vacuum tail as well as in the core.
///
/// # Errors
/// As [`eval_xc_eff_rks`].
pub fn eval_fxc_eff_rks(xc_code: &str, rho: &RhoEff) -> Result<FxcEff, PbcDftError> {
    let ty = XcType::of(xc_code)?;
    let ngrids = rho.ngrids;
    let nvar = ty.nvar();
    let mut data = vec![0.0_f64; nvar * nvar * ngrids];

    // Central difference of every `vp` row with respect to every density
    // variable — `transform_fxc`'s (nvar, nvar, N) kernel.
    for y in 0..nvar {
        let (plus, minus, h) = displaced_rks(xc_code, rho, y)?;
        for x in 0..nvar {
            for g in 0..ngrids {
                let step = h[g];
                let v = if step == 0.0 {
                    0.0
                } else {
                    (plus.row(0, x)[g] - minus.row(0, x)[g]) / (2.0 * step)
                };
                data[(x * nvar + y) * ngrids + g] = v;
            }
        }
    }
    Ok(FxcEff {
        data,
        nspin: 1,
        nvar,
        ngrids,
    })
}

/// `eval_xc_eff(..., deriv=2, spin=1)[2]` — the open-shell XC kernel, built the
/// same way as [`eval_fxc_eff_rks`].
///
/// # Errors
/// As [`eval_xc_eff_uks`].
pub fn eval_fxc_eff_uks(
    xc_code: &str,
    rho_a: &RhoEff,
    rho_b: &RhoEff,
) -> Result<FxcEff, PbcDftError> {
    let ty = XcType::of(xc_code)?;
    let ngrids = rho_a.ngrids;
    let nvar = ty.nvar();
    let mut data = vec![0.0_f64; 2 * nvar * 2 * nvar * ngrids];
    for b in 0..2 {
        for y in 0..nvar {
            let (plus, minus, h) = displaced_uks(xc_code, rho_a, rho_b, b, y)?;
            for a in 0..2 {
                for x in 0..nvar {
                    for g in 0..ngrids {
                        let step = h[g];
                        let v = if step == 0.0 {
                            0.0
                        } else {
                            (plus.row(a, x)[g] - minus.row(a, x)[g]) / (2.0 * step)
                        };
                        let idx = ((a * nvar + x) * 2 + b) * nvar + y;
                        data[idx * ngrids + g] = v;
                    }
                }
            }
        }
    }
    Ok(FxcEff {
        data,
        nspin: 2,
        nvar,
        ngrids,
    })
}

/// Per-point differentiation step for variable `y` of a density block.
fn steps(rho: &RhoEff, y: usize) -> Vec<f64> {
    const REL: f64 = 1e-5;
    const FLOOR: f64 = 1e-9;
    (0..rho.ngrids)
        .map(|g| {
            let scale = rho.row(y)[g].abs().max(rho.row(0)[g].abs());
            REL * scale.max(FLOOR)
        })
        .collect()
}

/// The `±h` displacement of variable `y`, and the step it used.
///
// The index walks `up`, `dn` and `h` together — the shared index IS the point.
#[allow(clippy::needless_range_loop)]
fn displaced_rks(
    xc_code: &str,
    rho: &RhoEff,
    y: usize,
) -> Result<(VxcEff, VxcEff, Vec<f64>), PbcDftError> {
    let h = steps(rho, y);
    let mut up = rho.clone();
    let mut dn = rho.clone();
    for g in 0..rho.ngrids {
        up.row_mut(y)[g] += h[g];
        dn.row_mut(y)[g] -= h[g];
    }
    Ok((
        eval_xc_eff_rks(xc_code, &up)?,
        eval_xc_eff_rks(xc_code, &dn)?,
        h,
    ))
}

// As `displaced_rks`: one index over four parallel blocks.
#[allow(clippy::needless_range_loop)]
fn displaced_uks(
    xc_code: &str,
    rho_a: &RhoEff,
    rho_b: &RhoEff,
    b: usize,
    y: usize,
) -> Result<(VxcEff, VxcEff, Vec<f64>), PbcDftError> {
    let target = if b == 0 { rho_a } else { rho_b };
    let h = steps(target, y);
    let mut ua = rho_a.clone();
    let mut ub = rho_b.clone();
    let mut da = rho_a.clone();
    let mut db = rho_b.clone();
    for g in 0..target.ngrids {
        if b == 0 {
            ua.row_mut(y)[g] += h[g];
            da.row_mut(y)[g] -= h[g];
        } else {
            ub.row_mut(y)[g] += h[g];
            db.row_mut(y)[g] -= h[g];
        }
    }
    Ok((
        eval_xc_eff_uks(xc_code, &ua, &ub)?,
        eval_xc_eff_uks(xc_code, &da, &db)?,
        h,
    ))
}

/// Drive the spin-resolved backend for a CLOSED-SHELL block: `(f, derivatives)`.
fn eval_raw1_rks(
    xc_code: &str,
    ty: XcType,
    rho: &RhoEff,
) -> Result<(Vec<f64>, Vec<Raw1>), PbcDftError> {
    let ngrids = rho.ngrids;
    let half: Vec<f64> = rho.row(0).iter().map(|r| r * 0.5).collect();
    let sigma_q: Option<Vec<f64>> = if ty == XcType::Gga {
        Some(
            (0..ngrids)
                .map(|g| {
                    let (x, y, z) = (rho.row(1)[g], rho.row(2)[g], rho.row(3)[g]);
                    (x * x + y * y + z * z) * 0.25
                })
                .collect(),
        )
    } else {
        None
    };
    backend_eval(
        xc_code,
        &half,
        &half,
        sigma_q.as_deref(),
        sigma_q.as_deref(),
        sigma_q.as_deref(),
    )
}

/// Drive the spin-resolved backend for an OPEN-SHELL pair.
fn eval_raw1_uks(
    xc_code: &str,
    ty: XcType,
    rho_a: &RhoEff,
    rho_b: &RhoEff,
) -> Result<(Vec<f64>, Vec<Raw1>), PbcDftError> {
    let ngrids = rho_a.ngrids;
    let ra = rho_a.row(0).to_vec();
    let rb = rho_b.row(0).to_vec();
    if ty == XcType::Lda {
        return backend_eval(xc_code, &ra, &rb, None, None, None);
    }
    let dot = |p: &RhoEff, q: &RhoEff| -> Vec<f64> {
        (0..ngrids)
            .map(|g| {
                p.row(1)[g] * q.row(1)[g] + p.row(2)[g] * q.row(2)[g] + p.row(3)[g] * q.row(3)[g]
            })
            .collect()
    };
    let saa = dot(rho_a, rho_a);
    let sab = dot(rho_a, rho_b);
    let sbb = dot(rho_b, rho_b);
    backend_eval(xc_code, &ra, &rb, Some(&saa), Some(&sab), Some(&sbb))
}

fn backend_eval(
    xc_code: &str,
    ra: &[f64],
    rb: &[f64],
    saa: Option<&[f64]>,
    sab: Option<&[f64]>,
    sbb: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<Raw1>), PbcDftError> {
    let backend = XcBackend::default();
    let spec = backend.parse(xc_code).map_err(dft_err)?;
    let out = backend
        .eval_uks(&spec, ra, rb, saa, sab, sbb, DerivOrder::Vxc)
        .map_err(dft_err)?;
    let n = ra.len();
    let mut d = vec![Raw1::default(); n];
    for (g, item) in d.iter_mut().enumerate() {
        item.fa = out.vrho_a[g];
        item.fb = out.vrho_b[g];
        if saa.is_some() {
            item.gaa = out.vsigma_aa[g];
            item.gab = out.vsigma_ab[g];
            item.gbb = out.vsigma_bb[g];
        }
    }
    Ok((out.exc, d))
}

pub(crate) fn err(msg: impl Into<String>) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(msg.into()),
    ))
}

fn dft_err(e: pyscf_dft::DftError) -> PbcDftError {
    err(format!("pbc NumInt: XC backend: {e}"))
}
