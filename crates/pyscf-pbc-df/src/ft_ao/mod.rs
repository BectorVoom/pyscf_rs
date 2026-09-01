//! Analytic Fourier transform of AO pairs — `pyscf/pbc/df/ft_ao.py` (plans 13-01
//! and 13-03).
//!
//! ```text
//! ft_aopair_kpts[k, μν, G] = Σ_L e^{i k·L} ∫ φ_μ(r) φ_ν(r − L) e^{−i(G+q)·r} dr
//! ```
//!
//! # This is a DIRECT lattice sum (D-PBC-21)
//!
//! Upstream spends ~600 of `ft_ao.py`'s 790 lines on `_RangeSeparatedCell` (a
//! partial basis de-contraction into steep/local/smooth blocks) and
//! `ExtendedMole` (a Born–von-Kármán supercell `Mole`). **Neither changes the
//! answer**: the first decontracts and then `recontract`s, and the second only
//! *drops* image shells whose Schwarz bound is below `cell.precision · 1e-2`.
//! Both are screening and cache-blocking devices for a C loop over a supermole.
//!
//! This port implements the definition above over
//! `get_lattice_Ls(rcut = estimate_rcut(cell).max())`, with upstream's own
//! per-shell-pair Schwarz screen ([`estimate_rcut`], ported verbatim). A
//! contracted shell is screened by its most diffuse primitive, which is exactly
//! what `_extract_pgto_params(cell, 'min')` selects, so the de-contraction is
//! never needed.
//!
//! # `rcut` is an explicit parameter, and that is load-bearing
//!
//! Measured on upstream (see `.planning/phases/13-ft-ao-aftdf/measurements/`):
//! `ft_ao.estimate_rcut` is **looser** than `cell.rcut` — 20.420 vs 21.319 Bohr
//! on diamond/`gth-szv` — so upstream's own `ft_aopair[G=0]` misses
//! `pbc_intor("int1e_ovlp")` by 1.554e-9 at gamma. Matching upstream to 1e-10
//! therefore REQUIRES upstream's screening, while proving the McMurchie–Davidson
//! algebra requires a converged sum. Those pull in opposite directions, so the
//! caller chooses: [`RcutChoice::Upstream`] reproduces upstream, and
//! [`RcutChoice::Scaled`] converges (the sum stops changing at ~1.5×`cell.rcut`;
//! 2.0× is bit-identical).
//!
//! # Tuning accuracy — the knob is `cell.precision`, not a radius multiplier
//!
//! `estimate_rcut` targets `cell.precision · 1e-2`, and the three screens
//! ([`FtScreen::Upstream`]) derive their cutoffs from `cell.precision` too, so
//! tightening it moves the radius AND the screens together. That is why it is
//! both the principled knob and the cheap one.
//!
//! **Set it at BUILD time.** `cell.rcut` is a cached field computed during
//! `Cell::build`; assigning `cell.precision = p` afterwards tightens only the
//! estimators that read `precision` at call time (this module's), leaving
//! `cell.rcut` — and therefore `pbc_intor`, Ewald and `eval_gto` — on the
//! original target. Both forms are useful and they measure different things.
//!
//! ## Post-hoc (`cell.precision = p`) — tightens `ft_aopair` alone
//!
//! Diamond/`gth-szv`, gamma. The ERI 8-fold-symmetry residue is a clean probe:
//! the exact ERI is symmetric, so any residue is pure screening error. It is
//! **independent of the mesh** — bit-identical from 1 331 to 19 683 G-vectors —
//! which is what rules out summation roundoff as the cause.
//!
//! | `precision` | `rcut` | images | ERI residue | `get_pp` anti-Hermitian |
//! |---|---|---|---|---|
//! | 1e-8 (default) | 20.420 | 675 | 1.966e-12 | 5.131e-11 |
//! | 1e-10 | 22.297 | 887 (1.31×) | 1.497e-14 | 5.000e-13 |
//! | **1e-12** | **24.020** | **1055 (1.56×)** | **3.842e-16** | **4.647e-15** |
//! | [`RcutChoice::Scaled(1.5)`](RcutChoice::Scaled) | 31.979 | 2315 (3.43×) | 2.914e-16 | 2.665e-15 |
//!
//! `precision = 1e-12` reaches the f64 floor for **2.2× fewer lattice images**
//! than `Scaled(1.5)`. `Scaled` inflates a radius that was sized for a looser
//! target while leaving the screens loose; tightening `precision` sizes both.
//!
//! ## Build time (`CellBuildArgs { precision, .. }`) — tightens the whole cell
//!
//! Now `pbc_intor` converges too, which is what Gate 1b's floor was made of:
//!
//! | `precision` | `cell.rcut` | `\|ft[G=0] − int1e_ovlp\|` |
//! |---|---|---|
//! | 1e-8 (default) | 21.319 | 1.189e-9 |
//! | 1e-10 | 23.193 | 1.142e-11 |
//! | **1e-12** | **24.910** | **8.416e-14** |
//!
//! **Summation is NOT the bottleneck and does not need `oracle_sum`.** Once
//! screening is converged the residue floor is 2.914e-16 at 1 331 G-vectors and
//! 2.923e-16 at 9 261 — a 0.3% rise over 7× more terms, extrapolating to
//! ~3.0e-16 at the default mesh 47. The G-sums are well conditioned, so
//! pairwise accumulation would buy nothing measurable.
//!
//! **The trade-off every accuracy setting shares:** converging the sum moves the
//! result AWAY from upstream's truncated value. Keep the default when
//! reproducing upstream is the goal (Gate 3); tighten `cell.precision` when the
//! exact answer is (Gate 1c).

pub mod mcmurchie;
pub mod rs_cell;
pub mod single;
pub mod supmol;

pub use rs_cell::{RsCell, LOCAL_BASIS, SMOOTH_BASIS, STEEP_BASIS};
pub use single::{fake_nuc, ft_ao_kpt, ft_ao_mol};
pub use supmol::ExtendedMole;

use pyscf_algebra::CTensor;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use pyscf_kernels::pbc::ft_aopair::FtAopairTables;
use pyscf_kernels::{cart_powers, common_fac_sp};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use mcmurchie::{e_coefficients, e_len, k_ab};

/// Upstream's `ft_ao.estimate_rcut` tightening factor: `precision · 1e-2`.
///
/// `ft_ao.py:749-753` explains it — the plain `cell.precision` converges the
/// integrals but leaves a hermitian-symmetry residue that post-HF methods
/// assume away.
pub const RCUT_PRECISION_SCALE: f64 = 1e-2;

/// Magnitude below which one `(shell pair, image, primitive pair)` record is
/// discarded.
///
/// Deliberately far tighter than `cell.precision`: the lattice sum has
/// `nimgs × nprim²` records per shell pair — tens of thousands on a real cell —
/// so a per-record cutoff at the integral's target accuracy accumulates to
/// hundreds of times that accuracy. Screening is the image list's job
/// ([`estimate_rcut`]); this constant only drops records that contribute
/// nothing at f64.
pub const FT_RECORD_EPS: f64 = 1e-18;

/// Which lattice-sum radius `ft_aopair` should use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RcutChoice {
    /// `estimate_rcut(cell).max()` — upstream's own screen. Use this whenever
    /// the result is compared against upstream (`Gate 3`).
    Upstream,
    /// `factor × cell.rcut`. `1.5` converges the sum (2.0 is bit-identical);
    /// use it to prove the algebra against a matching-`Ls` overlap (`Gate 1c`).
    ///
    /// **For accuracy, prefer tightening `cell.precision`** — see the module
    /// docs' "Tuning accuracy" section. It reaches the same machine-precision
    /// floor at a materially smaller radius, because it tightens the SCREENS
    /// too rather than only inflating the radius.
    Scaled(f64),
    /// An explicit radius in Bohr.
    Explicit(f64),
}

impl RcutChoice {
    /// The radius in Bohr this choice resolves to for `cell`.
    ///
    /// # Errors
    /// Propagates `cell.rcut` on an unbuilt cell.
    pub fn resolve_for(self, cell: &Cell) -> Result<f64, PbcDfError> {
        self.resolve(cell)
    }

    fn resolve(self, cell: &Cell) -> Result<f64, PbcDfError> {
        Ok(match self {
            RcutChoice::Upstream => estimate_rcut(cell, None)?
                .into_iter()
                .fold(0.0f64, f64::max),
            RcutChoice::Scaled(f) => f * cell.try_rcut().map_err(PbcDfError::from)?,
            RcutChoice::Explicit(r) => r,
        })
    }
}

/// How closely to reproduce upstream's truncation.
///
/// Upstream applies TWO screens this port would otherwise skip, and together
/// they are worth **1.55e-9** on diamond — the exact size of upstream's own
/// `ft_aopair[G=0]` vs `int1e_ovlp` gap. Reproducing them is what makes Gate 3
/// (match upstream to 1e-10) a real comparison rather than a measurement of the
/// screening difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FtScreen {
    /// `ExtendedMole.strip_basis` (`ft_ao.py:631-666`) — drop an image whose
    /// ket shell sits farther than that shell's own `estimate_rcut` from every
    /// reference-cell atom — plus `ExtendedMole.get_ovlp_mask`
    /// (`ft_ao.py:669-703`) — drop a `(bra shell, ket image shell)` pair whose
    /// overlap estimate falls under upstream's cutoff.
    Upstream,
    /// Keep every image in the list. **More accurate than upstream**, and
    /// therefore FARTHER from it: use this with a converged `rcut` for Gate 1c,
    /// never for Gate 3.
    None,
}

/// `ft_ao.estimate_rcut(cell, precision)` — `ft_ao.py:744-790`, both refinement
/// passes.
///
/// Per-shell Schwarz-style bound: only the MOST DIFFUSE primitive of each shell
/// is considered (`_extract_pgto_params(cell, 'min')`), and the bra is fixed to
/// the single most diffuse shell in the whole cell.
///
/// # Errors
/// Propagates `cell.rcut` / `cell.vol` access on an unbuilt cell.
pub fn estimate_rcut(cell: &Cell, precision: Option<f64>) -> Result<Vec<f64>, PbcDfError> {
    let precision = precision.unwrap_or(cell.precision * RCUT_PRECISION_SCALE);
    let nbas = cell.mol.nbas;
    if nbas == 0 {
        return Ok(vec![0.0]);
    }
    // `_extract_pgto_params(cell, 'min')`: the smallest exponent of each shell
    // and the `gto_norm`-scaled coefficient that goes with it.
    let (exps, cs) = extract_pgto_params_min(cell);
    let ls: Vec<u32> = (0..nbas)
        .map(|i| cell.mol._bas[i * BAS_SLOTS + ANG_OF] as u32)
        .collect();

    let ai_idx = exps
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite exponents"))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let ai = exps[ai_idx];
    let li = ls[ai_idx] as f64;
    let ci = cs[ai_idx];

    let vol = cell.vol();
    let r0_cell = cell.try_rcut().map_err(PbcDfError::from)?;

    let mut out = Vec::with_capacity(nbas);
    for j in 0..nbas {
        let aj = exps[j];
        let lj = ls[j] as f64;
        let cj = cs[j];
        let aij = ai + aj;
        let lij = li + lj;
        let norm_ang = ((2.0 * li + 1.0) * (2.0 * lj + 1.0)).sqrt() / (4.0 * std::f64::consts::PI);
        let c1 = ci * cj * norm_ang;
        let theta = ai * aj / aij;
        let aij1 = aij.powf(-0.5);
        let mut fac = std::f64::consts::PI.powf(1.5)
            * c1
            * aij1.powf(lij + 3.0)
            * (2.0 * aij / std::f64::consts::PI).powf(0.25)
            * aij.powf(lij);
        fac /= precision;

        // Two Newton-ish passes, exactly as upstream (`ft_ao.py:781-789`).
        let mut r0 = r0_cell;
        for _ in 0..2 {
            let dri = aj * aij1 * r0 + 1.0;
            let drj = ai * aij1 * r0 + 1.0;
            let fl = 2.0 * std::f64::consts::PI / vol * r0 / theta;
            r0 = ((fac * dri.powf(li) * drj.powf(lj) * fl + 1.0).ln() / theta).sqrt();
        }
        out.push(r0);
    }
    Ok(out)
}

/// `pbcgto.cell._extract_pgto_params(cell, 'min')` — the most diffuse primitive
/// of each shell, and the libcint contraction coefficient that goes with it.
///
/// **`cs` is the raw `_env` coefficient, NOT `gto_norm`.** `aft.estimate_ke_cutoff`
/// overwrites `cs` with `gto.gto_norm(ls, exps)`; `ft_ao.estimate_rcut` does not,
/// and conflating the two moves the radius by ~4% (21.186 vs 20.420 Bohr on
/// diamond) — enough to change the image count but not enough to look wrong.
fn extract_pgto_params_min(cell: &Cell) -> (Vec<f64>, Vec<f64>) {
    let mol = &cell.mol;
    let nbas = mol.nbas;
    let mut exps = Vec::with_capacity(nbas);
    let mut cs = Vec::with_capacity(nbas);
    for ib in 0..nbas {
        let row = ib * BAS_SLOTS;
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc = mol._bas[row + PTR_COEFF] as usize;
        // `e.argmin()` — the first minimum, matching numpy.
        let mut idx = 0usize;
        for p in 1..nprim {
            if mol._env[pe + p] < mol._env[pe + idx] {
                idx = p;
            }
        }
        exps.push(mol._env[pe + idx]);
        // `abs(c[idx]).max()` over the contraction columns (F-order).
        let cmax = (0..nctr)
            .map(|ictr| mol._env[pc + ictr * nprim + idx].abs())
            .fold(0.0f64, f64::max);
        cs.push(cmax);
    }
    (exps, cs)
}

/// `_RangeSeparatedCell`'s `ke_cut_threshold` (`ft_ao.py:36`, `KECUT_THRESHOLD`).
pub const KECUT_THRESHOLD: f64 = 10.0;
/// `_RangeSeparatedCell`'s `rcut_threshold` (`ft_ao.py:34`, `RCUT_THRESHOLD`).
pub const RCUT_THRESHOLD: f64 = 1.0;

/// One range-separated primitive group — the screening unit upstream actually
/// uses (`ft_ao.py:267-340`).
///
/// `_RangeSeparatedCell.from_cell` splits each shell's primitives into
/// steep / local / smooth blocks by their individual kinetic-energy cutoff and
/// radius, and each block then gets its OWN `estimate_rcut`. On
/// diamond/`gth-szv` that turns 4 shells into 6: each `s` shell becomes a
/// 3-primitive local block (rcut **16.123**) plus a 1-primitive smooth block
/// (rcut **20.114**), while the `p` shell stays whole (rcut **20.420**).
///
/// **This is not cosmetic.** Screening the steep primitives at 20.1 instead of
/// 16.1 leaves this port MORE converged than upstream and therefore further from
/// it: measured, the difference is 1.55e-9 vs upstream's `ft_aopair`, which is
/// the entire Gate-3 budget. D-PBC-21 declines to port the RS *cell* — the
/// de-contraction and `recontract` round trip — but the grouping RULE is what
/// makes the screens reproducible, so it lives here as a per-primitive label.
#[derive(Debug, Clone, Copy)]
struct PrimGroup {
    l: u32,
    /// Smallest exponent in the group.
    min_exp: f64,
    /// `abs(c[argmin]).max()` over the contraction columns.
    cmax: f64,
    /// This group's own `estimate_rcut`.
    rcut: f64,
}

/// Assign every `(shell, primitive)` to a range-separated group and give each
/// group its `estimate_rcut`.
///
/// Returns `(groups, group_of_prim)` where `group_of_prim[ib][p]` indexes
/// `groups`.
fn range_separated_groups(cell: &Cell) -> Result<(Vec<PrimGroup>, Vec<Vec<usize>>), PbcDfError> {
    let mol = &cell.mol;
    let precision = cell.precision;
    let vol = cell.vol();
    let cell_rcut = cell.try_rcut().map_err(PbcDfError::from)?;
    let pi = std::f64::consts::PI;

    let mut groups: Vec<PrimGroup> = Vec::new();
    let mut group_of_prim: Vec<Vec<usize>> = Vec::with_capacity(mol.nbas);

    for ib in 0..mol.nbas {
        let row = ib * BAS_SLOTS;
        let l = mol._bas[row + ANG_OF] as u32;
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc = mol._bas[row + PTR_COEFF] as usize;

        let es: Vec<f64> = (0..nprim).map(|p| mol._env[pe + p]).collect();
        let abs_cs: Vec<f64> = (0..nprim)
            .map(|p| {
                (0..nctr)
                    .map(|c| mol._env[pc + c * nprim + p].abs())
                    .fold(0.0f64, f64::max)
            })
            .collect();

        // `pbcgto.cell._estimate_ke_cutoff` — already ported per-primitive.
        let smooth: Vec<bool> = (0..nprim)
            .map(|p| {
                pyscf_pbc_gto::cutoff::estimate_ke_cutoff_pgto(
                    es[p], l as i32, abs_cs[p], precision, 0.0,
                ) < KECUT_THRESHOLD
            })
            .collect();

        // `ft_ao.py:329-336` — the per-primitive radius that separates steep
        // from local. Two passes, seeded from `cell.rcut`.
        let norm_ang = ((2.0 * l as f64 + 1.0) / (4.0 * pi)).sqrt();
        let steep: Vec<bool> = (0..nprim)
            .map(|p| {
                if smooth[p] {
                    return false;
                }
                let fac = 2.0 * pi * abs_cs[p] / vol * norm_ang / es[p] / precision;
                let mut r = cell_rcut;
                for _ in 0..2 {
                    r = ((fac * r.powf(l as f64 + 1.0) + 1.0).ln() / es[p]).sqrt();
                }
                r < RCUT_THRESHOLD
            })
            .collect();

        // 0 = steep, 1 = local, 2 = smooth (the STEEP/LOCAL/SMOOTH constants).
        let kind: Vec<u8> = (0..nprim)
            .map(|p| {
                if smooth[p] {
                    2
                } else if steep[p] {
                    0
                } else {
                    1
                }
            })
            .collect();

        let mut mine = vec![usize::MAX; nprim];
        for k in [0u8, 1, 2] {
            let members: Vec<usize> = (0..nprim).filter(|&p| kind[p] == k).collect();
            if members.is_empty() {
                continue;
            }
            // `_extract_pgto_params(rs_cell, 'min')` on this sub-shell.
            let mut idx = members[0];
            for &p in &members {
                if es[p] < es[idx] {
                    idx = p;
                }
            }
            let gid = groups.len();
            groups.push(PrimGroup {
                l,
                min_exp: es[idx],
                cmax: abs_cs[idx],
                rcut: 0.0,
            });
            for &p in &members {
                mine[p] = gid;
            }
        }
        group_of_prim.push(mine);
    }

    // `ft_ao.estimate_rcut(rs_cell)` — the bra is the single most diffuse group.
    if !groups.is_empty() {
        // `exps.argmin()` — the first minimum, matching numpy on ties.
        let mut ai_idx = 0usize;
        for g in 1..groups.len() {
            if groups[g].min_exp < groups[ai_idx].min_exp {
                ai_idx = g;
            }
        }
        let ai = groups[ai_idx].min_exp;
        let li = groups[ai_idx].l as f64;
        let ci = groups[ai_idx].cmax;
        let precision_r = precision * RCUT_PRECISION_SCALE;
        // `ai`/`li`/`ci` are already copied out, so each group's radius depends
        // only on itself and those three — iterate in place.
        for grp in groups.iter_mut() {
            let aj = grp.min_exp;
            let lj = grp.l as f64;
            let cj = grp.cmax;
            let aij = ai + aj;
            let lij = li + lj;
            let norm_ang = ((2.0 * li + 1.0) * (2.0 * lj + 1.0)).sqrt() / (4.0 * pi);
            let c1 = ci * cj * norm_ang;
            let theta = ai * aj / aij;
            let aij1 = aij.powf(-0.5);
            let mut fac = pi.powf(1.5)
                * c1
                * aij1.powf(lij + 3.0)
                * (2.0 * aij / pi).powf(0.25)
                * aij.powf(lij);
            fac /= precision_r;
            let mut r0 = cell_rcut;
            for _ in 0..2 {
                let dri = aj * aij1 * r0 + 1.0;
                let drj = ai * aij1 * r0 + 1.0;
                let fl = 2.0 * pi / vol * r0 / theta;
                r0 = ((fac * dri.powf(li) * drj.powf(lj) * fl + 1.0).ln() / theta).sqrt();
            }
            grp.rcut = r0;
        }
    }
    Ok((groups, group_of_prim))
}

/// The per-group `estimate_rcut` values, in `_RangeSeparatedCell` shell order.
///
/// Exposed so a test can pin them against upstream's
/// `ft_ao.estimate_rcut(_RangeSeparatedCell.from_cell(cell, 10.0, 1.0))`
/// without reaching into private state — the RS split is the subtlest thing in
/// plan 13-01 and it deserves a direct gate.
///
/// # Errors
/// Propagates cell access.
pub fn rs_group_rcuts(cell: &Cell) -> Result<Vec<f64>, PbcDfError> {
    Ok(range_separated_groups(cell)?.0.iter().map(|g| g.rcut).collect())
}

#[inline]
fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// One shell-contraction: the unit that owns a block of Cartesian AOs.
struct ShellCtr {
    l: u32,
    nprim: usize,
    /// Index into `_env` of the primitive exponents.
    pe: usize,
    /// Index into `_env` of THIS contraction's coefficient column (F-order:
    /// `PTR_COEFF + ictr*nprim`).
    pc: usize,
    /// Atom centre in Bohr.
    r: [f64; 3],
    /// Offset of this block in the Cartesian AO axis.
    cart_off: usize,
}

fn shell_contractions(cell: &Cell) -> Vec<ShellCtr> {
    let mol = &cell.mol;
    let mut v = Vec::new();
    let mut cart_off = 0usize;
    for ib in 0..mol.nbas {
        let row = ib * BAS_SLOTS;
        let l = mol._bas[row + ANG_OF] as u32;
        let nprim = mol._bas[row + NPRIM_OF] as usize;
        let nctr = mol._bas[row + NCTR_OF] as usize;
        let pe = mol._bas[row + PTR_EXP] as usize;
        let pc0 = mol._bas[row + PTR_COEFF] as usize;
        let atom = mol._bas[row + ATOM_OF] as usize;
        let pcoord = mol._atm[atom * ATM_SLOTS + PTR_COORD] as usize;
        let r = [
            mol._env[pcoord],
            mol._env[pcoord + 1],
            mol._env[pcoord + 2],
        ];
        for ictr in 0..nctr {
            v.push(ShellCtr {
                l,
                nprim,
                pe,
                pc: pc0 + ictr * nprim,
                r,
                cart_off,
            });
            cart_off += ncart(l);
        }
    }
    v
}

/// The dense `(ngrids, nao, nao)` planar result of one `ft_aopair` call.
#[derive(Debug, Clone, Default)]
pub struct FtAopairOut {
    /// Real plane, row-major `(ngrids, nao, nao)`.
    pub re: Vec<f64>,
    /// Imaginary plane, same layout.
    pub im: Vec<f64>,
    /// AO count (spherical unless `cell.cart`).
    pub nao: usize,
    /// Number of G-vectors.
    pub ngrids: usize,
}

impl FtAopairOut {
    /// The `nao × nao` matrix at G-index `g`, as a [`CTensor`].
    pub fn at(&self, g: usize) -> CTensor {
        let n = self.nao * self.nao;
        CTensor {
            re: self.re[g * n..(g + 1) * n].to_vec(),
            im: self.im[g * n..(g + 1) * n].to_vec(),
        }
    }
}

/// `ft_aopair_kpts` for ONE k-point — the plan 13-01 entry point.
///
/// Computes `Σ_L e^{i k·L} ∫ φ_μ(r) φ_ν(r−L) e^{−i(G+q)·r} dr` for every `G` in
/// `gv`, returning the dense `(ngrids, nao, nao)` planar result.
///
/// The Bloch phase convention is `e^{+i k·L}`, matching
/// `pyscf_kernels::pbc::bloch_phase` and therefore `pbc_intor` — which is what
/// makes the `G = 0` identity a meaningful gate.
///
/// # Errors
/// Propagates cell access, the lattice-sum build, and the kernel launch.
pub fn ft_aopair_kpt(
    cell: &Cell,
    gv: &[[f64; 3]],
    q: [f64; 3],
    kpt: [f64; 3],
    rcut: RcutChoice,
) -> Result<FtAopairOut, PbcDfError> {
    let radius = rcut.resolve(cell)?;
    let ls = pyscf_pbc_gto::lattice::get_lattice_ls(cell, Some(radius), None, true)
        .map_err(PbcDfError::from)?;
    // `RcutChoice::Upstream` means "be upstream", which includes its screens.
    let screen = if rcut == RcutChoice::Upstream {
        FtScreen::Upstream
    } else {
        FtScreen::None
    };
    ft_aopair_kpt_screened(cell, gv, q, kpt, &ls, screen)
}

/// [`ft_aopair_kpt`] against a caller-supplied image list.
///
/// This is what Gate 1c uses: the reference overlap is built with
/// `pyscf_pbc_gto::pbc_intor::intor_cross_with_images` over the SAME `Ls`, so
/// both sides are converged over identical images and nothing but the
/// McMurchie–Davidson algebra can move the difference.
///
/// # Errors
/// As [`ft_aopair_kpt`].
pub fn ft_aopair_kpt_with_images(
    cell: &Cell,
    gv: &[[f64; 3]],
    q: [f64; 3],
    kpt: [f64; 3],
    ls: &[[f64; 3]],
) -> Result<FtAopairOut, PbcDfError> {
    ft_aopair_kpt_screened(cell, gv, q, kpt, ls, FtScreen::None)
}

/// [`ft_aopair_kpt_with_images`] with an explicit [`FtScreen`].
///
/// # Errors
/// As [`ft_aopair_kpt`].
pub fn ft_aopair_kpt_screened(
    cell: &Cell,
    gv: &[[f64; 3]],
    q: [f64; 3],
    kpt: [f64; 3],
    ls: &[[f64; 3]],
    screen_mode: FtScreen,
) -> Result<FtAopairOut, PbcDfError> {
    FtKernel::build(cell, kpt, ls, screen_mode)?.eval(cell, gv, q)
}

/// The screened record table for one `(cell, k-point, image list, screen)` —
/// plan 13-03's `FtKernel`.
///
/// **Building this is the expensive half and it does not depend on `G`.** The
/// first cut of `ft_loop` rebuilt it per G-block, which made a single
/// `get_pp` at mesh 15 take minutes: the table is
/// `O(nimgs · nprim² · npairs)` McMurchie–Davidson recursions, and a mesh has
/// hundreds of blocks. Hoisting it is why this type exists — the consolidation
/// note in `aftdf.rs` was wrong and the measurement said so.
#[derive(Debug, Clone)]
pub struct FtKernel {
    tables: FtAopairTables,
    ncart_tot: usize,
    nsph: usize,
    cart: bool,
    c2s: Vec<f64>,
}

impl FtKernel {
    /// Screen the lattice sum and build the flat device tables.
    ///
    /// # Errors
    /// Propagates cell access and the cart→sph table lookup.
    pub fn build(
        cell: &Cell,
        kpt: [f64; 3],
        ls: &[[f64; 3]],
        screen_mode: FtScreen,
    ) -> Result<Self, PbcDfError> {
        build_tables(cell, kpt, ls, screen_mode)
    }

    /// Evaluate at a block of G-vectors, shifted by `q`.
    ///
    /// # Errors
    /// Propagates the kernel launch.
    pub fn eval(
        &self,
        cell: &Cell,
        gv: &[[f64; 3]],
        q: [f64; 3],
    ) -> Result<FtAopairOut, PbcDfError> {
        let ngrids = gv.len();
        if ngrids == 0 || self.tables.slot_pair.is_empty() {
            return Ok(FtAopairOut::default());
        }
        let mut t = self.tables.clone();
        t.gv = Vec::with_capacity(3 * ngrids);
        for g in gv {
            t.gv.push(g[0] + q[0]);
            t.gv.push(g[1] + q[1]);
            t.gv.push(g[2] + q[2]);
        }
        let client = pyscf_algebra::select_backend()
            .map_err(|e| PbcDfError::Backend(format!("ft_aopair: backend selection: {e}")))?
            .client;
        let (cre, cim) = pyscf_kernels::pbc::ft_aopair::ft_aopair(&client, &t)
            .map_err(|e| PbcDfError::Backend(format!("ft_aopair kernel: {e}")))?;
        let _ = cell;
        if self.cart {
            return Ok(FtAopairOut {
                re: cre,
                im: cim,
                nao: self.ncart_tot,
                ngrids,
            });
        }
        Ok(FtAopairOut {
            re: cart_to_sph_planes(&cre, &self.c2s, self.ncart_tot, self.nsph, ngrids),
            im: cart_to_sph_planes(&cim, &self.c2s, self.ncart_tot, self.nsph, ngrids),
            nao: self.nsph,
            ngrids,
        })
    }
}

fn build_tables(
    cell: &Cell,
    kpt: [f64; 3],
    ls: &[[f64; 3]],
    screen_mode: FtScreen,
) -> Result<FtKernel, PbcDfError> {
    let shells = shell_contractions(cell);
    let nshc = shells.len();
    let ncart_tot: usize = shells.iter().map(|s| ncart(s.l)).sum();
    if nshc == 0 {
        return Ok(FtKernel {
            tables: FtAopairTables::default(),
            ncart_tot: 0,
            nsph: 0,
            cart: cell.mol.cart,
            c2s: Vec::new(),
        });
    }
    // NOT `cell.precision * 1e-2`. That threshold is what `estimate_rcut`
    // integrates to derive the IMAGE LIST; applying it a second time as an
    // absolute per-primitive-pair cutoff drops ~1e-10 terms that then accumulate
    // over `nimgs × nprim²` records. Measured: it cost 1.66e-7 on diamond's
    // `p`-`p` block — 3 orders worse than Gate 1a's bar — while every
    // angular-off-diagonal element stayed exact to 1e-16, which is the
    // signature of a screen rather than an algebra bug. This threshold only
    // removes records that are numerically zero.
    let screen = FT_RECORD_EPS;
    let env = &cell.mol._env;

    let mut t = FtAopairTables {
        ncart: ncart_tot,
        ..Default::default()
    };
    let pi = std::f64::consts::PI;

    // ── upstream's two screens (ft_ao.py:631-666 and :669-703) ──────────
    // `shells` is per shell-CONTRACTION; both screens key off the shell, so map
    // each contraction back to its `_bas` row.
    let (groups, group_of_prim) = range_separated_groups(cell)?;
    // A contraction's `_bas` row: walk `_bas`, repeating each row `nctr` times,
    // which is exactly the order `shell_contractions` emitted.
    let mut shell_of_ctr = Vec::with_capacity(nshc);
    for ib in 0..cell.mol.nbas {
        let nctr = cell.mol._bas[ib * BAS_SLOTS + NCTR_OF] as usize;
        for _ in 0..nctr {
            shell_of_ctr.push(ib);
        }
    }
    let atom_coords = cell.mol.atom_coords();
    // `get_ovlp_mask`'s default cutoff (`ft_ao.py:678-684`).
    let cell_rcut = cell.try_rcut().map_err(PbcDfError::from)?;
    let vol = cell.vol();
    let theta_ij = groups
        .iter()
        .map(|g| g.min_exp)
        .fold(f64::INFINITY, f64::min)
        / 2.0;
    let lattice_sum_factor = (2.0 * pi * cell_rcut / (vol * theta_ij)).max(1.0);
    let ovlp_cutoff = cell.precision / lattice_sum_factor * 0.1;
    let expcutoff = cell.precision * 1e-4;

    // Cartesian power tables, once per l present.
    let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
    let powers: Vec<Vec<(u32, u32, u32)>> = (0..=lmax).map(cart_powers).collect();

    for (ip, si) in shells.iter().enumerate() {
        for (jp, sj) in shells.iter().enumerate() {
            let li = si.l;
            let lj = sj.l;
            let lij = li + lj;
            let estride = e_len(li, lj);
            let rec0 = t.rec_eoff.len() as u32;
            let cfac = common_fac_sp(li) * common_fac_sp(lj);

            for l_vec in ls {
                let b = [
                    sj.r[0] + l_vec[0],
                    sj.r[1] + l_vec[1],
                    sj.r[2] + l_vec[2],
                ];
                let d = [b[0] - si.r[0], b[1] - si.r[1], b[2] - si.r[2]];
                let ab2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

                // `strip_basis` needs only the image's distance to the
                // nearest reference-cell atom; the per-group radius comparison
                // happens inside the primitive loop.
                let nearest = if screen_mode == FtScreen::Upstream {
                    let mut n = f64::INFINITY;
                    for r in &atom_coords {
                        let v = [b[0] - r[0], b[1] - r[1], b[2] - r[2]];
                        n = n.min((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                    }
                    n
                } else {
                    0.0
                };
                let dr = ab2.sqrt();
                // e^{+i k·L}, the same convention as `pbc_intor`.
                let theta_k = kpt[0] * l_vec[0] + kpt[1] * l_vec[1] + kpt[2] * l_vec[2];
                let (ph_im, ph_re) = theta_k.sin_cos();

                for pa in 0..si.nprim {
                    let a = env[si.pe + pa];
                    let ca = env[si.pc + pa];
                    for pb in 0..sj.nprim {
                        let bexp = env[sj.pe + pb];
                        let cb = env[sj.pc + pb];
                        if screen_mode == FtScreen::Upstream {
                            // Both screens key off the RANGE-SEPARATED group of
                            // each primitive, not the shell it came from — that
                            // is the whole point of the RS split.
                            let gj = groups[group_of_prim[shell_of_ctr[jp]][pb]];
                            if nearest >= gj.rcut {
                                continue;
                            }
                            let gi = groups[group_of_prim[shell_of_ctr[ip]][pa]];
                            let (ei, ej) = (gi.min_exp, gj.min_exp);
                            let (lif, ljf) = (gi.l as f64, gj.l as f64);
                            let aij = ei + ej;
                            let theta = ei * ej / aij;
                            let aij1 = 1.0 / aij;
                            let aij2 = aij.powf(-0.5);
                            let dri = ej * aij1 * dr + aij2;
                            let drj = ei * aij1 * dr + aij2;
                            let norm_i = gi.cmax * ((2.0 * lif + 1.0) / (4.0 * pi)).sqrt();
                            let norm_j = gj.cmax * ((2.0 * ljf + 1.0) / (4.0 * pi)).sqrt();
                            let fl = 2.0 * pi / vol * dr / theta + 1.0;
                            let ovlp = pi.powf(1.5)
                                * norm_i
                                * norm_j
                                * (-theta * dr * dr).exp()
                                * dri.powf(lif)
                                * drj.powf(ljf)
                                * aij1.powf(1.5)
                                * fl;
                            if ovlp <= ovlp_cutoff {
                                continue;
                            }
                        }
                        let p = a + bexp;
                        let kab = k_ab(a, bexp, ab2);
                        if screen_mode == FtScreen::Upstream && kab < expcutoff {
                            // libcint's own per-primitive-pair cutoff:
                            // `supmol._env[PTR_EXPCUTOFF] = −log(precision·1e-4)`
                            // (`ft_ao.py:626`), i.e. skip when
                            // `K_AB = exp(−a·b/(a+b)·|A−B|²) < precision·1e-4`.
                            // The THIRD screen upstream applies, and the one
                            // that is easy to miss because it lives in an `_env`
                            // slot rather than in Python control flow.
                            continue;
                        }
                        let gauss = (pi / p).powf(1.5);
                        // Schwarz-style screen on the actual G=0 magnitude.
                        if (ca * cb * cfac * kab * gauss).abs() < screen {
                            continue;
                        }
                        let pc = [
                            (a * si.r[0] + bexp * b[0]) / p,
                            (a * si.r[1] + bexp * b[1]) / p,
                            (a * si.r[2] + bexp * b[2]) / p,
                        ];
                        t.rec_eoff.push(t.etab.len() as u32);
                        t.rec_estride.push(estride as u32);
                        // K_AB rides on the x axis ONLY — counting it three
                        // times is the classic bug and s-only tests miss it.
                        for axis in 0..3 {
                            let seed = if axis == 0 { kab } else { 1.0 };
                            let e = e_coefficients(
                                li,
                                lj,
                                p,
                                pc[axis] - si.r[axis],
                                pc[axis] - b[axis],
                                seed,
                            );
                            t.etab.extend_from_slice(&e.data);
                        }
                        t.rec_p.extend_from_slice(&[pc[0], pc[1], pc[2], p]);
                        let w = ca * cb * cfac * gauss;
                        t.rec_pref.push(w * ph_re);
                        t.rec_pref.push(w * ph_im);
                    }
                }
            }

            let nrec = t.rec_eoff.len() as u32 - rec0;
            if nrec == 0 {
                continue; // fully screened shell pair — output stays zero
            }
            let pair_id = t.pair_rec0.len() as u32;
            t.pair_rec0.push(rec0);
            t.pair_nrec.push(nrec);
            t.pair_lj.push(lj);
            t.pair_lij.push(lij);

            for (ci, &(ix, iy, iz)) in powers[li as usize].iter().enumerate() {
                for (cj, &(jx, jy, jz)) in powers[lj as usize].iter().enumerate() {
                    t.slot_pair.push(pair_id);
                    t.slot_out
                        .push(((si.cart_off + ci) * ncart_tot + sj.cart_off + cj) as u32);
                    t.slot_pow.extend_from_slice(&[ix, iy, iz, jx, jy, jz]);
                }
            }
        }
    }

    let (c2s, nsph) = if cell.mol.cart {
        (Vec::new(), ncart_tot)
    } else {
        pyscf_gto::cart2sph_coeff(&cell.mol).map_err(PbcDfError::from)?
    };
    Ok(FtKernel {
        tables: t,
        ncart_tot,
        nsph,
        cart: cell.mol.cart,
        c2s,
    })
}

/// `out[g] = Cᵀ · in[g] · C` for every G, with `C[cart, sph]` block-diagonal.
///
/// Two small dense passes per G rather than one big GEMM: `nao` is a handful and
/// `C` is mostly zeros, so the GEMM setup would cost more than the arithmetic.
/// Revisit if a large-basis profile says otherwise (carry-over, plan 13-03).
fn cart_to_sph_planes(
    src: &[f64],
    c: &[f64],
    ncart_tot: usize,
    nsph: usize,
    ngrids: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; ngrids * nsph * nsph];
    let mut tmp = vec![0.0f64; ncart_tot * nsph];
    for g in 0..ngrids {
        let block = &src[g * ncart_tot * ncart_tot..(g + 1) * ncart_tot * ncart_tot];
        // tmp[cart_i, n] = Σ_d block[cart_i, d] · C[d, n]
        tmp.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..ncart_tot {
            for d in 0..ncart_tot {
                let v = block[i * ncart_tot + d];
                if v == 0.0 {
                    continue;
                }
                for n in 0..nsph {
                    tmp[i * nsph + n] += v * c[d * nsph + n];
                }
            }
        }
        // out[m, n] = Σ_i C[i, m] · tmp[i, n]
        let dst = &mut out[g * nsph * nsph..(g + 1) * nsph * nsph];
        for i in 0..ncart_tot {
            for m in 0..nsph {
                let w = c[i * nsph + m];
                if w == 0.0 {
                    continue;
                }
                for n in 0..nsph {
                    dst[m * nsph + n] += w * tmp[i * nsph + n];
                }
            }
        }
    }
    out
}
