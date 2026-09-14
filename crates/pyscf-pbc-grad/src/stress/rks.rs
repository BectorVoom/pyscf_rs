//! Plan 18-12 — `pyscf/pbc/grad/rks_stress.py` (462 l), the CORE of the stress
//! half and the base of the other three stress modules.
//!
//! # The eight shared symbols
//!
//! Upstream's `krks_stress`, `kuks_stress` and `uks_stress` each import the same
//! eight symbols from `rks_stress` (`krks_stress.py:74-83` and two mirrors).
//! They are all `pub` here, and plan 18-13 imports them from here:
//!
//! | upstream (`rks_stress.py`) | here |
//! |---|---|
//! | `strain_tensor_dispalcement` (`:59`) | [`strain_tensor_displacement`] (typo fixed) |
//! | `_finite_diff_cells` (`:64`) | [`finite_diff_cells`] (re-export of 18-02 Task 5) |
//! | `_get_weight_strain_derivatives` (`:121`) | [`weight_strain_derivatives`] |
//! | `_get_coulG_strain_derivatives` (`:112`) | [`coulg_strain_derivatives`] |
//! | `_eval_ao_strain_derivatives` (`:127`) | [`eval_ao_strain_derivatives`] (18-11 wrapper) |
//! | `_get_vpplocG_strain_derivatives` (`:292`) | [`vpplocg_strain_derivatives`] |
//! | `_get_pp_nonloc_strain_derivatives` (`:309`) | [`pp_nonloc_strain_derivatives`] |
//! | `ewald` (`:387`) | [`ewald_strain`] |
//!
//! # The closed form (D-PBC-31 clause 2, `18-REVIEW §6.2`)
//!
//! Upstream writes `get_ovlp`/`get_kin` as 36 lattice sums between them (9 strain
//! components × 2 displaced cells × 2 operators, each over a rebuilt cell).
//! Upstream's OWN test replaces the whole finite difference with a closed form
//! and asserts it at **1e-9**, twice, in two algebraically distinct arrangements
//! (`test_rks_stress.py:53-71` for ovlp, `:88-107` for kin). This port ships the
//! closed form as the PRODUCTION path ([`ip_strain_closed_form`]) and keeps the
//! finite difference as the TEST ORACLE only (it lives in
//! `tests/rks_stress.rs`, not here):
//!
//! ```text
//! dS_k/dε_xy = −Σ_L e^{ik·L} [ ∇_x S[i,j;L]·R_{i,y} + ∇^{ket}_x S[i,j;L]·(R_j+L)_y ]
//! ```
//!
//! The weight splits: `R_{j,y}` is AO-indexed and applies AFTER the lattice sum;
//! `L_y` is an image scalar and folds straight into the Bloch-phase array —
//! 18-02 Task 7's hook
//! ([`intor_cross_with_image_weights`](pyscf_pbc_gto::intor_cross_with_image_weights)).
//! So the port needs no supercell and no new integral family: the whole 3×3
//! derivative is one `int1e_ipovlp`/`int1e_ipkin` lattice sum run with four phase
//! rows per k-point (weights `1`, `L_x`, `L_y`, `L_z`) instead of one. Per
//! operator: 18 lattice sums → 4 (1 unweighted + 3 image-weighted sharing ONE
//! `Ls`), 18 displaced cell builds → 0, truncation/cancellation/screening error
//! → none.
//!
//! Both of upstream's arrangements are ported ([`ip_strain_closed_form`] is
//! `-(ovlp10 + ovlp01)` from `intor_cross(scell, cell)`;
//! [`ip_strain_closed_form_alt`] is `-(ovlp10 - ovlp01)` from
//! `intor_cross(cell, scell)`). They differ by which side carries `∇`, so a sign
//! error in the ket-derivative term cancels in one and not the other, and
//! upstream asserts both. Here the ket derivative is the negated bra derivative
//! (`K_x[i,j;L] = −G_x[i,j;L]` pointwise, since `∂/∂R_i = −∂/∂R_j` on
//! `S(R_i − R_j − L)`), so the two paths are algebraically identical and
//! textually distinct: one ADDS the negated ket term, the other SUBTRACTS the
//! unnegated bra term. Dropping the negation breaks arrangement A and leaves B
//! green — exactly upstream's property.
//!
//! # What is NOT here
//!
//! `get_vxc`, the pseudopotential strain terms' CONSUMERS, `kernel` and Gates
//! A/D are plan 18-20. The block SIZING (D-PBC-30 clause 1) is here
//! ([`strain_block_size`]); the fused XC loop it sizes is 18-20's.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;

use crate::error::PbcGradError;

// Re-exported so 18-13 (and every later consumer) imports the strain FD from
// the stress base, mirroring upstream's `from ...rks_stress import ...`.
pub use crate::verify_fd::{StrainCells, finite_diff_cells};

/// Upstream's hard-coded half step for `get_ovlp`/`get_kin` (`rks_stress.py:89`,
/// `:102`: cells at `±disp`, divided by `2*disp`).
///
/// [`finite_diff_cells`] takes the FULL separation, so the oracle calls it with
/// `2 * STRAIN_FD_HALF_DISP`. This is a DIFFERENT step from the `1e-3` used by
/// the end-to-end stress tests (`18-CONTEXT` trap 6): the tight one is an
/// integral difference, the loose one an SCF difference. Do not unify them.
pub const STRAIN_FD_HALF_DISP: f64 = 1e-5;

// ---------------------------------------------------------------------------
// 1. strain_tensor_displacement — rks_stress.py:59
// ---------------------------------------------------------------------------

/// `E = I + disp · e_x ⊗ e_y` (`rks_stress.py:59-62`).
///
/// Upstream spells it `strain_tensor_dispalcement` (missing an `e`); the port
/// fixes the typo and records the mapping here so the import in 18-13 still
/// reads as the same symbol.
pub fn strain_tensor_displacement(x: usize, y: usize, disp: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if x < 3 && y < 3 && disp.is_finite() {
        out[x][y] += disp;
    }
    out
}

// ---------------------------------------------------------------------------
// 2. per-AO coordinates — test_rks_stress.py:51-53
// ---------------------------------------------------------------------------

/// `bas_coords = repeat(cell.atom_coords(), ao_repeats)` (`test_rks_stress.py`
/// `:51-53`, via `aoslice_by_atom()[:,3] − [:,2]`).
///
/// The 4-tuple `(shl0, shl1, p0, p1)` is 18-18's widening (`18-CONTEXT §1.8`);
/// only the AO half `[p0, p1)` is read here.
fn bas_coords(cell: &Cell) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let coords = cell.mol.atom_coords();
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol)?;
    let nao = cell.mol.nao_nr;
    let mut out = vec![[0.0; 3]; nao];
    for (ia, &(_, _, p0, p1)) in slices.iter().enumerate() {
        let coord = coords.get(ia).copied().ok_or_else(|| {
            PbcGradError::ShapeMismatch {
                expected: slices.len(),
                got: coords.len(),
            }
        })?;
        for mu in p0..p1 {
            if mu >= nao {
                return Err(PbcGradError::ShapeMismatch {
                    expected: nao,
                    got: mu + 1,
                }
                .into());
            }
            out[mu] = coord;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 3. weight strain derivatives — rks_stress.py:121-125 (D-PBC-31 clause 6)
// ---------------------------------------------------------------------------

/// `(weight_0, weight_1)` (`rks_stress.py:121-125`).
///
/// `weight_0 = vol / ngrids`; `weight_1 = eye(3) * weight_0` — EXACTLY diagonal,
/// so `:260`'s `einsum('g,g->', rho0[0], exc) * weight_1` is a scalar on the
/// diagonal (D-PBC-31 clause 6). The full 3×3 is returned for signature
/// compatibility; callers that only need the diagonal read `[0][0]`
/// (== `[1][1]` == `[2][2]`, bit-identical, not emergent-to-roundoff).
pub fn weight_strain_derivatives(vol: f64, ngrids: usize) -> Result<(f64, [[f64; 3]; 3]), PyscfRsError> {
    if !vol.is_finite() || vol <= 0.0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "weight strain derivatives: non-positive cell volume {vol}"
        ))));
    }
    if ngrids == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    let w0 = vol / ngrids as f64;
    let mut w1 = [[0.0; 3]; 3];
    for (i, row) in w1.iter_mut().enumerate() {
        row[i] = w0;
    }
    Ok((w0, w1))
}

// ---------------------------------------------------------------------------
// 4. coulG strain derivatives — rks_stress.py:112-119 (D-PBC-31 clause 6)
// ---------------------------------------------------------------------------

/// The 6 unique components of
/// `coulG_1 = einsum('gx,gy->xyg', Gv, Gv) * coulG_0 * 2/G2`
/// (`rks_stress.py:112-119`), stored once and mirrored.
///
/// SYMMETRIC in `(x,y)` BY CONSTRUCTION (D-PBC-31 clause 6): 9 components
/// stored where 6 are independent — 1.5× on a 1.7 MiB (`ngrids = 24389`) to
/// 15.5 MiB (`60³`) array. The total stress is NOT symmetric in general, so an
/// asymmetry that shows up must be attributable to a genuinely asymmetric term
/// rather than to roundoff in this one.
///
/// Order: `[xx, yy, zz, xy, xz, yz]`. `G == 0` contributes exactly `0`
/// (`G_x·G_y = 0` there regardless of `coulG_0[0]`, so no exxdiv convention
/// leaks in).
#[derive(Debug, Clone)]
pub struct CoulGStrain {
    /// Grid-point count.
    pub ngrids: usize,
    /// The 6 unique `(x,y)` planes, each `ngrids` long.
    pub g1: [Vec<f64>; 6],
}

impl CoulGStrain {
    /// Map `(x, y)` onto the stored unique plane.
    pub fn plane(&self, x: usize, y: usize) -> &[f64] {
        let (a, b) = if x <= y { (x, y) } else { (y, x) };
        let idx = match (a, b) {
            (0, 0) => 0,
            (1, 1) => 1,
            (2, 2) => 2,
            (0, 1) => 3,
            (0, 2) => 4,
            _ => 5,
        };
        &self.g1[idx]
    }
}

/// Build [`CoulGStrain`] from an explicit `Gv` table and the caller's `coulG_0`
/// (whatever `get_coulg` returned for this cell/mesh — the derivative formula
/// only reads it at `G != 0`).
pub fn coulg_strain_derivatives(
    gv: &[[f64; 3]],
    coulg0: &[f64],
) -> Result<CoulGStrain, PyscfRsError> {
    if gv.len() != coulg0.len() {
        return Err(PbcGradError::ShapeMismatch {
            expected: gv.len(),
            got: coulg0.len(),
        }
        .into());
    }
    if gv.iter().flatten().any(|v| !v.is_finite()) || coulg0.iter().any(|v| !v.is_finite()) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "coulG strain derivatives: non-finite Gv or coulG_0".into(),
        )));
    }
    let ngrids = gv.len();
    let mut g1: [Vec<f64>; 6] = Default::default();
    for (g, v) in gv.iter().enumerate() {
        let g2 = oracle_sum(&[v[0] * v[0], v[1] * v[1], v[2] * v[2]]);
        let factor = if g2 == 0.0 {
            0.0
        } else {
            coulg0[g] * 2.0 / g2
        };
        let comps = [
            v[0] * v[0] * factor,
            v[1] * v[1] * factor,
            v[2] * v[2] * factor,
            v[0] * v[1] * factor,
            v[0] * v[2] * factor,
            v[1] * v[2] * factor,
        ];
        for (plane, value) in g1.iter_mut().zip(comps) {
            plane.push(value);
        }
    }
    let _ = ngrids;
    Ok(CoulGStrain { ngrids, g1 })
}

// ---------------------------------------------------------------------------
// 5. eval_ao_strain_derivatives — rks_stress.py:127-147 (18-11 wrapper)
// ---------------------------------------------------------------------------

/// One k-point's `(3, 3, comp, ngrids, nao)` strain-AO table, flat per block.
///
/// Block `b = (x*3+y)*comp+c` starts at `b*ngrids*nao`, element `(g, mu)` at
/// `+ g + mu*ngrids` — the F-order per-component layout that reshapes to
/// `(3,3,comp,ngrids,nao)` C-order exactly as upstream's
/// `out.reshape(3,3,comp,ngrids,-1)` (`rks_stress.py:143-146`). This struct is
/// the reshape MAPPED, not recomputed: the kernel output of 18-11
/// (`eval_strain_ao`, via `Cell::pbc_eval_gto`) already has this layout.
#[derive(Debug, Clone)]
pub struct AoStrainTable {
    /// Strain order (`0` or `1`).
    pub deriv: u32,
    /// AO-derivative components: `(deriv+1)(deriv+2)(deriv+3)/6` (`:133`).
    pub comp: usize,
    /// Grid-point count.
    pub ngrids: usize,
    /// AO count.
    pub nao: usize,
    /// Real planes, one per k-point.
    pub re: Vec<Vec<f64>>,
    /// Imaginary planes, one per k-point (zeros at gamma).
    pub im: Vec<Vec<f64>>,
}

impl AoStrainTable {
    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.re.len()
    }

    /// `(re, im)` of strain `(x, y)`, derivative component `c`, grid `g`, AO `mu`.
    pub fn get(&self, k: usize, x: usize, y: usize, c: usize, g: usize, mu: usize) -> (f64, f64) {
        let b = (x * 3 + y) * self.comp + c;
        let p = b * self.ngrids * self.nao + g + mu * self.ngrids;
        (self.re[k][p], self.im[k][p])
    }
}

/// `_eval_ao_strain_derivatives` (`rks_stress.py:127-147`).
///
/// Selects the `feval` family by `cell.cart`
/// (`GTOval_{sph,cart}_deriv{deriv}_strain_tensor`), drives it through
/// `Cell::pbc_eval_gto` — the 18-11 kernel, which carries the per-image weight
/// `(R_A + L)_y` INSIDE the lattice sum (`18-REVIEW §9`) — and maps the
/// `(nkpts, 3, 3, comp, ngrids, nao)` table. The grid-response addition
/// (`rks_stress.py:215-226`, `ao_strain += einsum('xig,yg->xyig', ao, coords)`)
/// is deliberately NOT part of this function: it is the response of the grid
/// points, not of the basis function, and 18-20's block loop adds it.
///
/// `deriv >= 2` is the 18-11 Task-4 named refusal (no upstream caller, no
/// oracle), never a fall-through.
pub fn eval_ao_strain_derivatives(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    deriv: u32,
) -> Result<AoStrainTable, PyscfRsError> {
    let comp = match deriv {
        0 => 1usize,
        1 => 4usize,
        _ => {
            return Err(PyscfRsError::NotYetImplemented {
                phase: 18,
                what: "strain-tensor AO derivatives at deriv >= 2 (no upstream caller, no oracle)",
            });
        }
    };
    let family = if cell.mol.cart { "cart" } else { "sph" };
    let feval = format!("GTOval_{family}_deriv{deriv}_strain_tensor");
    let out = cell.pbc_eval_gto(&feval, coords, kpts)?;
    let want = 9 * comp;
    if out.comp != want {
        return Err(PbcGradError::ShapeMismatch {
            expected: want,
            got: out.comp,
        }
        .into());
    }
    let (mut re, mut im) = (Vec::with_capacity(out.nkpts()), Vec::with_capacity(out.nkpts()));
    for mat in &out.kaos {
        re.push(mat.re.clone());
        im.push(mat.im.clone());
    }
    Ok(AoStrainTable {
        deriv,
        comp,
        ngrids: out.ngrids,
        nao: out.nao,
        re,
        im,
    })
}

// ---------------------------------------------------------------------------
// 6. get_ovlp / get_kin closed form — rks_stress.py:86-111 via 18-REVIEW §6.2
// ---------------------------------------------------------------------------

/// One k-point's 3×3 one-electron strain derivative, complex planes.
///
/// `re`/`im` hold 9 `nao*nao` F-order matrices (`(i,j)` at `i + j*nao`,
/// matching [`PbcIntorOutput`](pyscf_pbc_gto::PbcIntorOutput)), index `x*3+y`.
/// At gamma the imaginary planes are zeros.
#[derive(Debug, Clone)]
pub struct IpStrainK {
    /// AO count.
    pub nao: usize,
    /// Real 3×3 strain matrices, F-order.
    pub re: [Vec<f64>; 9],
    /// Imaginary 3×3 strain matrices, F-order.
    pub im: [Vec<f64>; 9],
}

impl IpStrainK {
    /// `(re, im)` of strain `(x, y)`, element `(i, j)`.
    pub fn get(&self, x: usize, y: usize, i: usize, j: usize, nao: usize) -> (f64, f64) {
        let p = i + j * nao;
        (self.re[x * 3 + y][p], self.im[x * 3 + y][p])
    }
}

fn ip_sum(
    cell: &Cell,
    kpts: &[[f64; 3]],
    intor: &str,
    ls: &[[f64; 3]],
    weights: Option<&[f64]>,
) -> Result<pyscf_pbc_gto::PbcIntorOutput, PyscfRsError> {
    pyscf_pbc_gto::intor_cross_with_image_weights(
        intor,
        cell,
        cell,
        kpts,
        pyscf_pbc_gto::PbcIntorOpts {
            hermi: 0,
            ..Default::default()
        },
        ls,
        None,
        weights,
    )
}

/// The closed-form strain derivative of an `int1e_ip*` family — arrangement A:
///
/// ```text
/// dat = −(ovlp10 + ovlp01),  ovlp10[x,y,i,j] = G_x[i,j]·R_{i,y},
/// ovlp01 = ket term with K = −G  (intor_cross(scell, cell) side, :61-64)
/// ```
///
/// Production path for `get_ovlp` (`int1e_ipovlp`) and `get_kin`
/// (`int1e_ipkin`). ONE `Ls` (`lattice_images`, a single `get_lattice_ls`) is
/// shared by the unweighted sum and the three `L_y`-weighted sums — 18 lattice
/// sums → 4 per operator, 18 displaced cell builds → 0.
///
/// `kpts` are ABSOLUTE Cartesian k-points at fixed FRACTIONAL coordinates
/// (D-PBC-31 clause 3); empty means the single gamma point. `k·L` is
/// strain-invariant, so the Bloch phase participates in neither arrangement.
pub fn ip_strain_closed_form(
    cell: &Cell,
    kpts: &[[f64; 3]],
    intor: &str,
) -> Result<Vec<IpStrainK>, PyscfRsError> {
    let ls = pyscf_pbc_gto::lattice_images(cell, cell)?;
    let g = ip_sum(cell, kpts, intor, &ls, None)?;
    let mut gw = Vec::with_capacity(3);
    for y in 0..3 {
        let weights: Vec<f64> = ls.iter().map(|l| l[y]).collect();
        gw.push(ip_sum(cell, kpts, intor, &ls, Some(&weights))?);
    }
    combine_arrangement_a(cell, &g, &gw)
}

/// Same quantity as [`ip_strain_closed_form`] — arrangement B:
///
/// ```text
/// dat = −(ovlp10 − ovlp01'),  ovlp01' unnegated  (intor_cross(cell, scell) side, :67-70)
/// ```
///
/// Algebraically identical to A by `K = −G`; textually distinct (subtract the
/// unnegated bra term rather than add the negated ket term), so a ket-sign
/// error breaks A and leaves B green. The gate asserts both against the FD
/// oracle AND against each other.
pub fn ip_strain_closed_form_alt(
    cell: &Cell,
    kpts: &[[f64; 3]],
    intor: &str,
) -> Result<Vec<IpStrainK>, PyscfRsError> {
    let ls = pyscf_pbc_gto::lattice_images(cell, cell)?;
    let g = ip_sum(cell, kpts, intor, &ls, None)?;
    let mut gw = Vec::with_capacity(3);
    for y in 0..3 {
        let weights: Vec<f64> = ls.iter().map(|l| l[y]).collect();
        gw.push(ip_sum(cell, kpts, intor, &ls, Some(&weights))?);
    }
    combine_arrangement_b(cell, &g, &gw)
}

fn nkpts_of(cell_out_nk: usize) -> usize {
    cell_out_nk
}

fn combine_arrangement_a(
    cell: &Cell,
    g: &pyscf_pbc_gto::PbcIntorOutput,
    gw: &[pyscf_pbc_gto::PbcIntorOutput],
) -> Result<Vec<IpStrainK>, PyscfRsError> {
    let r = bas_coords(cell)?;
    let nao = g.ni;
    if g.nj != nao {
        return Err(PbcGradError::ShapeMismatch {
            expected: nao,
            got: g.nj,
        }
        .into());
    }
    let nk = nkpts_of(g.nkpts());
    let mut out = Vec::with_capacity(nk);
    for k in 0..nk {
        let (mut re, mut im): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (
            (0..9).map(|_| vec![0.0; nao * nao]).collect(),
            (0..9).map(|_| vec![0.0; nao * nao]).collect(),
        );
        for x in 0..3 {
            for y in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let p = x * nao * nao + i + j * nao;
                        let (gr, gi) = (g.kmats[k].re[p], g.kmats[k].im[p]);
                        let (wr, wi) = (gw[y].kmats[k].re[p], gw[y].kmats[k].im[p]);
                        // ovlp10 = G·R_i; ket01 = −(G·R_j + W); dat = −(ovlp10 + ket01).
                        let o_re = gr * r[i][y];
                        let o_im = gi * r[i][y];
                        let k_re = -(gr * r[j][y] + wr);
                        let k_im = -(gi * r[j][y] + wi);
                        let q = i + j * nao;
                        re[x * 3 + y][q] = oracle_sum(&[-o_re, -k_re]);
                        im[x * 3 + y][q] = oracle_sum(&[-o_im, -k_im]);
                    }
                }
            }
        }
        out.push(IpStrainK {
            nao,
            re: re.try_into().map_err(|_| {
                PyscfRsError::Core(CoreError::InvalidMolecule(
                    "strain closed form: 9-component assembly failed".into(),
                ))
            })?,
            im: im.try_into().map_err(|_| {
                PyscfRsError::Core(CoreError::InvalidMolecule(
                    "strain closed form: 9-component assembly failed".into(),
                ))
            })?,
        });
    }
    Ok(out)
}

fn combine_arrangement_b(
    cell: &Cell,
    g: &pyscf_pbc_gto::PbcIntorOutput,
    gw: &[pyscf_pbc_gto::PbcIntorOutput],
) -> Result<Vec<IpStrainK>, PyscfRsError> {
    let r = bas_coords(cell)?;
    let nao = g.ni;
    if g.nj != nao {
        return Err(PbcGradError::ShapeMismatch {
            expected: nao,
            got: g.nj,
        }
        .into());
    }
    let nk = nkpts_of(g.nkpts());
    let mut out = Vec::with_capacity(nk);
    for k in 0..nk {
        let (mut re, mut im): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (
            (0..9).map(|_| vec![0.0; nao * nao]).collect(),
            (0..9).map(|_| vec![0.0; nao * nao]).collect(),
        );
        for x in 0..3 {
            for y in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let p = x * nao * nao + i + j * nao;
                        let (gr, gi) = (g.kmats[k].re[p], g.kmats[k].im[p]);
                        let (wr, wi) = (gw[y].kmats[k].re[p], gw[y].kmats[k].im[p]);
                        // ovlp10 = G·R_i; bra01' = G·R_j + W (unnegated);
                        // dat = −(ovlp10 − bra01').
                        let o_re = gr * r[i][y];
                        let o_im = gi * r[i][y];
                        let b_re = oracle_sum(&[gr * r[j][y], wr]);
                        let b_im = oracle_sum(&[gi * r[j][y], wi]);
                        let q = i + j * nao;
                        re[x * 3 + y][q] = oracle_sum(&[o_re, -b_re]);
                        re[x * 3 + y][q] = -re[x * 3 + y][q];
                        im[x * 3 + y][q] = oracle_sum(&[o_im, -b_im]);
                        im[x * 3 + y][q] = -im[x * 3 + y][q];
                    }
                }
            }
        }
        out.push(IpStrainK {
            nao,
            re: re.try_into().map_err(|_| {
                PyscfRsError::Core(CoreError::InvalidMolecule(
                    "strain closed form: 9-component assembly failed".into(),
                ))
            })?,
            im: im.try_into().map_err(|_| {
                PyscfRsError::Core(CoreError::InvalidMolecule(
                    "strain closed form: 9-component assembly failed".into(),
                ))
            })?,
        });
    }
    Ok(out)
}

/// `get_ovlp` (`rks_stress.py:86-97`) at the gamma point — arrangement A.
///
/// Returns the 9 `(x*3+y)` strain matrices, F-order. The FD oracle for the gate
/// lives in `tests/rks_stress.rs` (D-PBC-31 clause 2: the FD is the oracle and
/// nothing else).
pub fn ovlp_strain_gamma(cell: &Cell) -> Result<IpStrainK, PyscfRsError> {
    Ok(ip_strain_closed_form(cell, &[], "int1e_ipovlp")?
        .into_iter()
        .next()
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(
                "ovlp strain: empty k-point output".into(),
            ))
        })?)
}

/// `get_kin` (`rks_stress.py:99-111`) at the gamma point — arrangement A.
pub fn kin_strain_gamma(cell: &Cell) -> Result<IpStrainK, PyscfRsError> {
    Ok(ip_strain_closed_form(cell, &[], "int1e_ipkin")?
        .into_iter()
        .next()
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(
                "kin strain: empty k-point output".into(),
            ))
        })?)
}

// ---------------------------------------------------------------------------
// 7. vpplocG strain derivatives — rks_stress.py:292-307
// ---------------------------------------------------------------------------

/// `(vpplocG_0, vpplocG_1)` (`rks_stress.py:292-307`).
///
/// `v0[g] = −Σ_ia SI[ia,g]·V[ia,g]` (complex) with `V = get_vlocG` and
/// `SI = cell.get_SI(mesh)`; `v1[x,y]` is its central difference at half step
/// [`STRAIN_FD_HALF_DISP`]. The structure factor is taken ONCE from the
/// undisplaced cell: under strain `G → G·E⁻¹` and `R → E·R`, so `G·R` — and
/// hence SI — is strain-invariant, and upstream reuses it on both sides
/// (`:297-304`). The G-vectors ARE rebuilt per displaced cell at the PINNED
/// mesh (`cell.get_Gv(mesh)`; [`finite_diff_cells`] pins it — trap 5).
pub struct VpplocGStrain {
    /// Grid-point count (`prod(mesh)`).
    pub ngrids: usize,
    /// Real / imaginary planes of `v0`.
    pub v0_re: Vec<f64>,
    /// Real / imaginary planes of `v0`.
    pub v0_im: Vec<f64>,
    /// Real planes of the 9 `v1[x*3+y]` strain derivatives.
    pub v1_re: [Vec<f64>; 9],
    /// Imaginary planes of the 9 `v1[x*3+y]` strain derivatives.
    pub v1_im: [Vec<f64>; 9],
}

fn contract_vpploc0(
    si: &CTensor,
    vlocg: &[f64],
    natm: usize,
    ngrids: usize,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    if si.re.len() != natm * ngrids || vlocg.len() != natm * ngrids {
        return Err(PbcGradError::ShapeMismatch {
            expected: natm * ngrids,
            got: si.re.len().min(vlocg.len()),
        }
        .into());
    }
    let (mut re, mut im) = (vec![0.0; ngrids], vec![0.0; ngrids]);
    for g in 0..ngrids {
        let mut terms_re = Vec::with_capacity(natm);
        let mut terms_im = Vec::with_capacity(natm);
        for ia in 0..natm {
            let (sr, si_im) = (si.re[ia * ngrids + g], si.im[ia * ngrids + g]);
            let v = vlocg[ia * ngrids + g];
            terms_re.push(-(sr * v));
            terms_im.push(-(si_im * v));
        }
        re[g] = oracle_sum(&terms_re);
        im[g] = oracle_sum(&terms_im);
    }
    Ok((re, im))
}

pub fn vpplocg_strain_derivatives(
    cell: &Cell,
    mesh: [usize; 3],
) -> Result<VpplocGStrain, PyscfRsError> {
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let natm = cell.mol.natm;
    let gv0 = pyscf_pbc_gto::get_gv(cell, Some(mesh))?;
    let si0 = pyscf_pbc_gto::get_si(cell, Some(&gv0), None, None)?;
    let vlocg0 = pyscf_pbc_gto::pseudo::get_vlocg(cell, &gv0)?;
    let (v0_re, v0_im) = contract_vpploc0(&si0, &vlocg0, natm, ngrids)?;
    let gamma: [[f64; 3]; 0] = [];
    let full_disp = 2.0 * STRAIN_FD_HALF_DISP;
    let mut v1_re: Vec<Vec<f64>> = Vec::with_capacity(9);
    let mut v1_im: Vec<Vec<f64>> = Vec::with_capacity(9);
    for x in 0..3 {
        for y in 0..3 {
            let pair = finite_diff_cells(cell, &gamma, x, y, full_disp)?;
            let gv_plus = pyscf_pbc_gto::get_gv(&pair.plus, Some(mesh))?;
            let gv_minus = pyscf_pbc_gto::get_gv(&pair.minus, Some(mesh))?;
            let v_plus = pyscf_pbc_gto::pseudo::get_vlocg(&pair.plus, &gv_plus)?;
            let v_minus = pyscf_pbc_gto::pseudo::get_vlocg(&pair.minus, &gv_minus)?;
            let (c_plus_re, c_plus_im) = contract_vpploc0(&si0, &v_plus, natm, ngrids)?;
            let (c_minus_re, c_minus_im) = contract_vpploc0(&si0, &v_minus, natm, ngrids)?;
            let mut row_re = Vec::with_capacity(ngrids);
            let mut row_im = Vec::with_capacity(ngrids);
            for g in 0..ngrids {
                row_re.push(oracle_sum(&[c_plus_re[g], -c_minus_re[g]]) / full_disp);
                row_im.push(oracle_sum(&[c_plus_im[g], -c_minus_im[g]]) / full_disp);
            }
            v1_re.push(row_re);
            v1_im.push(row_im);
        }
    }
    Ok(VpplocGStrain {
        ngrids,
        v0_re,
        v0_im,
        v1_re: v1_re.try_into().map_err(|_| {
            PyscfRsError::Core(CoreError::InvalidMolecule(
                "vpplocG strain: 9-component assembly failed".into(),
            ))
        })?,
        v1_im: v1_im.try_into().map_err(|_| {
            PyscfRsError::Core(CoreError::InvalidMolecule(
                "vpplocG strain: 9-component assembly failed".into(),
            ))
        })?,
    })
}

// ---------------------------------------------------------------------------
// 8. pp_nonloc strain derivatives — rks_stress.py:309-385
// ---------------------------------------------------------------------------

/// Non-local pseudopotential energy at one cell, gamma point:
///
/// ```text
/// E_nl = Tr(dm · Re V_nl) / vol
/// ```
///
/// `dm` is F-order (`(i,j)` at `i + j*nao`, matching [`CTensor`]); for the
/// symmetric densities of the stress path the index order is immaterial
/// (`18-CONTEXT` trap 9 is a gradient-path concern). `V_nl` is
/// [`get_pp_nl`](pyscf_pbc_gto::pseudo::get_pp_nl); the `/ vol` mirrors
/// upstream's `eval_pp_nonloc` (`:375`, `vppnl / (nkpts*vol)`).
pub fn pp_nonloc_energy(cell: &Cell, dm: &[f64]) -> Result<f64, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    if dm.len() != nao * nao {
        return Err(PbcGradError::ShapeMismatch {
            expected: nao * nao,
            got: dm.len(),
        }
        .into());
    }
    let vol = cell.vol();
    if !vol.is_finite() || vol <= 0.0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pp_nonloc energy: non-positive cell volume {vol}"
        ))));
    }
    let owned_gamma = [[0.0; 3]];
    let vnl = pyscf_pbc_gto::pseudo::get_pp_nl(cell, &owned_gamma)?;
    let mat = &vnl[0];
    let mut terms = Vec::with_capacity(nao * nao);
    for (d, v) in dm.iter().zip(mat.re.iter()) {
        terms.push(d * v);
    }
    Ok(oracle_sum(&terms) / vol)
}

/// `_get_pp_nonloc_strain_derivatives` (`rks_stress.py:309-385`).
///
/// Upstream finite-differences its `eval_pp_nonloc` (`ft_ao` + projectors,
/// `:377-384`); this port finite-differences [`pp_nonloc_energy`] (the
/// `get_pp_nl` contraction — the same scalar) at half step
/// `max(1e-5, sqrt(precision·0.1))` (`:377`), with k-points at fixed fractional
/// coordinates ([`finite_diff_cells`], D-PBC-31 clause 3). Cells with no
/// pseudopotential projectors yield exactly `0` (the `get_pp_nl` empty-blocks
/// branch), matching upstream's atom loop that `continue`s past them (`:343`).
///
/// The analytic replacement and the Gate-A3 assertion belong to 18-20; here the
/// symbol ships so 18-13 imports it from the base.
pub fn pp_nonloc_strain_derivatives(
    cell: &Cell,
    dm: &[f64],
) -> Result<[[f64; 3]; 3], PyscfRsError> {
    let disp = (cell.precision * 0.1).sqrt().max(1e-5);
    if !disp.is_finite() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "pp_nonloc strain: non-finite displacement from precision {}",
            cell.precision
        ))));
    }
    let full_disp = 2.0 * disp;
    let gamma: [[f64; 3]; 0] = [];
    let mut out = [[0.0; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            let pair = finite_diff_cells(cell, &gamma, x, y, full_disp)?;
            let e_plus = pp_nonloc_energy(&pair.plus, dm)?;
            let e_minus = pp_nonloc_energy(&pair.minus, dm)?;
            out[x][y] = oracle_sum(&[e_plus, -e_minus]) / full_disp;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 9. ewald strain — rks_stress.py:387-396
// ---------------------------------------------------------------------------

/// `ewald` (`rks_stress.py:387-396`): the Ewald nuclear-repulsion strain
/// derivative by central difference at half step
/// `max(1e-5, sqrt(precision·0.1))` (`:388`).
///
/// Symmetric fill (`out[j,i] = out[i,j]`, `:391-395`) — symmetric BY
/// CONSTRUCTION, like `coulG_1`. K-points are irrelevant (no electronic
/// states), so the gamma pair is used and the mesh stays pinned
/// ([`finite_diff_cells`], trap 5).
pub fn ewald_strain(cell: &Cell) -> Result<[[f64; 3]; 3], PyscfRsError> {
    let disp = (cell.precision * 0.1).sqrt().max(1e-5);
    if !disp.is_finite() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "ewald strain: non-finite displacement from precision {}",
            cell.precision
        ))));
    }
    let full_disp = 2.0 * disp;
    let gamma: [[f64; 3]; 0] = [];
    let mut out = [[0.0; 3]; 3];
    for x in 0..3 {
        for y in 0..=x {
            let pair = finite_diff_cells(cell, &gamma, x, y, full_disp)?;
            let e_plus = pyscf_pbc_gto::ewald(&pair.plus, None, None)?;
            let e_minus = pyscf_pbc_gto::ewald(&pair.minus, None, None)?;
            let v = oracle_sum(&[e_plus, -e_minus]) / full_disp;
            out[x][y] = v;
            out[y][x] = v;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 10. block sizing — D-PBC-30 clause 1 (18-12 Task 2)
// ---------------------------------------------------------------------------

/// `(ao_comps, strain_comps)` per strain order — the two arrays
/// `get_vxc`'s loop holds per block (`rks_stress.py:200-203`):
///
/// | order | `ao` from `block_loop(..., deriv+1)` | `ao_strain` |
/// |---|---|---|
/// | `deriv = 0` (LDA) | 4 | 9 |
/// | `deriv = 1` (GGA/MGGA) | 10 | 36 |
///
/// `None` for `deriv >= 2`: the same named refusal as the strain kernel
/// (no upstream caller, no oracle).
pub fn strain_ao_block_comps(deriv: u32) -> Option<(usize, usize)> {
    match deriv {
        0 => Some((4, 9)),
        1 => Some((10, 36)),
        _ => None,
    }
}

/// Choose the grid-block length against the SUM of both per-block AO arrays
/// (D-PBC-30 clause 1).
///
/// `block_loop` sizes `blk` from `max_memory` against the `ao` array ONLY — it
/// never sees `ao_strain`. Porting the caller as written therefore under-counts
/// the block's true footprint by **4.6× for GGA/MGGA** (`10` vs `10 + 36`) and
/// **3.25× for LDA** (`4` vs `4 + 9`); at the k-point scale 18-13 inherits that
/// is a 1.14 GiB block on diamond `gth-dzvp` 2×2×2 (`18-REVIEW §3.1`).
///
/// ```text
/// blk = floor(budget_bytes / (nao · (ao_comps + strain_comps) · bytes_per_elem)),
/// clamped to [1, ngrids].
/// ```
///
/// `bytes_per_elem` is 8 at gamma (real) and 16 at sampled k-points (complex).
/// Degenerate inputs (`ngrids == 0`, `nao == 0`, or a zero divisor) yield the
/// clamped empty/full-grid endpoints rather than a panic or a division by zero.
pub fn strain_block_size(
    ngrids: usize,
    nao: usize,
    ao_comps: usize,
    strain_comps: usize,
    bytes_per_elem: usize,
    max_memory_mb: f64,
) -> usize {
    if ngrids == 0 {
        return 0;
    }
    let per_grid = (nao as u128)
        .saturating_mul((ao_comps as u128).saturating_add(strain_comps as u128))
        .saturating_mul(bytes_per_elem as u128);
    if per_grid == 0 || !max_memory_mb.is_finite() || max_memory_mb <= 0.0 {
        return 1.min(ngrids);
    }
    let budget = (max_memory_mb * 1e6) as u128;
    let blk = (budget / per_grid).min(ngrids as u128).max(1);
    blk as usize
}

/// [`strain_block_size`] with the budget read from `PYSCF_MAX_MEMORY`
/// (MEGABYTES, the upstream convention; `aftdf.rs:84-85`), default `4000.0`.
///
/// Upstream's `blksize` subtracts `mem_now` twice (`fft_jk.py:361-363`,
/// `18-CONTEXT` trap 7); this port does not transcribe that arithmetic — one
/// budget, one division.
pub fn strain_block_size_from_env(
    ngrids: usize,
    nao: usize,
    ao_comps: usize,
    strain_comps: usize,
    bytes_per_elem: usize,
) -> usize {
    let budget_mb = std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(4000.0);
    strain_block_size(
        ngrids,
        nao,
        ao_comps,
        strain_comps,
        bytes_per_elem,
        budget_mb,
    )
}

/// The byte footprint of one block: `blk · nao · (ao_comps + strain_comps) ·
/// bytes_per_elem`. The gate asserts the CHOSEN `blk`'s footprint against the
/// budget — asserting the footprint, not just that the calculation runs.
pub fn strain_block_footprint(
    blk: usize,
    nao: usize,
    ao_comps: usize,
    strain_comps: usize,
    bytes_per_elem: usize,
) -> u128 {
    (blk as u128)
        .saturating_mul(nao as u128)
        .saturating_mul((ao_comps as u128).saturating_add(strain_comps as u128))
        .saturating_mul(bytes_per_elem as u128)
}

// ---------------------------------------------------------------------------
// 11. pressure units — rks_stress.py:19-21, kernel :456
// ---------------------------------------------------------------------------

/// Report a strain derivative as a stress tensor (`rks_stress.py:19-21`):
///
/// ```text
/// σ_ij = (1/V) dE/dε_ij.
/// ```
///
/// Upstream's assertion divides by `vol` (`test_rks_stress.py:406`,
/// `(e1-e2)/2e-3/vol`); the stress is a PRESSURE in Ha/Bohr³, never a bare
/// energy derivative (`18-CONTEXT §2.2`). The `kernel` (18-20) reports through
/// here so the dimension cannot drift again.
pub fn to_stress(ded_eps: [[f64; 3]; 3], vol: f64) -> Result<[[f64; 3]; 3], PyscfRsError> {
    if !vol.is_finite() || vol <= 0.0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "stress tensor: non-positive cell volume {vol} (stress is dE/deps divided by vol)"
        ))));
    }
    if ded_eps.iter().flatten().any(|v| !v.is_finite()) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "stress tensor: non-finite strain derivative".into(),
        )));
    }
    Ok(std::array::from_fn(|x| {
        std::array::from_fn(|y| ded_eps[x][y] / vol)
    }))
}
