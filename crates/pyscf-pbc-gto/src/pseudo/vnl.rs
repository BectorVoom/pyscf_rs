//! GTH **non-local** pseudopotential — `V_nl` (plan 10-06).
//!
//! Ports `pyscf/pbc/gto/pseudo/pp_int.py`:
//! `:577-624` (`fake_cell_vnl`), `:626-674` (`_int_vnl`) and `:408-438`
//! (`get_pp_nl`, the general k-point branch).
//!
//! # The form
//!
//! PRB 58, 3641 Eq (2): the non-local part is a low-rank projector sandwich
//!
//! ```text
//! V_nl = Σ_atoms Σ_l Σ_{i,j} |p_i^l> h^l_{ij} <p_j^l|
//! ```
//!
//! with Gaussian projectors `p_i^l(r) ∝ r^{l+2i} exp(−r²/2r_l²)`.
//!
//! # How it is evaluated
//!
//! `fake_cell_vnl` turns each projector into an ordinary **single-primitive
//! Gaussian shell** of exponent `α = 1/(2 r_l²)`, i.e. the `i = 0` projector
//! exactly. The higher projectors differ from it by a factor `r^{2i}`, which is
//! why the half-overlaps
//!
//! ```text
//! <p_0^l | phi_mu>,   <p_0^l | r² | phi_mu>,   <p_0^l | r⁴ | phi_mu>
//! ```
//!
//! — `int1e_ovlp`, `int1e_r2_origi`, `int1e_r4_origi`, all with the origin on
//! the projector centre — are enough for `nproj ≤ 3`, and why `h^l` is rescaled
//! by [`PLI_FAC`] to renormalise `r^{2i} p_0^l` back to a unit projector
//! (`pp_int.py:606-616`).
//!
//! Each half-overlap is a PERIODIC 2-centre integral (bra = projector, ket = the
//! translated AO), so it goes straight through
//! [`crate::pbc_intor::intor_cross`] — no new lattice-sum machinery.
//!
//! # cintx prerequisite
//!
//! `int1e_r{2,4}_origi` are cintx `unstable-source-api` symbols; this crate's
//! `gth-pp` feature (default-on) enables them, and
//! `tests/cintx_moment_weighted_available.rs` re-proves on every run that they
//! evaluate AND differ from the unweighted parent (PBC-MASTER-PLAN §2.4 / R-13).

use crate::cell::Cell;
use crate::pbc_intor::{PbcIntorOpts, PbcIntorOutput, intor_cross};
use crate::types::CellBuildArgs;
use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, ParsedBasis, PyscfRsError, ShellSpec};
use std::collections::HashMap;

/// `sqrt(Gamma(l+3/2) / Gamma(l+2i+3/2))` for `l = 0..=7`, `i = 0..=2` —
/// `pp_int.py:566-575`, stored as the SQUARED denominators upstream tabulates
/// (`_PLI_FAC = 1/sqrt(...)`), so `pli_fac(l, i) = 1/sqrt(PLI_FAC[l][i])`.
///
/// This is the renormalisation that lets one Gaussian shell stand in for all
/// `nproj` projectors of a channel: `r^{2i} p_0^l` is not unit-normalised, and
/// the factor is folded into `h^l` rather than into the basis.
pub const PLI_FAC: [[f64; 3]; 8] = [
    [1.0, 3.75, 59.0625],
    [1.0, 8.75, 216.5625],
    [1.0, 15.75, 563.0625],
    [1.0, 24.75, 1206.5625],
    [1.0, 35.75, 2279.0625],
    [1.0, 48.75, 3936.5625],
    [1.0, 63.75, 6359.0625],
    [1.0, 80.75, 9750.5625],
];

/// The largest `nproj` the [`PLI_FAC`] table (and hence the three half-overlap
/// operators) supports. Every GTH potential in the shipped `.dat` files is
/// within it.
pub const MAX_NPROJ: usize = 3;

/// The three half-overlap operators of `_int_vnl` (`pp_int.py:630`), in
/// projector order: `<p_0|·>`, `<p_0|r²|·>`, `<p_0|r⁴|·>`.
pub const VNL_INTORS: [&str; MAX_NPROJ] = ["int1e_ovlp", "int1e_r2_origi", "int1e_r4_origi"];

/// One `h^l` coupling block, already rescaled by [`PLI_FAC`].
#[derive(Debug, Clone, PartialEq)]
pub struct HlBlock {
    /// Index of the atom this channel sits on.
    pub atom: usize,
    /// Angular momentum of the channel.
    pub l: usize,
    /// `nproj` for this channel — the dimension of [`HlBlock::h`].
    pub dim: usize,
    /// The rescaled symmetric `dim x dim` matrix, row-major.
    pub h: Vec<f64>,
}

/// The projector basis, split by projector rank.
///
/// `cells[i]` holds the channels with `nproj > i`, because `_int_vnl`
/// (`pp_int.py:670-673`) evaluates operator `i` only over those — a channel with
/// one projector never needs `<p|r⁴|·>`. `None` means no channel reached that
/// rank, and the corresponding half-overlap is skipped entirely.
#[derive(Debug, Clone)]
pub struct FakeCellVnl {
    /// One projector cell per rank, `None` where no channel has that rank.
    pub cells: [Option<Cell>; MAX_NPROJ],
    /// The `h^l` blocks, in atom-major then angular-momentum order — the SAME
    /// order the projector cells lay their shells out in, which is what makes
    /// the running `offset` in [`get_pp_nl`] line up with the AO rows.
    pub blocks: Vec<HlBlock>,
}

/// `fake_cell_vnl(cell)` — `pp_int.py:577-624`.
///
/// Returns the projector cells and their (rescaled) `h^l` blocks. A cell with no
/// pseudopotential, or one whose potentials are purely local, yields empty
/// blocks and three `None` cells.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when a channel needs more than [`MAX_NPROJ`]
/// projectors (outside the [`PLI_FAC`] table), or when a projector cell fails
/// to build.
pub fn fake_cell_vnl(cell: &Cell) -> Result<FakeCellVnl, PyscfRsError> {
    let Some(pseudo) = cell.pseudo.as_ref() else {
        return Ok(FakeCellVnl {
            cells: [None, None, None],
            blocks: Vec::new(),
        });
    };

    // --- The h^l blocks, atom-major (pp_int.py:589-618). ---
    let charges = cell.atom_charges();
    let mut blocks: Vec<HlBlock> = Vec::new();
    for (ia, (label, _)) in cell.mol._atom.iter().enumerate() {
        // pp_int.py:590 — ghost atoms carry basis functions but no potential.
        if charges.get(ia).copied().unwrap_or(0) == 0 {
            continue;
        }
        let Some(pp) = pseudo.get(label) else {
            continue;
        };
        for (l, proj) in pp.projectors.iter().enumerate() {
            if proj.nproj == 0 {
                continue;
            }
            if proj.nproj > MAX_NPROJ {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "fake_cell_vnl: atom {ia} ('{label}') channel l={l} has nproj={} \
                     but the GTH projector renormalisation table covers at most {MAX_NPROJ}",
                    proj.nproj
                ))));
            }
            if l >= PLI_FAC.len() {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "fake_cell_vnl: atom {ia} ('{label}') has an l={l} projector channel; \
                     the renormalisation table stops at l={}",
                    PLI_FAC.len() - 1
                ))));
            }
            // pp_int.py:616 — hl = fac_i * hl_ij * fac_j, fac_i = PLI/rl^(2i).
            let n = proj.nproj;
            let fac: Vec<f64> = (0..n)
                .map(|i| 1.0 / PLI_FAC[l][i].sqrt() / proj.r.powi(2 * i as i32))
                .collect();
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] = fac[i] * proj.h[i * n + j] * fac[j];
                }
            }
            blocks.push(HlBlock {
                atom: ia,
                l,
                dim: n,
                h,
            });
        }
    }

    // --- The projector cells, one per rank. ---
    let mut cells: [Option<Cell>; MAX_NPROJ] = [None, None, None];
    for (rank, slot) in cells.iter_mut().enumerate() {
        *slot = build_projector_cell(cell, rank)?;
    }
    Ok(FakeCellVnl { cells, blocks })
}

/// The projector `Cell` holding every channel with `nproj > rank`.
///
/// The projector `p_0^l` is a single Gaussian primitive of exponent
/// `α = 1/(2 r_l²)` (`pp_int.py:604`), unit-normalised. Upstream writes the
/// normalisation constant into `_env` by hand (`fake_env.append([alpha, norm])`
/// with `norm = gto_norm(l, alpha)`); handing a raw coefficient of 1.0 to the
/// standard basis pipeline produces the identical number, because
/// `normalise_contractions` of a lone primitive IS `gto_norm`.
fn build_projector_cell(cell: &Cell, rank: usize) -> Result<Option<Cell>, PyscfRsError> {
    let pseudo = cell
        .pseudo
        .as_ref()
        .expect("caller checked cell.pseudo is Some");

    // Per-ELEMENT shells. Every atom of an element carries the same channels, so
    // the element-keyed basis reproduces upstream's per-atom loop exactly.
    let mut per_element: HashMap<String, pyscf_gto::BasisInput> = HashMap::new();
    let mut any = false;
    for (label, _) in cell.mol._atom.iter() {
        let key = crate::pseudo::normalise_symbol(label);
        if per_element.contains_key(&key) {
            continue;
        }
        let mut shells: Vec<ShellSpec> = Vec::new();
        if let Some(pp) = pseudo.get(label) {
            for (l, proj) in pp.projectors.iter().enumerate() {
                if proj.nproj > rank {
                    shells.push(ShellSpec {
                        l: l as u8,
                        exponents: vec![0.5 / (proj.r * proj.r)],
                        coeffs: vec![vec![1.0]],
                    });
                }
            }
        }
        any |= !shells.is_empty();
        per_element.insert(key, pyscf_gto::BasisInput::Parsed(ParsedBasis { shells }));
    }
    if !any {
        return Ok(None);
    }

    let atoms: Vec<(String, [f64; 3])> = cell.mol._atom.clone();
    // Upstream's fake cell is `cell.copy(deep=False)` with `_bas` swapped, so it
    // INHERITS `rcut`, `precision`, `mesh` rather than re-estimating them from
    // the (much more compact) projector basis. `_int_vnl:629` then takes
    // `max(cell.rcut, fakecell.rcut)`, which is `cell.rcut`. Pinning them here
    // reproduces that; letting `build` re-estimate would silently shorten the
    // lattice sum.
    let fake = Cell::build(CellBuildArgs {
        mole: pyscf_gto::MoleBuildArgs {
            atom: pyscf_gto::AtomInput::Tuples(atoms),
            basis: pyscf_gto::BasisInput::PerElement(per_element),
            unit: pyscf_core::Unit::Bohr,
            charge: cell.mol.charge,
            spin: cell.mol.spin,
            ..Default::default()
        },
        a: crate::types::ALattice::Matrix(cell.a),
        mesh: Some(cell.mesh),
        rcut: Some(cell.rcut),
        precision: cell.precision,
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        pseudo: cell.pseudo_name.clone(),
        ..Default::default()
    })?;
    Ok(Some(fake))
}

/// `_int_vnl(cell, fakecell, hl_blocks, kpts)` — `pp_int.py:626-674`.
///
/// Returns the three half-overlaps `<p_0^l | O_i | phi_mu>` (`O_0 = 1`,
/// `O_1 = r²`, `O_2 = r⁴`, origin on the projector centre), one entry per
/// projector rank; `None` where that rank has no channels.
///
/// Each is `[nkpts]` matrices of shape `(n_proj_ao_rank, nao)`, F-order.
///
/// # Errors
/// As [`intor_cross`]. A cintx failure on `int1e_r{2,4}_origi` means the
/// `gth-pp` feature is off — see the module docs.
pub fn int_vnl(
    cell: &Cell,
    fake: &FakeCellVnl,
    kpts: &[[f64; 3]],
) -> Result<[Option<PbcIntorOutput>; MAX_NPROJ], PyscfRsError> {
    let mut out: [Option<PbcIntorOutput>; MAX_NPROJ] = [None, None, None];
    for (rank, slot) in out.iter_mut().enumerate() {
        let Some(fcell) = fake.cells[rank].as_ref() else {
            continue;
        };
        // bra = projector (so `origi` puts the r^n origin on it), ket = the AO
        // that gets translated by L — `shls_slice = (cell.nbas, nbas, 0, cell.nbas)`
        // at `pp_int.py:645`.
        *slot = Some(intor_cross(
            VNL_INTORS[rank],
            fcell,
            cell,
            kpts,
            PbcIntorOpts::default(),
        )?);
    }
    Ok(out)
}

/// `get_pp_nl(cell, kpts)` — `pp_int.py:408-438`.
///
/// The k-resolved non-local pseudopotential matrix, one `nao x nao` F-order
/// `CTensor` per k-point:
///
/// ```text
/// V_nl^k[p, q] = Σ_blocks Σ_{i,j} Σ_m conj(P_i[m, p]) · h_{ij} · P_j[m, q]
/// ```
///
/// where `P_i[m, p]` is the half-overlap of the block's `m`-th angular component
/// with AO `p`. Ports the general-k numpy branch (`pp_int.py:422-437`); the
/// gamma branch upstream routes through a C driver computes the same sum, and
/// the port checks that by asserting the gamma result is real.
///
/// # Errors
/// As [`int_vnl`], plus [`CoreError::InvalidMolecule`] if a projector cell and
/// its `h^l` blocks disagree about how many AO rows exist (an internal
/// inconsistency that would otherwise read past the half-overlap).
pub fn get_pp_nl(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();

    let fake = fake_cell_vnl(cell)?;
    if fake.blocks.is_empty() {
        return Ok(vec![CTensor::zeros(nao * nao); nkpts]);
    }
    let halves = int_vnl(cell, &fake, kpts)?;

    let mut out = vec![CTensor::zeros(nao * nao); nkpts];
    for (k, out_k) in out.iter_mut().enumerate() {
        // `offset[i]` walks the AO rows of projector cell `i`, advancing by
        // `2l+1` for every block that HAS an i-th projector — the same running
        // counter as `pp_int.py:424,431-433`.
        let mut offset = [0usize; MAX_NPROJ];
        for block in &fake.blocks {
            let nd = 2 * block.l + 1;
            let dim = block.dim;

            // Gather P_i[m, p] for this block: dim x nd x nao.
            let mut p = vec![(0.0_f64, 0.0_f64); dim * nd * nao];
            for i in 0..dim {
                let half = halves[i].as_ref().ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "get_pp_nl: block (atom {}, l {}) claims {dim} projectors but \
                         projector cell {i} is empty",
                        block.atom, block.l
                    )))
                })?;
                let ni = half.ni;
                let row0 = offset[i];
                if row0 + nd > ni {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "get_pp_nl: projector cell {i} has {ni} AO rows but block \
                         (atom {}, l {}) needs rows {row0}..{}",
                        block.atom,
                        block.l,
                        row0 + nd
                    ))));
                }
                for m in 0..nd {
                    for q in 0..nao {
                        // F-order (ni, nao): element (row, col) at row + col*ni.
                        let src = (row0 + m) + q * ni;
                        p[(i * nd + m) * nao + q] = (half.at(k).re[src], half.at(k).im[src]);
                    }
                }
                offset[i] = row0 + nd;
            }

            // ppnl[p,q] += Σ_{i,j,m} conj(P_i[m,p]) h_ij P_j[m,q].
            for i in 0..dim {
                for j in 0..dim {
                    let hij = block.h[i * dim + j];
                    if hij == 0.0 {
                        continue;
                    }
                    for m in 0..nd {
                        let pi = &p[(i * nd + m) * nao..(i * nd + m + 1) * nao];
                        let pj = &p[(j * nd + m) * nao..(j * nd + m + 1) * nao];
                        for (pp_idx, (ar, ai)) in pi.iter().enumerate() {
                            // conj(P_i[m,p]) * h * P_j[m,q]
                            let (car, cai) = (*ar * hij, -*ai * hij);
                            for (qq, (br, bi)) in pj.iter().enumerate() {
                                let o = pp_idx + qq * nao;
                                out_k.re[o] += car * br - cai * bi;
                                out_k.im[o] += car * bi + cai * br;
                            }
                        }
                    }
                }
            }
        }
    }

    // `pp_int.py:262-268` takes `.real` of the half-overlaps at gamma; the
    // equivalent statement here is that the assembled matrix has no imaginary
    // part to lose.
    for (k, kpt) in kpts.iter().enumerate() {
        if crate::pbc_intor::is_gamma(kpt) {
            let max_im = out[k].im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_im > 1e-9 {
                tracing::warn!(
                    "get_pp_nl: gamma-point V_nl has max|Im| = {max_im:e}; \
                     upstream drops it, but this means the projector lattice sum is off"
                );
            }
            out[k].im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(out)
}
