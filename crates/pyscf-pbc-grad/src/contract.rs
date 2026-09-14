//! Gamma-point real derivative/density contraction, pbc/grad/rhf.py:90-131.
//! Unlike the k-point trace in krhf.grad_elec, upstream's C worker contracts
//! matching indices `vhf[x,i,j] * dm[i,j]`, NOT `dm[j,i]`.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::{Cell, NeighborList};

/// Default screening switch for `_contract_vhf_dm` at the gamma-gradient
/// call sites (`gamma_rhf`, `gamma_uhf`).
///
/// Named by 18-17 Task 1 (`measurements/sizings.md` §Task 1): direct
/// `_contract_vhf_dm` calls with a dense deterministic dm (`A+A.T`,
/// least-screenable input), screen True vs False, give
/// `max|de_screened − de_unscreened| = 0.0` — bitwise-identical output — on
/// all five reference cells (diamond, si, lif, he_fcc, graphene). The wall
/// ratio favouring unscreened (5–73×) is pure per-call
/// `build_neighbor_list_for_shlpairs` construction overhead dominating a
/// sub-ms contraction at `nao ≤ 8`, not the contraction cost, and does not
/// transfer to production sizes. Ruling: keep upstream's
/// `SCREEN_VHF_DM_CONTRA = True` (`pbc/grad/rhf.py:30`). Both branches stay
/// (the neighbour list already exists); the Gate-E finite-difference gate
/// runs against this screened default.
pub const SCREEN_VHF_DM_CONTRACT: bool = true;

fn invalid(message: &str) -> PyscfRsError {
    CoreError::InvalidMolecule(format!("contract_vhf_dm: {message}")).into()
}

/// Contract real gamma matrices stored in the port's column-major AO layout.
///
/// A supplied neighbor list screens shell pairs with no surviving images.
/// `None` evaluates all pairs. Screening is explicit at this low-level seam;
/// the method-level default is reserved for 18-17's completed measurements.
/// Imaginary input is refused because upstream's gamma C worker is real-only.
pub fn contract_vhf_dm(
    cell: &Cell,
    vhf: &[CTensor; 3],
    dm: &CTensor,
    neighbor_list: Option<&NeighborList>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let size = nao
        .checked_mul(nao)
        .ok_or_else(|| invalid("AO count overflow"))?;
    for matrix in std::iter::once(dm).chain(vhf.iter()) {
        if matrix.re.len() != size || matrix.im.len() != size {
            return Err(invalid("AO matrix plane shape mismatch"));
        }
        if matrix.re.iter().any(|x| !x.is_finite()) || matrix.im.iter().any(|&x| x != 0.0) {
            return Err(invalid("finite real gamma matrices required"));
        }
    }
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol)?;
    let atoms = pyscf_gto::aoslice::shell_atoms(&cell.mol)?;
    let ns = cell.mol.nbas;
    let mut active = vec![
        neighbor_list.is_none();
        ns.checked_mul(ns)
            .ok_or_else(|| invalid("shell count overflow"))?
    ];
    if let Some(nl) = neighbor_list {
        if nl.nish != ns || nl.njsh != ns || nl.per_image.len() != nl.nimgs {
            return Err(invalid("neighbor list shape mismatch"));
        }
        for pairs in &nl.per_image {
            for &(i, j) in pairs {
                if i >= ns || j >= ns {
                    return Err(invalid("neighbor shell index out of range"));
                }
                active[i * ns + j] = true;
            }
        }
    }
    let mut partials: Vec<[Vec<f64>; 3]> = slices
        .iter()
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();
    for ish in 0..ns {
        for jsh in 0..ns {
            if !active[ish * ns + jsh] {
                continue;
            }
            let p = cell.mol.ao_loc_nr[ish] as usize..cell.mol.ao_loc_nr[ish + 1] as usize;
            let q = cell.mol.ao_loc_nr[jsh] as usize..cell.mol.ao_loc_nr[jsh + 1] as usize;
            for c in 0..3 {
                let mut terms = Vec::new();
                for i in p.clone() {
                    for j in q.clone() {
                        let t = i + j * nao;
                        terms.push(vhf[c].re[t] * dm.re[t]);
                    }
                }
                partials[atoms[ish]][c].push(oracle_sum(&terms));
            }
        }
    }
    Ok(partials
        .into_iter()
        .map(|p| p.map(|v| oracle_sum(&v)))
        .collect())
}
