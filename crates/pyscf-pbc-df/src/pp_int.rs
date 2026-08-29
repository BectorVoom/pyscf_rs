//! `aft._IntPPBuilder.get_pp_loc_part2` — the k-resolved short-range half of
//! the GTH local pseudopotential.
//!
//! # The gap this closes
//!
//! Phase 10 ported `pp_int.get_pp_loc_part2` at **gamma only**
//! (`pseudo::vloc_part2`); its k-resolved counterpart is upstream's
//! `aft._IntPPBuilder`, which Phase 13 declined and plan 14-03 found blocking
//! every k-point pseudopotential path in AFTDF, GDF, MDF and RSDF. FFTDF was
//! unaffected because it evaluates the WHOLE local part in G-space through
//! `get_vlocG` and never calls part 2 — which is why Phases 11 and 12 could run
//! `KRHF`/`KRKS` on diamond 2×2×2 while this one could not.
//!
//! # It is `aux_e2` with a different operator, and nothing else
//!
//! `_IntPPBuilder` is an `Int3cBuilder` over the `fake_cell_vloc` auxiliary
//! basis with `j_only = True`, i.e. exactly
//!
//! ```text
//! V2[k][mu nu] = SUM_cn SUM_{L1,L2} e^{i k (L2 - L1)} ( mu(L1) nu(L2) | g_P^{cn}(0) )
//! ```
//!
//! which is [`crate::incore::aux_e2_intor`] with `kptij = [(k, k)]` and the
//! operator taken from `PART2_INTORS`. Plan 14-01 already built that sum,
//! Bloch phases included; this module supplies the auxiliary cell and the `cn`
//! loop.
//!
//! # Two conventions that are load-bearing
//!
//! * **`EPS_PPL = 1e-2`.** The auxiliary cell is screened far more loosely than
//!   `cell.precision`, because the `C_n` terms are a short-range CORRECTION
//!   whose absolute size is small (`pp_int.py:34`). It is applied by giving the
//!   fake cell that `precision`, so `rcut_by_shells` inside `aux_e2` picks it up.
//! * **The coefficients are RAW, not unit-normalised.**
//!   [`VlocAux::rescale_from_unit_norm`] converts, and it rides in the
//!   `AuxCell::modrho_scale` slot — the same seam
//!   `make_modrho_basis` uses for the density-fitting auxiliary basis.

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, ParsedBasis, PyscfRsError, ShellSpec};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::pseudo::vloc::{VlocAux, fake_cell_vloc};
use pyscf_pbc_gto::pseudo::vloc_part2::{EPS_PPL, PART2_INTORS};

use crate::error::PbcDfError;
use crate::incore::auxcell::build_aux_cell;
use crate::incore::int3c::KptPair;
use crate::incore::{Aosym, AuxCell, aux_e2_intor};

/// Wrap one `C_n` term's `fake_cell_vloc` output as an [`AuxCell`].
///
/// The auxiliary basis is one `s` primitive per atom, so a per-ELEMENT basis
/// map reproduces it exactly — provided every atom of an element carries the
/// same entry, which `fake_cell_vloc` guarantees because the exponent and
/// coefficient come from the element's pseudopotential. An element with SOME
/// atoms skipped (a ghost) would break that, and is refused rather than
/// silently mis-built.
fn vloc_auxcell(cell: &Cell, aux: &[VlocAux]) -> Result<AuxCell, PbcDfError> {
    let mut basis: std::collections::HashMap<String, ParsedBasis> =
        std::collections::HashMap::new();
    for a in aux {
        let sym = cell.mol._atom[a.atom].0.clone();
        let shell = ShellSpec {
            l: 0,
            exponents: vec![a.alpha],
            coeffs: vec![vec![1.0]],
        };
        match basis.get(&sym) {
            None => {
                basis.insert(sym, ParsedBasis { shells: vec![shell] });
            }
            Some(prev) => {
                if (prev.shells[0].exponents[0] - a.alpha).abs() > 1e-12 {
                    return Err(PbcDfError::Core(PyscfRsError::Core(
                        CoreError::InvalidMolecule(format!(
                            "get_pp_loc_part2_kpts: element '{sym}' has atoms with \
                             different local-pseudopotential exponents; the \
                             per-element auxiliary basis cannot express that"
                        )),
                    )));
                }
            }
        }
    }

    let mut fake = build_aux_cell(cell, basis)?;
    // `EPS_PPL` — the auxiliary screen upstream deliberately loosens.
    fake.precision = EPS_PPL;

    // Write the RAW `_env` coefficient the auxiliary function actually carries.
    //
    // `aux_e2`'s screens read `_env` (through `rcut_by_shells` and
    // `libcint_ctr_coeff_max`) while the cintx path reads `_basis`, which
    // `normalise_contractions` leaves UNIT-norm. Phase 10's gamma route screens
    // on `pgf_rcut(0, alpha, |coeff|, EPS_PPL)` with the raw coefficient, so
    // leaving `_env` unit-normalised here gives the two routes different image
    // sets — measured as a 1.04e-11 disagreement at gamma, where they should be
    // identical. `modrho_scale` still carries the integral scaling, so the
    // numbers are untouched; only the screening lines up.
    //
    // Atoms of one element share a `PTR_COEFF` slot (the 14-02 lesson), and
    // they carry the same coefficient here by construction, so writing once per
    // distinct pointer is both correct and idempotent.
    {
        use pyscf_core::raw_layout::{BAS_SLOTS, PTR_COEFF};
        let mut seen = std::collections::HashSet::new();
        for (ib, a) in aux.iter().enumerate().take(fake.mol.nbas) {
            let pc = fake.mol._bas[ib * BAS_SLOTS + PTR_COEFF].max(0) as usize;
            if seen.insert(pc) {
                fake.mol._env[pc] = a.coeff;
            }
        }
    }
    fake.rcut = pyscf_pbc_gto::estimate_rcut(&fake, EPS_PPL)?;
    fake._rcut_from_build = false;

    if fake.mol.nbas != aux.len() {
        return Err(PbcDfError::Core(PyscfRsError::Core(
            CoreError::InvalidMolecule(format!(
                "get_pp_loc_part2_kpts: the fake cell has {} shells for {} auxiliary \
                 functions — an atom of a pseudopotential element was skipped \
                 (a ghost?), which the per-element basis cannot express",
                fake.mol.nbas,
                aux.len()
            )),
        )));
    }

    // One AO per s shell, in atom order — the same order `fake_cell_vloc` emits.
    let modrho_scale: Vec<f64> = aux.iter().map(VlocAux::rescale_from_unit_norm).collect();
    Ok(AuxCell {
        cell: fake,
        modrho_scale,
    })
}

/// `_IntPPBuilder.get_pp_loc_part2()` — the k-resolved `C_1 … C_4` local terms.
///
/// Returns one ROW-MAJOR `nao × nao` complex matrix per k-point.
///
/// # Errors
/// Propagates [`fake_cell_vloc`] and the 3-centre lattice sum, and refuses a
/// cell whose atoms of one element carry different local pseudopotentials.
pub fn get_pp_loc_part2_kpts(
    cell: &Cell,
    kpts: &[[f64; 3]],
) -> Result<Vec<CTensor>, PbcDfError> {
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    let mut out: Vec<CTensor> = (0..nkpts).map(|_| CTensor::zeros(nao * nao)).collect();
    if cell.pseudo.is_none() || nao == 0 {
        return Ok(out);
    }

    // `j_only = True`: the part-2 matrix is diagonal in k.
    let kptij: Vec<KptPair> = kpts.iter().map(|&k| KptPair { ki: k, kj: k }).collect();

    for cn in 1..=4usize {
        let aux = fake_cell_vloc(cell, cn)?;
        if aux.is_empty() {
            continue;
        }
        let auxcell = vloc_auxcell(cell, &aux)?;
        // ONE image list for every C_n term (`pp_int.py:153`), taken from the
        // real cell exactly as the gamma route does.
        let rcut = cell.try_rcut()?;
        let blocks = aux_e2_intor(
            cell,
            &auxcell,
            PART2_INTORS[cn],
            Aosym::S1,
            &kptij,
            Some(rcut),
        )?;
        let naux = auxcell.naux();
        for (k, b) in blocks.iter().enumerate() {
            for p in 0..nao * nao {
                // Contract over the auxiliary index: the `C_n` coefficients are
                // already folded in through `modrho_scale`.
                for a in 0..naux {
                    out[k].re[p] += b.re[p * naux + a];
                    out[k].im[p] += b.im[p * naux + a];
                }
            }
        }
    }

    // The gamma block is real by construction (`pp_int.py`); drop the residue.
    for (k, m) in out.iter_mut().enumerate() {
        if pyscf_pbc_gto::is_zero(&kpts[k]) {
            m.im.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    Ok(out)
}
