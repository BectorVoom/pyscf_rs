//! 5 init_guess modes — declarations.
//! Plan 03-11 ships the '1e' body; minao/atom/huckel stay NotYetImplemented
//! until a Phase 3 follow-up plan or Phase 4 dependency lands.
//! Plan 03-06 ships the `Chkfile(path)` body (`init_guess_by_chkfile`).
use crate::{chkfile::load_scf_from_file, error::ScfError, InitGuessMode};
use pyscf_core::{Density, Mole, PyscfRsError};

pub fn default_get_init_guess(
    mol: &Mole,
    mode: &InitGuessMode,
) -> Result<Density, PyscfRsError> {
    match mode {
        InitGuessMode::Minao => {
            Err(ScfError::InitGuessNotYetImplemented("minao", "03-03 follow-up").into())
        }
        InitGuessMode::Atom => {
            Err(ScfError::InitGuessNotYetImplemented("atom", "03-03 follow-up").into())
        }
        InitGuessMode::OneElectron => init_guess_by_1e(mol),
        InitGuessMode::Huckel => {
            Err(ScfError::InitGuessNotYetImplemented("huckel", "03-03 follow-up").into())
        }
        InitGuessMode::Chkfile(path) => init_guess_by_chkfile(mol, path),
        InitGuessMode::UserDM(d) => Ok(d.clone()),
    }
}

/// `init_guess_by_chkfile(mol, path)` — port from `pyscf/scf/hf.py:673-763`.
///
/// Reads an upstream-PySCF-written chkfile (or pyscf-rs-written),
/// reconstructs the density matrix from `(mo_coeff, mo_occ)`, and returns
/// it for seeding SCF.
///
/// Density formula: `D[μν] = Σ_i mo_occ[i] * mo_coeff[μ, i] * mo_coeff[ν, i]`
///
/// Phase 3 ships the SIMPLE case — same basis (prior.nao == current nao).
/// Basis-projection (`pyscf/scf/hf.py:673-763` general case) is deferred:
/// returns `NotYetImplemented{phase:3}` on a basis mismatch so a future
/// plan can fill the projection path without changing this function's
/// signature.
pub(crate) fn init_guess_by_chkfile(
    mol: &Mole,
    path: &std::path::Path,
) -> Result<Density, PyscfRsError> {
    tracing::info!(
        target: "pyscf_scf::init_guess",
        chkfile = %path.display(),
        "init_guess_by_chkfile reading prior SCF state"
    );
    let prior = load_scf_from_file(path).map_err(|e| {
        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "chkfile read: {}",
            e
        )))
    })?;
    let nao = mol.nao_nr;
    if prior.mo_coeff.nao != nao {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "init_guess_by_chkfile basis projection (prior.nao != current nao); same-basis case only in Phase 3",
        });
    }
    // Build density (row-major, the Density.data convention).
    // D[mu,nu] = sum_i occ_i * C[mu,i] * C[nu,i]
    // MOCoefficients.data is F-order: C[mu, i] = data[mu + i * nao].
    let nmo = prior.mo_coeff.nmo;
    let mut data = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            // Materialize the i-axis terms into a scratch Vec so the
            // reduction goes through oracle_sum (Pitfall 9 mitigation —
            // matches plan 03-11's rdm.rs / energy.rs pattern).
            let terms: Vec<f64> = (0..nmo)
                .map(|i| {
                    prior.mo_occ[i]
                        * prior.mo_coeff.data[mu + i * nao]
                        * prior.mo_coeff.data[nu + i * nao]
                })
                .collect();
            data[mu * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Ok(Density { nao, data })
}

/// `init_guess_by_1e(mol)` — build the initial density by diagonalizing
/// the one-electron Hamiltonian `h_core = T + V_nuc`, Aufbau-filling the
/// MOs, and constructing the RDM1. Plan 03-11 fill (SCF-05 partial — '1e'
/// mode only; minao/atom/huckel stay NotYetImplemented).
///
/// Source: `pyscf/scf/hf.py:485-494` — `def init_guess_by_1e(mol)`.
pub(crate) fn init_guess_by_1e(mol: &Mole) -> Result<Density, PyscfRsError> {
    let h_core = crate::fock::default_get_hcore(mol)?;
    let s1e = crate::fock::default_get_ovlp(mol)?;
    let mut mo = crate::eig::default_eig(&h_core, &s1e)?;
    let occ = crate::occ::default_get_occ(&mo.energies, mol.nelectron)?;
    mo.occupations = occ;
    crate::rdm::default_make_rdm1(&mo)
}

/// Parse a string mode name (used by oracle Arm 4).
pub fn parse_init_guess_mode(name: &str) -> Result<InitGuessMode, PyscfRsError> {
    match name {
        "minao" => Ok(InitGuessMode::Minao),
        "atom" => Ok(InitGuessMode::Atom),
        "1e" => Ok(InitGuessMode::OneElectron),
        "huckel" => Ok(InitGuessMode::Huckel),
        other => Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!("unknown init_guess mode '{}'", other),
        ))),
    }
}
