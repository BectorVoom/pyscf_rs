//! 5 init_guess modes — declarations.
//! Plan 03-11 ships the '1e' body; minao/atom/huckel stay NotYetImplemented
//! until a Phase 3 follow-up plan or Phase 4 dependency lands.
//! Plan 03-06 ships the `Chkfile(path)` body (`init_guess_by_chkfile`).
use crate::{InitGuessMode, chkfile::load_scf_from_file, error::ScfError};
use pyscf_core::{Density, Mole, PyscfRsError};

pub fn default_get_init_guess(mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
    match mode {
        InitGuessMode::Minao => init_guess_by_minao(mol),
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

/// `init_guess_by_minao(mol)` — the DEFAULT init guess. Faithful port of
/// `pyscf/scf/hf.py:init_guess_by_minao`: build a minimal ANO-derived atomic
/// reference, then project its block-diagonal atomic density onto the working
/// basis. (Plan 03-13; SCF-05 — completes the default init-guess path.)
///
/// Algorithm:
///   1. Reference Mole `ano` = `mol`'s atoms with the `'ano'` basis.
///   2. Occupation vector over the ANO AOs: for each (element, l-shell), the
///      first `ndocc` contractions are doubly occupied (2.0), the frontier one
///      carries `frac`, the rest 0 — `(ndocc, frac) = frac_occ(Z, l)`.
///   3. `mo = S_working⁻¹ · S_cross` where `S_cross = <working|ANO>`
///      (`intor_cross`); this is `project_mo_nr2nr(ano, I, mol)`.
///   4. `dm[μν] = Σ_p occ_p · mo[μ,p] · mo[ν,p]` (oracle_sum, no bare `+=`).
///
/// Verified byte-for-byte against the upstream H2 docstring density. NOTE: the
/// vendored `ano.dat` may load one contraction per l for some elements; atoms
/// whose minimal occupation needs >1 contraction per l (e.g. C/O 1s+2s) rely on
/// that data's coverage — validated on H/H2O, see init_guess_minao.rs.
pub(crate) fn init_guess_by_minao(mol: &Mole) -> Result<Density, PyscfRsError> {
    use pyscf_core::{CoreError, Unit};
    use pyscf_gto::format_atom::charge_for_symbol;
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor, intor_cross};

    let nao = mol.nao_nr;

    // 1. ANO reference Mole — same atoms (Bohr), basis "ano".
    let ano = M(MoleBuildArgs {
        atom: AtomInput::Tuples(mol._atom.clone()),
        basis: BasisInput::Name("ano".to_string()),
        unit: Unit::Bohr,
        charge: mol.charge,
        spin: mol.spin,
        cart: mol.cart,
        ..Default::default()
    })?;
    let nao_ano = ano.nao_nr;
    if nao_ano == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "init_guess_by_minao: ANO reference built with 0 AOs".into(),
        )));
    }

    // 2. Occupation vector over the ANO AOs (walk atoms → shells → contractions
    //    → m, matching the cintx general-contraction AO layout build_atoms_and_shells
    //    produces). frac_occ gives the per-(element,l) minimal occupations.
    let mut occ = vec![0.0f64; nao_ano];
    let mut ao = 0usize;
    for (sym, _xyz) in &ano._atom {
        let elem: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
        let z = charge_for_symbol(&elem).unwrap_or(0).max(0) as usize;
        let upper = elem.to_ascii_uppercase();
        let parsed = ano._basis.get(&upper).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_minao: no ANO basis entry for '{upper}'"
            )))
        })?;
        for spec in &parsed.shells {
            let l = spec.l as usize;
            let nctr = spec.coeffs.len();
            let (ndocc, frac) = crate::atom_config::frac_occ(z, l);
            let m_degen = 2 * l + 1;
            for ctr in 0..nctr {
                let occ_ctr = if (ctr as u32) < ndocc {
                    2.0
                } else if ctr as u32 == ndocc {
                    frac
                } else {
                    0.0
                };
                for _m in 0..m_degen {
                    if ao >= nao_ano {
                        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "init_guess_by_minao: occ walk overran nao_ano={nao_ano} (ANO AO-layout mismatch)"
                        ))));
                    }
                    occ[ao] = occ_ctr;
                    ao += 1;
                }
            }
        }
    }
    if ao != nao_ano {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "init_guess_by_minao: occ walk covered {ao} AOs but ANO has {nao_ano} (AO-layout mismatch)"
        ))));
    }

    // 3. Overlaps: S_working (nao×nao) and S_cross = <working|ANO> (nao×nao_ano), F-order.
    let s_w = intor(mol, "int1e_ovlp_sph")?;
    let s_cross = intor_cross(mol, &ano, "int1e_ovlp_sph")?;

    // 4. mo[:,p] = S_working⁻¹ · S_cross[:,p] (per-ANO-column LU solve).
    //    mo stored F-order [nao, nao_ano]: mo[μ + p*nao].
    let mut mo = vec![0.0f64; nao * nao_ano];
    for p in 0..nao_ano {
        let rhs = &s_cross.values[p * nao..p * nao + nao];
        let col = pyscf_algebra::solve_linear(&s_w.values, rhs, nao).map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_minao: S_working solve failed (ANO col {p}): {e}"
            )))
        })?;
        mo[p * nao..p * nao + nao].copy_from_slice(&col);
    }

    // 5. dm[μ,ν] = Σ_p occ_p · mo[μ,p] · mo[ν,p] — row-major Density; oracle_sum
    //    over the (occupied) ANO columns (no bare +=).
    let mut data = vec![0.0f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            let terms: Vec<f64> = (0..nao_ano)
                .filter(|&p| occ[p] != 0.0)
                .map(|p| occ[p] * mo[mu + p * nao] * mo[nu + p * nao])
                .collect();
            data[mu * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }

    Ok(Density { nao, data })
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
