//! 5 init_guess modes — declarations.
//! Plan 03-11 ships the '1e' body; minao/atom/huckel stay NotYetImplemented
//! until a Phase 3 follow-up plan or Phase 4 dependency lands.
//! Plan 03-06 ships the `Chkfile(path)` body (`init_guess_by_chkfile`).
use crate::{InitGuessMode, chkfile::load_scf_from_file, error::ScfError};
use pyscf_core::{Density, Mole, PyscfRsError};

pub fn default_get_init_guess(mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
    match mode {
        InitGuessMode::Minao => init_guess_by_minao(mol),
        InitGuessMode::Atom => init_guess_by_atom(mol),
        InitGuessMode::OneElectron => init_guess_by_1e(mol),
        InitGuessMode::Huckel => init_guess_by_huckel(mol),
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

/// Per-atom molecular AO range `[lo, hi)` (the in-tree `aoslice_by_atom`
/// equivalent). For each atom `ia`, walks `mol._bas` rows whose `ATOM_OF`
/// slot equals `ia` and unions their `ao_loc_nr[shell]..ao_loc_nr[shell+1]`
/// AO ranges. Shells are atom-ordered post-build, so each atom's AO block is
/// contiguous. Returns `natm` `(lo, hi)` pairs.
///
/// Shared by `init_guess_by_atom` (block placement) and `init_guess_by_huckel`
/// (occupied-orbital scatter). Mirrors `analyze.rs`'s `ATOM_OF` + `ao_loc_nr`
/// walk. AO-range / atom-index overruns return `Err` (never panics —
/// T-03-14-PANIC).
fn aoslice_by_atom(mol: &Mole) -> Result<Vec<(usize, usize)>, PyscfRsError> {
    use pyscf_core::CoreError;
    use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};

    let natm = mol.natm;
    let nbas = mol.nbas;
    let nao = mol.nao_nr;
    if mol.ao_loc_nr.len() <= nbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aoslice_by_atom: ao_loc_nr len {} <= nbas {} (Mole not built?)",
            mol.ao_loc_nr.len(),
            nbas
        ))));
    }
    // Per-atom [lo, hi). Initialize lo = nao (sentinel), hi = 0.
    let mut slices = vec![(nao, 0usize); natm];
    for shell in 0..nbas {
        let atom = mol._bas[shell * BAS_SLOTS + ATOM_OF] as usize;
        if atom >= natm {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aoslice_by_atom: _bas[{shell}, ATOM_OF] = {atom} but natm = {natm}"
            ))));
        }
        let lo = mol.ao_loc_nr[shell] as usize;
        let hi = mol.ao_loc_nr[shell + 1] as usize;
        if hi > nao || lo > hi {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aoslice_by_atom: shell {shell} AO range [{lo},{hi}) invalid (nao={nao})"
            ))));
        }
        let (cur_lo, cur_hi) = slices[atom];
        slices[atom] = (cur_lo.min(lo), cur_hi.max(hi));
    }
    // Atoms with no basis get an empty [lo, lo) range (lo defaults to its
    // contiguous position; if never touched the sentinel collapses to (nao,0)
    // → normalize to an empty slice anchored at the running offset).
    let mut next = 0usize;
    for slot in slices.iter_mut() {
        if slot.0 == nao && slot.1 == 0 {
            // Basis-less atom: empty range at the current running offset.
            *slot = (next, next);
        } else {
            next = slot.1;
        }
    }
    Ok(slices)
}

/// `init_guess_by_atom(mol)` — superposition of spherically-averaged atomic
/// densities (SCF-05). Port of `pyscf/scf/hf.py:495-535`.
///
/// Algorithm:
///   1. `atm_scf = get_atm_nrhf(mol)` — per-unique-element `(mo_coeff c, mo_occ
///      occ)` from the spherically-averaged atomic RHF (atom_hf.rs).
///   2. For each atom, build the per-atom density block
///      `atm_dm[μ,ν] = Σ_p occ[p]·c[μ,p]·c[ν,p]` (row-major, oracle_sum over p).
///   3. Place each block on the block-diagonal of the molecular `nao×nao`
///      Density at the atom's molecular AO range `[lo, hi)` (the
///      `scipy.linalg.block_diag` placement — atom blocks at their AO offsets).
///
/// `mo_coeff.data` is F-order (`c[μ,p] = data[μ + p*nao_atm]`). The atomic dm
/// is built from normalized atomic orbitals, so `Tr(D·S) ≈ nelec` (the minao
/// non-normalization caveat does NOT apply here).
///
/// Cartesian basis (upstream's `cart2sph` branch at hf.py:528-531) is out of
/// scope: STO-3G is spherical. A cartesian Mole returns a clear `Err`.
pub(crate) fn init_guess_by_atom(mol: &Mole) -> Result<Density, PyscfRsError> {
    use pyscf_core::CoreError;

    if mol.cart {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "init_guess_by_atom cartesian-basis cart2sph branch (spherical basis only)",
        });
    }

    let nao = mol.nao_nr;
    let atm_scf = crate::atom_hf::get_atm_nrhf(mol)?;
    let slices = aoslice_by_atom(mol)?;

    let mut data = vec![0.0f64; nao * nao];

    for (ia, (sym, _xyz)) in mol._atom.iter().enumerate() {
        let (lo, hi) = slices[ia];
        let nao_atm = hi - lo;
        if nao_atm == 0 {
            continue; // basis-less atom → zero block.
        }
        // Look up the element's atomic-RHF result (keyed by the exact symbol).
        let res = atm_scf.get(sym).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_atom: no atomic-RHF result for element '{sym}'"
            )))
        })?;
        let c = &res.mo_coeff;
        if c.nao != nao_atm {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_atom: atom {ia} ({sym}) molecular AO span {nao_atm} \
                 != atomic nao {} (basis mismatch)",
                c.nao
            ))));
        }
        let occ = &res.mo_occ;
        let nmo = c.nmo;
        // Per-atom block atm_dm[i,j] = Σ_p occ[p]·c[i,p]·c[j,p] (row-major),
        // placed at molecular [lo+i, lo+j]. oracle_sum over the occupied p.
        for i in 0..nao_atm {
            for j in 0..nao_atm {
                let terms: Vec<f64> = (0..nmo)
                    .filter(|&p| occ[p] != 0.0)
                    .map(|p| occ[p] * c.data[i + p * nao_atm] * c.data[j + p * nao_atm])
                    .collect();
                data[(lo + i) * nao + (lo + j)] = pyscf_algebra::oracle_sum(&terms);
            }
        }
    }

    Ok(Density { nao, data })
}

/// The Generalized Wolfsberg-Helmholtz parameter (non-updated rule).
/// Source: `pyscf/scf/hf.py:563-575` — `Kgwh`. The default
/// `init_guess_by_huckel` uses `updated_rule=False`, so `Kgwh = 1.75`
/// (constant; the updated-rule Δ-dependent form is `init_guess_by_mod_huckel`,
/// out of scope here).
const KGWH: f64 = 1.75;

/// `init_guess_by_huckel(mol)` — extended-Hückel (GWH) init guess (SCF-05).
/// Port of `pyscf/scf/hf.py:537-555` + `_init_guess_huckel_orbitals`
/// (`:577-670`), `updated_rule=False`.
///
/// Algorithm:
///   1. `atm = get_atm_nrhf(mol)` — per-element atomic-RHF orbitals/energies/occ.
///   2. Collect the OCCUPIED atomic orbitals into the molecular AO basis:
///      `orb_C[lo..hi, iocc] = c[:, iorb]` for each atomic MO with `occ>0`
///      (`lo..hi` = the atom's molecular AO range), `orb_E[iocc] = e[iorb]`.
///   3. `orb_S = orb_Cᵀ · S · orb_C` (S = molecular overlap).
///   4. GWH Hückel matrix: `orb_H[io,io] = orb_E[io]`; off-diagonal
///      `orb_H[io,jo] = 0.5·KGWH·orb_S[io,jo]·(orb_E[io]+orb_E[jo])`.
///   5. `(mo_E, atmo_C) = eigh_gen(orb_H, orb_S, nocc)`; back-transform
///      `mo_C[μ,k] = Σ_o orb_C[μ,o]·atmo_C[o,k]` (F-order).
///   6. Aufbau-fill (`default_get_occ`) + `default_make_rdm1`.
///
/// All reductions go through `oracle_sum`/`oracle_dot` (T-03-14-NUM). Cartesian
/// basis (upstream's `atcart2sph` branch) is out of scope (spherical only).
pub(crate) fn init_guess_by_huckel(mol: &Mole) -> Result<Density, PyscfRsError> {
    use pyscf_core::{CoreError, MOCoefficients};

    if mol.cart {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "init_guess_by_huckel cartesian-basis atcart2sph branch (spherical basis only)",
        });
    }

    let nao = mol.nao_nr;
    let atm_scf = crate::atom_hf::get_atm_nrhf(mol)?;
    let slices = aoslice_by_atom(mol)?;

    // 1+2. Count occupied atomic orbitals and scatter them into orb_C (F-order
    //       [nao, nocc]) with energies orb_E.
    let mut nocc = 0usize;
    for (ia, (sym, _xyz)) in mol._atom.iter().enumerate() {
        let (lo, hi) = slices[ia];
        if hi == lo {
            continue;
        }
        let res = atm_scf.get(sym).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_huckel: no atomic-RHF result for element '{sym}'"
            )))
        })?;
        nocc += res.mo_occ.iter().filter(|&&o| o > 0.0).count();
    }

    if nocc == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "init_guess_by_huckel: no occupied atomic orbitals (all-GHOST molecule?)".into(),
        )));
    }

    let mut orb_c = vec![0.0f64; nao * nocc]; // F-order [nao, nocc]
    let mut orb_e = vec![0.0f64; nocc];
    let mut iocc = 0usize;
    for (ia, (sym, _xyz)) in mol._atom.iter().enumerate() {
        let (lo, hi) = slices[ia];
        let nao_atm = hi - lo;
        if nao_atm == 0 {
            continue;
        }
        let res = atm_scf.get(sym).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_huckel: no atomic-RHF result for element '{sym}'"
            )))
        })?;
        let c = &res.mo_coeff;
        if c.nao != nao_atm {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "init_guess_by_huckel: atom {ia} ({sym}) AO span {nao_atm} != atomic nao {}",
                c.nao
            ))));
        }
        for iorb in 0..c.nmo {
            if res.mo_occ[iorb] > 0.0 {
                // orb_C[lo+i, iocc] = c[i, iorb]  (both F-order).
                for i in 0..nao_atm {
                    orb_c[(lo + i) + iocc * nao] = c.data[i + iorb * nao_atm];
                }
                orb_e[iocc] = res.mo_energy[iorb];
                iocc += 1;
            }
        }
    }
    debug_assert_eq!(iocc, nocc);

    // 3. orb_S = orb_Cᵀ · S · orb_C  (nocc × nocc, row-major).
    //    First SC = S · orb_C (F-order [nao, nocc]), then orb_S = orb_Cᵀ · SC.
    let s1e = crate::fock::default_get_ovlp(mol)?; // row-major nao×nao
    // sc[μ, o] = Σ_ν S[μ,ν]·orb_C[ν,o]  (F-order [nao, nocc]).
    let mut sc = vec![0.0f64; nao * nocc];
    for o in 0..nocc {
        for mu in 0..nao {
            let terms: Vec<f64> = (0..nao)
                .map(|nu| s1e.data[mu * nao + nu] * orb_c[nu + o * nao])
                .collect();
            sc[mu + o * nao] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    // orb_S[i, j] = Σ_μ orb_C[μ,i]·SC[μ,j]  (row-major nocc×nocc).
    let mut orb_s = vec![0.0f64; nocc * nocc];
    for i in 0..nocc {
        let ci = &orb_c[i * nao..i * nao + nao];
        for j in 0..nocc {
            let scj = &sc[j * nao..j * nao + nao];
            orb_s[i * nocc + j] = pyscf_algebra::oracle_dot(ci, scj);
        }
    }

    // 4. GWH Hückel matrix (row-major nocc×nocc, symmetric).
    let mut orb_h = vec![0.0f64; nocc * nocc];
    for io in 0..nocc {
        orb_h[io * nocc + io] = orb_e[io];
        for jo in 0..io {
            let v = 0.5 * KGWH * orb_s[io * nocc + jo] * (orb_e[io] + orb_e[jo]);
            orb_h[io * nocc + jo] = v;
            orb_h[jo * nocc + io] = v;
        }
    }

    // 5. Solve generalized eig in the minimal orbital basis, back-transform to AO.
    let (mo_e, atmo_c) =
        pyscf_algebra::eigh_gen(&orb_h, &orb_s, nocc).map_err(ScfError::Algebra)?;
    // atmo_c is F-order nocc×nocc: atmo_C[o, k] = atmo_c[o + k*nocc].
    // mo_C[μ, k] = Σ_o orb_C[μ,o]·atmo_C[o,k]  (F-order [nao, nocc]).
    let mut mo_c = vec![0.0f64; nao * nocc];
    for k in 0..nocc {
        for mu in 0..nao {
            let terms: Vec<f64> = (0..nocc)
                .map(|o| orb_c[mu + o * nao] * atmo_c[o + k * nocc])
                .collect();
            mo_c[mu + k * nao] = pyscf_algebra::oracle_sum(&terms);
        }
    }

    // 6. Aufbau-fill the Hückel MOs, build the RDM1.
    let occ = crate::occ::default_get_occ(&mo_e, mol.nelectron)?;
    let mo = MOCoefficients {
        nao,
        nmo: nocc,
        data: mo_c,
        energies: mo_e,
        occupations: occ,
    };
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
