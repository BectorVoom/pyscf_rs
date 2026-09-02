//! 5 init_guess modes — declarations.
//! Plan 03-11 ships the '1e' body; minao/atom/huckel stay NotYetImplemented
//! until a Phase 3 follow-up plan or Phase 4 dependency lands.
//! Plan 03-06 ships the `Chkfile(path)` body (`init_guess_by_chkfile`).
use crate::{InitGuessMode, chkfile::load_scf_from_file, error::ScfError};
use pyscf_core::{Density, MOCoefficients, Mole, PyscfRsError};

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
/// Two cases:
///   - **Same basis** (`prior.nao == current nao`): direct RDM1 from the prior
///     MOs (byte-verified Phase 3 path).
///   - **Different basis** (`prior.nao != current nao`): the prior MOs are
///     projected onto the current basis first, via `project_mo_nr2nr`
///     (`pyscf/scf/addons.py`): `C2 = S22⁻¹·(S21·C1)` with `S22 = <cur|cur>`
///     and `S21 = <cur|prior>`. The prior molecule/basis is reconstructed from
///     the `mol` JSON stored in the same chkfile.
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
    if prior.mo_coeff.nao == nao {
        // Same-basis fast path (byte-verified): D[μν] = Σ_i occ_i C[μ,i] C[ν,i].
        return Ok(density_from_mos(
            nao,
            prior.mo_coeff.nmo,
            &prior.mo_coeff.data,
            &prior.mo_occ,
        ));
    }

    // Cross-basis: reconstruct the prior Mole from the chkfile's stored JSON,
    // then project the prior MOs onto the current basis before forming the seed.
    let mol_json = crate::chkfile::load_mol_json_from_file(path).map_err(|e| {
        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
            "chkfile read prior mol: {}",
            e
        )))
    })?;
    let prior_mol = pyscf_gto::loads(&mol_json)?;
    project_density_from_chkfile(mol, &prior_mol, &prior.mo_coeff, &prior.mo_occ)
}

/// Build a restricted RDM1 from MO coefficients: `D[μν] = Σ_i occ_i·C[μ,i]·C[ν,i]`.
///
/// `coeff` is F-order (`C[μ,i] = coeff[μ + i*nao]`, PySCF/LAPACK convention);
/// the output `Density.data` is row-major. The `i`-axis reduction goes through
/// `oracle_sum` (Pitfall 9 — no bare `+=`), matching rdm.rs / energy.rs.
fn density_from_mos(nao: usize, nmo: usize, coeff: &[f64], occ: &[f64]) -> Density {
    let mut data = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            let terms: Vec<f64> = (0..nmo)
                .map(|i| occ[i] * coeff[mu + i * nao] * coeff[nu + i * nao])
                .collect();
            data[mu * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Density { nao, data }
}

/// Project prior-basis MO coefficients onto the current basis, then build the
/// seed density. Faithful port of `project_mo_nr2nr` (`pyscf/scf/addons.py`):
///
/// ```text
/// C2 = S22⁻¹ · (S21 · C1),   S22 = <cur|cur> = int1e_ovlp(cur),
///                            S21 = <cur|prior> = intor_cross(cur, prior)
/// ```
///
/// then `D[μν] = Σ_i occ_i·C2[μ,i]·C2[ν,i]`. Mirrors the layout/solve idiom of
/// `init_guess_by_minao` (which is `project_mo_nr2nr(ano, I, mol)`): overlaps
/// are F-order in `IntorOutput.values`, and each MO column is recovered with a
/// per-column `solve_linear` against `S22`.
///
/// Verification scope: exercised in-tree by the same-basis identity invariant
/// (`prior == current ⇒ C2 == C1`) plus symmetry/finiteness checks. Exact
/// cross-basis numbers are NOT oracle-verified against live PySCF in-sandbox.
fn project_density_from_chkfile(
    mol: &Mole,
    prior_mol: &Mole,
    prior_coeff: &MOCoefficients,
    prior_occ: &[f64],
) -> Result<Density, PyscfRsError> {
    use pyscf_core::CoreError;
    use pyscf_gto::{intor, intor_cross};

    let nao = mol.nao_nr;
    let prior_nao = prior_coeff.nao;
    let nmo = prior_coeff.nmo;
    if prior_mol.nao_nr != prior_nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "chkfile projection: reconstructed prior mol nao_nr {} != stored mo_coeff nao {}",
            prior_mol.nao_nr, prior_nao
        ))));
    }
    if prior_occ.len() < nmo {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "chkfile projection: mo_occ len {} < nmo {}",
            prior_occ.len(),
            nmo
        ))));
    }

    // S22 = <cur|cur> (nao×nao, F-order); S21 = <cur|prior> (nao×prior_nao, F-order).
    let s22 = intor(mol, "int1e_ovlp_sph")?;
    let s21 = intor_cross(mol, prior_mol, "int1e_ovlp_sph")?;

    // B = S21 · C1  (nao × nmo, F-order). B[μ,i] = Σ_k S21[μ,k]·C1[k,i],
    //   S21[μ,k] = s21.values[μ + k*nao];  C1[k,i] = prior_coeff.data[k + i*prior_nao].
    let mut b = vec![0.0_f64; nao * nmo];
    for i in 0..nmo {
        for mu in 0..nao {
            let terms: Vec<f64> = (0..prior_nao)
                .map(|k| s21.values[mu + k * nao] * prior_coeff.data[k + i * prior_nao])
                .collect();
            b[mu + i * nao] = pyscf_algebra::oracle_sum(&terms);
        }
    }

    // C2[:,i] = S22⁻¹ · B[:,i] (per-column LU solve, as in init_guess_by_minao).
    let mut c2 = vec![0.0_f64; nao * nmo];
    for i in 0..nmo {
        let rhs = &b[i * nao..i * nao + nao];
        let col = pyscf_algebra::solve_linear(&s22.values, rhs, nao).map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "chkfile projection: S22 solve failed (MO col {i}): {e}"
            )))
        })?;
        c2[i * nao..i * nao + nao].copy_from_slice(&col);
    }

    Ok(density_from_mos(nao, nmo, &c2, prior_occ))
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

/// Per-atom molecular AO range `[lo, hi)` — re-exported from
/// [`pyscf_gto::aoslice_by_atom`].
///
/// U-02 step 5: this was one of three textually identical copies (here,
/// `pyscf-grad/src/rhf.rs`, and a `Cell`-shaped one in `pyscf-pbc-symm`). The
/// molecular one now lives beside `Mole` in `pyscf-gto`, which is what lets
/// `pyscf-pbc-scf` call it without depending on `pyscf-grad`.
use pyscf_gto::aoslice_by_atom;

/// Cartesian-basis init guess via a spherical sibling + `cart2sph` projection.
///
/// Upstream's `cart2sph` branch (`pyscf/scf/hf.py:528-531` / huckel `atcart2sph`)
/// builds the guess density in spherical space and projects it into the
/// cartesian molecular basis with `D_cart = C · D_sph · Cᵀ`, where
/// `C = mol.cart2sph_coeff('sp')`. We reuse the fully-validated spherical guess
/// by rebuilding a spherical sibling Mole (same atoms + basis, `cart=false`) via
/// the F-11 `Mole::build()` IoC hook, running `sph_guess` on it, then projecting.
///
/// `sph_guess` MUST be the spherical body of the caller (`init_guess_by_atom` /
/// `init_guess_by_huckel`) — invoked on the `cart=false` sibling it takes the
/// non-cart path, so this never recurses.
fn cart_init_guess_via_spherical(
    mol: &Mole,
    sph_guess: fn(&Mole) -> Result<Density, PyscfRsError>,
) -> Result<Density, PyscfRsError> {
    use pyscf_core::CoreError;

    // Spherical sibling. register_mole_builder() guarantees the F-11 hook is
    // armed regardless of whether a gto front-door ran earlier this process.
    pyscf_gto::register_mole_builder();
    let mut mol_sph = mol.clone();
    mol_sph.cart = false;
    mol_sph._built = false;
    mol_sph.build()?;

    let d_sph = sph_guess(&mol_sph)?;
    let (c, nao_sph) = pyscf_gto::cart2sph_coeff(mol)?;
    if d_sph.nao != nao_sph {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cart init guess: spherical density nao {} != cart2sph_coeff nao_sph {}",
            d_sph.nao, nao_sph
        ))));
    }
    let nao_cart = mol.nao_nr;
    Ok(project_dm_sph_to_cart(&d_sph.data, nao_sph, &c, nao_cart))
}

/// `D_cart = C · D_sph · Cᵀ`, with `C` row-major `[nao_cart × nao_sph]`
/// (`C[a*nao_sph + i]`) and `D_sph` row-major `[nao_sph × nao_sph]`. Reductions
/// go through `oracle_sum` (T-03-14-NUM).
fn project_dm_sph_to_cart(d_sph: &[f64], nao_sph: usize, c: &[f64], nao_cart: usize) -> Density {
    // M[a, j] = Σ_i C[a, i] · D_sph[i, j]   (intermediate [nao_cart × nao_sph]).
    let mut m = vec![0.0f64; nao_cart * nao_sph];
    for a in 0..nao_cart {
        for j in 0..nao_sph {
            let terms: Vec<f64> = (0..nao_sph)
                .map(|i| c[a * nao_sph + i] * d_sph[i * nao_sph + j])
                .collect();
            m[a * nao_sph + j] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    // D_cart[a, b] = Σ_j M[a, j] · C[b, j].
    let mut data = vec![0.0f64; nao_cart * nao_cart];
    for a in 0..nao_cart {
        for b in 0..nao_cart {
            let terms: Vec<f64> = (0..nao_sph)
                .map(|j| m[a * nao_sph + j] * c[b * nao_sph + j])
                .collect();
            data[a * nao_cart + b] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Density {
        nao: nao_cart,
        data,
    }
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
        // Cartesian branch: build D_sph on a spherical sibling, project to cart.
        return cart_init_guess_via_spherical(mol, init_guess_by_atom);
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
        // Cartesian branch: build D_sph on a spherical sibling, project to cart.
        return cart_init_guess_via_spherical(mol, init_guess_by_huckel);
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

#[cfg(test)]
mod chkfile_projection_tests {
    use super::{density_from_mos, project_density_from_chkfile};
    use pyscf_core::{MOCoefficients, Unit};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    fn h2(basis: &str) -> pyscf_core::Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Ang,
            ..Default::default()
        })
        .expect("build H2")
    }

    fn coeff(nao: usize, nmo: usize, data: Vec<f64>) -> MOCoefficients {
        MOCoefficients {
            nao,
            nmo,
            data,
            energies: vec![0.0; nmo],
            occupations: vec![0.0; nmo],
        }
    }

    /// Core correctness anchor: projecting onto the SAME basis must be the
    /// identity (`S22 == S21 == S ⇒ C2 = S⁻¹·S·C1 = C1`), so the projected
    /// density equals the direct same-basis RDM1. This exercises the full
    /// `S21·C1` contraction + per-column `S22` solve and proves it collapses
    /// to identity — independent of any external oracle.
    #[test]
    fn projection_same_basis_is_identity() {
        let mol = h2("sto-3g");
        let nao = mol.nao_nr;
        assert_eq!(nao, 2);
        // Arbitrary non-trivial prior coefficients (F-order [nao, nmo]).
        let c = coeff(nao, nao, vec![0.8, 0.1, 0.2, 0.9]);
        let occ = vec![2.0, 0.0];

        let projected = project_density_from_chkfile(&mol, &mol, &c, &occ).unwrap();
        let direct = density_from_mos(nao, nao, &c.data, &occ);

        assert_eq!(projected.nao, nao);
        for k in 0..nao * nao {
            assert!(
                (projected.data[k] - direct.data[k]).abs() < 1e-8,
                "elem {k}: projected {} vs direct {} (Δ={:.3e})",
                projected.data[k],
                direct.data[k],
                (projected.data[k] - direct.data[k]).abs()
            );
        }
    }

    /// Cross-basis projection (sto-3g → 6-31g) must produce a well-formed seed:
    /// shaped to the CURRENT basis (nao=4), finite, and symmetric. Exact
    /// numbers are not oracle-checked in-sandbox (see fn doc / AUDIT-FIX report).
    #[test]
    fn projection_cross_basis_is_finite_and_symmetric() {
        let prior_mol = h2("sto-3g");
        let cur_mol = h2("6-31g");
        let prior_nao = prior_mol.nao_nr;
        let cur_nao = cur_mol.nao_nr;
        assert_eq!(prior_nao, 2);
        assert!(cur_nao > prior_nao, "6-31g must be larger than sto-3g");

        // One doubly-occupied prior MO (bonding-like) + one virtual.
        let c = coeff(prior_nao, prior_nao, vec![0.55, 0.55, 1.0, -1.0]);
        let occ = vec![2.0, 0.0];

        let d = project_density_from_chkfile(&cur_mol, &prior_mol, &c, &occ).unwrap();
        assert_eq!(d.nao, cur_nao);
        assert_eq!(d.data.len(), cur_nao * cur_nao);
        assert!(
            d.data.iter().all(|x| x.is_finite()),
            "density has non-finite entries"
        );
        for i in 0..cur_nao {
            for j in 0..cur_nao {
                let a = d.data[i * cur_nao + j];
                let b = d.data[j * cur_nao + i];
                assert!((a - b).abs() < 1e-10, "asymmetry at ({i},{j}): {a} vs {b}");
            }
        }
    }
}
