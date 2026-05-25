//! Per-element spherically-averaged atomic RHF — the shared engine that BOTH
//! the `atom` and `huckel` init guesses consume (plan 03-14, SCF-05).
//!
//! Source-of-truth: `pyscf/scf/atom_hf.py:27-205`
//!   - `get_atm_nrhf(mol)` (`:27-86`) — for each UNIQUE element symbol, build a
//!     single-atom neutral Mole and run a spherically-averaged RHF, caching the
//!     `(mo_energy, mo_coeff, mo_occ)` per element.
//!   - `AtomSphAverageRHF.eig` (`:109-140`) — the ANGULAR-AVERAGED Fock solve:
//!     group AOs by angular momentum `l`, reshape each per-`l` block to
//!     `(nsh, degen, nsh, degen)`, average over the diagonal m-index
//!     (`einsum('piqi->pq') / degen`), `eigh` the `(nsh × nsh)` averaged block,
//!     then scatter the eigvecs back across the `(2l+1)` m-components.
//!   - `AtomSphAverageRHF.get_occ` (`:142-171`) — spherically-averaged
//!     fractional occupancy via `frac_occ(Z, l)` per l-shell, repeated over
//!     `degen` (no ECP core in scope here — STO-3G has no ECP).
//!   - `AtomHF1e` (`:192-193`) — 1-electron element: same eig, no 2e term.
//!
//! Conventions replicated (no exploration needed):
//!   - `Density.data` is ROW-MAJOR (`D[μ,ν] = data[μ*nao + ν]`).
//!   - `MOCoefficients.data` is COLUMN-MAJOR / F-order (`C[μ,i] = data[μ + i*nao]`).
//!   - `_bas[shell*BAS_SLOTS + ANG_OF]` = the shell's angular momentum `l`.
//!   - `ao_loc_nr[shell]..ao_loc_nr[shell+1]` = the shell's AO range.
//!   - all reductions go through `oracle_sum`/`oracle_dot` (Pitfall 9 / T-03-14-NUM).
//!   - no `unwrap()`/`expect()`/`panic!` in production (FOUND-07 / T-03-14-PANIC):
//!     every fallible op uses `?` or returns an explicit `Err`.

// Task 1 (this commit) ships the engine; Tasks 2 & 3 of plan 03-14 wire
// `get_atm_nrhf` into `init_guess_by_atom` / `init_guess_by_huckel`, at which
// point this allow becomes inert. Kept module-scoped so the Task-1 commit is
// independently `-D warnings`-clean while the consumers are still landing.
#![allow(dead_code)]

use std::collections::HashMap;

use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS};
use pyscf_core::{CoreError, Density, MOCoefficients, Mole, PyscfRsError, Unit};
use pyscf_gto::format_atom::charge_for_symbol;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

use crate::error::ScfError;

/// Maximum angular-momentum channel handled by the spherically-averaged
/// solve. Upstream uses `param.L_MAX = 8`; STO-3G / minimal bases never
/// exceed l = 3 (f), but we walk up to L_MAX for parity with upstream's
/// `for l in range(param.L_MAX)` loop.
const L_MAX: usize = 8;

/// Tight energy-convergence tolerance for the small per-atom SCF.
const ATOM_CONV_TOL: f64 = 1e-9;
/// Max SCF cycles for the per-atom solve.
const ATOM_MAX_CYCLE: u32 = 100;

/// Per-element spherically-averaged atomic-RHF result.
///
/// `mo_coeff` is F-order (`nao_atm × nao_atm`), `mo_energy` / `mo_occ` are
/// length `nao_atm`. For a GHOST / basis-less atom every field is empty / zero.
#[derive(Debug, Clone)]
pub(crate) struct AtomScfResult {
    pub(crate) mo_coeff: MOCoefficients,
    pub(crate) mo_energy: Vec<f64>,
    pub(crate) mo_occ: Vec<f64>,
}

/// `get_atm_nrhf(mol)` — per-UNIQUE-element spherically-averaged atomic RHF.
///
/// Returns a map from element symbol (as it appears in `mol._atom`, e.g. `"H"`,
/// `"O"`) to its `AtomScfResult`. Mirrors upstream's
/// `if element in atm_scf_result: continue` caching: each distinct symbol is
/// solved once.
///
/// Source: `pyscf/scf/atom_hf.py:27-86`.
pub(crate) fn get_atm_nrhf(mol: &Mole) -> Result<HashMap<String, AtomScfResult>, PyscfRsError> {
    let mut out: HashMap<String, AtomScfResult> = HashMap::new();

    for (sym, xyz) in &mol._atom {
        if out.contains_key(sym) {
            continue;
        }
        let result = solve_one_element(mol, sym, *xyz)?;
        out.insert(sym.clone(), result);
    }

    Ok(out)
}

/// Build a single-atom neutral Mole (working basis restricted to this element)
/// and run the spherically-averaged RHF.
fn solve_one_element(mol: &Mole, sym: &str, xyz: [f64; 3]) -> Result<AtomScfResult, PyscfRsError> {
    // Pure element symbol (strip any trailing index / ghost markers, mirroring
    // `init_guess_by_minao`'s `chars().take_while(is_alphabetic)`).
    let elem: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
    let z = charge_for_symbol(&elem).unwrap_or(0).max(0) as usize;

    // Per-element basis from the molecule's own `_basis` map (keyed UPPER).
    let upper = elem.to_ascii_uppercase();
    let parsed = match mol._basis.get(&upper) {
        Some(p) => p.clone(),
        None => {
            // No basis assigned to this element → GHOST-like (nao 0).
            return Ok(empty_result(0));
        }
    };

    // Spin parity per upstream `atm.spin = atm.nelectron % 2`.
    let spin = (z % 2) as i32;

    // Single-atom Mole at the same coordinates (Bohr), neutral, working basis.
    let atm = M(MoleBuildArgs {
        atom: AtomInput::Tuples(vec![(sym.to_string(), xyz)]),
        basis: BasisInput::Parsed(parsed),
        unit: Unit::Bohr,
        charge: 0,
        spin,
        cart: false, // AtomSphAverageRHF does not support cartesian basis.
        ..Default::default()
    })?;

    let nao = atm.nao_nr;
    // GHOST / no-basis / no-electron case (upstream atom_hf.py:58-61).
    if nao == 0 || atm.nelectron == 0 {
        return Ok(empty_result(nao));
    }

    // Pre-compute the per-l AO grouping for the angular-averaged eig.
    let groups = angular_groups(&atm)?;

    // Spherically-averaged fractional occupations (constant across the SCF;
    // upstream's get_occ does not depend on mo_energy ordering — it fills by
    // l-shell index). occ is length nao, in the AO order produced by `groups`.
    let mo_occ = spherical_occ(&atm, z, &groups)?;

    // One-electron pieces.
    let h_core = crate::fock::default_get_hcore(&atm)?;
    let s1e = crate::fock::default_get_ovlp(&atm)?;

    // 1-electron element: no 2e term (upstream AtomHF1e).
    let one_electron = atm.nelectron == 1;

    // Small self-consistent loop. Seed: diagonalize h_core, fill by mo_occ.
    let (_e0, c0) = angular_averaged_eig(&h_core, &s1e, nao, &groups)?;
    let mut dm = build_atomic_dm(&c0, &mo_occ, nao)?;

    let mut last_e = f64::INFINITY;
    let mut converged = false;
    let mut mo_energy = vec![0.0f64; nao];
    let mut mo_coeff_flat = vec![0.0f64; nao * nao];

    for _cycle in 0..ATOM_MAX_CYCLE {
        // Fock = h_core (+ veff if many-electron).
        let fock = if one_electron {
            h_core.clone()
        } else {
            let veff = crate::fock::default_get_veff(&atm, &dm)?;
            let data: Vec<f64> = h_core
                .data
                .iter()
                .zip(&veff.data)
                .map(|(a, b)| a + b)
                .collect();
            Density { nao, data }
        };

        // Angular-averaged eig.
        let (e, c) = angular_averaged_eig(&fock, &s1e, nao, &groups)?;

        // Energy: E = 0.5·Tr[D·(h_core + Fock)] for many-electron RHF;
        // E = Tr[D·h_core] for the 1-electron case.
        let e_elec = if one_electron {
            trace_dm_times(&dm, &h_core, nao)
        } else {
            let sum_hf: Vec<f64> = h_core
                .data
                .iter()
                .zip(&fock.data)
                .map(|(a, b)| a + b)
                .collect();
            0.5 * trace_dm_times(&dm, &Density { nao, data: sum_hf }, nao)
        };

        mo_energy = e;
        mo_coeff_flat = c;

        // New density from the fresh MOs.
        dm = build_atomic_dm(&mo_coeff_flat, &mo_occ, nao)?;

        if (e_elec - last_e).abs() < ATOM_CONV_TOL {
            converged = true;
            break;
        }
        last_e = e_elec;
    }

    if !converged {
        return Err(ScfError::ConvergenceFailure {
            cycles: ATOM_MAX_CYCLE,
            last_diff: (last_e - mo_energy.first().copied().unwrap_or(0.0)).abs(),
        }
        .into());
    }

    let mo_coeff = MOCoefficients {
        nao,
        nmo: nao,
        data: mo_coeff_flat,
        energies: mo_energy.clone(),
        occupations: mo_occ.clone(),
    };

    Ok(AtomScfResult {
        mo_coeff,
        mo_energy,
        mo_occ,
    })
}

/// A GHOST / basis-less element result: zero MOs / occ of size `nao`.
fn empty_result(nao: usize) -> AtomScfResult {
    AtomScfResult {
        mo_coeff: MOCoefficients {
            nao,
            nmo: nao,
            data: vec![0.0; nao * nao],
            energies: vec![0.0; nao],
            occupations: vec![0.0; nao],
        },
        mo_energy: vec![0.0; nao],
        mo_occ: vec![0.0; nao],
    }
}

/// Per-`l` AO grouping for the single-atom Mole.
///
/// Returns, for each angular momentum `l` that has shells on the atom, the
/// list of AO indices belonging to that `l`, ordered shell-by-shell and within
/// a shell m-major then contraction-major to match libcint's AO layout
/// (`ao_loc_nr[shell]..ao_loc_nr[shell+1]` is the contiguous AO block of one
/// shell, which spans `nctr * (2l+1)` AOs).
///
/// The returned `Vec<(l, ao_indices)>` is sorted by `l` ascending so the
/// reassembled MO/occ ordering is deterministic.
struct AngularGroups {
    /// `(l, degen, ao_indices)` per occupied angular channel.
    groups: Vec<(usize, usize, Vec<usize>)>,
}

fn angular_groups(atm: &Mole) -> Result<AngularGroups, PyscfRsError> {
    let nbas = atm.nbas;
    if atm.ao_loc_nr.len() <= nbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "atom_hf: ao_loc_nr len {} <= nbas {} (Mole not built?)",
            atm.ao_loc_nr.len(),
            nbas
        ))));
    }

    // Collect AO indices per l (across all shells of that l on this atom).
    let mut per_l: Vec<Vec<usize>> = vec![Vec::new(); L_MAX];
    for shell in 0..nbas {
        let l = atm._bas[shell * BAS_SLOTS + ANG_OF] as usize;
        if l >= L_MAX {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "atom_hf: shell {shell} has l={l} >= L_MAX={L_MAX}"
            ))));
        }
        let lo = atm.ao_loc_nr[shell] as usize;
        let hi = atm.ao_loc_nr[shell + 1] as usize;
        for ao in lo..hi {
            per_l[l].push(ao);
        }
    }

    let mut groups = Vec::new();
    for (l, aos) in per_l.into_iter().enumerate() {
        if aos.is_empty() {
            continue;
        }
        let degen = 2 * l + 1;
        if !aos.len().is_multiple_of(degen) {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "atom_hf: l={l} has {} AOs, not a multiple of degen={degen}",
                aos.len()
            ))));
        }
        groups.push((l, degen, aos));
    }

    Ok(AngularGroups { groups })
}

/// Spherically-averaged fractional occupations (length nao), built from
/// `frac_occ(Z, l)` per l-shell, repeated over `degen` and ordered to match the
/// reassembled MO ordering (per-l block, nsh MOs per block scattered over
/// degen). Mirrors upstream `AtomSphAverageRHF.get_occ` (no ECP core for STO-3G).
///
/// MO ordering: the angular-averaged eig produces, per l, `nsh` MOs each
/// repeated `degen` times (`numpy.repeat(e, degen)`). So the occupation vector
/// has, per l, for each of the `nsh` radial functions in energy order: the
/// first `ndocc` get 2.0, the `ndocc`-th gets `frac`, the rest 0.0; each
/// repeated `degen` times.
fn spherical_occ(atm: &Mole, z: usize, groups: &AngularGroups) -> Result<Vec<f64>, PyscfRsError> {
    let nao = atm.nao_nr;
    let mut occ = Vec::with_capacity(nao);
    for &(l, degen, ref aos) in &groups.groups {
        let nsh = aos.len() / degen;
        let (ndocc, frac) = crate::atom_config::frac_occ(z, l);
        for radial in 0..nsh {
            let occ_radial = if (radial as u32) < ndocc {
                2.0
            } else if radial as u32 == ndocc {
                frac
            } else {
                0.0
            };
            for _ in 0..degen {
                occ.push(occ_radial);
            }
        }
    }
    if occ.len() != nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "atom_hf: spherical_occ produced {} entries but nao={nao}",
            occ.len()
        ))));
    }
    Ok(occ)
}

/// Angular-averaged generalized eig (upstream `AtomSphAverageRHF.eig`).
///
/// For each l-channel: reshape the per-l Fock/overlap to (nsh, degen, nsh,
/// degen), average over the diagonal m-index (`einsum('piqi->pq')/degen`),
/// `eigh_gen` the (nsh × nsh) averaged block, then scatter the eigvecs back
/// across the (2l+1) m-components.
///
/// Returns `(mo_energy length nao, mo_coeff F-order nao×nao)` in the
/// per-l-block / scattered-m ordering matching `spherical_occ`.
fn angular_averaged_eig(
    fock: &Density,
    s1e: &Density,
    nao: usize,
    groups: &AngularGroups,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let mut mo_energy = Vec::with_capacity(nao);
    // F-order nao × nao; columns appended in per-l-block order.
    let mut mo_coeff = vec![0.0f64; nao * nao];
    let mut col = 0usize;

    for &(_l, degen, ref aos) in &groups.groups {
        let nsh = aos.len() / degen;
        let degen_f = degen as f64;

        // Build the averaged (nsh × nsh) Fock and overlap blocks (row-major).
        //   f_avg[p,q] = (1/degen) Σ_m f[ aos[p*degen + m], aos[q*degen + m] ]
        //   s_avg[p,q] = (1/degen) Σ_m s[ aos[p*degen + m], aos[q*degen + m] ]
        // The AO layout within a shell is m-major (consecutive m for a given
        // radial function), so radial function p occupies m-slots
        // p*degen .. p*degen+degen within `aos`.
        let mut f_avg = vec![0.0f64; nsh * nsh];
        let mut s_avg = vec![0.0f64; nsh * nsh];
        let mut f_terms = vec![0.0f64; degen];
        let mut s_terms = vec![0.0f64; degen];
        for p in 0..nsh {
            for q in 0..nsh {
                for m in 0..degen {
                    let mu = aos[p * degen + m];
                    let nu = aos[q * degen + m];
                    f_terms[m] = fock.data[mu * nao + nu];
                    s_terms[m] = s1e.data[mu * nao + nu];
                }
                f_avg[p * nsh + q] = pyscf_algebra::oracle_sum(&f_terms) / degen_f;
                s_avg[p * nsh + q] = pyscf_algebra::oracle_sum(&s_terms) / degen_f;
            }
        }

        // Solve the (nsh × nsh) generalized eigenproblem.
        let (eigvals, eigvecs) =
            pyscf_algebra::eigh_gen(&f_avg, &s_avg, nsh).map_err(ScfError::Algebra)?;
        // eigvecs is F-order nsh × nsh: c_sub[p, i] = eigvecs[p + i*nsh].

        // Scatter back: for each radial-MO i (energy order) and each m,
        // the molecular MO column `col` has AO entry at aos[p*degen + m] equal
        // to c_sub[p, i]. Each radial MO produces `degen` molecular columns
        // (one per m), and the energy is repeated `degen` times.
        for i in 0..nsh {
            for m in 0..degen {
                // Column `col` (m-th copy of radial MO i).
                for p in 0..nsh {
                    let mu = aos[p * degen + m];
                    mo_coeff[mu + col * nao] = eigvecs[p + i * nsh];
                }
                mo_energy.push(eigvals[i]);
                col += 1;
            }
        }
    }

    if mo_energy.len() != nao || col != nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "atom_hf: angular_averaged_eig filled {col} columns / {} energies but nao={nao}",
            mo_energy.len()
        ))));
    }

    Ok((mo_energy, mo_coeff))
}

/// Build the atomic density `D[μ,ν] = Σ_p occ[p]·C[μ,p]·C[ν,p]` (row-major).
/// `mo_coeff` is F-order (`C[μ,p] = mo_coeff[μ + p*nao]`). oracle_sum over p.
fn build_atomic_dm(mo_coeff: &[f64], occ: &[f64], nao: usize) -> Result<Density, PyscfRsError> {
    let mut data = vec![0.0f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            let terms: Vec<f64> = (0..nao)
                .filter(|&p| occ[p] != 0.0)
                .map(|p| occ[p] * mo_coeff[mu + p * nao] * mo_coeff[nu + p * nao])
                .collect();
            data[mu * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Ok(Density { nao, data })
}

/// `Tr[D·M] = Σ_{μν} D[μ,ν]·M[ν,μ]` for two row-major symmetric nao×nao
/// matrices (oracle_dot per row). Used for the per-atom electronic energy.
fn trace_dm_times(dm: &Density, m: &Density, nao: usize) -> f64 {
    let mut row_terms = vec![0.0f64; nao];
    let mut diag = vec![0.0f64; nao];
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            row_terms[nu] = dm.data[mu * nao + nu] * m.data[nu * nao + mu];
        }
        diag[mu] = pyscf_algebra::oracle_sum(&row_terms);
    }
    pyscf_algebra::oracle_sum(&diag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_single(sym: &str, basis: &str) -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String(format!("{sym} 0 0 0")),
            basis: BasisInput::Name(basis.to_string()),
            unit: Unit::Bohr,
            ..Default::default()
        })
        .expect("build single-atom Mole")
    }

    #[test]
    fn get_atm_nrhf_h_occ_sums_to_one() {
        // H spherically-averaged occ: 1 s electron → frac_occ(1,0) = (0, 1.0),
        // a single s radial function carries occ 1.0; total occ = 1.0.
        let mol = build_single("H", "sto-3g");
        let res = get_atm_nrhf(&mol).expect("H atom RHF");
        let h = res.get("H").expect("H result present");
        let total: f64 = pyscf_algebra::oracle_sum(&h.mo_occ);
        assert!(
            (total - 1.0).abs() < 1e-12,
            "H spherically-averaged occ should sum to 1.0, got {total}"
        );
        // MO coeff F-order nao×nao, finite.
        assert_eq!(h.mo_coeff.nao, mol.nao_nr);
        assert_eq!(h.mo_coeff.data.len(), mol.nao_nr * mol.nao_nr);
        assert!(h.mo_coeff.data.iter().all(|v| v.is_finite()));
        assert!(h.mo_energy.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn get_atm_nrhf_o_per_l_occ_matches_frac_occ() {
        // O (Z=8): s electrons = 8 → frac_occ(8,0) = (2, 0.0); p electrons = 4
        // → frac_occ(8,1) = (0, 4/3). STO-3G O has 2 s shells (1s,2s) and 1 p
        // shell (3 p AOs). The spherical occ should be:
        //   s block (2 radial fns): 2.0, 2.0
        //   p block (1 radial fn × 3 m): 4/3, 4/3, 4/3
        // total = 4 + 4 = 8 electrons.
        let mol = build_single("O", "sto-3g");
        let res = get_atm_nrhf(&mol).expect("O atom RHF");
        let o = res.get("O").expect("O result present");
        let total: f64 = pyscf_algebra::oracle_sum(&o.mo_occ);
        assert!(
            (total - 8.0).abs() < 1e-12,
            "O spherically-averaged occ should sum to Z=8, got {total}"
        );
        // The s/p occupations must come from frac_occ(8,0)/(8,1).
        let (ndocc_s, frac_s) = crate::atom_config::frac_occ(8, 0);
        let (ndocc_p, frac_p) = crate::atom_config::frac_occ(8, 1);
        assert_eq!((ndocc_s, frac_s), (2, 0.0));
        assert_eq!(ndocc_p, 0);
        assert!((frac_p - 4.0 / 3.0).abs() < 1e-12);
        // At least one occ entry equals the p frac (4/3) and one equals 2.0.
        assert!(o.mo_occ.iter().any(|&v| (v - 2.0).abs() < 1e-12));
        assert!(o.mo_occ.iter().any(|&v| (v - 4.0 / 3.0).abs() < 1e-12));
        assert_eq!(o.mo_coeff.data.len(), mol.nao_nr * mol.nao_nr);
        assert!(o.mo_coeff.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn get_atm_nrhf_caches_unique_elements() {
        // H2 has two H atoms but a single unique element → one entry.
        let mol = M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        })
        .expect("build H2");
        let res = get_atm_nrhf(&mol).expect("H2 atomic RHF");
        assert_eq!(res.len(), 1, "H2 has a single unique element");
        assert!(res.contains_key("H"));
    }
}
