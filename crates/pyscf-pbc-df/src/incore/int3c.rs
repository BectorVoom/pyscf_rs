//! The 3-centre integral `(mu nu | P)` over a DOUBLE lattice sum, and the
//! 2-centre auxiliary metric — `pyscf/pbc/df/incore.py:73-157, 440-489`
//! (plan 14-01, Task 4).
//!
//! # The sum
//!
//! ```text
//! T[ki,kj][mu nu, P] = SUM_{L1,L2} e^{-i ki.L1} e^{i kj.L2} ( mu(L1) nu(L2) | P(0) )
//! ```
//!
//! Both orbital centres are summed independently and **the auxiliary centre never
//! moves** — `pyscf/lib/pbc/fill_ints_screened.c:266-281` shifts `ish` by `iL`
//! and `jsh` by `jL` and leaves `ksh` in the origin cell. That is the same
//! convention `pseudo::vloc_part2` already ports at gamma, and this module is a
//! generalisation of it to k-points, general auxiliary angular momenta and
//! `aosym`.
//!
//! # THE ARGUMENT MUST BE THE FUSED AUXILIARY CELL
//!
//! Against a **charged** auxiliary cell — the plain `make_modrho_basis` output —
//! this lattice sum is only **conditionally convergent**: each auxiliary
//! function carries net charge, its Coulomb interaction with a distant AO pair
//! decays as `1/R`, and the sum over images therefore grows without bound. That
//! is measured, not asserted; the table is in
//! `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/README.md`:
//!
//! ```text
//! double sum, rcut = 9.532  ->  16.110074  15.972144  15.678667  15.270229
//! double sum, rcut = 14.0   ->  34.052358  33.914427  33.620951  33.212513
//! ```
//!
//! Upstream's own `incore.aux_e2` is `double(R)` minus a **P-independent**
//! offset (5.1814 at 9.532, 23.12 at 14.0 — four identical digits across every
//! `P`), i.e. it removes the divergent `G = 0` background-charge piece; a term
//! `prop. S_mu_nu * q_P` is P-independent exactly because every
//! modrho-normalised auxiliary function has the same monopole.
//!
//! With the **compensating charge** in place (`gdf_builder::fuse_auxcell`, plan
//! 14-02) the auxiliary functions are neutral, the `1/R` tail cancels and the
//! sum converges absolutely — upstream's `fuse(j3c)` is bit-identical at
//! `rcut` x1.0, x1.5 and x2.0. **That** is the quantity with a
//! screening-independent value, and it is the one plan 14-02 gates on at 1e-11.
//!
//! What this module gates on instead is the **isolated-cell identity**: on a
//! one-image cell the port matches upstream `incore.aux_e2` to ~1e-15 on every
//! auxiliary component, which pins the algebra, the normalisation and both
//! index orders with no lattice sum in the way.
//!
//! # Screening
//!
//! Two screens, both from upstream:
//!
//! 1. The neighbour-list screen `|R_s + L - R_P| < r_s + r_P` between each
//!    orbital shell and each auxiliary shell, which bounds BOTH image sums.
//! 2. [`estimate_rcut`] (`incore.py:440-480`), the two-pass Newton refinement
//!    that sets the image list radius.
//!
//! plus the conservative Gaussian-product prescreen `vloc_part2` introduced,
//! reused verbatim through [`prescreen_exponent`]. That bound carries an
//! `exp(-theta_abc |P_ab - C|^2)` factor, which is **exact for a neutralised
//! integrand and only approximate for a charged one** — another reason the
//! fused cell is the intended argument.

use cintx_core::{BasisSet as CintxBasisSet, Representation};
use cintx_ops::resolver::Resolver;
use cintx_rs::{EvaluationContext, SessionRequest};
use cintx_runtime::ExecutionOptions;
use pyscf_algebra::CTensor;
use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::pseudo::vloc_part2::prescreen_exponent;
use std::f64::consts::PI;

use super::auxcell::{AuxCell, gaussian_int};

/// Which half of the `(mu, nu)` block the output carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aosym {
    /// The full `nao x nao` block, row-major in `mu`.
    S1,
    /// The lower triangle `mu >= nu`, packed as `mu*(mu+1)/2 + nu`.
    S2,
}

impl Default for Aosym {
    /// `s2` — upstream's `aosym` default for the density-fitting drivers.
    fn default() -> Self {
        Self::S2
    }
}

impl Aosym {
    /// Number of `(mu, nu)` pairs the output carries.
    pub fn nao_pair(self, nao: usize) -> usize {
        match self {
            Aosym::S1 => nao * nao,
            Aosym::S2 => nao * (nao + 1) / 2,
        }
    }
}

/// The Gaussian-product prescreen threshold, matching
/// `pseudo::vloc_part2::PRESCREEN_EPS`. Everything dropped is provably below
/// this bound.
pub const PRESCREEN_EPS: f64 = 1e-14;

/// The live prescreen threshold. Reads `PBCDF_PRESCREEN_EPS` so the tolerance
/// can be swept against the oracle without a rebuild; defaults to
/// [`PRESCREEN_EPS`].
/// The work a worker must be handed for its `EvaluationContext` setup (~0.3 s)
/// to be worth paying. At ~28 us per shell triple this is ~0.6 s of integrals.
const MIN_TRIPLES_PER_THREAD: usize = 20_000;

/// How many worker threads the double lattice sum uses.
///
/// `PYSCF_NUM_THREADS` (upstream's own knob), then
/// `std::thread::available_parallelism`, then 1. Setting it to 1 makes the loop
/// literally serial — useful when bisecting a numerical difference, though the
/// result is identical either way by construction.
fn aux_e2_threads() -> usize {
    if let Ok(v) = std::env::var("PYSCF_NUM_THREADS")
        && let Ok(n) = v.parse::<usize>()
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

fn prescreen_eps() -> f64 {
    std::env::var("PBCDF_PRESCREEN_EPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(PRESCREEN_EPS)
}

/// How many KET images share one cintx `BasisSet` — see
/// `vloc_part2::KET_CHUNK` for the measurement behind the number.
const KET_CHUNK: usize = 16;

/// `estimate_rcut(cell, auxcell, precision)` — `incore.py:440-480`.
///
/// The radius of the 3-centre image list. Returns the MAXIMUM over the
/// per-ket-shell radii upstream returns as an array (`rcut.max()` is what every
/// caller takes).
///
/// **`cs` here is `gto_norm(l, e)`, not `_extract_pgto_params`.** The similarly
/// shaped `gdf_builder.estimate_rcut` (`gdf_builder.py:932`) uses the libcint
/// contraction coefficient instead, and the two functions also differ in the
/// `fac` prefactor and in the `(sfac*r0)` exponent — `l3 - 2` here against
/// `l3 - 1` there. They are NOT interchangeable: plan 14-02 ports the other one
/// separately. Phase 13's defect #2 was exactly this confusion in `ft_ao`.
///
/// Measured targets (`measurements/params.py`): **17.266040957536866** on
/// diamond/`gth-szv`, **9.53235156147295** on He-fcc/`sto-3g`.
pub fn estimate_rcut(cell: &Cell, auxcell: &Cell, precision: Option<f64>) -> f64 {
    let precision = precision.unwrap_or(cell.precision);
    if cell.mol.nbas == 0 || auxcell.mol.nbas == 0 {
        return 0.0;
    }

    // `cell_exps = [e.min() for e in cell.bas_exps()]`, `cs = gto_norm(ls, cell_exps)`.
    let cell_exps: Vec<f64> = (0..cell.mol.nbas)
        .map(|i| {
            pyscf_pbc_gto::cutoff::bas_exp(cell, i)
                .into_iter()
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let ls: Vec<i32> = (0..cell.mol.nbas)
        .map(|i| cell.mol._bas[i * BAS_SLOTS + ANG_OF].max(0))
        .collect();
    let cs: Vec<f64> = ls
        .iter()
        .zip(cell_exps.iter())
        .map(|(&l, &e)| gto_norm(l, e))
        .collect();

    let aux_exps: Vec<f64> = (0..auxcell.mol.nbas)
        .map(|i| {
            pyscf_pbc_gto::cutoff::bas_exp(auxcell, i)
                .into_iter()
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let ai_idx = argmin(&cell_exps);
    let ak_idx = argmin(&aux_exps);
    let ai = cell_exps[ai_idx];
    let ak = aux_exps[ak_idx];
    let li = f64::from(ls[ai_idx]);
    let lk_i = auxcell.mol._bas[ak_idx * BAS_SLOTS + ANG_OF].max(0);
    let ci = cs[ai_idx];
    // `ck` normalises the auxiliary basis so that `int chi_k dr == 1`.
    let ck = 1.0 / (4.0 * PI) / gaussian_int(lk_i + 2, ak);
    let lk = f64::from(lk_i);

    let r_start = cell.try_rcut().unwrap_or(20.0);

    let mut rcut = 0.0_f64;
    for (j, (&aj, &cj)) in cell_exps.iter().zip(cs.iter()).enumerate() {
        let lj = f64::from(ls[j]);
        let aij = ai + aj;
        let lij = li + lj;
        let l3 = lij + lk;
        let theta = 1.0 / (1.0 / aij + 1.0 / ak);
        let norm_ang = ((2.0 * li + 1.0) * (2.0 * lj + 1.0)).sqrt() / (4.0 * PI);
        let c1 = ci * cj * ck * norm_ang;
        let sfac = aij * aj / (aij * aj + ai * theta);
        let fl = 2.0_f64;
        // fac = 2**(li+1) * pi**3.5 * c1 * theta**(l3-1.5) / aij**(lij+1.5) / ak**(lk+1.5)
        let mut fac = 2.0_f64.powf(li + 1.0) * PI.powf(3.5) * c1 * theta.powf(l3 - 1.5);
        fac /= aij.powf(lij + 1.5) * ak.powf(lk + 1.5);
        fac *= (1.0 + ai / aj).powf(lj) * fl / precision;

        // Two fixed-point sweeps from `cell.rcut`, exactly as upstream.
        let step =
            |r0: f64| ((fac * r0 * (sfac * r0).powf(l3 - 2.0) + 1.0).ln() / (sfac * theta)).sqrt();
        let r0 = step(step(r_start));
        let _ = j;
        if r0.is_finite() && r0 > rcut {
            rcut = r0;
        }
    }
    rcut
}

/// `gto_norm(l, alpha) = 1 / sqrt(gaussian_int(2l + 2, 2 alpha))`.
/// `pyscf_gto::make_env::gto_norm` is `pub(crate)`, so it is restated here.
fn gto_norm(l: i32, alpha: f64) -> f64 {
    1.0 / gaussian_int(2 * l + 2, 2.0 * alpha).sqrt()
}

fn argmin(v: &[f64]) -> usize {
    let mut idx = 0usize;
    for (k, x) in v.iter().enumerate().skip(1) {
        if *x < v[idx] {
            idx = k;
        }
    }
    idx
}

/// `_conc_locs(ao_loc1, ao_loc2)` — `incore.py:483-489`. Concatenated AO
/// offsets, with the second block shifted past the first.
pub fn conc_locs(ao_loc1: &[i32], ao_loc2: &[i32]) -> Vec<i32> {
    let shift = ao_loc1.last().copied().unwrap_or(0);
    let mut out = ao_loc1.to_vec();
    out.extend(ao_loc2.iter().skip(1).map(|v| v + shift));
    out
}

/// `fill_2c2e(cell, auxcell, intor, hermi, kpt)` — `incore.py:144-157`.
///
/// The 2-centre auxiliary metric `(P|Q)` with the lattice sum applied on `|Q>`,
/// which is `auxcell.pbc_intor('int2c2e', hermi, kpts)` plus the modrho scale on
/// BOTH indices.
///
/// `omega` is upstream's `with auxcell.with_range_coulomb(omega):` wrapper
/// around the same call — the `_RSGDFBuilder.get_2c2e` half of range-separated
/// fitting. `None` is the full Coulomb metric and is byte-identical to before.
/// See [`pyscf_pbc_gto::pbc_intor::PbcIntorOpts::omega`] for the sign
/// convention and for what the image list does (and does not) do with ω.
///
/// # Errors
/// As [`pyscf_pbc_gto::pbc_intor::pbc_intor`].
pub fn fill_2c2e(
    aux: &AuxCell,
    hermi: i32,
    kpts: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<CTensor>, PyscfRsError> {
    let out = pyscf_pbc_gto::pbc_intor::pbc_intor(
        &aux.cell,
        "int2c2e",
        kpts,
        pyscf_pbc_gto::pbc_intor::PbcIntorOpts {
            hermi,
            // `pbcopt=lib.c_null_ptr()` — incore.py:155-157 explicitly disables
            // the AO-pair prescreen for the metric.
            screen: false,
            omega,
            ..Default::default()
        },
    )?;
    let naux = aux.naux();
    let s = &aux.modrho_scale;
    let mut mats = out.kmats;
    for m in &mut mats {
        for q in 0..naux {
            for p in 0..naux {
                let f = s[p] * s[q];
                let o = p + q * naux;
                m.re[o] *= f;
                m.im[o] *= f;
            }
        }
    }
    Ok(mats)
}

/// The `(ki, kj)` pairs one [`aux_e2`] call evaluates.
#[derive(Debug, Clone, Copy)]
pub struct KptPair {
    /// The bra k-point.
    pub ki: [f64; 3],
    /// The ket k-point.
    pub kj: [f64; 3],
}

/// `aux_e2(cell, auxcell, 'int3c2e', aosym, kptij_lst)` — `incore.py:73-116`.
///
/// Returns one `(nao_pair, naux)` complex tensor per entry of `kptij_lst`,
/// ROW-MAJOR in `(pair, aux)`.
///
/// **Pass the FUSED auxiliary cell.** Against a charged one the lattice sum is
/// only conditionally convergent and the result depends on `rcut` — see the
/// module docs for the measurement.
///
/// `omega` is upstream's `with cell.with_range_coulomb(omega):` around the
/// `int3c2e` call — the `_RSGDFBuilder.outcore_auxe2` half of range-separated
/// fitting, and the reason `_RSGDFBuilder` was blocked until cintx grew
/// `ExecutionOptions::range_omega` (D-PBC-24). `None` is full Coulomb and is
/// byte-identical to before. See
/// [`pyscf_pbc_gto::pbc_intor::PbcIntorOpts::omega`] for the sign convention.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] on a cintx failure, and
/// [`PyscfRsError::NotYetImplemented`] for the branches D-PBC-23 defers.
pub fn aux_e2(
    cell: &Cell,
    aux: &AuxCell,
    aosym: Aosym,
    kptij_lst: &[KptPair],
    rcut: Option<f64>,
    omega: Option<f64>,
) -> Result<Vec<CTensor>, PyscfRsError> {
    aux_e2_intor(cell, aux, "int3c2e", aosym, kptij_lst, rcut, omega)
}

/// [`aux_e2`] for an arbitrary 3-centre family.
///
/// The `int3c1e_r{2,4,6}_origk` members of
/// `pyscf_pbc_gto::pseudo::vloc_part2::PART2_INTORS` are what make this generic
/// worth having: with them, this function **is**
/// `aft._IntPPBuilder.get_pp_loc_part2` — the k-resolved local pseudopotential
/// Phase 13 declined to port and plan 14-03 found blocking every k-point
/// pseudopotential path. See [`crate::pp_int::get_pp_loc_part2_kpts`].
///
/// Unlike `int3c2e`, the `int3c1e*` operators are OVERLAP-like: the auxiliary
/// function carries no long-range Coulomb tail, so the lattice sum converges
/// absolutely and none of `aux_e2`'s compensating-charge caveats apply.
///
/// # Errors
/// As [`aux_e2`].
#[allow(clippy::too_many_arguments)]
pub fn aux_e2_intor(
    cell: &Cell,
    aux: &AuxCell,
    intor: &str,
    aosym: Aosym,
    kptij_lst: &[KptPair],
    rcut: Option<f64>,
    omega: Option<f64>,
) -> Result<Vec<CTensor>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let nbas = cell.mol.nbas;
    let naux = aux.naux();
    let nauxbas = aux.nbas();
    let nao_pair = aosym.nao_pair(nao);
    let nkij = kptij_lst.len();

    let mut out: Vec<CTensor> = (0..nkij).map(|_| CTensor::zeros(nao_pair * naux)).collect();
    if nao == 0 || naux == 0 || nkij == 0 {
        return Ok(out);
    }

    // --- the image list ---
    let rcut = rcut.unwrap_or_else(|| estimate_rcut(cell, &aux.cell, None));
    let ls = pyscf_pbc_gto::lattice::get_lattice_ls(cell, Some(rcut), None, true)?;
    let nimgs = ls.len();

    // --- Bloch phases: e^{-i ki.L1} for the bra, e^{+i kj.L2} for the ket ---
    let coords = cell.mol.atom_coords();
    let aux_coords = aux.cell.mol.atom_coords();
    let shell_atom = |m: &pyscf_core::Mole, s: usize| -> usize {
        use pyscf_core::raw_layout::ATOM_OF;
        m._bas[s * BAS_SLOTS + ATOM_OF] as usize
    };

    let mut ph_bra = vec![(0.0_f64, 0.0_f64); nkij * nimgs];
    let mut ph_ket = vec![(0.0_f64, 0.0_f64); nkij * nimgs];
    for (t, kp) in kptij_lst.iter().enumerate() {
        for (m, l) in ls.iter().enumerate() {
            let a = -(kp.ki[0] * l[0] + kp.ki[1] * l[1] + kp.ki[2] * l[2]);
            let b = kp.kj[0] * l[0] + kp.kj[1] * l[1] + kp.kj[2] * l[2];
            ph_bra[t * nimgs + m] = (a.cos(), a.sin());
            ph_ket[t * nimgs + m] = (b.cos(), b.sin());
        }
    }

    // --- screening: which images each (orbital shell, auxiliary shell) reaches ---
    let cell_rcut = cell.rcut_by_shells(None);
    // The neighbour-list radius is aggregated PER AUXILIARY ATOM, for the same
    // reason the Gaussian prescreen below is: a compact auxiliary function and
    // its diffuse model charge share a centre and are subtracted from each
    // other, so they must see the SAME image list. Screening them by their own
    // radii kept 12.881985 for the compact one and 11.939510 for its partner
    // over different image sets, and their difference — which is the whole
    // quantity — came out 4.35 wrong on a value of 0.94.
    let aux_rcut_shell = aux.cell.rcut_by_shells(None);
    let mut aux_rcut_atom = vec![0.0_f64; aux.cell.mol.natm];
    for k in 0..nauxbas {
        let ia = shell_atom(&aux.cell.mol, k);
        aux_rcut_atom[ia] = aux_rcut_atom[ia].max(aux_rcut_shell[k]);
    }
    let aux_rcut: Vec<f64> = (0..nauxbas)
        .map(|k| aux_rcut_atom[shell_atom(&aux.cell.mol, k)])
        .collect();
    let reach: Vec<Vec<Vec<usize>>> = (0..nauxbas)
        .map(|k| {
            let rp = aux_coords[shell_atom(&aux.cell.mol, k)];
            (0..nbas)
                .map(|s| {
                    let rmax = cell_rcut[s] + aux_rcut[k];
                    let rs = coords[shell_atom(&cell.mol, s)];
                    ls.iter()
                        .enumerate()
                        .filter(|(_, l)| {
                            let d = [
                                rs[0] + l[0] - rp[0],
                                rs[1] + l[1] - rp[1],
                                rs[2] + l[2] - rp[2],
                            ];
                            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < rmax
                        })
                        .map(|(m, _)| m)
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut used: Vec<usize> = reach
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    used.sort_unstable();
    if used.is_empty() {
        return Ok(out);
    }

    // --- cintx setup ---
    let full_name = pyscf_gto::add_suffix(intor, cell.mol.cart);
    let descriptor = Resolver::descriptor_by_symbol(&full_name).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know '{full_name}': {e}"
        )))
    })?;
    let representation = if cell.mol.cart {
        Representation::Cart
    } else {
        Representation::Spheric
    };
    // ω rides in the OPTIONS, not in the basis. That is what makes range
    // separation reachable from here at all: this driver builds its `BasisSet`
    // through `build_image_expanded_with_aux` from `cell.mol._atom`/`_basis`
    // and never materialises an `_env`, so `pyscf-gto`'s `OmegaGuard` trick of
    // writing `mol._env[8]` has nothing to write into. `ExecutionOptions`
    // sidesteps that entirely.
    //
    // It is set BEFORE `query_workspace` (in `eval3c`) and never after: short
    // range doubles the Rys roots, so ω sizes the workspace, and cintx rejects
    // a ω that changes between query and evaluate as backend contract drift.
    //
    // The image list and the Gaussian prescreen below are still the full-range
    // ones. Both are conservative under either branch — short range decays
    // faster than 1/r and long range shares its tail — so this is correct and
    // merely keeps more triples than short range needs. Tightening it is
    // `rsdf_builder::omega::estimate_rcut`'s job, at the caller, through the
    // `rcut` argument.
    let opts = ExecutionOptions {
        range_omega: omega,
        ..ExecutionOptions::default()
    };

    // AO offsets/counts, image-independent, off a probe basis.
    let (probe, probe_nbas, probe_naux) = build_basis(cell, &aux.cell, &ls, &[0])?;
    if probe_nbas != nbas || probe_naux != nauxbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aux_e2: basis layout mismatch (orbital shells {probe_nbas} vs {nbas}, \
             auxiliary {probe_naux} vs {nauxbas})"
        ))));
    }
    let pm = probe.meta();
    let ao_off: Vec<usize> = (0..nbas).map(|s| pm.shell_offset(s).unwrap_or(0)).collect();
    let ao_cnt: Vec<usize> = (0..nbas).map(|s| pm.ao_count(s).unwrap_or(0)).collect();
    let aux_off: Vec<usize> = (0..nauxbas)
        .map(|k| pm.shell_offset(nbas + k).unwrap_or(0).saturating_sub(nao))
        .collect();
    let aux_cnt: Vec<usize> = (0..nauxbas)
        .map(|k| pm.ao_count(nbas + k).unwrap_or(0))
        .collect();
    drop(probe);

    let diffuse: Vec<(f64, f64)> = (0..nbas).map(|s| most_diffuse(cell, s)).collect();

    // --- the prescreen's auxiliary factor, aggregated PER ATOM ---
    //
    // **The screening decision must not depend on which auxiliary shell is
    // being evaluated.** `aux_e2`'s caller is the FUSED cell, whose auxiliary
    // and model-charge functions sit on the SAME centre and are subtracted from
    // each other immediately afterwards (`FusedCell::fuse_rows`). Screening
    // them independently keeps a triple for one and drops it for the other, so
    // the cancellation that makes the compensated tensor converge is broken —
    // measured as a sign-flipped, 5x-too-large `fuse(j3c)` before this was
    // aggregated. Taking the most diffuse exponent and the largest coefficient
    // over every auxiliary shell ON THE ATOM makes the bound identical for a
    // function and its model charge, and is strictly more conservative than the
    // per-shell bound.
    let aux_natm = aux.cell.mol.natm;
    let mut aux_atom_screen: Vec<(f64, f64)> = vec![(f64::INFINITY, 0.0); aux_natm];
    for k in 0..nauxbas {
        let ia = shell_atom(&aux.cell.mol, k);
        let (e, c) = most_diffuse(&aux.cell, k);
        let slot = &mut aux_atom_screen[ia];
        slot.0 = slot.0.min(e);
        slot.1 = slot.1.max(c.abs());
    }
    for slot in &mut aux_atom_screen {
        if !slot.0.is_finite() {
            *slot = (1.0, 0.0);
        }
    }

    // --- the double lattice sum, bra image outer ---
    //
    // # Why this is threaded
    //
    // The sum is `Σ_{L1,L2} (mu(L1) nu(L2) | P(0))` and BOTH image sums are
    // real — pinning the bra to cell 0 was measured wrong by 3% on He-fcc
    // (0.91323 against 0.94247), so the `O(nimgs^2)` shape is intrinsic, not an
    // artefact. Diamond is 429 images, 10 `s2` shell pairs and 42 fused
    // auxiliary shells: **77 million shell triples**, one cintx
    // `SessionRequest` each at ~28 us. Serially that is over half an hour.
    //
    // The bra image is the natural axis to split on: each `mi` contributes an
    // independent additive term, and nothing inside the body writes shared
    // state. Each thread accumulates into its OWN output and the partials are
    // reduced **in chunk order**, so the floating-point summation order does
    // not depend on the thread count (FOUND-06 — the same guarantee
    // `oracle_sum` gives, obtained here by construction rather than by a
    // reduction primitive).
    //
    // `std::thread::scope` rather than rayon: rayon is not a workspace
    // dependency and adding one for a single loop would need a dependency-wall
    // decision it does not earn.
    let eps = prescreen_eps();

    // Each worker needs its OWN `EvaluationContext` (it owns a cubecl executor
    // and a host scratch arena behind a mutex, so sharing one would serialise
    // the loop). Standing one up is not free — measured at ~0.3 s — so a small
    // workload must not be split 16 ways: He-fcc's 91k triples went 2.56 s ->
    // 7.30 s when it was. Size the pool by the work instead.
    let mean_reach = {
        let n: usize = reach.iter().flatten().map(Vec::len).sum();
        let d = (nauxbas * nbas).max(1);
        n / d
    };
    let est_triples = used
        .len()
        .saturating_mul(nauxbas)
        .saturating_mul(aosym.nao_pair(nbas).max(1))
        .saturating_mul(mean_reach.max(1));
    let nthreads = aux_e2_threads()
        .min(used.len().max(1))
        .min((est_triples / MIN_TRIPLES_PER_THREAD).max(1));
    tracing::debug!(
        "aux_e2: {} bra images, ~{est_triples} shell triples, {nthreads} thread(s)",
        used.len()
    );

    let per_bra = |mi: usize,
                   ctx: &EvaluationContext,
                   out: &mut [CTensor],
                   triples: &mut Vec<(usize, usize, usize, usize)>|
     -> Result<(), PyscfRsError> {
        triples.clear();
        for k in 0..nauxbas {
            if aux_cnt[k] == 0 {
                continue;
            }
            let ka = shell_atom(&aux.cell.mol, k);
            let rp = aux_coords[ka];
            let (ck, cck) = aux_atom_screen[ka];
            for ish in 0..nbas {
                if ao_cnt[ish] == 0 || !reach[k][ish].contains(&mi) {
                    continue;
                }
                let ri = coords[shell_atom(&cell.mol, ish)];
                let ai = [ri[0] + ls[mi][0], ri[1] + ls[mi][1], ri[2] + ls[mi][2]];
                let (ea, ca) = diffuse[ish];
                for jsh in 0..nbas {
                    if ao_cnt[jsh] == 0 {
                        continue;
                    }
                    // `aosym = s2` evaluates the ish >= jsh half only.
                    if aosym == Aosym::S2 && ish < jsh {
                        continue;
                    }
                    let rj = coords[shell_atom(&cell.mol, jsh)];
                    let (eb, cb) = diffuse[jsh];
                    let cmax = ca * cb * cck.abs();
                    for &mj in &reach[k][jsh] {
                        let bj = [rj[0] + ls[mj][0], rj[1] + ls[mj][1], rj[2] + ls[mj][2]];
                        if prescreen_exponent(ea, &ai, eb, &bj, ck, &rp) * cmax < eps {
                            continue;
                        }
                        triples.push((mj, ish, jsh, k));
                    }
                }
            }
        }
        if triples.is_empty() {
            return Ok(());
        }
        triples.sort_unstable();

        let mut cursor = 0usize;
        while cursor < triples.len() {
            let mut kets: Vec<usize> = Vec::with_capacity(KET_CHUNK);
            let start = cursor;
            while cursor < triples.len() && kets.len() <= KET_CHUNK {
                let mj = triples[cursor].0;
                if kets.last() != Some(&mj) {
                    if kets.len() == KET_CHUNK {
                        break;
                    }
                    kets.push(mj);
                }
                cursor += 1;
            }
            let mut shifts: Vec<usize> = Vec::with_capacity(1 + kets.len());
            shifts.push(mi);
            shifts.extend_from_slice(&kets);
            let (basis, _, _) = build_basis(cell, &aux.cell, &ls, &shifts)?;
            let aux_shell0 = shifts.len() * nbas;

            for &(mj, ish, jsh, k) in &triples[start..cursor] {
                let p = kets.iter().position(|m| *m == mj).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(
                        "aux_e2: ket image missing from its own chunk".into(),
                    ))
                })?;
                let di = ao_cnt[ish];
                let dj = ao_cnt[jsh];
                let dk = aux_cnt[k];
                let block = eval3c(
                    &basis,
                    ctx,
                    descriptor.id,
                    representation,
                    &opts,
                    ish,
                    (1 + p) * nbas + jsh,
                    aux_shell0 + k,
                    di * dj * dk,
                    &full_name,
                )?;

                let oi = ao_off[ish];
                let oj = ao_off[jsh];
                let oa = aux_off[k];
                for t in 0..kptij_lst.len() {
                    let (br, bi) = ph_bra[t * nimgs + mi];
                    let (kr, ki_) = ph_ket[t * nimgs + mj];
                    // e^{-i ki.L1} * e^{i kj.L2}
                    let pr = br * kr - bi * ki_;
                    let pi = br * ki_ + bi * kr;
                    let mat = &mut out[t];
                    for kk in 0..dk {
                        let sc = aux.modrho_scale[oa + kk];
                        for jj in 0..dj {
                            for ii in 0..di {
                                let mu = oi + ii;
                                let nu = oj + jj;
                                let Some(row) = pair_index(aosym, nao, mu, nu) else {
                                    continue;
                                };
                                let v = block[ii + jj * di + kk * di * dj] * sc;
                                let o = row * naux + (oa + kk);
                                mat.re[o] += pr * v;
                                mat.im[o] += pi * v;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    };

    if nthreads <= 1 {
        let ctx = EvaluationContext::new();
        let mut triples: Vec<(usize, usize, usize, usize)> = Vec::new();
        for &mi in &used {
            per_bra(mi, &ctx, &mut out, &mut triples)?;
        }
        return Ok(out);
    }

    // Deterministic chunking: contiguous, in `used` order, reduced in the same
    // order. Nothing about the result depends on `nthreads`.
    let chunk = used.len().div_ceil(nthreads);
    let partials: Vec<Result<Vec<CTensor>, PyscfRsError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = used
            .chunks(chunk)
            .map(|slice| {
                let per_bra = &per_bra;
                scope.spawn(move || -> Result<Vec<CTensor>, PyscfRsError> {
                    let ctx = EvaluationContext::new();
                    let mut local: Vec<CTensor> =
                        (0..nkij).map(|_| CTensor::zeros(nao_pair * naux)).collect();
                    let mut triples: Vec<(usize, usize, usize, usize)> = Vec::new();
                    for &mi in slice {
                        per_bra(mi, &ctx, &mut local, &mut triples)?;
                    }
                    Ok(local)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                        "aux_e2: a worker thread panicked".into(),
                    )))
                })
            })
            .collect()
    });

    for part in partials {
        let part = part?;
        for (t, m) in part.iter().enumerate() {
            for (o, v) in m.re.iter().enumerate() {
                out[t].re[o] += v;
            }
            for (o, v) in m.im.iter().enumerate() {
                out[t].im[o] += v;
            }
        }
    }
    Ok(out)
}

/// Row index of the `(mu, nu)` pair, or `None` when `s2` discards it.
fn pair_index(aosym: Aosym, nao: usize, mu: usize, nu: usize) -> Option<usize> {
    match aosym {
        Aosym::S1 => Some(mu * nao + nu),
        Aosym::S2 => {
            if mu >= nu {
                Some(mu * (mu + 1) / 2 + nu)
            } else {
                None
            }
        }
    }
}

/// `[cell + Ls[shifts[0]] | cell + Ls[shifts[1]] | ... | auxcell]`.
fn build_basis(
    cell: &Cell,
    auxcell: &Cell,
    ls: &[[f64; 3]],
    shifts: &[usize],
) -> Result<(std::sync::Arc<CintxBasisSet>, usize, usize), PyscfRsError> {
    let translations: Vec<[f64; 3]> = shifts.iter().map(|m| ls[*m]).collect();
    pyscf_gto::build_image_expanded_with_aux(
        &cell.mol._atom,
        &cell.mol._basis,
        cell.mol.cart,
        &translations,
        &auxcell.mol._atom,
        &auxcell.mol._basis,
        auxcell.mol.cart,
    )
}

#[allow(clippy::too_many_arguments)]
fn eval3c(
    basis: &CintxBasisSet,
    ctx: &EvaluationContext,
    operator: cintx_core::OperatorId,
    representation: Representation,
    opts: &ExecutionOptions,
    i: usize,
    j: usize,
    k: usize,
    expected: usize,
    name: &str,
) -> Result<Vec<f64>, PyscfRsError> {
    let shells = basis.shell_tuple_for_indices([i, j, k]).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "shell_tuple_for_indices({i}, {j}, {k}) failed for '{name}': {e}"
        )))
    })?;
    let outcome = SessionRequest::new(operator, representation, basis, shells, opts.clone())
        .query_workspace_in(ctx)
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cintx workspace query failed for '{name}' triple ({i},{j},{k}): {e}"
            )))
        })?
        .evaluate()
        .map_err(|e| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cintx evaluate failed for '{name}' triple ({i},{j},{k}): {e}"
            )))
        })?;
    let v = &outcome.tensor.owned_values;
    if v.len() != expected {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx returned {} elements for '{name}' triple ({i},{j},{k}), expected \
             {expected} (extents {:?})",
            v.len(),
            outcome.tensor.extents,
        ))));
    }
    Ok(v.clone())
}

/// `(exponent, |coefficient|)` of a shell's most DIFFUSE primitive.
fn most_diffuse(cell: &Cell, bas_id: usize) -> (f64, f64) {
    let es = pyscf_pbc_gto::cutoff::bas_exp(cell, bas_id);
    let cs = pyscf_pbc_gto::cutoff::libcint_ctr_coeff_max(cell, bas_id);
    let n = es.len().min(cs.len());
    let mut best = (f64::INFINITY, 0.0);
    for p in 0..n {
        if es[p] < best.0 {
            best = (es[p], cs[p]);
        }
    }
    if best.0.is_finite() { best } else { (1.0, 0.0) }
}
