//! The density-fitting AUXILIARY cell — `pyscf/pbc/df/incore.py:52-66`
//! (`make_auxcell`) and `pyscf/pbc/df/df.py:64-121` (`make_modrho_basis`).
//! Plan 14-01, Task 3.
//!
//! # Two things happen here, and only the second is subtle
//!
//! 1. **`make_auxcell`** builds a `Cell` on the same atoms with the auxiliary
//!    basis, and then *re-runs the periodic estimators on it* — `rcut`,
//!    `ke_cutoff`, `mesh`. Phase 11's `super_cell` defect is the precedent to
//!    fear: a periodic cell built through the molecular constructor without the
//!    periodic post-pass is silently wrong. [`make_auxcell`] asserts `rcut` came
//!    out finite and positive before returning.
//!
//! 2. **`make_modrho_basis`** renormalises every auxiliary function so its
//!    **monopole** — not its square norm — is `half_sph_norm = sqrt(0.25/pi)`:
//!
//!    ```text
//!    int1_p = gaussian_int(2l + 2, e_p)
//!    s_i    = SUM_p  c_libcint[p][i] * int1_p
//!    c[p][i] <- c_libcint[p][i] * half_sph_norm / s_i
//!    ```
//!
//!    That convention is what makes `gdf_builder::auxbar` and the compensating
//!    charge express as simply as they do (`df.py:98-104`). Getting it wrong is
//!    invisible until plan 14-02's `j2c`.
//!
//! # Why the scale is carried BESIDE the cell, not inside it
//!
//! `pyscf_gto::make_env::normalise_contractions` divides each contraction column
//! by its own norm, so it is **scale-invariant**: no coefficient a caller writes
//! into `_basis` survives a rebuild. Upstream sidesteps this by writing straight
//! into `auxcell._env` after `make_env`, but this port evaluates integrals
//! through a cintx `BasisSet` built from `_basis`, which `_env` does not feed.
//!
//! So [`AuxCell`] does both: it rewrites `_env` — which is what
//! `estimate_rcut` / `estimate_ke_cutoff` / `_extract_pgto_params` read, making
//! those upstream-exact — and it records [`AuxCell::modrho_scale`], one factor
//! per auxiliary AO, for the integral path to apply. This is the same
//! separation `pseudo::vloc_part2` already uses for the `fake_cell_vloc`
//! coefficients (`VlocAux::rescale_from_unit_norm`).

use std::collections::HashMap;

use pyscf_core::{CoreError, ParsedBasis, PyscfRsError};
use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// `half_sph_norm = sqrt(.25/pi)` — `df.py:75`. The monopole every auxiliary
/// function is normalised to.
pub const HALF_SPH_NORM: f64 = 0.28209479177387814; // (0.25/pi).sqrt()

/// An auxiliary cell plus the per-AO factor that turns its square-norm
/// functions into `make_modrho_basis`'s monopole-normalised ones.
#[derive(Debug, Clone)]
pub struct AuxCell {
    /// The auxiliary `Cell`. Its `_env` already carries the modrho
    /// coefficients, so every estimator that reads `_env` is upstream-exact.
    pub cell: Cell,
    /// One factor per auxiliary AO, in AO order. Multiply any integral block
    /// carrying auxiliary index `P` by `modrho_scale[P]`.
    pub modrho_scale: Vec<f64>,
}

impl AuxCell {
    /// Number of auxiliary AOs.
    pub fn naux(&self) -> usize {
        self.cell.mol.nao_nr
    }
    /// Number of auxiliary shells.
    pub fn nbas(&self) -> usize {
        self.cell.mol.nbas
    }
}

/// `gaussian_int(n, alpha) = 0.5 * Gamma((n+1)/2) / alpha^((n+1)/2)`.
///
/// `pyscf_gto::make_env::gaussian_int` is `pub(crate)`, so it is restated here.
/// Phase 13's defect #3 was a half-integer Gamma that stopped one reduction step
/// early; `libm::tgamma` is used for exactly that reason.
pub fn gaussian_int(n: i32, alpha: f64) -> f64 {
    let h = (f64::from(n) + 1.0) * 0.5;
    0.5 * libm::tgamma(h) / alpha.powf(h)
}

/// `make_auxcell(cell, auxbasis)` — `incore.py:52-66`.
///
/// `auxbasis = None` runs `pyscf_df::make_auxbasis`, which is the Psi4 table,
/// then the BSE metadata, then even-tempered Gaussians. A `Some(name)` is a
/// named basis applied to every element.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when the auxiliary basis cannot be resolved,
/// produces no AOs, or leaves `rcut` unset.
pub fn make_auxcell(cell: &Cell, auxbasis: Option<&str>) -> Result<Cell, PyscfRsError> {
    build_aux_cell(cell, resolve_auxbasis(cell, auxbasis)?)
}

/// The per-element auxiliary basis `make_auxcell` would use — the Psi4 table,
/// then the BSE metadata, then even-tempered Gaussians. Exposed so
/// `gdf_builder::fuse_auxcell` can AUGMENT it with the model-charge shells
/// before the cell is built, rather than concatenating two built cells the way
/// upstream's `gto.conc_env` does.
///
/// # Errors
/// As [`make_auxcell`].
pub fn resolve_auxbasis(
    cell: &Cell,
    auxbasis: Option<&str>,
) -> Result<HashMap<String, ParsedBasis>, PyscfRsError> {
    let aux_basis: HashMap<String, ParsedBasis> = match auxbasis {
        Some(name) => {
            let mut m = HashMap::new();
            for (sym, _) in &cell.mol._atom {
                if m.contains_key(sym) {
                    continue;
                }
                let b = pyscf_gto::basis::load_basis(name, sym).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "make_auxcell: auxiliary basis '{name}' does not cover '{sym}': {e}"
                    )))
                })?;
                m.insert(sym.clone(), b);
            }
            m
        }
        None => {
            // `mol.basis` stores the `{:?}` echo of the `BasisInput` the cell
            // was built with (`Name("gth-szv")`), which `strip_name_echo`
            // reduces to the bare name. A basis given as text or as parsed
            // shells has no name and therefore no table entry — which is
            // exactly upstream's `if not isinstance(obs, str): continue`, so it
            // falls through to the even-tempered route.
            let name = pyscf_gto::strip_name_echo(&cell.mol.basis);
            let names: HashMap<String, String> = cell
                .mol
                ._atom
                .iter()
                .map(|(s, _)| (s.clone(), name.clone()))
                .collect();
            pyscf_df::make_auxbasis(
                &cell.mol._atom,
                &names,
                &cell.mol._basis,
                |s| pyscf_gto::format_atom::charge_for_symbol(s).and_then(|z| usize::try_from(z).ok()),
                // `xc = 'HF'` is upstream's default and `is_hybrid_xc('HF')` is true.
                true,
                false,
            )?
        }
    };
    Ok(aux_basis)
}

/// Build a `Cell` on `cell`'s atoms and lattice carrying `aux_basis`.
///
/// # Errors
/// As [`make_auxcell`].
pub fn build_aux_cell(
    cell: &Cell,
    aux_basis: HashMap<String, ParsedBasis>,
) -> Result<Cell, PyscfRsError> {
    let per_element: HashMap<String, BasisInput> = aux_basis
        .into_iter()
        .map(|(k, v)| (k, BasisInput::Parsed(v)))
        .collect();

    // The atoms are already in Bohr (`_atom` invariant), so the lattice must be
    // handed over in Bohr too. NO pseudopotential: the auxiliary basis fits the
    // DENSITY, and giving it `gth-pade` would rewrite its atom charges.
    let auxcell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(cell.mol._atom.clone()),
            basis: BasisInput::PerElement(per_element),
            unit: pyscf_core::Unit::Bohr,
            cart: cell.mol.cart,
            ..Default::default()
        },
        a: ALattice::Matrix(cell.a),
        precision: cell.precision,
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        ..Default::default()
    })?;

    if auxcell.mol.nao_nr == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "make_auxcell: the auxiliary basis produced no AOs".into(),
        )));
    }
    // Phase-11 `super_cell` lesson: a periodic cell whose periodic post-pass did
    // not run is silently wrong, and a zero rcut makes every lattice sum empty.
    let rcut = auxcell.try_rcut()?;
    if !(rcut.is_finite() && rcut > 0.0) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "make_auxcell: auxcell.rcut = {rcut} is not a usable lattice-sum radius"
        ))));
    }
    Ok(auxcell)
}

/// `make_modrho_basis(cell, auxbasis, drop_eta)` — `df.py:64-121`.
///
/// Renormalises the auxiliary functions to unit MONOPOLE (see the module docs),
/// drops primitives with `exponent < drop_eta`, and recomputes `rcut` from the
/// renormalised coefficients.
///
/// # Errors
/// As [`make_auxcell`], plus [`CoreError::InvalidMolecule`] when a shell's
/// monopole comes out zero (which would make the scale infinite).
pub fn make_modrho_basis(
    cell: &Cell,
    auxbasis: Option<&str>,
    drop_eta: Option<f64>,
) -> Result<AuxCell, PyscfRsError> {
    if let Some(eta) = drop_eta {
        // `df.py:88-92` drops diffuse primitives from the auxiliary basis. No
        // caller in Phase 14 sets it (GDF.exp_to_discard defaults to None), and
        // dropping primitives silently changes naux, so refuse rather than
        // ignore (D-PBC-20).
        return Err(PyscfRsError::NotYetImplemented {
            phase: 14,
            what: "make_modrho_basis(drop_eta) — GDF.exp_to_discard is not wired \
                   (df.py:88-92); pass None",
        }
        .tap_eta(eta));
    }

    let auxcell = make_auxcell(cell, auxbasis)?;
    apply_modrho(auxcell, cell.precision)
}

/// Rewrite a built auxiliary cell's `_env` contraction coefficients to the
/// MONOPOLE normalisation and return the per-AO factor the integral path needs.
///
/// Shared by [`make_modrho_basis`] and `gdf_builder::fuse_auxcell` — the fused
/// cell's model-charge shells take exactly the same normalisation as the
/// auxiliary ones (upstream writes `half_sph_norm / gaussian_int(2l+2, eta)`
/// straight into `chgcell._env`, which is this formula for a single-primitive
/// shell), so one pass covers both halves.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when a contraction's monopole is zero.
pub fn apply_modrho(mut auxcell: Cell, precision: f64) -> Result<AuxCell, PyscfRsError> {
    let nbas = auxcell.mol.nbas;
    let mut modrho_scale: Vec<f64> = Vec::with_capacity(auxcell.mol.nao_nr);
    let mut rcut_max = 0.0_f64;

    // PASS 1 — the per-contraction scale, keyed by the `_env` COEFFICIENT
    // POINTER.
    //
    // **libcint deduplicates identical basis blocks.** Two atoms of the same
    // element share one `PTR_COEFF` slot, so a naive shell loop would scale the
    // same `_env` entries once per atom: the second atom would read
    // already-normalised coefficients, compute `scale = 1`, and leave `_env`
    // squared. On diamond that made the auxiliary metric 4495 where upstream
    // says 252 — with atom 0 correct and atom 1 untouched, which is exactly the
    // signature to look for. Keying on the pointer is what makes this
    // idempotent.
    let mut scales: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
    for ib in 0..nbas {
        let l = auxcell.mol._bas[ib * BAS_SLOTS + ANG_OF].max(0);
        let nprim = auxcell.mol._bas[ib * BAS_SLOTS + NPRIM_OF].max(0) as usize;
        let nctr = auxcell.mol._bas[ib * BAS_SLOTS + NCTR_OF].max(0) as usize;
        let pe = auxcell.mol._bas[ib * BAS_SLOTS + PTR_EXP].max(0) as usize;
        let pc = auxcell.mol._bas[ib * BAS_SLOTS + PTR_COEFF].max(0) as usize;
        if scales.contains_key(&pc) {
            continue;
        }
        let es: Vec<f64> = auxcell.mol._env[pe..pe + nprim].to_vec();
        // `int1 = gaussian_int(l*2+2, es)` — the MULTIPOLE, `df.py:97-99`.
        let int1: Vec<f64> = es.iter().map(|&e| gaussian_int(2 * l + 2, e)).collect();
        let mut per_ctr = Vec::with_capacity(nctr);
        for ic in 0..nctr {
            // `_env` holds the block column-major as (nctr, nprim):
            // `_env[pc + ic*nprim + p]`.
            let s: f64 = (0..nprim)
                .map(|p| auxcell.mol._env[pc + ic * nprim + p] * int1[p])
                .sum();
            if s == 0.0 || !s.is_finite() {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "make_modrho_basis: auxiliary shell {ib} contraction {ic} has \
                     monopole {s}; cannot normalise to half_sph_norm"
                ))));
            }
            per_ctr.push(HALF_SPH_NORM / s);
        }
        scales.insert(pc, per_ctr);
    }

    // PASS 2 — apply each scale exactly once, then read the per-AO factor and
    // the per-shell radius off the result.
    for (&pc, per_ctr) in &scales {
        // Find one shell using this block, for its `nprim`.
        let ib = (0..nbas)
            .find(|&i| auxcell.mol._bas[i * BAS_SLOTS + PTR_COEFF].max(0) as usize == pc)
            .ok_or_else(|| {
                PyscfRsError::Core(CoreError::InvalidMolecule(
                    "apply_modrho: coefficient pointer with no owning shell".into(),
                ))
            })?;
        let nprim = auxcell.mol._bas[ib * BAS_SLOTS + NPRIM_OF].max(0) as usize;
        for (ic, scale) in per_ctr.iter().enumerate() {
            for p in 0..nprim {
                auxcell.mol._env[pc + ic * nprim + p] *= scale;
            }
        }
    }

    for ib in 0..nbas {
        let l = auxcell.mol._bas[ib * BAS_SLOTS + ANG_OF].max(0);
        let nprim = auxcell.mol._bas[ib * BAS_SLOTS + NPRIM_OF].max(0) as usize;
        let nctr = auxcell.mol._bas[ib * BAS_SLOTS + NCTR_OF].max(0) as usize;
        let pe = auxcell.mol._bas[ib * BAS_SLOTS + PTR_EXP].max(0) as usize;
        let pc = auxcell.mol._bas[ib * BAS_SLOTS + PTR_COEFF].max(0) as usize;
        let per_ctr = scales.get(&pc).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "apply_modrho: no scale recorded for shell {ib}"
            )))
        })?;
        let ncomp = if auxcell.mol.cart {
            ((l + 1) * (l + 2) / 2) as usize
        } else {
            (2 * l + 1) as usize
        };
        // The cintx basis is built from `_basis`, whose contraction columns are
        // unit-NORM, so the integral path needs this factor explicitly.
        for ic in 0..nctr {
            for _ in 0..ncomp {
                modrho_scale.push(per_ctr[ic]);
            }
        }

        // `r = _estimate_rcut(es, l, abs(cs).max(axis=1), cell.precision)`,
        // maximised over primitives — `df.py:110-112`.
        let es: Vec<f64> = auxcell.mol._env[pe..pe + nprim].to_vec();
        for (p, &e) in es.iter().enumerate() {
            let cmax = (0..nctr).fold(0.0_f64, |m, ic| {
                m.max(auxcell.mol._env[pc + ic * nprim + p].abs())
            });
            let r = pyscf_pbc_gto::estimate_rcut_pgto(e, l, cmax, precision);
            if r > rcut_max {
                rcut_max = r;
            }
        }
    }

    if modrho_scale.len() != auxcell.mol.nao_nr {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "make_modrho_basis: built {} scale factors for {} auxiliary AOs",
            modrho_scale.len(),
            auxcell.mol.nao_nr
        ))));
    }

    // `auxcell.rcut = max(rcut)` — df.py:114.
    auxcell.rcut = rcut_max;
    auxcell._rcut_from_build = false;
    // `make_auxcell` also pins the mesh off the auxiliary ke_cutoff (incore.py:58-65).
    let ke = pyscf_pbc_gto::estimate_ke_cutoff(&auxcell, auxcell.precision);
    auxcell.mesh = auxcell.cutoff_to_mesh(ke)?;
    auxcell._mesh_from_build = false;

    Ok(AuxCell {
        cell: auxcell,
        modrho_scale,
    })
}

/// Tiny helper so the `drop_eta` refusal can still name the value it refused.
trait TapEta {
    fn tap_eta(self, eta: f64) -> Self;
}
impl TapEta for PyscfRsError {
    fn tap_eta(self, eta: f64) -> Self {
        tracing::warn!("make_modrho_basis: refusing drop_eta = {eta}");
        self
    }
}
