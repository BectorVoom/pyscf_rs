//! The compensating charge — `pyscf/pbc/df/gdf_builder.py:729-931`
//! (plan 14-02, Task 4).
//!
//! # What `fuse` does, and why the port builds ONE cell where upstream builds two
//!
//! Upstream makes a separate `chgcell` of smooth Gaussians (one per
//! `(atom, l)`, all sharing the exponent `eta`) and `gto.conc_env`s it onto the
//! auxiliary cell, so the fused AO layout is `[all aux | all model charge]` and
//! `fuse` is a slice split: `Lpq[:naux] -= chgLpq[...]`.
//!
//! This port cannot concatenate two built cells — `Cell::build` goes through the
//! per-element basis map — so it appends the model-charge shells to each
//! element's basis BEFORE building. The fused layout is therefore ATOM-major:
//! `[atom0 aux | atom0 chg | atom1 aux | atom1 chg | ...]`. Two consequences,
//! both handled by [`FusedCell`]:
//!
//! * the auxiliary AOs are no longer contiguous, so [`FusedCell::aux_ao`] maps
//!   auxiliary-cell AO index to fused AO index. **Within that map the order is
//!   still the auxiliary cell's own**, because the auxiliary cell is atom-major
//!   too — so `fuse`'s output is directly indexable by the auxiliary AO index,
//!   exactly as upstream's is;
//! * the model-charge partner of an auxiliary AO is found through
//!   [`FusedCell::partner`] rather than by an offset table.
//!
//! # One normalisation pass covers both halves
//!
//! `make_modchg_basis` writes `half_sph_norm / gaussian_int(2l+2, eta)` straight
//! into `chgcell._env` — which is exactly what
//! [`crate::incore::auxcell::apply_modrho`] computes for a single-primitive
//! shell. So building the fused basis and running the auxiliary normalisation
//! over all of it reproduces upstream's coefficients on both halves, with no
//! second code path.
//!
//! # Measured (`measurements/params.py`)
//!
//! | cell | `fused_cell.nao` / `.nbas` | `auxbar` nnz / norm |
//! |---|---|---|
//! | diamond 2x2x2 | 126 / 42 | 12 / 0.23012787965177506 |
//! | He-fcc 2x2x2 | 32 / 12 | 4 / 0.3187837520926407 |

use std::collections::HashMap;

use pyscf_core::raw_layout::{ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};
use pyscf_core::{CoreError, ParsedBasis, PyscfRsError, ShellSpec};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::incore::auxcell::{HALF_SPH_NORM, apply_modrho, build_aux_cell, gaussian_int, resolve_auxbasis};
use crate::incore::AuxCell;

/// The auxiliary cell fused with its model-charge partner, plus the index maps
/// [`FusedCell::fuse`] needs.
#[derive(Debug, Clone)]
pub struct FusedCell {
    /// The fused cell — auxiliary + model-charge shells, modrho-normalised.
    pub fused: AuxCell,
    /// The plain auxiliary cell, for `naux` and for callers that need it alone.
    pub auxcell: AuxCell,
    /// `aux_ao[a]` — the fused AO index of auxiliary AO `a`.
    pub aux_ao: Vec<usize>,
    /// `partner[a]` — the fused AO index of auxiliary AO `a`'s model charge.
    /// `None` only when the auxiliary shell's `l` has no model-charge shell,
    /// which upstream's `modchg_offset == -1` also allows.
    pub partner: Vec<Option<usize>>,
    /// The model-charge exponent.
    pub eta: f64,
}

impl FusedCell {
    /// Number of auxiliary AOs — upstream's `naux`.
    pub fn naux(&self) -> usize {
        self.auxcell.naux()
    }
    /// Number of fused AOs — upstream's `nauxc`.
    pub fn nauxc(&self) -> usize {
        self.fused.naux()
    }

    /// `fuse(Lpq, axis=0)` — `gdf_builder.py:844-886`, the spherical branch.
    ///
    /// Takes a `(nauxc, ncol)` block over the FUSED auxiliary index and returns
    /// the `(naux, ncol)` block over the plain auxiliary index, with each
    /// auxiliary row reduced by its model-charge partner.
    ///
    /// **This subtraction is what makes the 3-centre lattice sum converge.**
    /// Plan 14-01 measured the raw sum diverging with `rcut` while upstream's
    /// `fuse(j3c)` is bit-identical at `rcut` x1.0/x1.5/x2.0.
    pub fn fuse_rows(&self, lpq: &[f64], ncol: usize) -> Vec<f64> {
        let naux = self.naux();
        let mut out = vec![0.0_f64; naux * ncol];
        for a in 0..naux {
            let src = self.aux_ao[a] * ncol;
            let dst = a * ncol;
            match self.partner[a] {
                None => out[dst..dst + ncol].copy_from_slice(&lpq[src..src + ncol]),
                Some(p) => {
                    let q = p * ncol;
                    for c in 0..ncol {
                        out[dst + c] = lpq[src + c] - lpq[q + c];
                    }
                }
            }
        }
        out
    }

    /// [`Self::fuse_rows`] on a planar-complex block.
    pub fn fuse_rows_complex(
        &self,
        re: &[f64],
        im: &[f64],
        ncol: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        (self.fuse_rows(re, ncol), self.fuse_rows(im, ncol))
    }

    /// `fuse(Lpq, axis=1)` — the same reduction on the SECOND index of a
    /// `(nrow, nauxc)` row-major block, giving `(nrow, naux)`.
    pub fn fuse_cols(&self, lpq: &[f64], nrow: usize) -> Vec<f64> {
        let naux = self.naux();
        let nauxc = self.nauxc();
        let mut out = vec![0.0_f64; nrow * naux];
        for r in 0..nrow {
            for a in 0..naux {
                let v = lpq[r * nauxc + self.aux_ao[a]];
                out[r * naux + a] = match self.partner[a] {
                    None => v,
                    Some(p) => v - lpq[r * nauxc + p],
                };
            }
        }
        out
    }
}

/// `make_modchg_basis(auxcell, smooth_eta)` — `gdf_builder.py:766-799`.
///
/// One single-primitive shell per `(atom, l)` present in the auxiliary basis,
/// all sharing the exponent `eta`. Returned as a per-element shell list so the
/// caller can append it before the cell is built; the normalisation is applied
/// later by [`apply_modrho`], which reproduces upstream's
/// `half_sph_norm / gaussian_int(2l+2, eta)` exactly.
pub fn make_modchg_basis(
    aux_basis: &HashMap<String, ParsedBasis>,
    eta: f64,
) -> HashMap<String, Vec<ShellSpec>> {
    aux_basis
        .iter()
        .map(|(sym, b)| {
            let mut ls: Vec<u8> = b.shells.iter().map(|s| s.l).collect();
            ls.sort_unstable();
            ls.dedup();
            let shells = ls
                .into_iter()
                .map(|l| ShellSpec {
                    l,
                    exponents: vec![eta],
                    coeffs: vec![vec![1.0]],
                })
                .collect();
            (sym.clone(), shells)
        })
        .collect()
}

/// `fuse_auxcell(auxcell, eta)` — `gdf_builder.py:801-886`.
///
/// # Errors
/// [`PbcDfError::Core`] when the auxiliary basis cannot be resolved or built,
/// and [`PyscfRsError::NotYetImplemented`] for a cartesian cell (upstream runs
/// an extra `CINTc2s_ket_sph` contraction there, which this port does not).
pub fn fuse_auxcell(
    cell: &Cell,
    auxbasis: Option<&str>,
    eta: f64,
) -> Result<FusedCell, PbcDfError> {
    if cell.mol.cart {
        return Err(PbcDfError::Core(PyscfRsError::NotYetImplemented {
            phase: 19,
            what: "cartesian fused auxcell — gdf_builder.fuse_auxcell runs an extra \
                   CINTc2s_ket_sph contraction per shell (gdf_builder.py:848-880)",
        }));
    }

    let aux_basis = resolve_auxbasis(cell, auxbasis)?;
    let auxcell = apply_modrho(build_aux_cell(cell, aux_basis.clone())?, cell.precision)?;

    let chg = make_modchg_basis(&aux_basis, eta);
    let mut fused_basis: HashMap<String, ParsedBasis> = HashMap::new();
    // `n_aux_shells[sym]` — how many leading shells of each element are auxiliary.
    let mut n_aux_shells: HashMap<String, usize> = HashMap::new();
    for (sym, b) in &aux_basis {
        let mut shells = b.shells.clone();
        n_aux_shells.insert(sym.clone(), shells.len());
        if let Some(c) = chg.get(sym) {
            shells.extend(c.iter().cloned());
        }
        fused_basis.insert(sym.clone(), ParsedBasis { shells });
    }

    let mut fused = apply_modrho(build_aux_cell(cell, fused_basis)?, cell.precision)?;

    // `fused_cell.rcut = max(auxcell.rcut, chgcell.rcut)` — gdf_builder.py:806.
    // `chgcell.rcut = _estimate_rcut(smooth_eta, l_max, 1., auxcell.precision)`.
    let l_max = aux_basis
        .values()
        .flat_map(|b| b.shells.iter().map(|s| s.l))
        .max()
        .unwrap_or(0);
    let chg_rcut =
        pyscf_pbc_gto::estimate_rcut_pgto(eta, i32::from(l_max), 1.0, auxcell.cell.precision);
    fused.cell.rcut = auxcell.cell.rcut.max(chg_rcut);

    let (aux_ao, partner) = build_index_maps(&auxcell.cell, &fused.cell, &n_aux_shells)?;

    Ok(FusedCell {
        fused,
        auxcell,
        aux_ao,
        partner,
        eta,
    })
}

/// Walk both cells shell by shell and pair each auxiliary AO with its fused
/// twin and its model-charge partner.
///
/// The cells share atoms and ordering, and the fused element basis is
/// `aux_shells ++ chg_shells`, so within one atom the first
/// `n_aux_shells[element]` fused shells ARE the auxiliary ones, in order.
fn build_index_maps(
    auxcell: &Cell,
    fused: &Cell,
    n_aux_shells: &HashMap<String, usize>,
) -> Result<(Vec<usize>, Vec<Option<usize>>), PyscfRsError> {
    let ao_count = |c: &Cell, ib: usize| -> usize {
        let l = c.mol._bas[ib * BAS_SLOTS + ANG_OF].max(0) as usize;
        let nctr = c.mol._bas[ib * BAS_SLOTS + NCTR_OF].max(0) as usize;
        nctr * if c.mol.cart { (l + 1) * (l + 2) / 2 } else { 2 * l + 1 }
    };
    let atom_of = |c: &Cell, ib: usize| c.mol._bas[ib * BAS_SLOTS + ATOM_OF] as usize;
    let ang_of = |c: &Cell, ib: usize| c.mol._bas[ib * BAS_SLOTS + ANG_OF].max(0);

    // Fused AO offset per shell, and the (atom, l) -> first chg AO map.
    let mut fused_off = Vec::with_capacity(fused.mol.nbas);
    let mut off = 0usize;
    for ib in 0..fused.mol.nbas {
        fused_off.push(off);
        off += ao_count(fused, ib);
    }

    // Which fused shells are auxiliary, and which are model charge.
    let mut seen_per_atom: HashMap<usize, usize> = HashMap::new();
    let mut is_aux = vec![false; fused.mol.nbas];
    let mut chg_start: HashMap<(usize, i32), usize> = HashMap::new();
    for ib in 0..fused.mol.nbas {
        let ia = atom_of(fused, ib);
        let sym = &fused.mol._atom[ia].0;
        let n_aux = *n_aux_shells.get(sym).or_else(|| {
            n_aux_shells
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(sym))
                .map(|(_, v)| v)
        }).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "fuse_auxcell: no auxiliary shell count for element '{sym}'"
            )))
        })?;
        let seen = seen_per_atom.entry(ia).or_insert(0);
        if *seen < n_aux {
            is_aux[ib] = true;
        } else {
            chg_start.insert((ia, ang_of(fused, ib)), fused_off[ib]);
        }
        *seen += 1;
    }

    let mut aux_ao = Vec::with_capacity(auxcell.mol.nao_nr);
    let mut partner = Vec::with_capacity(auxcell.mol.nao_nr);
    let fused_aux_shells: Vec<usize> = (0..fused.mol.nbas).filter(|ib| is_aux[*ib]).collect();
    if fused_aux_shells.len() != auxcell.mol.nbas {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "fuse_auxcell: {} auxiliary shells in the fused cell, {} in the auxiliary cell",
            fused_aux_shells.len(),
            auxcell.mol.nbas
        ))));
    }
    for (ai, &fb) in fused_aux_shells.iter().enumerate() {
        let n = ao_count(auxcell, ai);
        if n != ao_count(fused, fb) {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "fuse_auxcell: auxiliary shell {ai} has {n} AOs, its fused twin {fb} has {}",
                ao_count(fused, fb)
            ))));
        }
        let ia = atom_of(fused, fb);
        let l = ang_of(fused, fb);
        let nd = if fused.mol.cart {
            ((l + 1) * (l + 2) / 2) as usize
        } else {
            (2 * l + 1) as usize
        };
        let p0 = chg_start.get(&(ia, l)).copied();
        // `for i0, i1 in lib.prange(aux_loc[i], aux_loc[i+1], nd)` — every
        // CONTRACTION of the shell is reduced by the SAME model-charge block.
        for c in 0..n {
            aux_ao.push(fused_off[fb] + c);
            partner.push(p0.map(|p| p + c % nd));
        }
    }
    Ok((aux_ao, partner))
}

/// `auxbar(fused_cell)` — `gdf_builder.py:729-764`.
///
/// The interaction between the background charge and each auxiliary function:
/// `-pi/vol * SUM_p c_p / e_p` for `l = 0` shells, zero for the rest. It is
/// subtracted from the gamma-point `j3c` (times the periodic overlap), which is
/// how the chargeless-density convention `int (rho - C) V` is imposed.
///
/// Returned over the FUSED index, exactly as upstream — the caller applies
/// [`FusedCell::fuse_rows`] to it.
pub fn auxbar(fused: &Cell) -> Vec<f64> {
    let naux = fused.mol.nao_nr;
    let mut vbar = vec![0.0_f64; naux];
    // `if fused_cell.dimension < 2 or fused_cell.omega < 0: return vbar`
    if fused.dimension < 2 || pyscf_pbc_gto::cutoff::omega(fused) < 0.0 {
        return vbar;
    }

    let mut off = 0usize;
    for ib in 0..fused.mol.nbas {
        let l = fused.mol._bas[ib * BAS_SLOTS + ANG_OF].max(0);
        let nprim = fused.mol._bas[ib * BAS_SLOTS + NPRIM_OF].max(0) as usize;
        let nctr = fused.mol._bas[ib * BAS_SLOTS + NCTR_OF].max(0) as usize;
        let pe = fused.mol._bas[ib * BAS_SLOTS + PTR_EXP].max(0) as usize;
        let pc = fused.mol._bas[ib * BAS_SLOTS + PTR_COEFF].max(0) as usize;
        let ncomp = if fused.mol.cart {
            ((l + 1) * (l + 2) / 2) as usize
        } else {
            (2 * l + 1) as usize
        };

        if l == 0 {
            let es: Vec<f64> = fused.mol._env[pe..pe + nprim].to_vec();
            if nprim == 1 {
                // `vbar[aux_loc[i]] = -1/es[0]` — only the FIRST AO of the shell.
                for ic in 0..nctr {
                    vbar[off + ic * ncomp] = -1.0 / es[0];
                }
            } else {
                // `norms = half_sph_norm / gaussian_int(2, es)`;
                // `cs = einsum('i,ij->ij', 1/norms, _libcint_ctr_coeff(i))`;
                // `vbar[...] = einsum('in,i->n', cs, -1/es)`.
                let norms: Vec<f64> = es
                    .iter()
                    .map(|&e| HALF_SPH_NORM / gaussian_int(2, e))
                    .collect();
                for ic in 0..nctr {
                    let mut acc = 0.0_f64;
                    for p in 0..nprim {
                        let c = fused.mol._env[pc + ic * nprim + p] / norms[p];
                        acc += c * (-1.0 / es[p]);
                    }
                    vbar[off + ic * ncomp] = acc;
                }
            }
        }
        off += nctr * ncomp;
    }

    let f = std::f64::consts::PI / fused.vol();
    for v in &mut vbar {
        *v *= f;
    }
    vbar
}

/// `_compensate_nuccell(cell, eta)` — `gdf_builder.py:918-931`.
///
/// A cell of ONE `s` Gaussian of exponent `eta` per atom, carrying the
/// compensating nuclear charge. Used by plan 14-03's `_CCNucBuilder`; defined
/// here because it shares the model-charge normalisation.
///
/// # Errors
/// As [`build_aux_cell`].
pub fn compensate_nuccell(cell: &Cell, eta: f64) -> Result<AuxCell, PbcDfError> {
    let mut basis: HashMap<String, ParsedBasis> = HashMap::new();
    for (sym, _) in &cell.mol._atom {
        basis.entry(sym.clone()).or_insert_with(|| ParsedBasis {
            shells: vec![ShellSpec {
                l: 0,
                exponents: vec![eta],
                coeffs: vec![vec![1.0]],
            }],
        });
    }
    Ok(apply_modrho(build_aux_cell(cell, basis)?, cell.precision)?)
}
