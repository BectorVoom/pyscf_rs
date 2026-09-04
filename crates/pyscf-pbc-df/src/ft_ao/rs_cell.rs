//! `ft_ao._RangeSeparatedCell` — `ft_ao.py:253-564`. Plan 17-10 Task 1.
//!
//! # What this is, in one paragraph
//!
//! Every shell of `cell` is split by its own primitives' exponents into up to
//! THREE new shells — STEEP (needs a fine planewave grid), LOCAL (the default
//! bucket) and SMOOTH (converges slowly in real space, cheap in reciprocal
//! space) — using the SAME primitive coefficients, reordered and regrouped but
//! never rescaled. [`RsCell::recontract`] sums the (up to three) decontracted
//! rows for each original AO back together, and because no coefficient is ever
//! renormalised, the sum reproduces the original contracted quantity exactly.
//!
//! # Why this is NOT built through [`Cell::build`]
//!
//! `Cell::build`'s ordinary path re-derives every shell's contraction
//! coefficients from raw (un-normalised) basis-set text via
//! `pyscf_gto::make_env::normalise_contractions`, which rescales EACH shell's
//! contraction column by `1/sqrt(cᵀSc)` computed over THAT shell's OWN
//! primitive set. Route a decontracted (smaller-nprim) shell through that path
//! and the rescale factor changes — the whole point of upstream's `_env`
//! surgery (`ft_ao.py:340-347`, in-place reorder within the ORIGINAL
//! `PTR_EXP`/`PTR_COEFF` window, values untouched) is that it does NOT do
//! this. This port therefore builds `RsCell` by copying the ALREADY-NORMALISED
//! `Shell`s straight out of `cell.mol.basis_set()` (and the matching raw
//! `_bas`/`_env` window) and only ever reorders/slices — mirroring upstream's
//! `_env`-splice at the level this port's dual (raw-array + cintx `BasisSet`)
//! representation requires. See `pyscf-pbc-df/src/incore/auxcell.rs`'s module
//! docs for the precedent (`AuxCell::modrho_scale`) — same class of problem,
//! different fix, because THIS transform needs no rescale at all.

use std::sync::Arc;

use cintx_core::{Atom as CintxAtom, BasisSet, Shell as CintxShell};
use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;

/// `ft_ao.STEEP_BASIS`.
pub const STEEP_BASIS: i32 = 0;
/// `ft_ao.LOCAL_BASIS`.
pub const LOCAL_BASIS: i32 = 1;
/// `ft_ao.SMOOTH_BASIS`.
pub const SMOOTH_BASIS: i32 = 2;

/// `ft_ao.RCUT_THRESHOLD` — `pbc_scf_rsjk_rcut_threshold`, default `1.0`.
pub const RCUT_THRESHOLD: f64 = 1.0;
/// `ft_ao.KECUT_THRESHOLD` — `pbc_scf_rsjk_kecut_threshold`, default `10.0`.
pub const KECUT_THRESHOLD: f64 = 10.0;

/// `ft_ao._RangeSeparatedCell` — a `Cell` with a partially de-contracted basis.
///
/// `Deref`s to [`Cell`] (D-PBC-01's own convention, one level deeper).
#[derive(Debug, Clone)]
pub struct RsCell {
    /// The de-contracted cell itself.
    pub cell: Cell,
    /// `ref_cell` — the ORIGINAL cell this was decontracted from.
    pub ref_cell: Cell,
    /// For each shell of `self.cell`, the shell id in `ref_cell`.
    pub bas_map: Vec<i32>,
    /// `STEEP_BASIS` / `LOCAL_BASIS` / `SMOOTH_BASIS` per shell of `self.cell`.
    pub bas_type: Vec<i32>,
    /// `sh_loc[ib]..sh_loc[ib+1]` — the shell range of `self.cell` that
    /// `ref_cell` shell `ib` decontracted into. `ref_cell.nbas + 1` entries.
    pub sh_loc: Vec<i32>,
}

impl std::ops::Deref for RsCell {
    type Target = Cell;
    fn deref(&self) -> &Cell {
        &self.cell
    }
}

/// One (exponent, per-contraction coefficient row) primitive record, IN THE
/// SAME layout the cintx `Shell` stores (`coefficients[prim*nctr + ctr]`) —
/// carried as owned rows so a primitive can be freely reordered.
#[derive(Debug, Clone)]
struct Primitive {
    exp: f64,
    /// One coefficient per contraction (`nctr` entries).
    coeffs: Vec<f64>,
}

fn core_err(msg: impl Into<String>) -> PbcDfError {
    PbcDfError::Core(PyscfRsError::Core(CoreError::InvalidMolecule(msg.into())))
}

impl RsCell {
    /// The `ke_cut_threshold = None` trivial wrap, infallible (no shell ever
    /// splits) -- used by callers that need an [`RsCell`] shape (e.g.
    /// [`crate::gdf_builder::eta::estimate_rcut`]) without ever decontracting.
    pub fn trivial_wrap(cell: &Cell) -> RsCell {
        let nbas = cell.mol.nbas;
        RsCell {
            cell: cell.clone(),
            ref_cell: cell.clone(),
            bas_map: (0..nbas as i32).collect(),
            bas_type: vec![LOCAL_BASIS; nbas],
            sh_loc: (0..=nbas as i32).collect(),
        }
    }

    /// `_RangeSeparatedCell.from_cell(cell, ke_cut_threshold, rcut_threshold,
    /// in_rsjk)` — `ft_ao.py:266-399`.
    ///
    /// `ke_cut_threshold = None` is the upstream short-circuit (`:275-280`):
    /// every shell becomes its own trivial one-shell "group", tagged
    /// `LOCAL_BASIS`.
    ///
    /// # Errors
    /// [`PbcDfError`] if `cell` has no built `basis_set`, or a shell/env index
    /// is inconsistent (should not happen on a `Cell` that passed
    /// [`pyscf_pbc_gto::Cell::build`]).
    pub fn from_cell(
        cell: &Cell,
        ke_cut_threshold: Option<f64>,
        rcut_threshold: Option<f64>,
        in_rsjk: bool,
    ) -> Result<RsCell, PbcDfError> {
        let nbas = cell.mol.nbas;

        let Some(ke_cut_threshold) = ke_cut_threshold else {
            return Ok(RsCell {
                cell: cell.clone(),
                ref_cell: cell.clone(),
                bas_map: (0..nbas as i32).collect(),
                bas_type: vec![LOCAL_BASIS; nbas],
                sh_loc: (0..=nbas as i32).collect(),
            });
        };

        let bset = cell
            .mol
            .basis_set()
            .ok_or_else(|| core_err("_RangeSeparatedCell::from_cell: cell has no basis_set"))?
            .clone();
        let shells = bset.shells();
        if shells.len() != nbas {
            return Err(core_err(format!(
                "_RangeSeparatedCell::from_cell: basis_set has {} shells, cell.nbas = {nbas}",
                shells.len()
            )));
        }

        let precision = cell.precision;
        let vol = cell.vol();
        let r0_cell = cell.try_rcut().map_err(PbcDfError::from)?;

        let mut env = cell.mol._env.clone();
        let mut decontracted_bas: Vec<i32> = Vec::new();
        let mut decontracted_shells: Vec<Arc<CintxShell>> = Vec::new();
        let mut bas_type: Vec<i32> = Vec::new();
        let mut bas_map: Vec<i32> = Vec::new();
        let mut sh_loc: Vec<i32> = vec![0];

        for (ib, shell) in shells.iter().enumerate() {
            let row = &cell.mol._bas[ib * BAS_SLOTS..ib * BAS_SLOTS + BAS_SLOTS];
            let nprim = row[NPRIM_OF].max(0) as usize;
            let nctr = row[NCTR_OF].max(0) as usize;
            let l = row[ANG_OF].max(0);
            let pexp = row[PTR_EXP].max(0) as usize;
            let pcoeff = row[PTR_COEFF].max(0) as usize;
            debug_assert_eq!(shell.nprim as usize, nprim);
            debug_assert_eq!(shell.nctr as usize, nctr);

            // `es_idx = es.argsort()[::-1]` — stable ascending sort, then
            // reverse. Matches numpy's tie-break exactly (see module docs).
            let mut prims: Vec<Primitive> = (0..nprim)
                .map(|p| Primitive {
                    exp: shell.exponents[p],
                    coeffs: (0..nctr)
                        .map(|c| shell.coefficients[p * nctr + c])
                        .collect(),
                })
                .collect();
            let mut idx: Vec<usize> = (0..nprim).collect();
            idx.sort_by(|&a, &b| prims[a].exp.partial_cmp(&prims[b].exp).expect("finite exp"));
            idx.reverse();
            prims = idx.into_iter().map(|i| prims[i].clone()).collect();

            // `abs_cs = abs(cs).max(axis=1)`.
            let abs_cs: Vec<f64> = prims
                .iter()
                .map(|p| p.coeffs.iter().fold(0.0_f64, |m, &c| m.max(c.abs())))
                .collect();
            let es: Vec<f64> = prims.iter().map(|p| p.exp).collect();

            // `ke = aft._estimate_ke_cutoff(...)` / `cell._estimate_ke_cutoff(...)`.
            let ke: Vec<f64> = (0..nprim)
                .map(|k| {
                    if in_rsjk {
                        crate::rsdf_builder::omega::estimate_ke_cutoff_pgto_4c(
                            es[k], l, abs_cs[k], precision, 0.0,
                        )
                    } else {
                        pyscf_pbc_gto::cutoff::estimate_ke_cutoff_pgto(
                            es[k], l, abs_cs[k], precision, 0.0,
                        )
                    }
                })
                .collect();

            let smooth_mask: Vec<bool> = ke.iter().map(|&k| k < ke_cut_threshold).collect();
            let (steep_mask, local_mask) = match rcut_threshold {
                None => {
                    let local: Vec<bool> = smooth_mask.iter().map(|&s| !s).collect();
                    (vec![false; nprim], local)
                }
                Some(rcut_thr) => {
                    let norm_ang = ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI)).sqrt();
                    let mut steep = vec![false; nprim];
                    let mut local = vec![false; nprim];
                    for k in 0..nprim {
                        if smooth_mask[k] {
                            continue;
                        }
                        let fac = 2.0 * std::f64::consts::PI * abs_cs[k] / vol * norm_ang
                            / es[k]
                            / precision;
                        let mut r0 = r0_cell;
                        for _ in 0..2 {
                            r0 = ((fac * r0.powf(l as f64 + 1.0) + 1.0).ln() / es[k]).sqrt();
                        }
                        steep[k] = r0 < rcut_thr;
                        local[k] = !steep[k];
                    }
                    (steep, local)
                }
            };

            // Partition, preserving the sorted-descending relative order
            // within each group (upstream's boolean masking does the same).
            let mut groups: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for k in 0..nprim {
                if steep_mask[k] {
                    groups[STEEP_BASIS as usize].push(k);
                } else if smooth_mask[k] {
                    groups[SMOOTH_BASIS as usize].push(k);
                } else {
                    debug_assert!(local_mask[k]);
                    groups[LOCAL_BASIS as usize].push(k);
                }
            }

            // Overwrite `env[pexp..]` / `env[pcoeff..]` IN PLACE, steep then
            // local then smooth — `ft_ao.py:340-347`. Total size is unchanged;
            // only the primitive order/grouping moves.
            let mut e_offset = 0usize;
            for &(group_idx, group_type) in
                &[(0usize, STEEP_BASIS), (1, LOCAL_BASIS), (2, SMOOTH_BASIS)]
            {
                let members = &groups[group_idx];
                let n = members.len();
                if n == 0 {
                    continue;
                }
                for (local_p, &orig_sorted_p) in members.iter().enumerate() {
                    env[pexp + e_offset + local_p] = prims[orig_sorted_p].exp;
                    for c in 0..nctr {
                        // column-major block: `_env[pcoeff + ic*nprim_group + p]`
                        env[pcoeff + e_offset * nctr + c * n + local_p] =
                            prims[orig_sorted_p].coeffs[c];
                    }
                }

                let mut new_row = row.to_vec();
                new_row[NPRIM_OF] = n as i32;
                new_row[PTR_EXP] = (pexp + e_offset) as i32;
                new_row[PTR_COEFF] = (pcoeff + e_offset * nctr) as i32;
                decontracted_bas.extend_from_slice(&new_row);
                bas_type.push(group_type);
                bas_map.push(ib as i32);

                let exps: Arc<[f64]> = Arc::from(
                    (0..n)
                        .map(|p| env[pexp + e_offset + p])
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                let mut coeffs_flat = vec![0.0_f64; n * nctr];
                for p in 0..n {
                    for c in 0..nctr {
                        coeffs_flat[p * nctr + c] = env[pcoeff + e_offset * nctr + c * n + p];
                    }
                }
                let new_shell = CintxShell::try_new(
                    shell.atom_index,
                    l as u8,
                    n as u16,
                    nctr as u16,
                    shell.kappa,
                    shell.representation,
                    exps,
                    Arc::from(coeffs_flat.into_boxed_slice()),
                )
                .map_err(|e| core_err(format!("RsCell::from_cell: Shell::try_new: {e}")))?;
                decontracted_shells.push(Arc::new(new_shell));

                e_offset += n;
            }
            debug_assert_eq!(e_offset, nprim);
            sh_loc.push(decontracted_bas.len() as i32 / BAS_SLOTS as i32);
        }

        let new_nbas = decontracted_bas.len() / BAS_SLOTS;
        let atoms: Arc<[CintxAtom]> = Arc::from(bset.atoms().to_vec().into_boxed_slice());
        let new_basis_set =
            BasisSet::try_new(atoms, Arc::from(decontracted_shells.into_boxed_slice()))
                .map_err(|e| core_err(format!("RsCell::from_cell: BasisSet::try_new: {e}")))?;

        let mut ao_loc: Vec<i32> = Vec::with_capacity(new_nbas + 1);
        let mut acc = 0i32;
        ao_loc.push(0);
        for ib in 0..new_nbas {
            let row = &decontracted_bas[ib * BAS_SLOTS..ib * BAS_SLOTS + BAS_SLOTS];
            let l = row[ANG_OF];
            let nctr = row[NCTR_OF];
            let dim_per_ctr = if cell.mol.cart {
                (l + 1) * (l + 2) / 2
            } else {
                2 * l + 1
            };
            acc += dim_per_ctr * nctr;
            ao_loc.push(acc);
        }

        let mut out = cell.clone();
        out.mol._bas = decontracted_bas;
        out.mol._env = env;
        out.mol.nbas = new_nbas;
        out.mol.ao_loc_nr = ao_loc;
        out.mol.nao_nr = acc as usize;
        out.mol.basis_set = Some(Arc::new(new_basis_set));
        out.ke_cutoff = Some(ke_cut_threshold);

        Ok(RsCell {
            cell: out,
            ref_cell: cell.clone(),
            bas_map,
            bas_type,
            sh_loc,
        })
    }

    /// `_RangeSeparatedCell._reverse_bas_map(bas_map)` — `ft_ao.py:399-406`.
    ///
    /// For each shell of `ref_cell`, the FIRST shell id of its decontracted
    /// children in `self.cell`, plus a trailing sentinel `= bas_map.len()`.
    /// Upstream's own debug assertion is `reverse_bas_map(bas_map) == sh_loc`;
    /// see `tests/rs_cell.rs`.
    ///
    /// # Panics
    /// If `bas_map` is not `0..=max` with every value present (upstream
    /// asserts the same: `uniq_bas[-1] == len(uniq_bas) - 1`).
    pub fn reverse_bas_map(bas_map: &[i32]) -> Vec<i32> {
        let mut uniq: Vec<i32> = bas_map.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let max = *uniq.last().expect("non-empty bas_map");
        assert_eq!(
            max,
            uniq.len() as i32 - 1,
            "bas_map is not a contiguous 0..n range"
        );
        let mut first_idx = vec![0i32; uniq.len()];
        for (i, &v) in bas_map.iter().enumerate() {
            // first occurrence only — bas_map is grouped by construction, so a
            // simple "earliest wins" scan matches `np.unique(..., return_index=True)`.
            if i == 0 || bas_map[i - 1] != v {
                first_idx[v as usize] = i as i32;
            }
        }
        first_idx.push(bas_map.len() as i32);
        first_idx
    }

    /// `smooth_basis_cell()` — `ft_ao.py:408-423`. A plain [`Cell`] carrying
    /// only the SMOOTH shells, sharing `self.cell`'s `_env`/`_atm` (the
    /// `PTR_EXP`/`PTR_COEFF` offsets remain valid — this is a shell subset,
    /// not a rebuild).
    ///
    /// # Errors
    /// As [`pyscf_pbc_gto::estimate_rcut`] / [`Cell::cutoff_to_mesh`] on the
    /// filtered cell.
    pub fn smooth_basis_cell(&self) -> Result<Cell, PbcDfError> {
        let mut cell_d = self.cell.clone();
        let keep: Vec<usize> = (0..self.bas_type.len())
            .filter(|&i| self.bas_type[i] == SMOOTH_BASIS)
            .collect();

        // `cell_d.nbas == 0` — upstream returns early (`ft_ao.py:415-416`)
        // before touching `ke_cutoff`/`mesh`. `BasisSet::try_new` rejects an
        // empty shell list, so this port must short-circuit BEFORE it too.
        if keep.is_empty() {
            cell_d.mol._bas = Vec::new();
            cell_d.mol.nbas = 0;
            cell_d.mol.ao_loc_nr = vec![0];
            cell_d.mol.nao_nr = 0;
            cell_d.mol.basis_set = None;
            return Ok(cell_d);
        }

        let mut new_bas = Vec::with_capacity(keep.len() * BAS_SLOTS);
        for &i in &keep {
            new_bas
                .extend_from_slice(&self.cell.mol._bas[i * BAS_SLOTS..i * BAS_SLOTS + BAS_SLOTS]);
        }
        let bset = self
            .cell
            .mol
            .basis_set()
            .expect("built RsCell has a basis_set");
        let shells: Vec<Arc<CintxShell>> = keep.iter().map(|&i| bset.shells()[i].clone()).collect();
        let atoms: Arc<[CintxAtom]> = Arc::from(bset.atoms().to_vec().into_boxed_slice());
        let new_bset = BasisSet::try_new(atoms, Arc::from(shells.into_boxed_slice()))
            .map_err(|e| core_err(format!("smooth_basis_cell: BasisSet::try_new: {e}")))?;

        let nbas = keep.len();
        let mut ao_loc = Vec::with_capacity(nbas + 1);
        let mut acc = 0i32;
        ao_loc.push(0);
        for ib in 0..nbas {
            let row = &new_bas[ib * BAS_SLOTS..ib * BAS_SLOTS + BAS_SLOTS];
            let l = row[ANG_OF];
            let nctr = row[NCTR_OF];
            let dim_per_ctr = if cell_d.mol.cart {
                (l + 1) * (l + 2) / 2
            } else {
                2 * l + 1
            };
            acc += dim_per_ctr * nctr;
            ao_loc.push(acc);
        }

        cell_d.mol._bas = new_bas;
        cell_d.mol.nbas = nbas;
        cell_d.mol.ao_loc_nr = ao_loc;
        cell_d.mol.nao_nr = acc as usize;
        cell_d.mol.basis_set = Some(Arc::new(new_bset));

        let ke_cutoff = pyscf_pbc_gto::cutoff::estimate_ke_cutoff(&cell_d, cell_d.precision);
        cell_d.ke_cutoff = Some(ke_cutoff);
        cell_d.mesh = cell_d.cutoff_to_mesh(ke_cutoff).map_err(PbcDfError::from)?;
        Ok(cell_d)
    }

    /// `compact_basis_cell()` — `ft_ao.py:426-434`. An [`RsCell`] carrying
    /// only the STEEP and LOCAL shells (i.e. everything but SMOOTH).
    ///
    /// # Errors
    /// As [`pyscf_pbc_gto::estimate_rcut`].
    pub fn compact_basis_cell(&self) -> Result<RsCell, PbcDfError> {
        let keep: Vec<usize> = (0..self.bas_type.len())
            .filter(|&i| self.bas_type[i] != SMOOTH_BASIS)
            .collect();

        let mut new_bas = Vec::with_capacity(keep.len() * BAS_SLOTS);
        for &i in &keep {
            new_bas
                .extend_from_slice(&self.cell.mol._bas[i * BAS_SLOTS..i * BAS_SLOTS + BAS_SLOTS]);
        }
        let bset = self
            .cell
            .mol
            .basis_set()
            .expect("built RsCell has a basis_set");
        let shells: Vec<Arc<CintxShell>> = keep.iter().map(|&i| bset.shells()[i].clone()).collect();
        let atoms: Arc<[CintxAtom]> = Arc::from(bset.atoms().to_vec().into_boxed_slice());
        let new_bset = BasisSet::try_new(atoms, Arc::from(shells.into_boxed_slice()))
            .map_err(|e| core_err(format!("compact_basis_cell: BasisSet::try_new: {e}")))?;

        let nbas = keep.len();
        let mut ao_loc = Vec::with_capacity(nbas + 1);
        let mut acc = 0i32;
        ao_loc.push(0);
        for ib in 0..nbas {
            let row = &new_bas[ib * BAS_SLOTS..ib * BAS_SLOTS + BAS_SLOTS];
            let l = row[ANG_OF];
            let nctr = row[NCTR_OF];
            let dim_per_ctr = if self.cell.mol.cart {
                (l + 1) * (l + 2) / 2
            } else {
                2 * l + 1
            };
            acc += dim_per_ctr * nctr;
            ao_loc.push(acc);
        }

        let mut cell_c = self.cell.clone();
        cell_c.mol._bas = new_bas;
        cell_c.mol.nbas = nbas;
        cell_c.mol.ao_loc_nr = ao_loc;
        cell_c.mol.nao_nr = acc as usize;
        cell_c.mol.basis_set = Some(Arc::new(new_bset));

        let new_bas_map: Vec<i32> = keep.iter().map(|&i| self.bas_map[i]).collect();
        let new_bas_type: Vec<i32> = keep.iter().map(|&i| self.bas_type[i]).collect();

        // `segs = sh_loc[1:] - sh_loc[:-1]; segs[bas_map[~mask]] -= 1; cumsum`.
        let mut segs: Vec<i32> = self.sh_loc.windows(2).map(|w| w[1] - w[0]).collect();
        for (i, &t) in self.bas_type.iter().enumerate() {
            if t == SMOOTH_BASIS {
                segs[self.bas_map[i] as usize] -= 1;
            }
        }
        let mut new_sh_loc = Vec::with_capacity(segs.len() + 1);
        new_sh_loc.push(0i32);
        let mut acc_seg = 0i32;
        for s in segs {
            acc_seg += s;
            new_sh_loc.push(acc_seg);
        }

        cell_c.rcut =
            pyscf_pbc_gto::estimate_rcut(&cell_c, self.cell.precision).map_err(PbcDfError::from)?;

        Ok(RsCell {
            cell: cell_c,
            ref_cell: self.ref_cell.clone(),
            bas_map: new_bas_map,
            bas_type: new_bas_type,
            sh_loc: new_sh_loc,
        })
    }

    /// The `ref_cell`-AO index each `self.cell` AO scatter-adds into —
    /// `get_ao_indices(bas_map, ref_cell.ao_loc)`.
    fn ao_map(&self) -> Vec<usize> {
        let ref_ao_loc = &self.ref_cell.mol.ao_loc_nr;
        let this_ao_loc = &self.cell.mol.ao_loc_nr;
        let mut out = Vec::with_capacity(self.cell.mol.nao_nr);
        for (ib, &orig) in self.bas_map.iter().enumerate() {
            let n = (this_ao_loc[ib + 1] - this_ao_loc[ib]) as usize;
            let base = ref_ao_loc[orig as usize] as usize;
            out.extend(base..base + n);
        }
        out
    }

    /// `recontract(dim=2)` — `ft_ao.py:477-506`. `a` is `(rs_nao, rs_nao)`
    /// row-major; the return is `(ref_nao, ref_nao)`, scatter-ADDED (upstream
    /// `lib.takebak_2d`, `out[idx,None][:,idy] += a` — see the module docs).
    pub fn recontract2d(&self, a: &[f64]) -> Vec<f64> {
        let map = self.ao_map();
        let rs_nao = self.cell.mol.nao_nr;
        let ref_nao = self.ref_cell.mol.nao_nr;
        debug_assert_eq!(a.len(), rs_nao * rs_nao);
        let mut out = vec![0.0_f64; ref_nao * ref_nao];
        for i in 0..rs_nao {
            let oi = map[i];
            for j in 0..rs_nao {
                out[oi * ref_nao + map[j]] += a[i * rs_nao + j];
            }
        }
        out
    }

    /// `recontract(dim=1)` — `ft_ao.py:459-476`. `a` is `(rs_nao, ngrids)`
    /// row-major; the return is `(ref_nao, ngrids)`, scatter-ADDED along the
    /// AO axis only.
    pub fn recontract1d(&self, a: &[f64], ngrids: usize) -> Vec<f64> {
        let map = self.ao_map();
        let rs_nao = self.cell.mol.nao_nr;
        let ref_nao = self.ref_cell.mol.nao_nr;
        debug_assert_eq!(a.len(), rs_nao * ngrids);
        let mut out = vec![0.0_f64; ref_nao * ngrids];
        for i in 0..rs_nao {
            let oi = map[i];
            for g in 0..ngrids {
                out[oi * ngrids + g] += a[i * ngrids + g];
            }
        }
        out
    }

    /// The `ref_cell` AO indices of every SMOOTH AO of `self.cell`, in
    /// ascending order (upstream's `merge_diffused_block`'s `smooth_ao_idx`,
    /// `ft_ao.py:446-447`). Monotonically increasing, since it is built by
    /// walking `ref_cell`'s shells in order and keeping whichever ones have a
    /// SMOOTH decontracted child.
    pub fn smooth_ao_indices(&self) -> Vec<usize> {
        let ref_ao_loc = &self.ref_cell.mol.ao_loc_nr;
        let mut out = Vec::new();
        for (ib, &t) in self.bas_type.iter().enumerate() {
            if t != SMOOTH_BASIS {
                continue;
            }
            let orig = self.bas_map[ib] as usize;
            let (a0, a1) = (ref_ao_loc[orig] as usize, ref_ao_loc[orig + 1] as usize);
            out.extend(a0..a1);
        }
        out
    }

    /// `get_ao_type()` — `ft_ao.py:514-527`. One `STEEP_BASIS` /
    /// `LOCAL_BASIS` / `SMOOTH_BASIS` tag per AO of `self.cell`.
    pub fn get_ao_type(&self) -> Vec<i32> {
        let ao_loc = &self.cell.mol.ao_loc_nr;
        let mut out = vec![LOCAL_BASIS; self.cell.mol.nao_nr];
        for (ib, &t) in self.bas_type.iter().enumerate() {
            let (a0, a1) = (ao_loc[ib] as usize, ao_loc[ib + 1] as usize);
            for ao in a0..a1 {
                out[ao] = t;
            }
        }
        out
    }
}
