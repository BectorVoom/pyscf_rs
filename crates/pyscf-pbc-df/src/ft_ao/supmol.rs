//! `ft_ao.ExtendedMole` — `ft_ao.py:565-743`. Plan 17-10 Task 2.
//!
//! # This port does NOT materialise a literal giant `Mole`
//!
//! Upstream builds `ExtendedMole` by literally replicating `rs_cell`'s atoms
//! and shells once per `(image L, bvk K)` translation into one big `gto.Mole`
//! (`ft_ao.py:606-618`, via `pbctools._build_supcell_`), so it can hand the
//! result to a cint driver as an ordinary finite molecule.
//!
//! Every quantity this plan gates — [`ExtendedMole::strip_basis`]'s surviving
//! `(bvk, shell, image)` triples, [`ExtendedMole::get_ovlp_mask`]'s screen — is
//! a function of shell PARAMETERS (exponent, angular momentum, coefficient,
//! taken verbatim from [`super::rs_cell::RsCell`], since a replica is a rigid
//! translation and touches none of them) and GEOMETRY (the replica's atom
//! position, `ref_atom_coord + L + K`). Neither needs a materialised shell
//! list, so this port represents `ExtendedMole` as `(rs_cell, Ls, bvkmesh_Ls,
//! bas_mask)` and computes replica positions on demand.
//!
//! **What this saves**: literally rebuilding a `cintx::BasisSet` of
//! `nimgs · bvk_ncells · rs_cell.nbas` shells (tens of thousands on a real
//! cell) with none of `RsCell`'s renormalisation hazard, since no shell is
//! ever reconstructed at all — every replica reads `rs_cell`'s ALREADY-BUILT
//! `Shell` by index.
//!
//! **What this does not (yet) give**: an actual cint driver over the extended
//! mole — upstream's `_get_jk_sr` (`rsjk.py:267-436`) needs exactly that, and
//! `crates/pyscf-pbc-scf/src/rsjk.rs`'s own module docs name it as `rsjk`'s
//! SECOND, still-open blocker (Task 5 of this plan). Wiring `ExtendedMole` into
//! this port's EXISTING 3-centre driver (`incore::int3c`, which already sums
//! images directly over `estimate_rcut` and — per `14-VERIFICATION` defect (4)
//! — is *more* converged than upstream's `strip_basis`-narrowed sum) is a
//! separate, later decision; see this plan's `17-10-SUMMARY.md`.

use pyscf_pbc_gto::cutoff::{PgtoOp, extract_pgto_params};
use pyscf_pbc_gto::inv3;
use pyscf_pbc_tools::supercell::{image_atom_coords, scale_lattice, super_cell_translations};

use super::rs_cell::{RsCell, SMOOTH_BASIS};
use crate::error::PbcDfError;

fn frac(r: &[f64; 3], inv_a: &[[f64; 3]; 3]) -> [f64; 3] {
    // Row-vector times matrix: upstream's `atom_coords . inv(a)`.
    let mut o = [0.0_f64; 3];
    for (j, oj) in o.iter_mut().enumerate() {
        *oj = r[0] * inv_a[0][j] + r[1] * inv_a[1][j] + r[2] * inv_a[2][j];
    }
    o
}

fn norm(r: &[f64; 3]) -> f64 {
    (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
}

/// `ft_ao.ExtendedMole` — the Born-von-Karman supercell that "mimics
/// periodicity", represented compactly (see module docs).
#[derive(Debug, Clone)]
pub struct ExtendedMole {
    /// The de-contracted cell every replica shell is a rigid copy of.
    pub rs_cell: RsCell,
    /// `bvk_kmesh`.
    pub bvk_kmesh: [usize; 3],
    /// The rcut-screened image list, sorted by ascending norm — `self.Ls`.
    pub ls: Vec<[f64; 3]>,
    /// The `kmesh`-sized Born-von-Karman translations — `self.bvkmesh_Ls`.
    pub bvkmesh_ls: Vec<[f64; 3]>,
    /// `bas_mask[bvk, shell, image]` — `(bvk_ncells, rs_cell.nbas, nimgs)`,
    /// row-major. Whether that replica shell is kept.
    pub bas_mask: Vec<bool>,
    /// `seg_loc` — shell id in the (bvk-cell, rs-cell-shell) grid to segment
    /// id in the flattened (surviving) shell list.
    pub seg_loc: Vec<i32>,
    /// `seg2sh` — segment id to the flattened surviving-shell-list offset
    /// (`seg2sh[n+1] - seg2sh[n]` = image count kept for that segment).
    pub seg2sh: Vec<i32>,
    pub precision: f64,
}

impl ExtendedMole {
    fn nimgs(&self) -> usize {
        self.ls.len()
    }
    fn bvk_ncells(&self) -> usize {
        self.bvkmesh_ls.len()
    }
    fn rs_nbas(&self) -> usize {
        self.rs_cell.cell.mol.nbas
    }
    fn mask_idx(&self, bvk: usize, shell: usize, img: usize) -> usize {
        (bvk * self.rs_nbas() + shell) * self.nimgs() + img
    }

    /// `ExtendedMole.from_cell(cell, kmesh, rcut, verbose)` — `ft_ao.py:594-630`.
    ///
    /// # Errors
    /// Propagates lattice access on `rs_cell`.
    pub fn from_cell(
        rs_cell: &RsCell,
        kmesh: [usize; 3],
        rcut: Option<f64>,
    ) -> Result<ExtendedMole, PbcDfError> {
        let cell = &rs_cell.cell;
        let rcut = match rcut {
            Some(r) => r,
            None => cell.try_rcut().map_err(PbcDfError::from)?,
        };
        let a = cell.lattice_vectors();

        // `bvkcell = super_cell(cell, kmesh, wrap_around=True)` — geometry
        // only (module docs): `super_cell_translations` IS
        // `k2gamma.translation_vectors_for_kmesh` (identical wrap-around +
        // cartesian_prod + lattice-dot algebra), so `bvkmesh_ls` serves both
        // as `self.bvkmesh_Ls` and as `bvkcell`'s own atom-replica geometry.
        let bvkmesh_ls = super_cell_translations(&a, &kmesh, true);
        let bvk_atom_coords = image_atom_coords(&bvkmesh_ls, &cell.mol.atom_coords());
        let a_bvk = scale_lattice(&a, &kmesh);
        let inv_a_bvk = inv3(&a_bvk).map_err(PbcDfError::from)?;
        let scaled_bvk_coords: Vec<[f64; 3]> = bvk_atom_coords
            .iter()
            .map(|r| frac(r, &inv_a_bvk))
            .collect();

        let dim = pyscf_pbc_gto::lattice::lattice_sum_dimension(cell);
        let mut ls = pyscf_pbc_tools::lattice::get_lattice_ls(
            &a_bvk,
            &scaled_bvk_coords,
            &bvk_atom_coords,
            rcut,
            dim,
            true,
        );
        // `Ls[np.linalg.norm(Ls, axis=1).argsort()]` — stable ascending sort.
        ls.sort_by(|p, q| norm(p).partial_cmp(&norm(q)).expect("finite"));

        let bvk_ncells = bvkmesh_ls.len();
        let nimgs = ls.len();
        let rs_nbas = cell.mol.nbas;
        let bas_mask = vec![true; bvk_ncells * rs_nbas * nimgs];

        let (seg_loc, seg2sh) = bas_mask_to_segment(rs_cell, &bas_mask, bvk_ncells, rs_nbas, nimgs);

        Ok(ExtendedMole {
            rs_cell: rs_cell.clone(),
            bvk_kmesh: kmesh,
            ls,
            bvkmesh_ls,
            bas_mask,
            seg_loc,
            seg2sh,
            precision: cell.precision,
        })
    }

    /// The atom-index a given `rs_cell` shell belongs to.
    fn atom_of(&self, shell: usize) -> usize {
        use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
        self.rs_cell.cell.mol._bas[shell * BAS_SLOTS + ATOM_OF] as usize
    }

    /// The Cartesian position of replica `(bvk, image, atom)` — the atom of
    /// `rs_cell.ref_cell`... no: of `rs_cell` itself (`self.atom_coords()` in
    /// upstream is the DECONTRACTED cell's atoms, same positions as
    /// `ref_cell`'s since decontraction never moves an atom).
    fn replica_atom_coord(&self, bvk: usize, image: usize, atom: usize) -> [f64; 3] {
        let r = self.rs_cell.cell.mol.atom_coords()[atom];
        let l = self.ls[image];
        let k = self.bvkmesh_ls[bvk];
        [r[0] + l[0] + k[0], r[1] + l[1] + k[1], r[2] + l[2] + k[2]]
    }

    /// `strip_basis(rcut)` — `ft_ao.py:631-668`. `rcut` has one entry per
    /// `rs_cell` shell (upstream's `estimate_rcut(rs_cell, auxcell, ...)`).
    ///
    /// Drops any replica shell whose atom sits farther than that shell's own
    /// `rcut` from every atom of `rs_cell` (the reference cell), then
    /// recomputes [`ExtendedMole::seg_loc`] / [`ExtendedMole::seg2sh`].
    ///
    /// A no-op (returns `self` unchanged, matching upstream's `dim == 0`
    /// early return) when `rs_cell.dimension == 0`.
    pub fn strip_basis(&mut self, rcut: &[f64]) {
        if self.rs_cell.cell.dimension == 0 {
            return;
        }
        let ref_coords = self.rs_cell.cell.mol.atom_coords();
        let (bvk_ncells, rs_nbas, nimgs) = (self.bvk_ncells(), self.rs_nbas(), self.nimgs());
        for bvk in 0..bvk_ncells {
            for shell in 0..rs_nbas {
                let atom = self.atom_of(shell);
                let r = rcut[shell];
                for img in 0..nimgs {
                    let idx = self.mask_idx(bvk, shell, img);
                    if !self.bas_mask[idx] {
                        continue;
                    }
                    let p = self.replica_atom_coord(bvk, img, atom);
                    let shortest = ref_coords
                        .iter()
                        .map(|q| {
                            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                            norm(&d)
                        })
                        .fold(f64::INFINITY, f64::min);
                    self.bas_mask[idx] = shortest < r;
                }
            }
        }
        let (seg_loc, seg2sh) =
            bas_mask_to_segment(&self.rs_cell, &self.bas_mask, bvk_ncells, rs_nbas, nimgs);
        self.seg_loc = seg_loc;
        self.seg2sh = seg2sh;
    }

    /// `get_ovlp_mask(cutoff)` — `ft_ao.py:669-703`. Returns a flat
    /// `(rs_cell.nbas, n_active)` row-major boolean screen, `n_active` being
    /// the number of `True` entries currently in [`ExtendedMole::bas_mask`]
    /// (upstream's `supmol.nbas`) — in `(bvk, shell, image)` row-major order,
    /// matching `self.bas_mask.ravel()`.
    ///
    /// # Errors
    /// Propagates `rs_cell.vol()` / precision access (infallible today; kept
    /// `Result` for API symmetry with the rest of this module).
    pub fn get_ovlp_mask(&self, cutoff: Option<f64>) -> Result<(Vec<bool>, usize), PbcDfError> {
        let rs = &self.rs_cell;
        let (cell_exps, cell_cs) = extract_pgto_params(&rs.cell, PgtoOp::Min);
        let cell_l: Vec<i32> = {
            use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS};
            (0..rs.cell.mol.nbas)
                .map(|i| rs.cell.mol._bas[i * BAS_SLOTS + ANG_OF].max(0))
                .collect()
        };
        let cell_coords = rs.cell.mol.atom_coords();
        let cell_bas_coords: Vec<[f64; 3]> = (0..rs.cell.mol.nbas)
            .map(|ib| cell_coords[self.atom_of(ib)])
            .collect();

        let cutoff = cutoff.unwrap_or_else(|| {
            let theta_ij = cell_exps.iter().cloned().fold(f64::INFINITY, f64::min) / 2.0;
            let vol = rs.cell.vol();
            let lattice_sum_factor =
                (2.0 * std::f64::consts::PI * rs.cell.rcut / (vol * theta_ij)).max(1.0);
            rs.cell.precision / lattice_sum_factor * 0.1
        });

        // Active (bvk,shell,image) triples, in ravel order.
        let (bvk_ncells, rs_nbas, nimgs) = (self.bvk_ncells(), self.rs_nbas(), self.nimgs());
        let mut active: Vec<(usize, usize, usize)> = Vec::new();
        for bvk in 0..bvk_ncells {
            for shell in 0..rs_nbas {
                for img in 0..nimgs {
                    if self.bas_mask[self.mask_idx(bvk, shell, img)] {
                        active.push((bvk, shell, img));
                    }
                }
            }
        }
        let n_active = active.len();
        let mut out = vec![false; rs_nbas * n_active];
        for i in 0..rs_nbas {
            let ei = cell_exps[i];
            let ci = cell_cs[i];
            let li = cell_l[i] as f64;
            let ri = cell_bas_coords[i];
            let norm_i = ci * ((2.0 * li + 1.0) / (4.0 * std::f64::consts::PI)).sqrt();
            for (col, &(bvk, shell, img)) in active.iter().enumerate() {
                let ej = cell_exps[shell];
                let cj = cell_cs[shell];
                let lj = cell_l[shell] as f64;
                let atom = self.atom_of(shell);
                let rj = self.replica_atom_coord(bvk, img, atom);
                let aij = ei + ej;
                let theta = ei * ej / aij;
                let dr = {
                    let d = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
                    norm(&d)
                };
                let aij1 = 1.0 / aij;
                let aij2 = aij.powf(-0.5);
                let dri = ej * aij1 * dr + aij2;
                let drj = ei * aij1 * dr + aij2;
                let norm_j = cj * ((2.0 * lj + 1.0) / (4.0 * std::f64::consts::PI)).sqrt();
                let fl = 2.0 * std::f64::consts::PI / rs.cell.vol() * dr / theta + 1.0;
                let ovlp = std::f64::consts::PI.powf(1.5)
                    * norm_i
                    * norm_j
                    * (-theta * dr * dr).exp()
                    * dri.powf(li)
                    * drj.powf(lj)
                    * aij1.powf(1.5)
                    * fl;
                out[i * n_active + col] = ovlp > cutoff;
            }
        }
        Ok((out, n_active))
    }

    /// `bas_type_to_indices(type_code)` — `ft_ao.py:730-739`. Indices into the
    /// active-triple (ravel `bas_mask`) list — same order as
    /// [`ExtendedMole::get_ovlp_mask`]'s columns — whose `rs_cell` shell has
    /// the requested `bas_type`.
    pub fn bas_type_to_indices(&self, type_code: i32) -> Vec<usize> {
        let (bvk_ncells, rs_nbas, nimgs) = (self.bvk_ncells(), self.rs_nbas(), self.nimgs());
        let mut out = Vec::new();
        let mut col = 0usize;
        for bvk in 0..bvk_ncells {
            for shell in 0..rs_nbas {
                let is_type = self.rs_cell.bas_type[shell] == type_code;
                for img in 0..nimgs {
                    if self.bas_mask[self.mask_idx(bvk, shell, img)] {
                        if is_type {
                            out.push(col);
                        }
                        col += 1;
                    }
                }
            }
        }
        out
    }

    /// Convenience: [`ExtendedMole::bas_type_to_indices`] for `SMOOTH_BASIS`.
    pub fn smooth_indices(&self) -> Vec<usize> {
        self.bas_type_to_indices(SMOOTH_BASIS)
    }

    /// `self.sh_loc` property (`ft_ao.py:585-588`) -- `seg2sh[seg_loc]`. One
    /// entry per (bvk-cell, ref_cell-shell) pair plus a trailing sentinel,
    /// giving the supmol shell-offset range for every reference shell.
    pub fn sh_loc(&self) -> Vec<i32> {
        self.seg_loc
            .iter()
            .map(|&i| self.seg2sh[i as usize])
            .collect()
    }

    /// `self.bas_map` property (`ft_ao.py:590-593`) -- ravel indices of the
    /// surviving `(bvk, shell, image)` triples, in row-major order.
    pub fn bas_map(&self) -> Vec<i32> {
        self.bas_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i as i32))
            .collect()
    }
}

/// `bas_mask_to_segment(rs_cell, bas_mask, verbose)` — `ft_ao.py:705-728`.
///
/// Two-level map, exactly as upstream: `seg_loc` maps a shell of the
/// **reference** (`ref_cell`) cell, per bvk-cell, to the offset of its first
/// decontracted child in the (bvk, rs-cell-shell) grid -- via `rs_cell.sh_loc`,
/// whose `ref_cell.nbas + 1` entries this indexes by `[:-1]`, exactly as
/// upstream's `rs_cell.sh_loc[:-1]` broadcast (shape `(bvk_ncells, ref_nbas)`)
/// does. `seg2sh` then maps that (bvk, rs-cell-shell) offset to the
/// cumulative surviving-image count -- the flattened supmol shell offset.
fn bas_mask_to_segment(
    rs_cell: &RsCell,
    bas_mask: &[bool],
    bvk_ncells: usize,
    rs_nbas: usize,
    nimgs: usize,
) -> (Vec<i32>, Vec<i32>) {
    // `images_count = count_nonzero(bas_mask, axis=2)` -- shape (bvk_ncells, nbas).
    let mut images_count = vec![0i32; bvk_ncells * rs_nbas];
    for bvk in 0..bvk_ncells {
        for shell in 0..rs_nbas {
            let base = (bvk * rs_nbas + shell) * nimgs;
            images_count[bvk * rs_nbas + shell] =
                bas_mask[base..base + nimgs].iter().filter(|&&b| b).count() as i32;
        }
    }

    // `seg_loc = arange(bvk_ncells)[:,None]*cell_rs_nbas + rs_cell.sh_loc[:-1]`,
    // ravelled, then append `bvk_ncells*cell_rs_nbas` -- `ft_ao.py:719-720`.
    let ref_nbas = rs_cell.sh_loc.len() - 1;
    let mut seg_loc = Vec::with_capacity(bvk_ncells * ref_nbas + 1);
    for bvk in 0..bvk_ncells {
        for ib in 0..ref_nbas {
            seg_loc.push((bvk * rs_nbas) as i32 + rs_cell.sh_loc[ib]);
        }
    }
    seg_loc.push((bvk_ncells * rs_nbas) as i32);

    // `seg2sh = append(0, cumsum(images_count.ravel()))` -- `ft_ao.py:721`.
    let mut seg2sh = Vec::with_capacity(bvk_ncells * rs_nbas + 1);
    seg2sh.push(0i32);
    let mut acc = 0i32;
    for &c in &images_count {
        acc += c;
        seg2sh.push(acc);
    }
    (seg_loc, seg2sh)
}
