//! Gamma-point unrestricted gradient over `MultiGridNumInt2` — `pbc/grad/uhf.py` (103 l).
//!
//! Same shape as the k-point `18-06` note: the `h1ao` and `s1` terms take
//! spin-summed densities, the `vhf` term stays spin-resolved
//! (`uhf.py:72-75`):
//!
//! ```python
//! de += rhf_grad._contract_vhf_dm(mf_grad, h1ao, dm0_sf) * 2
//! for s in range(2):
//!     de += rhf_grad._contract_vhf_dm(mf_grad, vhf[s], dm0[s]) * 2
//! de += rhf_grad._contract_vhf_dm(mf_grad, s1, dme0_sf) * -2
//! ```
//!
//! `uhf.py:95` is `class Gradients(rhf_grad.GradientsBase)` — the RHF
//! module's base surface (trait [`Gradients`]), not its `Gradients` class.
//! The numint assertion (`uhf.py:38-43`) and the non-gamma refusal
//! (`uhf.py:78-79`) are the same two named errors as the restricted body.
//!
//! `get_veff` (`uhf.py:89-93`) passes `spin=1` with the same `xc_code`
//! convention (`None` for HF). This port's `get_veff_ip1` is a closed-shell
//! primitive, so each spin channel is contracted through it separately —
//! the spin-resolved half of the 18-06 shape. The pseudo branch consumes the
//! spin-summed density (`dm0_sf`, `uhf.py:63-65`); its cached `rhoG` is
//! rebuilt from that sum (`None` — upstream's cache-miss path) rather than
//! summed from per-spin handles, so no density-linearity claim is needed.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_dft::multigrid::pair::MultiGridNumInt2;
use pyscf_pbc_gto::Cell;

use crate::contract::{SCREEN_VHF_DM_CONTRACT, contract_vhf_dm};
use crate::error::PbcGradError;
use crate::gamma_rhf::{GammaCoulombEngine, negated_ip_intor, require_gamma_kpt};
use crate::gradients::{Gradient, Gradients};

fn wrap_dft(context: &str, error: impl std::fmt::Display) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "gamma UHF gradient ({context}): {error}"
    )))
}

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

/// Gamma-point unrestricted (Hartree-Fock or Kohn-Sham) nuclear gradient —
/// `pbc/grad/uhf.py`'s `Gradients` class.
///
/// Densities are `[alpha, beta]` pairs of `nao × nao` symmetric real
/// matrices; `dm_sf = dm[0] + dm[1]` is formed once (`uhf.py:54-55`) and the
/// `h1ao`/`s1` contractions consume the sum.
pub struct GammaUhfGradients<'a> {
    cell: &'a Cell,
    ni: &'a MultiGridNumInt2,
    dm: [Vec<f64>; 2],
    dme: [Vec<f64>; 2],
    kpts: [[f64; 3]; 1],
    xc_code: Option<String>,
    atmlst: Option<Vec<usize>>,
    screened: bool,
    vpploc_part1_precomputed: bool,
}

impl<'a> GammaUhfGradients<'a> {
    /// Resolve the Coulomb engine (`uhf.py:38-43`) and validate both spin
    /// channels. K-point handling mirrors the restricted body.
    pub fn new(
        cell: &'a Cell,
        engine: GammaCoulombEngine<'a>,
        dm: [Vec<f64>; 2],
        dme: [Vec<f64>; 2],
    ) -> Result<Self, PyscfRsError> {
        let ni = engine.resolve()?;
        let nao = cell.mol.nao_nr;
        let size = nao
            .checked_mul(nao)
            .ok_or_else(|| invalid("gamma UHF gradient: AO count overflow"))?;
        for (label, matrices) in [("dm", &dm), ("dme", &dme)] {
            for (s, m) in matrices.iter().enumerate() {
                if m.len() != size {
                    return Err(invalid(format!(
                        "gamma UHF gradient: {label}[{s}] has {} elements, expected nao x nao = {size}",
                        m.len()
                    )));
                }
                if m.iter().any(|x| !x.is_finite()) {
                    return Err(invalid(format!(
                        "gamma UHF gradient: {label}[{s}] must be finite"
                    )));
                }
            }
        }
        Ok(Self {
            cell,
            ni,
            dm,
            dme,
            kpts: [[0.0; 3]],
            xc_code: None,
            atmlst: None,
            screened: SCREEN_VHF_DM_CONTRACT,
            vpploc_part1_precomputed: false,
        })
    }

    /// K-point for the gradient (`uhf.py:30`, default gamma).
    pub fn with_kpt(mut self, kpt: [f64; 3]) -> Self {
        self.kpts = [kpt];
        self
    }

    /// Exchange-correlation code (`getattr(mf, 'xc', None)` at `uhf.py:91`).
    /// `None` is HF.
    pub fn with_xc(mut self, xc: Option<&str>) -> Self {
        self.xc_code = xc.map(str::to_string);
        self
    }

    /// Atom subset (`de = de[atmlst]`, `uhf.py:77`).
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Self {
        self.atmlst = Some(atmlst);
        self
    }

    /// Screening switch for every `_contract_vhf_dm` call. Defaults to
    /// [`SCREEN_VHF_DM_CONTRACT`].
    pub fn with_screened(mut self, screened: bool) -> Self {
        self.screened = screened;
        self
    }

    /// Whether the part-1 local potential was precomputed
    /// (`uhf.py:66-68`, same conditional as the restricted body).
    pub fn with_vpploc_part1_precomputed(mut self, precomputed: bool) -> Self {
        self.vpploc_part1_precomputed = precomputed;
        self
    }

    fn xc_str(&self) -> &str {
        self.xc_code.as_deref().unwrap_or("HF")
    }

    fn has_pseudo(&self) -> bool {
        (0..self.cell.natm).any(|ia| self.cell.atom_pseudo(ia).is_some())
    }

    fn spin_sum(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// `get_ovlp` via the restricted seam (`uhf.py` inherits it from
    /// `GradientsBase`): `-cell.pbc_intor('int1e_ipovlp', kpt)`.
    pub fn overlap_ip(&self) -> Result<Vec<f64>, PyscfRsError> {
        negated_ip_intor(self.cell, &self.kpts, "int1e_ipovlp")
    }

    /// `get_veff` (`uhf.py:89-93`), per spin channel with `spin=1`: each
    /// entry is `-get_veff_ip1(dm[s])`. The minus is here, per channel.
    pub fn veff_ip(&self) -> Result<[Vec<f64>; 2], PyscfRsError> {
        let mut out = [Vec::new(), Vec::new()];
        for (s, dm_s) in self.dm.iter().enumerate() {
            let result = self
                .ni
                .get_veff_ip1(self.cell, self.xc_str(), dm_s, &self.kpts)
                .map_err(|e| wrap_dft("get_veff_ip1", e))?;
            out[s] = result.veff_ip1.into_iter().map(|v| -v).collect();
        }
        Ok(out)
    }

    /// `grad_nuc` via `GradientsBase`: 18-03's `ewald_nuc_grad`.
    pub fn nuclear_gradient(&self) -> Result<Gradient, PyscfRsError> {
        pyscf_pbc_gto::ewald_nuc_grad(self.cell, None, None)
    }

    fn neighbor_list(&self) -> Result<Option<pyscf_pbc_gto::NeighborList>, PyscfRsError> {
        if !self.screened {
            return Ok(None);
        }
        let ls = pyscf_pbc_gto::get_lattice_ls_default(self.cell)?;
        Ok(Some(pyscf_pbc_gto::build_neighbor_list_for_shlpairs(
            self.cell, &ls,
        )?))
    }

    fn contract_planes(&self, planes: &[f64], dm: &[f64]) -> Result<Gradient, PyscfRsError> {
        let nao = self.cell.mol.nao_nr;
        let size = nao * nao;
        if planes.len() != 3 * size || dm.len() != size {
            return Err(invalid(
                "gamma UHF gradient: contraction planes are not (3, nao, nao)",
            ));
        }
        let vhf: [CTensor; 3] = std::array::from_fn(|c| CTensor {
            re: planes[c * size..(c + 1) * size].to_vec(),
            im: vec![0.0; size],
        });
        let dm_tensor = CTensor {
            re: dm.to_vec(),
            im: vec![0.0; size],
        };
        let neighbor = self.neighbor_list()?;
        contract_vhf_dm(self.cell, &vhf, &dm_tensor, neighbor.as_ref())
    }

    /// `grad_elec` (`uhf.py:30-87`), in upstream's order.
    ///
    /// `extra_force` (`uhf.py:81-82`) contributes nothing here, as in the
    /// restricted body.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        // uhf.py:78-79 — the non-gamma refusal, before any work.
        require_gamma_kpt(&self.kpts[0])?;
        let natm = self.cell.natm;
        let nao = self.cell.mol.nao_nr;
        let size = nao * nao;

        let dm_sf = Self::spin_sum(&self.dm[0], &self.dm[1]);
        let dme_sf = Self::spin_sum(&self.dme[0], &self.dme[1]);

        let s1 = self.overlap_ip()?;
        // Spin-resolved veff, one closed-shell primitive call per channel
        // (`spin=1`, uhf.py:93). Raw sign kept for the assembly below.
        let mut vhf_raw = [Vec::new(), Vec::new()];
        for (s, dm_s) in self.dm.iter().enumerate() {
            vhf_raw[s] = self
                .ni
                .get_veff_ip1(self.cell, self.xc_str(), dm_s, &self.kpts)
                .map_err(|e| wrap_dft("get_veff_ip1", e))?
                .veff_ip1;
            debug_assert_eq!(vhf_raw[s].len(), 3 * size);
        }
        let mut h1ao = negated_ip_intor(self.cell, &self.kpts, "int1e_ipkin")?;

        // uhf.py:60-77. Spin-summed densities for `h1ao`/`s1`, resolved
        // `vhf` below.
        let mut de = if self.has_pseudo() {
            let mut de = self
                .ni
                .vpploc_part1_nuc_grad(self.cell, &dm_sf, &self.kpts, None, None)
                .map_err(|e| wrap_dft("vpploc_part1_nuc_grad", e))?;
            let part2 = pyscf_pbc_gto::pseudo::vpploc_part2_nuc_grad(
                self.cell, &dm_sf, &self.kpts,
            )?;
            let nonloc =
                pyscf_pbc_gto::pseudo::vppnl_nuc_grad(self.cell, &dm_sf, &self.kpts)?;
            for ia in 0..natm {
                for c in 0..3 {
                    de[ia][c] = oracle_sum(&[de[ia][c], part2[ia][c], nonloc[ia][c]]);
                }
            }
            // uhf.py:66-68 — the same precompute conditional.
            if !self.vpploc_part1_precomputed {
                let correction = self
                    .ni
                    .get_vpploc_part1_ip1(self.cell, &self.kpts)
                    .map_err(|e| wrap_dft("get_vpploc_part1_ip1", e))?;
                for (h, v) in h1ao.iter_mut().zip(&correction) {
                    *h -= v;
                }
            }
            de
        } else {
            let de = self
                .ni
                .get_nuc_nuc_grad(self.cell, &dm_sf, &self.kpts, None)
                .map_err(|e| wrap_dft("get_nuc_nuc_grad", e))?;
            let nuc_ip1 = self
                .ni
                .get_nuc_ip1(self.cell, &self.kpts)
                .map_err(|e| wrap_dft("get_nuc_ip1", e))?;
            for (h, v) in h1ao.iter_mut().zip(&nuc_ip1) {
                *h -= v;
            }
            de
        };

        // uhf.py:72-75. `h1ao` contracts the spin SUM; then one `vhf[s]`
        // contraction per spin (`vhf[s] = get_veff = -veff_ip1`, so the raw
        // planes negate). Every `* 2` / `* -2` is exact.
        let h_contrib = self.contract_planes(&h1ao, &dm_sf)?;
        for ia in 0..natm {
            for c in 0..3 {
                de[ia][c] = oracle_sum(&[de[ia][c], 2.0 * h_contrib[ia][c]]);
            }
        }
        for (s, raw) in vhf_raw.iter().enumerate() {
            // `vhf[s] = get_veff = -veff_ip1`, so negate the raw planes.
            let neg: Vec<f64> = raw.iter().map(|v| -v).collect();
            let spin_contrib = self.contract_planes(&neg, &self.dm[s])?;
            for ia in 0..natm {
                for c in 0..3 {
                    de[ia][c] = oracle_sum(&[de[ia][c], 2.0 * spin_contrib[ia][c]]);
                }
            }
        }
        let s_contrib = self.contract_planes(&s1, &dme_sf)?;
        for ia in 0..natm {
            for c in 0..3 {
                de[ia][c] = oracle_sum(&[de[ia][c], -2.0 * s_contrib[ia][c]]);
            }
        }

        // uhf.py:77.
        if let Some(atmlst) = &self.atmlst {
            let mut selected = Vec::with_capacity(atmlst.len());
            for &ia in atmlst {
                if ia >= natm {
                    return Err(invalid(format!(
                        "gamma UHF gradient: atmlst id {ia} out of range for {natm} atoms"
                    )));
                }
                selected.push(de[ia]);
            }
            return Ok(selected);
        }
        Ok(de)
    }

    /// `kernel`: electronic plus nuclear parts.
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let elec = self.electronic_gradient()?;
        let nuc = self.nuclear_gradient()?;
        if elec.len() != nuc.len() {
            return Err(invalid(
                "gamma UHF gradient: electronic and nuclear parts disagree on atom count",
            ));
        }
        Ok(elec
            .into_iter()
            .zip(nuc)
            .map(|(e, n)| {
                [
                    oracle_sum(&[e[0], n[0]]),
                    oracle_sum(&[e[1], n[1]]),
                    oracle_sum(&[e[2], n[2]]),
                ]
            })
            .collect())
    }
}

impl<'a> Gradients for GammaUhfGradients<'a> {
    fn cell(&self) -> &Cell {
        self.cell
    }

    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.electronic_gradient()
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.nuclear_gradient()
    }

    fn make_rdm1e(&self) -> Result<pyscf_pbc_scf::types::KDms, PyscfRsError> {
        Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "gamma UHF make_rdm1e from SCF orbitals (pass dme explicitly; see gamma_make_rdm1e)",
        }
        .into())
    }
}

/// Per-spin energy-weighted density at gamma — the
/// `mol_uhf.Gradients.make_rdm1e` line `uhf.py:102` delegates to. Same
/// column-major convention as [`crate::gamma_rhf::gamma_make_rdm1e`],
/// applied per channel.
pub fn gamma_make_rdm1e_uhf(
    mo_coeff_col_major: [&[f64]; 2],
    mo_energy: [&[f64]; 2],
    mo_occ: [&[f64]; 2],
    nao: usize,
) -> Result<[Vec<f64>; 2], PyscfRsError> {
    Ok([
        crate::gamma_rhf::gamma_make_rdm1e(mo_coeff_col_major[0], mo_energy[0], mo_occ[0], nao)?,
        crate::gamma_rhf::gamma_make_rdm1e(mo_coeff_col_major[1], mo_energy[1], mo_occ[1], nao)?,
    ])
}

/// `grad/uks.py:22` — `class Gradients(uhf.Gradients)` with a
/// `pass`-equivalent body (29 l upstream).
///
/// Shipped as a re-export, not a body: the KS unrestricted gamma gradient
/// IS the UHF assembly with an `xc_code` carried through `get_veff`
/// (`uhf.py:91`). (`get_stress` defers to 18-13's `uks_stress`.)
pub type GammaUksGradients<'a> = GammaUhfGradients<'a>;
