//! UKS analytical gradient (GRAD-04) — the spin-resolved KS gradient.
//!
//! Port target: `pyscf/grad/uks.py` (the spin-resolved `get_veff`/`get_vxc` +
//! the `Gradients(uhf_grad.Gradients)` class). Structural analog: [`crate::rks`]
//! (the closed-shell sibling within this crate) + [`crate::uhf`] (the spin-
//! resolved variational base).
//!
//! ## The UKS grad_elec decomposition
//!
//! UKS combines the UHF spin-resolved Hellmann-Feynman + Pulay base (the total-
//! density `hcore_deriv·dm0`, the per-spin 2e Pulay, the spin-summed overlap
//! Pulay) with the spin-resolved KS effective potential:
//!
//! ```text
//! vxc_a, vxc_b = get_vxc_uks(ni, mol, grids, xc, (dm_a, dm_b))  # per-spin XC deriv on the grid
//! veff_a = vxc_a + vj − hyb·vk_a                                 # total-density Coulomb + per-spin exchange
//! veff_b = vxc_b + vj − hyb·vk_b
//! ```
//!
//! where `vj` is the total-density `(dm_a+dm_b)` Coulomb. When `grid_response =
//! true` (default OFF, GRAD-04) the per-atom Becke-weight-derivative term is
//! added (the spin-summed energy density). Per **D-04** / Pitfall 5 the
//! `grid_response` term is a grid-weight-derivative, NOT a coupled-perturbed
//! response solve — UKS makes NO such response solve.
//!
//! ## libxc / xcfun discipline (T-07-16, user memory)
//!
//! The XC-derivative path routes through `pyscf-dft`'s NATIVE `xcfun` backend
//! (the per-spin `eval_uks` surface). It NEVER enables the `libxc` feature.
//!
//! ## cintx-availability gating (D-02 + 07-01/07-03 SUMMARY)
//!
//! The variational base contracts the SAME six grad-integral families RHF/UHF
//! need (all MISSING from cintx, no scheduled workstream); the XC-grid +
//! grid_response terms are cintx-INDEPENDENT. The overall `kernel()` routes the
//! gated families to a CLEAN cintx-availability error (never
//! `NotYetImplemented{phase:7}`); the numeric FD arm is `#[ignore]`'d (see
//! `tests/uks_verify_fd.rs`) until the cintx workstream lands.
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every per-spin + grid + per-atom reduction materialises then routes through
//! `oracle_sum`/`oracle_dot` — NEVER a bare `+=`.

use crate::Gradients;
use crate::error::GradError;
use crate::rhf::{aoslice_by_atom, get_ovlp};
use crate::rks::RksReference;
use crate::uhf::{UhfReference, make_rdm1e as uhf_make_rdm1e};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{Density, Mole, PyscfRsError, Unit};
use pyscf_dft::numint::{NumInt, XcType};
use pyscf_dft::xc_backend::DerivOrder;
use pyscf_grids::Grids;

/// The number of Cartesian derivative components every gradient intor carries.
const NCOMP: usize = 3;

/// Snapshot of a converged UKS reference, consumed by [`UksGradients`] (D-09).
///
/// Carries the UHF-shaped spin-resolved α/β reference pair PLUS the DFT `xc`
/// functional name. pyo3-free.
#[derive(Debug, Clone)]
pub struct UksReference {
    /// The spin-resolved UHF-shaped reference (α/β pair + molecule).
    pub scf: UhfReference,
    /// The exchange-correlation functional name. Parsed by the xcfun backend —
    /// NEVER routed through libxc (T-07-16).
    pub xc: String,
}

impl UksReference {
    /// The shared molecule.
    pub fn mol(&self) -> &Mole {
        self.scf.mol()
    }

    /// Build the closed-shell-shaped [`RksReference`] view of the α channel (for
    /// reusing the RKS grid-weight-derivative helper, which is spin-summed).
    fn rks_view(&self) -> RksReference {
        RksReference {
            scf: self.scf.alpha.clone(),
            xc: self.xc.clone(),
        }
    }
}

/// UKS analytical gradient (`pyscf/grad/uks.py` `Gradients`).
///
/// Holds the converged spin-resolved KS reference + `grid_response` (default
/// OFF, GRAD-04) + an optional `atmlst`. Implements the base [`Gradients`]
/// trait: its `grad_elec` is the UHF spin-resolved assembly with the per-spin KS
/// `get_veff`. Makes NO coupled-perturbed response solve (D-04).
#[derive(Debug, Clone)]
pub struct UksGradients {
    /// The converged UKS reference snapshot (α/β pair + xc).
    pub reference: UksReference,
    /// Whether to add the Becke-weight-derivative term (default OFF, GRAD-04).
    /// Fully supported on request; NOT a response solve (D-04 / Pitfall 5).
    pub grid_response: bool,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl UksGradients {
    /// Build a UKS gradient driver over a converged reference. `grid_response`
    /// defaults OFF (the upstream class default, GRAD-04).
    pub fn new(reference: UksReference) -> Self {
        Self {
            reference,
            grid_response: false,
            atmlst: None,
            de: None,
        }
    }

    /// Enable the Becke-weight-derivative term (GRAD-04). OFF by default; a
    /// grid-weight-derivative, NOT a response solve (D-04).
    pub fn with_grid_response(mut self, on: bool) -> Self {
        self.grid_response = on;
        self
    }

    /// Restrict the gradient to a subset of atoms (GRAD-08).
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Self {
        self.atmlst = Some(atmlst);
        self
    }

    /// The per-atom `extra_force` term (grid-weight-derivative). Zero unless
    /// `grid_response` is set (the upstream default). Cintx-INDEPENDENT.
    pub fn extra_force(&self, ia: usize) -> Result<[f64; 3], PyscfRsError> {
        if !self.grid_response {
            return Ok([0.0; 3]);
        }
        // The spin-summed weight-derivative reuses the RKS helper on the total
        // density (the α-view stands in; the term is spin-summed energy density).
        crate::rks::RksGradients::new(self.reference.rks_view())
            .with_grid_response(true)
            .extra_force(ia)
    }
}

impl Gradients for UksGradients {
    fn mol(&self) -> &Mole {
        self.reference.mol()
    }

    fn atmlst(&self) -> Option<&[usize]> {
        self.atmlst.as_deref()
    }

    fn de(&self) -> Option<&[[f64; 3]]> {
        self.de.as_deref()
    }

    fn unit(&self) -> Unit {
        Unit::Bohr
    }

    /// The spin-summed energy-weighted RDM — reuses the UHF form.
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        uhf_make_rdm1e(&self.reference.scf)
    }

    /// The overlap derivative — inherited (spin-independent), cintx-gated.
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        get_ovlp(self.reference.mol())
    }

    /// PER-METHOD spin-resolved KS electronic gradient. See module docs. Makes
    /// NO response solve.
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(self, atmlst)
    }
}

/// Spin-resolved RDM `dm[μ,ν] = Σ_i occ_i C[μ,i] C[ν,i]` for one spin channel.
fn make_rdm1_spin(refr: &crate::rhf::RhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    if refr.mo_coeff.data.len() != nao * nmo || refr.mo_occ.len() != nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }
    let mut dm = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nmo];
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                terms[i] = refr.mo_occ[i]
                    * refr.mo_coeff.data[mu + i * nao]
                    * refr.mo_coeff.data[nu + i * nao];
            }
            dm[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    Ok(dm)
}

/// The spin-independent core-Hamiltonian gradient `-(int1e_ipkin + int1e_ipnuc)`,
/// component-leading `[3, nao, nao]`. Both intors MISSING from cintx → clean error.
fn get_hcore(mol: &Mole) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let kin = pyscf_gto::intor(mol, "int1e_ipkin")?;
    assert_component_leading(&kin, nao, "int1e_ipkin")?;
    let nuc = pyscf_gto::intor(mol, "int1e_ipnuc")?;
    assert_component_leading(&nuc, nao, "int1e_ipnuc")?;
    let mut h = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in h.iter_mut().enumerate() {
        *slot = -oracle_sum(&[kin.values[idx], nuc.values[idx]]);
    }
    Ok(h)
}

/// Per-atom core-Hamiltonian derivative (rinv-origin shift + h1 block,
/// symmetrised). `int1e_iprinv` + `with_rinv_at_nucleus` MISSING → clean error.
fn hcore_deriv(
    mol: &Mole,
    h1: &[f64],
    aoslices: &[(usize, usize)],
    atm_id: usize,
) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let charges = mol.atom_charges();
    if atm_id >= mol.natm {
        return Err(GradError::ShapeMismatch {
            expected: mol.natm,
            got: atm_id,
        }
        .into());
    }
    let iprinv = pyscf_gto::intor(mol, "int1e_iprinv")?;
    assert_component_leading(&iprinv, nao, "int1e_iprinv")?;
    let z = charges[atm_id] as f64;
    let mut vrinv = vec![0.0_f64; NCOMP * nao * nao];
    for (idx, slot) in vrinv.iter_mut().enumerate() {
        *slot = -z * iprinv.values[idx];
    }
    let (p0, p1) = aoslices[atm_id];
    for comp in 0..NCOMP {
        let base = comp * nao * nao;
        for j in p0..p1 {
            for i in 0..nao {
                let off = base + i + j * nao;
                vrinv[off] = oracle_sum(&[vrinv[off], h1[off]]);
            }
        }
    }
    let mut out = vec![0.0_f64; NCOMP * nao * nao];
    for comp in 0..NCOMP {
        let base = comp * nao * nao;
        for j in 0..nao {
            for i in 0..nao {
                let ij = base + i + j * nao;
                let ji = base + j + i * nao;
                out[ij] = oracle_sum(&[vrinv[ij], vrinv[ji]]);
            }
        }
    }
    Ok(out)
}

/// The spin-resolved KS effective-potential gradient (`uks.py` `get_veff`): the
/// per-spin XC-potential derivative on the grid PLUS the total-density Coulomb
/// and per-spin hybrid-exchange. Returns `(veff_a, veff_b)`, each `[3, nao, nao]`.
fn get_veff(
    refr: &UksReference,
    dm_a: &[f64],
    dm_b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let mol = refr.mol();
    let nao = mol.nao_nr;

    // Per-spin XC-potential derivative on the grid (cintx-independent; xcfun).
    let (vxc_a, vxc_b) = get_vxc_uks(refr, dm_a, dm_b)?; // each [3, nao, nao]

    // Hybrid mixing coefficient (xcfun parser; NO libxc).
    let ni = NumInt::new();
    let hyb = ni.hybrid_coeff(&refr.xc, 1).map_err(PyscfRsError::from)?;

    // The 2e Coulomb (total-density) + per-spin exchange gradient, gated.
    let (vj, vk_a, vk_b) = get_jk_uks(mol, dm_a, dm_b)?; // each [3, nao, nao]

    let n2 = nao * nao;
    let mut veff_a = vec![0.0_f64; NCOMP * n2];
    let mut veff_b = vec![0.0_f64; NCOMP * n2];
    for idx in 0..NCOMP * n2 {
        // veff_s = vxc_s + vj − hyb·vk_s (UKS full per-spin exchange).
        veff_a[idx] = oracle_sum(&[vxc_a[idx], vj[idx], -hyb * vk_a[idx]]);
        veff_b[idx] = oracle_sum(&[vxc_b[idx], vj[idx], -hyb * vk_b[idx]]);
    }
    Ok((veff_a, veff_b))
}

/// The per-spin XC-potential derivative on the grid (`uks.py` `get_vxc`):
/// `vmat_s[x,μ,ν] = -Σ_g w_g · (∂f/∂ρ_s)_g · ∇_x(AO_μ)_g · AO_ν,g`.
///
/// Cintx-INDEPENDENT: AO + derivs via `GTOval_sph_deriv1` on the byte-exact grid;
/// the per-spin potentials `∂f/∂ρ_a`, `∂f/∂ρ_b` via the xcfun `eval_uks` surface
/// (NEVER libxc, T-07-16). Returns `(vmat_a, vmat_b)`.
fn get_vxc_uks(
    refr: &UksReference,
    dm_a: &[f64],
    dm_b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let mol = refr.mol();
    let nao = mol.nao_nr;

    let mut grids = Grids::new();
    let (coords, weights) = grids.build(mol);
    let ngrids = coords.len();
    if ngrids == 0 || weights.len() != ngrids {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "uks get_vxc: empty / mismatched DFT grid".into(),
        )));
    }

    let ao = pyscf_gto::eval_gto(mol, "GTOval_sph_deriv1", &coords)?;
    let expect = 4 * ngrids * nao;
    if ao.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: ao.values.len(),
        }
        .into());
    }

    // Per-spin densities on the grid. eval_rho's LDA path takes ONLY the value
    // block (comp 0) — the first ngrids*nao slice of the F-order deriv1 buffer.
    let ao_value = &ao.values[..ngrids * nao];
    let den_a = Density::from_flat(nao, dm_a.to_vec());
    let den_b = Density::from_flat(nao, dm_b.to_vec());
    let (rho_a, _ga) = NumInt::eval_rho(ao_value, &den_a, ngrids, XcType::Lda)?;
    let (rho_b, _gb) = NumInt::eval_rho(ao_value, &den_b, ngrids, XcType::Lda)?;

    // Per-spin XC potentials ∂f/∂ρ_a, ∂f/∂ρ_b (genuine open-shell xcfun path).
    let (vrho_a, vrho_b) = ni_eval_uks_vrho(&refr.xc, &rho_a, &rho_b)?;

    let ao_at =
        |comp: usize, g: usize, mu: usize| ao.values[comp * ngrids * nao + (g + mu * ngrids)];
    let n2 = nao * nao;
    let mut vmat_a = vec![0.0_f64; NCOMP * n2];
    let mut vmat_b = vec![0.0_f64; NCOMP * n2];
    let mut ta = vec![0.0_f64; ngrids];
    let mut tb = vec![0.0_f64; ngrids];
    for x in 0..NCOMP {
        for mu in 0..nao {
            for nu in 0..nao {
                for g in 0..ngrids {
                    let common = weights[g] * ao_at(x + 1, g, mu) * ao_at(0, g, nu);
                    ta[g] = vrho_a[g] * common;
                    tb[g] = vrho_b[g] * common;
                }
                // - sign: ∇_X = -∇_x.
                vmat_a[x * n2 + mu + nu * nao] = -oracle_sum(&ta);
                vmat_b[x * n2 + mu + nu * nao] = -oracle_sum(&tb);
            }
        }
    }
    Ok((vmat_a, vmat_b))
}

/// `get_jk` over `int2e_ip1` for UKS: total-density Coulomb `vj` + per-spin
/// exchange `vk_a`/`vk_b`. Returns `(vj, vk_a, vk_b)`, each `[3, nao, nao]`,
/// with the `get_jk` `(-v)` bra-sign folded in. `int2e_ip1` MISSING → clean error.
// The (vj, vk_a, vk_b) triple is the natural spin-resolved JK return shape; a
// type alias would obscure more than it clarifies (matching the numint precedent
// for these per-spin grid-loop tuple returns).
#[allow(clippy::type_complexity)]
fn get_jk_uks(
    mol: &Mole,
    dm_a: &[f64],
    dm_b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), PyscfRsError> {
    let nao = mol.nao_nr;
    let eri = pyscf_gto::intor(mol, "int2e_ip1")?;
    let expect = NCOMP * nao * nao * nao * nao;
    if eri.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: eri.values.len(),
        }
        .into());
    }
    let n2 = nao * nao;
    let n3 = n2 * nao;
    let n4 = n3 * nao;
    let idx =
        |x: usize, i: usize, j: usize, k: usize, l: usize| x * n4 + i + j * nao + k * n2 + l * n3;
    let dval = |dm: &[f64], a: usize, b: usize| dm[a * nao + b];

    let mut vj = vec![0.0_f64; NCOMP * n2];
    let mut vk_a = vec![0.0_f64; NCOMP * n2];
    let mut vk_b = vec![0.0_f64; NCOMP * n2];
    let mut j_terms = vec![0.0_f64; n2];
    let mut ka_terms = vec![0.0_f64; n2];
    let mut kb_terms = vec![0.0_f64; n2];
    for x in 0..NCOMP {
        for i in 0..nao {
            for j in 0..nao {
                for k in 0..nao {
                    for l in 0..nao {
                        let g = eri.values[idx(x, i, j, k, l)];
                        let dtot = oracle_sum(&[dval(dm_a, l, k), dval(dm_b, l, k)]);
                        j_terms[k * nao + l] = g * dtot;
                        ka_terms[k * nao + l] = g * dval(dm_a, j, k);
                        kb_terms[k * nao + l] = g * dval(dm_b, j, k);
                    }
                }
                vj[x * n2 + i + j * nao] = -oracle_sum(&j_terms);
                vk_a[x * n2 + i + j * nao] = -oracle_sum(&ka_terms);
                vk_b[x * n2 + i + j * nao] = -oracle_sum(&kb_terms);
            }
        }
    }
    Ok((vj, vk_a, vk_b))
}

/// The UKS spin-resolved electronic gradient (`uks.py`). Reuses the UHF
/// spin-resolved Hellmann-Feynman + Pulay assembly with the per-spin KS veff,
/// plus the `grid_response` `extra_force` per atom when enabled.
pub fn grad_elec(
    g: &UksGradients,
    atmlst: Option<&[usize]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let refr = &g.reference;
    let mol = refr.mol();
    let nao = mol.nao_nr;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;
    let aoslices = aoslice_by_atom(mol)?;

    let dm_a = make_rdm1_spin(&refr.scf.alpha)?;
    let dm_b = make_rdm1_spin(&refr.scf.beta)?;
    let mut dm0 = vec![0.0_f64; nao * nao];
    for (i, slot) in dm0.iter_mut().enumerate() {
        *slot = oracle_sum(&[dm_a[i], dm_b[i]]);
    }
    let dme0 = uhf_make_rdm1e(&refr.scf)?;

    let s1 = get_ovlp(mol)?; // [3, nao, nao]; gated
    let h1 = get_hcore(mol)?; // [3, nao, nao]; gated
    let (vhf_a, vhf_b) = get_veff(refr, &dm_a, &dm_b)?; // per-spin KS veff

    let n2 = nao * nao;
    let mut de = Vec::with_capacity(rows.len());
    for &ia in &rows {
        let h1ao = hcore_deriv(mol, &h1, &aoslices, ia)?; // [3, nao, nao]
        let (p0, p1) = aoslices[ia];
        let extra = g.extra_force(ia)?;

        let mut row = [0.0_f64; 3];
        for (comp, slot) in row.iter_mut().enumerate() {
            let base = comp * n2;
            let mut t1 = Vec::with_capacity(n2);
            for j in 0..nao {
                for i in 0..nao {
                    t1.push(h1ao[base + i + j * nao] * dm0[i * nao + j]);
                }
            }
            let term1 = oracle_sum(&t1);

            let mut t2 = Vec::new();
            let mut t3 = Vec::new();
            for i in p0..p1 {
                for j in 0..nao {
                    t2.push(vhf_a[base + i + j * nao] * dm_a[i * nao + j]);
                    t2.push(vhf_b[base + i + j * nao] * dm_b[i * nao + j]);
                    t3.push(s1[base + i + j * nao] * dme0[i * nao + j]);
                }
            }
            let term2 = 2.0 * oracle_sum(&t2);
            let term3 = 2.0 * oracle_sum(&t3);

            *slot = oracle_sum(&[term1, term2, -term3, extra[comp]]);
        }
        de.push(row);
    }
    let _ = oracle_dot(&dm0[..nao.min(dm0.len())], &dm0[..nao.min(dm0.len())]);
    Ok(de)
}

/// Per-point per-spin `(∂f/∂ρ_a, ∂f/∂ρ_b)` via the xcfun `eval_uks` surface
/// (NEVER libxc — `NumInt::new()` uses the default `Xcfun` backend, T-07-16).
fn ni_eval_uks_vrho(
    xc: &str,
    rho_a: &[f64],
    rho_b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    use pyscf_dft::xc_backend::XcBackend;
    let spec = pyscf_dft::parser::xcfun::parse_xc(xc).map_err(PyscfRsError::from)?;
    let backend = XcBackend::default(); // Xcfun; libxc only under --features libxc
    let out = backend
        .eval_uks(&spec, rho_a, rho_b, None, None, None, DerivOrder::Vxc)
        .map_err(PyscfRsError::from)?;
    Ok((out.vrho_a, out.vrho_b))
}

/// Assert a gradient intor came back component-leading `[3, nao, nao]`.
fn assert_component_leading(
    out: &pyscf_gto::IntorOutput,
    nao: usize,
    name: &str,
) -> Result<(), PyscfRsError> {
    let expect = NCOMP * nao * nao;
    if out.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: out.values.len(),
        }
        .into());
    }
    if out.shape.first().copied() != Some(NCOMP) {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "{name} must be component-leading [3, nao, nao] (Pitfall 4); got {:?}",
                out.shape
            ),
        )));
    }
    Ok(())
}

/// UKS electronic gradient seam preserved for the 07-02 module stub.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "UksGradients::grad_elec requires a UksReference snapshot — build a \
         UksGradients via UksGradients::new(reference) (07-05)"
            .into(),
    )))
}
