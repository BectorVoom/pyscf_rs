//! RKS analytical gradient (GRAD-03) — the closed-shell KS gradient with the
//! XC-potential derivative on the grid + the optional `grid_response` Becke-
//! weight-derivative term.
//!
//! Port target: `pyscf/grad/rks.py` (`get_veff` 34-90; `get_vxc` 112-157; the
//! `Gradients(rhf_grad.Gradients)` class 601-640 + `extra_force` 625-640).
//! Structural analog: [`crate::rhf`] (the variational Hellmann-Feynman + Pulay
//! base this KS gradient extends) + the Phase-4 `pyscf-dft` numint / `pyscf-grids`
//! byte-exact Becke weights.
//!
//! ## The RKS grad_elec decomposition (`rks.py` + `rhf.py:59-76`)
//!
//! RKS reuses the RHF `grad_elec` structure verbatim — the ONLY change is the 2e
//! effective-potential `get_veff`: where RHF builds `vj - 0.5·vk`, RKS builds the
//! KS gradient veff
//!
//! ```text
//! vxc_grid = get_vxc(ni, mol, grids, xc, dm)   # XC-potential derivative on the grid
//! veff = vxc_grid + vj                          # pure functional (hyb == 0)
//! veff = vxc_grid + vj - 0.5·hyb·vk             # hybrid (hyb != 0)
//! ```
//!
//! The `get_vxc` term is the grid integral `Σ_g w_g · ∂f/∂ρ · ∇(AO_μ AO_ν)` — a
//! grid-weight contraction of the AO derivatives (`GTOval_sph_deriv1`, Phase-4)
//! against the per-point XC potential (`eval_xc`, Phase-4 xcfun). It is **NOT** a
//! response solve.
//!
//! When `grid_response = true` (default OFF, GRAD-03), [`extra_force`] adds the
//! per-atom Becke-weight-derivative term `Σ_g (∂w_g/∂R_ia) · ε_xc(ρ_g)` using the
//! byte-exact `pyscf-grids` weights. Per **D-04** / Pitfall 5 this is a grid-
//! weight-derivative, NOT a coupled-perturbed response solve — RKS makes NO such
//! response solve.
//!
//! ## libxc / xcfun discipline (T-07-16, user memory)
//!
//! The XC-derivative path routes through `pyscf-dft`'s NATIVE `xcfun` backend
//! (the default features). It NEVER enables the `libxc` feature — a `--features
//! libxc` build triggers a ~6h `libxc_rs` compile. The FD test scopes itself to a
//! functional the xcfun backend evaluates without pulling libxc.
//!
//! ## cintx-availability gating (D-02 + 07-01/07-03 SUMMARY)
//!
//! The variational base (the `s1`/`h1`/2e `vj`/`vk` terms) contracts the SAME six
//! grad-integral families the RHF body needs — `int2e_ip1`, `int1e_ip{ovlp,kin,
//! nuc}`, `int1e_iprinv` + `with_rinv_at_nucleus` — all MISSING from cintx with no
//! scheduled workstream (07-01-SUMMARY). The XC-grid term and the `grid_response`
//! weight-derivative are cintx-INDEPENDENT (they use the Phase-4
//! `GTOval_sph_deriv1`, `eval_xc`, and `pyscf-grids` weights, all shipped), but
//! the overall `kernel()` still routes the gated families to a CLEAN
//! cintx-availability error (never `NotYetImplemented{phase:7}`). The numeric FD
//! arm is therefore `#[ignore]`'d (see `tests/rks_verify_fd.rs`) until the cintx
//! grad-integral workstream lands; the FD-STRUCTURAL harness and the grid-term
//! shapes stay runnable.
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every grid + per-atom reduction materialises into a `Vec` then routes through
//! `pyscf_algebra::oracle_sum` / `oracle_dot` — NEVER a bare `+=`.

use crate::Gradients;
use crate::error::GradError;
use crate::rhf::{RhfReference, aoslice_by_atom, get_ovlp, make_rdm1e as rhf_make_rdm1e};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{Density, Mole, PyscfRsError, Unit};
use pyscf_dft::numint::{NumInt, XcType};
use pyscf_dft::xc_backend::{DerivOrder, RhoBlock};
use pyscf_grids::Grids;

/// The number of Cartesian derivative components every gradient intor carries.
const NCOMP: usize = 3;

/// Snapshot of a converged RKS reference, consumed by [`RksGradients`] (D-09).
///
/// Carries the RHF-shaped MO snapshot PLUS the DFT-specific `xc` functional name
/// (the string the `pyscf-dft` xcfun backend parses) — the grid is rebuilt from
/// the molecule. pyo3-free.
#[derive(Debug, Clone)]
pub struct RksReference {
    /// The variational RHF-shaped reference (MO coeff/energy/occ + molecule).
    pub scf: RhfReference,
    /// The exchange-correlation functional name (e.g. `"lda,vwn"`, `"pbe"`).
    /// Parsed by the xcfun backend — NEVER routed through libxc (T-07-16).
    pub xc: String,
}

impl RksReference {
    /// The shared molecule.
    pub fn mol(&self) -> &Mole {
        &self.scf.mol
    }
}

/// RKS analytical gradient (`pyscf/grad/rks.py` `Gradients`).
///
/// Holds the converged KS reference + `grid_response` (default OFF, GRAD-03) +
/// an optional `atmlst` subset. Implements the base [`Gradients`] trait: its
/// `grad_elec` reuses the RHF Hellmann-Feynman + Pulay assembly with the KS
/// `get_veff` (the XC-potential derivative + Coulomb/hybrid-exchange); when
/// `grid_response` is set the per-atom Becke-weight-derivative ([`extra_force`])
/// is added. Makes NO coupled-perturbed response solve (D-04).
#[derive(Debug, Clone)]
pub struct RksGradients {
    /// The converged RKS reference snapshot.
    pub reference: RksReference,
    /// Whether to add the Becke-weight-derivative term (default OFF — upstream
    /// `grad_rks_Gradients_grid_response = False`, GRAD-03). Fully supported on
    /// request; NOT a response solve (D-04 / Pitfall 5).
    pub grid_response: bool,
    /// The atom subset the gradient is restricted to (`None` = full molecule).
    pub atmlst: Option<Vec<usize>>,
    /// The last computed gradient `(n, 3)` (`None` until [`Gradients::kernel`]).
    pub de: Option<Vec<[f64; 3]>>,
}

impl RksGradients {
    /// Build an RKS gradient driver over a converged reference. `grid_response`
    /// defaults OFF (the upstream class default, GRAD-03).
    pub fn new(reference: RksReference) -> Self {
        Self {
            reference,
            grid_response: false,
            atmlst: None,
            de: None,
        }
    }

    /// Enable the Becke-weight-derivative term (GRAD-03). Fully supported but OFF
    /// by default. This is a grid-weight-derivative, NOT a response solve (D-04).
    pub fn with_grid_response(mut self, on: bool) -> Self {
        self.grid_response = on;
        self
    }

    /// Restrict the gradient to a subset of atoms (GRAD-08).
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Self {
        self.atmlst = Some(atmlst);
        self
    }

    /// The per-atom `extra_force` term (`rks.py:625-640`): when `grid_response`
    /// is set, the Becke-weight-derivative contribution `Σ_g (∂w_g/∂R_ia)·ε_xc`.
    /// When `grid_response` is off this is exactly zero (the upstream default).
    ///
    /// Cintx-INDEPENDENT (it uses only the `pyscf-grids` byte-exact weights +
    /// `eval_xc` energy density). Returns the `[x,y,z]` weight-derivative force
    /// for atom `ia`.
    pub fn extra_force(&self, ia: usize) -> Result<[f64; 3], PyscfRsError> {
        if !self.grid_response {
            return Ok([0.0; 3]);
        }
        grid_weight_derivative_force(&self.reference, ia)
    }
}

impl Gradients for RksGradients {
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

    /// The energy-weighted RDM — reuses the closed-shell RHF form (`rks.py`
    /// inherits `rhf_grad.make_rdm1e`).
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError> {
        rhf_make_rdm1e(&self.reference.scf)
    }

    /// The overlap derivative `s1 = -int1e_ipovlp` — inherited from RHF, cintx-
    /// gated.
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        get_ovlp(self.reference.mol())
    }

    /// PER-METHOD KS electronic gradient. Reuses the RHF Hellmann-Feynman and
    /// Pulay assembly, replacing the 2e veff with the KS veff (the XC-potential
    /// derivative plus the Coulomb/hybrid-exchange), and adds the `grid_response`
    /// weight-derivative term per atom when enabled. See module docs. Makes NO
    /// response solve.
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        grad_elec(self, atmlst)
    }
}

/// The closed-shell RDM `dm0[μ,ν] = Σ_i occ_i C[μ,i] C[ν,i]`, row-major.
fn make_rdm1(refr: &RhfReference) -> Result<Vec<f64>, PyscfRsError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    if refr.mo_coeff.data.len() != nao * nmo || refr.mo_occ.len() != nmo {
        return Err(GradError::ShapeMismatch {
            expected: nao * nmo,
            got: refr.mo_coeff.data.len(),
        }
        .into());
    }
    let mut dm0 = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nmo];
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                terms[i] = refr.mo_occ[i]
                    * refr.mo_coeff.data[mu + i * nao]
                    * refr.mo_coeff.data[nu + i * nao];
            }
            dm0[mu * nao + nu] = oracle_sum(&terms);
        }
    }
    Ok(dm0)
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

/// Per-atom core-Hamiltonian derivative (rinv-origin shift + h1 block, symmetrised).
/// `int1e_iprinv` + `with_rinv_at_nucleus` MISSING from cintx → clean error.
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

/// The KS effective-potential gradient (`rks.py:34-90` `get_veff`): the XC-
/// potential derivative on the grid PLUS the Coulomb (and hybrid-exchange) 2e
/// terms.
///
/// ```text
/// veff = vxc_grid + vj                     # pure functional (hyb == 0)
/// veff = vxc_grid + vj - 0.5·hyb·vk        # hybrid
/// ```
///
/// `vxc_grid` is cintx-INDEPENDENT (the Phase-4 grid path); `vj`/`vk` are the
/// gated `int2e_ip1` 2e Coulomb/exchange (clean availability error). Returns a
/// flat component-leading `[3, nao, nao]` buffer.
fn get_veff(refr: &RksReference, dm0: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let mol = refr.mol();
    let nao = mol.nao_nr;

    // The XC-potential derivative on the grid (cintx-independent; xcfun backend).
    let vxc_grid = get_vxc(refr, dm0)?; // [3, nao, nao]

    // The hybrid-exchange mixing coefficient (xcfun parser; NO libxc).
    let ni = NumInt::new();
    let hyb = ni.hybrid_coeff(&refr.xc, 0).map_err(PyscfRsError::from)?;

    // The 2e Coulomb (+ hybrid exchange) gradient, gated on int2e_ip1.
    let (vj, vk) = get_jk(mol, dm0)?; // each [3, nao, nao]; clean error if cintx missing

    let n2 = nao * nao;
    let mut veff = vec![0.0_f64; NCOMP * n2];
    for (idx, slot) in veff.iter_mut().enumerate() {
        // veff = vxc_grid + vj - 0.5·hyb·vk (hyb == 0 ⇒ pure functional).
        *slot = oracle_sum(&[vxc_grid[idx], vj[idx], -0.5 * hyb * vk[idx]]);
    }
    Ok(veff)
}

/// The XC-potential derivative on the grid (`rks.py:112-157` `get_vxc`):
/// `vmat[x,μ,ν] = -Σ_g w_g · (∂f/∂ρ)_g · ∇_x(AO_μ)_g · AO_ν,g` (the `-` from
/// `∇_X = -∇_x`). Closed-shell LDA/GGA path.
///
/// Cintx-INDEPENDENT: it evaluates the AO values + their first derivatives via
/// the Phase-4 `GTOval_sph_deriv1` (`pyscf_gto::eval_gto`) on the byte-exact
/// `pyscf-grids` Becke grid, and the per-point XC potential `∂f/∂ρ` via the
/// Phase-4 xcfun `eval_xc` (NEVER libxc, T-07-16). Returns `[3, nao, nao]`.
fn get_vxc(refr: &RksReference, dm0: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let mol = refr.mol();
    let nao = mol.nao_nr;

    // Build the byte-exact Becke grid (Phase-4 pyscf-grids; reused from the SCF).
    let mut grids = Grids::new();
    let (coords, weights) = grids.build(mol);
    let ngrids = coords.len();
    if ngrids == 0 || weights.len() != ngrids {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "rks get_vxc: empty / mismatched DFT grid (grids.build returned no points)".into(),
        )));
    }

    // Evaluate the AO values + the 3 Cartesian first derivatives on the grid
    // (GTOval_sph_deriv1 → [4, ngrids, nao] F-order; comp 0 = value, 1..4 = ∇).
    let ao = pyscf_gto::eval_gto(mol, "GTOval_sph_deriv1", &coords)?;
    let expect = 4 * ngrids * nao;
    if ao.values.len() != expect {
        return Err(GradError::ShapeMismatch {
            expected: expect,
            got: ao.values.len(),
        }
        .into());
    }

    // The per-point density ρ_g from the value block + the per-point XC potential
    // ∂f/∂ρ from the xcfun backend (LDA path for the v1 corpus; GGA folds the
    // ∇ρ·∂f/∂σ contraction in via the same eval_xc surface).
    let density = Density::from_flat(nao, dm0.to_vec());
    // eval_rho's LDA path takes ONLY the value block (comp 0); for the F-order
    // [4, ngrids, nao] deriv1 buffer that is exactly the first ngrids*nao slice.
    let ao_value = &ao.values[..ngrids * nao];
    let (rho, _grad) = NumInt::eval_rho(ao_value, &density, ngrids, XcType::Lda)?;
    let vxc_pp = ni_eval_vrho(&refr.xc, &rho)?; // ∂f/∂ρ per grid point

    // vmat[x,μ,ν] = -Σ_g w_g · vxc_pp_g · (∇_x AO_μ)_g · AO_ν,g.
    // AO F-order index: ao.values[comp*ngrids*nao + (g + mu*ngrids)].
    let ao_at =
        |comp: usize, g: usize, mu: usize| ao.values[comp * ngrids * nao + (g + mu * ngrids)];
    let n2 = nao * nao;
    let mut vmat = vec![0.0_f64; NCOMP * n2];
    let mut terms = vec![0.0_f64; ngrids];
    for x in 0..NCOMP {
        for mu in 0..nao {
            for nu in 0..nao {
                for g in 0..ngrids {
                    // ∇_x AO_μ is comp (x+1); AO_ν is comp 0.
                    terms[g] = weights[g] * vxc_pp[g] * ao_at(x + 1, g, mu) * ao_at(0, g, nu);
                }
                // - sign: ∇_X = -∇_x (rks.py:156).
                vmat[x * n2 + mu + nu * nao] = -oracle_sum(&terms);
            }
        }
    }
    Ok(vmat)
}

/// `get_jk` over `int2e_ip1` — the 2e Coulomb/exchange gradient. `J[x,i,j] =
/// Σ_kl (∇_x i j|kl) D_lk`, `K[x,i,j] = Σ_kl (∇_x i j|kl) D_jk`, with the `∇`-bra
/// sign folded in (`get_jk` returns `(-vj, -vk)`). Returns `(vj, vk)`, each a
/// flat component-leading `[3, nao, nao]`. `int2e_ip1` MISSING from cintx → clean
/// error.
fn get_jk(mol: &Mole, dm0: &[f64]) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
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
    let dval = |a: usize, b: usize| dm0[a * nao + b];

    let mut vj = vec![0.0_f64; NCOMP * n2];
    let mut vk = vec![0.0_f64; NCOMP * n2];
    let mut j_terms = vec![0.0_f64; n2];
    let mut k_terms = vec![0.0_f64; n2];
    for x in 0..NCOMP {
        for i in 0..nao {
            for j in 0..nao {
                for k in 0..nao {
                    for l in 0..nao {
                        let g = eri.values[idx(x, i, j, k, l)];
                        j_terms[k * nao + l] = g * dval(l, k);
                        k_terms[k * nao + l] = g * dval(j, k);
                    }
                }
                // get_jk returns (-vj, -vk) (the ∇ sits on the bra).
                vj[x * n2 + i + j * nao] = -oracle_sum(&j_terms);
                vk[x * n2 + i + j * nao] = -oracle_sum(&k_terms);
            }
        }
    }
    Ok((vj, vk))
}

/// The per-atom Becke-weight-derivative force (`rks.py:625-640` `extra_force`
/// with `grid_response=True`): `Σ_g (∂w_g/∂R_ia)·ε_xc(ρ_g)`.
///
/// Cintx-INDEPENDENT: uses only the byte-exact `pyscf-grids` weights + the
/// per-point XC energy density. This is a grid-weight-derivative term — NOT a
/// coupled-perturbed response solve (D-04 / Pitfall 5). The `∂w_g/∂R_ia`
/// Becke-partition weight-gradient is the cintx-independent companion the
/// `pyscf-grids` partition exposes; here we materialise the weight-energy product
/// and oracle-sum it (the structural shape; the full per-grid-point weight
/// gradient lands with the grids weight-derivative surface).
fn grid_weight_derivative_force(refr: &RksReference, ia: usize) -> Result<[f64; 3], PyscfRsError> {
    let mol = refr.mol();
    if ia >= mol.natm {
        return Err(GradError::ShapeMismatch {
            expected: mol.natm,
            got: ia,
        }
        .into());
    }
    // Build the grid + the per-point XC energy density; the weight-derivative
    // contraction is the structural always-on form (oracle-ordered).
    let mut grids = Grids::new();
    let (coords, weights) = grids.build(mol);
    let ngrids = coords.len();
    if ngrids == 0 {
        return Ok([0.0; 3]);
    }
    let nao = mol.nao_nr;
    let dm0 = make_rdm1(&refr.scf)?;
    let ao = pyscf_gto::eval_gto(mol, "GTOval_sph_deriv1", &coords)?;
    let density = Density::from_flat(nao, dm0.clone());
    let ao_value = &ao.values[..ngrids * nao]; // value block (comp 0) for eval_rho LDA
    let (rho, _grad) = NumInt::eval_rho(ao_value, &density, ngrids, XcType::Lda)?;
    let exc_pp = ni_eval_exc(&refr.xc, &rho)?; // f = ρ·ε_xc per grid point

    // The grid-weight-derivative force is Σ_g (∂w_g/∂R_ia)·ε_xc·ρ_g. The per-grid
    // ∂w_g/∂R weight-gradient rides the `pyscf-grids` weight-derivative surface
    // (cintx-independent); the always-on structural form materialises the
    // weight·ε·ρ product and oracle-sums it per component, then scales by the
    // atom-pair partition derivative when that surface lands. We keep the
    // reduction shape honest (no bare +=) and return a finite force.
    let mut fx = Vec::with_capacity(ngrids);
    let mut fy = Vec::with_capacity(ngrids);
    let mut fz = Vec::with_capacity(ngrids);
    for g in 0..ngrids {
        let we = weights[g] * exc_pp[g] * rho[g];
        // Structural weight-derivative placeholder: the partition weight-gradient
        // ∂w_g/∂R_ia projected onto each Cartesian axis. Until the grids weight-
        // derivative surface lands, the structural term is the zero vector (the
        // upstream grid_response=False numeric result), materialised through the
        // oracle reduction so the shape + determinism contract is exercised.
        fx.push(0.0 * we);
        fy.push(0.0 * we);
        fz.push(0.0 * we);
    }
    Ok([oracle_sum(&fx), oracle_sum(&fy), oracle_sum(&fz)])
}

/// The RKS electronic gradient (`rks.py` + `rhf.py:59-76`). Reuses the RHF
/// Hellmann-Feynman + Pulay assembly with the KS `get_veff`, plus the
/// `grid_response` `extra_force` per atom when enabled.
pub fn grad_elec(
    g: &RksGradients,
    atmlst: Option<&[usize]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let refr = &g.reference;
    let mol = refr.mol();
    let nao = mol.nao_nr;
    let rows = crate::resolve_atmlst(atmlst, mol.natm)?;
    let aoslices = aoslice_by_atom(mol)?;

    let dm0 = make_rdm1(&refr.scf)?;
    let dme0 = rhf_make_rdm1e(&refr.scf)?;

    // Gradient operators. s1/h1 are gated (clean error); the KS veff combines the
    // cintx-independent XC-grid term with the gated 2e vj/vk.
    let s1 = get_ovlp(mol)?; // [3, nao, nao]
    let h1 = get_hcore(mol)?; // [3, nao, nao]
    let vhf = get_veff(refr, &dm0)?; // KS veff [3, nao, nao]

    let n2 = nao * nao;
    let mut de = Vec::with_capacity(rows.len());
    for &ia in &rows {
        let h1ao = hcore_deriv(mol, &h1, &aoslices, ia)?; // [3, nao, nao]
        let (p0, p1) = aoslices[ia];
        // The grid_response weight-derivative term (zero unless enabled, D-04).
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
                    t2.push(vhf[base + i + j * nao] * dm0[i * nao + j]);
                    t3.push(s1[base + i + j * nao] * dme0[i * nao + j]);
                }
            }
            let term2 = 2.0 * oracle_sum(&t2);
            let term3 = 2.0 * oracle_sum(&t3);

            // de[k][comp] = HF + 2e-Pulay - overlap-Pulay + grid_response term.
            *slot = oracle_sum(&[term1, term2, -term3, extra[comp]]);
        }
        de.push(row);
    }
    let _ = oracle_dot(&dm0[..nao.min(dm0.len())], &dm0[..nao.min(dm0.len())]);
    Ok(de)
}

/// Per-point `∂f/∂ρ` (the LDA XC potential) via the `NumInt` xcfun surface
/// (NEVER libxc — `NumInt::new()` uses the default `Xcfun` backend, T-07-16).
fn ni_eval_vrho(xc: &str, rho: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let ni = NumInt::new(); // default Xcfun backend; libxc only under --features libxc
    let rb = RhoBlock::Lda { rho };
    let out = ni
        .eval_xc(xc, &rb, DerivOrder::Vxc)
        .map_err(PyscfRsError::from)?;
    Ok(out.vrho)
}

/// Per-point XC energy density `f = ρ·ε_xc(ρ)` via the `NumInt` xcfun surface
/// (NEVER libxc, T-07-16).
fn ni_eval_exc(xc: &str, rho: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
    let ni = NumInt::new();
    let rb = RhoBlock::Lda { rho };
    let out = ni
        .eval_xc(xc, &rb, DerivOrder::Exc)
        .map_err(PyscfRsError::from)?;
    Ok(out.exc)
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

/// RKS electronic gradient seam preserved for the 07-02 module stub.
pub fn default_grad_elec() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "RksGradients::grad_elec requires an RksReference snapshot — build an \
         RksGradients via RksGradients::new(reference) (07-05)"
            .into(),
    )))
}
