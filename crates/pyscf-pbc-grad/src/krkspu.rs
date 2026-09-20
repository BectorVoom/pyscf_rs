//! KRKS+U k-point analytic nuclear gradient — `pyscf/pbc/grad/krkspu.py` (142 l).
//!
//! The restricted DFT+U gradient. Upstream declares it as
//! `class Gradients(krks_grad.Gradients)` (`krkspu.py:135`): 18-07's assembly
//! with `extra_force` extended by the Hubbard-`U` term (`:140-142`). This
//! module ports the two bodies that are genuinely new —
//! [`first_order_local_orbitals`] (`generate_first_order_local_orbitals`,
//! `:27-87`) and [`hubbard_u_deriv1`] (`_hubbard_U_deriv1`, `:89-133`) — and
//! wraps 18-07's [`KrksGradients`](crate::krks::KrksGradients) in
//! [`KrkspuGradients`], which adds the `U` rows to the kernel output exactly
//! where upstream's `extra_force` sits (per atom, after `/nkpts`, before the
//! whole-array `vppnl` term — so adding to `kernel()` is exact).
//!
//! # Upstream correspondence (`krkspu.py` line → here)
//!
//! | upstream | here |
//! |---|```
//! | `generate_first_order_local_orbitals` `:27-60` (C0 + Löwdin eigendecomposition) | [`first_order_local_orbitals`] (C0 via `kspu::make_minao_lo`, `S0` eigendecomposition inline) |
//! | `make_coeff` `:61-86` (atom-scoped `sAA1`/`sAB1`, Sylvester solve, `C1`) | [`make_coeff`] |
//! | `_hubbard_U_deriv1` `:89-133` | [`hubbard_u_deriv1`] |
//! | `Gradients(krks_grad.Gradients)` `:135-142` | [`KrkspuGradients`] |
//!
//! # What is called, never reimplemented (18-08 Task 1)
//!
//! `_set_U`, `_make_minao_lo` and `reference_mol` are **called** from
//! `pyscf_pbc_dft::kspu` ([`set_u`](pyscf_pbc_dft::kspu::set_u),
//! [`make_minao_lo`](pyscf_pbc_dft::kspu::make_minao_lo),
//! [`reference_cell`](pyscf_pbc_dft::kspu::reference_cell)), never
//! reimplemented. Two local-orbital constructions in one workspace is how a
//! plausible wrong number ships (`16-CONTEXT §1.1`'s ruling); the crate's own
//! test asserts neither symbol is defined here.
//!
//! # Index orders (18-CONTEXT trap 9)
//!
//! Every density is indexed **`ji`-aware**: the contractions below keep
//! upstream's `einsum` index order literally (`'pq,xqi->xpi'`,
//! `'pj,xjq->xpq'`, `'xii->x'`, `'xij,ji->x'`), with the row-major density
//! this port's [`make_rdm1`](pyscf_pbc_scf::krdm::make_rdm1) produces.
//! The port's integral tables are F-order per component; [`make_coeff`]
//! reads them through [`PbcIntorOutput::element`](pyscf_pbc_gto::pbc_intor::PbcIntorOutput::element)
//! so the ket-derivative blocks land on the right `(i, j)` whatever the
//! storage order. The `.real` is taken as the real part of an explicitly
//! accumulated complex pair (D-PBC-31 clause 4 at the scalar level).
//!
//! # No new kernels (ALG-06, D-PBC-17)
//!
//! All arithmetic is host-side complex pairs over materialised buffers,
//! reduced through [`oracle_sum`](pyscf_algebra::oracle_sum) — never a bare
//! `+=`. `pyscf-pbc-grad` names no `cubecl-*` (`xtask
//! check-dependency-wall`); the CubeCL manual's generics-`Float` discipline
//! has no device kernel to apply to in this file.
//!
//! # Gate B tier
//!
//! Gate B for these two bodies is **5e-6, not 1e-6** — upstream's own DFT+U
//! assertions are 5-decimal (`test_krkspu.py:87,92`; `test_kukspu.py:67,72`)
//! where every other body in the directory is 6-decimal (`18-CONTEXT §2.3`).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::PyscfRsError;
use pyscf_pbc_dft::kspu::HubbardU;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::pbc_intor::{PbcIntorOpts, PbcIntorOutput, intor_cross};
use pyscf_pbc_scf::krdm::make_rdm1;
use pyscf_pbc_scf::types::KMats;

use crate::error::PbcGradError;
use crate::gradients::{Gradient, Gradients};
use crate::krks::KrksGradients;

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(message.into()))
}

fn lift_dft<T>(r: Result<T, pyscf_pbc_dft::PbcDftError>) -> Result<T, PyscfRsError> {
    r.map_err(|e| invalid(format!("KRKS+U gradient: DFT+U substrate failed: {e}")))
}

fn lift_alg<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> Result<T, PyscfRsError> {
    r.map_err(|e| invalid(format!("KRKS+U gradient: {what} failed: {e}")))
}

/// Ordered complex dot over materialised `(re, im)` pairs — the scalar-level
/// D-PBC-17 discipline. Returns `(re, im)`.
fn csum(terms: &[(f64, f64)]) -> (f64, f64) {
    let re: Vec<f64> = terms.iter().map(|(r, _)| *r).collect();
    let im: Vec<f64> = terms.iter().map(|(_, i)| *i).collect();
    (oracle_sum(&re), oracle_sum(&im))
}

/// F-order `ni × nj` plane → row-major. The port's integral tables are
/// F-order per component; the algebra below is row-major throughout, so the
/// conversion happens once at the boundary, never per access.
fn f_to_c(f: &[f64], ni: usize, nj: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            out[i * nj + j] = f[i + j * ni];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// `generate_first_order_local_orbitals` — `krkspu.py:27-60`.
// ---------------------------------------------------------------------------

/// Per-k Löwdin data shared by every atom's [`make_coeff`] call —
/// `krkspu.py:34-50` hoisted out of the closure.
///
/// CAREFUL: `c0` here is the **raw** least-squares projection
/// `C0_minao = cho_solve(sAA, sAB)` (`:43`) — NOT the Löwdin-orthogonalized
/// `C_ao_lo` that [`make_minao_lo`](pyscf_pbc_dft::kspu::make_minao_lo)
/// returns. `make_coeff` differentiates the orthogonalisation itself, so it
/// needs the pre-orthogonalisation factor (`:75-85` all read `C0_minao`);
/// the orthogonalized set belongs to [`hubbard_u_deriv1`], which builds it
/// separately exactly as `_hubbard_U_deriv1` does (`:102`).
#[derive(Debug, Clone)]
pub struct LocalOrbBase {
    /// AO count.
    pub nao: usize,
    /// Local-orbital count.
    pub nlo: usize,
    /// `C0_minao[k]`, COLUMN-MAJOR `nao × nlo` (raw projection).
    pub c0: Vec<CTensor>,
    /// `w = sqrt(eig(S0))` per k-point (`krkspu.py:48`), with
    /// `S0 = sAB^H·C0_raw`.
    pub w: Vec<Vec<f64>>,
    /// `v`, the eigenvectors of `S0`, COLUMN-MAJOR `nlo × nlo` (the
    /// `zeigh_gen` convention, as in `kspu::vec_lowdin`).
    pub v: Vec<CTensor>,
    /// `S0_lowdin[k] = (v/w)·v^H`, ROW-MAJOR `nlo × nlo` (`:50`).
    pub s0_lowdin: Vec<CTensor>,
    /// `S0[k] = sAB^H·C0`, ROW-MAJOR `nao × nao` overlap at each k-point
    /// (kept row-major for the `C_inv` contraction in [`hubbard_u_deriv1`]).
    pub saa0: Vec<CTensor>,
}

impl LocalOrbBase {
    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.c0.len()
    }
}

/// `generate_first_order_local_orbitals(cell, minao_ref, kpts)` minus the
/// closure — `krkspu.py:27-60`.
///
/// Builds the MINAO reference cell, the raw least-squares projection
/// `C0 = S^{-1}·S12` per k-point (`la.cho_solve`, `:42-43` — solved here
/// column by column through `zsolve_linear`, the same call `kspu` uses)
/// and the Löwdin eigendecomposition of `S0 = sAB^H·C0` at each k-point.
/// The overlap tables it consumes are 18-05's `get_ovlp` integrals
/// (`int1e_ovlp` + `intor_cross`), with no sign applied here — the minus
/// lives in the gradient's `get_ovlp`, not in the integrals
/// (`18-CONTEXT §1.4` lineage).
///
/// # Errors
/// Shape disagreement, a singular Löwdin metric, or a failed integral / solve.
pub fn first_order_local_orbitals(
    cell: &Cell,
    minao_ref: &str,
    kpts: &[[f64; 3]],
) -> Result<LocalOrbBase, PyscfRsError> {
    use pyscf_pbc_dft::kspu::reference_cell;
    use pyscf_pbc_gto::get_ovlp;

    let nkpts = kpts.len();
    if nkpts == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    let nao = cell.mol.nao_nr;
    let pcell = lift_dft(reference_cell(cell, minao_ref))?;
    let nlo = pcell.mol.nao_nr;
    if nlo == 0 {
        return Err(PbcGradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    // `sAA[k]`, F-order → row-major for the solves below.
    let saa_f =
        get_ovlp(cell, kpts).map_err(|e| invalid(format!("KRKS+U C0: get_ovlp failed: {e}")))?;
    if saa_f.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: saa_f.len(),
        }
        .into());
    }
    // `sAB[k]`, F-order `nao × nlo` → row-major.
    let sab_f = intor_cross("int1e_ovlp", cell, &pcell, kpts, PbcIntorOpts::default())
        .map_err(|e| invalid(format!("KRKS+U C0: intor_cross ovlp failed: {e}")))?;
    if sab_f.nkpts() != nkpts || sab_f.ni != nao || sab_f.nj != nlo {
        return Err(invalid(format!(
            "KRKS+U C0: intor_cross ovlp has nkpts/ni/nj = {}/{}/{} for {nkpts}/{nao}/{nlo}",
            sab_f.nkpts(),
            sab_f.ni,
            sab_f.nj,
        )));
    }

    let mut c0_all = Vec::with_capacity(nkpts);
    let mut w_all = Vec::with_capacity(nkpts);
    let mut v_all = Vec::with_capacity(nkpts);
    let mut s0l_all = Vec::with_capacity(nkpts);
    let mut saa0 = Vec::with_capacity(nkpts);
    for k in 0..nkpts {
        let saa_row = CTensor::from_planes(
            f_to_c(&saa_f[k].re, nao, nao),
            f_to_c(&saa_f[k].im, nao, nao),
        );
        // `C0_minao = cho_solve(sAA, sAB)` (`:42-43`), COLUMN-MAJOR
        // `nao × nlo` — one `zsolve_linear` per reference column.
        let mut c0 = CTensor::zeros(nao * nlo);
        for j in 0..nlo {
            let rhs = CTensor::from_planes(
                (0..nao).map(|i| sab_f.kmats[k].re[i + j * nao]).collect(),
                (0..nao).map(|i| sab_f.kmats[k].im[i + j * nao]).collect(),
            );
            let x = lift_alg(
                pyscf_algebra::zsolve_linear(&saa_row, &rhs, nao),
                "C0 projection solve",
            )?;
            for i in 0..nao {
                c0.re[i + j * nao] = x.re[i];
                c0.im[i + j * nao] = x.im[i];
            }
        }
        // `S0 = sAB[k]^H · C0_minao[k]` (`:46`), row-major `nlo × nlo`.
        let mut s0 = CTensor::zeros(nlo * nlo);
        for a in 0..nlo {
            for b in 0..nlo {
                let mut terms = Vec::with_capacity(nao);
                for i in 0..nao {
                    let (sr, si) = (
                        sab_f.kmats[k].re[i + a * nao],
                        sab_f.kmats[k].im[i + a * nao],
                    );
                    // conj(sAB[i,a]).
                    let (cr, ci) = (sr, -si);
                    // C0[i,b].
                    let q = i + b * nao;
                    let (dr, di) = (c0.re[q], c0.im[q]);
                    terms.push((cr * dr - ci * di, cr * di + ci * dr));
                }
                let (r, im) = csum(&terms);
                s0.re[a * nlo + b] = r;
                s0.im[a * nlo + b] = im;
            }
        }
        // `w2, v = la.eigh(S0); w = sqrt(w2)` (`:47-48`).
        let mut ident = CTensor::zeros(nlo * nlo);
        for i in 0..nlo {
            ident.re[i * nlo + i] = 1.0;
        }
        let (w2, v) = lift_alg(
            pyscf_algebra::zeigh_gen(&s0, &ident, nlo),
            "Löwdin metric eigensolve",
        )?;
        let mut w = vec![0.0_f64; nlo];
        for (a, wn) in w2.iter().enumerate() {
            if !wn.is_finite() || *wn <= 0.0 {
                return Err(invalid(format!(
                    "KRKS+U C0: the local-orbital metric is singular (k = {k}, eig {a} = {wn:.3e}); \
                     check `minao_ref`"
                )));
            }
            w[a] = wn.sqrt();
        }
        // `S0_lowdin = (v/w)·v^H` (`:50`): column-scaled, `vw[i,a] = v[i,a]/w[a]`.
        let mut s0l = CTensor::zeros(nlo * nlo);
        for i in 0..nlo {
            for j in 0..nlo {
                let mut terms = Vec::with_capacity(nlo);
                for a in 0..nlo {
                    // v[i,a]/w[a], column-major v at a*nlo+i.
                    let (ar, ai) = (v.re[a * nlo + i] / w[a], v.im[a * nlo + i] / w[a]);
                    // conj(v[j,a]).
                    let (br, bi) = (v.re[a * nlo + j], -v.im[a * nlo + j]);
                    terms.push((ar * br - ai * bi, ar * bi + ai * br));
                }
                let (r, im) = csum(&terms);
                s0l.re[i * nlo + j] = r;
                s0l.im[i * nlo + j] = im;
            }
        }
        saa0.push(saa_row);
        c0_all.push(c0);
        w_all.push(w);
        v_all.push(v);
        s0l_all.push(s0l);
    }
    Ok(LocalOrbBase {
        nao,
        nlo,
        c0: c0_all,
        w: w_all,
        v: v_all,
        s0_lowdin: s0l_all,
        saa0,
    })
}

// ---------------------------------------------------------------------------
// `make_coeff` — `krkspu.py:61-87`.
// ---------------------------------------------------------------------------

/// The three derivative-integral tables `make_coeff` consumes, kept F-order
/// per component and read through
/// [`PbcIntorOutput::element`](pyscf_pbc_gto::pbc_intor::PbcIntorOutput::element).
pub struct IpTables {
    /// `cell.pbc_intor('int1e_ipovlp')`: `(3, nao, nao)`.
    pub saa: PbcIntorOutput,
    /// `intor_cross('int1e_ipovlp', cell, pcell)`: `(3, nao, nlo)`.
    pub sab: PbcIntorOutput,
    /// `intor_cross('int1e_ipovlp', pcell, cell)`: `(3, nlo, nao)`.
    pub sba: PbcIntorOutput,
}

/// Build the three tables for [`make_coeff`].
///
/// # Errors
/// A failed integral or a shape disagreement.
pub fn ip_tables(cell: &Cell, pcell: &Cell, kpts: &[[f64; 3]]) -> Result<IpTables, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let nlo = pcell.mol.nao_nr;
    let nkpts = kpts.len();
    let saa = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())
        .map_err(|e| invalid(format!("KRKS+U C1: int1e_ipovlp failed: {e}")))?;
    let sab =
        intor_cross("int1e_ipovlp", cell, pcell, kpts, PbcIntorOpts::default()).map_err(|e| {
            invalid(format!(
                "KRKS+U C1: intor_cross cell/pcell ipovlp failed: {e}"
            ))
        })?;
    let sba =
        intor_cross("int1e_ipovlp", pcell, cell, kpts, PbcIntorOpts::default()).map_err(|e| {
            invalid(format!(
                "KRKS+U C1: intor_cross pcell/cell ipovlp failed: {e}"
            ))
        })?;
    for (name, t, ni, nj) in [
        ("sAA_ip1", &saa, nao, nao),
        ("sAB_ip1", &sab, nao, nlo),
        ("sBA_ip1", &sba, nlo, nao),
    ] {
        if t.nkpts() != nkpts || t.ni != ni || t.nj != nj || saa.comp != 3 {
            return Err(invalid(format!(
                "KRKS+U C1: {name} has nkpts/ni/nj/comp = {}/{}/{}/{} for {nkpts}/{ni}/{nj}/3",
                t.nkpts(),
                t.ni,
                t.nj,
                t.comp,
            )));
        }
    }
    Ok(IpTables { saa, sab, sba })
}

/// `make_coeff(atm_id)` — `krkspu.py:61-87`: the first-order local orbitals
/// for one displaced atom.
///
/// `(p0, p1)` is the atom's AO range in `cell`, `(q0, q1)` its range in the
/// MINAO reference cell (`ao_slice[atm_id,2:]`, `minao_slice[atm_id,2:]` —
/// the 4-tuple 18-18 widened `aoslice_by_atom` to). Returns the three
/// Cartesian components, each ROW-MAJOR `nao × nlo`.
///
/// The body differentiates the **Löwdin symmetric orthogonalisation**:
/// `S^{-1/2}`'s derivative is not `-½ S^{-3/2} ∂S` — `S` and `∂S` do not
/// commute — so upstream solves the Sylvester-type relation in the eigenbasis
/// of `S` (`:78-79`, `S1 /= (w[:,None] + w)`), transcribed literally below.
///
/// # Errors
/// A failed linear solve or an out-of-range atom/slice.
#[allow(clippy::too_many_arguments)]
pub fn make_coeff(
    base: &LocalOrbBase,
    k: usize,
    p0: usize,
    p1: usize,
    q0: usize,
    q1: usize,
    tables: &IpTables,
) -> Result<[CTensor; 3], PyscfRsError> {
    let (nao, nlo) = (base.nao, base.nlo);
    if k >= base.nkpts() {
        return Err(PbcGradError::ShapeMismatch {
            expected: base.nkpts() as usize,
            got: k,
        }
        .into());
    }
    if p1 > nao || q1 > nlo || p0 > p1 || q0 > q1 {
        return Err(invalid(format!(
            "KRKS+U C1: atom slices (p0,p1,q0,q1) = ({p0},{p1},{q0},{q1}) exceed nao/nlo = {nao}/{nlo}"
        )));
    }
    let w = &base.w[k];
    let v = &base.v[k];
    // C0 column-major: c0(i,j) at i+j*nao.
    let c0 = |i: usize, j: usize| -> (f64, f64) {
        let q = i + j * nao;
        (base.c0[k].re[q], base.c0[k].im[q])
    };
    let mut out_re = vec![vec![0.0_f64; nao * nlo]; 3];
    let mut out_im = vec![vec![0.0_f64; nao * nlo]; 3];
    for n in 0..3 {
        // `sAA1[p0:p1,:] -= sAA_ip1[n,p0:p1]`; `sAA1[:,p0:p1] -= conj`
        // (`:68-70`). Row-major `nao × nao`.
        let mut saa1_re = vec![0.0_f64; nao * nao];
        let mut saa1_im = vec![0.0_f64; nao * nao];
        for i in p0..p1 {
            for j in 0..nao {
                let (r, im) = tables.saa.element(k, n, i, j);
                saa1_re[i * nao + j] = oracle_sum(&[saa1_re[i * nao + j], -r]);
                saa1_im[i * nao + j] = oracle_sum(&[saa1_im[i * nao + j], -im]);
            }
        }
        for j in p0..p1 {
            for i in 0..nao {
                // conj(ip[n,j,i]).
                let (r, im) = tables.saa.element(k, n, j, i);
                saa1_re[i * nao + j] = oracle_sum(&[saa1_re[i * nao + j], -r]);
                saa1_im[i * nao + j] = oracle_sum(&[saa1_im[i * nao + j], im]);
            }
        }
        // `sAB1[p0:p1,:] -= sAB_ip1[n,p0:p1]`;
        // `sAB1[:,q0:q1] -= sBA_ip1[n,q0:q1]^H` (`:71-73`).
        let mut sab1_re = vec![0.0_f64; nao * nlo];
        let mut sab1_im = vec![0.0_f64; nao * nlo];
        for i in p0..p1 {
            for j in 0..nlo {
                let (r, im) = tables.sab.element(k, n, i, j);
                sab1_re[i * nlo + j] = oracle_sum(&[sab1_re[i * nlo + j], -r]);
                sab1_im[i * nlo + j] = oracle_sum(&[sab1_im[i * nlo + j], -im]);
            }
        }
        for j in q0..q1 {
            for i in 0..nao {
                // conj(sBA[n,j,i]).
                let (r, im) = tables.sba.element(k, n, j, i);
                sab1_re[i * nlo + j] = oracle_sum(&[sab1_re[i * nlo + j], -r]);
                sab1_im[i * nlo + j] = oracle_sum(&[sab1_im[i * nlo + j], im]);
            }
        }
        // `S1 = C0^H·sAB1 + hc - C0^H·sAA1·C0` (`:75-77`),
        // row-major `nlo × nlo`. `A = C0^H·sAB1`, `B = C0^H·sAA1·C0`.
        // ORDER MATTERS: the Hermitian completion applies to `A` ONLY —
        // `S1 = (A + A^H) − B` (`:76` runs before `:77`). `B` is not
        // Hermitian (the `sAA1` assembly is a derivative block, not a
        // metric), so completing the difference would add a spurious `−B^H`.
        let mut a_re = vec![0.0_f64; nlo * nlo];
        let mut a_im = vec![0.0_f64; nlo * nlo];
        let mut b_re = vec![0.0_f64; nlo * nlo];
        let mut b_im = vec![0.0_f64; nlo * nlo];
        for a in 0..nlo {
            for b in 0..nlo {
                // A[a,b] = Σ_i conj(C0[i,a])·sAB1[i,b].
                let mut ta = Vec::with_capacity(nao);
                for i in 0..nao {
                    let (cr, ci) = c0(i, a);
                    let (dr, di) = (sab1_re[i * nlo + b], sab1_im[i * nlo + b]);
                    ta.push((cr * dr + ci * di, cr * di - ci * dr));
                }
                let (aar, aai) = csum(&ta);
                a_re[a * nlo + b] = aar;
                a_im[a * nlo + b] = aai;
                // B[a,b] = Σ_{i,q} conj(C0[i,a])·sAA1[i,q]·C0[q,b].
                let mut tb = Vec::with_capacity(nao * nao);
                for i in 0..nao {
                    let (cr, ci) = c0(i, a);
                    for q in 0..nao {
                        let (sr, si) = (saa1_re[i * nao + q], saa1_im[i * nao + q]);
                        // conj(C0[i,a])·sAA1[i,q].
                        let (ur, ui) = (cr * sr + ci * si, cr * si - ci * sr);
                        let (er, ei) = c0(q, b);
                        tb.push((ur * er - ui * ei, ur * ei + ui * er));
                    }
                }
                let (bbr, bbi) = csum(&tb);
                b_re[a * nlo + b] = bbr;
                b_im[a * nlo + b] = bbi;
            }
        }
        let mut s1_re = vec![0.0_f64; nlo * nlo];
        let mut s1_im = vec![0.0_f64; nlo * nlo];
        for a in 0..nlo {
            for b in 0..nlo {
                // (A + A^H)[a,b] − B[a,b]; A^H[a,b] = conj(A[b,a]).
                s1_re[a * nlo + b] =
                    oracle_sum(&[a_re[a * nlo + b], a_re[b * nlo + a], -b_re[a * nlo + b]]);
                s1_im[a * nlo + b] =
                    oracle_sum(&[a_im[a * nlo + b], -a_im[b * nlo + a], -b_im[a * nlo + b]]);
            }
        }
        // `S1 = V^H·(−S1)·V; S1 /= (w[:,None] + w)` (`:78-79`).
        // T1[a,b] = −Σ_{p,q} conj(v[p,a])·S1[p,q]·v[q,b] / (w[a]+w[b]).
        // `v` column-major: v(p,a) at a*nlo+p.
        let mut t1_re = vec![0.0_f64; nlo * nlo];
        let mut t1_im = vec![0.0_f64; nlo * nlo];
        for a in 0..nlo {
            for b in 0..nlo {
                let mut terms = Vec::with_capacity(nlo * nlo);
                for p in 0..nlo {
                    let (cr, ci) = (v.re[a * nlo + p], -v.im[a * nlo + p]);
                    for q in 0..nlo {
                        let (sr, si) = (s1_re[p * nlo + q], s1_im[p * nlo + q]);
                        let (ur, ui) = (cr * sr - ci * si, cr * si + ci * sr);
                        let (er, ei) = (v.re[b * nlo + q], v.im[b * nlo + q]);
                        terms.push((ur * er - ui * ei, ur * ei + ui * er));
                    }
                }
                let (rr, ri) = csum(&terms);
                let denom = oracle_sum(&[w[a], w[b]]);
                t1_re[a * nlo + b] = -rr / denom;
                t1_im[a * nlo + b] = -ri / denom;
            }
        }
        // `S1_lowdin = (V/w)·T1·(V/w)^H` (`:80-81`).
        let mut s1l_re = vec![0.0_f64; nlo * nlo];
        let mut s1l_im = vec![0.0_f64; nlo * nlo];
        for i in 0..nlo {
            for j in 0..nlo {
                let mut terms = Vec::with_capacity(nlo * nlo);
                for a in 0..nlo {
                    for b in 0..nlo {
                        // (V/w)[i,a] = v[i,a]/w[a].
                        let (ar, ai) = (v.re[a * nlo + i] / w[a], v.im[a * nlo + i] / w[a]);
                        let (tr, ti) = (t1_re[a * nlo + b], t1_im[a * nlo + b]);
                        let (ur, ui) = (ar * tr - ai * ti, ar * ti + ai * tr);
                        // conj((V/w)[j,b]).
                        let (er, ei) = (v.re[b * nlo + j] / w[b], -v.im[b * nlo + j] / w[b]);
                        terms.push((ur * er - ui * ei, ur * ei + ui * er));
                    }
                }
                let (rr, ri) = csum(&terms);
                s1l_re[i * nlo + j] = rr;
                s1l_im[i * nlo + j] = ri;
            }
        }
        // `C1_minao = cho_solve(sAA_cd, sAB1 − sAA1·C0)` (`:83`): D then one
        // `zsolve_linear` per column against row-major `saa0`.
        // D[i,j] = sAB1[i,j] − Σ_q sAA1[i,q]·C0[q,j].
        let mut d_re = vec![0.0_f64; nao * nlo];
        let mut d_im = vec![0.0_f64; nao * nlo];
        for i in 0..nao {
            for j in 0..nlo {
                let mut terms = Vec::with_capacity(nao);
                for q in 0..nao {
                    let (sr, si) = (saa1_re[i * nao + q], saa1_im[i * nao + q]);
                    let (er, ei) = c0(q, j);
                    terms.push((sr * er - si * ei, sr * ei + si * er));
                }
                let (pr, pi) = csum(&terms);
                d_re[i * nlo + j] = oracle_sum(&[sab1_re[i * nlo + j], -pr]);
                d_im[i * nlo + j] = oracle_sum(&[sab1_im[i * nlo + j], -pi]);
            }
        }
        let mut c1m_re = vec![0.0_f64; nao * nlo];
        let mut c1m_im = vec![0.0_f64; nao * nlo];
        for j in 0..nlo {
            let rhs = CTensor::from_planes(
                (0..nao).map(|i| d_re[i * nlo + j]).collect(),
                (0..nao).map(|i| d_im[i * nlo + j]).collect(),
            );
            let x = lift_alg(
                pyscf_algebra::zsolve_linear(&base.saa0[k], &rhs, nao),
                "C1 cho_solve",
            )?;
            for i in 0..nao {
                c1m_re[i * nlo + j] = x.re[i];
                c1m_im[i * nlo + j] = x.im[i];
            }
        }
        // `C1 = C1_minao·S0_lowdin + C0·S1_lowdin` (`:84-85`), row-major.
        let s0l = &base.s0_lowdin[k];
        for i in 0..nao {
            for j in 0..nlo {
                let mut terms = Vec::with_capacity(2 * nlo);
                for t in 0..nlo {
                    let (ar, ai) = (c1m_re[i * nlo + t], c1m_im[i * nlo + t]);
                    let (br, bi) = (s0l.re[t * nlo + j], s0l.im[t * nlo + j]);
                    terms.push((ar * br - ai * bi, ar * bi + ai * br));
                    let (cr, ci) = c0(i, t);
                    let (dr, di) = (s1l_re[t * nlo + j], s1l_im[t * nlo + j]);
                    terms.push((cr * dr - ci * di, cr * di + ci * dr));
                }
                let (rr, ri) = csum(&terms);
                out_re[n][i * nlo + j] = rr;
                out_im[n][i * nlo + j] = ri;
            }
        }
    }
    Ok([
        CTensor::from_planes(
            std::mem::take(&mut out_re[0]),
            std::mem::take(&mut out_im[0]),
        ),
        CTensor::from_planes(
            std::mem::take(&mut out_re[1]),
            std::mem::take(&mut out_im[1]),
        ),
        CTensor::from_planes(
            std::mem::take(&mut out_re[2]),
            std::mem::take(&mut out_im[2]),
        ),
    ])
}

// ---------------------------------------------------------------------------
// `_hubbard_U_deriv1` — `krkspu.py:89-133`.
// ---------------------------------------------------------------------------

/// Check the three upstream asserts (`krkspu.py:90-92`, mirrored by
/// `krks_stress.py:364-366` on the strain path): no linear-response
/// perturbation, no caller-supplied local orbitals, a named MINAO reference.
pub(crate) fn check_cfg(cfg: &HubbardU) -> Result<(), PyscfRsError> {
    if !cfg.alpha.is_empty() {
        return Err(invalid(
            "KRKS+U gradient: linear-response alpha is not supported (krkspu.py:90 asserts mf.alpha is None)",
        ));
    }
    if cfg.c_ao_lo.is_some() {
        return Err(invalid(
            "KRKS+U gradient: caller-supplied C_ao_lo is not supported (krkspu.py:91 asserts mf.C_ao_lo is None)",
        ));
    }
    if cfg.minao_ref.is_empty() {
        return Err(invalid(
            "KRKS+U gradient: a named minao_ref is required (krkspu.py:92)",
        ));
    }
    Ok(())
}

/// `_hubbard_U_deriv1(mf, dm, kpts)` — `krkspu.py:89-133`: the Hubbard-`U`
/// nuclear gradient for a closed-shell k-point density.
///
/// `dm` is the single row-major density set (`dm[k]`, `nao × nao`), held
/// fixed — the derivative is exact for ANY fixed Hermitian `dm`, so the gate
/// feeds a converged (or 1-cycle, as upstream's own test does) density and
/// finite-differences `E_U` at fixed `dm`. `U` values come from `cfg` in eV
/// via [`set_u`](pyscf_pbc_dft::kspu::set_u); the `1/nkpts` weighting is
/// `:116`.
///
/// 17-08's D-17-08-02 applies here and is not re-litigated: DFT+U rotates
/// **no** projectors under k-point symmetry — if a rotation appears in this
/// body, it is a bug, not a port.
///
/// # Errors
/// A refused configuration, a shape disagreement, or a failed integral.
pub fn hubbard_u_deriv1(
    cell: &Cell,
    dm: &KMats,
    kpts: &[[f64; 3]],
    cfg: &HubbardU,
) -> Result<Gradient, PyscfRsError> {
    use pyscf_pbc_dft::kspu::{make_minao_lo, reference_cell, set_u};

    check_cfg(cfg)?;
    let nao = cell.mol.nao_nr;
    let nkpts = kpts.len();
    if nkpts == 0 || dm.len() != nkpts {
        return Err(PbcGradError::ShapeMismatch {
            expected: nkpts,
            got: dm.len(),
        }
        .into());
    }
    for (k, m) in dm.iter().enumerate() {
        if m.re.len() != nao * nao || m.im.len() != nao * nao {
            return Err(PbcGradError::ShapeMismatch {
                expected: nao * nao,
                got: m.re.len().min(m.im.len()),
            }
            .into());
        }
        let _ = k;
    }

    // Construct orthogonal MINAO local orbitals (`:101-102`).
    let pcell = lift_dft(reference_cell(cell, &cfg.minao_ref))?;
    let c_lo = lift_dft(make_minao_lo(cell, &pcell, kpts))?;
    let resolved = lift_dft(set_u(&pcell, cfg))?;
    if resolved.indices.is_empty() {
        return Ok(vec![[0.0; 3]; cell.natm]);
    }
    // `C0 = [C_k[:,U_idx_stack]]` (`:105`), column-major.
    let stack: Vec<usize> = resolved.indices.iter().flatten().copied().collect();
    let nu = stack.len();
    let nlo = c_lo[0].re.len() / nao;
    for &u in &stack {
        if u >= nlo {
            return Err(invalid(format!(
                "KRKS+U gradient: Hubbard index {u} exceeds nlo = {nlo}"
            )));
        }
    }
    let c0u = |ck: &CTensor, i: usize, a: usize| -> (f64, f64) {
        let q = i + stack[a] * nao;
        (ck.re[q], ck.im[q])
    };

    // `ovlp0`, row-major; `ovlp1`, F-order per component via `.element`.
    let ovlp0_f = pyscf_pbc_gto::get_ovlp(cell, kpts)
        .map_err(|e| invalid(format!("KRKS+U U1: get_ovlp failed: {e}")))?;
    let ovlp1 = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())
        .map_err(|e| invalid(format!("KRKS+U U1: int1e_ipovlp failed: {e}")))?;
    let mut s0: Vec<CTensor> = Vec::with_capacity(nkpts);
    for m in &ovlp0_f {
        s0.push(CTensor::from_planes(
            f_to_c(&m.re, nao, nao),
            f_to_c(&m.im, nao, nao),
        ));
    }

    let flo = first_order_local_orbitals(cell, &cfg.minao_ref, kpts)?;
    if flo.nlo != nlo {
        return Err(PbcGradError::ShapeMismatch {
            expected: nlo,
            got: flo.nlo,
        }
        .into());
    }
    let tables = ip_tables(cell, &pcell, kpts)?;
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
        .map_err(|e| invalid(format!("KRKS+U U1: aoslice failed: {e}")))?;
    let mslices = pyscf_gto::aoslice_by_atom(&pcell.mol)
        .map_err(|e| invalid(format!("KRKS+U U1: MINAO aoslice failed: {e}")))?;
    let natm = cell.natm;
    if slices.len() != natm || mslices.len() != natm {
        return Err(invalid(format!(
            "KRKS+U U1: aoslice counts {}/{} for natm = {natm}",
            slices.len(),
            mslices.len()
        )));
    }

    let weight = 1.0 / nkpts as f64;
    let mut de = vec![[0.0_f64; 3]; natm];
    for (atm_id, &(_, _, p0, p1)) in slices.iter().enumerate() {
        let (_, _, q0, q1) = mslices[atm_id];
        // Per-k first-order orbitals for this atom: row-major `(3, nao, nu)`.
        let mut c1: Vec<[CTensor; 3]> = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let full = make_coeff(&flo, k, p0, p1, q0, q1, &tables)?;
            c1.push(std::array::from_fn(|n| {
                let mut re = vec![0.0_f64; nao * nu];
                let mut im = vec![0.0_f64; nao * nu];
                for i in 0..nao {
                    for (b, &u) in stack.iter().enumerate() {
                        re[i * nu + b] = full[n].re[i * nlo + u];
                        im[i * nu + b] = full[n].im[i * nlo + u];
                    }
                }
                CTensor::from_planes(re, im)
            }));
        }
        for k in 0..nkpts {
            // `C_inv = C0^H · S0` (`:109`), `(nu × nao)` row-major.
            let mut cinv_re = vec![0.0_f64; nu * nao];
            let mut cinv_im = vec![0.0_f64; nu * nao];
            for a in 0..nu {
                for b in 0..nao {
                    let mut terms = Vec::with_capacity(nao);
                    for i in 0..nao {
                        let (cr, ci) = c0u(&c_lo[k], i, a);
                        let (sr, si) = (s0[k].re[i * nao + b], s0[k].im[i * nao + b]);
                        terms.push((cr * sr + ci * si, cr * si - ci * sr));
                    }
                    let (rr, ri) = csum(&terms);
                    cinv_re[a * nao + b] = rr;
                    cinv_im[a * nao + b] = ri;
                }
            }
            // `T = C_inv · dm[k]` (`:124` inner factor), `(nu × nao)`.
            let mut t_re = vec![0.0_f64; nu * nao];
            let mut t_im = vec![0.0_f64; nu * nao];
            for a in 0..nu {
                for j in 0..nao {
                    let mut terms = Vec::with_capacity(nao);
                    for b in 0..nao {
                        let (cr, ci) = (cinv_re[a * nao + b], cinv_im[a * nao + b]);
                        let (dr, di) = (dm[k].re[b * nao + j], dm[k].im[b * nao + j]);
                        terms.push((cr * dr - ci * di, cr * di + ci * dr));
                    }
                    let (rr, ri) = csum(&terms);
                    t_re[a * nao + j] = rr;
                    t_im[a * nao + j] = ri;
                }
            }
            // `dm_deriv0 = T · C_inv^H` (`:110`), `(nu × nu)`, full complex —
            // `:132` takes `.real` only AFTER the `P1·P0` product.
            let mut p0_re = vec![0.0_f64; nu * nu];
            let mut p0_im = vec![0.0_f64; nu * nu];
            for a in 0..nu {
                for b in 0..nu {
                    let mut terms = Vec::with_capacity(nao);
                    for j in 0..nao {
                        let (tr, ti) = (t_re[a * nao + j], t_im[a * nao + j]);
                        let (cr, ci) = (cinv_re[b * nao + j], cinv_im[b * nao + j]);
                        terms.push((tr * cr + ti * ci, ti * cr - tr * ci));
                    }
                    let (rr, ri) = csum(&terms);
                    p0_re[a * nu + b] = rr;
                    p0_im[a * nu + b] = ri;
                }
            }
            // `SC1 = S0·C1 + S1·C0` in einsum form (`:121-123`).
            // SC1[x,p,i] row-major `(3, nao, nu)`.
            let mut sc1_re = vec![0.0_f64; 3 * nao * nu];
            let mut sc1_im = vec![0.0_f64; 3 * nao * nu];
            for x in 0..3 {
                for p in 0..nao {
                    for i in 0..nu {
                        // Σ_q S[p,q]·C1[x,q,i].
                        let mut terms = Vec::with_capacity(nao + 2 * (p1 - p0));
                        for q in 0..nao {
                            let (sr, si) = (s0[k].re[p * nao + q], s0[k].im[p * nao + q]);
                            let (cr, ci) = (c1[k][x].re[q * nu + i], c1[k][x].im[q * nu + i]);
                            terms.push((sr * cr - si * ci, sr * ci + si * cr));
                        }
                        // −Σ_{q∈atom} conj(ip[x,q,p])·C0[q,i] (`:122`).
                        for q in p0..p1 {
                            let (r, im) = ovlp1.element(k, x, q, p);
                            let (er, ei) = c0u(&c_lo[k], q, i);
                            // conj(ip)·C0.
                            terms.push((-(r * er + im * ei), -(im * er - r * ei)));
                        }
                        // p∈atom: −Σ_q ip[x,p,q]·C0[q,i] (`:123`).
                        if (p0..p1).contains(&p) {
                            for q in 0..nao {
                                let (r, im) = ovlp1.element(k, x, p, q);
                                let (er, ei) = c0u(&c_lo[k], q, i);
                                terms.push((-(r * er - im * ei), -(r * ei + im * er)));
                            }
                        }
                        let (rr, ri) = csum(&terms);
                        sc1_re[(x * nao + p) * nu + i] = rr;
                        sc1_im[(x * nao + p) * nu + i] = ri;
                    }
                }
            }
            // `dm_deriv1 = T·SC1` (`:124`), `(3, nu, nu)`, full complex.
            // P1[x,a,b] = Σ_j T[a,j]·SC1[x,j,b].
            let mut p1_re = vec![0.0_f64; 3 * nu * nu];
            let mut p1_im = vec![0.0_f64; 3 * nu * nu];
            for x in 0..3 {
                for a in 0..nu {
                    for b in 0..nu {
                        let mut terms = Vec::with_capacity(nao);
                        for j in 0..nao {
                            let (tr, ti) = (t_re[a * nao + j], t_im[a * nao + j]);
                            let q = (x * nao + j) * nu + b;
                            let (sr, si) = (sc1_re[q], sc1_im[q]);
                            terms.push((tr * sr - ti * si, tr * si + ti * sr));
                        }
                        let (rr, ri) = csum(&terms);
                        p1_re[(x * nu + a) * nu + b] = rr;
                        p1_im[(x * nu + a) * nu + b] = ri;
                    }
                }
            }
            // `:126-132` over the site blocks:
            // `weight * (U/2) * (2·Re tr P1 − 2·Re tr(P1·P0))` — the `*2`
            // is upstream's `*2 for P1+P1.T`.
            let mut row = [0.0_f64; 3];
            let mut off = 0usize;
            for (idx, val) in resolved.indices.iter().zip(resolved.u_val.iter()) {
                let n = idx.len();
                let wgt = weight * val * 0.5;
                for x in 0..3 {
                    let mut d_terms = Vec::with_capacity(n);
                    let mut p_terms = Vec::with_capacity(n * n);
                    for ai in 0..n {
                        let p = off + ai;
                        // Re P1[x,p,p] (`einsum('xii->x', P1).real`).
                        d_terms.push(p1_re[(x * nu + p) * nu + p]);
                        for bi in 0..n {
                            let q = off + bi;
                            // Re(P1[x,p,q]·P0[q,p]) (`einsum('xij,ji->x', P1, P0).real`).
                            let (ar, ai_) =
                                (p1_re[(x * nu + p) * nu + q], p1_im[(x * nu + p) * nu + q]);
                            let (br, bi_) = (p0_re[q * nu + p], p0_im[q * nu + p]);
                            p_terms.push(ar * br - ai_ * bi_);
                        }
                    }
                    let term = oracle_sum(&[
                        wgt * 2.0 * oracle_sum(&d_terms),
                        -(wgt * 2.0 * oracle_sum(&p_terms)),
                    ]);
                    row[x] = oracle_sum(&[row[x], term]);
                }
                off += n;
            }
            for x in 0..3 {
                de[atm_id][x] = oracle_sum(&[de[atm_id][x], row[x]]);
            }
        }
    }
    Ok(de)
}

// ---------------------------------------------------------------------------
// `Gradients(krks_grad.Gradients)` — `krkspu.py:135-142`.
// ---------------------------------------------------------------------------

/// KRKS+U k-point nuclear gradient — `pbc/grad/krkspu.py`'s `Gradients` class.
///
/// 18-07's [`KrksGradients`] with `extra_force` extended by the `U` term
/// (`krkspu.py:140-142`). `d_e_u` is computed once at construction (from the
/// converged orbitals, exactly as upstream's `get_veff` stashes `_dE_U`
/// before the assembly runs) and added per atom in [`kernel`](Self::kernel),
/// which is where upstream's `extra_force` sits.
pub struct KrkspuGradients<'a> {
    base: KrksGradients<'a>,
    d_e_u: Gradient,
    atmlst: Option<Vec<usize>>,
}

impl<'a> KrkspuGradients<'a> {
    /// Build from converged orbitals plus the Hubbard configuration. The
    /// density is rebuilt from the orbitals through
    /// [`make_rdm1`](pyscf_pbc_scf::krdm::make_rdm1) — the same call
    /// [`KrksGradients::new`](crate::krks::KrksGradients::new) makes — so the
    /// `U` term and the base see one density.
    ///
    /// # Errors
    /// Whatever [`KrksGradients::new`](crate::krks::KrksGradients::new) or
    /// [`hubbard_u_deriv1`] report.
    pub fn new(
        mf: &'a pyscf_pbc_dft::krks::Krks,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
        u: &HubbardU,
    ) -> Result<Self, PyscfRsError> {
        let nao = mf.cell().mol.nao_nr;
        let dm0 = vec![make_rdm1(&mo_coeff, &mo_occ, nao)];
        let d_e_u = if u.sites.is_empty() {
            vec![[0.0; 3]; mf.cell().natm]
        } else {
            hubbard_u_deriv1(mf.cell(), &dm0[0], mf.kpts(), u)?
        };
        let base = KrksGradients::new(mf, mo_energy, mo_coeff, mo_occ)?;
        Ok(Self {
            base,
            d_e_u,
            atmlst: None,
        })
    }

    /// Atom subset (`de = de[atmlst]`). `None` (default) is all atoms.
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Result<Self, PyscfRsError> {
        self.base = self.base.with_atmlst(atmlst.clone())?;
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    /// `kernel` — 18-07's kernel plus the `U` rows (`krkspu.py:140-142`).
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let mut out = self.base.kernel()?;
        let list: Vec<usize> = match &self.atmlst {
            Some(l) => l.clone(),
            None => (0..self.d_e_u.len()).collect(),
        };
        if out.len() != list.len() {
            return Err(invalid(format!(
                "KRKS+U kernel: base returned {} rows for {} atoms",
                out.len(),
                list.len()
            )));
        }
        for (row, &ia) in out.iter_mut().zip(list.iter()) {
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], self.d_e_u[ia][x]]);
            }
        }
        Ok(out)
    }
}

impl<'a> Gradients for KrkspuGradients<'a> {
    fn cell(&self) -> &Cell {
        self.base.cell()
    }

    fn kpts(&self) -> &[[f64; 3]] {
        self.base.kpts()
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.base.grad_elec()
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.base.grad_nuc()
    }
}
