//! KRHF k-point analytic nuclear gradient — `pyscf/pbc/grad/krhf.py` (418 l).
//!
//! The k-point restricted Hartree–Fock nuclear gradient, and the base class
//! every other k-point gradient in Phase 18 inherits (`kuhf.py:86`,
//! `krks.py:119` via `rhf_grad.Gradients`). First plan in the phase that
//! produces a number a user would ask for, and the first one `verify_fd`
//! gates end to end (Gate B).
//!
//! # Upstream correspondence (`krhf.py` line → here)
//!
//! | upstream | here |
//! |---|---|
//! | `get_hcore` `:88-113` (kinetic + PP-local on the uniform grid) | [`get_hcore`] |
//! | `hcore_generator` `:117-147` (per-atom `(3,nkpts,nao,nao)` closure) | [`KrhfGradients::hcore_generator`] via [`hcore_deriv_matrices`] |
//! | `grad_elec` `:34-74` (three contractions + scopes) | [`Gradients::grad_elec`] for [`KrhfGradients`] |
//! | `get_ovlp` `:114-115` | [`KrhfGradients::get_ovlp`] |
//! | `get_jk`/`get_j`/`get_k` `:260-284` (FFTDF `get_jk_e1`) | [`KrhfGradients::get_jk`] |
//! | `get_veff` `:219-222` (`vj - vk·.5`) | [`KrhfGradients::veff`] |
//! | `make_rdm1e` `:224-228` | [`make_rdm1e_kpts`] |
//! | `grad_nuc` `:286-288` (Ewald) | [`KrhfGradients::grad_nuc`] |
//! | `kernel` `:380-397` (`grad_elec + grad_nuc`) | [`KrhfGradients::kernel`] |
//! | `as_scanner` `:300-320,371` + `SCF_GradScanner` `:322-342` | [`crate::scanner`] (`as_scanner`, `SCF_GradScanner`) |
//!
//! The driver surface (`as_scanner`, `optimizer`) lives in 18-19:
//! [`KrhfGradients`] implements the trait's `as_scanner` energy seam over the
//! shared-state scanner, and inherits the `optimizer` refusal unchanged
//! (`'geometric'` raising is upstream's behaviour — 18-CONTEXT §1.7 — not a
//! gap).
//!
//! # Index orders (18-CONTEXT traps 9 + `krhf.py:144-145`)
//!
//! * This port's `mo_coeff` is COLUMN-MAJOR `nao × nmo`
//!   (`pyscf-pbc-scf/src/types.rs:9-10`); every `nao × nao` matrix below is
//!   ROW-MAJOR (`[i·nao+j]`), the `KScfResult` convention.
//! * The density is indexed **`ji`, not `ij`**, on all three `grad_elec`
//!   terms: `einsum('xkij,kji->x', h1ao, dm0)` (`krhf.py:63-66`). The doc
//!   comment on each contraction helper repeats this, and a deliberately
//!   non-symmetric DM gates it.
//! * `hcore_generator`'s second subtraction (`:145`) is the **conjugate
//!   transpose** of the same `h1` block, not a second slice:
//!   `hcore[x,k,i,j∈A] -= conj(h1[x,k,j,i])`. [`assemble_hcore_deriv`] is a
//!   pure function of its inputs so a non-Hermitian test matrix gates the
//!   `.conj()`.
//!
//! # Reductions and kernels (ALG-06, D-PBC-17, D-PBC-31)
//!
//! Every reduction routes through `pyscf_algebra::oracle_sum` over a
//! materialised partial buffer — never a bare `+=` over grid or AO terms —
//! so results are bit-identical for any `RAYON_NUM_THREADS`.
//!
//! No CubeCL kernel is added here (ALG-06: `pyscf-pbc-grad` may not depend on
//! `cubecl-*` AT ALL; `xtask check-dependency-wall` enforces it). The CubeCL
//! manual (`INDEX.md` + `Cubecl_generics.md`, `Float` generics) was read
//! before writing; there is no device kernel in this file for it to apply
//! to. The one device computation consumed is the clause-10 shared
//! `(natm,3) ← (natm·3,ngrids)×(ngrids)` primitive
//! (`pyscf_kernels::pbc::multigrid_grad::contract_atom_grid`, itself a
//! generics-`Float` kernel) inside [`fused_local_contraction`].
//!
//! # The fused local-PP contraction (D-PBC-31 clause 7)
//!
//! Upstream's closure allocates `(3,nkpts,nao,nao)` per atom and fills it
//! with `vloc = einsum('gi,gj,g->ij', ao.conj(), ao, vloc_R)` at cost
//! `natm·3·nkpts·ngrids·nao²`, and `grad_elec:63` reduces the whole thing to
//! three numbers against `dm0`. But
//! `Σ_ij vloc[k,i,j]·dm0[k,j,i] = Σ_g vloc_R[g]·ρ_k[g] =
//! Σ_G conj(vloc_g[G])·ρ_k[G]` (Parseval), and ρ depends on neither `atm_id`
//! nor the Cartesian component. [`fused_local_contraction`] is that
//! contraction — `nkpts` value-AO evaluations, `nkpts` density builds, zero
//! inverse FFTs — and it is what [`Gradients::grad_elec`] USES.
//!
//! [`hcore_deriv_matrices`] (via [`KrhfGradients::hcore_generator`]) REMAINS
//! the public seam returning the matrix: `grad_elec:68` passes `locals()` to
//! `extra_force` and 18-08's DFT+U term reads it. A test asserts the two
//! agree at `measurements/parseval.md`'s number — G-space and real-space are
//! different summations, never bit-identity.

use pyscf_algebra::{CTensor, oracle_sum, select_backend};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_df::JkOpts;
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::types::{KDms, KMats};
use pyscf_pbc_scf::{Krhf, krdm::make_rdm1};

use crate::error::PbcGradError;
use crate::gradients::{EnergyScanner, GradMatrices, Gradient, Gradients};
use crate::scanner::{KrhfScannerConfig, ScfGradScanner};

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

fn df_err(context: &'static str, e: pyscf_pbc_df::PbcDfError) -> PyscfRsError {
    match e {
        pyscf_pbc_df::PbcDfError::Core(c) => c,
        other => PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "KRHF gradient ({context}): density fitting failed: {other}"
        ))),
    }
}

/// Upstream `cell._pseudo` truthiness: at least one atom carries a GTH
/// pseudopotential. Mirrors the gamma gradient's `has_pseudo`.
fn has_pseudo(cell: &Cell) -> bool {
    (0..cell.natm).any(|ia| cell.atom_pseudo(ia).is_some())
}

/// Number of k-points, treating an empty list as the single gamma point
/// (the `pbc_intor` convention).
fn nkpts_of(kpts: &[[f64; 3]]) -> usize {
    if kpts.is_empty() { 1 } else { kpts.len() }
}

/// Take one half of an 18-04 gradient JK result (`[x][set][k]`, single set)
/// as `[x][k]` row-major matrices. Rejects a wrong `(x, set, k)` shape by
/// name — a silent squeeze would misattribute a builder regression.
fn take_grad_mats(
    mats: pyscf_pbc_df::fft_jk_grad::GradMats,
    nkpts: usize,
    nao: usize,
    what: &'static str,
) -> Result<GradMatrices, PyscfRsError> {
    if mats.len() != 3 || mats.iter().any(|s| s.len() != 1 || s[0].len() != nkpts) {
        return Err(invalid(format!(
            "KRHF get_jk: {what} has the wrong (x, set, k) shape for nkpts = {nkpts}"
        )));
    }
    Ok(std::array::from_fn(|x| {
        mats[x][0]
            .iter()
            .map(|m| {
                debug_assert_eq!(m.re.len(), nao * nao);
                m.clone()
            })
            .collect()
    }))
}

// ---------------------------------------------------------------------------
// `get_hcore` — krhf.py:88-113
// ---------------------------------------------------------------------------

/// Everything `get_hcore` and `hcore_generator` share that does not depend on
/// the atom: the kinetic-derivative matrices, the uniform-grid fields, and
/// the grid itself. Built ONCE per gradient (upstream builds `h1` once in
/// `hcore_generator`, outside the closure); the per-atom closure only reads
/// it.
pub struct HcoreTables {
    /// `int1e_ipkin` + the PP-local term, `[x][k]` row-major `nao × nao`.
    pub h1: GradMatrices,
    /// Structure factor, `(natm, ngrids)` row-major split planes.
    pub si_re: Vec<f64>,
    /// Structure factor, imaginary plane.
    pub si_im: Vec<f64>,
    /// Local-PP kernel `Vloc(G)`, `(natm, ngrids)` real.
    pub vlocg: Vec<f64>,
    /// G-vectors, `(ngrids, 3)`.
    pub gv: Vec<[f64; 3]>,
    /// Uniform-grid coordinates (upstream's `coords`, `wrap_around=True`).
    pub coords: Vec<[f64; 3]>,
    /// FFT mesh.
    pub mesh: [usize; 3],
    /// Grid-point count.
    pub ngrids: usize,
    /// Atom count.
    pub natm: usize,
    /// AO count.
    pub nao: usize,
    /// `true` when every k-point is gamma (the `.real` branch).
    pub all_gamma: bool,
}

/// `get_hcore(cell, kpts)` — `krhf.py:88-113`.
///
/// `h1 = cell.pbc_intor('int1e_ipkin', kpts=kpts)` (18-03 Task 2 verified it),
/// then — for a pseudopotential cell — the local term assembled on the
/// uniform grid from `get_vlocG` / `get_alphas` (`pbc/gto/pseudo/pp.py`),
/// inverse-FFT'd and contracted against `eval_ao_kpts(deriv=1)`.
///
/// `krhf.py:112` is `else: raise NotImplementedError` — an all-electron
/// periodic cell has NO `get_hcore` gradient upstream. This refuses
/// identically ([`PbcGradError::NotYetImplemented`]); it never falls back to
/// `int1e_ipnuc` (which is the NUCLEAR-attraction derivative, a different
/// operator that happens to be nearby in the layout table).
///
/// # Errors
/// * [`PbcGradError::NotYetImplemented`] for all-electron cells.
/// * Propagates the integrals, the grid builds, the AO evaluations and the FFT.
pub fn get_hcore(cell: &Cell, kpts: &[[f64; 3]]) -> Result<GradMatrices, PyscfRsError> {
    Ok(precompute_hcore(cell, kpts)?.h1)
}

/// Shared build behind [`get_hcore`] and [`hcore_deriv_matrices`].
///
/// # Errors
/// As [`get_hcore`].
pub fn precompute_hcore(cell: &Cell, kpts: &[[f64; 3]]) -> Result<HcoreTables, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let natm = cell.natm;
    let nkpts = nkpts_of(kpts);
    if !has_pseudo(cell) {
        return Err(PbcGradError::NotYetImplemented {
            phase: 18,
            what: "KRHF get_hcore for all-electron cells (upstream krhf.py:111 \
                   raises NotImplementedError; never falls back to int1e_ipnuc)",
        }
        .into());
    }
    // krhf.py:94 — F-order `(c,i,j)` per k becomes row-major `[x][k]`.
    let ipkin = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipkin", kpts, Default::default())?;
    if ipkin.comp != 3 || ipkin.ni != nao || ipkin.nj != nao || ipkin.nkpts() != nkpts {
        return Err(invalid(format!(
            "KRHF get_hcore: int1e_ipkin has comp = {}, ni/nj = {}/{}, nkpts = {} \
             for nao = {nao}, nkpts = {nkpts}",
            ipkin.comp,
            ipkin.ni,
            ipkin.nj,
            ipkin.nkpts(),
        )));
    }
    let mut h1: GradMatrices = std::array::from_fn(|_| Vec::with_capacity(nkpts));
    for k in 0..nkpts {
        for x in 0..3 {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for i in 0..nao {
                for j in 0..nao {
                    let (r, imv) = ipkin.element(k, x, i, j);
                    re[i * nao + j] = r;
                    im[i * nao + j] = imv;
                }
            }
            h1[x].push(CTensor::from_planes(re, im));
        }
    }

    // krhf.py:97-102 — the local term on the uniform grid.
    let mesh = cell.try_mesh()?;
    if mesh.contains(&0) {
        return Err(invalid(format!(
            "KRHF get_hcore: mesh {mesh:?} has a zero axis"
        )));
    }
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let gv = cell.get_gv(Some(mesh))?;
    let si = cell.get_si(None, Some(mesh), None)?;
    let vlocg = pyscf_pbc_gto::pseudo::get_vlocg(cell, &gv)?;
    let alphas = pyscf_pbc_gto::pseudo::get_alphas(cell)?;
    if si.re.len() != natm * ngrids || vlocg.len() != natm * ngrids || alphas.len() != natm {
        return Err(invalid(format!(
            "KRHF get_hcore: SI/vlocG/alphas shapes disagree with natm = {natm}, ngrids = {ngrids}"
        )));
    }
    // vpplocG[g] = -Σ_a SI[a,g]·vlocG[a,g] (`einsum('ij,ij->j', SI, vlocG)`),
    // vpplocG[0] = Σ_a alphas[a]. Fixed atom order — deterministic.
    let mut vg_re = vec![0.0_f64; ngrids];
    let mut vg_im = vec![0.0_f64; ngrids];
    for g in 0..ngrids {
        let terms_re: Vec<f64> = (0..natm)
            .map(|a| si.re[a * ngrids + g] * vlocg[a * ngrids + g])
            .collect();
        let terms_im: Vec<f64> = (0..natm)
            .map(|a| si.im[a * ngrids + g] * vlocg[a * ngrids + g])
            .collect();
        vg_re[g] = -oracle_sum(&terms_re);
        vg_im[g] = -oracle_sum(&terms_im);
    }
    let mut alpha_terms = Vec::with_capacity(natm.max(1));
    for a in 0..natm {
        alpha_terms.push(alphas[a]);
    }
    vg_re[0] = oracle_sum(&alpha_terms);
    vg_im[0] = 0.0;
    let vpploc_r = pyscf_pbc_tools::ifft(&CTensor::from_planes(vg_re, vg_im), mesh)
        .map_err(|e| invalid(format!("KRHF get_hcore: ifft failed: {e}")))?
        .re;

    let coords = cell.get_uniform_grids(Some(mesh), true)?;
    let owned_gamma = [[0.0_f64; 3]];
    let kpts_nz: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let all_gamma = kpts_nz.iter().all(pyscf_pbc_gto::pbc_intor::is_gamma);
    // krhf.py:103-109 — one deriv-1 table per k (the `nkpts`, not
    // `natm·nkpts`, AO budget 18-04's `ao_cache` governs for 18-05's `h1`
    // build; the per-atom tables inside `hcore_generator` are Task 2's
    // literal port, subsumed for `grad_elec` by clause 7).
    let deriv1 = pyscf_pbc_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", &coords, kpts_nz)
        .map_err(|e| invalid(format!("KRHF get_hcore: deriv-1 AO evaluation failed: {e}")))?;
    if deriv1.comp != 4 {
        return Err(invalid(format!(
            "KRHF get_hcore: deriv-1 table has comp = {}, expected 4",
            deriv1.comp
        )));
    }
    for k in 0..nkpts {
        let kao = &deriv1.kaos[k];
        // vloc[x,i,j] = Σ_g conj(dao[x,i,g])·vpplocR[g]·ao[j,g]
        // (`einsum('agi,g,gj->aij', aos[1:].conj(), vpplocR, aos[0])`).
        // kao block: `[(c·nao+mu)·ngrids+g]`, c = 0 value, 1..3 derivatives.
        for x in 0..3 {
            let mut vr = vec![0.0_f64; nao * nao];
            let mut vi = vec![0.0_f64; nao * nao];
            for i in 0..nao {
                for j in 0..nao {
                    let mut pr = Vec::with_capacity(ngrids);
                    let mut pi = Vec::with_capacity(ngrids);
                    for g in 0..ngrids {
                        let (dr, di) = (
                            kao.re[((x + 1) * nao + i) * ngrids + g],
                            -kao.im[((x + 1) * nao + i) * ngrids + g],
                        );
                        let (ar, ai) = (kao.re[j * ngrids + g], kao.im[j * ngrids + g]);
                        let vr_g = vpploc_r[g];
                        pr.push((dr * ar - di * ai) * vr_g);
                        pi.push((dr * ai + di * ar) * vr_g);
                    }
                    vr[i * nao + j] = oracle_sum(&pr);
                    vi[i * nao + j] = oracle_sum(&pi);
                }
            }
            if all_gamma {
                // krhf.py:106-107 — `dtype == float64` adds `.real`. The
                // lattice-sum imaginary residue at gamma is roundoff; the
                // intor layer already drops it on its gamma planes, so the
                // local term does the same rather than carry a second,
                // inconsistent imaginary part.
                debug_assert!(
                    vi.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < 1e-9,
                    "KRHF get_hcore: non-roundoff imaginary local term at gamma"
                );
                for (h, v) in h1[x][k].re.iter_mut().zip(&vr) {
                    *h += v;
                }
            } else {
                for (h, v) in h1[x][k].re.iter_mut().zip(&vr) {
                    *h += v;
                }
                for (h, v) in h1[x][k].im.iter_mut().zip(&vi) {
                    *h += v;
                }
            }
        }
    }

    Ok(HcoreTables {
        h1,
        si_re: si.re,
        si_im: si.im,
        vlocg,
        gv,
        coords,
        mesh,
        ngrids,
        natm,
        nao,
        all_gamma,
    })
}

// ---------------------------------------------------------------------------
// `hcore_generator` — krhf.py:117-147
// ---------------------------------------------------------------------------

/// `vloc_g[ax,G] = 1j·Gv[G,ax]·SI[atm,G]·vlocG[atm,G]` (`krhf.py:134`) as
/// split planes, per Cartesian component. Shared by the literal seam and the
/// fused contraction: the closure inverse-FFTs these per `(atom, ax)`, the
/// fused route contracts them in G-space directly (Parseval).
pub fn vloc_g_atom(
    si_re: &[f64],
    si_im: &[f64],
    vlocg_row: &[f64],
    gv: &[[f64; 3]],
) -> (Vec<f64>, Vec<f64>) {
    let ngrids = gv.len();
    let mut re = vec![0.0_f64; 3 * ngrids];
    let mut im = vec![0.0_f64; 3 * ngrids];
    for ax in 0..3 {
        for g in 0..ngrids {
            let w = gv[g][ax] * vlocg_row[g];
            // 1j·(sr + i·si)·w = (−si·w, sr·w).
            re[ax * ngrids + g] = -si_im[g] * w;
            im[ax * ngrids + g] = si_re[g] * w;
        }
    }
    (re, im)
}

/// Assemble one atom's `(3,nkpts,nao,nao)` derivative from its local-PP part
/// and the shared `h1` — the body of upstream's `hcore_deriv` closure
/// (`krhf.py:132-147`) minus the AO evaluations and FFTs, as a pure function
/// so the index order is unit-testable:
///
/// ```text
/// hcore[x,k] = vloc[x,k]
/// hcore[x,k,i∈A,j]   -= h1[x,k,i,j]          (:144, the rows of atom A)
/// hcore[x,k,i,j∈A]   -= conj(h1[x,k,j,i])    (:145, the CONJUGATE TRANSPOSE)
/// ```
///
/// `vloc` and `h1` are `[x][k]` row-major `nao × nao`; `mo_coeff` elsewhere
/// in this port is column-major, so the `(p0,p1)` slice cuts ROWS in the
/// first subtraction and COLUMNS in the second — 14-05's `decompose_j2c`
/// misread (+6 306 866.73 Ha) was exactly this confusion.
pub fn assemble_hcore_deriv(
    vloc: &[KMats; 3],
    h1: &GradMatrices,
    p0: usize,
    p1: usize,
    nao: usize,
    nkpts: usize,
) -> [KMats; 3] {
    std::array::from_fn(|x| {
        vloc[x]
            .iter()
            .take(nkpts)
            .zip(h1[x].iter())
            .map(|(v, h)| {
                let mut re = v.re.clone();
                let mut im = v.im.clone();
                for i in p0.min(nao)..p1.min(nao) {
                    for j in 0..nao {
                        re[i * nao + j] -= h.re[i * nao + j];
                        im[i * nao + j] -= h.im[i * nao + j];
                    }
                }
                for i in 0..nao {
                    for j in p0.min(nao)..p1.min(nao) {
                        // conj(h[j,i]): re subtracts as-is, im ADDS.
                        re[i * nao + j] -= h.re[j * nao + i];
                        im[i * nao + j] += h.im[j * nao + i];
                    }
                }
                CTensor::from_planes(re, im)
            })
            .collect()
    })
}

/// The literal per-atom `(3,nkpts,nao,nao)` matrix — upstream's
/// `hcore_deriv(atm_id)` closure body (`krhf.py:132-147`), ported literally
/// INCLUDING trap 1 (the AO table is re-evaluated per atom; `grad_elec`
/// does NOT use this path — clause 7).
///
/// `hcore[ax,kn] += einsum('gi,gj,g->ij', ao.conj(), ao, vloc_R)` with
/// `vloc_R = ifft(vloc_g[ax]).real`, then the `:144-145` row/column
/// subtractions via [`assemble_hcore_deriv`].
///
/// # Errors
/// Propagates the AO evaluations and the FFTs.
pub fn hcore_deriv_matrices(
    tables: &HcoreTables,
    cell: &Cell,
    kpts: &[[f64; 3]],
    atom: usize,
) -> Result<[KMats; 3], PyscfRsError> {
    let owned_gamma = [[0.0_f64; 3]];
    let kpts_nz: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    let nkpts = tables.h1[0].len();
    if atom >= tables.natm {
        return Err(invalid(format!(
            "KRHF hcore_generator: atom {atom} out of range for {} atoms",
            tables.natm
        )));
    }
    let (_, _, p0, p1) = pyscf_gto::aoslice_by_atom(&cell.mol)
        .map_err(|e| invalid(format!("KRHF hcore_generator: aoslice failed: {e}")))?
        .get(atom)
        .copied()
        .ok_or_else(|| {
            invalid(format!(
                "KRHF hcore_generator: no aoslice for atom {atom} (natm = {})",
                tables.natm
            ))
        })?;
    let si_re = &tables.si_re[atom * tables.ngrids..(atom + 1) * tables.ngrids];
    let si_im = &tables.si_im[atom * tables.ngrids..(atom + 1) * tables.ngrids];
    let vlocg_row = &tables.vlocg[atom * tables.ngrids..(atom + 1) * tables.ngrids];
    let (vg_re, vg_im) = vloc_g_atom(si_re, si_im, vlocg_row, &tables.gv);

    let mut vloc: [KMats; 3] = std::array::from_fn(|_| Vec::with_capacity(nkpts));
    for kpt in kpts_nz {
        // Trap 1, literally: the full value-AO table per (atom, k).
        let ao = pyscf_pbc_gto::eval_ao_kpts(
            cell,
            "GTOval_sph",
            &tables.coords,
            std::slice::from_ref(kpt),
        )
        .map_err(|e| invalid(format!("KRHF hcore_generator: value AO failed: {e}")))?;
        for ax in 0..3 {
            let field = CTensor::from_planes(
                vg_re[ax * tables.ngrids..(ax + 1) * tables.ngrids].to_vec(),
                vg_im[ax * tables.ngrids..(ax + 1) * tables.ngrids].to_vec(),
            );
            let vloc_r = pyscf_pbc_tools::ifft(&field, tables.mesh)
                .map_err(|e| invalid(format!("KRHF hcore_generator: ifft failed: {e}")))?
                .re;
            let mut mr = vec![0.0_f64; tables.nao * tables.nao];
            let mut mi = vec![0.0_f64; tables.nao * tables.nao];
            for i in 0..tables.nao {
                for j in 0..tables.nao {
                    let mut pr = Vec::with_capacity(tables.ngrids);
                    let mut pi = Vec::with_capacity(tables.ngrids);
                    for g in 0..tables.ngrids {
                        let (ar, ai) = (
                            ao.kaos[0].re[g + i * tables.ngrids],
                            -ao.kaos[0].im[g + i * tables.ngrids],
                        );
                        let (br, bi) = (
                            ao.kaos[0].re[g + j * tables.ngrids],
                            ao.kaos[0].im[g + j * tables.ngrids],
                        );
                        let w = vloc_r[g];
                        pr.push((ar * br - ai * bi) * w);
                        pi.push((ar * bi + ai * br) * w);
                    }
                    mr[i * tables.nao + j] = oracle_sum(&pr);
                    mi[i * tables.nao + j] = oracle_sum(&pi);
                }
            }
            vloc[ax].push(CTensor::from_planes(mr, mi));
        }
    }
    Ok(assemble_hcore_deriv(
        &vloc, &tables.h1, p0, p1, tables.nao, nkpts,
    ))
}

// ---------------------------------------------------------------------------
// Fused G-space local-PP contraction (D-PBC-31 clause 7)
// ---------------------------------------------------------------------------

/// What [`fused_local_contraction`] actually did — the two counters Task 3's
/// tests gate: value-AO evaluations must be `nkpts` (not `natm·nkpts`) and
/// the `ngrids·nao²` density builds must not scale with `natm`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HcoreFusedStats {
    /// Value-AO table evaluations (`eval_ao_kpts` calls).
    pub value_ao_evals: usize,
    /// Per-k density builds (the `ngrids·nao²` work).
    pub density_builds: usize,
}

/// The clause-7 fused contraction: the local-PP half of `grad_elec:63`
/// WITHOUT materialising any `(3,nkpts,nao,nao)` matrix.
///
/// Per k-point: build the density `ρ_k[g] = Σ_ij conj(ao[i,g])·dm[k,j,i]·
/// ao[j,g]` (one `ngrids·nao²` build — the DM is indexed `ji`, exactly as
/// the einsum), FFT it, and contract
/// `Re Σ_G conj(vloc_g[atm,ax,G])·ρ̂_k[G]·weight/ngrids` for all atoms and
/// components at once through the clause-10 shared primitive
/// (`contract_atom_grid`: `(natm,3) ← (natm·3,ngrids)×(ngrids)`, one
/// `oracle_sum` discipline, one determinism test — NOT a local copy "for
/// now"). If 18-09 has landed with its own call site, both call the same
/// kernel.
///
/// Parseval is exact in exact arithmetic and ~1e-13 relative here
/// (`measurements/parseval.md`); G-space and real-space are different
/// summations, so the fused-vs-literal test gates at that measured residual
/// and never claims bit-identity.
///
/// # Errors
/// Propagates the AO evaluations, the FFTs and the device contraction.
pub fn fused_local_contraction(
    tables: &HcoreTables,
    cell: &Cell,
    kpts: &[[f64; 3]],
    dm0: &KMats,
    stats: &mut HcoreFusedStats,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let owned_gamma = [[0.0_f64; 3]];
    let kpts_nz: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    if dm0.len() != kpts_nz.len()
        || dm0.iter().any(|m| {
            m.re.len() != tables.nao * tables.nao
                || m.im.len() != tables.nao * tables.nao
                || m.re.iter().chain(&m.im).any(|v| !v.is_finite())
        })
    {
        return Err(invalid(format!(
            "KRHF fused contraction: need {} finite nao x nao = {} densities, got {}",
            kpts_nz.len(),
            tables.nao * tables.nao,
            dm0.len(),
        )));
    }
    let selection = select_backend()
        .map_err(|e| {
            invalid(format!(
                "KRHF fused contraction: backend selection failed: {e}"
            ))
        })?
        .client;
    // Parseval scale 1/N — and NOTHING else: upstream's matrix route contracts
    // `vloc_R = ifft(vloc_g).real` with no quadrature weight
    // (`einsum('gi,gj,g->ij', ...)`, krhf.py:141), and that route is
    // FD-correct upstream, so `Vloc(G)` already carries the cell volume and
    // the inverse FFT's 1/N completes the quadrature. An extra `vol/ngrids`
    // here moved every fused value by exactly N/vol (measured
    // literal/fused = 1356.1904557890555 = N/vol to 13 digits on diamond
    // 47³ before the fix) — the weight lives in the field, never in the
    // contraction.
    let scale = 1.0 / tables.ngrids as f64;
    let mut acc = vec![[0.0_f64; 3]; tables.natm];
    for (k, kpt) in kpts_nz.iter().enumerate() {
        // nkpts value-AO evaluations TOTAL — shared across atoms (this is
        // the clause-7 ruling: no per-atom AO cost left to hoist).
        let ao = pyscf_pbc_gto::eval_ao_kpts(
            cell,
            "GTOval_sph",
            &tables.coords,
            std::slice::from_ref(kpt),
        )
        .map_err(|e| invalid(format!("KRHF fused contraction: value AO failed: {e}")))?;
        stats.value_ao_evals += 1;
        // ρ_k[g], real part of the ji-indexed contraction.
        let mut rho_re = vec![0.0_f64; tables.ngrids];
        for g in 0..tables.ngrids {
            let mut pr = Vec::with_capacity(tables.nao * tables.nao);
            let mut pi = Vec::with_capacity(tables.nao * tables.nao);
            for i in 0..tables.nao {
                let (ar, ai) = (
                    ao.kaos[0].re[g + i * tables.ngrids],
                    -ao.kaos[0].im[g + i * tables.ngrids],
                );
                for j in 0..tables.nao {
                    let (br, bi) = (dm0[k].re[j * tables.nao + i], dm0[k].im[j * tables.nao + i]);
                    let (cr, ci) = (
                        ao.kaos[0].re[g + j * tables.ngrids],
                        ao.kaos[0].im[g + j * tables.ngrids],
                    );
                    // (a·b)·c, fixed (i, j) order.
                    let qr = ar * br - ai * bi;
                    let qi = ar * bi + ai * br;
                    pr.push(qr * cr - qi * ci);
                    pi.push(qr * ci + qi * cr);
                }
            }
            rho_re[g] = oracle_sum(&pr);
            let _ = oracle_sum(&pi);
        }
        stats.density_builds += 1;
        let mut rhog = pyscf_pbc_tools::fft(
            &CTensor::from_planes(rho_re, vec![0.0; tables.ngrids]),
            tables.mesh,
        )
        .map_err(|e| invalid(format!("KRHF fused contraction: fft failed: {e}")))?;
        for v in rhog.re.iter_mut() {
            *v *= scale;
        }
        for v in rhog.im.iter_mut() {
            *v *= scale;
        }
        // Field: conj(vloc_g) for every (atom, component) — (natm·3, ngrids).
        let mut fr = vec![0.0_f64; tables.natm * 3 * tables.ngrids];
        let mut fi = vec![0.0_f64; tables.natm * 3 * tables.ngrids];
        for ia in 0..tables.natm {
            let si_re = &tables.si_re[ia * tables.ngrids..(ia + 1) * tables.ngrids];
            let si_im = &tables.si_im[ia * tables.ngrids..(ia + 1) * tables.ngrids];
            let vlocg_row = &tables.vlocg[ia * tables.ngrids..(ia + 1) * tables.ngrids];
            let (gr, gi) = vloc_g_atom(si_re, si_im, vlocg_row, &tables.gv);
            for ax in 0..3 {
                for g in 0..tables.ngrids {
                    fr[(ia * 3 + ax) * tables.ngrids + g] = gr[ax * tables.ngrids + g];
                    fi[(ia * 3 + ax) * tables.ngrids + g] = -gi[ax * tables.ngrids + g];
                }
            }
        }
        let part = pyscf_kernels::pbc::multigrid_grad::contract_atom_grid(
            &selection,
            tables.natm,
            &fr,
            &fi,
            &rhog.re,
            &rhog.im,
        )
        .map_err(|e| {
            invalid(format!(
                "KRHF fused contraction: device reduction failed: {e}"
            ))
        })?;
        for (ia, row) in acc.iter_mut().enumerate() {
            for ax in 0..3 {
                row[ax] = oracle_sum(&[row[ax], part[ia][ax]]);
            }
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// `grad_elec` contractions — krhf.py:60-69, real INSIDE (D-PBC-31 clause 4)
// ---------------------------------------------------------------------------

/// The `.real` lives INSIDE each contraction: `Re(Σ a·b)` is computed as
/// `Σ (Re a·Re b − Im a·Im b)` over a materialised REAL partial buffer
/// through [`oracle_sum`]. Exact for a fixed summation order (complex
/// addition is componentwise), halves the multiplies (2 per product instead
/// of 4), and removes the complex accumulator entirely. A test asserts
/// bit-identity with the complex-then-`.real` route on a non-symmetric DM.
///
/// All three helpers contract the DM **`ji`-indexed** (`18-CONTEXT` trap 9):
/// `M[x,k,i,j]·dm[k,j,i]`. A non-symmetric DM gates the order — `ij` gives a
/// different number.
///
/// # Which index the atom slice cuts
///
/// Upstream slices the SECOND axis: `vhf[:,:,p0:p1]` on `(x,k,i,j)` is
/// `(x,k,i∈A,j)` — ROWS (the bra belongs to atom A; nabla was applied on
/// the bra, `*2` for `nabla|ket>`, `krhf.py:64`) — and `s1[:,:,p0:p1]` on
/// `(k,x,i,j)` is likewise `(k,x,i∈A,j)`. The ket index `j` always runs over
/// ALL AOs. Slicing columns instead passes every self-consistency test and
/// fails Gate B by 6e-2 — the slice axis is gated against upstream's
/// numbers, not just against itself.
/// (`hcore_deriv` needs no slice: it carries its own row/column
/// subtractions.)
///
/// # Layouts
/// Every matrix is `[x][k]` row-major `nao × nao` (`GradMatrices`); `dm0`
/// and `dme0` are `KMats` (per-k row-major).
pub mod contractions {
    use super::*;

    /// Literal `einsum('xkij,kji->x', h1ao, dm0).real` (`krhf.py:63`) over a
    /// full per-atom matrix — the REFERENCE the fused route is gated against
    /// (Task 3), not what `grad_elec` uses.
    pub fn contract_hcore_matrix(h1ao: &GradMatrices, dm0: &KMats, nao: usize) -> [f64; 3] {
        std::array::from_fn(|x| {
            let mut terms = Vec::new();
            for (k, (h, d)) in h1ao[x].iter().zip(dm0.iter()).enumerate() {
                let _ = k;
                for i in 0..nao {
                    for j in 0..nao {
                        let (ar, ai) = (h.re[i * nao + j], h.im[i * nao + j]);
                        let (br, bi) = (d.re[j * nao + i], d.im[j * nao + i]);
                        terms.push(ar * br - ai * bi);
                    }
                }
            }
            oracle_sum(&terms)
        })
    }

    /// The `-h1` rows/columns half of the h-term: what remains of
    /// `contract_hcore_matrix` once the fused contraction has taken the
    /// local-PP part. `hcore = vloc − rows − colsᵀ`
    /// (`hcore_generator:144-145`), so both enter negated:
    /// `−Σ_{k,i∈A,j} h1[x,k,i,j]·dm[k,j,i] − Σ_{k,i,j∈A}
    /// conj(h1[x,k,j,i])·dm[k,j,i]`.
    pub fn contract_h1_atom(
        h1: &GradMatrices,
        dm0: &KMats,
        p0: usize,
        p1: usize,
        nao: usize,
    ) -> [f64; 3] {
        std::array::from_fn(|x| {
            let mut terms = Vec::new();
            for (h, d) in h1[x].iter().zip(dm0.iter()) {
                for i in p0.min(nao)..p1.min(nao) {
                    for j in 0..nao {
                        let (ar, ai) = (h.re[i * nao + j], h.im[i * nao + j]);
                        let (br, bi) = (d.re[j * nao + i], d.im[j * nao + i]);
                        terms.push(-(ar * br - ai * bi));
                    }
                }
                for i in 0..nao {
                    for j in p0.min(nao)..p1.min(nao) {
                        // conj(h[j,i])·dm[j,i].
                        let (ar, ai) = (h.re[j * nao + i], -h.im[j * nao + i]);
                        let (br, bi) = (d.re[j * nao + i], d.im[j * nao + i]);
                        terms.push(-(ar * br - ai * bi));
                    }
                }
            }
            oracle_sum(&terms)
        })
    }

    /// `einsum('xkij,kji->x', vhf[:,:,p0:p1], dm0[:,:,p0:p1]).real * 2`
    /// (`krhf.py:65`; nabla was applied on the bra, `*2` for `nabla|ket>`).
    /// `i` runs over atom A's ROWS on BOTH factors: `Σ_{k,i∈A,j}
    /// vhf[x,k,i,j]·dm[k,j,i]`.
    pub fn contract_vhf_atom(
        vhf: &GradMatrices,
        dm0: &KMats,
        p0: usize,
        p1: usize,
        nao: usize,
    ) -> [f64; 3] {
        std::array::from_fn(|x| {
            let mut terms = Vec::new();
            for (v, d) in vhf[x].iter().zip(dm0.iter()) {
                for i in p0.min(nao)..p1.min(nao) {
                    for j in 0..nao {
                        let (ar, ai) = (v.re[i * nao + j], v.im[i * nao + j]);
                        let (br, bi) = (d.re[j * nao + i], d.im[j * nao + i]);
                        terms.push(2.0 * (ar * br - ai * bi));
                    }
                }
            }
            oracle_sum(&terms)
        })
    }

    /// `-einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2`
    /// (`krhf.py:66`). Upstream's `s1` is `(k,x,i,j)` (k first —
    /// `pbc_intor`'s native order) while `h1ao`/`vhf` are `(x,k,i,j)`; the
    /// port stores all three as `[x][k]`, so the axis swap is a storage
    /// detail and the contracted math is identical: `Σ_{k,i∈A,j}
    /// s1[x,k,i,j]·dme[k,j,i]` (rows in A, like `vhf`).
    pub fn contract_ovlp_atom(
        s1: &GradMatrices,
        dme0: &KMats,
        p0: usize,
        p1: usize,
        nao: usize,
    ) -> [f64; 3] {
        std::array::from_fn(|x| {
            let mut terms = Vec::new();
            for (s, d) in s1[x].iter().zip(dme0.iter()) {
                for i in p0.min(nao)..p1.min(nao) {
                    for j in 0..nao {
                        let (ar, ai) = (s.re[i * nao + j], s.im[i * nao + j]);
                        let (br, bi) = (d.re[j * nao + i], d.im[j * nao + i]);
                        terms.push(-2.0 * (ar * br - ai * bi));
                    }
                }
            }
            oracle_sum(&terms)
        })
    }
}

// ---------------------------------------------------------------------------
// `make_rdm1e` — krhf.py:224-228 via molgrad.make_rdm1e
// ---------------------------------------------------------------------------

/// `make_rdm1e(mo_energy, mo_coeff, mo_occ)` — `krhf.py:224-228`: per k,
/// `[molgrad.make_rdm1e(mo_energy[k], mo_coeff[k], mo_occ[k]) for k]`, and
/// `molgrad` is `dme[i,j] = Σ_m C[i,m]·e[m]·n[m]·conj(C[j,m])`
/// (`pyscf/grad/rhf.py:185-189`: `mo0e = mo0·(e·n)`,
/// `dot(mo0e, mo0.T.conj())`).
///
/// `mo_coeff` is COLUMN-MAJOR `nao × nmo` (this port's convention); output is
/// one row-major `nao × nao` [`CTensor`] per k-point. The index order is the
/// one trap 9 folds into at gamma, stated here so it cannot drift.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] on shape disagreement or non-finite input.
pub fn make_rdm1e_kpts(
    mo_coeff: &[CTensor],
    mo_energy: &[Vec<f64>],
    mo_occ: &[Vec<f64>],
    nao: usize,
) -> Result<KMats, PyscfRsError> {
    let nkpts = mo_coeff.len();
    if mo_energy.len() != nkpts || mo_occ.len() != nkpts {
        return Err(invalid(format!(
            "KRHF make_rdm1e: {nkpts} coeff blocks but {}/{} energy/occ blocks",
            mo_energy.len(),
            mo_occ.len(),
        )));
    }
    let mut out = Vec::with_capacity(nkpts);
    for (k, ((c, e), n)) in mo_coeff.iter().zip(mo_energy).zip(mo_occ).enumerate() {
        let nmo = n.len();
        if e.len() != nmo
            || c.re.len() != nao * nmo
            || c.im.len() != nao * nmo
            || e.iter().chain(n.iter()).any(|v| !v.is_finite())
            || c.re.iter().chain(&c.im).any(|v| !v.is_finite())
        {
            return Err(invalid(format!(
                "KRHF make_rdm1e: k-point {k} shapes disagree (nao = {nao}, nmo = {nmo})"
            )));
        }
        let at = |i: usize, m: usize| (c.re[i + m * nao], c.im[i + m * nao]);
        let mut re = vec![0.0_f64; nao * nao];
        let mut im = vec![0.0_f64; nao * nao];
        for i in 0..nao {
            for j in 0..nao {
                let mut pr = Vec::with_capacity(nmo);
                let mut pi = Vec::with_capacity(nmo);
                for m in 0..nmo {
                    let w = e[m] * n[m];
                    if w == 0.0 {
                        continue;
                    }
                    let (ar, ai) = at(i, m);
                    // conj(C[j,m]).
                    let (br, bi) = (c.re[j + m * nao], -c.im[j + m * nao]);
                    pr.push(w * (ar * br - ai * bi));
                    pi.push(w * (ar * bi + ai * br));
                }
                re[i * nao + j] = oracle_sum(&pr);
                im[i * nao + j] = oracle_sum(&pi);
            }
        }
        out.push(CTensor::from_planes(re, im));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// `KrhfGradients` — the `Gradients` class (krhf.py:230-397)
// ---------------------------------------------------------------------------

/// KRHF k-point nuclear gradient — `pbc/grad/krhf.py`'s `Gradients` class.
///
/// Borrowed mean field ([`Krhf`], owning `with_df`, `cell`, `kpts`,
/// `exxdiv`) plus caller-held SCF products: the density `dm0` and the
/// energy-weighted density `dme0` are built by [`KrhfGradients::new`] from
/// the converged orbitals through [`make_rdm1`] and [`make_rdm1e_kpts`].
/// No SCF runs here, so there is no convergence path and no
/// second-solution noise.
///
/// `extra_force` is the shared zero default (`krhf.py:359-368` returns `0`
/// on the base class; DFT+U / dispersion extras override it in later
/// plans).
pub struct KrhfGradients<'a> {
    mf: &'a Krhf,
    dm0: KDms,
    dme0: KMats,
    atmlst: Option<Vec<usize>>,
}

impl<'a> KrhfGradients<'a> {
    /// Build from converged orbitals. `mo_coeff` is column-major `nao × nmo`
    /// per k-point; `mo_energy`/`mo_occ` are per-orbital per k-point.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] on k-point/shape disagreement or
    /// non-finite input.
    pub fn new(
        mf: &'a Krhf,
        mo_energy: Vec<Vec<f64>>,
        mo_coeff: Vec<CTensor>,
        mo_occ: Vec<Vec<f64>>,
    ) -> Result<Self, PyscfRsError> {
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        if mo_coeff.len() != nkpts || mo_energy.len() != nkpts || mo_occ.len() != nkpts {
            return Err(invalid(format!(
                "KRHF gradient: {nkpts} k-points but {}/{}/{} coeff/energy/occ blocks",
                mo_coeff.len(),
                mo_energy.len(),
                mo_occ.len(),
            )));
        }
        let dm0 = vec![make_rdm1(&mo_coeff, &mo_occ, nao)];
        for m in &dm0[0] {
            if m.re.iter().chain(&m.im).any(|v| !v.is_finite()) {
                return Err(invalid("KRHF gradient: density is non-finite"));
            }
        }
        let dme0 = make_rdm1e_kpts(&mo_coeff, &mo_energy, &mo_occ, nao)?;
        Ok(Self {
            mf,
            dm0,
            dme0,
            atmlst: None,
        })
    }

    /// Atom subset (`de = de[atmlst]`). `None` (default) is all atoms.
    pub fn with_atmlst(mut self, atmlst: Vec<usize>) -> Result<Self, PyscfRsError> {
        for &ia in &atmlst {
            if ia >= self.mf.cell().natm {
                return Err(invalid(format!(
                    "KRHF gradient: atmlst id {ia} out of range for {} atoms",
                    self.mf.cell().natm
                )));
            }
        }
        self.atmlst = Some(atmlst);
        Ok(self)
    }

    fn atom_list(&self) -> Vec<usize> {
        match &self.atmlst {
            Some(list) => list.clone(),
            None => (0..self.mf.cell().natm).collect(),
        }
    }

    /// `get_ovlp(cell, kpts)` — `krhf.py:114-115`:
    /// `-cell.pbc_intor('int1e_ipovlp', kpts)`, `[x][k]` row-major.
    pub fn overlap_deriv(&self) -> Result<GradMatrices, PyscfRsError> {
        let cell = self.mf.cell();
        let kpts = self.mf.kpts();
        let nao = cell.mol.nao_nr;
        let out = pyscf_pbc_gto::pbc_intor(cell, "int1e_ipovlp", kpts, Default::default())?;
        if out.comp != 3 || out.ni != nao || out.nj != nao {
            return Err(invalid(format!(
                "KRHF get_ovlp: int1e_ipovlp has comp = {}, ni/nj = {}/{} for nao = {nao}",
                out.comp, out.ni, out.nj
            )));
        }
        Ok(std::array::from_fn(|x| {
            out.kmats
                .iter()
                .map(|m| {
                    let mut re = vec![0.0_f64; nao * nao];
                    let mut im = vec![0.0_f64; nao * nao];
                    for i in 0..nao {
                        for j in 0..nao {
                            let (r, v) = (
                                m.re[x * nao * nao + i + j * nao],
                                m.im[x * nao * nao + i + j * nao],
                            );
                            re[i * nao + j] = -r;
                            im[i * nao + j] = -v;
                        }
                    }
                    CTensor::from_planes(re, im)
                })
                .collect()
        }))
    }

    /// The exchange-divergence treatment carried by the mean field.
    pub fn exxdiv(&self) -> Option<ExxDiv> {
        self.mf.exxdiv
    }

    /// `get_jk(dm, kpts)` — `krhf.py:260-267`: FFTDF `get_jk_e1`
    /// (`fft.py:324-328`), the ONLY density-fitting route with a gradient
    /// (18-CONTEXT §1.2). Any other route serves the inherited named refusal
    /// (`traits.rs`), never a fallback.
    ///
    /// The gradient entry point takes NO k-pair symmetry flag (D-PBC-30
    /// clause 4b): the derivative sits on the bra, so the energy-path
    /// conjugate identity does not close. `kk_symmetry: false` is passed
    /// explicitly — inheriting the SCF's env default could refuse a correct
    /// calculation.
    pub fn jk_deriv(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let res = mf
            .with_df
            .get_jk_e1(
                dm,
                mf.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: mf.exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
                None,
            )
            .map_err(|e| df_err("get_jk", e))?;
        let vj = take_grad_mats(
            res.vj.ok_or_else(|| invalid("KRHF get_jk: missing vj"))?,
            nkpts,
            nao,
            "vj",
        )?;
        let vk = take_grad_mats(
            res.vk.ok_or_else(|| invalid("KRHF get_jk: missing vk"))?,
            nkpts,
            nao,
            "vk",
        )?;
        Ok((vj, vk))
    }

    /// `get_j(dm, kpts)` — `krhf.py:269-275`: FFTDF `get_j_e1`
    /// (`fft.py:330-333`). Routed directly (not split out of [`Self::jk_deriv`]),
    /// exactly as upstream routes it, so a non-FFTDF builder serves 18-04's
    /// named `get_j_e1` refusal rather than a `get_jk_e1` one.
    ///
    /// # Errors
    /// The named route refusal on GDF/MDF/RSDF/AFTDF; otherwise as
    /// [`Self::jk_deriv`].
    pub fn j_deriv(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let vj = mf
            .with_df
            .get_j_e1(dm, mf.kpts(), None)
            .map_err(|e| df_err("get_j", e))?;
        take_grad_mats(vj, nkpts, nao, "vj")
    }

    /// `get_k(dm, kpts)` — `krhf.py:277-284`: FFTDF `get_k_e1`
    /// (`fft.py:335-340`), with `mf.exxdiv`. Routed directly for the same
    /// reason as [`Self::j_deriv`].
    ///
    /// # Errors
    /// The named route refusal on GDF/MDF/RSDF/AFTDF; otherwise as
    /// [`Self::jk_deriv`].
    pub fn k_deriv(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        let mf = self.mf;
        let nkpts = nkpts_of(mf.kpts());
        let nao = mf.cell().mol.nao_nr;
        let vk = mf
            .with_df
            .get_k_e1(dm, mf.kpts(), None, mf.exxdiv, None, None)
            .map_err(|e| df_err("get_k", e))?;
        take_grad_mats(vk, nkpts, nao, "vk")
    }

    /// `get_veff(dm, kpts)` — `krhf.py:219-222`: `vj - vk·.5`.
    pub fn veff(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        let (vj, vk) = self.jk_deriv(dm)?;
        Ok(std::array::from_fn(|x| {
            vj[x]
                .iter()
                .zip(vk[x].iter())
                .map(|(j, k)| {
                    CTensor::from_planes(
                        j.re.iter().zip(&k.re).map(|(a, b)| a - 0.5 * b).collect(),
                        j.im.iter().zip(&k.im).map(|(a, b)| a - 0.5 * b).collect(),
                    )
                })
                .collect()
        }))
    }

    /// `grad_elec` — `krhf.py:34-74`, in upstream's exact order, with the
    /// clause-7 fused local-PP contraction in place of the materialised
    /// `hcore_deriv(ia)` matrix:
    ///
    /// ```text
    /// h1ao/fused → de[x] += h-part                      (:63, fused local + matrix kinetic)
    /// vhf       → de[x] += 2·vhf-part                   (:65)
    /// s1        → de[x] -= 2·ovlp-part                  (:66)
    ///             de[x] /= nkpts        INSIDE atom loop (:67)
    ///             de    += extra_force  AFTER division   (:68, zero here)
    /// de += vppnl_nuc_grad/nkpts       WHOLE array      (:69, after loop)
    /// ```
    ///
    /// Three scopes in five lines (18-CONTEXT trap 3): reproduce literally.
    /// A reordering here is invisible to a `lib.fp` comparison dominated by
    /// one term.
    pub fn electronic_gradient(&self) -> Result<Gradient, PyscfRsError> {
        use contractions::{contract_h1_atom, contract_ovlp_atom, contract_vhf_atom};

        let mf = self.mf;
        let cell = mf.cell();
        let kpts = mf.kpts();
        let nkpts = nkpts_of(kpts) as f64;
        let nao = cell.mol.nao_nr;
        let dm0 = &self.dm0[0];

        let tables = precompute_hcore(cell, kpts)?;
        let s1 = self.overlap_deriv()?;
        let vhf = self.veff(&self.dm0)?;
        let mut fused_stats = HcoreFusedStats::default();
        let fused = fused_local_contraction(&tables, cell, kpts, dm0, &mut fused_stats)?;
        let slices = pyscf_gto::aoslice_by_atom(&cell.mol)
            .map_err(|e| invalid(format!("KRHF grad_elec: aoslice failed: {e}")))?;

        let atmlst = self.atom_list();
        let mut de = Vec::with_capacity(atmlst.len());
        for &ia in &atmlst {
            let (_, _, p0, p1) = slices.get(ia).copied().ok_or_else(|| {
                invalid(format!(
                    "KRHF grad_elec: no aoslice for atom {ia} (natm = {})",
                    cell.natm
                ))
            })?;
            let h1 = contract_h1_atom(&tables.h1, dm0, p0, p1, nao);
            let vv = contract_vhf_atom(&vhf, dm0, p0, p1, nao);
            let ss = contract_ovlp_atom(&s1, &self.dme0, p0, p1, nao);
            let mut row = [0.0_f64; 3];
            for x in 0..3 {
                // krhf.py:63/65/66 accumulate, :67 divides INSIDE the loop.
                row[x] = oracle_sum(&[fused[ia][x], h1[x], vv[x], ss[x]]) / nkpts;
            }
            // krhf.py:68 — AFTER the division (a no-op here: the base-class
            // extra_force is zero, and DFT+U extras override it in 18-08).
            let extra = self.extra_force(ia, &[])?;
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], extra[x]]);
            }
            de.push(row);
        }
        // krhf.py:69 — the whole array, after the loop (trap 3).
        let vppnl = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(cell, dm0, kpts)?;
        for (row, &ia) in de.iter_mut().zip(atmlst.iter()) {
            for x in 0..3 {
                row[x] = oracle_sum(&[row[x], vppnl[ia][x] / nkpts]);
            }
        }
        Ok(de)
    }

    /// `grad_nuc` — `krhf.py:286-288`: 18-03's `ewald_nuc_grad`.
    pub fn nuclear_gradient(&self) -> Result<Gradient, PyscfRsError> {
        pyscf_pbc_gto::ewald_nuc_grad(self.mf.cell(), None, None)
    }

    /// `kernel` — `krhf.py:380-397`: `grad_elec + grad_nuc`.
    pub fn kernel(&self) -> Result<Gradient, PyscfRsError> {
        let elec = self.electronic_gradient()?;
        let nuc = self.nuclear_gradient()?;
        let atmlst = self.atom_list();
        if nuc.len() != self.mf.cell().natm {
            return Err(invalid(
                "KRHF kernel: nuclear part has the wrong atom count",
            ));
        }
        Ok(elec
            .into_iter()
            .zip(atmlst)
            .map(|(e, ia)| {
                [
                    oracle_sum(&[e[0], nuc[ia][0]]),
                    oracle_sum(&[e[1], nuc[ia][1]]),
                    oracle_sum(&[e[2], nuc[ia][2]]),
                ]
            })
            .collect())
    }
}

impl<'a> Gradients for KrhfGradients<'a> {
    fn cell(&self) -> &Cell {
        self.mf.cell()
    }

    fn kpts(&self) -> &[[f64; 3]] {
        self.mf.kpts()
    }

    fn grad_elec(&self) -> Result<Gradient, PyscfRsError> {
        self.electronic_gradient()
    }

    fn get_hcore(&self) -> Result<GradMatrices, PyscfRsError> {
        get_hcore(self.mf.cell(), self.mf.kpts())
    }

    fn hcore_generator(&self, atom: usize) -> Result<[KMats; 3], PyscfRsError> {
        let tables = precompute_hcore(self.mf.cell(), self.mf.kpts())?;
        hcore_deriv_matrices(&tables, self.mf.cell(), self.mf.kpts(), atom)
    }

    fn get_ovlp(&self) -> Result<GradMatrices, PyscfRsError> {
        self.overlap_deriv()
    }

    fn get_jk(&self, dm: &KDms) -> Result<(GradMatrices, GradMatrices), PyscfRsError> {
        self.jk_deriv(dm)
    }

    fn get_j(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.j_deriv(dm)
    }

    fn get_k(&self, dm: &KDms) -> Result<GradMatrices, PyscfRsError> {
        self.k_deriv(dm)
    }

    fn grad_nuc(&self) -> Result<Gradient, PyscfRsError> {
        self.nuclear_gradient()
    }

    fn make_rdm1e(&self) -> Result<KDms, PyscfRsError> {
        Ok(vec![self.dme0.clone()])
    }

    /// The energy seam [`crate::verify_fd`] consumes (`krhf.py:300-320`,
    /// `371`): an [`EnergyScanner`] over the shared-state
    /// [`ScfGradScanner`] at cell-derived SCF defaults
    /// (`conv_tol = max(precision·10, 1e-8)`).
    ///
    /// Gates that need tighter convergence (Gate B) build the
    /// [`ScfGradScanner`] explicitly with [`KrhfScannerConfig::tight`]
    /// instead of going through this default.
    fn as_scanner(&self) -> Result<EnergyScanner, PyscfRsError> {
        let precision = self.mf.cell().precision;
        let scanner = ScfGradScanner::new(
            self.mf.cell().clone(),
            KrhfScannerConfig {
                kpts: self.mf.kpts().to_vec(),
                exxdiv: self.mf.exxdiv,
                conv_tol: (precision * 10.0).max(1e-8),
                conv_tol_grad: None,
                max_cycle: 50,
            },
        );
        Ok(scanner.energy_scanner())
    }
}
