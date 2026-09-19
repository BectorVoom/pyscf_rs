//! GTH **non-local** nuclear gradients — `vppnl_nuc_grad` (plan 18-03, Task 4).
//!
//! Ports `pyscf/pbc/gto/pseudo/pp_int.py:443-509` (`vppnl_nuc_grad`) with its
//! worker `:300-405` (`_contract_ppnl_nuc_grad`) and the C driver
//! `pyscf/lib/pbc/pp.c:277-405` (`ppnl_nuc_grad_fill_gs1`,
//! `contract_ppnl_nuc_grad`, `_contract_vnuc_ip1_dm`).
//!
//! # What it computes
//!
//! The nuclear gradient of `V_nl` contracted with a density matrix, as a
//! `(natm, 3)` array in Ha/Bohr (`krhf.grad_elec:69` adds it to the WHOLE `de`
//! array, outside the atom loop and after `extra_force` — `18-CONTEXT §3.3`).
//! It consumes `int1e_r{2,4}_origi_ip2` through `_int_vnl`'s `ppnl_half_ip2`
//! argument, alongside the `ppnl_half` the energy path already builds.
//!
//! # Shared projector data
//!
//! `_prepare_hl_data` (`pp_int.py:211-228`) is this port's
//! [`FakeCellVnl`](crate::pseudo::FakeCellVnl) + [`HlBlock`](crate::pseudo::HlBlock):
//! [`vppnl_nuc_grad`] calls [`fake_cell_vnl`](crate::pseudo::fake_cell_vnl)
//! ONCE and feeds the same blocks to both the half-overlap builds and the
//! contraction — never re-derived (two projector-block layouts is how a
//! plausible wrong number ships; `16-CONTEXT §1.1`'s ruling, applied here).
//!
//! # Index order (18-CONTEXT trap 9)
//!
//! Upstream's k-path contracts `dppnl` against the TRANSPOSED density,
//! `einsum('dpq,qp->d', dppnl_k, dm_dmH[k])` (`pp_int.py:497`), and the gamma
//! C driver contracts `buf` against `dm[i0+a, j0+b]` (`pp.c:340-352`). With
//! `dm` F-order (`dm[p + q*nao]`, the workspace column-major convention):
//!
//! ```text
//! gamma: t[c] = Σ_{a,b} buf[c, a, b] · dm[oi+a + (oj+b)*nao],
//!        buf[c, a, b] = Σ_{i,j} Σ_m hl[j, i] · dP[i, c, m, a] · P[j, m, b]
//! k:     grad[d] += Σ_{p,q} dppnl[d, p, q] · dmH[q, p],
//!        dppnl[d, p, q] = Σ_{i,j} Σ_m conj(dP[i, d, m, p]) · hl[i, j] · P[j, m, q]
//! ```
//!
//! (`hl` is symmetric, so `hl[j,i] = hl[i,j]`.) Every reduction routes through
//! [`pyscf_algebra::oracle_sum`].
//!
//! # cintx prerequisite
//!
//! `int1e_r{2,4}_origi_ip2` are cintx `unstable-source-api` symbols; this
//! crate's `gth-pp` feature (default-on) enables them, and
//! `tests/cintx_moment_weighted_available.rs` re-proves on every run that the
//! derivative half evaluates AND differs from the unweighted parent
//! (`18-CONTEXT §1.3`). `oracle_covered = false`: the numeric gate is upstream
//! PySCF, not cintx's own oracle.

use crate::cell::Cell;
use crate::pbc_intor::{PbcIntorOpts, PbcIntorOutput, intor_cross};
use crate::pseudo::vnl::{FakeCellVnl, MAX_NPROJ, fake_cell_vnl};
use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::{CoreError, PyscfRsError};

/// The three derivative half-overlap operators of `_int_vnl`'s `ppnl_half_ip2`
/// (`pp_int.py:454`), in projector order.
///
/// Rank 0 is `int1e_ipovlp` — the BRA derivative — negated after evaluation to
/// the ket derivative `_int_vnl` needs (`pp_int.py:457-460`: "`int1e_ipovlp`
/// computes ip1 so multiply -1 to get ip2"). Ranks 1-2 are genuine `ip2`
/// (ket-derivative) symbols.
pub const VNL_IP2_INTORS: [&str; MAX_NPROJ] =
    ["int1e_ipovlp", "int1e_r2_origi_ip2", "int1e_r4_origi_ip2"];

/// `_int_vnl(..., intors, comp=3)` — `pp_int.py:626-674` with the derivative
/// operators: the `<p_0^l| O_i |d/dR phi_mu>` halves (`O_0 = 1`,
/// `O_1 = r²`, `O_2 = r⁴`, derivative on the AO/ket centre), one entry per
/// projector rank; `None` where that rank has no channels.
///
/// Each is `[nkpts]` matrices of shape `(n_proj_ao_rank, nao)` per component,
/// F-order per component (`(c, i, j)` at `c*ni*nj + i + j*ni`).
///
/// # Errors
/// As [`intor_cross`]. A cintx failure on `int1e_r{2,4}_origi_ip2` means the
/// `gth-pp` feature is off — see the module docs.
pub fn int_vnl_ip2(
    cell: &Cell,
    fake: &FakeCellVnl,
    kpts: &[[f64; 3]],
) -> Result<[Option<PbcIntorOutput>; MAX_NPROJ], PyscfRsError> {
    let mut out: [Option<PbcIntorOutput>; MAX_NPROJ] = [None, None, None];
    for (rank, slot) in out.iter_mut().enumerate() {
        let Some(fcell) = fake.cells[rank].as_ref() else {
            continue;
        };
        // Same geometry as the energy halves (bra = projector, ket = the
        // translated AO); only the operator changes. `comp` defaults to the
        // operator's natural count — 3, from the layout table.
        *slot = Some(intor_cross(
            VNL_IP2_INTORS[rank],
            fcell,
            cell,
            kpts,
            PbcIntorOpts::default(),
        )?);
    }
    Ok(out)
}

/// `vppnl_nuc_grad(cell, dm, kpts=None)` — `pp_int.py:443-509`.
///
/// The `(natm, 3)` non-local-PP nuclear gradient contracted with the real
/// symmetric density matrix `dm` (F-order `nao x nao`). At non-gamma k-points
/// the same `dm` is used at every k (upstream's `dm.shape == (nkpts, nao,
/// nao)` with a shared matrix; `dm_dmH = dm + dm.T.conj()` is formed
/// internally).
///
/// * Gamma (`pp_int.py:462-466`): the C-driver contraction
///   [`_contract_ppnl_nuc_grad`], then `grad *= -2`.
/// * k-points (`pp_int.py:468-509`): the pure-numpy path — per-k `dppnl`,
///   projector-atom `+=`, AO-slice `−=`, real part taken (upstream warns when
///   `max|Im| >= 1e-8`; so does this port).
///
/// # Errors
/// As [`int_vnl_ip2`], plus [`CoreError::InvalidMolecule`] when `dm` is not
/// `nao x nao` or when a projector cell and its `h^l` blocks disagree.
pub fn vppnl_nuc_grad(
    cell: &Cell,
    dm: &[f64],
    kpts: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    if dm.len() != nao * nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "vppnl_nuc_grad: dm has {} elements, expected nao x nao = {}",
            dm.len(),
            nao * nao
        ))));
    }
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };

    // _prepare_hl_data, shared with the energy path: ONE call, both consumers.
    let fake = fake_cell_vnl(cell)?;
    if fake.blocks.is_empty() {
        return Ok(vec![[0.0; 3]; cell.mol.natm]);
    }

    if kpts.iter().all(|k| crate::pbc_intor::is_gamma(k)) {
        let halves = crate::pseudo::vnl::int_vnl(cell, &fake, kpts)?;
        let mut halves_ip2 = int_vnl_ip2(cell, &fake, kpts)?;
        // pp_int.py:457-460 — int1e_ipovlp differentiates the bra (projector);
        // the ket derivative is its negation (translation invariance).
        if let Some(rank0) = halves_ip2[0].as_mut() {
            for m in rank0.kmats.iter_mut() {
                for v in m.re.iter_mut() {
                    *v = -*v;
                }
                for v in m.im.iter_mut() {
                    *v = -*v;
                }
            }
        }
        let mut grad = contract_ppnl_nuc_grad(cell, &fake, dm, &halves, &halves_ip2, kpts)?;
        for row in grad.iter_mut() {
            for c in 0..3 {
                row[c] *= -2.0;
            }
        }
        return Ok(grad);
    }

    // dmH[q, p] = dm[q,p] + dm[p,q], real (F-order storage), shared at
    // every k — the 18-03 contract. [`vppnl_nuc_grad_kdm`] builds the per-k
    // complex planes instead.
    let mut dmh = vec![0.0_f64; nao * nao];
    for q in 0..nao {
        for p in 0..nao {
            dmh[q + p * nao] = dm[q + p * nao] + dm[p + q * nao];
        }
    }
    let zeros = vec![0.0_f64; nao * nao];
    let dmh_re = vec![dmh; kpts.len()];
    let dmh_im = vec![zeros; kpts.len()];
    vppnl_nuc_grad_kpts(cell, &fake, &dmh_re, &dmh_im, kpts)
}

/// `vppnl_nuc_grad(cell, dm, kpts)` for a k-point density — plan 18-05.
///
/// Same quantity as [`vppnl_nuc_grad`] (`pp_int.py:443-509`), but the density
/// is the SCF's per-k complex `dm[k]`, `nao × nao` row-major
/// (`pyscf-pbc-scf` convention), instead of one real matrix shared at every
/// k. Upstream's k-path (`pp_int.py:468-509`) forms
/// `dm_dmH = dm + dm.transpose(0,2,1).conj()` per k and takes `.real` at the
/// end (warning when `max|Im| >= 1e-8`); this port does exactly that, with
/// the Hermitian sum pre-built per k and every reduction through
/// [`oracle_sum`].
///
/// At all-gamma k-points this delegates to [`vppnl_nuc_grad`] on the real
/// parts (upstream's gamma branch takes `dm.real`, `pp_int.py:351`).
///
/// # Errors
/// As [`vppnl_nuc_grad`], plus [`CoreError::InvalidMolecule`] when `dm` has
/// the wrong k-point count or plane shape, or holds non-finite entries.
pub fn vppnl_nuc_grad_kdm(
    cell: &Cell,
    dm: &[CTensor],
    kpts: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };
    if dm.len() != kpts.len()
        || dm.iter().any(|m| {
            m.re.len() != nao * nao
                || m.im.len() != nao * nao
                || m.re.iter().chain(&m.im).any(|v| !v.is_finite())
        })
    {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "vppnl_nuc_grad_kdm: need {} finite nao x nao = {} planes, got {}",
            kpts.len(),
            nao * nao,
            dm.len(),
        ))));
    }
    if kpts.iter().all(|k| crate::pbc_intor::is_gamma(k)) {
        // F-order real parts into the 18-03 entry point (upstream's `.real`).
        let mut flat = vec![0.0_f64; nao * nao];
        for q in 0..nao {
            for p in 0..nao {
                flat[q + p * nao] = dm[0].re[q * nao + p];
            }
        }
        return vppnl_nuc_grad(cell, &flat, kpts);
    }
    let fake = fake_cell_vnl(cell)?;
    if fake.blocks.is_empty() {
        return Ok(vec![[0.0; 3]; cell.mol.natm]);
    }
    // dmH[k, q, p] = dm[k, q, p] + conj(dm[k, p, q]), F-order planes.
    let mut dmh_re = Vec::with_capacity(kpts.len());
    let mut dmh_im = Vec::with_capacity(kpts.len());
    for m in dm {
        let mut wr = vec![0.0_f64; nao * nao];
        let mut wi = vec![0.0_f64; nao * nao];
        for q in 0..nao {
            for p in 0..nao {
                wr[q + p * nao] = m.re[q * nao + p] + m.re[p * nao + q];
                wi[q + p * nao] = m.im[q * nao + p] - m.im[p * nao + q];
            }
        }
        dmh_re.push(wr);
        dmh_im.push(wi);
    }
    vppnl_nuc_grad_kpts(cell, &fake, &dmh_re, &dmh_im, kpts)
}

/// `_contract_ppnl_nuc_grad` — `pp_int.py:300-405` at the gamma point
/// (`pp.c:277-355`, `ppnl_nuc_grad_fill_gs1` + `_contract_vnuc_ip1_dm`).
///
/// Per ordered shell pair `(ish, jsh)`, per projector block:
/// `buf[c,a,b] = Σ_{i,j,m} hl[j,i]·dP[i,c,m,a]·P[j,m,b]`, then
/// `t[c] = Σ_{a,b} buf·dm`, `grad[atom(ish)] += t`, `grad[atom(block)] −= t`.
/// The caller applies the final `*= -2` (`pp_int.py:465`).
///
/// `halves_ip2[0]` must ALREADY carry the `int1e_ipovlp` negation (done by
/// [`vppnl_nuc_grad`]); this function contracts what it is given.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] on a block/cell layout inconsistency.
pub fn contract_ppnl_nuc_grad(
    cell: &Cell,
    fake: &FakeCellVnl,
    dm: &[f64],
    halves: &[Option<PbcIntorOutput>; MAX_NPROJ],
    halves_ip2: &[Option<PbcIntorOutput>; MAX_NPROJ],
    kpts: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let nao = cell.mol.nao_nr;
    let natm = cell.mol.natm;
    let nkpts = kpts.len();
    if nkpts != 1 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "contract_ppnl_nuc_grad: gamma path needs exactly one k-point, got {nkpts}"
        ))));
    }
    let mut grad = vec![[0.0_f64; 3]; natm];

    // AO layout of the real cell: shell -> (atom, ao offset, ao count).
    let (sh_atom, sh_off, sh_cnt) = cell_shell_map(cell);
    // Upstream's gamma path drops the imaginary plane; a large residue means
    // the projector lattice sum is off, so warn first like `get_pp_nl` does.
    for halves_set in [halves, halves_ip2] {
        for half in halves_set.iter().flatten() {
            let max_im = half.max_abs_imag();
            if max_im > 1e-9 {
                tracing::warn!(
                    "contract_ppnl_nuc_grad: gamma-point half-overlap has max|Im| = \
                     {max_im:e}; upstream drops it, but this means the projector \
                     lattice sum is off"
                );
            }
        }
    }

    // `offset[i]` walks the AO rows of projector cell `i` — the same running
    // counter as `get_pp_nl` and `pp_int.py:431-433`.
    let mut offset = [0usize; MAX_NPROJ];
    for block in &fake.blocks {
        let nd = 2 * block.l + 1;
        let dim = block.dim;
        // Block-local rows present in each rank's half matrix.
        let mut rows: Vec<Vec<usize>> = Vec::with_capacity(dim);
        for i in 0..dim {
            let half = halves[i]
                .as_ref()
                .ok_or_else(|| missing_rank(block, dim, i))?;
            let row0 = offset[i];
            if row0 + nd > half.ni {
                return Err(layout_error(block, i, half.ni, row0, nd));
            }
            rows.push((row0..row0 + nd).collect());
        }
        // Rank-0-negated derivative halves share the same row layout; check.
        for i in 0..dim {
            let half = halves_ip2[i]
                .as_ref()
                .ok_or_else(|| missing_rank(block, dim, i))?;
            let row0 = offset[i];
            if row0 + nd > half.ni {
                return Err(layout_error(block, i, half.ni, row0, nd));
            }
            if half.comp != 3 {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "contract_ppnl_nuc_grad: rank-{i} derivative half has comp = {}, \
                     expected 3",
                    half.comp
                ))));
            }
        }

        // Per (ish, jsh) shell pair: build buf from the block rows, contract
        // with the dm block, route to bra atom (+) and projector atom (−).
        for ish in 0..cell.mol.nbas {
            let di = sh_cnt[ish];
            if di == 0 {
                continue;
            }
            for jsh in 0..cell.mol.nbas {
                let dj = sh_cnt[jsh];
                if dj == 0 {
                    continue;
                }
                let oi = sh_off[ish];
                let oj = sh_off[jsh];
                let mut buf = vec![0.0_f64; 3 * di * dj];
                for i in 0..dim {
                    for j in 0..dim {
                        let hij = block.h[j * dim + i];
                        if hij == 0.0 {
                            continue;
                        }
                        let p_half = &halves[j].as_ref().expect("checked").kmats[0].re;
                        let ni_j = halves[j].as_ref().expect("checked").ni;
                        let d_half = &halves_ip2[i].as_ref().expect("checked").kmats[0].re;
                        let ni_i = halves_ip2[i].as_ref().expect("checked").ni;
                        for m in 0..nd {
                            let prow = rows[j][m];
                            let drow = rows[i][m];
                            for (aa, a) in (oi..oi + di).enumerate() {
                                // dP[i, c, m, a] for all c.
                                let mut dip = [0.0_f64; 3];
                                for c in 0..3 {
                                    dip[c] = d_half[c * ni_i * nao + drow + a * ni_i];
                                }
                                for (bb, b) in (oj..oj + dj).enumerate() {
                                    let pv = p_half[prow + b * ni_j];
                                    for c in 0..3 {
                                        buf[c * di * dj + aa + bb * di] += hij * dip[c] * pv;
                                    }
                                }
                            }
                        }
                    }
                }
                // _contract_vnuc_ip1_dm.
                let ia = sh_atom[ish];
                let ka = block.atom;
                let mut t = [0.0_f64; 3];
                for c in 0..3 {
                    let mut terms = Vec::with_capacity(di * dj);
                    for bb in 0..dj {
                        for aa in 0..di {
                            terms.push(
                                buf[c * di * dj + aa + bb * di] * dm[(oi + aa) + (oj + bb) * nao],
                            );
                        }
                    }
                    t[c] = oracle_sum(&terms);
                }
                for c in 0..3 {
                    grad[ia][c] += t[c];
                    grad[ka][c] -= t[c];
                }
            }
        }
        for i in 0..dim {
            offset[i] += 2 * block.l + 1;
        }
    }
    Ok(grad)
}

/// The k-point path — `pp_int.py:468-509`, pure numpy (no C driver).
///
/// Per k, per block: `dppnl[d,p,q] = Σ_{i,j,m} conj(dP[i,d,m,p])·hl[i,j]·P[j,m,q]`
/// feeds `grad[proj] += Σ dppnl·dmH` immediately and accumulates into the full
/// `dppnl[k]` whose AO-slice rows feed `grad[ia] −= Σ dppnl[k][:,p∈A,:]·dmH`.
/// `dmH[k]` is the caller-built Hermitian sum `dm[k] + dm[k]ᴴ`, F-order
/// `(q + p·nao)` planes with independent real/imaginary parts; the real part
/// of the total is taken at the end.
fn vppnl_nuc_grad_kpts(
    cell: &Cell,
    fake: &FakeCellVnl,
    dmh_re: &[Vec<f64>],
    dmh_im: &[Vec<f64>],
    kpts: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    use pyscf_gto::aoslice_by_atom;

    let nao = cell.mol.nao_nr;
    let natm = cell.mol.natm;
    let nkpts = kpts.len();
    let slices = aoslice_by_atom(&cell.mol)?;
    if dmh_re.len() != nkpts
        || dmh_im.len() != nkpts
        || dmh_re.iter().any(|m| m.len() != nao * nao)
        || dmh_im.iter().any(|m| m.len() != nao * nao)
    {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "vppnl_nuc_grad: need {nkpts} dmH planes of nao x nao = {}, got {}/{}",
            nao * nao,
            dmh_re.len(),
            dmh_im.len(),
        ))));
    }

    let halves = crate::pseudo::vnl::int_vnl(cell, fake, kpts)?;
    let mut halves_ip2 = int_vnl_ip2(cell, fake, kpts)?;
    if let Some(rank0) = halves_ip2[0].as_mut() {
        for m in rank0.kmats.iter_mut() {
            for v in m.re.iter_mut() {
                *v = -*v;
            }
            for v in m.im.iter_mut() {
                *v = -*v;
            }
        }
    }

    // dmH[q, p] planes arrive pre-built from the caller (F-order storage):
    // the shared real `dm + dmᵀ` for [`vppnl_nuc_grad`], the per-k complex
    // `dm[k] + dm[k]ᴴ` for [`vppnl_nuc_grad_kdm`]. With zero imaginary weights
    // the products below are bit-identical to the pre-18-05 real-only loop
    // (`x·0 = ±0`, `x ± 0 = x`), so the 18-03 gate still pins this path.

    let mut grad_re = vec![[0.0_f64; 3]; natm];
    let mut grad_im = vec![[0.0_f64; 3]; natm];
    // Full dppnl[k] per k, accumulated over blocks for the AO-slice term.
    let mut dall_re = vec![vec![0.0_f64; 3 * nao * nao]; nkpts];
    let mut dall_im = vec![vec![0.0_f64; 3 * nao * nao]; nkpts];

    let mut offset = [0usize; MAX_NPROJ];
    for block in &fake.blocks {
        let nd = 2 * block.l + 1;
        let dim = block.dim;
        for i in 0..dim {
            let (Some(h), Some(dh)) = (halves[i].as_ref(), halves_ip2[i].as_ref()) else {
                return Err(missing_rank(block, dim, i));
            };
            if offset[i] + nd > h.ni || offset[i] + nd > dh.ni {
                return Err(layout_error(block, i, h.ni.min(dh.ni), offset[i], nd));
            }
            if dh.comp != 3 {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "vppnl_nuc_grad: rank-{i} derivative half has comp = {}, expected 3",
                    dh.comp
                ))));
            }
        }
        for (k, _) in kpts.iter().enumerate() {
            // dppnl_k[d, p, q] for this block.
            let mut dk_re = vec![0.0_f64; 3 * nao * nao];
            let mut dk_im = vec![0.0_f64; 3 * nao * nao];
            for i in 0..dim {
                for j in 0..dim {
                    let hij = block.h[i * dim + j];
                    if hij == 0.0 {
                        continue;
                    }
                    let ph = &halves[j].as_ref().expect("checked").kmats[k];
                    let dh = &halves_ip2[i].as_ref().expect("checked").kmats[k];
                    let ni_j = halves[j].as_ref().expect("checked").ni;
                    let ni_i = halves_ip2[i].as_ref().expect("checked").ni;
                    for m in 0..nd {
                        let prow = offset[j] + m;
                        let drow = offset[i] + m;
                        for p in 0..nao {
                            let mut dip_re = [0.0_f64; 3];
                            let mut dip_im = [0.0_f64; 3];
                            for d in 0..3 {
                                dip_re[d] = dh.re[d * ni_i * nao + drow + p * ni_i];
                                dip_im[d] = dh.im[d * ni_i * nao + drow + p * ni_i];
                            }
                            for q in 0..nao {
                                let (pr, pi) = (ph.re[prow + q * ni_j], ph.im[prow + q * ni_j]);
                                for d in 0..3 {
                                    // conj(dP) · hl · P.
                                    let (cr, ci) = (dip_re[d] * hij, -dip_im[d] * hij);
                                    let o = d * nao * nao + p + q * nao;
                                    dk_re[o] += cr * pr - ci * pi;
                                    dk_im[o] += cr * pi + ci * pr;
                                }
                            }
                        }
                    }
                }
            }
            // grad[proj] += einsum('dpq,qp->d', dppnl_k, dmH).
            let ka = block.atom;
            for d in 0..3 {
                let mut tr = Vec::with_capacity(nao * nao);
                let mut ti = Vec::with_capacity(nao * nao);
                for q in 0..nao {
                    for p in 0..nao {
                        let (wr, wi) = (dmh_re[k][q + p * nao], dmh_im[k][q + p * nao]);
                        let o = d * nao * nao + p + q * nao;
                        tr.push(dk_re[o] * wr - dk_im[o] * wi);
                        ti.push(dk_re[o] * wi + dk_im[o] * wr);
                    }
                }
                grad_re[ka][d] += oracle_sum(&tr);
                grad_im[ka][d] += oracle_sum(&ti);
            }
            for (dst, src) in dall_re[k].iter_mut().zip(dk_re.iter()) {
                *dst += *src;
            }
            for (dst, src) in dall_im[k].iter_mut().zip(dk_im.iter()) {
                *dst += *src;
            }
        }
        for i in 0..dim {
            offset[i] += nd;
        }
    }

    // grad[ia] −= einsum('kdpq,kqp->d', dppnl[:, :, p0:p1, :], dmH[:, :, p0:p1]).
    for ia in 0..natm {
        let (_, _, p0, p1) = slices.get(ia).copied().unwrap_or((0, 0, 0, 0));
        for (k, _) in kpts.iter().enumerate() {
            for d in 0..3 {
                let mut tr = Vec::new();
                let mut ti = Vec::new();
                for q in 0..nao {
                    for p in p0.min(nao)..p1.min(nao) {
                        let (wr, wi) = (dmh_re[k][q + p * nao], dmh_im[k][q + p * nao]);
                        let o = d * nao * nao + p + q * nao;
                        tr.push(dall_re[k][o] * wr - dall_im[k][o] * wi);
                        ti.push(dall_re[k][o] * wi + dall_im[k][o] * wr);
                    }
                }
                grad_re[ia][d] -= oracle_sum(&tr);
                grad_im[ia][d] -= oracle_sum(&ti);
            }
        }
    }

    let max_im = grad_im
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    if max_im >= 1e-8 {
        tracing::warn!(
            "vppnl_nuc_grad: large imaginary part ({max_im:e}) from pseudopotential \
             non-local term gradient (upstream pp_int.py:504-506 warns here too)"
        );
    }
    Ok(grad_re)
}

/// Per-shell `(atom, AO offset, AO count)` of the real cell.
fn cell_shell_map(cell: &Cell) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
    let nbas = cell.mol.nbas;
    let mut sh_atom = vec![0usize; nbas];
    let mut sh_off = vec![0usize; nbas];
    let mut sh_cnt = vec![0usize; nbas];
    for s in 0..nbas {
        sh_atom[s] = cell.mol._bas[s * BAS_SLOTS + ATOM_OF] as usize;
    }
    for (s, off) in cell.mol.ao_loc_nr.iter().enumerate().take(nbas) {
        sh_off[s] = (*off).max(0) as usize;
    }
    for s in 0..nbas {
        let end = cell
            .mol
            .ao_loc_nr
            .get(s + 1)
            .copied()
            .unwrap_or(sh_off[s] as i32);
        sh_cnt[s] = (end.max(0) as usize).saturating_sub(sh_off[s]);
    }
    (sh_atom, sh_off, sh_cnt)
}

fn missing_rank(block: &crate::pseudo::HlBlock, dim: usize, i: usize) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "vppnl_nuc_grad: block (atom {}, l {}) claims {dim} projectors but \
         projector cell {i} is empty",
        block.atom, block.l
    )))
}

fn layout_error(
    block: &crate::pseudo::HlBlock,
    i: usize,
    ni: usize,
    row0: usize,
    nd: usize,
) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "vppnl_nuc_grad: projector cell {i} has {ni} AO rows but block \
         (atom {}, l {}) needs rows {row0}..{}",
        block.atom,
        block.l,
        row0 + nd
    )))
}
