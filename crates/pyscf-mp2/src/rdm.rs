//! `make_rdm1` / `make_rdm2` + `gamma1_intermediates` (MP2-05).
//!
//! Ports the closed-shell (RMP2) reduced density matrices from
//! `pyscf/mp/mp2.py` (`make_rdm1:151`, `_gamma1_intermediates:175`,
//! `make_rdm2:275`) plus the `ccsd_rdm._make_rdm1:246` assembly the upstream
//! `make_rdm1` delegates to.
//!
//! Port order follows Pitfall 5: `make_rdm1` (the simpler doo/dvv assembly)
//! FIRST, unit-tested, THEN `make_rdm2` (the `nmo^4` occ/vir/frozen sub-block
//! tensor) layered on top with explicit flat-index maps doc-commented at every
//! placement (T-05-04-LAYOUT — a silent transpose corrupts the RDM).
//!
//! Every contraction materializes its products into a slice then reduces via
//! [`oracle_sum`]/[`oracle_dot`] (NO `+=` accumulation) — bit-exact and
//! thread-count invariant (T-05-04-FP, Pitfall 1/2).
//!
//! All index ranges are validated before use; the `make_rdm2` fancy-indexing
//! NEVER indexes OOB (T-05-04-FFI, Pitfall 5).
//!
//! ## Amplitude layout consumed
//! The stored RMP2 [`Amplitudes`] `t2` is flat `[nocc,nocc,nvir,nvir]` C-order:
//! element `(i,j,a,b)` at `((i*nocc + j)*nvir + a)*nvir + b` (the layout
//! `rmp2_kernel` produces). `t2[i]` (the upstream per-`i` block) is the
//! contiguous `[nocc,nvir,nvir]` slice starting at `i*nocc*nvir*nvir`, indexed
//! `[j,a,b]` at `(j*nvir + a)*nvir + b`.

use crate::frozen::{self, Frozen};
use pyscf_algebra::oracle_sum;
use pyscf_core::{Amplitudes, Density, MOCoefficients, PyscfRsError};

/// MP2 `_gamma1_intermediates` (`mp2.py:175-203`): the occ-occ (`doo`) and
/// vir-vir (`dvv`) correlation blocks of the 1-RDM.
///
/// For each occupied `i`, with the per-`i` block `t2i[j,a,b] = t2[i,j,a,b]`
/// (and `l2i == t2i` for real amplitudes):
///
/// - `dm1vir[b,a] += 2·Σ_{j,c} t2i[j,c,a]·t2i[j,c,b] − Σ_{j,c} t2i[j,c,a]·t2i[j,b,c]`
/// - `dm1occ[p,q] += 2·Σ_{a,b} t2i[p,a,b]·t2i[q,a,b] − Σ_{a,b} t2i[p,a,b]·t2i[q,b,a]`
///
/// Returns `(doo, dvv)` where `doo = −dm1occ` (flat `[nocc,nocc]` row-major,
/// element `(p,q)` at `p*nocc + q`) and `dvv = dm1vir` (flat `[nvir,nvir]`
/// row-major, element `(b,a)` at `b*nvir + a`) — matching the upstream
/// `return -dm1occ, dm1vir`.
///
/// Each scalar `dm1*[..]` entry is a fresh per-`(i)` `Σ` materialized into a
/// scratch `Vec` and reduced via [`oracle_sum`] (T-05-04-FP).
///
/// # Errors
/// Returns [`PyscfRsError`] (via [`crate::error::Mp2Error::ShapeMismatch`]) when
/// `t2.t2_slice().len() != nocc*nocc*nvir*nvir`. Never panics / indexes OOB.
pub fn gamma1_intermediates(
    t2: &Amplitudes,
    nocc: usize,
    nvir: usize,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    // D-01: t2 is now an opaque AmplitudeStore. MP2 RDMs always receive owned
    // (in-memory) amplitudes; pull the resident slice (a pooled CCSD store has
    // no resident slice and is not a valid MP2 RDM input).
    let t2_data = t2.t2_slice().ok_or(crate::error::Mp2Error::ShapeMismatch {
        expected: nocc * nocc * nvir * nvir,
        got: 0,
    })?;
    let expected = nocc * nocc * nvir * nvir;
    if t2_data.len() != expected {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected,
            got: t2_data.len(),
        }
        .into());
    }
    // Flat (i,j,a,b) index into t2_data (C-order [nocc,nocc,nvir,nvir]).
    let t = |i: usize, j: usize, a: usize, b: usize| t2_data[((i * nocc + j) * nvir + a) * nvir + b];

    // dm1occ[p,q] = Σ_i [ 2·Σ_ab t2[i,p,a,b]·t2[i,q,a,b] − Σ_ab t2[i,p,a,b]·t2[i,q,b,a] ].
    let mut doo = vec![0.0_f64; nocc * nocc]; // doo = −dm1occ.
    let mut buf_ab: Vec<f64> = Vec::with_capacity(nvir * nvir);
    for p in 0..nocc {
        for q in 0..nocc {
            // Accumulate the per-i contributions into a single Vec, then reduce.
            let mut per_i: Vec<f64> = Vec::with_capacity(nocc);
            for i in 0..nocc {
                // 2·Σ_ab t2[i,p,a,b]·t2[i,q,a,b].
                buf_ab.clear();
                for a in 0..nvir {
                    for b in 0..nvir {
                        buf_ab.push(t(i, p, a, b) * t(i, q, a, b));
                    }
                }
                let direct = 2.0 * oracle_sum(&buf_ab);
                // Σ_ab t2[i,p,a,b]·t2[i,q,b,a].
                buf_ab.clear();
                for a in 0..nvir {
                    for b in 0..nvir {
                        buf_ab.push(t(i, p, a, b) * t(i, q, b, a));
                    }
                }
                let exch = oracle_sum(&buf_ab);
                per_i.push(direct - exch);
            }
            // doo = −dm1occ.
            doo[p * nocc + q] = -oracle_sum(&per_i);
        }
    }

    // dm1vir[b,a] = Σ_i [ 2·Σ_jc t2[i,j,c,a]·t2[i,j,c,b] − Σ_jc t2[i,j,c,a]·t2[i,j,b,c] ].
    let mut dvv = vec![0.0_f64; nvir * nvir];
    let mut buf_jc: Vec<f64> = Vec::with_capacity(nocc * nvir);
    for b in 0..nvir {
        for a in 0..nvir {
            let mut per_i: Vec<f64> = Vec::with_capacity(nocc);
            for i in 0..nocc {
                // 2·Σ_jc t2[i,j,c,a]·t2[i,j,c,b].
                buf_jc.clear();
                for j in 0..nocc {
                    for c in 0..nvir {
                        buf_jc.push(t(i, j, c, a) * t(i, j, c, b));
                    }
                }
                let direct = 2.0 * oracle_sum(&buf_jc);
                // Σ_jc t2[i,j,c,a]·t2[i,j,b,c].
                buf_jc.clear();
                for j in 0..nocc {
                    for c in 0..nvir {
                        buf_jc.push(t(i, j, c, a) * t(i, j, b, c));
                    }
                }
                let exch = oracle_sum(&buf_jc);
                per_i.push(direct - exch);
            }
            dvv[b * nvir + a] = oracle_sum(&per_i);
        }
    }

    Ok((doo, dvv))
}

/// MP2 spin-traced one-particle reduced density matrix (`mp2.py:make_rdm1:151`
/// → `ccsd_rdm._make_rdm1:246`).
///
/// Assembles the MO-basis `nmo×nmo` 1-RDM from the `doo`/`dvv` intermediates
/// (`dov = dvo = 0` for MP2): with `nact = nocc + nvir` the ACTIVE space,
/// - `dm[p,q] = doo[p,q] + doo[q,p]` for `p,q` occupied
/// - `dm[a',b'] = dvv[a',b'] + dvv[b',a']` for `a',b'` virtual
/// - the occupied diagonal gets `+2` (the mean-field reference, `with_mf`)
///
/// When `with_frozen` and `frozen != None`, the active block is embedded into a
/// full `nmo×nmo` matrix with the frozen-occupied diagonal set to `2`. When
/// `ao_repr`, the MO-basis matrix is back-transformed `C·γ·Cᵀ` to the AO basis.
///
/// The returned [`Density`] holds the matrix row-major in `data`, with `nao`
/// set to the matrix dimension (`nmo` for the MO-basis result, the AO count for
/// the `ao_repr` result).
///
/// # Errors
/// Propagates the [`gamma1_intermediates`] shape check and the frozen-mask
/// resolution; validates the AO-transform dimension. Never panics / OOB.
// Mirrors the upstream `make_rdm1(mp, t2, eris, ao_repr, with_frozen)` surface
// plus the explicit `mo_coeff`/`mo_occ` (pyo3-free: no `mp` instance to read
// them from) — the argument count is the upstream contract, not incidental.
#[allow(clippy::too_many_arguments)]
pub fn make_rdm1(
    t2: &Amplitudes,
    nocc: usize,
    nmo: usize,
    frozen: &Frozen,
    ao_repr: bool,
    with_frozen: bool,
    mo_coeff: &MOCoefficients,
    mo_occ: &[f64],
) -> Result<Density, PyscfRsError> {
    let nvir = nmo - nocc;
    let (doo, dvv) = gamma1_intermediates(t2, nocc, nvir)?;

    // Active-space MO-basis 1-RDM (nact = nocc + nvir).
    let nact = nocc + nvir;
    let mut dm = vec![0.0_f64; nact * nact];
    // Occupied-occupied block: dm[p,q] = doo[p,q] + doo[q,p].
    for p in 0..nocc {
        for q in 0..nocc {
            dm[p * nact + q] = doo[p * nocc + q] + doo[q * nocc + p];
        }
    }
    // Virtual-virtual block: dm[nocc+a, nocc+b] = dvv[a,b] + dvv[b,a].
    for a in 0..nvir {
        for b in 0..nvir {
            dm[(nocc + a) * nact + (nocc + b)] = dvv[a * nvir + b] + dvv[b * nvir + a];
        }
    }
    // Mean-field reference: occupied diagonal += 2.
    for p in 0..nocc {
        dm[p * nact + p] += 2.0;
    }

    // Embed into the full nmo space when frozen orbitals are present.
    let (dim, mo_dm) = if with_frozen && *frozen != Frozen::None {
        let elements: Vec<u32> = Vec::new(); // chemcore handled at kernel layer; mask via mo_occ.
        let mask = frozen::frozen_mask(frozen, mo_occ, &[], &elements)?;
        let full_nmo = mo_occ.len();
        // The active columns (mask==true), in column order.
        let active_cols: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(c, &a)| if a { Some(c) } else { None })
            .collect();
        if active_cols.len() != nact {
            return Err(crate::error::Mp2Error::ShapeMismatch {
                expected: nact,
                got: active_cols.len(),
            }
            .into());
        }
        let mut full = vec![0.0_f64; full_nmo * full_nmo];
        // Frozen-occupied diagonal = 2 (mean-field core, with_mf).
        for (c, &occ) in mo_occ.iter().enumerate() {
            if occ > 0.0 && !mask[c] {
                full[c * full_nmo + c] = 2.0;
            }
        }
        // Scatter the active block into the active rows/columns:
        // full[active[p], active[q]] = dm[p,q].
        for (p, &rp) in active_cols.iter().enumerate() {
            for (q, &rq) in active_cols.iter().enumerate() {
                full[rp * full_nmo + rq] = dm[p * nact + q];
            }
        }
        (full_nmo, full)
    } else {
        (nact, dm)
    };

    if !ao_repr {
        return Ok(Density {
            nao: dim,
            data: mo_dm,
        });
    }

    // AO representation: dm_ao = C · dm_mo · Cᵀ.
    // mo_coeff.data is column-major [nao, nmo_full]: C[ao, mo] at ao + mo*nao.
    let nao = mo_coeff.nao;
    let n = dim; // MO dimension of mo_dm.
    if mo_coeff.nmo < n {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: n,
            got: mo_coeff.nmo,
        }
        .into());
    }
    let c = |ao: usize, mo: usize| mo_coeff.data[ao + mo * nao];
    // tmp[ao, q] = Σ_p C[ao,p]·dm[p,q].
    let mut tmp = vec![0.0_f64; nao * n];
    let mut buf: Vec<f64> = Vec::with_capacity(n);
    for ao in 0..nao {
        for q in 0..n {
            buf.clear();
            for p in 0..n {
                buf.push(c(ao, p) * mo_dm[p * n + q]);
            }
            tmp[ao * n + q] = oracle_sum(&buf);
        }
    }
    // dm_ao[ao, bo] = Σ_q tmp[ao,q]·C[bo,q].
    let mut dm_ao = vec![0.0_f64; nao * nao];
    for ao in 0..nao {
        for bo in 0..nao {
            buf.clear();
            for q in 0..n {
                buf.push(tmp[ao * n + q] * c(bo, q));
            }
            dm_ao[ao * nao + bo] = oracle_sum(&buf);
        }
    }
    Ok(Density { nao, data: dm_ao })
}

/// MP2 spin-traced two-particle reduced density matrix (`mp2.py:make_rdm2:275`).
///
/// Builds the `nmo0^4` Chemist-notation 2-RDM. The frozen path (`frozen !=
/// None`) embeds active occ/vir sub-blocks into the FULL `nmo0` space; the
/// no-frozen path uses the active space directly (`nmo0 == nact`).
///
/// Layout: `dm2` is flat C-order `[nmo0,nmo0,nmo0,nmo0]`; element `(p,q,r,s)`
/// lives at `((p*nmo0 + q)*nmo0 + r)*nmo0 + s`.
///
/// Build steps (mirroring upstream):
/// 1. `dm1 = make_rdm1(...)`, then `dm1[diag(nocc0)] -= 2` (the separable HF
///    diagonal is added back explicitly in step 4).
/// 2. For each occupied `i`, `dovov[a,j,b] = 2·(2·t2i[j,a,b] − t2i[j,b,a])`
///    (the `dovov = (t2i.transpose(1,0,2)*2 − t2i.transpose(2,0,1)) * 2`), placed
///    at `dm2[oi, va, oj, vb]` and the Chemist-transpose at
///    `dm2[va, oi, vb, oj]`.
/// 3. The dm1 contribution: for each occupied `i0`,
///    `dm2[i0,i0,:,:] += dm1ᵀ·2`, `dm2[:,:,i0,i0] += dm1ᵀ·2`,
///    `dm2[:,i0,i0,:] −= dm1ᵀ`, `dm2[i0,:,:,i0] −= dm1`.
/// 4. The separable HF part: `dm2[i,i,j,j] += 4`, `dm2[i,j,j,i] −= 2`.
/// 5. When `ao_repr`, back-transform to the AO basis (NotYetImplemented this
///    plan — the nmo^4 AO transform lands with the gradient consumer; returns
///    `NotYetImplemented` so callers never get a silently wrong tensor).
///
/// # Errors
/// Propagates the [`make_rdm1`] / shape checks; returns
/// [`crate::error::Mp2Error::NotYetImplemented`] for `ao_repr=true` (the
/// nmo^4 AO transform is deferred). Never panics / indexes OOB (T-05-04-FFI).
#[allow(clippy::too_many_arguments)]
pub fn make_rdm2(
    t2: &Amplitudes,
    nocc: usize,
    nmo: usize,
    frozen: &Frozen,
    ao_repr: bool,
    mo_coeff: &MOCoefficients,
    mo_occ: &[f64],
) -> Result<Vec<f64>, PyscfRsError> {
    if ao_repr {
        // The nmo^4 AO back-transform is deferred to the gradient consumer
        // (Phase 7); never substitute a wrong tensor (T-05-04-FFI).
        return Err(crate::error::Mp2Error::NotYetImplemented { plan: 4 }.into());
    }
    let nvir = nmo - nocc;
    let nact = nocc + nvir;

    // Resolve the frozen embedding. nmo0/nocc0 are the FULL counts; oidx/vidx
    // map active occ/vir positions to full-space orbital indices.
    let (nmo0, nocc0, oidx, vidx) = if *frozen != Frozen::None {
        let mask = frozen::frozen_mask(frozen, mo_occ, &[], &[])?;
        let full_nmo = mo_occ.len();
        let full_nocc = mo_occ.iter().filter(|&&o| o > 0.0).count();
        // Active occupied / virtual full-space indices (mask && occ>0 / mask && occ==0).
        let oidx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(c, &a)| {
                if a && mo_occ.get(c).copied().unwrap_or(0.0) > 0.0 {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        let vidx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(c, &a)| {
                if a && mo_occ.get(c).copied().unwrap_or(0.0) == 0.0 {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        if oidx.len() != nocc || vidx.len() != nvir {
            return Err(crate::error::Mp2Error::ShapeMismatch {
                expected: nocc * nvir,
                got: oidx.len() * vidx.len(),
            }
            .into());
        }
        (full_nmo, full_nocc, oidx, vidx)
    } else {
        // No frozen: active == full; oidx = 0..nocc, vidx = nocc..nact.
        let oidx: Vec<usize> = (0..nocc).collect();
        let vidx: Vec<usize> = (nocc..nact).collect();
        (nact, nocc, oidx, vidx)
    };

    // D-01: pull the resident slice from the opaque AmplitudeStore (MP2 RDMs
    // always receive owned amplitudes).
    let t2_data = t2.t2_slice().ok_or(crate::error::Mp2Error::ShapeMismatch {
        expected: nocc * nocc * nvir * nvir,
        got: 0,
    })?;
    let expected_t2 = nocc * nocc * nvir * nvir;
    if t2_data.len() != expected_t2 {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: expected_t2,
            got: t2_data.len(),
        }
        .into());
    }

    // 1. dm1 (MO basis, NOT ao_repr, with_frozen) then subtract 2 on the
    //    occupied-full diagonal.
    let dm1 = make_rdm1(t2, nocc, nmo, frozen, false, true, mo_coeff, mo_occ)?;
    let mut dm1 = dm1.data; // flat [nmo0,nmo0] row-major.
    for i in 0..nocc0 {
        dm1[i * nmo0 + i] -= 2.0;
    }
    let d1 = |p: usize, q: usize| dm1[p * nmo0 + q];

    // dm2 flat C-order [nmo0,nmo0,nmo0,nmo0].
    let mut dm2 = vec![0.0_f64; nmo0 * nmo0 * nmo0 * nmo0];
    let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * nmo0 + q) * nmo0 + r) * nmo0 + s;
    // t2 per-i block accessor: t2i[j,a,b] = t2[i,j,a,b].
    let t = |i: usize, j: usize, a: usize, b: usize| t2_data[((i * nocc + j) * nvir + a) * nvir + b];

    // 2. dovov placement (mp2.py:309-327).
    //    dovov[a,j,b] = (2·t2i[j,a,b] − t2i[j,b,a]) · 2.
    //    dm2[oidx[i], vidx[a], oidx[j], vidx[b]] = dovov[a,j,b]
    //    dm2[vidx[a], oidx[i], vidx[b], oidx[j]] = dovov_conj.transpose(0,2,1)[a,j,b]
    //      = dovov[a,b... ] — the conj-transpose(0,2,1) swaps the (j,b) axes:
    //      placed value at [va, oi, vb, oj] equals dovov[a, j, b] with axes
    //      (a, b, j) → here we place dovov[a,j,b] at [va,oi,vb,oj] using the
    //      transpose(0,2,1): element [a, j, b] of dovov maps to position
    //      [a, b, j] of the transposed tensor, i.e. dm2[va, oi, vb, oj] gets
    //      dovov[a, j, b] read as transposed → see explicit loop below.
    for i in 0..nocc {
        for a in 0..nvir {
            for j in 0..nocc {
                for b in 0..nvir {
                    let dovov = (2.0 * t(i, j, a, b) - t(i, j, b, a)) * 2.0;
                    let oi = oidx[i];
                    let oj = oidx[j];
                    let va = vidx[a];
                    let vb = vidx[b];
                    // dm2[i,nocc:,:nocc,nocc:] = dovov  (p=oi, q=va, r=oj, s=vb).
                    dm2[idx4(oi, va, oj, vb)] = dovov;
                    // dm2[nocc:,i,nocc:,:nocc] = dovov.transpose(0,2,1):
                    //   element (a,j,b) → stored at axis order (a,b,j), so
                    //   dm2[va, oi, vb, oj] = dovov(a, j, b).
                    dm2[idx4(va, oi, vb, oj)] = dovov;
                }
            }
        }
    }

    // 3. dm1 contribution (mp2.py:334-338). dm1ᵀ[p,q] = d1(q,p).
    //    dm2[i0,i0,:,:] += dm1ᵀ·2 ; dm2[:,:,i0,i0] += dm1ᵀ·2 ;
    //    dm2[:,i0,i0,:] −= dm1ᵀ   ; dm2[i0,:,:,i0] −= dm1.
    for i0 in 0..nocc0 {
        for p in 0..nmo0 {
            for q in 0..nmo0 {
                let dt = d1(q, p); // dm1ᵀ[p,q]
                dm2[idx4(i0, i0, p, q)] += 2.0 * dt;
                dm2[idx4(p, q, i0, i0)] += 2.0 * dt;
                dm2[idx4(p, i0, i0, q)] -= dt;
                dm2[idx4(i0, p, q, i0)] -= d1(p, q); // − dm1 (not transposed)
            }
        }
    }

    // 4. separable HF part (mp2.py:340-343).
    for i in 0..nocc0 {
        for j in 0..nocc0 {
            dm2[idx4(i, i, j, j)] += 4.0;
            dm2[idx4(i, j, j, i)] -= 2.0;
        }
    }

    Ok(dm2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a toy RMP2 `Amplitudes` with given (nocc, nvir) and a fill closure
    /// over (i,j,a,b).
    fn toy_amps(
        nocc: usize,
        nvir: usize,
        fill: impl Fn(usize, usize, usize, usize) -> f64,
    ) -> Amplitudes {
        let mut t2 = vec![0.0; nocc * nocc * nvir * nvir];
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        t2[((i * nocc + j) * nvir + a) * nvir + b] = fill(i, j, a, b);
                    }
                }
            }
        }
        Amplitudes::from_vec(nocc, nvir, Vec::new(), t2)
    }

    fn identity_mo(nmo: usize, mo_occ: &[f64], mo_energy: &[f64]) -> MOCoefficients {
        let mut data = vec![0.0; nmo * nmo];
        for i in 0..nmo {
            data[i * nmo + i] = 1.0;
        }
        MOCoefficients {
            nao: nmo,
            nmo,
            data,
            energies: mo_energy.to_vec(),
            occupations: mo_occ.to_vec(),
        }
    }

    #[test]
    fn gamma1_shapes() {
        let nocc = 2;
        let nvir = 3;
        let amps = toy_amps(nocc, nvir, |i, j, a, b| {
            0.01 * (i + 1) as f64
                + 0.02 * (j + 1) as f64
                + 0.03 * (a + 1) as f64
                + 0.04 * (b + 1) as f64
        });
        let (doo, dvv) = gamma1_intermediates(&amps, nocc, nvir).expect("gamma1");
        assert_eq!(doo.len(), nocc * nocc);
        assert_eq!(dvv.len(), nvir * nvir);
    }

    #[test]
    fn make_rdm1_trace_equals_nelec_no_frozen() {
        // No-frozen closed-shell: Tr(γ) must equal nelec = 2·nocc. The
        // correlation correction is traceless (doo is occ-occ, dvv vir-vir, and
        // doo+dooᵀ / dvv+dvvᵀ contributions cancel against the occupied
        // diagonal at the mean-field level only in the FULL nelec sense:
        // Tr(γ) = 2·nocc + Tr(doo+dooᵀ) + Tr(dvv+dvvᵀ); MP2 1-RDM IS NOT
        // idempotent so the trace is conserved to 2·nocc only when the doo/dvv
        // traces cancel — which they do for the spin-traced MP2 1-RDM since the
        // number of electrons is conserved by construction).
        let nocc = 2;
        let nvir = 2;
        let nmo = nocc + nvir;
        let amps = toy_amps(nocc, nvir, |i, j, a, b| 0.05 / ((i + j + a + b + 2) as f64));
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mo = identity_mo(nmo, &mo_occ, &[-1.0, -0.9, 0.5, 0.7]);
        let dm =
            make_rdm1(&amps, nocc, nmo, &Frozen::None, false, true, &mo, &mo_occ).expect("rdm1");
        assert_eq!(dm.nao, nmo);
        assert_eq!(dm.data.len(), nmo * nmo);
        let trace: f64 = (0..nmo).map(|p| dm.data[p * nmo + p]).sum();
        let nelec = 2.0 * nocc as f64;
        assert!(
            (trace - nelec).abs() < 1e-10,
            "Tr(gamma) = {} but nelec = {}",
            trace,
            nelec
        );
    }

    #[test]
    fn make_rdm1_with_frozen_places_core_diagonal() {
        // Freeze the inner-most occupied orbital (Count(1)) of a 4-orbital
        // system [occ, occ, vir, vir]. Active space: nocc=1, nvir=2 → nact=3,
        // embedded into the full nmo=4. The full nocc is 2 (orbitals 0,1), so
        // Tr(γ) must equal nelec = 2·2 = 4 (the frozen-core electrons are placed
        // on the full diagonal as 2, and the trace is conserved by the active
        // MP2 1-RDM as in the no-frozen case).
        let full_nmo = 4;
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let nocc = 1; // active occupied after freezing the inner 1
        let nvir = 2; // both virtuals are active
        let nmo = nocc + nvir; // active nmo passed to the kernel
        let amps = toy_amps(nocc, nvir, |_, _, a, b| 0.03 * (a + b + 1) as f64);
        let mo = identity_mo(full_nmo, &mo_occ, &[-2.0, -1.0, 0.5, 0.7]);
        let dm = make_rdm1(
            &amps,
            nocc,
            nmo,
            &Frozen::Count(1),
            false,
            true,
            &mo,
            &mo_occ,
        )
        .expect("rdm1 frozen");
        assert_eq!(dm.nao, full_nmo);
        // Frozen-occupied diagonal (orbital 0) must be exactly 2.
        assert!((dm.data[0] - 2.0).abs() < 1e-12, "frozen core diag != 2");
        let trace: f64 = (0..full_nmo).map(|p| dm.data[p * full_nmo + p]).sum();
        assert!(
            (trace - 4.0).abs() < 1e-10,
            "frozen Tr(gamma) = {} but nelec = 4",
            trace
        );
    }

    #[test]
    fn make_rdm2_length_is_nmo4_no_frozen() {
        let nocc = 2;
        let nvir = 2;
        let nmo = nocc + nvir;
        let amps = toy_amps(nocc, nvir, |i, j, a, b| {
            0.02 * (i + 1) as f64 + 0.01 * (j + a + b + 1) as f64
        });
        let mo_occ = [2.0, 2.0, 0.0, 0.0];
        let mo = identity_mo(nmo, &mo_occ, &[-1.0, -0.9, 0.5, 0.7]);
        let dm2 = make_rdm2(&amps, nocc, nmo, &Frozen::None, false, &mo, &mo_occ).expect("rdm2");
        assert_eq!(dm2.len(), nmo * nmo * nmo * nmo);
    }

    #[test]
    fn make_rdm2_known_subblock_placement() {
        // nocc=1, nvir=2 so the dovov antisymmetrizer distinguishes (a,b) from
        // (b,a). t2[0,0,a,b] picked asymmetric: t[0,0,0,1]=0.3, t[0,0,1,0]=0.1,
        // t[0,0,0,0]=t[0,0,1,1]=0. dovov[a,j=0,b] = (2·t[0,0,a,b] − t[0,0,b,a])·2.
        // For (a=0,b=1): (2·0.3 − 0.1)·2 = 1.0. For (a=1,b=0): (2·0.1 − 0.3)·2 = −0.2.
        let nocc = 1;
        let nvir = 2;
        let nmo = nocc + nvir; // 3; oidx=[0], vidx=[1,2].
        let amps = toy_amps(nocc, nvir, |_, _, a, b| match (a, b) {
            (0, 1) => 0.3,
            (1, 0) => 0.1,
            _ => 0.0,
        });
        let mo_occ = [2.0, 0.0, 0.0];
        let mo = identity_mo(nmo, &mo_occ, &[-1.0, 0.5, 0.7]);
        let dm2 = make_rdm2(&amps, nocc, nmo, &Frozen::None, false, &mo, &mo_occ).expect("rdm2");
        let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * nmo + q) * nmo + r) * nmo + s;
        // dovov placement: dm2[oi=0, va=vidx[a], oj=0, vb=vidx[b]] = dovov[a,0,b].
        // (a=0,b=1): vidx[0]=1, vidx[1]=2 → dm2[0,1,0,2] = 1.0.
        assert!(
            (dm2[idx4(0, 1, 0, 2)] - 1.0).abs() < 1e-12,
            "dovov(0,1) misplaced: {}",
            dm2[idx4(0, 1, 0, 2)]
        );
        // (a=1,b=0): dm2[0,2,0,1] = −0.2.
        assert!(
            (dm2[idx4(0, 2, 0, 1)] - (-0.2)).abs() < 1e-12,
            "dovov(1,0) misplaced: {}",
            dm2[idx4(0, 2, 0, 1)]
        );
        // Chemist transpose: dm2[va,oi,vb,oj] = dovov[a,0,b].
        // (a=0,b=1): dm2[1,0,2,0] = 1.0.
        assert!((dm2[idx4(1, 0, 2, 0)] - 1.0).abs() < 1e-12);
        // (a=1,b=0): dm2[2,0,1,0] = −0.2.
        assert!((dm2[idx4(2, 0, 1, 0)] - (-0.2)).abs() < 1e-12);
        // HF separable on the occ-occ corner: dm2[0,0,0,0] += 4 then −= 2
        // (i=j=0 of both HF terms) plus the dm1 contribution. Must be finite.
        assert!(dm2[idx4(0, 0, 0, 0)].is_finite());
    }

    #[test]
    fn make_rdm2_ao_repr_is_not_yet_implemented() {
        let nocc = 1;
        let nvir = 1;
        let nmo = 2;
        let amps = toy_amps(nocc, nvir, |_, _, _, _| 0.5);
        let mo_occ = [2.0, 0.0];
        let mo = identity_mo(nmo, &mo_occ, &[-1.0, 0.5]);
        let r = make_rdm2(&amps, nocc, nmo, &Frozen::None, true, &mo, &mo_occ);
        assert!(r.is_err(), "ao_repr should be NotYetImplemented this plan");
    }
}
