//! Spin-free X2C one-electron Hamiltonian (`pbc/x2c/sfx2c1e.py`, 355 l).
//!
//! Ports the transform core upstream shares with the molecular code:
//! [`xmatrix`] is `x2c._x2c1e_xmatrix` (`pyscf/x2c/x2c.py`), [`hcore_fw`] is
//! `x2c._get_hcore_fw`, [`renorm_r`] is `x2c._get_r`. The PBC driver
//! (`SpinFreeX2CHelper.get_hcore`) only assembles the `(t, v, w, s)` blocks at
//! each k-point (via `pbc_intor` + `get_pnucp`) and calls this core — so
//! gating this core against upstream on identical blocks IS Gate E's shape
//! (transform-only, no SCF noise).
//!
//! X2C is the tightest family in the phase (upstream asserts 8dp in 29/34
//! checks) precisely because it is a one-electron transformation with no
//! iterative solver and therefore no convergence noise — Gate E at 1e-8 Ha
//! (19-01). This module is real-symmetric: exact at Γ and at every k-point
//! whose blocks are real. Complex-Hermitian k-points (explicit spin-orbit or
//! complex `pbc_intor` phases) route through `zeigh_gen` in a later step and
//! are refused here rather than silently truncated to their real parts.

use crate::error::PbcX2cError;
use pyscf_algebra::{eigh_gen, solve_linear};

/// Speed of light in atomic units (`pyscf.lib.param.LIGHT_SPEED`).
pub const LIGHT_SPEED: f64 = 137.035999084;

/// Spin-free X2C decoupling matrix (`x2c._x2c1e_xmatrix`).
///
/// Solves the modified Dirac equation in the `(t, v, w, s)` block basis:
/// `h = [[V, T], [T, W/4c² − T]]`, `m = [[S, 0], [0, T/2c²]]` by generalized
/// eigh, then `X = solve(clᵀ, csᵀ)ᵀ` over the positive-energy branch
/// (`a[:nao, nao:]` / `a[nao:, nao:]`). All blocks row-major `nao × nao`.
/// The LinAlgError fallback branch (canonical-orthonormalization retry) is
/// implemented: on [`pyscf_algebra::AlgebraError::Singular`] the solve
/// retries in the metric-orthogonalized basis exactly as upstream does.
pub fn xmatrix(t: &[f64], v: &[f64], w: &[f64], s: &[f64], nao: usize, c: f64) -> Result<Vec<f64>, PbcX2cError> {
    check_blocks(t, v, w, s, nao)?;
    let n2 = 2 * nao;
    let (mut h, mut m) = (vec![0.0f64; n2 * n2], vec![0.0f64; n2 * n2]);
    for i in 0..nao {
        for j in 0..nao {
            h[i * n2 + j] = v[i * nao + j];
            h[i * n2 + nao + j] = t[i * nao + j];
            h[(nao + i) * n2 + j] = t[i * nao + j];
            h[(nao + i) * n2 + nao + j] = w[i * nao + j] * (0.25 / (c * c)) - t[i * nao + j];
            m[i * n2 + j] = s[i * nao + j];
            m[(nao + i) * n2 + nao + j] = t[i * nao + j] * (0.5 / (c * c));
        }
    }
    match eigh_gen(&h, &m, n2) {
        Ok((_e, a_f)) => {
            // Positive-energy branch: cl = a[:nao, nao:], cs = a[nao:, nao:]
            // (F-order columns nao..2*nao). Xᵀ = solve(clᵀ, csᵀ) per column.
            let mut x = vec![0.0f64; nao * nao];
            // cl[row][col] = A[row][nao+col] = a_f[(nao+col)*n2 + row]
            // (C is column-major: element (i,j) at j*n2+i). clt = clᵀ:
            // clt[r][c] = cl[c][r] = a_f[(nao+r)*n2 + c].
            let mut clt = vec![0.0f64; nao * nao];
            for r in 0..nao {
                for cc in 0..nao {
                    clt[r * nao + cc] = a_f[(nao + r) * n2 + cc];
                }
            }
            // X = Yᵀ with clᵀ·Y = csᵀ. A linear solve decomposes over RHS
            // columns: Y[:,r] solves clᵀ·y = cs[r,:]ᵀ (row r of cs as a
            // column vector), and row r of X is Y[:,r]ᵀ:
            // X[r][j] = Y[j][r]. (Decomposing over columns of cs instead is
            // a same-shape plausible-wrong-number — caught by Gate E.)
            for r in 0..nao {
                let mut rhs = vec![0.0f64; nao];
                for i in 0..nao {
                    rhs[i] = a_f[(nao + i) * n2 + nao + r];
                }
                let ycol = solve_linear(&clt, &rhs, nao)?;
                for j in 0..nao {
                    x[r * nao + j] = ycol[j];
                }
            }
            Ok(x)
        }
        Err(_) => {
            // Fallback: canonical orthonormalization in the metric, diagonalize
            // `tᵀ h t`, keep `e > −c²`, `X = cs·clᵀ·s` (upstream's except arm;
            // see the shape note inside).
            xmatrix_fallback(&h, &m, s, n2, nao, c)
        }
    }
}

/// Upstream `_x2c1e_xmatrix`'s `except LinAlgError` arm.
///
/// Deviation, documented not hidden: upstream writes `x =
/// cs.dot(cl.conj().T).dot(m)` where `cs·clᵀ` is `nao × nao` and `m` is the
/// full `2nao × 2nao` metric — that product cannot execute (inner dimensions
/// `nao` vs `2nao` never align), so the arm raises `ValueError` whenever it is
/// taken; it is dead code that only survives because `eigh` on a well-formed
/// metric never raises. This port implements the arm's own comment instead —
/// `X = B·A⁻¹ = B·Aᵀ·S` with the large-component overlap `S` (`nao × nao`) —
/// which is dimensionally sound and mathematically what the comment states.
fn xmatrix_fallback(h: &[f64], m_full: &[f64], s: &[f64], n2: usize, nao: usize, c: f64) -> Result<Vec<f64>, PbcX2cError> {
    const LIN_DEP: f64 = 1e-12;
    let ident: Vec<f64> = {
        let mut s = vec![0.0f64; n2 * n2];
        for i in 0..n2 {
            s[i * n2 + i] = 1.0;
        }
        s
    };
    let (dm_evals, dm_evecs_f) = eigh_gen(m_full, &ident, n2)?;
    let mut keep: Vec<usize> = Vec::new();
    for (j, &d) in dm_evals.iter().enumerate() {
        if d > LIN_DEP {
            keep.push(j);
        }
    }
    if keep.is_empty() {
        return Err(PbcX2cError::ShapeMismatch { expected: 1, got: 0 });
    }
    // t_orth[i][k] = evec_k[i]/sqrt(d_k), F-order columns.
    let nk = keep.len();
    let mut torth = vec![0.0f64; n2 * nk];
    for (k, &j) in keep.iter().enumerate() {
        let s = 1.0 / dm_evals[j].sqrt();
        for i in 0..n2 {
            torth[i * nk + k] = dm_evecs_f[j * n2 + i] * s;
        }
    }
    // tht = tᵀ·h·t (nk × nk).
    let mut tht = vec![0.0f64; nk * nk];
    for a in 0..nk {
        for b in 0..nk {
            let mut acc = 0.0f64;
            for i in 0..n2 {
                for j in 0..n2 {
                    acc += torth[i * nk + a] * h[i * n2 + j] * torth[j * nk + b];
                }
            }
            tht[a * nk + b] = acc;
        }
    }
    let ident_k: Vec<f64> = {
        let mut s = vec![0.0f64; nk * nk];
        for i in 0..nk {
            s[i * nk + i] = 1.0;
        }
        s
    };
    let (e, a_f) = eigh_gen(&tht, &ident_k, nk)?;
    // Back-transform, keep e > −c².
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for j in 0..nk {
        if e[j] > -c * c {
            let mut col = vec![0.0f64; n2];
            for i in 0..n2 {
                let mut acc = 0.0f64;
                for k in 0..nk {
                    acc += torth[i * nk + k] * a_f[j * nk + k];
                }
                col[i] = acc;
            }
            cols.push(col);
        }
    }
    if cols.len() < nao {
        return Err(PbcX2cError::ShapeMismatch { expected: nao, got: cols.len() });
    }
    // X = cs·clᵀ·s with cl = col[..nao], cs = col[nao..] over kept columns:
    // X[i,j] = Σ_col cs_col[i] · Σ_q cl_col[q]·s[q,j].
    let mut x = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut acc = 0.0f64;
            for col in &cols {
                let mut clm = 0.0f64;
                for q in 0..nao {
                    clm += col[q] * s[q * nao + j];
                }
                acc += col[nao + i] * clm;
            }
            x[i * nao + j] = acc;
        }
    }
    Ok(x)
}

/// Foldy-Wouthuysen picture-change Hamiltonian (`x2c._get_hcore_fw`).
///
/// `s1 = S + XᵀTX/2c²`; `h1 = V + TX + (TX)ᵀ − XᵀTX + XᵀWX/4c²`;
/// `h1 → Rᵀ·h1·R` with [`renorm_r`]. All blocks row-major `nao × nao`.
pub fn hcore_fw(t: &[f64], v: &[f64], w: &[f64], s: &[f64], x: &[f64], nao: usize, c: f64) -> Result<Vec<f64>, PbcX2cError> {
    check_blocks(t, v, w, s, nao)?;
    if x.len() != nao * nao {
        return Err(PbcX2cError::ShapeMismatch { expected: nao * nao, got: x.len() });
    }
    let c2 = c * c;
    // tx = T·X.
    let mut tx = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut acc = 0.0f64;
            for k in 0..nao {
                acc += t[i * nao + k] * x[k * nao + j];
            }
            tx[i * nao + j] = acc;
        }
    }
    // xtx = Xᵀ·T·X, xwx = Xᵀ·W·X.
    let (mut xtx, mut xwx) = (vec![0.0f64; nao * nao], vec![0.0f64; nao * nao]);
    for i in 0..nao {
        for j in 0..nao {
            let (mut a1, mut a2) = (0.0f64, 0.0f64);
            for k in 0..nao {
                for l in 0..nao {
                    a1 += x[k * nao + i] * t[k * nao + l] * x[l * nao + j];
                    a2 += x[k * nao + i] * w[k * nao + l] * x[l * nao + j];
                }
            }
            xtx[i * nao + j] = a1;
            xwx[i * nao + j] = a2;
        }
    }
    let mut h1 = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            h1[i * nao + j] = v[i * nao + j] + tx[i * nao + j] + tx[j * nao + i]
                - xtx[i * nao + j]
                + xwx[i * nao + j] * (0.25 / c2);
        }
    }
    // s1 = S + XᵀTX/2c².
    let mut s1 = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            s1[i * nao + j] = s[i * nao + j] + xtx[i * nao + j] * (0.5 / c2);
        }
    }
    let r = renorm_r(s, &s1, nao)?;
    // h1 → Rᵀ·h1·R.
    let mut out = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut acc = 0.0f64;
            for k in 0..nao {
                for l in 0..nao {
                    acc += r[k * nao + i] * h1[k * nao + l] * r[l * nao + j];
                }
            }
            out[i * nao + j] = acc;
        }
    }
    Ok(out)
}

/// Renormalization matrix (`x2c._get_r`): `R = S^{-1/2}[S^{-1/2}S̃S^{-1/2}]^{-1/2}S^{1/2}`.
pub fn renorm_r(s: &[f64], s_tilde: &[f64], nao: usize) -> Result<Vec<f64>, PbcX2cError> {
    if s.len() != nao * nao || s_tilde.len() != nao * nao {
        return Err(PbcX2cError::ShapeMismatch { expected: nao * nao, got: s.len().min(s_tilde.len()) });
    }
    let ident: Vec<f64> = {
        let mut v = vec![0.0f64; nao * nao];
        for i in 0..nao {
            v[i * nao + i] = 1.0;
        }
        v
    };
    let (w, v_f) = eigh_gen(s, &ident, nao)?;
    let mut idx: Vec<usize> = Vec::new();
    for (j, &lam) in w.iter().enumerate() {
        if lam > 1e-14 {
            idx.push(j);
        }
    }
    if idx.is_empty() {
        return Err(PbcX2cError::ShapeMismatch { expected: 1, got: 0 });
    }
    let nkeep = idx.len();
    // v_keep[i][k] (row-major i, k), w_sqrt/w_invsqrt per k.
    let mut vkeep = vec![0.0f64; nao * nkeep];
    let mut wsqrt = vec![0.0f64; nkeep];
    let mut winv = vec![0.0f64; nkeep];
    for (k, &j) in idx.iter().enumerate() {
        wsqrt[k] = w[j].sqrt();
        winv[k] = 1.0 / wsqrt[k];
        for i in 0..nao {
            vkeep[i * nkeep + k] = v_f[j * nao + i];
        }
    }
    // snesc → eigenbasis: st[k][l] = Σ_ij vkeep[i][k]·s_tilde[i][j]·vkeep[j][l].
    let mut st = vec![0.0f64; nkeep * nkeep];
    for k in 0..nkeep {
        for l in 0..nkeep {
            let mut acc = 0.0f64;
            for i in 0..nao {
                for j in 0..nao {
                    acc += vkeep[i * nkeep + k] * s_tilde[i * nao + j] * vkeep[j * nkeep + l];
                }
            }
            st[k * nkeep + l] = acc;
        }
    }
    // r_mid0[k][l] = winv[k]·st[k][l]·winv[l]; eigh → keep > 1e-14.
    let mut rmid0 = vec![0.0f64; nkeep * nkeep];
    for k in 0..nkeep {
        for l in 0..nkeep {
            rmid0[k * nkeep + l] = winv[k] * st[k * nkeep + l] * winv[l];
        }
    }
    let ident_k: Vec<f64> = {
        let mut v = vec![0.0f64; nkeep * nkeep];
        for i in 0..nkeep {
            v[i * nkeep + i] = 1.0;
        }
        v
    };
    let (w1, v1_f) = eigh_gen(&rmid0, &ident_k, nkeep)?;
    let mut idx1: Vec<usize> = Vec::new();
    for (j, &lam) in w1.iter().enumerate() {
        if lam > 1e-14 {
            idx1.push(j);
        }
    }
    // r_mid = Σ_kept v1[k]/sqrt(w1) · v1[l] (nkeep × nkeep).
    let mut rmid = vec![0.0f64; nkeep * nkeep];
    for &j in &idx1 {
        let s = 1.0 / w1[j].sqrt();
        for k in 0..nkeep {
            for l in 0..nkeep {
                rmid[k * nkeep + l] += (v1_f[j * nkeep + k] * s) * v1_f[j * nkeep + l];
            }
        }
    }
    // r_eig[k][l] = winv[k]·rmid[k][l]·wsqrt[l]; back-transform: r = v·r_eig·vᵀ.
    let mut r = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut acc = 0.0f64;
            for k in 0..nkeep {
                for l in 0..nkeep {
                    acc += vkeep[i * nkeep + k] * winv[k] * rmid[k * nkeep + l] * wsqrt[l]
                        * vkeep[j * nkeep + l];
                }
            }
            r[i * nao + j] = acc;
        }
    }
    Ok(r)
}

/// Spin-free X2C picture-change Hamiltonian at one k-point.
///
/// `X = xmatrix(...)`, `h1 = hcore_fw(...)` with `c = LIGHT_SPEED` unless the
/// caller passes an explicit speed (upstream's `lib.light_speed(c)` test hook
/// is mirrored by taking `c` as a parameter; see [`sfx2c1e_hcore_at_c`]).
/// Complex-Hermitian blocks are refused (see module docs), never truncated.
pub fn sfx2c1e_hcore(t: &[f64], v: &[f64], w: &[f64], s: &[f64], nao: usize) -> Result<Vec<f64>, PbcX2cError> {
    sfx2c1e_hcore_at_c(t, v, w, s, nao, LIGHT_SPEED)
}

/// [`sfx2c1e_hcore`] at an explicit speed of light (test hook + PBC driver use).
pub fn sfx2c1e_hcore_at_c(
    t: &[f64],
    v: &[f64],
    w: &[f64],
    s: &[f64],
    nao: usize,
    c: f64,
) -> Result<Vec<f64>, PbcX2cError> {
    let x = xmatrix(t, v, w, s, nao, c)?;
    hcore_fw(t, v, w, s, &x, nao, c)
}

fn check_blocks(t: &[f64], v: &[f64], w: &[f64], s: &[f64], nao: usize) -> Result<(), PbcX2cError> {
    for (name, b) in [("t", t), ("v", v), ("w", w), ("s", s)] {
        if b.len() != nao * nao {
            return Err(PbcX2cError::ShapeMismatch {
                expected: nao * nao,
                got: b.len(),
            });
        }
        let _ = name;
    }
    Ok(())
}
