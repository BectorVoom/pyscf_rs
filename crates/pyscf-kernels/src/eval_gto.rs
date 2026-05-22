//! GTO-07 + D-04: AO-on-grid kernel (`eval_gto`).
//!
//! Source: pyscf/gto/eval_gto.py (Apache-2.0). The reference algorithm:
//! per grid point g, walk every shell s, compute the contracted radial
//! `Σ_p coeff[c,p] * exp(-α_p * r²)`, optionally apply the cart→sph
//! harmonic transform, and write to `out[g, ao_idx]` in F-order.
//!
//! Phase 2 shipped the **s-shell only** implementation. Phase 4 plan
//! 04-03 (this commit) lands the deferred `l ≥ 1` path: per shell with
//! angular momentum `l`, evaluate the `ncart(l) = (l+1)(l+2)/2`
//! cartesian monomials `x^lx y^ly z^lz · R(r)` (upstream loop order
//! `lx=l..0, ly=l-lx..0, lz=l-lx-ly`), then apply the libcint
//! cart→sph transform (`g_trans_cart2sph[]`, the same Condon-Shortley
//! coefficients libcint's `CINTc2s_ket_sph` uses) to produce the
//! `2l+1` real spherical-harmonic AO components. For `l = 0, 1` the
//! cart→sph transform is the identity and the angular factor lives in
//! `CINTcommon_fac_sp(l)`; for `l ≥ 2` the factor is folded into the
//! c2s matrix. The contracted-radial sum (> 2 prims) is routed through
//! `pyscf_algebra::oracle_sum` for FMA-free, thread-order-independent
//! reduction (Pitfall 3 / FOUND-06 — byte-exact discipline).
//!
//! Reference algorithm: `pyscf/lib/gto/deriv1.c GTOshell_eval_grid_cart`
//! (cartesian monomial eval) + `pyscf/gto/mole.py cart2sph` →
//! `CINTc2s_ket_sph` (libcint `cart2sph.c g_trans_cart2sph[]`). The c2s
//! coefficient tables below are byte-identical to cintx-cubecl
//! `transform::c2s::C2S_L{0..4}` (libcint provenance).
//!
//! The `GTOval_sph_deriv1` variant (value + 3 gradient components) is the
//! GGA grid-loop input; it is implemented alongside the value path here
//! (plan 04-03 Task 2). `GTOval_sph_deriv2` / `GTOval_ip*` / `GTOval_ig*`
//! remain dispatched at the user-facing wrapper (`pyscf-gto::eval_gto`)
//! and return clean `NotYetImplemented{phase:4|7}` — no kernel cost.
//!
//! ALG-06 algebra-wall: this module imports `cubecl-*` directly via the
//! Wave 0 W0-T4 allowlist update. The PUBLIC function `eval_gto_sph`
//! takes ONLY `pyscf-algebra` types (`AlgebraClient`) so that
//! `pyscf-gto`'s wrapper (the next layer up) imports this without ever
//! naming a cubecl type. `xtask::check-dependency-wall` enforces the
//! containment.
//!
//! ### Plan deviation (Rule 3): cubecl macro deferred to Phase 4
//!
//! The plan's draft kernel (`#[cube(launch_unchecked)] fn
//! eval_gto_sph_kernel(..., #[comptime] _spherical: bool)`) hits multiple
//! cubecl 0.10.0 macro-expansion issues that don't appear in the plan's
//! spec:
//!
//!   - `ScalarArg::new` is not a public type in cubecl 0.10.0 (the
//!     replacement is `InputScalar` in `cubecl::frontend::scalar`)
//!   - `ArrayArg::from_raw_parts` takes `(handle, length)` — no turbofish
//!     for element type
//!   - `ABSOLUTE_POS` returns `usize`, not `u32` — silent type mismatch
//!     in the plan's draft
//!   - `let bas_slots: u32 = 8u32;` triggers `from_lit` on
//!     `NativeExpand<u32>` which is not satisfied by `From<NativeExpand<u32>>
//!     for ConstantValue` in cubecl 0.10.0
//!   - inlined `f64::exp` works inside `#[cube]` (verified via
//!     `impl_unary_func!(Exp, exp, …, f64)`) but only after fixing all
//!     of the above syntax issues
//!
//! Wave 0 (`tests/wave0_cubecl_smoke.rs`) already proved cubecl-cpu can
//! launch a `#[cube(launch_unchecked)]` kernel from this crate. Plan
//! 02-06 (this commit) preserves that proof and ships a host CPU
//! implementation behind the same `AlgebraClient`-typed public surface
//! (`eval_gto_sph(&AlgebraClient, …)`). The host path is in lockstep
//! with `pyscf-algebra::host_fallback::{eigh, cholesky, qr, svd}` (which
//! also routes the eigh family to `faer 0.24` on host per ALG-05); the
//! algebra wall is preserved without forcing this Phase-2 plan to land a
//! production-ready cubecl macro that the upstream API hasn't fully
//! frozen.
//!
//! Phase 4 DFT (or a dedicated Phase 8 GPU-enable plan) extends this
//! file with the actual `#[cube(launch_unchecked)]` kernel for l ≥ 1
//! cart2sph transforms + deriv1/deriv2 stencils — that's also when the
//! cubecl-macro surface really earns its keep (large grids on GPU).
//! The host CPU path stays as a fallback for the algebra-wall and as
//! the FMA-free oracle target (FOUND-05).

use pyscf_algebra::{oracle_sum, AlgebraClient};
use pyscf_runtime::BackendKind;

// `cubecl` is reachable from this crate per the ALG-06 carve-out
// (`xtask/check_dependency_wall.rs:47` lists `pyscf-kernels` in
// `ALLOWED_CRATES`). The Wave 0 smoke test
// (`tests/wave0_cubecl_smoke.rs`) keeps the launch path warm — see
// the file-level doc comment for the deferral rationale.
#[allow(unused_imports)]
use cubecl::prelude::*;

use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};

// ── cart→sph angular machinery (libcint provenance) ─────────────────────
//
// Source: libcint `cart2sph.c g_trans_cart2sph[]` (the matrices
// `CINTc2s_ket_sph` applies; `pyscf/gto/mole.py cart2sph` uses the same
// routine). FROZEN f64 — byte-identical to cintx-cubecl
// `transform::c2s::C2S_L{0..4}`. Rows = m = -l..+l, cols = libcint
// cartesian order (the `GTOshell_eval_grid_cart` monomial order:
// lx=l..0, ly=l-lx..0, lz=l-lx-ly). Changing any value breaks bit-exact
// agreement with upstream PySCF.

/// libcint `CINTcommon_fac_sp` (g1e.c:566). l=0,1 carry the angular
/// prefactor in the radial part; l≥2 fold it into the c2s matrix.
#[inline]
fn common_fac_sp(l: u32) -> f64 {
    match l {
        0 => 0.282094791773878143,
        1 => 0.488602511902919921,
        _ => 1.0,
    }
}

/// Number of cartesian components for angular momentum `l`.
#[inline]
fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Number of spherical components for angular momentum `l` (`2l+1`).
#[inline]
fn nsph(l: u32) -> usize {
    2 * l as usize + 1
}

/// Cartesian monomial powers `(lx, ly, lz)` for cart column `c`, in the
/// upstream `GTOshell_eval_grid_cart` loop order
/// (`for lx=l..0 { for ly=l-lx..0 { lz=l-lx-ly }}`). This ordering is
/// what `ao_loc_nr` + the c2s columns assume — Pitfall 8/17 lives here.
fn cart_powers(l: u32) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::with_capacity(ncart(l));
    let li = l as i32;
    let mut lx = li;
    while lx >= 0 {
        let mut ly = li - lx;
        while ly >= 0 {
            v.push((lx as u32, ly as u32, (li - lx - ly) as u32));
            ly -= 1;
        }
        lx -= 1;
    }
    v
}

/// libcint `g_trans_cart2sph` coefficient `T[l][m_row][cart_col]`.
/// Returns the FROZEN Condon-Shortley value. `l ≤ 4` supported (g-shells);
/// higher `l` is not in the v1 corpus (max cc-pVTZ f = l 3) and panics so
/// a future basis with l>4 fails loudly rather than silently writing 0.
fn c2s_coeff(l: u32, m_row: usize, cart_col: usize) -> f64 {
    // s (l=0): 1×1 identity.
    const L0: [[f64; 1]; 1] = [[1.0]];
    // p (l=1): identity (px,py,pz); the 0.4886 prefactor is in fac1.
    const L1: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // d (l=2): 5×6. cols: xx,xy,xz,yy,yz,zz. rows: m=-2..+2.
    const L2: [[f64; 6]; 5] = [
        [0.0, 1.092548430592079070, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.092548430592079070, 0.0],
        [-0.315391565252520002, 0.0, 0.0, -0.315391565252520002, 0.0, 0.630783130505040012],
        [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],
        [0.546274215296039535, 0.0, 0.0, -0.546274215296039535, 0.0, 0.0],
    ];
    // f (l=3): 7×10. cols: xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz.
    const L3: [[f64; 10]; 7] = [
        [0.0, 1.770130769779930531, 0.0, 0.0, 0.0, 0.0, -0.590043589926643510, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 2.890611442640554055, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.457045799464465739, 0.0, 0.0, 0.0, 0.0, -0.457045799464465739, 0.0, 1.828183197857862944, 0.0],
        [0.0, 0.0, -1.119528997770346170, 0.0, 0.0, 0.0, 0.0, -1.119528997770346170, 0.0, 0.746352665180230782],
        [-0.457045799464465739, 0.0, 0.0, -0.457045799464465739, 0.0, 1.828183197857862944, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.445305721320277020, 0.0, 0.0, 0.0, 0.0, -1.445305721320277020, 0.0, 0.0],
        [0.590043589926643510, 0.0, 0.0, -1.770130769779930530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    // g (l=4): 9×15. cols: xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,
    // xzzz,yyyy,yyyz,yyzz,yzzz,zzzz.
    const L4: [[f64; 15]; 9] = [
        [0.0, 2.503342941796704538, 0.0, 0.0, 0.0, 0.0, -2.503342941796704530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 5.310392309339791593, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.770130769779930530, 0.0, 0.0, 0.0],
        [0.0, -0.946174695757560014, 0.0, 0.0, 0.0, 0.0, -0.946174695757560014, 0.0, 5.677048174545360108, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 2.676186174229156671, 0.0],
        [0.317356640745612911, 0.0, 0.0, 0.634713281491225822, 0.0, -2.538853125964903290, 0.0, 0.0, 0.0, 0.0, 0.317356640745612911, 0.0, -2.538853125964903290, 0.0, 0.846284375321634430],
        [0.0, 0.0, -2.007139630671867500, 0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 2.676186174229156671, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.473087347878780002, 0.0, 0.0, 0.0, 0.0, 2.838524087272680054, 0.0, 0.0, 0.0, 0.0, 0.473087347878780009, 0.0, -2.838524087272680050, 0.0, 0.0],
        [0.0, 0.0, 1.770130769779930531, 0.0, 0.0, 0.0, 0.0, -5.310392309339791590, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.625835735449176134, 0.0, 0.0, -3.755014412695056800, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.625835735449176134, 0.0, 0.0, 0.0, 0.0],
    ];
    match l {
        0 => L0[m_row][cart_col],
        1 => L1[m_row][cart_col],
        2 => L2[m_row][cart_col],
        3 => L3[m_row][cart_col],
        4 => L4[m_row][cart_col],
        _ => panic!(
            "eval_gto: cart→sph transform only supports l<=4 (g shells); got l={l}. \
             v1 corpus tops out at f (l=3). Add the g_trans_cart2sph row for l>4 if needed."
        ),
    }
}

/// Output of an eval_gto call. Flat F-order buffer + shape descriptor.
#[derive(Debug, Clone)]
pub struct EvalGtoBuffers {
    /// Flat F-order buffer. For scalar variants (`GTOval`, `GTOval_sph`,
    /// `GTOval_cart`): `out[g + ao * ngrids]`. For derivative variants
    /// (Phase 4 DFT extension): leading axis indexes the derivative
    /// component.
    pub values: Vec<f64>,
    /// Logical shape — `[ngrids, nao]` for scalar variants; future
    /// `[ncomp, ngrids, nao]` for derivative variants.
    pub shape: Vec<usize>,
}

/// Evaluate `GTOval_sph` (or `GTOval_cart`) on the given grid for the
/// supplied basis. Public surface uses pyscf-algebra types only —
/// `pyscf-gto`'s wrapper calls this without ever naming `cubecl::*`.
///
/// # Arguments
///
/// - `client`: the resolved `AlgebraClient`. Phase 2 ships CPU only;
///   GPU backends fall back to the CPU path with a `tracing::warn!`.
///   Phase 4 DFT (or Phase 8 GPU enable) wires the GPU arms with a
///   `#[cube(launch_unchecked)]` kernel.
/// - `coords`: flat F-order grid coordinates. Length `ngrids * 3`.
///   Layout: `x[0..ngrids], y[ngrids..2*ngrids], z[2*ngrids..3*ngrids]`.
/// - `atm` / `bas` / `env` / `ao_loc`: the libcint flat arrays from
///   `mol._atm`, `mol._bas`, `mol._env`, `mol.ao_loc_nr` (built in 02-04).
/// - `nao`: total number of AOs (`mol.nao_nr`).
/// - `spherical`: `true` → apply `cart2sph` (Phase 4 DFT extension; for
///   l = 0 a no-op so the s-shell smoke test passes either way);
///   `false` → return raw cartesian.
pub fn eval_gto_sph(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    spherical: bool,
) -> EvalGtoBuffers {
    // The per-backend match still lives at the public surface so future
    // GPU arms (Phase 8) drop in cleanly. With only the `cpu` feature
    // enabled, `BackendKind` only has the `Cpu` variant constructible
    // from `AlgebraClient::kind()`; the fallback arm logs a warn for
    // any future GPU backend until Phase 8 wires them.
    let kind = client.kind();
    if kind != BackendKind::Cpu {
        tracing::warn!(
            backend = ?kind,
            "eval_gto: backend not yet wired (Phase 8 GPU enable); falling back to CPU"
        );
    }
    eval_gto_sph_cpu(coords, ngrids, atm, bas, env, ao_loc, nao, spherical)
}

/// CPU-host implementation of the s-shell AO-on-grid kernel.
///
/// One pass per grid point evaluates EVERY shell on that point. Output
/// is F-order `(ngrids, nao)`: index `g + ao * ngrids`.
///
/// **Phase 2 scope** (this commit): the `l == 0` path computes the full
/// contracted radial `Σ_p coeff[c,p] * exp(-α_p * r²)` and writes it
/// directly to the AO slot (Y_00 is absorbed into the normalised
/// coefficient — see `pyscf-gto::make_env::normalise_contractions`). The
/// `l >= 1` branch (plan 04-03) evaluates the cartesian monomials and
/// applies the libcint cart→sph transform.
///
/// `_spherical`: l = 0 is identical for sph and cart (Y_00 == the
/// cartesian s norm). For l >= 1 this kernel always emits the SPHERICAL
/// AOs (the plan-04-03 / DFT scope is `GTOval_sph`). `GTOval_cart` with
/// l >= 1 needs the cartesian `ao_loc`/`nao` (more AOs than spherical)
/// which the caller does not pass here; it stays deferred and the
/// `pyscf-gto::eval_gto` wrapper only routes `spherical=true` into the
/// l >= 1 path for the corpus bases (sp shells excepted — none in v1).
fn eval_gto_sph_cpu(
    coords_host: &[f64],
    ngrids: usize,
    atm_host: &[i32],
    bas_host: &[i32],
    env_host: &[f64],
    ao_loc_host: &[i32],
    nao: usize,
    _spherical: bool,
) -> EvalGtoBuffers {
    debug_assert_eq!(
        coords_host.len(),
        ngrids * 3,
        "coords flat buffer must be ngrids*3 (got {} for ngrids={})",
        coords_host.len(),
        ngrids
    );
    debug_assert!(
        bas_host.len() % BAS_SLOTS == 0,
        "bas length {} not a multiple of BAS_SLOTS={}",
        bas_host.len(),
        BAS_SLOTS
    );

    let nbas = bas_host.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    // Empty grid → empty output, skip the loop entirely.
    if out_len == 0 {
        return EvalGtoBuffers { values: Vec::new(), shape: vec![ngrids, nao] };
    }

    let mut out = vec![0.0_f64; out_len];

    // Per-grid-point evaluation. `coords` is F-order: x[0..ngrids],
    // y[ngrids..2*ngrids], z[2*ngrids..3*ngrids].
    for g in 0..ngrids {
        let gx = coords_host[g];
        let gy = coords_host[g + ngrids];
        let gz = coords_host[g + 2 * ngrids];

        for shell_idx in 0..nbas {
            let bas_row = shell_idx * BAS_SLOTS;
            let atom_id = bas_host[bas_row + ATOM_OF] as usize;
            let l = bas_host[bas_row + ANG_OF] as u32;
            let nprim = bas_host[bas_row + NPRIM_OF] as usize;
            let nctr = bas_host[bas_row + NCTR_OF] as usize;
            let ptr_exp = bas_host[bas_row + PTR_EXP] as usize;
            let ptr_coeff = bas_host[bas_row + PTR_COEFF] as usize;

            let atm_row = atom_id * ATM_SLOTS;
            let ptr_coord = atm_host[atm_row + PTR_COORD] as usize;
            let ax = env_host[ptr_coord];
            let ay = env_host[ptr_coord + 1];
            let az = env_host[ptr_coord + 2];

            let dx = gx - ax;
            let dy = gy - ay;
            let dz = gz - az;
            let r2 = dx * dx + dy * dy + dz * dz;

            let ao_off = ao_loc_host[shell_idx] as usize;

            if l == 0 {
                // s-shell path: contracted radial × Y_00.
                //
                // 02-04 `make_env::normalise_contractions` applies the
                // *radial* normalisation only (per-prim gto_norm + the
                // `_nomalize_contracted_ao` factor). The angular factor
                // Y_00 = (1/(4π))^{1/2} = 1/(2*sqrt(π)) is applied here
                // — upstream `pyscf/gto/eval_gto.py` calls `_cart2sph_l(0)`
                // which is the [[1/(2*sqrt(π))]] 1×1 matrix. For s-shells
                // the cartesian normalisation factor is identical to
                // Y_00, so the same multiplier covers both `GTOval_sph`
                // and `GTOval_cart` (`cart_variant_works_for_s_shells`
                // smoke fixture verifies the equality).
                let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
                for c_idx in 0..nctr {
                    let mut acc: f64 = 0.0;
                    for p_idx in 0..nprim {
                        let alpha = env_host[ptr_exp + p_idx];
                        // Coefficient matrix is F-order:
                        //   ptr_coeff + c_idx * nprim + p_idx
                        let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                        acc += coef * (-alpha * r2).exp();
                    }
                    let ao_idx = ao_off + c_idx;
                    out[g + ao_idx * ngrids] = acc * y00;
                }
            } else {
                // l ≥ 1 path (plan 04-03): cartesian monomials × radial,
                // then the libcint cart→sph transform. Mirrors
                // `GTOshell_eval_grid_cart` + `CINTc2s_ket_sph`.
                let fac1 = common_fac_sp(l);
                let powers = cart_powers(l);
                let ncart_l = ncart(l);
                let nsph_l = nsph(l);

                // Precompute the cartesian monomial geometric factors
                // (x^lx · y^ly · z^lz) — radial-independent, shared by
                // every contraction column.
                let mut mono = vec![0.0_f64; ncart_l];
                for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                    mono[ci] = dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32);
                }

                let mut cart_vals = vec![0.0_f64; ncart_l];
                for c_idx in 0..nctr {
                    // Ordered, FMA-free contracted radial. > 2 prims →
                    // oracle_sum (Pitfall 3 / FOUND-06); ≤ 2 prims fold
                    // into the same materialised-then-summed path so the
                    // reduction tree shape depends only on length.
                    let radial = if nprim > 2 {
                        let terms: Vec<f64> = (0..nprim)
                            .map(|p_idx| {
                                let alpha = env_host[ptr_exp + p_idx];
                                let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                                coef * (-alpha * r2).exp()
                            })
                            .collect();
                        oracle_sum(&terms)
                    } else {
                        let mut acc = 0.0_f64;
                        for p_idx in 0..nprim {
                            let alpha = env_host[ptr_exp + p_idx];
                            let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                            acc += coef * (-alpha * r2).exp();
                        }
                        acc
                    };
                    let radial = radial * fac1;

                    // cartesian AO values for this contraction.
                    for ci in 0..ncart_l {
                        cart_vals[ci] = mono[ci] * radial;
                    }

                    // cart → sph: row m = Σ_c T[l][m][c] * cart_vals[c].
                    for m_idx in 0..nsph_l {
                        let mut v = 0.0_f64;
                        for ci in 0..ncart_l {
                            v += c2s_coeff(l, m_idx, ci) * cart_vals[ci];
                        }
                        let ao_idx = ao_off + c_idx * nsph_l + m_idx;
                        out[g + ao_idx * ngrids] = v;
                    }
                }
            }
        }
    }

    EvalGtoBuffers { values: out, shape: vec![ngrids, nao] }
}
