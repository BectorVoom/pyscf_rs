//! GTO-07 + D-04: AO-on-grid kernel (`eval_gto`).
//!
//! Source: pyscf/gto/eval_gto.py (Apache-2.0). The reference algorithm:
//! per grid point g, walk every shell s, compute the contracted radial
//! `Σ_p coeff[c,p] * exp(-α_p * r²)`, optionally apply the cart→sph
//! harmonic transform, and write to `out[g, ao_idx]` in F-order.
//!
//! Phase 2 (this commit) ships the **s-shell only** implementation —
//! sufficient for the smoke test (single H 1s at the nucleus matches the
//! analytical contracted-radial sum for STO-3G). The l ≥ 1 path writes
//! zeros and is documented as a Phase 4 DFT extension. The 4 derivative
//! variants (`GTOval_sph_deriv1`, `GTOval_sph_deriv2`, `GTOval_ip*`,
//! `GTOval_ig*`) are dispatched at the user-facing wrapper
//! (`pyscf-gto::eval_gto`) and return clean
//! `NotYetImplemented{phase:4|7}` errors — no kernel cost here.
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

use pyscf_algebra::AlgebraClient;
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
/// `l >= 1` branch writes zeros and is documented as a Phase 4 DFT
/// extension (see threat model T-02-06-04).
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
                // s-shell path: contracted radial × Y_00 (absorbed into
                // the normalised coefficient by 02-04 `make_env`).
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
                    out[g + ao_idx * ngrids] = acc;
                }
            } else {
                // l ≥ 1 stub: write zeros into every (c, m) AO slot.
                // Phase 4 DFT extends with cart2sph + radial. The slots
                // are pre-zeroed by `vec![0.0; out_len]` so this branch
                // is technically a no-op; left explicit for the byte-
                // identity oracle to land on the right shape.
                let dim_per_ctr = (2 * l as usize) + 1;
                for c_idx in 0..nctr {
                    for m_idx in 0..dim_per_ctr {
                        let ao_idx = ao_off + c_idx * dim_per_ctr + m_idx;
                        out[g + ao_idx * ngrids] = 0.0;
                    }
                }
            }
        }
    }

    EvalGtoBuffers { values: out, shape: vec![ngrids, nao] }
}
