//! GTO-07 user-facing wrapper for `eval_gto`.
//!
//! Source: `pyscf/gto/eval_gto.py:1-248` (Apache-2.0). Mirrors the
//! upstream public surface — Python users call
//! `mol.eval_gto("GTOval_sph", coords)`, the Rust public-API equivalent
//! is `pyscf_gto::eval_gto(&mol, "GTOval_sph", &coords)`.
//!
//! ### Algebra wall
//!
//! Per ALG-06 + D-04, this module imports ONLY `pyscf-algebra` and
//! `pyscf-kernels` (which itself imports cubecl behind the wall). It
//! NEVER names a `cubecl::*` type. The `cargo run -p xtask --bin
//! check-dependency-wall` lint catches any future dep slip;
//! `! grep -E '\bcubecl(_[a-z]+)?\b' crates/pyscf-gto/src/eval_gto.rs`
//! is the per-file static guarantee documented in the plan.
//!
//! ### Variant catalogue
//!
//! Shipped:
//!   - `GTOval` — alias; routes to `GTOval_cart` if `mol.cart` else
//!     `GTOval_sph`.
//!   - `GTOval_sph` — spherical; the priority MVP. Phase 4 plan 04-03
//!     landed the `l ≥ 1` (p/d/f) cart→sph evaluation in the kernel.
//!   - `GTOval_cart` — cartesian. For l = 0 shells (the smoke fixture)
//!     this returns identical numerics to `GTOval_sph` because the
//!     l = 0 cart→sph transform is the identity. (l ≥ 1 cartesian still
//!     uses the spherical AO count; cartesian-specific output is N/A in
//!     v1 — the DFT grid loop is spherical.)
//!   - `GTOval_sph_deriv1` — Phase 4 plan 04-03: value + 3 Cartesian
//!     gradient components `[value, ∂x, ∂y, ∂z]` (the GGA ∇ρ input).
//!     Output layout `[4, ngrids, nao]` (component-leading).
//!   - `GTOval_ip_sph` / `GTOval_ip` (spherical) — `nabla |AO>`, 3 components
//!     `[3, ngrids, nao]`. Byte-identical to `GTOval_sph_deriv1[1:4]`
//!     (no sign flip; verified vs upstream), so it reuses the deriv1 kernel
//!     and slices off the value block (F-01).
//!
//! Deferred:
//!   - `GTOval_cart_deriv1` → `NotYetImplemented{phase: 4}` (needs
//!     cartesian `ao_loc`; v1 DFT uses the spherical deriv1).
//!   - `GTOval_sph_deriv2` / `GTOval_cart_deriv2` → `NotYetImplemented
//!     {phase: 4}` (deriv2 not needed for v1 DFT energy).
//!   - `GTOval_ip_cart` → `NotYetImplemented{phase: 4}` (needs cart deriv1, F-02).
//!   - `GTOval_ig*` (magnetic-gauge) → `NotYetImplemented{phase: 7}` — a distinct
//!     GIAO kernel, not derivable from deriv1.
//!
//! Unknown variants return `CoreError::InvalidMolecule` (consistent
//! with `mol.intor("foo")` per 02-05). Unbuilt Mole returns
//! `CoreError::InvalidMolecule("Mole not built …")`.

use pyscf_algebra::select_backend;
use pyscf_core::{CoreError, Mole, PyscfRsError};

/// Output of an eval_gto call. Mirrors `pyscf-kernels::EvalGtoBuffers`
/// at the user surface — provides a `values` flat F-order buffer and
/// the logical `shape` descriptor.
#[derive(Debug, Clone)]
pub struct EvalGtoOutput {
    /// Flat F-order buffer. For scalar variants (`GTOval`, `GTOval_sph`,
    /// `GTOval_cart`): index `g + ao * ngrids`. For `GTOval_sph_deriv1`:
    /// component-leading `[4, ngrids, nao]` — component
    /// `c ∈ {0=value,1=∂x,2=∂y,3=∂z}` occupies `values[c*ngrids*nao..]`
    /// and within each component the index is `g + ao*ngrids`.
    pub values: Vec<f64>,
    /// Logical shape — `[ngrids, nao]` for scalar variants;
    /// `[4, ngrids, nao]` for `GTOval_sph_deriv1`.
    pub shape: Vec<usize>,
}

/// Evaluate AO values at the given grid coordinates.
///
/// `coords` shape: `(ngrids, 3)` C-order — `coords[g][xyz]`.
///
/// Supports `GTOval`, `GTOval_sph`, `GTOval_cart`, and (Phase 4 plan
/// 04-03) `GTOval_sph_deriv1`. The remaining derivative variants return
/// `NotYetImplemented{phase:4|7}` per the module-level catalogue.
pub fn eval_gto(
    mol: &Mole,
    eval_name: &str,
    coords: &[[f64; 3]],
) -> Result<EvalGtoOutput, PyscfRsError> {
    if !mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "Mole not built — call mol.build() first".into(),
        )));
    }

    // Resolve alias `GTOval` to `GTOval_sph` or `GTOval_cart` per
    // mol.cart. Mirrors upstream `pyscf/gto/eval_gto.py:152-160`.
    let resolved = match eval_name {
        "GTOval" => {
            if mol.cart {
                "GTOval_cart"
            } else {
                "GTOval_sph"
            }
        }
        // Bare derivative aliases get the rep suffix per mol.cart, mirroring
        // upstream `_get_intor_and_comp` (`pyscf/gto/eval_gto.py`).
        "GTOval_ip" => {
            if mol.cart {
                "GTOval_ip_cart"
            } else {
                "GTOval_ip_sph"
            }
        }
        "GTOval_ig" => {
            if mol.cart {
                "GTOval_ig_cart"
            } else {
                "GTOval_ig_sph"
            }
        }
        other => other,
    };

    match resolved {
        "GTOval_sph" | "GTOval_cart" => {
            let spherical = resolved == "GTOval_sph";
            let ngrids = coords.len();

            // Pack coords into F-order flat buffer: x[0..N], y[N..2N], z[2N..3N].
            let mut coords_flat = Vec::with_capacity(ngrids * 3);
            coords_flat.extend(coords.iter().map(|c| c[0]));
            coords_flat.extend(coords.iter().map(|c| c[1]));
            coords_flat.extend(coords.iter().map(|c| c[2]));

            // Resolve the AlgebraClient via PYSCF_BACKEND. The `select_backend`
            // helper logs a `tracing::info!` line per ALG-08 and falls back to
            // CPU on any unrecognised env value (ALG-04).
            let selection = select_backend().map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_gto: backend selection failed: {e}"
                )))
            })?;

            let buffers = pyscf_kernels::eval_gto_sph(
                &selection.client,
                &coords_flat,
                ngrids,
                &mol._atm,
                &mol._bas,
                &mol._env,
                &mol.ao_loc_nr,
                mol.nao_nr,
                spherical,
            )?;
            Ok(EvalGtoOutput {
                values: buffers.values,
                shape: buffers.shape,
            })
        }
        "GTOval_sph_deriv1" => {
            let ngrids = coords.len();
            // F-order flat coords: x[0..N], y[N..2N], z[2N..3N].
            let mut coords_flat = Vec::with_capacity(ngrids * 3);
            coords_flat.extend(coords.iter().map(|c| c[0]));
            coords_flat.extend(coords.iter().map(|c| c[1]));
            coords_flat.extend(coords.iter().map(|c| c[2]));

            let selection = select_backend().map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_gto: backend selection failed: {e}"
                )))
            })?;

            let buffers = pyscf_kernels::eval_gto_sph_deriv1(
                &selection.client,
                &coords_flat,
                ngrids,
                &mol._atm,
                &mol._bas,
                &mol._env,
                &mol.ao_loc_nr,
                mol.nao_nr,
            )?;
            Ok(EvalGtoOutput {
                values: buffers.values,
                shape: buffers.shape,
            })
        }
        // Cartesian deriv1 needs the cartesian `ao_loc`/`nao` (more AOs
        // than spherical) which the kernel surface does not take; it
        // stays deferred. v1 DFT uses the spherical path.
        "GTOval_cart_deriv1" => Err(PyscfRsError::NotYetImplemented {
            phase: 4,
            what: "GTOval_cart_deriv1 (cartesian deriv1 needs cart ao_loc; v1 DFT uses GTOval_sph_deriv1)",
        }),
        "GTOval_sph_deriv2" | "GTOval_cart_deriv2" => Err(PyscfRsError::NotYetImplemented {
            phase: 4,
            what: "GTOval_sph_deriv2 (Phase 4 DFT grid integration extends the kernel)",
        }),
        // `GTOval_ip_sph` = `nabla |AO>` (3 components) — verified byte-identical
        // to the gradient block `GTOval_sph_deriv1[1:4]` of the Phase-4 deriv1
        // kernel (upstream `pyscf/gto/eval_gto.py`: `GTOval_ip_sph` ≡
        // `+GTOval_sph_deriv1[1:4]`, no sign flip). So we evaluate deriv1 and
        // drop the value component, yielding `[3, ngrids, nao]`.
        "GTOval_ip_sph" => {
            let ngrids = coords.len();
            let mut coords_flat = Vec::with_capacity(ngrids * 3);
            coords_flat.extend(coords.iter().map(|c| c[0]));
            coords_flat.extend(coords.iter().map(|c| c[1]));
            coords_flat.extend(coords.iter().map(|c| c[2]));

            let selection = select_backend().map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "eval_gto: backend selection failed: {e}"
                )))
            })?;

            let buffers = pyscf_kernels::eval_gto_sph_deriv1(
                &selection.client,
                &coords_flat,
                ngrids,
                &mol._atm,
                &mol._bas,
                &mol._env,
                &mol.ao_loc_nr,
                mol.nao_nr,
            )?;

            // deriv1 layout: component-leading `[4, ngrids, nao]`, component c
            // at `values[c*block..]` with `block = ngrids*nao`. The 3 gradient
            // components (c=1,2,3) are already contiguous in order — slice off
            // the leading value block (c=0).
            let block = ngrids.checked_mul(mol.nao_nr).ok_or_else(|| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "GTOval_ip_sph: size overflow ngrids={ngrids} nao={}",
                    mol.nao_nr
                )))
            })?;
            // Defensive: the deriv1 kernel must return exactly 4 blocks.
            if buffers.values.len() != 4 * block {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "GTOval_ip_sph: deriv1 returned {} values, expected {} (4*ngrids*nao)",
                    buffers.values.len(),
                    4 * block
                ))));
            }
            let grad = buffers.values[block..4 * block].to_vec();
            Ok(EvalGtoOutput {
                values: grad,
                shape: vec![3, ngrids, mol.nao_nr],
            })
        }
        // Cartesian gradient needs the cart `ao_loc`/`nao` deriv1 surface, which
        // is still deferred (same blocker as `GTOval_cart_deriv1`, F-02).
        "GTOval_ip_cart" => Err(PyscfRsError::NotYetImplemented {
            phase: 4,
            what: "GTOval_ip_cart (cartesian gradient needs cart ao_loc deriv1; \
                   spherical GTOval_ip_sph is live)",
        }),
        "GTOval_ig_sph" | "GTOval_ig_cart" | "GTOval_ig" => Err(PyscfRsError::NotYetImplemented {
            phase: 7,
            what: "GTOval_ig (magnetic-gauge variant; Phase 7 grad)",
        }),
        other => Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "unknown eval_gto variant: {} (Phase 2 ships GTOval, GTOval_sph, GTOval_cart; \
             deriv variants stub to Phase 4/7)",
            other
        )))),
    }
}
