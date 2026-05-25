//! W0-T1 smoke test: cintx round-trip from inside pyscf-gto.
//!
//! Proves the cintx integration reach for Phase 2:
//!   1. cintx-{core, ops, rs, runtime, compat} are reachable as path deps
//!   2. `Resolver::descriptor_by_symbol("int1e_ovlp_sph")` resolves the operator
//!   3. `SessionRequest::new(...).query_workspace()?.evaluate()` round-trips
//!      without error, returning an `IntegralTensor` with the expected extents.
//!
//! NOTE on the cintx-rs safe-API contract observed during W0-T1:
//!
//!   1. `IntegralTensor.extents` is `[ao_per_shell_i for shell_i in tuple]`,
//!      i.e. ONE block per `SessionRequest` (the shell-pair block, libcint
//!      convention). For two H 1s shells (l=0, spheric, 1 AO each) the block
//!      is `[1, 1]` with 1 element — NOT the full 2x2 overlap matrix in a
//!      single call. Callers iterate all shell-pair combinations to build
//!      the full matrix; that loop is plan 02-05's job.
//!
//!   2. The current safe-API executor (`CubeClExecutor::execute` in
//!      cintx-rs::api) populates `owned_values` via `fill_staging_values` —
//!      a synthetic pattern (`((idx + 1) as f64) * 0.5` for spherical) used
//!      while the safe API is being landed end-to-end. Real integral values
//!      flow through `cintx-compat::raw::eval_raw` (the cintx oracle suite).
//!      Plan 02-05 will switch the dispatcher to whichever path is real by
//!      the time it lands.
//!
//!   For Wave 0 the de-risk is path-dep reachability + ABI shape + finite
//!   output buffer — not numerical correctness (which the cintx oracle suite
//!   covers separately). This test acts as a reachability smoke and a
//!   change-detector if cintx flips its safe-API output behaviour.

mod common;

use cintx_core::Representation;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;

#[test]
fn cintx_h2_sto3g_int1e_ovlp_sph_round_trip() {
    let (basis, shells) = common::h2_sto3g_basis();

    // Resolve the int1e_ovlp_sph operator id from cintx-ops manifest.
    let descriptor = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
        .expect("int1e_ovlp_sph must be present in cintx-ops manifest (W0-T1)");
    let operator = descriptor.id;

    let request = SessionRequest::new(
        operator,
        Representation::Spheric,
        &basis,
        shells,
        ExecutionOptions::default(),
    );

    let query = request
        .query_workspace()
        .expect("workspace query must succeed for int1e_ovlp_sph H2/STO-3G");

    let outcome = query
        .evaluate()
        .expect("evaluate must succeed for int1e_ovlp_sph H2/STO-3G");

    let tensor = outcome.tensor;
    // cintx contract: extents has one entry per shell in the tuple, equal to
    // `ao_per_shell()` for that shell. Two H 1s shells (l=0, Spheric) ⇒
    // extents == [1, 1] for the shell-pair block. The full 2x2 overlap matrix
    // is assembled by 02-05's intor dispatcher iterating shell pairs.
    assert_eq!(
        tensor.extents,
        vec![1, 1],
        "H2/STO-3G int1e_ovlp_sph one-shell-pair block must be 1x1 \
         (one AO per H 1s shell in spheric); got extents {:?}",
        tensor.extents
    );
    assert_eq!(
        tensor.owned_values.len(),
        1,
        "1x1 shell-pair block must hold 1 element (got {})",
        tensor.owned_values.len()
    );

    // Sanity: every value must be finite. This catches uninitialised buffers
    // / NaN propagation in the safe pipeline regardless of whether the values
    // are real overlaps or the current synthetic staging pattern.
    for (i, v) in tensor.owned_values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "owned_values[{i}] must be finite (got {v}); pipeline produced bad data"
        );
    }

    // bytes_written must agree with element count × f64 size.
    assert_eq!(
        outcome.bytes_written,
        tensor.owned_values.len() * std::mem::size_of::<f64>(),
        "bytes_written must equal owned_values.len() * sizeof(f64)"
    );

    // workspace_bytes is a positive planning quantity.
    assert!(
        outcome.workspace_bytes > 0,
        "workspace planner must report a positive byte budget (got {})",
        outcome.workspace_bytes
    );
}
