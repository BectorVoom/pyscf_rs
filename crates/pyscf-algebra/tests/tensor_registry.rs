//! quick-260529-mtx: end-to-end test of the Phase-2 device-buffer registry and
//! the `Tensor`-based ops wired through it (`axpy`, `scal`, `dot`, `reduce_sum`,
//! `gemm`, `gemv`, `transpose`).
//!
//! Strategy (per docs/rust_crate_test_guideline.md — verify the stateful
//! upload→op→download workflow and its error contracts, not just a happy path):
//! since quick-260529-mtx-2 the registry keeps buffers **resident on the device**
//! and ops launch directly on the resident handles (no per-op re-upload). The
//! strongest correctness check is still **equivalence** — the `Tensor` op must
//! produce the exact same bytes as the corresponding `*_dense` function on the
//! same inputs. `check_resident_chain` additionally exercises the residency
//! itself: a chain of ops on the same buffers with no intermediate download. We
//! also pin host references and exercise every error contract
//! (placeholder/unallocated, backend metadata, double-release, bad shapes).
//!
//! - `*_on_cpu` always runs (default `cpu` feature).
//! - `*_on_rocm` (`#[cfg(feature = "rocm")]`) runs the SAME checks on real AMD
//!   hardware (gfx1152) via `cubecl_hip::HipRuntime`.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_algebra::{
    AlgebraClient, AlgebraError, Tensor, axpy, axpy_dense, dot, dot_dense, download, gemm,
    gemm_dense, gemv, gemv_dense, reduce_sum, reduce_sum_axis_dense, reduce_sum_dense, release,
    scal, scal_dense, transpose, transpose_dense, upload,
};

/// Exact-equality tolerance: the Tensor path calls the identical `*_dense`
/// launcher on the identical inputs, so equivalence is bit-for-bit (`==`). The
/// host-reference cross-checks use a tiny tolerance to absorb backend
/// rounding-mode quirks while still catching a wrong wiring.
const TOL: f64 = 1e-12;

/// `axpy` over `Tensor` must equal `axpy_dense` and update `y` in place.
fn check_axpy(client: &AlgebraClient, label: &str) {
    let x = vec![1.5_f64, -2.0, 3.25, 0.0, 7.0];
    let y0 = vec![0.5_f64, 1.0, -1.0, 4.0, 2.0];
    let alpha = 2.5;

    // dense reference (the oracle-tested path)
    let mut dense = y0.clone();
    axpy_dense::<f64>(client, alpha, &x, &mut dense).expect("axpy_dense");

    // Tensor path through the registry
    let xt = upload::<f64>(client, &x, vec![5]).expect("upload x");
    let mut yt = upload::<f64>(client, &y0, vec![5]).expect("upload y");
    axpy(client, alpha, &xt, &mut yt).expect("axpy over Tensor");
    let got = download::<f64>(client, &yt).expect("download y");

    assert_eq!(
        got, dense,
        "{label}: axpy Tensor path must equal axpy_dense"
    );
    for (g, (xi, yi)) in got.iter().zip(x.iter().zip(y0.iter())) {
        assert!(
            (g - (yi + alpha * xi)).abs() < TOL,
            "{label}: axpy result drifted from host reference"
        );
    }
    // x is untouched; y holds the update.
    assert_eq!(
        download::<f64>(client, &xt).expect("download x"),
        x,
        "{label}: x intact"
    );

    release::<f64>(&xt);
    release::<f64>(&yt);
}

/// `scal` over `Tensor` must equal `scal_dense` and update `x` in place.
fn check_scal(client: &AlgebraClient, label: &str) {
    let x0 = vec![1.0_f64, -2.5, 3.0, 100.0];
    let alpha = -3.0;

    let mut dense = x0.clone();
    scal_dense::<f64>(client, alpha, &mut dense).expect("scal_dense");

    let mut xt = upload::<f64>(client, &x0, vec![4]).expect("upload x");
    scal(client, alpha, &mut xt).expect("scal over Tensor");
    let got = download::<f64>(client, &xt).expect("download x");

    assert_eq!(
        got, dense,
        "{label}: scal Tensor path must equal scal_dense"
    );
    release::<f64>(&xt);
}

/// `dot` over `Tensor` must equal `dot_dense`.
fn check_dot(client: &AlgebraClient, label: &str) {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];

    let dense = dot_dense::<f64>(client, &x, &y).expect("dot_dense");

    let xt = upload::<f64>(client, &x, vec![5]).expect("upload x");
    let yt = upload::<f64>(client, &y, vec![5]).expect("upload y");
    let got = dot(client, &xt, &yt).expect("dot over Tensor");

    assert_eq!(got, dense, "{label}: dot Tensor path must equal dot_dense");
    let host_ref: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    assert!(
        (got - host_ref).abs() < TOL,
        "{label}: dot drifted from host ref"
    );

    release::<f64>(&xt);
    release::<f64>(&yt);
}

/// `reduce_sum` over `Tensor` covers (a) the 1-D total-sum case — where per-axis
/// axis-0 collapses to the scalar total, equal to `reduce_sum_dense` — and (b)
/// genuine 2-D per-axis reduction along both axes, equal to
/// `reduce_sum_axis_dense` and a host reference, with the reduced `out` shape set.
fn check_reduce_sum(client: &AlgebraClient, label: &str) {
    // (a) 1-D: reducing the only axis is the total sum → scalar out.
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let total = reduce_sum_dense::<f64>(client, &x).expect("reduce_sum_dense");
    let xt = upload::<f64>(client, &x, vec![7]).expect("upload 1-D x");
    let mut out = upload::<f64>(client, &[0.0_f64], vec![1]).expect("upload scalar out");
    reduce_sum(client, &xt, 0, &mut out).expect("reduce_sum 1-D over Tensor");
    let got = download::<f64>(client, &out).expect("download out");
    assert_eq!(got.len(), 1, "{label}: 1-D reduce_sum out is a scalar");
    assert_eq!(
        got[0], total,
        "{label}: 1-D reduce_sum must equal total sum"
    );
    assert!(
        (got[0] - x.iter().sum::<f64>()).abs() < TOL,
        "{label}: 1-D sum drifted"
    );
    release::<f64>(&xt);
    release::<f64>(&out);

    // (b) 2-D 2×3 row-major: [[1,2,3],[4,5,6]].
    let m = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (rows, cols) = (2usize, 3usize);

    // axis 0 → sum down columns → len `cols` = [5, 7, 9].
    let dense0 = reduce_sum_axis_dense::<f64>(client, &m, &[rows, cols], 0).expect("axis0 dense");
    let mt = upload::<f64>(client, &m, vec![rows, cols]).expect("upload M");
    let mut out0 = upload::<f64>(client, &vec![0.0_f64; cols], vec![cols]).expect("upload out0");
    reduce_sum(client, &mt, 0, &mut out0).expect("reduce_sum axis 0");
    let got0 = download::<f64>(client, &out0).expect("download out0");
    assert_eq!(
        got0, dense0,
        "{label}: axis-0 Tensor path must equal axis dense"
    );
    assert_eq!(got0, vec![5.0, 7.0, 9.0], "{label}: axis-0 host reference");
    assert_eq!(out0.shape, vec![cols], "{label}: axis-0 reduced shape set");

    // axis 1 → sum across rows → len `rows` = [6, 15].
    let dense1 = reduce_sum_axis_dense::<f64>(client, &m, &[rows, cols], 1).expect("axis1 dense");
    let mut out1 = upload::<f64>(client, &vec![0.0_f64; rows], vec![rows]).expect("upload out1");
    reduce_sum(client, &mt, 1, &mut out1).expect("reduce_sum axis 1");
    let got1 = download::<f64>(client, &out1).expect("download out1");
    assert_eq!(
        got1, dense1,
        "{label}: axis-1 Tensor path must equal axis dense"
    );
    assert_eq!(got1, vec![6.0, 15.0], "{label}: axis-1 host reference");
    assert_eq!(out1.shape, vec![rows], "{label}: axis-1 reduced shape set");

    // Out-of-range axis is a clean DimensionMismatch, not a panic.
    let mut junk = upload::<f64>(client, &[0.0_f64], vec![1]).expect("upload junk");
    assert!(
        matches!(
            reduce_sum(client, &mt, 2, &mut junk),
            Err(AlgebraError::DimensionMismatch { .. })
        ),
        "{label}: reduce_sum must reject out-of-range axis"
    );
    // Wrong-size `out` for the reduction is a DimensionMismatch too.
    let mut wrong = upload::<f64>(client, &[0.0_f64; 5], vec![5]).expect("upload wrong");
    assert!(
        matches!(
            reduce_sum(client, &mt, 0, &mut wrong),
            Err(AlgebraError::DimensionMismatch { .. })
        ),
        "{label}: reduce_sum must reject wrong-size out"
    );

    release::<f64>(&mt);
    release::<f64>(&out0);
    release::<f64>(&out1);
    release::<f64>(&junk);
    release::<f64>(&wrong);
}

/// `gemm` over `Tensor` must equal `gemm_dense` and fill the `out` buffer.
fn check_gemm(client: &AlgebraClient, label: &str) {
    // lhs 2×3, rhs 3×2 → out 2×2 (row-major).
    let lhs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let (m, k, n) = (2usize, 3usize, 2usize);

    let dense = gemm_dense::<f64>(client, &lhs, &rhs, m, k, n).expect("gemm_dense");

    let lt = upload::<f64>(client, &lhs, vec![m, k]).expect("upload lhs");
    let rt = upload::<f64>(client, &rhs, vec![k, n]).expect("upload rhs");
    let mut ot = upload::<f64>(client, &vec![0.0_f64; m * n], vec![m, n]).expect("upload out");
    gemm(client, &lt, &rt, &mut ot).expect("gemm over Tensor");
    let got = download::<f64>(client, &ot).expect("download out");

    assert_eq!(
        got, dense,
        "{label}: gemm Tensor path must equal gemm_dense"
    );
    // Spot-check one host-computed element: out[0,0] = 1*7 + 2*9 + 3*11 = 58.
    assert!((got[0] - 58.0).abs() < TOL, "{label}: gemm[0,0] wrong");

    release::<f64>(&lt);
    release::<f64>(&rt);
    release::<f64>(&ot);
}

/// `gemv` over `Tensor` must equal `gemv_dense` (and GEMM with N=1).
fn check_gemv(client: &AlgebraClient, label: &str) {
    // A 3×2, x len 2 → y len 3.
    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![7.0_f64, 8.0];
    let (m, n) = (3usize, 2usize);

    let dense = gemv_dense::<f64>(client, &a, &x, m, n).expect("gemv_dense");

    let at = upload::<f64>(client, &a, vec![m, n]).expect("upload A");
    let xt = upload::<f64>(client, &x, vec![n]).expect("upload x");
    let mut yt = upload::<f64>(client, &vec![0.0_f64; m], vec![m]).expect("upload y");
    gemv(client, &at, &xt, &mut yt).expect("gemv over Tensor");
    let got = download::<f64>(client, &yt).expect("download y");

    assert_eq!(
        got, dense,
        "{label}: gemv Tensor path must equal gemv_dense"
    );
    // host ref: y[i] = A[i,0]*x[0] + A[i,1]*x[1].
    for (i, g) in got.iter().enumerate() {
        let r = a[i * n] * x[0] + a[i * n + 1] * x[1];
        assert!(
            (g - r).abs() < TOL,
            "{label}: gemv[{i}] drifted from host ref"
        );
    }

    release::<f64>(&at);
    release::<f64>(&xt);
    release::<f64>(&yt);
}

/// `transpose` over `Tensor` must equal `transpose_dense` and the host
/// permutation.
fn check_transpose(client: &AlgebraClient, label: &str) {
    // 2×3 input → 3×2 output (row-major).
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (m, n) = (2usize, 3usize);

    let dense = transpose_dense::<f64>(client, &x, m, n).expect("transpose_dense");

    let xt = upload::<f64>(client, &x, vec![m, n]).expect("upload x");
    let mut ot = upload::<f64>(client, &vec![0.0_f64; m * n], vec![n, m]).expect("upload out");
    transpose(client, &xt, &mut ot).expect("transpose over Tensor");
    let got = download::<f64>(client, &ot).expect("download out");

    assert_eq!(
        got, dense,
        "{label}: transpose Tensor path must equal transpose_dense"
    );
    // host reference permutation: out[c*m + r] = x[r*n + c].
    for r in 0..m {
        for c in 0..n {
            assert!(
                (got[c * m + r] - x[r * n + c]).abs() < TOL,
                "{label}: transpose[{c},{r}] wrong"
            );
        }
    }

    release::<f64>(&xt);
    release::<f64>(&ot);
}

/// Error contracts that must hold regardless of backend.
fn check_error_contracts(client: &AlgebraClient, label: &str) {
    // Placeholder tensors carry the sentinel BufferId — never uploaded — so
    // every wired op must reject them with UnallocatedBuffer.
    let ph = Tensor::placeholder(vec![3]);
    let mut ph_mut = Tensor::placeholder(vec![3]);
    assert!(
        matches!(
            axpy(client, 1.0, &ph, &mut ph_mut),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: axpy must reject placeholder"
    );
    assert!(
        matches!(
            scal(client, 2.0, &mut ph_mut),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: scal must reject placeholder"
    );
    assert!(
        matches!(
            dot(client, &ph, &ph),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: dot must reject placeholder"
    );
    // reduce_sum into a scalar placeholder out: x is the placeholder → rejected.
    let mut scalar_out = upload::<f64>(client, &[0.0_f64], vec![1]).expect("upload out");
    assert!(
        matches!(
            reduce_sum(client, &ph, 0, &mut scalar_out),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: reduce_sum must reject placeholder x"
    );

    // gemm/gemv/transpose: 2-D placeholders pass the rank/shape checks but were
    // never uploaded, so the registry lookup misses → UnallocatedBuffer.
    let m2 = Tensor::placeholder(vec![2, 2]);
    let mut m2_out = Tensor::placeholder(vec![2, 2]);
    assert!(
        matches!(
            gemm(client, &m2, &m2, &mut m2_out),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: gemm must reject placeholder"
    );
    let vec2 = Tensor::placeholder(vec![2]);
    let mut vec2_out = Tensor::placeholder(vec![2]);
    assert!(
        matches!(
            gemv(client, &m2, &vec2, &mut vec2_out),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: gemv must reject placeholder"
    );
    assert!(
        matches!(
            transpose(client, &m2, &mut m2_out),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: transpose must reject placeholder"
    );

    // Non-rank-2 inputs are a clean DimensionMismatch, not a panic.
    let one_d = upload::<f64>(client, &[1.0_f64, 2.0, 3.0, 4.0], vec![4]).expect("upload 1-D");
    let mut one_d_out = upload::<f64>(client, &[0.0_f64; 4], vec![4]).expect("upload 1-D out");
    assert!(
        matches!(
            transpose(client, &one_d, &mut one_d_out),
            Err(AlgebraError::DimensionMismatch { .. })
        ),
        "{label}: transpose must reject non-rank-2 input"
    );
    release::<f64>(&one_d);
    release::<f64>(&one_d_out);

    // Length mismatch is enforced by the dense launcher under the Tensor path.
    let a = upload::<f64>(client, &[1.0_f64, 2.0], vec![2]).expect("upload a");
    let mut b = upload::<f64>(client, &[1.0_f64, 2.0, 3.0], vec![3]).expect("upload b");
    assert!(
        matches!(
            axpy(client, 1.0, &a, &mut b),
            Err(AlgebraError::DimensionMismatch { .. })
        ),
        "{label}: axpy must reject mismatched lengths"
    );

    // Double-release is a no-op (idempotent) and the buffer is gone afterward.
    release::<f64>(&a);
    release::<f64>(&a);
    assert!(
        matches!(
            download::<f64>(client, &a),
            Err(AlgebraError::UnallocatedBuffer { .. })
        ),
        "{label}: released buffer must be gone"
    );
    release::<f64>(&b);
    release::<f64>(&scalar_out);
}

/// Device residency: a chain of ops on the SAME resident tensors with NO
/// intermediate `download` must compose correctly. This is the whole point of
/// handle caching — `scal`'s in-place result feeds `axpy`, whose in-place result
/// feeds `dot`, all without re-uploading between ops. We download only at the end
/// and check every buffer plus the final reduction against host references.
fn check_resident_chain(client: &AlgebraClient, label: &str) {
    let x0 = vec![1.0_f64, 2.0, 3.0, 4.0];
    let y0 = vec![10.0_f64, 20.0, 30.0, 40.0];

    let mut xt = upload::<f64>(client, &x0, vec![4]).expect("upload x");
    let mut yt = upload::<f64>(client, &y0, vec![4]).expect("upload y");

    // x *= 2  → x stays resident, mutated in place.
    scal(client, 2.0, &mut xt).expect("scal");
    // y += 1 * x  → reads resident x, mutates resident y in place (no re-upload).
    axpy(client, 1.0, &xt, &mut yt).expect("axpy");
    // d = dot(x, y) → reduces the two resident buffers (no re-upload).
    let d = dot(client, &xt, &yt).expect("dot");

    // Host reference for the same chain.
    let x_ref: Vec<f64> = x0.iter().map(|v| v * 2.0).collect();
    let y_ref: Vec<f64> = y0.iter().zip(&x_ref).map(|(y, x)| y + x).collect();
    let d_ref: f64 = x_ref.iter().zip(&y_ref).map(|(a, b)| a * b).sum();

    assert_eq!(
        download::<f64>(client, &xt).expect("download x"),
        x_ref,
        "{label}: chained scal left x wrong"
    );
    assert_eq!(
        download::<f64>(client, &yt).expect("download y"),
        y_ref,
        "{label}: chained axpy (over resident x) left y wrong"
    );
    assert!(
        (d - d_ref).abs() < TOL,
        "{label}: chained dot over resident buffers wrong: {d} vs {d_ref}"
    );

    release::<f64>(&xt);
    release::<f64>(&yt);
}

fn run_all(client: &AlgebraClient, label: &str) {
    check_axpy(client, label);
    check_scal(client, label);
    check_dot(client, label);
    check_reduce_sum(client, label);
    check_gemm(client, label);
    check_gemv(client, label);
    check_transpose(client, label);
    check_resident_chain(client, label);
    check_error_contracts(client, label);
}

#[test]
fn tensor_registry_roundtrip_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    run_all(&client, "CPU");
}

#[cfg(feature = "rocm")]
#[test]
fn tensor_registry_roundtrip_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    run_all(&client, "ROCm");
}
