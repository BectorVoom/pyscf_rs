//! Plan 20-07 — the complex, k-resolved NumPy boundary (`numpy_io.rs`).
//!
//! Round-trips planar `CTensor { re, im }` (D-PBC-02) through the interleaved
//! `Complex64` wire format and back, and asserts `to_bits()` equality on BOTH
//! planes. This is same-implementation A/B (pure element moves, no arithmetic),
//! so bit-exactness IS the correct standard here — never an epsilon.
//!
//! These tests drive the pyo3-free core (`ctensor_to_array` / `array_to_ctensor`
//! / `kmats_to_arrays` / `arrays_to_kmats`) on `ndarray` views. The Python-facing
//! wrappers (`ctensor_to_pyarray`, `to_ctensor`, `kmats_to_pylist`, …) are thin
//! shells over this core; they need a live interpreter and are covered by
//! `python/pyscf/tests/test_complex_boundary.py` (the default `abi3-py310`
//! feature turns on `pyo3/extension-module`, so no test binary here links
//! libpython — the same reason every other test in this directory is pyo3-free).

use numpy::Complex64;
use numpy::ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, ShapeBuilder, s};
use pyscf_algebra::CTensor;
use pyscf_py::numpy_io::{
    BufOrder, array_to_ctensor, arrays_to_kdms, arrays_to_kmats, ctensor_to_array, kdms_to_arrays,
    kmats_to_arrays,
};

/// Deterministic, bit-pattern-rich planes: signed zeros, subnormals, huge and
/// tiny magnitudes, and values whose last ulp matters.
fn planes(n: usize, seed: u64) -> CTensor {
    let special = [
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        -5e-324,
        1.0 + f64::EPSILON,
        -1e308,
        std::f64::consts::PI,
        0.1 + 0.2,
    ];
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map the high bits to a finite f64 with a full mantissa
        let bits = (state >> 12) | 0x3FF0_0000_0000_0000;
        let v = f64::from_bits(bits) - 1.5;
        if state & 1 == 0 { v } else { -v * 1e-7 }
    };
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for i in 0..n {
        if i < special.len() {
            re.push(special[i]);
            im.push(special[special.len() - 1 - i]);
        } else {
            re.push(next());
            im.push(next());
        }
    }
    CTensor { re, im }
}

fn assert_bits_eq(a: &CTensor, b: &CTensor, what: &str) {
    assert_eq!(a.re.len(), b.re.len(), "{what}: re length");
    assert_eq!(a.im.len(), b.im.len(), "{what}: im length");
    for (i, (x, y)) in a.re.iter().zip(&b.re).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: re[{i}] {x:e} vs {y:e}");
    }
    for (i, (x, y)) in a.im.iter().zip(&b.im).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: im[{i}] {x:e} vs {y:e}");
    }
}

#[test]
fn f_order_buffer_round_trips_bitwise_and_indexes_column_major() {
    let (n0, n1) = (3usize, 5usize);
    let t = planes(n0 * n1, 1);
    let arr = ctensor_to_array(&t, &[n0, n1], BufOrder::F).expect("shape matches");
    // element (i, j) of an F-order buffer lives at i + j*n0 — the `.f()` trick.
    for i in 0..n0 {
        for j in 0..n1 {
            let z = arr[[i, j]];
            assert_eq!(z.re.to_bits(), t.re[i + j * n0].to_bits(), "re ({i},{j})");
            assert_eq!(z.im.to_bits(), t.im[i + j * n0].to_bits(), "im ({i},{j})");
        }
    }
    let back = array_to_ctensor(arr.view(), BufOrder::F);
    assert_bits_eq(&t, &back, "F round-trip");
}

#[test]
fn c_order_buffer_round_trips_bitwise_and_indexes_row_major() {
    let (n0, n1) = (4usize, 2usize);
    let t = planes(n0 * n1, 2);
    let arr = ctensor_to_array(&t, &[n0, n1], BufOrder::C).expect("shape matches");
    for i in 0..n0 {
        for j in 0..n1 {
            assert_eq!(arr[[i, j]].re.to_bits(), t.re[i * n1 + j].to_bits());
            assert_eq!(arr[[i, j]].im.to_bits(), t.im[i * n1 + j].to_bits());
        }
    }
    let back = array_to_ctensor(arr.view(), BufOrder::C);
    assert_bits_eq(&t, &back, "C round-trip");
}

#[test]
fn c_contiguous_input_normalises_to_the_requested_order() {
    // A C-contiguous numpy-style array read back as an F-order CTensor must
    // be the element-wise column-major flattening, not the raw memory.
    let (n0, n1, n2) = (2usize, 3usize, 4usize);
    let data: Vec<Complex64> = (0..n0 * n1 * n2)
        .map(|k| Complex64::new(k as f64 + 0.25, -(k as f64) - 0.5))
        .collect();
    let c_arr = ArrayD::from_shape_vec(IxDyn(&[n0, n1, n2]), data).expect("shape");
    assert!(c_arr.is_standard_layout());
    let t = array_to_ctensor(c_arr.view(), BufOrder::F);
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let flat = i + j * n0 + k * n0 * n1;
                assert_eq!(t.re[flat].to_bits(), c_arr[[i, j, k]].re.to_bits());
                assert_eq!(t.im[flat].to_bits(), c_arr[[i, j, k]].im.to_bits());
            }
        }
    }
    // and back through the F writer reproduces the same logical array
    let again = ctensor_to_array(&t, &[n0, n1, n2], BufOrder::F).expect("shape");
    assert_eq!(again.shape(), c_arr.shape());
    for (a, b) in again.iter().zip(c_arr.iter()) {
        assert_eq!(a.re.to_bits(), b.re.to_bits());
        assert_eq!(a.im.to_bits(), b.im.to_bits());
    }
}

#[test]
fn non_contiguous_transposed_input_does_not_silently_transpose() {
    // stride-fuzz spirit (ci.yml stride-fuzz): a.T, a[::2], a[:, 1:5] and a
    // negative-stride view must all read by LOGICAL index.
    let (n0, n1) = (6usize, 7usize);
    let t = planes(n0 * n1, 3);
    let arr = ctensor_to_array(&t, &[n0, n1], BufOrder::C).expect("shape");

    let views: Vec<(&str, ArrayViewD<'_, Complex64>)> = vec![
        ("a.T", arr.t()),
        ("a[::2]", arr.slice(s![..;2, ..]).into_dyn()),
        ("a[:,1:5]", arr.slice(s![.., 1..5]).into_dyn()),
        ("a[::-1, ::3]", arr.slice(s![..;-1, ..;3]).into_dyn()),
    ];
    for (name, v) in views {
        assert!(
            !v.is_standard_layout() || v.len() <= 1,
            "{name} should be non-contiguous for this test to mean anything"
        );
        let shape = v.shape().to_vec();
        for order in [BufOrder::C, BufOrder::F] {
            let ct = array_to_ctensor(v.view(), order);
            let rebuilt = ctensor_to_array(&ct, &shape, order).expect("shape");
            assert_eq!(rebuilt.shape(), v.shape(), "{name} {order:?}: shape");
            for (idx, z) in v.indexed_iter() {
                let r = rebuilt[idx.clone()];
                assert_eq!(
                    r.re.to_bits(),
                    z.re.to_bits(),
                    "{name} {order:?} re {idx:?}"
                );
                assert_eq!(
                    r.im.to_bits(),
                    z.im.to_bits(),
                    "{name} {order:?} im {idx:?}"
                );
            }
        }
    }

    // and the transpose explicitly: T[i,j] == A[j,i], bitwise, never A[i,j].
    let tt = array_to_ctensor(arr.t(), BufOrder::C);
    for i in 0..n1 {
        for j in 0..n0 {
            assert_eq!(tt.re[i * n0 + j].to_bits(), t.re[j * n1 + i].to_bits());
            assert_eq!(tt.im[i * n0 + j].to_bits(), t.im[j * n1 + i].to_bits());
        }
    }
}

#[test]
fn length_or_plane_mismatch_errors_and_never_truncates() {
    let t = planes(12, 4);
    assert!(
        ctensor_to_array(&t, &[3, 5], BufOrder::F).is_err(),
        "15 != 12"
    );
    assert!(
        ctensor_to_array(&t, &[2, 5], BufOrder::C).is_err(),
        "10 != 12"
    );
    let mut bad = t.clone();
    bad.im.pop();
    assert!(
        ctensor_to_array(&bad, &[bad.re.len()], BufOrder::F).is_err(),
        "unequal planes must error"
    );
}

#[test]
fn kmats_of_eight_blocks_round_trip_as_a_list_of_eight_arrays() {
    let nao = 4usize;
    let kmats: Vec<CTensor> = (0..8).map(|k| planes(nao * nao, 100 + k)).collect();
    let shapes = vec![vec![nao, nao]; 8];
    let arrays = kmats_to_arrays(&kmats, &shapes, BufOrder::C).expect("shapes match");
    assert_eq!(
        arrays.len(),
        8,
        "one array per k-point, not a stacked block"
    );
    for a in &arrays {
        assert_eq!(a.ndim(), 2, "each k block stays 2-D");
    }
    let views: Vec<ArrayViewD<'_, Complex64>> = arrays.iter().map(|a| a.view()).collect();
    let (back, back_shapes) = arrays_to_kmats(&views, BufOrder::C);
    assert_eq!(back_shapes, shapes);
    for (k, (a, b)) in kmats.iter().zip(&back).enumerate() {
        assert_bits_eq(a, b, &format!("kmats[{k}]"));
    }
}

#[test]
fn ksymm_blocks_of_different_shapes_require_the_list_form() {
    // Under k-symmetry per-k blocks need not share a shape (e.g. a truncated
    // MO space at one k). A stacked (nk, n, m) array cannot hold these.
    let shapes = vec![vec![5usize, 3], vec![5usize, 2]];
    let kmats: Vec<CTensor> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| planes(s[0] * s[1], 200 + k as u64))
        .collect();
    let arrays = kmats_to_arrays(&kmats, &shapes, BufOrder::F).expect("shapes match");
    assert_ne!(arrays[0].shape(), arrays[1].shape());
    // proof that the stacked form is impossible for this data
    let stacked = numpy::ndarray::stack(
        Axis(0),
        &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
    );
    assert!(stacked.is_err(), "heterogeneous k blocks cannot be stacked");

    let views: Vec<ArrayViewD<'_, Complex64>> = arrays.iter().map(|a| a.view()).collect();
    let (back, back_shapes) = arrays_to_kmats(&views, BufOrder::F);
    assert_eq!(back_shapes, shapes);
    for (k, (a, b)) in kmats.iter().zip(&back).enumerate() {
        assert_bits_eq(a, b, &format!("ksymm kmats[{k}]"));
    }

    // a shape list of the wrong length errors instead of zipping short
    assert!(kmats_to_arrays(&kmats, &shapes[..1], BufOrder::F).is_err());
}

#[test]
fn kdms_carry_the_spin_axis_as_a_nested_list() {
    let nao = 3usize;
    let nset = 2usize;
    let nk = 4usize;
    let kdms: Vec<Vec<CTensor>> = (0..nset)
        .map(|s| {
            (0..nk)
                .map(|k| planes(nao * nao, 300 + (s * nk + k) as u64))
                .collect()
        })
        .collect();
    let shapes = vec![vec![vec![nao, nao]; nk]; nset];
    let nested = kdms_to_arrays(&kdms, &shapes, BufOrder::C).expect("shapes match");
    assert_eq!(nested.len(), nset);
    assert!(nested.iter().all(|kset| kset.len() == nk));

    let views: Vec<Vec<ArrayViewD<'_, Complex64>>> = nested
        .iter()
        .map(|kset| kset.iter().map(|a| a.view()).collect())
        .collect();
    let (back, back_shapes) = arrays_to_kdms(&views, BufOrder::C);
    assert_eq!(back_shapes, shapes);
    for s in 0..nset {
        for k in 0..nk {
            assert_bits_eq(&kdms[s][k], &back[s][k], &format!("kdms[{s}][{k}]"));
        }
    }
}

#[test]
fn f_order_writer_matches_the_gto_intor_spinor_precedent() {
    // gto.rs intor_spinor: interleave then ArrayD::from_shape_vec(IxDyn(&shape).f(), data).
    let shape = [2usize, 3, 2];
    let t = planes(12, 5);
    let data: Vec<Complex64> =
        t.re.iter()
            .zip(t.im.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
    let precedent = ArrayD::from_shape_vec(IxDyn(&shape).f(), data).expect("shape");
    let ours = ctensor_to_array(&t, &shape, BufOrder::F).expect("shape");
    assert_eq!(ours.shape(), precedent.shape());
    assert_eq!(
        ours.strides(),
        precedent.strides(),
        "same F-order memory layout"
    );
    for (a, b) in ours.iter().zip(precedent.iter()) {
        assert_eq!(a.re.to_bits(), b.re.to_bits());
        assert_eq!(a.im.to_bits(), b.im.to_bits());
    }
}
