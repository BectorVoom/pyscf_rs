//! K-08 — the resident [`AoKAccumulator`] against the single-shot slice API.
//!
//! The two surfaces must fold a lattice-image sequence into exactly the same
//! `(nkpts, n)` planes. That is the property the driver in `pyscf-pbc-gto`
//! depends on when it switched from passing both planes through the host on
//! every image to keeping them on the device for the whole loop: the change is
//! only worth anything if it is also invisible in the result.
//!
//! Equality here is BIT-exact, not approximate. Both paths run the same kernel
//! over the same images in the same order, and the accumulation is elementwise
//! (`out[i] += p*v`), so nothing reassociates between them — a tolerance would
//! hide a real difference rather than absorb a legitimate one.

use pyscf_algebra::select_backend;
use pyscf_kernels::pbc::{AoKAccumulator, eval_ao_k_accumulate};

/// Deterministic pseudo-random fill; a fixed LCG so a failure reproduces.
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// Fold `nimgs` synthetic images through both surfaces and compare.
///
/// `n` values are chosen to exercise both a vectorizable width and one that
/// forces the width down to 1 (an odd `n`), since the resident kernel indexes
/// `p` in vectors while the accumulation is per-scalar.
fn both_paths_agree(nkpts: usize, n: usize, nimgs: usize) {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;

    let mut acc = AoKAccumulator::zeros(client, nkpts, n);
    assert_eq!(acc.shape(), (nkpts, n));

    let mut ref_re = vec![0.0f64; nkpts * n];
    let mut ref_im = vec![0.0f64; nkpts * n];

    for m in 0..nimgs {
        let ao = fill(n, 0x51ed_270b + m as u64);
        let pr = fill(nkpts, 0x9e37_79b9 + m as u64);
        let pi = fill(nkpts, 0xc2b2_ae35 + m as u64);

        acc.accumulate(client, &ao, &pr, &pi)
            .expect("resident accumulate");

        let (re, im) = eval_ao_k_accumulate(client, &ao, &pr, &pi, &ref_re, &ref_im)
            .expect("single-shot accumulate");
        ref_re = re;
        ref_im = im;
    }

    let (got_re, got_im) = acc.into_planes(client);
    assert_eq!(got_re.len(), nkpts * n, "re plane length");
    assert_eq!(got_im.len(), nkpts * n, "im plane length");

    for i in 0..nkpts * n {
        assert_eq!(
            got_re[i],
            ref_re[i],
            "re[{i}] (k {}, p {}) after {nimgs} images, shape ({nkpts}, {n})",
            i / n,
            i % n
        );
        assert_eq!(
            got_im[i],
            ref_im[i],
            "im[{i}] (k {}, p {}) after {nimgs} images, shape ({nkpts}, {n})",
            i / n,
            i % n
        );
    }
}

#[test]
fn resident_accumulator_matches_the_single_shot_api() {
    // Wide, vectorizable n; several k; enough images that an accumulator that
    // failed to persist across launches would be obvious.
    both_paths_agree(4, 256, 7);
}

#[test]
fn resident_accumulator_matches_with_an_unvectorizable_width() {
    // n prime → `line_size_for` falls back to a width of 1, exercising the
    // scalar degeneration of the vectorized kernel.
    both_paths_agree(3, 97, 5);
}

#[test]
fn resident_accumulator_handles_a_single_kpoint_and_one_image() {
    both_paths_agree(1, 64, 1);
}

/// An empty grid or an empty basis reaches K-08 with `n == 0`; a single k-point
/// mesh with no images reaches it with nothing to fold. Neither may allocate a
/// zero-length device buffer or launch an empty grid.
#[test]
fn degenerate_shapes_produce_empty_planes_without_launching() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;

    for (nkpts, n) in [(0usize, 0usize), (4, 0), (0, 32)] {
        let mut acc = AoKAccumulator::zeros(client, nkpts, n);
        assert_eq!(acc.shape(), (nkpts, n));
        // Accumulating a correctly-shaped (but empty) image is a no-op, not an error.
        acc.accumulate(client, &vec![0.0; n], &vec![0.0; nkpts], &vec![0.0; nkpts])
            .expect("degenerate accumulate is a no-op");
        let (re, im) = acc.into_planes(client);
        assert!(re.is_empty(), "re plane for shape ({nkpts}, {n})");
        assert!(im.is_empty(), "im plane for shape ({nkpts}, {n})");
    }
}

/// Shape errors are reported, not silently launched with mismatched buffers.
#[test]
fn accumulate_rejects_mismatched_operands() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;

    let mut acc = AoKAccumulator::zeros(client, 3, 16);
    assert!(
        acc.accumulate(client, &fill(15, 1), &fill(3, 2), &fill(3, 3))
            .is_err(),
        "an AO block of the wrong length must be rejected"
    );
    assert!(
        acc.accumulate(client, &fill(16, 1), &fill(2, 2), &fill(3, 3))
            .is_err(),
        "phase vectors disagreeing with nkpts must be rejected"
    );
}

#[test]
fn device_path_matches_slice_path_bit_exact() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    let (nkpts, n) = (3, 96);
    let ao = fill(n, 41);
    let pr = fill(nkpts, 42);
    let pi = fill(nkpts, 43);
    let device = pyscf_kernels::AoBlockDevice::from_values(client, &ao, vec![n]);
    let mut acc = AoKAccumulator::zeros(client, nkpts, n);
    acc.accumulate_device(client, &device, &pr, &pi)
        .expect("device accumulate");
    let got = acc.into_planes(client);
    let expected = eval_ao_k_accumulate(
        client,
        &ao,
        &pr,
        &pi,
        &vec![0.0; nkpts * n],
        &vec![0.0; nkpts * n],
    )
    .expect("slice accumulate");
    assert_eq!(got, expected);
}

#[test]
fn device_scatter_path_matches_zero_filled_slice_bit_exact() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    let (nkpts, ngrids, nao, comp) = (3, 17, 4, 2);
    let index = vec![0usize, 2, 3, 8, 13, 16];
    let sub = fill(comp * index.len() * nao, 51);
    let mut full = vec![0.0f64; comp * ngrids * nao];
    for c in 0..comp {
        for a in 0..nao {
            for (j, &g) in index.iter().enumerate() {
                full[c * ngrids * nao + a * ngrids + g] =
                    sub[c * index.len() * nao + a * index.len() + j];
            }
        }
    }
    let pr = fill(nkpts, 52);
    let pi = fill(nkpts, 53);
    let device =
        pyscf_kernels::AoBlockDevice::from_values(client, &sub, vec![comp, index.len(), nao]);
    let mut acc = AoKAccumulator::zeros(client, nkpts, full.len());
    acc.accumulate_device_scatter(client, &device, &index, ngrids, nao, comp, &pr, &pi)
        .expect("device scatter accumulate");
    let got = acc.into_planes(client);
    let expected = eval_ao_k_accumulate(
        client,
        &full,
        &pr,
        &pi,
        &vec![0.0; nkpts * full.len()],
        &vec![0.0; nkpts * full.len()],
    )
    .expect("zero-filled slice accumulate");
    assert_eq!(got, expected);
}
