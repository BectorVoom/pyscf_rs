//! Plan 09-02 Task 8 — `CTensor` (PBC-MASTER-PLAN §5.1) host-type contract.
//!
//! Verified here: interleaved <-> planar round-trip is EXACT (bit-for-bit, not
//! epsilon-tolerant) over 1000 pseudo-random values; `conj` negates only the
//! imaginary plane; `is_real` respects the tolerance; the trivial constructors
//! keep the two planes equal-length.
//!
//! Not verified here: device operations over `CTensor` — see `zgemm.rs`,
//! `zeigh.rs`, `zoracle_determinism.rs`.

use pyscf_algebra::CTensor;

/// Deterministic LCG (Knuth/MMIX constants) — reproducible "random" values
/// without pulling in `rand`. Same shape as `tests/gemm_oracle.rs`.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        u * 2.0 - 1.0 // [-1, 1)
    }
}

#[test]
fn interleaved_round_trip_is_bit_exact_for_1000_values() {
    let mut rng = Lcg::new(0x5EED_0902);
    // 1000 complex values -> 2000 interleaved f64.
    let z: Vec<f64> = (0..2000).map(|_| rng.next_f64()).collect();

    let c = CTensor::from_interleaved(&z);
    assert_eq!(c.len(), 1000);
    assert_eq!(c.re.len(), c.im.len());

    // Planes carry exactly the even/odd positions.
    for k in 0..1000 {
        assert_eq!(c.re[k].to_bits(), z[2 * k].to_bits(), "re plane at {k}");
        assert_eq!(c.im[k].to_bits(), z[2 * k + 1].to_bits(), "im plane at {k}");
    }

    // Round-trip is bit-identical — the conversion moves elements, never
    // arithmetic, so no rounding can occur.
    let back = c.to_interleaved();
    assert_eq!(back.len(), z.len());
    for (i, (a, b)) in back.iter().zip(z.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "round-trip differs at index {i}");
    }
}

#[test]
fn round_trip_preserves_non_finite_and_signed_zero() {
    let z = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        f64::MIN_POSITIVE,
    ];
    let back = CTensor::from_interleaved(&z).to_interleaved();
    for (i, (a, b)) in back.iter().zip(z.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "index {i}");
    }
}

#[test]
fn conj_negates_only_the_imaginary_plane() {
    let mut rng = Lcg::new(0xC0FFEE);
    let re: Vec<f64> = (0..64).map(|_| rng.next_f64()).collect();
    let im: Vec<f64> = (0..64).map(|_| rng.next_f64()).collect();
    let c = CTensor::from_interleaved(
        &re.iter()
            .zip(im.iter())
            .flat_map(|(r, i)| [*r, *i])
            .collect::<Vec<f64>>(),
    );

    let k = c.conj();
    assert_eq!(k.len(), c.len());
    for i in 0..c.len() {
        assert_eq!(k.re[i].to_bits(), c.re[i].to_bits(), "re changed at {i}");
        assert_eq!(k.im[i].to_bits(), (-c.im[i]).to_bits(), "im at {i}");
    }
    // Involution: conj(conj(x)) == x.
    let kk = k.conj();
    assert_eq!(kk.re, c.re);
    assert_eq!(kk.im, c.im);
}

#[test]
fn is_real_respects_the_tolerance() {
    let mut c = CTensor::from_real(&[1.0, 2.0, 3.0]);
    assert!(c.im.iter().all(|v| *v == 0.0), "from_real must zero `im`");
    // max|im| = 0 < any positive tol.
    assert!(c.is_real(1e-300));
    // Strict `<` (§5.1 says `max|im| < tol`), so tol = 0 is never satisfied.
    assert!(!c.is_real(0.0));

    c.im[1] = 1e-9;
    assert!(c.is_real(1e-8), "1e-9 < 1e-8 must count as real");
    assert!(!c.is_real(1e-9), "1e-9 < 1e-9 is false — strict comparison");
    assert!(!c.is_real(1e-10));

    // Sign of the imaginary part is irrelevant — the test is on |im|.
    c.im[1] = -1e-9;
    assert!(c.is_real(1e-8));
    assert!(!c.is_real(1e-10));
}

#[test]
fn constructors_keep_planes_equal_length() {
    assert_eq!(CTensor::zeros(7).re.len(), 7);
    assert_eq!(CTensor::zeros(7).im.len(), 7);
    assert_eq!(CTensor::zeros(7).len(), 7);
    assert!(CTensor::zeros(0).is_empty());
    assert!(!CTensor::zeros(1).is_empty());
    assert_eq!(CTensor::from_real(&[1.0, 2.0]).len(), 2);
    assert_eq!(CTensor::from_interleaved(&[]).len(), 0);
    assert_eq!(CTensor::default().len(), 0);
    assert!(CTensor::default().is_empty());
}
