//! Plan 16-07 Task 4 test 1 — `spatial2spin` / `spin2spatial`, both ranks.
//!
//! ORACLE-FREE. `kccsd.py:237-329`'s packing folds the `aa`, `ab`, `bb` and
//! `abba` blocks into one `(nocc², nvir²)` view through four `takebak_2d`
//! calls with TRANSPOSED index products — precisely where an off-by-one is
//! silent, because every wrong version still has the right shape.

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::kccsd::{
    restricted_t2_to_aa, spatial2spin_t1, spatial2spin_t2, spin2spatial_t1, spin2spatial_t2,
};
use pyscf_pbc_cc::ZArr;
use pyscf_pbc_lib::get_kconserv;

fn lattice() -> [[f64; 3]; 3] {
    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
}

fn kpts(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| [0.0, 0.0, i as f64 * std::f64::consts::PI / n as f64])
        .collect()
}

fn ramp(shape: &[usize], off: f64) -> ZArr {
    let n: usize = shape.iter().product();
    ZArr::from_ctensor(
        shape,
        CTensor::from_planes(
            (0..n).map(|i| ((i as f64 + off) * 0.37).sin()).collect(),
            (0..n).map(|i| ((i as f64 + off) * 0.11).cos()).collect(),
        ),
    )
    .expect("shape")
}

/// `orbspin` as `kccsd.py:519-521` guesses it with no tag: alternating
/// alpha/beta over the whole `nmo`.
fn orbspin(nkpts: usize, nmo: usize) -> Vec<Vec<u8>> {
    (0..nkpts)
        .map(|_| (0..nmo).map(|p| (p % 2) as u8).collect())
        .collect()
}

/// `spin2spatial(spatial2spin(x)) == x` BIT-IDENTICALLY, for `t1`.
#[test]
fn t1_roundtrip_is_bit_identical() {
    let (nk, nocc, nvir) = (2usize, 4usize, 4usize);
    let os = orbspin(nk, nocc + nvir);
    let t1a = ramp(&[nk, nocc / 2, nvir / 2], 0.5);
    let t1b = ramp(&[nk, nocc / 2, nvir / 2], 7.5);
    let spin = spatial2spin_t1(&t1a, &t1b, &os, nocc, nvir).expect("spatial2spin");
    assert_eq!(spin.shape(), &[nk, nocc, nvir]);
    let (a, b) = spin2spatial_t1(&spin, &os, nocc).expect("spin2spatial");
    assert_eq!(a, t1a, "t1a does not survive the round trip");
    assert_eq!(b, t1b, "t1b does not survive the round trip");
}

/// The forward `t1` direction against a hand-built case: with alternating
/// `orbspin`, the alpha block lands on the EVEN spin-orbitals and the beta
/// block on the odd ones.
#[test]
fn t1_forward_places_the_spin_blocks_where_orbspin_says() {
    let (nk, nocc, nvir) = (1usize, 2usize, 2usize);
    let os = orbspin(nk, nocc + nvir);
    let mut t1a = ZArr::zeros(&[nk, 1, 1]);
    t1a.data_mut().re[0] = 3.0;
    let mut t1b = ZArr::zeros(&[nk, 1, 1]);
    t1b.data_mut().re[0] = -5.0;
    let s = spatial2spin_t1(&t1a, &t1b, &os, nocc, nvir).expect("spatial2spin");
    // occupied 0 is alpha, 1 is beta; virtual 0 is alpha, 1 is beta.
    assert_eq!(s.at(&[0, 0, 0]).unwrap(), (3.0, 0.0), "alpha -> (o0, v0)");
    assert_eq!(s.at(&[0, 1, 1]).unwrap(), (-5.0, 0.0), "beta -> (o1, v1)");
    assert_eq!(s.at(&[0, 0, 1]).unwrap(), (0.0, 0.0), "no alpha-beta leakage");
    assert_eq!(s.at(&[0, 1, 0]).unwrap(), (0.0, 0.0), "no beta-alpha leakage");
}

/// `spin2spatial(spatial2spin(x)) == x` for `t2`'s three blocks.
///
/// The `ab` block is the one that matters: `spatial2spin_t2` writes it into
/// FOUR k-blocks (`[ki,kj,ka]`, `[kj,ki,kb]`, and the two NEGATED `abba`
/// blocks), and `spin2spatial_t2` reads it back from only the first.
#[test]
fn t2_roundtrip_recovers_the_three_spin_blocks() {
    let (nk, nocc, nvir) = (2usize, 4usize, 4usize);
    let (no, nv) = (nocc / 2, nvir / 2);
    let os = orbspin(nk, nocc + nvir);
    let kc = get_kconserv(&lattice(), &kpts(nk));

    let t2aa = ramp(&[nk, nk, nk, no, no, nv, nv], 1.0);
    let t2ab = ramp(&[nk, nk, nk, no, no, nv, nv], 40.0);
    let t2bb = ramp(&[nk, nk, nk, no, no, nv, nv], 90.0);

    let spin = spatial2spin_t2(&t2aa, &t2ab, &t2bb, &os, &kc, nocc, nvir)
        .expect("spatial2spin_t2");
    assert_eq!(spin.shape(), &[nk, nk, nk, nocc, nocc, nvir, nvir]);
    let (a, b, c) = spin2spatial_t2(&spin, &os, &kc, nocc).expect("spin2spatial_t2");
    assert_eq!(a, t2aa, "the aa block does not survive the round trip");
    assert_eq!(b, t2ab, "the ab block does not survive the round trip");
    assert_eq!(c, t2bb, "the bb block does not survive the round trip");
}

/// The spin-orbital `t2` is ANTISYMMETRIC under exchanging the two virtuals
/// together with their k-points — the property that makes it a spin-orbital
/// amplitude at all, and the one a wrong `abba` sign would destroy.
#[test]
fn the_lifted_t2_is_antisymmetric() {
    let (nk, nocc, nvir) = (2usize, 4usize, 4usize);
    let (no, nv) = (nocc / 2, nvir / 2);
    let os = orbspin(nk, nocc + nvir);
    let kc = get_kconserv(&lattice(), &kpts(nk));

    // A RESTRICTED t2 lifted the way `kccsd.py:231-236` does.
    let t2 = ramp(&[nk, nk, nk, no, no, nv, nv], 3.0);
    let t2aa = restricted_t2_to_aa(&t2, &kc).expect("restricted_t2_to_aa");
    let spin =
        spatial2spin_t2(&t2aa, &t2, &t2aa, &os, &kc, nocc, nvir).expect("spatial2spin_t2");

    let mut worst = 0.0_f64;
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kc.get(ki, ka, kj) as usize;
                let x = spin.slice_leading(&[ki, kj, ka]).unwrap();
                // t2[ki,kj,ka][i,j,a,b] == -t2[ki,kj,kb][i,j,b,a]
                let y = spin
                    .slice_leading(&[ki, kj, kb])
                    .unwrap()
                    .transpose(&[0, 1, 3, 2])
                    .unwrap();
                for f in 0..x.len() {
                    worst = worst
                        .max((x.data().re[f] + y.data().re[f]).abs())
                        .max((x.data().im[f] + y.data().im[f]).abs());
                }
            }
        }
    }
    assert!(
        worst < 1e-13,
        "the lifted t2 is not antisymmetric in (a,b): worst {worst:e}"
    );
}

/// `restricted_t2_to_aa` is the antisymmetriser `kccsd.py:231-236` applies, and
/// its output is antisymmetric on its own.
#[test]
fn restricted_lift_produces_an_antisymmetric_aa_block() {
    let (nk, no, nv) = (2usize, 2usize, 2usize);
    let kc = get_kconserv(&lattice(), &kpts(nk));
    let t2 = ramp(&[nk, nk, nk, no, no, nv, nv], 11.0);
    let aa = restricted_t2_to_aa(&t2, &kc).expect("aa");
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kc.get(ki, ka, kj) as usize;
                let x = aa.slice_leading(&[ki, kj, ka]).unwrap();
                let y = aa
                    .slice_leading(&[ki, kj, kb])
                    .unwrap()
                    .transpose(&[0, 1, 3, 2])
                    .unwrap();
                for f in 0..x.len() {
                    assert!(
                        (x.data().re[f] + y.data().re[f]).abs() < 1e-14
                            && (x.data().im[f] + y.data().im[f]).abs() < 1e-14,
                        "aa is not antisymmetric at k ({ki},{kj},{ka})"
                    );
                }
            }
        }
    }
}
