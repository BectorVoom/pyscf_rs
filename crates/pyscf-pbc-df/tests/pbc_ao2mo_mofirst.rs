mod common;

use common::he_all_electron;
use pyscf_pbc_df::pbc_ao2mo::{
    CoulGCache, aft_general, aft_general_mo_first, aft_get_eri, aft_get_eri_with_cache,
    fft_general, fft_general_mo_first, fft_get_eri, fft_get_eri_with_cache,
};
use pyscf_pbc_df::{Aftdf, Fftdf, MoCoeff};

fn max_dev(a: &pyscf_algebra::CTensor, b: &pyscf_algebra::CTensor) -> f64 {
    a.re.iter()
        .zip(&b.re)
        .map(|(x, y)| (x - y).abs())
        .chain(a.im.iter().zip(&b.im).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn cache_is_bit_identical() {
    let k = [[0.0; 3]];
    let f = Fftdf::with_mesh(he_all_electron(), &k, [9, 9, 9]).expect("fftdf");
    let a = Aftdf::with_mesh(he_all_electron(), &k, [9, 9, 9]).expect("aftdf");
    let cache = CoulGCache::build(&f.cell, f.mesh, [0.0; 3]).expect("cache");
    let k4 = [[0.0; 3]; 4];
    assert_eq!(
        fft_get_eri(&f, k4).unwrap(),
        fft_get_eri_with_cache(&f, k4, &cache).unwrap()
    );
    assert_eq!(
        aft_get_eri(&a, k4).unwrap(),
        aft_get_eri_with_cache(&a, k4, &cache).unwrap()
    );
    let mo = MoCoeff::identity(f.cell.mol.nao_nr);
    let mos = [&mo, &mo, &mo, &mo];
    let fft_plain = fft_general_mo_first(&f, mos, k4, None).unwrap();
    let fft_cached = fft_general_mo_first(&f, mos, k4, Some(&cache)).unwrap();
    assert_eq!(fft_plain.data, fft_cached.data);
    assert_eq!(fft_plain.row, fft_cached.row);
    assert_eq!(fft_plain.col, fft_cached.col);
    let aft_plain = aft_general_mo_first(&a, mos, k4, None).unwrap();
    let aft_cached = aft_general_mo_first(&a, mos, k4, Some(&cache)).unwrap();
    assert_eq!(aft_plain.data, aft_cached.data);
    assert_eq!(aft_plain.row, aft_cached.row);
    assert_eq!(aft_plain.col, aft_cached.col);
}

#[test]
fn mo_first_matches_ao_first() {
    let k = [[0.0; 3]];
    let f = Fftdf::with_mesh(he_all_electron(), &k, [9, 9, 9]).expect("fftdf");
    let a = Aftdf::with_mesh(he_all_electron(), &k, [9, 9, 9]).expect("aftdf");
    let mo = MoCoeff::identity(f.cell.mol.nao_nr);
    let mos = [&mo, &mo, &mo, &mo];
    let k4 = [[0.0; 3]; 4];
    let ff0 = fft_general(&f, mos, k4).unwrap();
    let ff1 = fft_general_mo_first(&f, mos, k4, None).unwrap();
    let aa0 = aft_general(&a, mos, k4).unwrap();
    let aa1 = aft_general_mo_first(&a, mos, k4, None).unwrap();
    assert!(
        max_dev(&ff0.data, &ff1.data) < 2e-11,
        "FFT residual {}",
        max_dev(&ff0.data, &ff1.data)
    );
    assert!(
        max_dev(&aa0.data, &aa1.data) < 2e-11,
        "AFT residual {}",
        max_dev(&aa0.data, &aa1.data)
    );
}

#[test]
fn mo_first_matches_ao_first_away_from_gamma() {
    let cell = he_all_electron();
    let kpts = cell.make_kpts([1, 1, 2]).expect("k-points");
    let f = Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("fftdf");
    let a = Aftdf::with_mesh(cell, &kpts, [9, 9, 9]).expect("aftdf");
    let mo = MoCoeff::identity(f.cell.mol.nao_nr);
    let mos = [&mo, &mo, &mo, &mo];
    let kc = pyscf_pbc_lib::get_kconserv(&f.cell.a, &kpts);
    for ki in 0..2 {
        for ka in 0..2 {
            for kj in 0..2 {
                let kb = kc.get(ki, ka, kj) as usize;
                let k4 = [kpts[ki], kpts[ka], kpts[kj], kpts[kb]];
                let ff0 = fft_general(&f, mos, k4).unwrap();
                let ff1 = fft_general_mo_first(&f, mos, k4, None).unwrap();
                let aa0 = aft_general(&a, mos, k4).unwrap();
                let aa1 = aft_general_mo_first(&a, mos, k4, None).unwrap();
                let fr = max_dev(&ff0.data, &ff1.data);
                let ar = max_dev(&aa0.data, &aa1.data);
                assert!(
                    fr < 2e-11,
                    "FFT ({ki},{ka},{kj},{kb}) residual {fr}; ao={:?}+i{:?}, mo={:?}+i{:?}",
                    ff0.data.re,
                    ff0.data.im,
                    ff1.data.re,
                    ff1.data.im
                );
                assert!(ar < 2e-11, "AFT ({ki},{ka},{kj},{kb}) residual {ar}");
            }
        }
    }
}

/// A deterministic COMPLEX, non-identity MO block.
///
/// `mo_first_matches_ao_first_away_from_gamma` above runs on `he_all_electron`,
/// which is `sto-3g` — **one** AO — with `MoCoeff::identity`, so its MO
/// transform is the 1x1 identity and it cannot see a coefficient-handling
/// defect at all. It missed one: `aft_general_mo_first` conjugated the ket pair
/// AFTER the MO transform where `aft_ao2mo.py:215-216` conjugates it BEFORE,
/// which is the same thing for real coefficients and wrong by O(1) for complex
/// ones (measured `4.596e-1` against the AO-first route on the block below).
fn complex_mo(nao: usize, seed: u64) -> MoCoeff {
    let mut c = pyscf_algebra::CTensor::zeros(nao * nao);
    let mut s = seed;
    for v in 0..nao * nao {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        c.re[v] = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        c.im[v] = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
    }
    MoCoeff::new(nao, nao, c)
}

#[test]
fn mo_first_matches_ao_first_with_complex_coefficients() {
    let cell = common::he_631g();
    let kpts = cell.make_kpts([1, 1, 2]).expect("k-points");
    let f = Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("fftdf");
    let a = Aftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("aftdf");
    let nao = f.cell.mol.nao_nr;
    assert!(nao > 1, "this test is worthless on a one-AO cell");
    let m: Vec<MoCoeff> = (0..kpts.len())
        .map(|k| complex_mo(nao, 20 + k as u64))
        .collect();
    let kc = pyscf_pbc_lib::get_kconserv(&cell.a, &kpts);
    for ki in 0..kpts.len() {
        for ka in 0..kpts.len() {
            for kj in 0..kpts.len() {
                let kb = kc.get(ki, ka, kj) as usize;
                let k4 = [kpts[ki], kpts[ka], kpts[kj], kpts[kb]];
                let q = [&m[ki], &m[ka], &m[kj], &m[kb]];
                let fr = max_dev(
                    &fft_general(&f, q, k4).unwrap().data,
                    &fft_general_mo_first(&f, q, k4, None).unwrap().data,
                );
                let ar = max_dev(
                    &aft_general(&a, q, k4).unwrap().data,
                    &aft_general_mo_first(&a, q, k4, None).unwrap().data,
                );
                assert!(fr < 2e-13, "FFT ({ki},{ka},{kj},{kb}) residual {fr:.3e}");
                assert!(ar < 2e-13, "AFT ({ki},{ka},{kj},{kb}) residual {ar:.3e}");
            }
        }
    }
}
