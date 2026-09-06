//! S-07 (session 3 of `KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`): when
//! `kpts_band` is a subset of the sampling k-list — which is what every
//! k-symmetric driver passes (`kpts_ibz ⊂ kpts`) — the potential's AO table
//! is served from the sampling table instead of a second `eval_ao_kpts`.
//!
//! The claim is bit-exactness, and it is asserted, not argued: the same
//! `nr_rks` / `nr_uks` call with the reuse on and with it off
//! (`PYSCF_PBC_BAND_AO_REUSE=0`, the profiler's kill switch) must agree to the
//! bit on every output — `vmat` at every band point, `nelec`, `excsum`.
//!
//! Both routes live in ONE test function because the kill switch is a
//! process-wide environment variable.

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::test_systems::si;
use pyscf_pbc_scf::{KInitGuess, KScfConfig, Krhf};
use pyscf_pbc_symm::kpts::make_kpts;

const TIME_REVERSAL: bool = false;

fn same_bits(what: &str, a: &CTensor, b: &CTensor) {
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
fn band_subset_reuse_is_bit_exact_for_rks_and_uks() {
    let cell = si();
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    assert!(kpts.nkpts_ibz() < kpts.nkpts(), "the fixture must fold");
    // S-09: the reuse can only fire when the IBZ list is a BITWISE subset of
    // the sampling list. Without this the two arms below would both take the
    // second-evaluation path and agree trivially — which is exactly what
    // happened before S-09 (the profiler counted 4 cold AO tables with the
    // reuse "on").
    for (i, &bz) in kpts.ibz2bz.iter().enumerate() {
        for ax in 0..3 {
            assert_eq!(
                kpts.kpts_ibz[i][ax].to_bits(),
                kpts.kpts[bz][ax].to_bits(),
                "kpts_ibz[{i}] is not the bitwise copy of kpts[{bz}] (axis {ax})"
            );
        }
    }

    let mf = Krhf::new(cell.clone(), &kpts.kpts).expect("Krhf");
    let r = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-8,
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        })
        .expect("SCF");
    assert!(r.converged, "SCF did not converge");
    let grids = PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("uniform grids");
    let band = kpts.kpts_ibz.clone();
    let dms = vec![r.dm[0].clone()];
    // A genuinely open-shell pair, so the two spin channels differ.
    let half: Vec<CTensor> = r.dm[0]
        .iter()
        .map(|m| CTensor {
            re: m.re.iter().map(|x| 0.5 * x).collect(),
            im: m.im.iter().map(|x| 0.5 * x).collect(),
        })
        .collect();
    let third: Vec<CTensor> = r.dm[0]
        .iter()
        .map(|m| CTensor {
            re: m.re.iter().map(|x| x / 3.0).collect(),
            im: m.im.iter().map(|x| x / 3.0).collect(),
        })
        .collect();
    let udms = [vec![half], vec![third]];

    for xc in ["lda,vwn", "pbe"] {
        // Reuse ON (the default): one numint, one AO table.
        let ni_on = KNumInt::with_symmetry(&kpts);
        let on = ni_on
            .nr_rks(&cell, &grids, xc, &dms, 1, Some(&band))
            .expect("nr_rks with reuse");
        let on_u = ni_on
            .nr_uks(&cell, &grids, xc, &udms, 1, Some(&band))
            .expect("nr_uks with reuse");

        // Reuse OFF: a fresh numint (its own AO cache), the second
        // evaluation restored.
        // SAFETY: single-threaded test body; no other thread reads the
        // variable concurrently.
        unsafe { std::env::set_var("PYSCF_PBC_BAND_AO_REUSE", "0") };
        let ni_off = KNumInt::with_symmetry(&kpts);
        let off = ni_off.nr_rks(&cell, &grids, xc, &dms, 1, Some(&band));
        let off_u = ni_off.nr_uks(&cell, &grids, xc, &udms, 1, Some(&band));
        unsafe { std::env::remove_var("PYSCF_PBC_BAND_AO_REUSE") };
        let off = off.expect("nr_rks without reuse");
        let off_u = off_u.expect("nr_uks without reuse");

        assert_eq!(
            on.vmat[0].len(),
            band.len(),
            "{xc}: vmat is at the band points"
        );
        for (k, (a, b)) in on.vmat[0].iter().zip(&off.vmat[0]).enumerate() {
            same_bits(&format!("{xc} rks vmat[{k}]"), a, b);
        }
        assert_eq!(on.nelec[0].to_bits(), off.nelec[0].to_bits(), "{xc}: nelec");
        assert_eq!(
            on.excsum[0].to_bits(),
            off.excsum[0].to_bits(),
            "{xc}: excsum"
        );

        for spin in 0..2 {
            for (k, (a, b)) in on_u.vmat[spin][0]
                .iter()
                .zip(&off_u.vmat[spin][0])
                .enumerate()
            {
                same_bits(&format!("{xc} uks spin {spin} vmat[{k}]"), a, b);
            }
        }
        assert_eq!(
            on_u.nelec[0].0.to_bits(),
            off_u.nelec[0].0.to_bits(),
            "{xc}: nelec a"
        );
        assert_eq!(
            on_u.nelec[0].1.to_bits(),
            off_u.nelec[0].1.to_bits(),
            "{xc}: nelec b"
        );
        assert_eq!(
            on_u.excsum[0].to_bits(),
            off_u.excsum[0].to_bits(),
            "{xc}: uks excsum"
        );
        println!(
            "{xc}: reuse on/off bit-identical at {} band points",
            band.len()
        );
    }
}

/// S-08 — the same identity in the FFT density-fitting layer: `get_j_kpts`
/// and `get_k_kpts` at `kpts_band = kpts_ibz` with the band AO table served
/// from the sampling table (reuse on) vs evaluated separately (reuse off).
#[test]
fn fftdf_band_subset_reuse_is_bit_exact_for_j_and_k() {
    use pyscf_pbc_df::{Fftdf, get_j_kpts, get_k_kpts};
    let cell = si();
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    let mf = Krhf::new(cell.clone(), &kpts.kpts).expect("Krhf");
    let r = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-8,
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        })
        .expect("SCF");
    let band = kpts.kpts_ibz.clone();
    let dms = vec![r.dm[0].clone()];

    let df_on = Fftdf::with_mesh(cell.clone(), &kpts.kpts, cell.mesh).expect("FFTDF");
    let j_on = get_j_kpts(&df_on, &dms, 1, &kpts.kpts, Some(&band), None).expect("vj on");
    let k_on = get_k_kpts(&df_on, &dms, 1, &kpts.kpts, Some(&band), None, None).expect("vk on");

    // SAFETY: single-threaded test body.
    unsafe { std::env::set_var("PYSCF_PBC_BAND_AO_REUSE", "0") };
    let df_off = Fftdf::with_mesh(cell.clone(), &kpts.kpts, cell.mesh).expect("FFTDF");
    let j_off = get_j_kpts(&df_off, &dms, 1, &kpts.kpts, Some(&band), None);
    let k_off = get_k_kpts(&df_off, &dms, 1, &kpts.kpts, Some(&band), None, None);
    unsafe { std::env::remove_var("PYSCF_PBC_BAND_AO_REUSE") };
    let j_off = j_off.expect("vj off");
    let k_off = k_off.expect("vk off");

    assert_eq!(j_on[0].len(), band.len());
    for (k, (a, b)) in j_on[0].iter().zip(&j_off[0]).enumerate() {
        same_bits(&format!("fftdf vj[{k}]"), a, b);
    }
    for (k, (a, b)) in k_on[0].iter().zip(&k_off[0]).enumerate() {
        same_bits(&format!("fftdf vk[{k}]"), a, b);
    }
    println!("fftdf: band-subset reuse on/off bit-identical for vj and vk");
}

/// A band list that is NOT a subset must still take the second evaluation —
/// the map refuses, the old path runs, the result is the old result.
#[test]
fn band_outside_the_sampling_list_is_still_evaluated() {
    let cell = si();
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    let mf = Krhf::new(cell.clone(), &kpts.kpts).expect("Krhf");
    let r = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-8,
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        })
        .expect("SCF");
    let grids = PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("uniform grids");
    let dms = vec![r.dm[0].clone()];
    // Shift one band point off the sampling list.
    let mut band = kpts.kpts_ibz.clone();
    band[0][0] += 1e-3;
    let ni = KNumInt::with_symmetry(&kpts);
    let out = ni
        .nr_rks(&cell, &grids, "lda,vwn", &dms, 1, Some(&band))
        .expect("nr_rks off-list band");
    assert_eq!(out.vmat[0].len(), band.len());
    assert!(out.vmat[0][0].re.iter().all(|x| x.is_finite()));
}
