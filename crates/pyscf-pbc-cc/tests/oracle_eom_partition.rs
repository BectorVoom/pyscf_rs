//! Opt-in oracle checks for the `eom.partition` branches of all three EOM
//! modules, against the vendored PySCF 2.12.1 tree.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_eom_partition -- --ignored --nocapture
//! ```
//!
//! # What is gated here, and why it needs an oracle at all
//!
//! No caller reaches `eom.partition` through upstream's public API: `ipccsd`
//! and `eaccsd` (`eom_kccsd_ghf.py:618`, `:905`) raise `NotImplementedError`
//! for both `'mp'` and `'full'`, and all three modules inherit those drivers.
//! The branches behind that refusal still exist, and this file gates the ones
//! that compute something against upstream's own arithmetic — the emitter
//! reaches them exactly the way a user would have to, by setting
//! `eom.partition` and calling the matvec or the diagonal directly.
//!
//! Four of the six branches read `eom.eris.fock`, and `eom.eris` does not
//! exist on an EOM object. The emitter records the `AttributeError` and then
//! SUPPLIES `eom.eris` so upstream's own code runs; `imds.eris` is the same
//! `_ERIS`, which is what this port reads. See
//! [`pyscf_pbc_cc::eom_kccsd_ghf::Partition`].
//!
//! The tolerance is `IMDS_BLOCK` (`1e-6`), the same integral-transform floor
//! `oracle_phase16.rs` gates every other EOM block at — measured in
//! `measurements/README.md §1`, not invented here.

mod common;

use common::{
    block, cblock, diamond_scf, emit, eris_on_upstream_mf, maxdiff, scalar, synthetic, upstream_mos,
};

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::eom_kccsd_ghf::{self as eomg, Partition};
use pyscf_pbc_cc::eom_kccsd_rhf as eomr;
use pyscf_pbc_cc::eom_kccsd_uhf as eomu;
use pyscf_pbc_cc::{PbcCcError, ZArr};
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;

/// The EOM-block gate — `oracle_phase16.rs`'s `IMDS_BLOCK`, same reason.
const IMDS_BLOCK: f64 = 1e-6;

/// Assert an oracle flag is `1.0`, i.e. upstream raised what this port claims.
fn refused(out: &str, key: &str) {
    let v = scalar(out, key);
    assert!(
        (v - 1.0).abs() < 1e-12,
        "{key} = {v}: upstream did NOT behave the way this port's refusal claims"
    );
}

/// **The twelve driver refusals, and the two UHF diagonals' own refusal.**
///
/// This is the load-bearing test for every `NotImplementedUpstream` this
/// crate returns for a partition: if upstream ever starts implementing one,
/// this fails and the refusal stops being honest.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn upstream_refuses_every_partition_the_drivers_are_given() {
    let Some(out) = emit("partition") else { return };

    for module in ["rhf", "ghf", "uhf"] {
        for kind in ["ip", "ea"] {
            for part in ["mp", "full"] {
                refused(&out, &format!("{module}_{kind}_{part}_refused"));
            }
        }
    }
    println!("all 12 driver entry points raise NotImplementedError");

    // The UHF diagonals refuse a second time, on their own.
    refused(&out, "uhf_ip_diag_mp_raises");
    refused(&out, "uhf_ea_diag_mp_raises");
    println!("both UHF diagonals raise Exception(\"MP diag is not tested\")");

    // The four branches that need an attribute upstream never sets.
    for key in [
        "rhf_ip_diag_mp_needs_eom_eris",
        "rhf_ea_matvec_mp_needs_eom_eris",
        "ghf_ip_diag_mp_needs_eom_eris",
        "ghf_ea_diag_mp_needs_eom_eris",
    ] {
        refused(&out, key);
    }
    println!("four 'mp' branches raise AttributeError on eom.eris as shipped");

    // `'full'` computes nothing: both RHF matvecs raise before any arithmetic.
    refused(&out, "rhf_ip_matvec_full_typeerror");
    refused(&out, "rhf_ea_matvec_full_typeerror");

    // And this port refuses in the same places, with a payload that names
    // the upstream line the refusal is inherited from.
    for (label, err) in [
        ("ghf/rhf", eomg::partition_refusal()),
        ("uhf", eomu::partition_refusal()),
    ] {
        match err {
            PbcCcError::NotImplementedUpstream { upstream, what } => {
                println!("{label} refuses at {upstream}: {what}");
                assert!(upstream.starts_with("pbc/cc/eom_kccsd_"), "{upstream}");
            }
            other => panic!("{label}: expected NotImplementedUpstream, got {other:?}"),
        }
    }
    // `'full'` is refused a second time, inside the branches themselves.
    let err = Partition::refuse_full(Partition::Full).expect_err("'full' must refuse");
    assert!(matches!(err, PbcCcError::NotImplementedUpstream { .. }));
    Partition::refuse_full(Partition::Mp).expect("'mp' is not refused there");
    Partition::refuse_full(Partition::None).expect("None is not refused there");
}

/// **The RHF `'mp'` matvecs and diagonals, element-wise against upstream.**
///
/// Both matvecs and both diagonals, at every `kshift`, on the same fixed
/// synthetic amplitudes `krccsd_eom` uses — so this measures the equations,
/// not a converged answer.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn rhf_partition_mp_matches_upstream() {
    let Some(out) = emit("partition") else { return };
    let f = diamond_scf([1, 1, 2]);
    let up = upstream_mos(&out);
    let (eris, khelper) = eris_on_upstream_mf(&f, &up);
    let kc = &khelper.kconserv;
    let (nkpts, nocc, nvir) = (up.nkpts, up.nocc, up.nmo - up.nocc);

    let (st1, st2) = synthetic(
        &[nkpts, nocc, nvir],
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
    );

    // `make_ip('mp')` skips `Woooo` and the whole shared-2e set; `make_ea('mp')`
    // keeps `Wvvvv` here because `t1` is not zero (`:1618`). Both are asserted
    // against the flags upstream emitted rather than assumed.
    let imds_ip = eomr::RhfEomImds::make_shared_1e(&st1, &st2, &eris, kc)
        .expect("shared 1e")
        .make_ip_partition(kc, Partition::Mp)
        .expect("IP mp imds");
    assert!(
        imds_ip.woooo.is_none(),
        "make_ip('mp') must not build Woooo"
    );
    assert_eq!(
        scalar(&out, "rhf_make_ip_mp_has_woooo"),
        0.0,
        "upstream's make_ip('mp') built Woooo after all"
    );

    let imds_ea = eomr::RhfEomImds::make_shared_1e(&st1, &st2, &eris, kc)
        .expect("shared 1e")
        .make_ea_partition(kc, Partition::Mp)
        .expect("EA mp imds");
    assert!(
        imds_ea.wvvvv.is_some(),
        "make_ea('mp') keeps Wvvvv when t1 is nonzero"
    );
    assert_eq!(
        scalar(&out, "rhf_make_ea_mp_t1nonzero_has_wvvvv"),
        1.0,
        "upstream kept Wvvvv for a nonzero t1"
    );
    // ... and drops it when `t1` is identically zero, which is the OTHER half
    // of `:1618`'s `and`.
    let zero_t1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    let imds_ea0 = eomr::RhfEomImds::make_shared_1e(&zero_t1, &st2, &eris, kc)
        .expect("shared 1e")
        .make_ea_partition(kc, Partition::Mp)
        .expect("EA mp imds, zero t1");
    assert!(
        imds_ea0.wvvvv.is_none(),
        "make_ea('mp') with t1 == 0 must not build Wvvvv"
    );
    assert_eq!(
        scalar(&out, "rhf_make_ea_mp_t1zero_has_wvvvv"),
        0.0,
        "upstream dropped Wvvvv for a zero t1"
    );

    let n_ip = scalar(&out, "rhf_ip_vector_size") as usize;
    let n_ea = scalar(&out, "rhf_ea_vector_size") as usize;
    assert_eq!(n_ip, eomr::ip_vector_size(nkpts, nocc, nvir));
    assert_eq!(n_ea, eomr::ea_vector_size(nkpts, nocc, nvir));
    let v_ip = ZArr::from_ctensor(&[n_ip], cblock(&out, "rhf_ip_vec")).expect("ip vec");
    let v_ea = ZArr::from_ctensor(&[n_ea], cblock(&out, "rhf_ea_vec")).expect("ea vec");

    let mut failures: Vec<String> = Vec::new();
    for kshift in 0..nkpts {
        for (got, name) in [
            (
                eomr::ipccsd_matvec_partition(&v_ip, kshift, &imds_ip, kc, Partition::Mp)
                    .expect("ip mp matvec"),
                format!("rhf_ip_matvec_mp_{kshift}"),
            ),
            (
                eomr::ipccsd_diag_partition(kshift, &imds_ip, kc, Partition::Mp)
                    .expect("ip mp diag"),
                format!("rhf_ip_diag_mp_{kshift}"),
            ),
            (
                eomr::eaccsd_matvec_partition(&v_ea, kshift, &imds_ea, kc, Partition::Mp)
                    .expect("ea mp matvec"),
                format!("rhf_ea_matvec_mp_{kshift}"),
            ),
            (
                eomr::eaccsd_diag_partition(kshift, &imds_ea, kc, Partition::Mp)
                    .expect("ea mp diag"),
                format!("rhf_ea_diag_mp_{kshift}"),
            ),
        ] {
            let d = maxdiff(&got, &cblock(&out, &name), &name);
            println!("  {name:26} max|Δ| {d:e}");
            if !(d < IMDS_BLOCK) {
                failures.push(format!("{name} {d:e}"));
            }
        }
    }
    assert!(failures.is_empty(), "RHF 'mp' above the gate: {failures:?}");
}

/// **The two spin-orbital `'mp'` diagonals, element-wise against upstream.**
///
/// The spin-orbital matvecs have no partition branch at all, so the diagonals
/// are the whole of `eom_kccsd_ghf`'s `'mp'` surface. Note `eaccsd_diag`'s
/// `'mp'` branch carries a sign upstream's own `None` branch does not — the
/// gate is what settles that it was transcribed and not "corrected".
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ghf_partition_mp_diagonals_match_upstream() {
    let Some(out) = emit("partition") else { return };
    let f = diamond_scf([1, 1, 2]);

    let nkpts = scalar(&out, "nkpts") as usize;
    let nocc = scalar(&out, "ghf_nocc") as usize;
    let nmo = scalar(&out, "ghf_nmo") as usize;
    let nao = scalar(&out, "ghf_nao") as usize;
    let nvir = nmo - nocc;

    let c = cblock(&out, "ghf_mo_coeff");
    let mo_coeff: Vec<MoCoeff> = (0..nkpts)
        .map(|k| {
            let off = k * nao * nmo;
            MoCoeff::new(
                nao,
                nmo,
                CTensor {
                    re: c.re[off..off + nao * nmo].to_vec(),
                    im: c.im[off..off + nao * nmo].to_vec(),
                },
            )
        })
        .collect();
    let me = block(&out, "ghf_mo_energy");
    let mo_energy: Vec<Vec<f64>> = (0..nkpts)
        .map(|k| me[k * nmo..(k + 1) * nmo].to_vec())
        .collect();
    let fock = ZArr::from_ctensor(&[nkpts, nmo, nmo], cblock(&out, "ghf_fock")).expect("fock");
    let khelper = KptsHelper::without_symm_map(&f.cell.a, PeriodicDf::kpts(&f.df));
    let eris = pyscf_pbc_cc::kccsd::KgEris::from_parts(
        &f.df,
        &khelper,
        &mo_coeff,
        fock,
        mo_energy,
        nocc,
        4_000_000_000,
    )
    .expect("spin-orbital _ERIS");
    let kc = &khelper.kconserv;

    let (st1, st2) = synthetic(
        &[nkpts, nocc, nvir],
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
    );
    let imds_ip = eomg::EomImds::make_shared(&st1, &st2, &eris, kc)
        .expect("shared")
        .make_ip(kc)
        .expect("IP imds");
    let imds_ea = eomg::EomImds::make_shared(&st1, &st2, &eris, kc)
        .expect("shared")
        .make_ea(kc)
        .expect("EA imds");

    let mut failures: Vec<String> = Vec::new();
    for kshift in 0..nkpts {
        for (got, name) in [
            (
                eomg::ipccsd_diag_mp(kshift, &imds_ip, kc).expect("ghf ip mp diag"),
                format!("ghf_ip_diag_mp_{kshift}"),
            ),
            (
                eomg::eaccsd_diag_mp(kshift, &imds_ea, kc).expect("ghf ea mp diag"),
                format!("ghf_ea_diag_mp_{kshift}"),
            ),
        ] {
            let d = maxdiff(&got, &cblock(&out, &name), &name);
            println!("  {name:26} max|Δ| {d:e}");
            if !(d < IMDS_BLOCK) {
                failures.push(format!("{name} {d:e}"));
            }
        }
    }
    assert!(failures.is_empty(), "GHF 'mp' above the gate: {failures:?}");
}

/// `eom_kccsd_uhf` ships no `'mp'` branch, and says where.
#[test]
fn uhf_refuses_partition_and_says_where() {
    // A pure refusal test: it never touches Python, so it is not `#[ignore]`d.
    match eomu::partition_refusal() {
        PbcCcError::NotImplementedUpstream { upstream, what } => {
            assert!(upstream.contains("eom_kccsd_uhf.py"), "{upstream}");
            assert!(what.contains("MP diag is not tested"), "{what}");
        }
        other => panic!("expected a NotImplementedUpstream, got {other:?}"),
    }
}
