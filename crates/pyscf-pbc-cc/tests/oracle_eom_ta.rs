//! Opt-in oracle checks for the `T3[2]` intermediates and the `EOMIP_Ta` /
//! `EOMEA_Ta` variants, against PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_eom_ta -- --ignored --nocapture
//! ```
//!
//! # What `(T)a` is
//!
//! `EOMIP_Ta` differs from `EOMIP` in one thing: its `make_imds` runs
//! `get_t3p2_imds*` first, replaces `t1`/`t2` with the perturbatively
//! corrected `pt1`/`pt2`, rebuilds EVERY intermediate from those, and only
//! then adds the `Wmcik` / `Wacek` term. So the gate has three layers, and
//! this file checks all three: the `T3[2]` output itself, the rebuilt
//! `Wovoo` / `Wvvvo`, and the roots that stand on them.
//!
//! # The spin-adapted floor is upstream's own
//!
//! `kintermediates_rhf` has TWO implementations — the loop-explicit
//! `get_t3p2_imds_slow` and the blocked `get_t3p2_imds` that `_IMDS` calls —
//! and they do not agree to machine precision. The emitter measures the gap
//! and this file prints it next to its own, so the reader can see which floor
//! is whose.

mod common;

use common::{
    block, cblock, diamond_scf, emit, eris_on_upstream_mf, maxdiff, scalar, upstream_mos,
};

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::eom_kccsd_ghf::{self as eomg, EomOpts, Excitation, KLattice};
use pyscf_pbc_cc::eom_kccsd_rhf as eomr;
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;

/// The EOM-block gate — `oracle_phase16.rs`'s `IMDS_BLOCK`, the measured
/// integral-transform floor at the pinned `[15,15,15]` mesh.
const IMDS_BLOCK: f64 = 1e-6;

/// The ROOT gate, `1e-5`, measured in `measurements/README.md §1`: upstream's
/// own spread over `conv_tol` and `nroots` on these roots reaches `5.1e-7`,
/// and its own test suite asserts EOM roots at 3 decimals.
const ROOT_GATE: f64 = 1e-5;

fn lattice(out: &str) -> ([[f64; 3]; 3], Vec<[f64; 3]>) {
    let a = block(out, "lattice");
    let mut lat = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            lat[i][j] = a[i * 3 + j];
        }
    }
    let k = block(out, "kpts");
    let kpts: Vec<[f64; 3]> = (0..k.len() / 3)
        .map(|i| [k[3 * i], k[3 * i + 1], k[3 * i + 2]])
        .collect();
    (lat, kpts)
}

fn eom_opts(nroots: usize) -> EomOpts {
    EomOpts {
        conv_tol: 1e-8,
        max_cycle: 100,
        nroots,
        ..Default::default()
    }
}

/// **The spin-orbital `T3[2]` and the `(T)a` roots.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ghf_t3p2_and_ta_roots_match_upstream() {
    let Some(out) = emit("t3p2_ghf") else { return };
    let f = diamond_scf([1, 1, 2]);
    let nkpts = scalar(&out, "nkpts") as usize;
    let nocc = scalar(&out, "nocc") as usize;
    let nmo = scalar(&out, "nmo") as usize;
    let nao = scalar(&out, "nao") as usize;
    let nvir = nmo - nocc;

    let c = cblock(&out, "mo_coeff");
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
    let me = block(&out, "mo_energy");
    let mo_energy: Vec<Vec<f64>> = (0..nkpts)
        .map(|k| me[k * nmo..(k + 1) * nmo].to_vec())
        .collect();
    let fock = ZArr::from_ctensor(&[nkpts, nmo, nmo], cblock(&out, "fock")).expect("fock");
    let nocc_per_kpt: Vec<usize> = block(&out, "nocc_per_kpt")
        .iter()
        .map(|v| *v as usize)
        .collect();
    let padded = PaddedMos {
        mo_coeff: mo_coeff.clone(),
        mo_energy: mo_energy.clone(),
        nmo_per_kpt: vec![nmo; nkpts],
        nocc_per_kpt,
        nmo,
        nocc,
    };
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
    let padding = eomg::padding_from(&padded).expect("padding");
    let (lat_a, kpts) = lattice(&out);
    let lat = KLattice {
        a: &lat_a,
        kpts: &kpts,
    };

    let t1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "t1")).expect("t1");
    let t2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "t2"),
    )
    .expect("t2");

    // --- layer 1: `get_t3p2_imds_slow` itself.
    let p = pyscf_pbc_cc::kintermediates::get_t3p2_imds_slow(&t1, &t2, &eris, kc, &padding, &lat)
        .expect("T3[2]");
    let want_delta = scalar(&out, "delta_ccsd_energy");
    println!(
        "  delta_ccsd_energy {:.15e} vs {:.15e} -> {:e}",
        p.delta_ccsd_energy,
        want_delta,
        (p.delta_ccsd_energy - want_delta).abs()
    );
    let mut failures: Vec<String> = Vec::new();
    if !((p.delta_ccsd_energy - want_delta).abs() < IMDS_BLOCK) {
        failures.push("delta_ccsd_energy".into());
    }
    for (got, name) in [
        (&p.pt1, "pt1"),
        (&p.pt2, "pt2"),
        (&p.wovoo, "Wmcik"),
        (&p.wvvvo, "Wacek"),
    ] {
        let d = maxdiff(got, &cblock(&out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // --- layer 2: the REBUILT Wovoo / Wvvvo the `_Ta` classes carry.
    let (imds_ip, d_ip) =
        eomg::EomImds::make_t3p2_ip(&t1, &t2, &eris, kc, &padding, &lat).expect("Ta IP imds");
    let (imds_ea, d_ea) =
        eomg::EomImds::make_t3p2_ea(&t1, &t2, &eris, kc, &padding, &lat).expect("Ta EA imds");
    assert!((d_ip - d_ea).abs() < 1e-14, "one T3[2], two deltas");
    for (got, name) in [
        (imds_ip.wovoo.as_ref().expect("Wovoo"), "Ta_Wovoo"),
        (imds_ea.wvvvo.as_ref().expect("Wvvvo"), "Ta_Wvvvo"),
    ] {
        let d = maxdiff(got, &cblock(&out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // --- layer 3: the roots.
    let nroots = scalar(&out, "nroots") as usize;
    for (kind, imds, tag) in [
        (Excitation::Ip, &imds_ip, "ip"),
        (Excitation::Ea, &imds_ea, "ea"),
    ] {
        for kshift in 0..nkpts {
            let roots = eomg::eom_kernel(kind, kshift, imds, &padding, kc, &eom_opts(nroots))
                .expect("Ta roots");
            let want = block(&out, &format!("Ta_{tag}_roots_{kshift}"));
            let conv = block(&out, &format!("Ta_{tag}_conv_{kshift}"));
            for (n, w) in want.iter().enumerate() {
                if conv[n] == 0.0 {
                    println!("  Ta_{tag}_roots_{kshift}[{n}]: upstream did not converge, skipped");
                    continue;
                }
                let d = (roots.e[n] - w).abs();
                println!(
                    "  Ta_{tag}_roots_{kshift}[{n}] {:.10} vs {w:.10} -> {d:e}",
                    roots.e[n]
                );
                if !(d < ROOT_GATE) {
                    failures.push(format!("Ta_{tag}_roots_{kshift}[{n}] {d:e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "GHF (T)a above the gate: {failures:?}");
}

/// **The spin-adapted `T3[2]` and the `(T)a` roots.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn rhf_t3p2_and_ta_roots_match_upstream() {
    let Some(out) = emit("t3p2_rhf") else { return };
    let f = diamond_scf([1, 1, 2]);
    let up = upstream_mos(&out);
    let (eris, khelper) = eris_on_upstream_mf(&f, &up);
    let kc = &khelper.kconserv;
    let (nkpts, nocc, nvir) = (up.nkpts, up.nocc, up.nmo - up.nocc);
    let padding = eomg::padding_from(&up.padded).expect("padding");
    let (lat_a, kpts) = lattice(&out);
    let lat = KLattice {
        a: &lat_a,
        kpts: &kpts,
    };

    let t1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "t1")).expect("t1");
    let t2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "t2"),
    )
    .expect("t2");

    // Upstream's own two implementations, for scale.
    println!("UPSTREAM slow-vs-fast (reported, not gated):");
    for name in ["pt1", "pt2", "Wmcik", "Wacek"] {
        println!(
            "  {name:8} {:e}",
            scalar(&out, &format!("upstream_{name}_slow_vs_fast"))
        );
    }

    let p =
        pyscf_pbc_cc::kintermediates_rhf::get_t3p2_imds_slow(&t1, &t2, &eris, kc, &padding, &lat)
            .expect("T3[2]");
    let want_delta = scalar(&out, "delta_ccsd_energy");
    println!(
        "  delta_ccsd_energy {:.15e} vs {:.15e} (upstream fast {:.15e})",
        p.delta_ccsd_energy,
        want_delta,
        scalar(&out, "delta_ccsd_energy_fast")
    );
    let mut failures: Vec<String> = Vec::new();
    if !((p.delta_ccsd_energy - want_delta).abs() < IMDS_BLOCK) {
        failures.push("delta_ccsd_energy".into());
    }
    for (got, name) in [
        (&p.pt1, "pt1"),
        (&p.pt2, "pt2"),
        (&p.wovoo, "Wmcik"),
        (&p.wvvvo, "Wacek"),
    ] {
        let d = maxdiff(got, &cblock(&out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // The `_Ta` intermediates. Upstream builds these from its FAST `T3[2]`,
    // so the residual here carries the slow-vs-fast gap printed above on top
    // of the integral floor.
    let (imds_ip, d_ip) =
        eomr::RhfEomImds::make_t3p2_ip(&t1, &t2, &eris, kc, &padding, &lat).expect("Ta IP imds");
    let (imds_ea, _) =
        eomr::RhfEomImds::make_t3p2_ea(&t1, &t2, &eris, kc, &padding, &lat).expect("Ta EA imds");
    println!("  delta from make_t3p2_ip {d_ip:.15e}");
    for (got, name) in [
        (imds_ip.wovoo.as_ref().expect("Wovoo"), "Ta_Wovoo"),
        (imds_ea.wvvvo.as_ref().expect("Wvvvo"), "Ta_Wvvvo"),
    ] {
        let d = maxdiff(got, &cblock(&out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // `make_t3p2_ip_ea` must produce exactly what the two separate calls do.
    let (imds_both, d_both) =
        eomr::RhfEomImds::make_t3p2_ip_ea(&t1, &t2, &eris, kc, &padding, &lat)
            .expect("Ta IP+EA imds");
    assert!((d_both - d_ip).abs() < 1e-14);
    let d = maxdiff(
        imds_both.wovoo.as_ref().expect("Wovoo"),
        &imds_ip.wovoo.as_ref().expect("Wovoo").data().clone(),
        "make_t3p2_ip_ea Wovoo",
    );
    assert!(d == 0.0, "make_t3p2_ip_ea's Wovoo differs by {d:e}");

    let nroots = scalar(&out, "nroots") as usize;
    for (kind, imds, tag) in [
        (Excitation::Ip, &imds_ip, "ip"),
        (Excitation::Ea, &imds_ea, "ea"),
    ] {
        for kshift in 0..nkpts {
            let roots = eomr::eom_kernel(kind, kshift, imds, &padding, kc, &eom_opts(nroots))
                .expect("Ta roots");
            let want = block(&out, &format!("Ta_{tag}_roots_{kshift}"));
            let conv = block(&out, &format!("Ta_{tag}_conv_{kshift}"));
            for (n, w) in want.iter().enumerate() {
                if conv[n] == 0.0 {
                    println!("  Ta_{tag}_roots_{kshift}[{n}]: upstream did not converge, skipped");
                    continue;
                }
                let d = (roots.e[n] - w).abs();
                println!(
                    "  Ta_{tag}_roots_{kshift}[{n}] {:.10} vs {w:.10} -> {d:e}",
                    roots.e[n]
                );
                if !(d < ROOT_GATE) {
                    failures.push(format!("Ta_{tag}_roots_{kshift}[{n}] {d:e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "RHF (T)a above the gate: {failures:?}");
}
