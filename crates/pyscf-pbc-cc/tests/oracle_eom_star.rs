//! Opt-in oracle checks for the CCSD\* (`ipccsd_star` / `eaccsd_star`)
//! corrections, spin-orbital and spin-adapted, against PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_eom_star -- --ignored --nocapture
//! ```
//!
//! # The gate contracts UPSTREAM's own eigenvectors
//!
//! A Davidson eigenvector is fixed only up to a phase, and `<L|R> = 1` is
//! enforced by scaling the LEFT vector alone — so two solvers that agree on
//! every root can still hand the correction differently-scaled amplitudes.
//! Comparing `e_star` end-to-end would absorb that. Instead the emitter runs
//! upstream's right and left solves, pairs them with
//! `_sort_left_right_eigensystem`, and emits the PAIRS; this file feeds those
//! exact vectors to `*_star_contract` and diffs the resulting `e_star`. What
//! is measured is the correction, not two eigensolves.
//!
//! # The tolerance
//!
//! `1e-6`, the same integral-transform floor at the pinned `[15,15,15]` mesh
//! that `oracle_phase16.rs` gates every EOM block at
//! (`measurements/README.md §1`). The corrections themselves are `1e-4 …
//! 4e-3`, so the gate is two to three orders below the signal.

mod common;

use common::{block, cblock, diamond_scf, emit, eris_on_upstream_mf, scalar, upstream_mos};

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::eom_kccsd_ghf::{self as eomg, KLattice, StarPair};
use pyscf_pbc_cc::eom_kccsd_rhf as eomr;
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;

/// The EOM-block gate — see the module doc.
const STAR_GATE: f64 = 1e-6;

/// The k-mesh and lattice the emitter used, as `get_kconserv3` wants them.
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

/// The emitted `(eval, right, left)` triples for one `kshift`.
fn pairs_at(
    out: &str,
    prefix: &str,
    tag: &str,
    kshift: usize,
    size: usize,
) -> (Vec<f64>, Vec<ZArr>, Vec<ZArr>) {
    let n = scalar(out, &format!("{prefix}_{tag}_npair_{kshift}")) as usize;
    let evals = block(out, &format!("{prefix}_{tag}_evals_{kshift}"));
    assert_eq!(evals.len(), n, "{prefix}_{tag} evals at kshift {kshift}");
    let unpack = |name: &str| -> Vec<ZArr> {
        let c = cblock(out, name);
        assert_eq!(c.re.len(), n * size, "{name}: {} vs {n}·{size}", c.re.len());
        (0..n)
            .map(|i| {
                let o = i * size;
                ZArr::from_ctensor(
                    &[size],
                    CTensor {
                        re: c.re[o..o + size].to_vec(),
                        im: c.im[o..o + size].to_vec(),
                    },
                )
                .expect("evec shape")
            })
            .collect()
    };
    let r = unpack(&format!("{prefix}_{tag}_revecs_{kshift}"));
    let l = unpack(&format!("{prefix}_{tag}_levecs_{kshift}"));
    (evals, r, l)
}

/// Report the worst absolute `e_star` difference over one `kshift`.
fn diff_estar(got: &[pyscf_pbc_cc::eom_kccsd_ghf::StarRoot], want: &[f64], name: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: {} roots vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0_f64;
    for (g, w) in got.iter().zip(want) {
        println!(
            "  {name}: e_star {:.12} vs {:.12}  (delta_e {:.3e}, <L|R> {:.3e}+{:.3e}i, Im(deltaE) {:.3e})",
            g.e_star, w, g.delta_e, g.ldotr.0, g.ldotr.1, g.delta_e_imag
        );
        worst = worst.max((g.e_star - w).abs());
    }
    worst
}

/// **The spin-adapted IP/EA-CCSD\* corrections, root by root.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn rhf_ccsd_star_matches_upstream() {
    let Some(out) = emit("star_rhf") else { return };
    let f = diamond_scf([1, 1, 2]);
    let up = upstream_mos(&out);
    let (eris, khelper) = eris_on_upstream_mf(&f, &up);
    let kc = &khelper.kconserv;
    let (nkpts, nocc, nvir) = (up.nkpts, up.nocc, up.nmo - up.nocc);
    let (lat_a, kpts) = lattice(&out);
    let lat = KLattice {
        a: &lat_a,
        kpts: &kpts,
    };
    let padding = eomg::padding_from(&up.padded).expect("padding");

    // UPSTREAM's converged amplitudes: the correction is a property of the
    // CCSD solution, and two CCSD convergences would be a second measurement.
    let t1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "t1")).expect("t1");
    let t2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "t2"),
    )
    .expect("t2");

    let mut failures: Vec<String> = Vec::new();
    for (tag, is_ip) in [("ip", true), ("ea", false)] {
        let imds = eomr::RhfEomImds::make_shared(&t1, &t2, &eris, kc).expect("shared");
        let imds = if is_ip {
            imds.make_ip(kc).expect("IP imds")
        } else {
            imds.make_ea(kc).expect("EA imds")
        };
        let size = scalar(&out, &format!("rhf_{tag}_vector_size")) as usize;
        assert_eq!(
            size,
            if is_ip {
                eomr::ip_vector_size(nkpts, nocc, nvir)
            } else {
                eomr::ea_vector_size(nkpts, nocc, nvir)
            }
        );
        for kshift in 0..nkpts {
            let (evals, rv, lv) = pairs_at(&out, "rhf", tag, kshift, size);
            let pairs: Vec<StarPair<'_>> = (0..evals.len())
                .map(|i| StarPair {
                    eval: evals[i],
                    r: &rv[i],
                    l: &lv[i],
                })
                .collect();
            let got = if is_ip {
                eomr::ipccsd_star_contract(&pairs, kshift, &imds, &padding, kc, &lat)
            } else {
                eomr::eaccsd_star_contract(&pairs, kshift, &imds, &padding, kc, &lat)
            }
            .expect("star contract");
            let want = block(&out, &format!("rhf_{tag}_estar_{kshift}"));
            let name = format!("rhf_{tag}_estar_{kshift}");
            let d = diff_estar(&got, &want, &name);
            println!("  {name} worst |Δ| {d:e}");
            if !(d < STAR_GATE) {
                failures.push(format!("{name} {d:e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "RHF CCSD* above the gate: {failures:?}"
    );
}

/// **The spin-orbital IP/EA-CCSD\* corrections, root by root.**
///
/// The spin-orbital form is a different equation, not the spin-adapted one
/// with `nocc` doubled: its `P(ijk)` has three unit-weight terms where the
/// spin-adapted one has six weighted `4,1,1,−2,−2,−2`, it permutes BOTH sides
/// where the spin-adapted one permutes only the left, and its prefactor is
/// `1/12` against `1/2`.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ghf_ccsd_star_matches_upstream() {
    let Some(out) = emit("star_ghf") else { return };
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

    let mut failures: Vec<String> = Vec::new();
    for (tag, is_ip) in [("ip", true), ("ea", false)] {
        let imds = eomg::EomImds::make_shared(&t1, &t2, &eris, kc).expect("shared");
        let imds = if is_ip {
            imds.make_ip(kc).expect("IP imds")
        } else {
            imds.make_ea(kc).expect("EA imds")
        };
        let size = scalar(&out, &format!("ghf_{tag}_vector_size")) as usize;
        for kshift in 0..nkpts {
            let (evals, rv, lv) = pairs_at(&out, "ghf", tag, kshift, size);
            let pairs: Vec<StarPair<'_>> = (0..evals.len())
                .map(|i| StarPair {
                    eval: evals[i],
                    r: &rv[i],
                    l: &lv[i],
                })
                .collect();
            let got = if is_ip {
                eomg::ipccsd_star_contract(&pairs, kshift, &imds, &padding, kc, &lat)
            } else {
                eomg::eaccsd_star_contract(&pairs, kshift, &imds, &padding, kc, &lat)
            }
            .expect("star contract");
            let want = block(&out, &format!("ghf_{tag}_estar_{kshift}"));
            let name = format!("ghf_{tag}_estar_{kshift}");
            let d = diff_estar(&got, &want, &name);
            println!("  {name} worst |Δ| {d:e}");
            if !(d < STAR_GATE) {
                failures.push(format!("{name} {d:e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "GHF CCSD* above the gate: {failures:?}"
    );
}

/// `_sort_left_right_eigensystem` drops a right root with no converged left
/// partner, and never claims one left root twice.
#[test]
fn sorting_left_and_right_drops_the_unpaired() {
    use pyscf_pbc_cc::eom_kccsd_ghf::{EomRoots, sort_left_right_eigensystem};
    let v = |n: usize| (0..n).map(|_| ZArr::zeros(&[1])).collect::<Vec<_>>();
    let right = EomRoots {
        kshift: 0,
        conv: vec![true, true, false],
        e: vec![1.0, 2.0, 3.0],
        v: v(3),
        qp_weight: vec![1.0; 3],
    };
    let left = EomRoots {
        kshift: 0,
        // 2.0 appears twice; only the first may be claimed. 1.0 never
        // converged, so the right root at 1.0 has no partner.
        conv: vec![false, true, true],
        e: vec![1.0, 2.0, 2.0],
        v: v(3),
        qp_weight: vec![1.0; 3],
    };
    let pairs = sort_left_right_eigensystem(&right, &left, 1e-6);
    assert_eq!(pairs.len(), 1, "only the 2.0 root pairs");
    assert!((pairs[0].eval - 2.0).abs() < 1e-12);
}
