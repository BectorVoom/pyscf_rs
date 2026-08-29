//! `pyscf_pbc_df::gdf` — the `GDF` class, the `_cderi` store and the
//! nuclear-attraction builder (plan 14-03).

mod common;

use pyscf_pbc_df::gdf::Gdf;
use pyscf_pbc_df::incore::Aosym;
use pyscf_pbc_df::traits::PeriodicDf;

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

fn built(cell: pyscf_pbc_gto::Cell, km: [usize; 3], aosym: Aosym) -> Gdf {
    let k = kpts(&cell, km);
    let mut d = Gdf::new(cell, &k);
    d.aosym = aosym;
    d.build().expect("GDF::build");
    d
}

/// GDF builds LAZILY, exactly as upstream does at the head of `get_j_kpts` —
/// the k-point SCF drivers hand the builder to `get_jk` and never call
/// `build()`. A GDF that required an explicit build would fail there, and did.
#[test]
fn cderi_is_built_on_first_use() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let d = Gdf::new(cell, &k);
    // No `build()` call anywhere.
    assert_eq!(d.get_naoaux().expect("lazy build"), 23);
    assert!(d.sr_loop(0, 0, true).is_ok());
}

// ---------------------------------------------------------------------------
// Task 1 — the store.
// ---------------------------------------------------------------------------

#[test]
fn get_naoaux_matches_the_auxiliary_basis() {
    let d = built(common::he_all_electron(), [1, 1, 1], Aosym::S2);
    assert_eq!(d.get_naoaux().expect("naoaux"), 23);
    assert_eq!(d.name(), "GDF");
}

/// GDF's own mesh is the COMPENSATING-CHARGE mesh, not the density one — that
/// is why it is cheap. He-fcc: `[9,9,9]` against FFTDF's `[43,43,43]`.
#[test]
fn gdf_mesh_is_the_model_charge_mesh() {
    let d = built(common::he_all_electron(), [2, 2, 2], Aosym::S2);
    assert_eq!(d.mesh(), [9, 9, 9]);
}

/// The `_cderi` file round-trips bit-identically, in upstream's layout, so a
/// file this port writes is one Phase 15/16 can read.
#[test]
fn cderi_file_round_trips() {
    use pyscf_pbc_df::gdf::CderiFile;
    let d = built(common::he_all_electron(), [1, 1, 1], Aosym::S2);
    let c = d.cderi().expect("cderi");
    let path = std::env::temp_dir().join(format!("pbcdf_cderi_{}.h5", std::process::id()));
    {
        let _f = CderiFile::save(c, &path, true).expect("save");
    }
    let back = CderiFile::load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.kpts.len(), c.kpts.len());
    assert_eq!(back.aosym, c.aosym);
    assert_eq!(back.blocks.len(), c.blocks.len());
    for (k, b) in &c.blocks {
        let g = back.blocks.get(k).expect("block survives the round trip");
        assert_eq!(g.rank, b.rank);
        assert_eq!(g.nao_pair, b.nao_pair);
        assert_eq!(g.data.re, b.data.re, "block {k} real part");
        assert_eq!(g.data.im, b.data.im, "block {k} imaginary part");
    }
}

/// `compact = false` unpacks the `s2` triangle into the full square. The
/// IMAGINARY part unpacks ANTI-hermitian (`lib.ANTIHERMI`) — invisible at
/// gamma, where it is zero, and wrong everywhere else.
///
/// Tested on a SYNTHETIC `Cderi` rather than a built one: the packing is pure
/// index arithmetic, and driving it through a real `make_j3c` on a multi-AO
/// cell would spend minutes in the lattice sum to exercise ten lines. The
/// values are deliberately complex and asymmetric so a wrong sign shows up.
#[test]
fn sr_loop_unpacks_the_triangle() {
    use pyscf_algebra::CTensor;
    use pyscf_pbc_df::gdf::sr_loop;
    use pyscf_pbc_df::gdf_builder::j3c::{Cderi, CderiBlock};

    let nao = 4usize;
    let npair = nao * (nao + 1) / 2;
    let rank = 3usize;
    let mut re = vec![0.0_f64; rank * npair];
    let mut im = vec![0.0_f64; rank * npair];
    for l in 0..rank {
        for p in 0..npair {
            re[l * npair + p] = (l * npair + p) as f64 + 1.0;
            im[l * npair + p] = -((l * npair + p) as f64) - 0.5;
        }
    }
    let mut blocks = std::collections::HashMap::new();
    blocks.insert(
        0,
        CderiBlock {
            data: CTensor { re, im },
            rank,
            nao_pair: npair,
            negative: None,
        },
    );
    let c = Cderi {
        blocks,
        kpts: vec![[0.0; 3]],
        aosym: Aosym::S2,
    };

    let packed = sr_loop(&c, 0, 0, nao, true).expect("compact");
    let square = sr_loop(&c, 0, 0, nao, false).expect("square");
    assert_eq!(packed[0].ncol, npair);
    assert_eq!(square[0].ncol, nao * nao);
    assert_eq!(packed[0].naux, rank);

    for l in 0..rank {
        for mu in 0..nao {
            for nu in 0..nao {
                let (lo, hi) = if mu >= nu { (nu, mu) } else { (mu, nu) };
                let tri = hi * (hi + 1) / 2 + lo;
                let got = square[0].re[l * nao * nao + mu * nao + nu];
                let want = packed[0].re[l * npair + tri];
                assert!(
                    (got - want).abs() < 1e-15,
                    "L={l} ({mu},{nu}): square {got} != packed {want}"
                );
                // `lib.ANTIHERMI`: the upper triangle is the NEGATED lower one.
                let gi = square[0].im[l * nao * nao + mu * nao + nu];
                let wi = packed[0].im[l * npair + tri];
                let wi = if mu >= nu { wi } else { -wi };
                assert!(
                    (gi - wi).abs() < 1e-15,
                    "L={l} ({mu},{nu}) imaginary: {gi} != {wi}"
                );
            }
        }
    }
    // And the round trip back: s1 -> s2 keeps only the lower triangle.
    let c1 = Cderi {
        blocks: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                0,
                CderiBlock {
                    data: CTensor {
                        re: square[0].re.clone(),
                        im: square[0].im.clone(),
                    },
                    rank,
                    nao_pair: nao * nao,
                    negative: None,
                },
            );
            m
        },
        kpts: vec![[0.0; 3]],
        aosym: Aosym::S1,
    };
    let back = sr_loop(&c1, 0, 0, nao, true).expect("repack");
    assert_eq!(back[0].re, packed[0].re);
    assert_eq!(back[0].im, packed[0].im);
}

/// A pair that was never built must be REFUSED, not silently zero — `j_only`
/// keeps only the diagonal.
#[test]
fn j_only_refuses_an_off_diagonal_pair() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);
    let mut d = Gdf::new(cell, &k);
    d.j_only = true;
    d.build().expect("build");
    assert!(d.sr_loop(0, 0, true).is_ok(), "the diagonal is built");
    assert!(
        d.sr_loop(0, 1, true).is_err(),
        "an off-diagonal pair must be refused under j_only"
    );
}

// ---------------------------------------------------------------------------
// Task 2 — the nuclear builder.
// ---------------------------------------------------------------------------

/// `_CCNucBuilder`'s `eta` is NOT `_CCGDFBuilder`'s: it is
/// `max(0.5/(0.5 + nkpts^(1/9)), ETA_MIN)`.
#[test]
fn nuc_eta_matches_upstream_formula() {
    use pyscf_pbc_df::gdf::nuc::nuc_eta;
    for (nk, want) in [(1usize, 0.5 / 1.5), (8, 0.5 / (0.5 + 8f64.powf(1.0 / 9.0)))] {
        let got = nuc_eta(nk);
        assert!((got - want).abs() < 1e-15, "nkpts={nk}: {got} != {want}");
    }
    // The ETA_MIN floor bites for a very large k-mesh.
    assert!(nuc_eta(1_000_000) >= pyscf_pbc_df::gdf_builder::ETA_MIN);
}

#[test]
fn get_nuc_is_hermitian_at_k_points() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);
    let d = Gdf::new(cell.clone(), &k);
    let nao = cell.mol.nao_nr;
    let mats = d.get_nuc(&k).expect("get_nuc");
    assert_eq!(mats.len(), k.len());
    for (ik, m) in mats.iter().enumerate() {
        let mut worst = 0.0_f64;
        for p in 0..nao {
            for q in 0..nao {
                let (a, b) = (p * nao + q, q * nao + p);
                worst = worst.max((m.re[a] - m.re[b]).abs());
                worst = worst.max((m.im[a] + m.im[b]).abs());
            }
        }
        assert!(worst < 1e-11, "he k={ik}: get_nuc asymmetry {worst:e}");
    }
}

/// `get_pp` works at GAMMA and is Hermitian there.
///
/// Phase 13 MEASURED the residue this gate has to clear: `ft_aopair`'s
/// screening leaves `get_pp` anti-Hermitian by **5.131e-11** at the default
/// `cell.precision = 1e-8`, falling to 4.647e-15 at 1e-12 (see the table in
/// `ft_ao/mod.rs`). Gating at 1e-11 would fail a correct implementation.
#[test]
fn get_pp_is_hermitian_at_gamma() {
    let cell = common::diamond();
    let k = kpts(&cell, [1, 1, 1]);
    let d = Gdf::new(cell.clone(), &k);
    let nao = cell.mol.nao_nr;
    let mats = d.get_pp(&k).expect("get_pp at gamma");
    let m = &mats[0];
    let mut worst = 0.0_f64;
    for p in 0..nao {
        for q in 0..nao {
            worst = worst.max((m.re[p * nao + q] - m.re[q * nao + p]).abs());
        }
    }
    assert!(worst < 1e-9, "diamond gamma: get_pp asymmetry {worst:e}");
}

/// `get_pp` at K-POINTS — the gap plan 14-03 surfaced, now CLOSED.
///
/// Phase 10's `pseudo::vloc_part2::get_pp_loc_part2` is gamma-only; its
/// k-resolved counterpart is upstream's `aft._IntPPBuilder`, which Phase 13
/// declined. It is ported in `pp_int::get_pp_loc_part2_kpts` on top of
/// `incore::aux_e2_intor` — the same double lattice sum with the
/// `int3c1e_r{2,4,6}_origk` operators — and reproduces the Phase-10 gamma route
/// to <1e-12 (`tests/pp_int.rs`). Every k-point pseudopotential path in AFTDF,
/// GDF, MDF and RSDF was blocked on this.
///
/// **Slow**: diamond's part-2 lattice sum runs to `cell.rcut = 21.3` Bohr, so
/// this is an opt-in acceptance run rather than a per-commit gate.
#[test]
#[ignore = "slow — diamond's part-2 lattice sum runs to cell.rcut"]
fn get_pp_works_at_k_points() {
    let cell = common::diamond();
    let nao = cell.mol.nao_nr;
    let k = kpts(&cell, [2, 2, 2]);
    let d = Gdf::new(cell, &k);
    let mats = d.get_pp(&k).expect("get_pp at k-points");
    assert_eq!(mats.len(), k.len());
    for (ik, m) in mats.iter().enumerate() {
        let mut worst = 0.0_f64;
        for p in 0..nao {
            for q in 0..nao {
                let (a, b) = (p * nao + q, q * nao + p);
                worst = worst.max((m.re[a] - m.re[b]).abs());
                worst = worst.max((m.im[a] + m.im[b]).abs());
            }
        }
        // Phase 13 measured `ft_aopair`'s screening leaving `get_pp`
        // anti-Hermitian by 5.131e-11 at the default `cell.precision`.
        assert!(worst < 1e-9, "diamond k={ik}: get_pp asymmetry {worst:e}");
    }
}

// ---------------------------------------------------------------------------
// Task 4 — the driver seam.
// ---------------------------------------------------------------------------

/// D-PBC-22: every k-point driver takes `Box<dyn PeriodicDf>`, so `GDF` has to
/// be usable through the trait object with no driver change.
#[test]
fn gdf_is_usable_as_a_boxed_periodic_df() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let mut df: Box<dyn PeriodicDf> = Box::new(Gdf::new(cell, &k));
    assert_eq!(df.name(), "GDF");
    df.build().expect("build through the trait object");
    assert_eq!(df.kpts().len(), 1);
    let nuc = df.get_nuc(&k).expect("get_nuc through the trait object");
    assert_eq!(nuc.len(), 1);
    // `get_jk` landed in plan 14-04; an EMPTY density list is a legitimate
    // (degenerate) request and must come back with empty matrices rather than
    // an error — the SCF drivers never send one, but `nset = 0` must not panic.
    let out = df
        .get_jk(&[], &k, pyscf_pbc_df::traits::JkOpts::hermitian())
        .expect("get_jk through the trait object");
    assert_eq!(out.vj.as_ref().expect("vj").len(), 0);
    assert_eq!(out.vk.as_ref().expect("vk").len(), 0);
}

/// The range-separated route is upstream's DEFAULT and this port does not have
/// it yet, so asking for it must be refused with the measured cost named.
#[test]
fn prefer_ccdf_false_is_refused() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let mut d = Gdf::new(cell, &k);
    d.prefer_ccdf = false;
    let e = d.build().expect_err("the RS route is plan 14-07");
    assert!(format!("{e}").contains("14-07"), "got: {e}");
}
