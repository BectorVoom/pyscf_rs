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

/// GDF's own mesh is TINY next to the density one — that is why it is cheap —
/// and **which** small mesh it is depends on the route.
///
/// He-fcc 2x2x2, against FFTDF's `[43,43,43]`:
///
/// | route | mesh | estimator |
/// |---|---|---|
/// | RS (the default since Task 7d) | `[11,11,11]` | `_guess_omega`, carrying the long-range half |
/// | CC | `[9,9,9]` | `_guess_eta`, resolving the model charge |
///
/// This test asserted only the `[9,9,9]` before plan 14-07 Task 7d flipped the
/// default; both are pinned now, because "the mesh is small" is true of either
/// and would not have caught the change of route.
#[test]
fn gdf_mesh_is_the_routes_own_small_mesh() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);

    let rs = Gdf::new(cell.clone(), &k);
    assert!(!rs.prefer_ccdf, "Task 7d: the RS route is the default");
    assert_eq!(rs.mesh(), [11, 11, 11], "RS: _guess_omega's mesh");

    let mut cc = Gdf::new(cell, &k);
    cc.prefer_ccdf = true;
    assert_eq!(cc.mesh(), [9, 9, 9], "CC: _guess_eta's model-charge mesh");
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

/// **An `s2` block holds HALF of an off-diagonal k-pair, and the other half
/// lives in the CONJUGATE pair's block.**
///
/// `(L | mu^{ki} nu^{kj})` is Hermitian in `(mu, nu)` only at `ki == kj`. Away
/// from the diagonal, upstream assembles the square from TWO stored triangles —
/// `_KPair3CLoader.__getitem__` (`df.py:990-1009`) hands `PBCunpack_tril_triu`
/// (`pyscf/lib/pbc/fill_ints.c:1460-1483`) `tril` from `(ki, kj)` and `triu`
/// from `(kj, ki)`:
///
/// ```text
/// out[mu, nu] = tril[mu, nu]           mu >= nu
/// out[nu, mu] = conj(triu[mu, nu])     mu >  nu
/// ```
///
/// Reconstructing the upper triangle from `lib.ANTIHERMI` on the SAME block
/// instead — correct at gamma, where the block is Hermitian — silently
/// substitutes the wrong pair's integrals off the diagonal, and no test on a
/// one-AO cell can see it: `nao = 1` makes `s2` and `s1` the same array.
///
/// Synthetic, for `sr_loop_unpacks_the_triangle`'s reason: the assembly is pure
/// index arithmetic, and the two blocks are given DELIBERATELY UNRELATED values
/// so that taking either from the wrong one is a visible failure.
#[test]
fn sr_loop_takes_the_upper_triangle_from_the_conjugate_pair() {
    use pyscf_algebra::CTensor;
    use pyscf_pbc_df::gdf::sr_loop;
    use pyscf_pbc_df::gdf_builder::j3c::{Cderi, CderiBlock};

    let nao = 3usize;
    let npair = nao * (nao + 1) / 2;
    let rank = 2usize;
    let tri = |mu: usize, nu: usize| {
        let (hi, lo) = if mu >= nu { (mu, nu) } else { (nu, mu) };
        hi * (hi + 1) / 2 + lo
    };

    // The tensor the store is HALVES OF: `L(0,1)[L, mu, nu]`, deliberately
    // NEITHER symmetric nor Hermitian in `(mu, nu)`, so every wrong way of
    // filling the upper triangle produces a different array.
    let m = |l: usize, mu: usize, nu: usize| {
        let i = (l * nao + mu) * nao + nu;
        (1.0 + i as f64, 0.25 - 0.5 * i as f64 + 3.0 * mu as f64)
    };
    // What a `s2` store actually holds: the lower triangle of each pair, with
    // `L(1,0)[mu, nu] = conj(L(0,1)[nu, mu])` (`fill_ints.c:1460-1483`).
    let mut b01 = CderiBlock {
        data: CTensor {
            re: vec![0.0; rank * npair],
            im: vec![0.0; rank * npair],
        },
        rank,
        nao_pair: npair,
        negative: None,
    };
    let mut b10 = CderiBlock { ..b01.clone() };
    for l in 0..rank {
        for mu in 0..nao {
            for nu in 0..=mu {
                let t = l * npair + tri(mu, nu);
                let (re, im) = m(l, mu, nu);
                b01.data.re[t] = re;
                b01.data.im[t] = im;
                // conj(L(0,1)[nu, mu]) -- the TRANSPOSED element.
                let (re, im) = m(l, nu, mu);
                b10.data.re[t] = re;
                b10.data.im[t] = -im;
            }
        }
    }

    let mut blocks = std::collections::HashMap::new();
    blocks.insert(1, b01.clone()); // ki*nkpts + kj = 0*2 + 1
    blocks.insert(2, b10.clone()); // 1*2 + 0
    let c = Cderi {
        blocks,
        kpts: vec![[0.0; 3], [0.1, 0.2, 0.3]],
        aosym: Aosym::S2,
    };

    // 1. A COMPACT request is the stored `(ki, kj)` block, verbatim. Serving it
    //    from the conjugate pair is not a re-packing -- the packed store has no
    //    `mu < nu` entry to transpose into place -- it is a different integral,
    //    and `b01 != b10` here precisely because `L` is not symmetric.
    let packed = sr_loop(&c, 0, 1, nao, true).expect("compact (0,1)");
    assert_eq!(packed[0].re, b01.data.re, "compact (0,1) real");
    assert_eq!(packed[0].im, b01.data.im, "compact (0,1) imaginary");
    let packed10 = sr_loop(&c, 1, 0, nao, true).expect("compact (1,0)");
    assert_eq!(packed10[0].re, b10.data.re, "compact (1,0) real");
    assert_eq!(packed10[0].im, b10.data.im, "compact (1,0) imaginary");
    assert_ne!(
        b01.data.re, b10.data.re,
        "the fixture is pointless unless the two blocks differ"
    );

    // 2. The SQUARE reassembles `L(0,1)` EXACTLY -- both triangles, from the
    //    two halves the store keeps.
    let sq = sr_loop(&c, 0, 1, nao, false).expect("square (0,1)");
    assert_eq!(sq[0].ncol, nao * nao);
    for l in 0..rank {
        for mu in 0..nao {
            for nu in 0..nao {
                let at = l * nao * nao + mu * nao + nu;
                assert_eq!(
                    (sq[0].re[at], sq[0].im[at]),
                    m(l, mu, nu),
                    "square (0,1) L={l} ({mu},{nu})"
                );
            }
        }
    }

    // 3. And the identity that assembly exists to preserve, stated on the
    //    OUTPUT: `L(ki,kj)[mu,nu] == conj(L(kj,ki)[nu,mu])` over the WHOLE
    //    square, not just the half either block stores.
    let sq10 = sr_loop(&c, 1, 0, nao, false).expect("square (1,0)");
    for l in 0..rank {
        for mu in 0..nao {
            for nu in 0..nao {
                let a = l * nao * nao + mu * nao + nu;
                let b = l * nao * nao + nu * nao + mu;
                assert_eq!(sq[0].re[a], sq10[0].re[b], "hermiticity re L={l} ({mu},{nu})");
                assert_eq!(
                    sq[0].im[a], -sq10[0].im[b],
                    "hermiticity im L={l} ({mu},{nu})"
                );
            }
        }
    }

    // 4. A square that needs a conjugate pair the store never computed is
    //    REFUSED, not silently filled from the wrong block.
    let half = Cderi {
        blocks: std::iter::once((1, b01.clone())).collect(),
        kpts: c.kpts.clone(),
        aosym: Aosym::S2,
    };
    assert!(
        sr_loop(&half, 0, 1, nao, true).is_ok(),
        "a compact request never needs the conjugate pair"
    );
    assert!(
        sr_loop(&half, 0, 1, nao, false).is_err(),
        "the upper triangle is unreachable without the conjugate pair"
    );
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

/// The range-separated route is upstream's DEFAULT, and since plan 14-07
/// sub-tasks 7b/7c (on D-PBC-24's cintx `range_omega`) this port has it.
///
/// This test used to assert the REFUSAL. It now asserts the route runs and
/// produces a usable `cderi` of the same shape as the compensated one — the
/// numbers themselves are gated in `tests/rsdf_builder.rs` against upstream.
#[test]
fn prefer_ccdf_false_builds_the_range_separated_route() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);

    let mut rs = Gdf::new(cell.clone(), &k);
    rs.prefer_ccdf = false;
    rs.build().expect("the range-separated route builds");
    let rs_naux = rs.cderi().expect("rs cderi").naoaux().expect("rs naoaux");

    let mut cc = Gdf::new(cell, &k);
    cc.prefer_ccdf = true;
    cc.build().expect("the compensated route builds");
    let cc_naux = cc.cderi().expect("cc cderi").naoaux().expect("cc naoaux");

    assert_eq!(
        rs_naux, cc_naux,
        "both routes fit in the same auxiliary basis, so the rank must agree"
    );
    // The two routes carry the LONG-range half on different grids, so their
    // meshes differ by construction — `_guess_omega`'s against `_guess_eta`'s.
    assert_ne!(
        pyscf_pbc_df::PeriodicDf::mesh(&rs),
        pyscf_pbc_df::PeriodicDf::mesh(&cc),
        "the RS mesh is _guess_omega's, the CC mesh is _guess_eta's"
    );
}
