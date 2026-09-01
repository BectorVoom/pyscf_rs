//! Oracle-free tests for the `KPoints` k-tuple machinery — 17-05-PLAN.md
//! Task 5.
//!
//! None of these needs a reference number: each is an identity the tables
//! must satisfy to be tables at all. They are the analogue of Task 1's
//! `stars_ops[i][j] == stars_ops_bz[stars[i][j]]` — cheap, and each catches
//! a different bug.

// `needless_range_loop` is allowed throughout: these loops index SEVERAL
// parallel arrays by the same k / p / q (upstream's own index convention),
// and rewriting them as iterator zips would obscure which array each index
// belongs to.
#![allow(clippy::needless_range_loop)]


use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::diamond;
use pyscf_pbc_symm::kpts::{KPoints, KQuartets, make_kpts};
use pyscf_pbc_symm::symmetry::build_lattice_symmetry;

fn fixture(mesh: [usize; 3]) -> (Cell, KPoints) {
    let mut cell = diamond();
    cell.space_group_symmetry = true;
    cell.symmorphic = false;
    let check_mesh_symmetry = !cell._mesh_from_build;
    build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");
    let kmesh = pyscf_pbc_gto::make_kpts_default(&cell, mesh).expect("make_kpts_default");
    // `time_reversal = false`: with it on, `k2opk` has `2*nop` columns while
    // `ops` has `nop` entries, and `little_cogroups` refuses (see its doc —
    // upstream raises IndexError on the same input).
    let kpts = make_kpts(&cell, &kmesh, true, false).expect("make_kpts");
    (cell, kpts)
}

/// `add(k, inv(k)) == Gamma` for every `k`, plus the two structural
/// properties that make the tables tables: `addition_table` is a Latin
/// square (each row is a permutation of `0..nkpts`) and `inverse_table` is
/// an involution.
#[test]
fn addition_and_inverse_tables_are_consistent() {
    let (_cell, kpts) = fixture([3, 3, 3]);
    let nk = kpts.nkpts();
    let add = kpts.addition_table();
    let inv = kpts.inverse_table();

    // Gamma is the zeroth point of a gamma-centred mesh.
    let gamma = 0usize;
    assert!(
        kpts.kpts_scaled[gamma].iter().all(|x| x.abs() < 1e-12),
        "the zeroth k-point of a gamma-centred mesh must be Gamma"
    );

    for k in 0..nk {
        let ik = inv[k] as usize;
        assert_eq!(
            add[k * nk + ik] as usize,
            gamma,
            "add(k={k}, inv(k)={ik}) must be Gamma"
        );
        assert_eq!(inv[ik] as usize, k, "inverse_table is not an involution at {k}");
    }

    for i in 0..nk {
        let mut row: Vec<i32> = add[i * nk..(i + 1) * nk].to_vec();
        row.sort_unstable();
        assert_eq!(
            row,
            (0..nk as i32).collect::<Vec<_>>(),
            "addition_table row {i} is not a permutation"
        );
        // The group is abelian: k_i + k_j == k_j + k_i.
        for j in 0..nk {
            assert_eq!(add[i * nk + j], add[j * nk + i], "addition_table is not symmetric");
        }
    }
}

/// `get_kconserv` DELEGATES to `pyscf_pbc_lib::kpts_helper::get_kconserv`
/// (17-05-PLAN.md Task 5: two kconservs would drift). This asserts the
/// delegation lands on the same table upstream's own fast path computes,
/// `add_tab[add_tab[:, inv_tab], :]` (`kpts.py:1079-1081`) — i.e. that the
/// two routes agree, which is exactly what NOT re-porting buys.
#[test]
fn get_kconserv_matches_the_addition_inverse_table_route() {
    let (cell, kpts) = fixture([3, 3, 3]);
    let nk = kpts.nkpts();
    let kconserv = kpts.get_kconserv(&cell);
    let add = kpts.addition_table();
    let inv = kpts.inverse_table();

    for i in 0..nk {
        for j in 0..nk {
            // A[i][j] = index of k_i - k_j
            let a = add[i * nk + inv[j] as usize] as usize;
            for m in 0..nk {
                assert_eq!(
                    kconserv.get(i, j, m),
                    add[a * nk + m],
                    "kconserv[{i},{j},{m}] disagrees with add[add[i, inv[j]], m]"
                );
            }
        }
    }
}

/// `ktuple_to_index` / `index_to_ktuple` round-trip over the WHOLE range,
/// for every tuple order the workspace uses (2 for k-pairs, 3 for the
/// quartet generator).
#[test]
fn ktuple_index_round_trips_over_the_whole_range() {
    let (_cell, kpts) = fixture([2, 2, 2]);
    let nk = kpts.nkpts();
    for ntuple in [1usize, 2, 3] {
        let n = nk.pow(ntuple as u32);
        for t in 0..n {
            let digits = kpts.index_to_ktuple(t, ntuple);
            assert_eq!(digits.len(), ntuple);
            assert!(digits.iter().all(|&d| d < nk), "digit out of range at t = {t}");
            assert_eq!(kpts.ktuple_to_index(&digits), t, "round-trip failed at t = {t}");
        }
    }
}

/// `make_ktuples_ibz(ntuple = 2)` unfolds to exactly `nkpts^2` pairs with no
/// duplicates — the stars PARTITION the full tuple space. A wrong
/// `np.unique` port, a wrong `bz2bz` sentinel slot or a wrong reversal all
/// break this and nothing else.
#[test]
fn make_ktuples_ibz_partitions_the_full_tuple_space() {
    let (_cell, kpts) = fixture([2, 2, 2]);
    let nk = kpts.nkpts();

    for ntuple in [2usize, 3] {
        let n = nk.pow(ntuple as u32);
        let t = kpts.make_ktuples_ibz(ntuple);

        let total: usize = t.stars.iter().map(|s| s.len()).sum();
        assert_eq!(total, n, "ntuple={ntuple}: stars do not cover {n} tuples");

        let mut seen = vec![false; n];
        for star in &t.stars {
            for &k in star {
                assert!(!seen[k], "ntuple={ntuple}: tuple {k} appears in two stars");
                seen[k] = true;
            }
        }
        assert!(seen.iter().all(|b| *b), "ntuple={ntuple}: some tuple is in no star");

        assert_eq!(t.ibz2bz.len(), t.stars.len());
        assert_eq!(t.weight_ibz.len(), t.stars.len());
        let wsum: f64 = t.weight_ibz.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-14, "ntuple={ntuple}: weights sum to {wsum}");
        for (i, star) in t.stars.iter().enumerate() {
            assert!((t.weight_ibz[i] - star.len() as f64 / n as f64).abs() < 1e-15);
            assert!(star.contains(&t.ibz2bz[i]), "the representative is not in its own star");
            // stars_ops[i][j] == stars_ops_bz[stars[i][j]] — the same
            // consistency identity Task 1 asserts for single k-points.
            for (j, &k) in star.iter().enumerate() {
                assert_eq!(t.stars_ops[i][j], t.stars_ops_bz[k]);
            }
            assert_eq!(t.bz2ibz[t.ibz2bz[i]], i);
        }
    }
}

/// `make_k4_ibz(sym = "s1")` — every quartet must satisfy momentum
/// conservation by construction, and the quartet list must be exactly the
/// 3-tuple IBZ with the fourth index appended.
#[test]
fn make_k4_ibz_s1_quartets_conserve_momentum() {
    let (cell, kpts) = fixture([2, 2, 2]);
    let k4 = kpts.make_k4_ibz(&cell, "s1").expect("s1");
    let kconserv = kpts.get_kconserv(&cell);
    let t3 = kpts.make_ktuples_ibz(3);

    assert_eq!(k4.k4.len(), t3.ibz2bz.len());
    for (i, q) in k4.k4.iter().enumerate() {
        let [ki, kj, ka, kb] = *q;
        assert_eq!(kpts.index_to_ktuple(t3.ibz2bz[i], 3), vec![ki, kj, ka]);
        assert_eq!(kconserv.get(ki, ka, kj) as usize, kb);
    }

    // s2 / s4 are 17-09's, and say so rather than silently returning s1.
    assert!(kpts.make_k4_ibz(&cell, "s2").is_err());
    assert!(kpts.make_k4_ibz(&cell, "s4").is_err());
    assert!(kpts.make_k4_ibz(&cell, "nonsense").is_err());
}

/// `KQuartets` builds, and every stabiliser element genuinely stabilises the
/// quartet's first index.
#[test]
fn kquartets_stabilizer_fixes_the_first_index() {
    let (cell, kpts) = fixture([2, 2, 2]);
    let mut kq = KQuartets::build(&kpts, &cell).expect("KQuartets");
    kq.cache_stabilizer(&kpts);

    for (i, quartet) in kq.kqrts_ibz.iter().enumerate() {
        for (klcd, iop) in kq.loop_stabilizer(i) {
            assert_eq!(
                kpts.k2opk[quartet[0]][iop] as usize,
                quartet[0],
                "stabilizer op {iop} does not fix quartet {i}'s first index"
            );
            for d in 0..4 {
                assert_eq!(klcd[d], kpts.k2opk[quartet[d]][iop] as usize);
            }
        }
    }
}

/// `little_cogroups` / `little_cogroup_rep`, the pair 17-04's
/// `symm_adapted_basis` takes as parameters.
///
/// Oracle-free identities: every full-BZ point's little co-group has the
/// SAME order as its IBZ representative's (conjugation is an isomorphism);
/// `indices[ki]` is a permutation; and `little_cogroup_rep` returns a
/// representation whose character vector has one entry per group element.
#[test]
fn little_cogroups_are_conjugates_of_the_ibz_representatives() {
    let (_cell, kpts) = fixture([2, 2, 2]);
    let (copgs, indices) = kpts.little_cogroups().expect("little_cogroups");

    assert_eq!(copgs.len(), kpts.nkpts());
    assert_eq!(indices.len(), kpts.nkpts());

    for ki in 0..kpts.nkpts() {
        let ki_ibz = kpts.bz2ibz[ki];
        let order_ibz = kpts.little_cogroup_ops[ki_ibz].len();
        assert_eq!(
            copgs[ki].order(),
            order_ibz,
            "little co-group of BZ point {ki} has a different order from its IBZ representative"
        );
        let mut perm = indices[ki].clone();
        perm.sort_unstable();
        assert_eq!(perm, (0..order_ibz).collect::<Vec<_>>(), "indices[{ki}] is not a permutation");
        // Every op in the little co-group really does fix the IBZ k-point.
        for &iop in &kpts.little_cogroup_ops[ki_ibz] {
            assert_eq!(kpts.k2opk[kpts.ibz2bz[ki_ibz]][iop] as usize, kpts.ibz2bz[ki_ibz]);
        }
    }

    // little_cogroup_rep: one character per element of the target group.
    for ki in 0..kpts.nkpts() {
        let rep = kpts.little_cogroup_rep(ki, 0).expect("little_cogroup_rep");
        assert_eq!(rep.chi.len(), copgs[ki].order());
        assert_eq!(rep.group.order(), copgs[ki].order());
    }
}

/// `make_gdf_kptij_lst_jk` — `nkpts` diagonal pairs first, then every
/// `(k_ibz, k_bz)` pair whose `k_bz` is not the representative itself.
#[test]
fn gdf_kptij_lst_has_the_diagonal_first_and_no_self_pairs_after() {
    let (_cell, kpts) = fixture([2, 2, 2]);
    let lst = kpts.make_gdf_kptij_lst_jk();
    let nk = kpts.nkpts();

    for i in 0..nk {
        assert_eq!(lst[i].0, kpts.kpts[i]);
        assert_eq!(lst[i].1, kpts.kpts[i]);
    }
    let mut expected = nk;
    for i in 0..kpts.nkpts_ibz() {
        expected += nk - pyscf_pbc_gto::member(&kpts.kpts_ibz[i], &kpts.kpts).len();
    }
    assert_eq!(lst.len(), expected);
    for pair in &lst[nk..] {
        assert_ne!(pair.0, pair.1, "an off-diagonal entry is a self-pair");
    }
}
