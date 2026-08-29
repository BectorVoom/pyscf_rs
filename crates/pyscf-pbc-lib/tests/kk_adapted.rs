//! `kk_adapted_iter` / `group_by_conj_pairs` / `unique_with_wrap_around` —
//! plan 14-02 Task 2.
//!
//! Every expectation is upstream PySCF 2.12.1 output on the diamond reference
//! cell. Re-derive with:
//!
//! ```text
//! PYTHONPATH=. .venv/bin/python -c "
//! import sys; sys.path.insert(0,'.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements')
//! from _cells import diamond
//! from pyscf.pbc.lib.kpts_helper import kk_adapted_iter
//! c = diamond(); kpts = c.make_kpts([2,2,2])
//! for kpt, ki, kj, sc in kk_adapted_iter(c, kpts): print(sc, list(ki), list(kj))"
//! ```
//!
//! **Why this is tested here and not where it is consumed.** The k-index
//! bookkeeping is the classic source of a silently-wrong exchange matrix: every
//! group is one `get_2c2e` call and one `cderi` block, so an off-by-one in
//! `kj_idx` produces a plausible number rather than a crash.

use pyscf_pbc_lib::kpts_helper::{group_by_conj_pairs, kk_adapted_iter, unique_with_wrap_around};

/// A Monkhorst-Pack mesh in SCALED coordinates — `cell.make_kpts` with
/// `with_gamma_point = True`: `n/N` for `n = 0..N-1`.
fn mp_scaled(km: [usize; 3]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for a in 0..km[0] {
        for b in 0..km[1] {
            for c in 0..km[2] {
                out.push([
                    a as f64 / km[0] as f64,
                    b as f64 / km[1] as f64,
                    c as f64 / km[2] as f64,
                ]);
            }
        }
    }
    out
}

/// `scaled_dk[i*n + j] = scaled[j] - scaled[i]` — upstream's
/// `dk = (kpts[None,:,:] - kpts[:,None,:]).reshape(-1,3)`.
fn dk(scaled: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = scaled.len();
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            out.push([
                scaled[j][0] - scaled[i][0],
                scaled[j][1] - scaled[i][1],
                scaled[j][2] - scaled[i][2],
            ]);
        }
    }
    out
}

fn groups(km: [usize; 3]) -> Vec<(bool, Vec<usize>, Vec<usize>)> {
    let scaled = mp_scaled(km);
    let n = scaled.len();
    let d = dk(&scaled);
    kk_adapted_iter(n, &d, &d, None, true)
        .expect("kk_adapted_iter")
        .into_iter()
        .map(|g| (g.self_conj, g.ki_idx, g.kj_idx))
        .collect()
}

#[test]
fn diamond_2x2x2_matches_upstream() {
    let g = groups([2, 2, 2]);
    // Upstream yields 8 groups, all self-conjugate — every k-difference on a
    // 2x2x2 mesh is half a reciprocal-lattice vector, hence its own negative.
    assert_eq!(g.len(), 8);
    let all: Vec<usize> = (0..8).collect();
    let want_kj: [[usize; 8]; 8] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 7, 4, 5, 2, 3, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ];
    for (i, (self_conj, ki, kj)) in g.iter().enumerate() {
        assert!(self_conj, "group {i} should be self-conjugate");
        assert_eq!(ki, &all, "group {i} ki");
        assert_eq!(kj, &want_kj[i].to_vec(), "group {i} kj");
    }
}

#[test]
fn gamma_is_one_self_conjugate_group() {
    let g = groups([1, 1, 1]);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0], (true, vec![0], vec![0]));
}

#[test]
fn diamond_2x1x1_matches_upstream() {
    let g = groups([2, 1, 1]);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0], (true, vec![0, 1], vec![0, 1]));
    assert_eq!(g[1], (true, vec![0, 1], vec![1, 0]));
}

/// A 3-fold mesh is where a NON-self-conjugate group first appears: `+1/3` and
/// `-1/3` are distinct points of the first Brillouin zone. Upstream yields two
/// groups, the second with `self_conj = False`, and with
/// `time_reversal_symmetry = True` (its default) it does NOT also yield the
/// conjugate partner.
#[test]
fn diamond_3x1x1_has_a_non_self_conjugate_group() {
    let g = groups([3, 1, 1]);
    assert_eq!(g.len(), 2, "upstream yields 2 groups, not 3");
    assert_eq!(g[0], (true, vec![0, 1, 2], vec![0, 1, 2]));
    assert_eq!(g[1], (false, vec![0, 1, 2], vec![1, 2, 0]));
}

/// With time-reversal symmetry OFF the conjugate partner IS yielded, so a
/// 3-fold mesh gives three groups instead of two.
#[test]
fn time_reversal_off_yields_the_conjugate_partner() {
    let scaled = mp_scaled([3, 1, 1]);
    let d = dk(&scaled);
    let g = kk_adapted_iter(3, &d, &d, None, false).expect("iter");
    assert_eq!(g.len(), 3);
    assert!(g[0].self_conj);
    assert!(!g[1].self_conj);
    assert!(!g[2].self_conj);
    // The partner group is the transpose: kj of one is ki of the other.
    assert_eq!(g[1].kj_idx, vec![1, 2, 0]);
    assert_eq!(g[2].kj_idx, vec![2, 0, 1]);
}

/// Upstream refuses `kk_idx` combined with time-reversal symmetry
/// (`NotImplementedError`), and so does this port.
#[test]
fn kk_idx_with_time_reversal_is_refused() {
    let scaled = mp_scaled([2, 2, 2]);
    let d = dk(&scaled);
    let sel: Vec<usize> = (0..8).map(|k| k * 8 + k).collect();
    assert!(kk_adapted_iter(8, &d, &d, Some(&sel), true).is_err());
    assert!(kk_adapted_iter(8, &d, &d, Some(&sel), false).is_ok());
}

#[test]
fn unique_with_wrap_around_folds_reciprocal_vectors() {
    // 0.5 and -0.5 are the same point of the first Brillouin zone; 1.0 is gamma.
    let k = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let (index, inverse) = unique_with_wrap_around(&k);
    assert_eq!(index.len(), 2, "only gamma and the zone boundary are distinct");
    assert_eq!(inverse[0], inverse[3], "1.0 folds onto gamma");
    assert_eq!(inverse[1], inverse[2], "-0.5 folds onto +0.5");
    assert_ne!(inverse[0], inverse[1]);
}

#[test]
fn group_by_conj_pairs_labels_all_three_cases() {
    // gamma is self-conjugate; +1/3 and -1/3 pair; +1/4 alone has no partner.
    let k = [
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 0.0, 0.0],
        [-1.0 / 3.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ];
    let p = group_by_conj_pairs(&k, true);
    assert_eq!(p[0], (0, Some(0)), "gamma is self-conjugate and comes first");
    assert!(p.contains(&(1, Some(2))) || p.contains(&(2, Some(1))));
    assert!(p.contains(&(3, None)), "+1/4 has no conjugate in the set");
}
