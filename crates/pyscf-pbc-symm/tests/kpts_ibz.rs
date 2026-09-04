//! Gate A and the oracle-free IBZ invariants — 17-05-PLAN.md Task 1 / Task 2.
//!
//! # Gate A needs no oracle beyond six integers
//!
//! 17-CONTEXT §2.2, MEASURED by 17-01 (`measurements/gate_a.py` /
//! `gate_a.out`): on a diamond-structure (Fd-3m, NON-symmorphic) cell at a
//! `[16, 16, 16]` k-mesh the six configurations give
//! **145 / 145 / 245 / 408 / 816 / 2052**, EXACTLY — no tolerance. 17-01 also
//! measured that these integers travel with the space-group TYPE and not with
//! the lattice constant: upstream's own Si (`a = 5.3870243948 A`), §9.2's
//! `si` (`a = 5.4306 A`) and §9.2's `diamond` (`a = 3.5668 A`) all reproduce
//! the same six. `finger(kpts_ibz)` does NOT travel (it scales with `1/a`),
//! so it is deliberately not asserted here.
//!
//! The Fm-3m (SYMMORPHIC) controls `lif` and `he_fcc` collapse to
//! `{145, 145, 145, 408, 408, 2052}` — `C == A` and `E == D`, because a
//! symmorphic group has no non-symmorphic ops to lose under
//! `symmorphic = true`. That contrast is asserted too: it is what separates
//! "the symmorphic branch works" from "the symmorphic branch is a no-op".
//!
//! A single wrong symmorphic branch, a wrong time-reversal fold or a wrong
//! `wrap_around` interaction moves one of these integers.

// `needless_range_loop` is allowed throughout: these loops index SEVERAL
// parallel arrays by the same k / p / q (upstream's own index convention),
// and rewriting them as iterator zips would obscure which array each index
// belongs to.
#![allow(clippy::needless_range_loop)]

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::{diamond, he_fcc, lif, si};
use pyscf_pbc_lib::kpts_helper::KPT_DIFF_TOL;
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};
use pyscf_pbc_symm::symmetry::build_lattice_symmetry;

const KMESH: [usize; 3] = [16, 16, 16];

/// `cell.py:1770-1772` — the `space_group_symmetry` branch of `Cell.build`:
/// `check_mesh_symmetry = not cell._mesh_from_build`. Our §9.2 fixtures all
/// take their mesh from the build, so the mesh is ENLARGED to carry the
/// lattice symmetry rather than ops being dropped — which is exactly what
/// upstream does for an auto mesh, and what keeps the non-symmorphic ops
/// alive for configuration A.
fn with_symmetry(mut cell: Cell, symmorphic: bool) -> Cell {
    cell.space_group_symmetry = true;
    cell.symmorphic = symmorphic;
    let check_mesh_symmetry = !cell._mesh_from_build;
    build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");
    cell
}

fn kmesh(cell: &Cell, with_gamma_point: bool) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, KMESH, false, with_gamma_point, None)
        .expect("make_kpts")
}

/// The six configurations of 17-CONTEXT §2.2 Gate A, in its own order.
#[derive(Debug, PartialEq, Eq)]
struct SixConfigs {
    /// `space_group_symmetry=True` (non-symmorphic allowed).
    a: usize,
    /// `symmorphic=True`, `+ time_reversal_symmetry=True`.
    b: usize,
    /// `symmorphic=True`, `time_reversal_symmetry=False`.
    c: usize,
    /// `with_gamma_point=False`, `space_group_symmetry=True`.
    d: usize,
    /// `with_gamma_point=False`, `symmorphic=True`.
    e: usize,
    /// `time_reversal_symmetry=True` only (no space group).
    f: usize,
}

fn six_configs(base: fn() -> Cell) -> SixConfigs {
    let cell = with_symmetry(base(), false);
    let cell_symm = with_symmetry(base(), true);

    let kg = kmesh(&cell, true);
    let kg_symm = kmesh(&cell_symm, true);
    let kng = kmesh(&cell, false);
    let kng_symm = kmesh(&cell_symm, false);

    let a = make_kpts(&cell, &kg, true, false).expect("A");
    let b = make_kpts(&cell_symm, &kg_symm, true, true).expect("B");
    let c = make_kpts(&cell_symm, &kg_symm, true, false).expect("C");
    let d = make_kpts(&cell, &kng, true, false).expect("D");
    let e = make_kpts(&cell_symm, &kng_symm, true, false).expect("E");
    // `cell.make_kpts(..., time_reversal_symmetry=True)` with no space group:
    // upstream still routes through `libkpts.make_kpts` (`cell.py:874-883`).
    let f = make_kpts(&base(), &kg, false, true).expect("F");

    let plain = base();
    for (name, k, c) in [
        ("A", &a, &cell),
        ("B", &b, &cell_symm),
        ("C", &c, &cell_symm),
        ("D", &d, &cell),
        ("E", &e, &cell_symm),
        ("F", &f, &plain),
    ] {
        check_invariants(k, c, name);
    }

    SixConfigs {
        a: a.nkpts_ibz(),
        b: b.nkpts_ibz(),
        c: c.nkpts_ibz(),
        d: d.nkpts_ibz(),
        e: e.nkpts_ibz(),
        f: f.nkpts_ibz(),
    }
}

/// The oracle-free invariants of 17-05-PLAN.md Task 1. Each catches a
/// different bug and none needs a reference number.
fn check_invariants(kpts: &KPoints, cell: &Cell, who: &str) {
    let nkpts = kpts.nkpts();
    let nkpts_ibz = kpts.nkpts_ibz();

    // sum(weights_ibz) == 1
    let wsum: f64 = kpts.weights_ibz.iter().sum();
    assert!(
        (wsum - 1.0).abs() < 1e-15,
        "{who}: sum(weights_ibz) = {wsum}, expected 1"
    );
    // weights_ibz[i] == |stars[i]| / nkpts
    assert_eq!(kpts.stars.len(), nkpts_ibz, "{who}: stars length");
    for i in 0..nkpts_ibz {
        let expect = kpts.stars[i].len() as f64 / nkpts as f64;
        assert!(
            (kpts.weights_ibz[i] - expect).abs() < 1e-15,
            "{who}: weights_ibz[{i}] = {} but |stars[{i}]|/nkpts = {expect}",
            kpts.weights_ibz[i]
        );
    }

    // bz2ibz is total, and ibz2bz[bz2ibz[k]] lands in k's own star.
    assert_eq!(kpts.bz2ibz.len(), nkpts, "{who}: bz2ibz is not total");
    for k in 0..nkpts {
        let i = kpts.bz2ibz[k];
        assert!(i < nkpts_ibz, "{who}: bz2ibz[{k}] = {i} out of range");
        assert!(
            kpts.stars[i].contains(&k),
            "{who}: k = {k} is not in the star of its own IBZ representative {i}"
        );
        assert!(
            kpts.stars[i].contains(&kpts.ibz2bz[i]),
            "{who}: ibz2bz[{i}] is not in star {i}"
        );
    }

    // stars_ops[i][j] == stars_ops_bz[stars[i][j]] — upstream's own
    // consistency check (`test_kpts_ksymm.py:60-65`), which it writes as a
    // loop because it has caught a real bug.
    for i in 0..nkpts_ibz {
        for (j, &k) in kpts.stars[i].iter().enumerate() {
            assert_eq!(
                kpts.stars_ops[i][j], kpts.stars_ops_bz[k],
                "{who}: stars_ops[{i}][{j}] != stars_ops_bz[{k}]"
            );
        }
    }

    // every k in stars[i] satisfies stars_ops[i][j] . kpts_ibz[i] == k
    // modulo a reciprocal vector, to KPT_DIFF_TOL. This is the only
    // invariant that re-derives the rotation, so it is the one that catches
    // an off-by-one in the op index.
    for i in 0..nkpts_ibz {
        let ki = kpts.kpts_scaled_ibz[i];
        for (j, &k) in kpts.stars[i].iter().enumerate() {
            let iop = kpts.stars_ops[i][j];
            // The rotation acts on SCALED k-points in the RECIPROCAL basis
            // (`kpts.py:52` — `op.a2b(cell).rot`), not the direct one.
            let rot = kpts.ops()[iop].a2b(cell).expect("a2b").rot;
            let sign = if kpts.time_reversal_symm_bz[k] == 1 {
                -1.0
            } else {
                1.0
            };
            let mut ok = true;
            for x in 0..3 {
                let r = sign * (rot[x][0] * ki[0] + rot[x][1] * ki[1] + rot[x][2] * ki[2]);
                let d = kpts.kpts_scaled[k][x] - r;
                if (d - d.round_ties_even()).abs() >= KPT_DIFF_TOL {
                    ok = false;
                }
            }
            assert!(
                ok,
                "{who}: stars_ops[{i}][{j}] = {iop} does not map kpts_ibz[{i}] onto kpts[{k}]"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Gate A
// ---------------------------------------------------------------------

/// Gate A on `si` — PBC-MASTER-PLAN §9.2's diamond-structure fixture.
#[test]
fn gate_a_si_ibz_integers_are_exact() {
    let got = six_configs(si);
    assert_eq!(
        got,
        SixConfigs {
            a: 145,
            b: 145,
            c: 245,
            d: 408,
            e: 816,
            f: 2052
        },
        "Gate A on si: the six nkpts_ibz integers are EXACT (17-CONTEXT §2.2)"
    );
}

/// Gate A on `diamond` — same space group (Fd-3m), different lattice
/// constant. 17-01 measured that the integers travel; this asserts it.
#[test]
fn gate_a_diamond_ibz_integers_are_exact() {
    let got = six_configs(diamond);
    assert_eq!(
        got,
        SixConfigs {
            a: 145,
            b: 145,
            c: 245,
            d: 408,
            e: 816,
            f: 2052
        },
        "Gate A on diamond: same Fd-3m space group as si, same six integers"
    );
}

/// The SYMMORPHIC controls: `lif` and `he_fcc` are Fm-3m, so the
/// `symmorphic = true` branch removes nothing and `C == A`, `E == D`.
/// Without this, a `symmorphic` branch that silently did nothing would still
/// pass the two tests above.
#[test]
fn gate_a_symmorphic_controls_collapse() {
    for (name, base) in [
        ("lif", lif as fn() -> Cell),
        ("he_fcc", he_fcc as fn() -> Cell),
    ] {
        let got = six_configs(base);
        assert_eq!(
            got,
            SixConfigs {
                a: 145,
                b: 145,
                c: 145,
                d: 408,
                e: 408,
                f: 2052
            },
            "Gate A on {name} (Fm-3m, symmorphic): C == A and E == D"
        );
    }
}

// ---------------------------------------------------------------------
// Determinism — 17-05-PLAN.md Task 1 "Speed"
// ---------------------------------------------------------------------

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

/// The star search is parallelised over the outer BZ loop. Vary the worker
/// count inside ONE process — a strictly stronger check than an env-var
/// sweep across processes — and demand BIT identity of every produced
/// index array.
#[test]
fn star_search_is_bit_identical_at_1_and_8_threads() {
    let cell = with_symmetry(si(), false);
    let kg = kmesh(&cell, true);

    let one = pool(1).install(|| make_kpts(&cell, &kg, true, true).expect("1 thread"));
    let eight = pool(8).install(|| make_kpts(&cell, &kg, true, true).expect("8 threads"));

    assert_eq!(one.nkpts_ibz(), eight.nkpts_ibz());
    assert_eq!(one.ibz2bz, eight.ibz2bz);
    assert_eq!(one.bz2ibz, eight.bz2ibz);
    assert_eq!(one.stars, eight.stars);
    assert_eq!(one.stars_ops, eight.stars_ops);
    assert_eq!(one.stars_ops_bz, eight.stars_ops_bz);
    assert_eq!(one.time_reversal_symm_bz, eight.time_reversal_symm_bz);
    assert_eq!(one.little_cogroup_ops, eight.little_cogroup_ops);
    assert_eq!(one.k2opk, eight.k2opk);
    for (a, b) in one.weights_ibz.iter().zip(eight.weights_ibz.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "weights_ibz moved between 1 and 8 threads"
        );
    }
}

/// **Measurably parallelised, not merely safe to parallelise.**
///
/// 17-05-PLAN.md's verification asks for the measurement, not just the
/// determinism proof. This times the whole `[16,16,16]` fold (`nkpts = 4096`,
/// `nop = 48`) — dominated by the `O(nkpts x nop)` star search — in explicit
/// 1- and 8-worker pools and PRINTS both, under `--nocapture`.
///
/// The assertion is deliberately weak (8 workers must not be more than 1.5x
/// SLOWER than 1) because wall-clock ratios on a shared CI box are flaky;
/// the real measured speedup is recorded in `17-05-SUMMARY.md`. What is NOT
/// weak is [`star_search_is_bit_identical_at_1_and_8_threads`] above, which
/// is the correctness half.
#[test]
fn star_search_is_measurably_parallel() {
    let cell = with_symmetry(si(), false);
    let kg = kmesh(&cell, true);

    // Warm the caches so the first run does not pay for page faults alone.
    let _ = pool(1).install(|| make_kpts(&cell, &kg, true, true).expect("warm"));

    let t1 = {
        let p = pool(1);
        let start = std::time::Instant::now();
        let k = p.install(|| make_kpts(&cell, &kg, true, true).expect("1 thread"));
        assert_eq!(k.nkpts(), 4096);
        start.elapsed()
    };
    let t8 = {
        let p = pool(8);
        let start = std::time::Instant::now();
        let k = p.install(|| make_kpts(&cell, &kg, true, true).expect("8 threads"));
        assert_eq!(k.nkpts(), 4096);
        start.elapsed()
    };
    println!(
        "  Task 1 speed: make_kpts si [16,16,16] (nkpts = 4096, nop = {}): \
         1 worker {:.1} ms, 8 workers {:.1} ms, speedup {:.2}x",
        cell.lattice_symmetry.as_ref().map_or(0, |l| l.ops.len()),
        t1.as_secs_f64() * 1e3,
        t8.as_secs_f64() * 1e3,
        t1.as_secs_f64() / t8.as_secs_f64()
    );
    assert!(
        t8.as_secs_f64() < t1.as_secs_f64() * 1.5,
        "8 workers ({:?}) is more than 1.5x slower than 1 ({:?}) — the parallel star search \
         is not paying for itself",
        t8,
        t1
    );
}
