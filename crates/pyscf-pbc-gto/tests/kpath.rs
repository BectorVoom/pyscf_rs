//! Band-structure k-paths: the special-point tables and the sampling machinery.
//!
//! # How the hardcoded tables are checked without ASE
//!
//! The special-point coordinates are transcribed from Setyawan-Curtarolo, and
//! ASE — the only reference implementation PySCF itself uses — is not a
//! dependency here. So instead of comparing numbers to a table, these tests
//! check the *defining geometric property*: every non-Gamma high-symmetry point
//! lies on the Brillouin-zone boundary, i.e. there is a nonzero reciprocal
//! lattice vector `G` with `k . G == |G|^2 / 2`. A mistyped coordinate almost
//! certainly falls off every Bragg plane, so this catches transcription errors
//! while depending on nothing but the cell's own reciprocal lattice.
//!
//! The cells are built from light elements: a k-path depends only on the
//! lattice, never on what sits at the sites, so using Li here keeps the test
//! free of any basis-set download.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, BravaisLattice, Cell, CellBuildArgs, band_path, band_path_from_segments,
    detect_lattice,
};

fn cell_with(a: [[f64; 3]; 3]) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("Li".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        ..Default::default()
    })
    .expect("cell must build")
}

/// Primitive BCC vectors for lattice parameter `a` — europium's lattice.
fn bcc(a: f64) -> [[f64; 3]; 3] {
    let h = a / 2.0;
    [[-h, h, h], [h, -h, h], [h, h, -h]]
}

/// Primitive FCC vectors.
fn fcc(a: f64) -> [[f64; 3]; 3] {
    let h = a / 2.0;
    [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
}

fn cubic(a: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
}

fn dot(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

/// Is `k` on a Bragg plane of the reciprocal lattice spanned by `b`?
fn on_bragg_plane(k: &[f64; 3], b: &[[f64; 3]; 3]) -> bool {
    for n1 in -2i32..=2 {
        for n2 in -2i32..=2 {
            for n3 in -2i32..=2 {
                if n1 == 0 && n2 == 0 && n3 == 0 {
                    continue;
                }
                let g = [
                    n1 as f64 * b[0][0] + n2 as f64 * b[1][0] + n3 as f64 * b[2][0],
                    n1 as f64 * b[0][1] + n2 as f64 * b[1][1] + n3 as f64 * b[2][1],
                    n1 as f64 * b[0][2] + n2 as f64 * b[1][2] + n3 as f64 * b[2][2],
                ];
                let gg = dot(&g, &g);
                if gg <= 0.0 {
                    continue;
                }
                // Scale-free comparison: k.G / (|G|^2/2) == 1.
                if ((dot(k, &g) / (0.5 * gg)) - 1.0).abs() < 1e-9 {
                    return true;
                }
            }
        }
    }
    false
}

/// Every non-Gamma special point must sit on the Brillouin-zone boundary.
#[test]
fn special_points_lie_on_the_zone_boundary() {
    let cases = [
        (BravaisLattice::Cubic, cubic(6.0)),
        (BravaisLattice::Fcc, fcc(6.8)),
        (BravaisLattice::Bcc, bcc(7.2)),
        (
            BravaisLattice::Tetragonal,
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 8.0]],
        ),
    ];
    for (lat, a) in cases {
        let cell = cell_with(a);
        let b = cell
            .reciprocal_vectors_2pi()
            .expect("reciprocal vectors must exist");
        for (label, frac) in lat.special_points() {
            let abs = cell.get_abs_kpts(&[*frac]).expect("abs k")[0];
            if *label == "G" {
                assert_eq!(*frac, [0.0, 0.0, 0.0], "{lat:?}: G must be the origin");
                continue;
            }
            assert!(
                on_bragg_plane(&abs, &b),
                "{lat:?}: special point {label} at fractional {frac:?} is not on any \
                 Bragg plane of this lattice, so the coordinate is very likely mistyped"
            );
        }
    }
}

/// The hexagonal case gets its own row: its reciprocal lattice is not cubic, so
/// it exercises the same invariant on a genuinely different metric.
#[test]
fn hexagonal_special_points_lie_on_the_zone_boundary() {
    let a = 5.0_f64;
    let c = 8.0_f64;
    let cell = cell_with([
        [a, 0.0, 0.0],
        [-a / 2.0, a * (3.0_f64).sqrt() / 2.0, 0.0],
        [0.0, 0.0, c],
    ]);
    let b = cell.reciprocal_vectors_2pi().expect("reciprocal vectors");
    for (label, frac) in BravaisLattice::Hexagonal.special_points() {
        if *label == "G" {
            continue;
        }
        let abs = cell.get_abs_kpts(&[*frac]).expect("abs k")[0];
        assert!(
            on_bragg_plane(&abs, &b),
            "hexagonal: {label} at {frac:?} is not on a Bragg plane"
        );
    }
}

/// The sampling machinery: endpoints, monotone axis, ticks, and the jump.
#[test]
fn the_bcc_path_is_sampled_consistently() {
    let cell = cell_with(bcc(7.2));
    let path = band_path(&cell, BravaisLattice::Bcc, 60).expect("bcc path");

    assert!(path.len() >= 60, "asked for 60 points, got {}", path.len());
    assert_eq!(path.scaled.len(), path.abs.len());
    assert_eq!(path.x.len(), path.abs.len());

    // `abs` must be exactly what get_abs_kpts makes of `scaled` — the two are
    // handed to different consumers (get_bands vs plotting) and must agree.
    let recomputed = cell.get_abs_kpts(&path.scaled).expect("abs");
    for (i, (p, q)) in path.abs.iter().zip(recomputed.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                (p[c] - q[c]).abs() < 1e-13,
                "point {i} component {c}: abs {} vs get_abs_kpts {}",
                p[c],
                q[c]
            );
        }
    }

    // The plot axis starts at zero and never goes backwards.
    assert!(path.x[0].abs() < 1e-15, "the axis must start at 0");
    assert!(
        path.x.windows(2).all(|w| w[1] >= w[0] - 1e-12),
        "the cumulative axis must be non-decreasing"
    );

    // BCC's conventional path is G-H-N-G-P-H then a jump to P-N: eight labelled
    // points, with the jump collapsing H and P into a single "H|P" tick.
    assert_eq!(
        path.tick_labels.len(),
        path.tick_x.len(),
        "one x per tick label"
    );
    assert_eq!(
        path.tick_labels,
        vec!["G", "H", "N", "G", "P", "H|P", "N"],
        "the jump between the two BCC segments must render as one H|P tick"
    );

    // Every tick must coincide with a k-point actually on the path, or the
    // plotted band would not pass through its own axis label.
    for (t, label) in path.tick_x.iter().zip(&path.tick_labels) {
        assert!(
            path.x.iter().any(|x| (x - t).abs() < 1e-9),
            "tick {label} at x = {t} does not land on a sampled k-point"
        );
    }

    // The first and last k-points are the path's endpoints: G and N.
    assert_eq!(path.scaled[0], [0.0, 0.0, 0.0], "the path starts at Gamma");
    let last = path.scaled[path.len() - 1];
    assert_eq!(last, [0.0, 0.0, 0.5], "the BCC path ends at N");
}

/// A caller-supplied path is honoured verbatim, and a single straight segment
/// is sampled uniformly.
#[test]
fn a_custom_segment_is_sampled_uniformly() {
    let cell = cell_with(cubic(6.0));
    let segments = vec![vec![
        ("G".to_string(), [0.0, 0.0, 0.0]),
        ("X".to_string(), [0.0, 0.5, 0.0]),
    ]];
    let path = band_path_from_segments(&cell, &segments, 10).expect("custom path");

    assert_eq!(path.tick_labels, vec!["G", "X"]);
    assert!(path.len() >= 11, "10 intervals means 11 points");

    // Uniform spacing along a single leg.
    let steps: Vec<f64> = path.x.windows(2).map(|w| w[1] - w[0]).collect();
    let first = steps[0];
    for (i, s) in steps.iter().enumerate() {
        assert!(
            (s - first).abs() < 1e-12,
            "step {i} = {s} differs from the first step {first}; a straight leg \
             must be sampled uniformly"
        );
    }
}

/// Recognition must identify the standard forms and decline everything else.
#[test]
fn lattice_recognition_identifies_or_declines() {
    assert_eq!(
        detect_lattice(&cell_with(cubic(6.0))),
        Some(BravaisLattice::Cubic)
    );
    assert_eq!(
        detect_lattice(&cell_with(fcc(6.8))),
        Some(BravaisLattice::Fcc)
    );
    assert_eq!(
        detect_lattice(&cell_with(bcc(7.2))),
        Some(BravaisLattice::Bcc),
        "bcc is europium's lattice; it must be recognised"
    );
    assert_eq!(
        detect_lattice(&cell_with([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 8.0]
        ])),
        Some(BravaisLattice::Tetragonal)
    );

    // A skewed cell matches no standard form, and must NOT be forced into the
    // nearest one — a band path for the wrong lattice is silently wrong.
    assert_eq!(
        detect_lattice(&cell_with([
            [6.0, 0.0, 0.0],
            [0.7, 5.3, 0.0],
            [0.2, 0.4, 7.1]
        ])),
        None,
        "an unrecognised lattice must decline rather than guess"
    );

    // A BCC cell perturbed well past the tolerance is no longer BCC.
    let mut nearly = bcc(7.2);
    nearly[0][0] += 0.05;
    assert_eq!(
        detect_lattice(&cell_with(nearly)),
        None,
        "recognition must be strict, not nearest-match"
    );
}
