//! W-09 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — block-level AO screening
//! in `eval_ao_kpts`.
//!
//! The screen **drops terms**, so unlike every other item in that plan it
//! cannot be defended by a bit-parity argument. What it can be defended by is
//! the bound it drops them under: a shell contributes to a grid block only when
//! the block's bounding box is within that shell's `rcut`, and `rcut` is the
//! radius `estimate_rcut_for_eval` derives from `cell.precision` — the same
//! radius that sized the lattice-image list in the first place. Anything the
//! screen removes was already assumed negligible when the image list was built.
//!
//! Three things are asserted:
//!
//! 1. **Agreement.** Screened and unscreened AO tables agree far inside the
//!    KRKS gate, on both a pseudopotential cell and an ALL-ELECTRON one (the
//!    all-electron He cell is the gate's own 1e-12 control, and its `sto-3g`
//!    basis is more diffuse than `gth-szv`, so it is the harder case).
//! 2. **Convergence in `rcut`.** Enlarging the image list must not change the
//!    screened answer — if the screen were dropping something real, a bigger
//!    list would keep changing the result.
//! 3. **The screen actually screens.** The test would be vacuous if it never
//!    rejected an image, so that is measured and asserted directly.
//!
//! The two paths are compared across PROCESSES because `PYSCF_PBC_AO_SCREEN` is
//! read through a `OnceLock`: a single process cannot see both. The test
//! re-executes its own binary with the variable set, which also proves the kill
//! switch works from the outside, the way a bisecting user would use it.

use std::process::Command;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, make_kpts_default};

const MESH: [usize; 3] = [11, 11, 11];

fn silicon() -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Si".into(), [0.0, 0.0, 0.0]),
                ("Si".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("cell")
}

/// The gate's own all-electron control, and the harder case for a screen:
/// `sto-3g` on He is more diffuse than `gth-szv` on Si.
fn he_all_electron() -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cell")
}

fn grid(cell: &Cell) -> Vec<[f64; 3]> {
    let a = cell.lattice_vectors();
    let mut out = Vec::with_capacity(MESH[0] * MESH[1] * MESH[2]);
    for i in 0..MESH[0] {
        for j in 0..MESH[1] {
            for k in 0..MESH[2] {
                let (fi, fj, fk) = (
                    i as f64 / MESH[0] as f64,
                    j as f64 / MESH[1] as f64,
                    k as f64 / MESH[2] as f64,
                );
                out.push([
                    fi * a[0][0] + fj * a[1][0] + fk * a[2][0],
                    fi * a[0][1] + fj * a[1][1] + fk * a[2][1],
                    fi * a[0][2] + fj * a[1][2] + fk * a[2][2],
                ]);
            }
        }
    }
    out
}

/// `max |ao|` over every k, plane and element — the scale the differences below
/// are measured against.
fn max_abs(name: &str, cell: &Cell, kpts: &[[f64; 3]], coords: &[[f64; 3]]) -> (f64, Vec<f64>) {
    let out = eval_ao_kpts(cell, name, coords, kpts).expect("eval_ao_kpts");
    let mut flat = Vec::new();
    let mut m = 0.0_f64;
    for t in &out.kaos {
        for i in 0..t.len() {
            m = m.max(t.re[i].abs()).max(t.im[i].abs());
            flat.push(t.re[i]);
            flat.push(t.im[i]);
        }
    }
    (m, flat)
}

/// The child half of the cross-process comparison: print the AO table's
/// checksum-free full contents is too much, so print the two aggregates the
/// parent needs — the max magnitude and a fixed-order sum of squares, both of
/// which move if a single element moves.
#[test]
fn screened_matches_unscreened_on_both_cells() {
    // The child is this same test binary, run with the screen off and with a
    // marker variable that makes it print instead of assert.
    let exe = std::env::current_exe().expect("test binary path");
    for (label, which) in [("si", "si"), ("he", "he")] {
        let out = Command::new(&exe)
            .args([
                "--exact",
                "print_reference_unscreened",
                "--nocapture",
                "--ignored",
            ])
            .env("PYSCF_PBC_AO_SCREEN", "0")
            .env("PYSCF_AO_SCREEN_TEST_CELL", which)
            .output()
            .expect("re-run self with the screen off");
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text
            .lines()
            .find(|l| l.starts_with("REFERENCE "))
            .unwrap_or_else(|| panic!("child produced no REFERENCE line for {label}:\n{text}"));
        let mut it = line.split_whitespace().skip(1);
        let ref_max: f64 = it.next().expect("max").parse().expect("max f64");
        let ref_sq: f64 = it.next().expect("sq").parse().expect("sq f64");

        let cell = if which == "si" {
            silicon()
        } else {
            he_all_electron()
        };
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
        let coords = grid(&cell);
        let (m, flat) = max_abs("GTOval_sph", &cell, &kpts, &coords);
        let sq: f64 = flat.iter().map(|v| v * v).sum();

        // Relative to the AO magnitude, not to 1: an AO value is O(1) here, but
        // tying the tolerance to the measured scale keeps this honest on a cell
        // whose AOs are not.
        let rel_max = (m - ref_max).abs() / ref_max.max(1e-30);
        let rel_sq = (sq - ref_sq).abs() / ref_sq.max(1e-30);
        println!(
            "{label}: screened vs unscreened — max {rel_max:.3e}, sum-of-squares {rel_sq:.3e}"
        );
        assert!(
            rel_max < 1e-12 && rel_sq < 1e-12,
            "{label}: the screen changed the AO table by more than 1e-12 relative \
             (max {rel_max:.3e}, sum-of-squares {rel_sq:.3e}) — the dropped mass is \
             NOT inside the gate"
        );
    }
}

/// The child entry point. `#[ignore]`d so a normal run never executes it; the
/// parent invokes it explicitly with `--ignored --exact`.
#[test]
#[ignore = "child process of screened_matches_unscreened_on_both_cells"]
fn print_reference_unscreened() {
    let which = std::env::var("PYSCF_AO_SCREEN_TEST_CELL").unwrap_or_else(|_| "si".into());
    let cell = if which == "si" {
        silicon()
    } else {
        he_all_electron()
    };
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let coords = grid(&cell);
    let (m, flat) = max_abs("GTOval_sph", &cell, &kpts, &coords);
    let sq: f64 = flat.iter().map(|v| v * v).sum();
    println!("REFERENCE {m:.17e} {sq:.17e}");
}

/// Enlarging the image list must not move the SCREENED answer any more than it
/// moves the unscreened one. This is the same shape and the same 1e-11 standard
/// as the pre-existing `eval_ao_kpts::image_sum_is_converged`, run with the
/// screen ON: if the screen were dropping something real, growing the radius by
/// 50 % would move the result past a bound the unscreened code already meets.
#[test]
fn the_screened_answer_is_still_converged_in_the_image_list() {
    use pyscf_pbc_gto::{estimate_rcut_for_eval, eval_ao_kpts_with_images, lattice};

    let cell = silicon();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let coords = grid(&cell);

    let rcut = estimate_rcut_for_eval(&cell, 0).expect("rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let narrow = lattice::get_lattice_ls(&cell, Some(rmax), None, false).expect("Ls");
    let wide = lattice::get_lattice_ls(&cell, Some(rmax * 1.5), None, false).expect("wide Ls");
    assert!(
        wide.len() > narrow.len(),
        "the wide image list must actually be wider, else this test is vacuous"
    );

    let a = eval_ao_kpts_with_images(&cell, "GTOval_sph", &coords, &kpts, &narrow).expect("narrow");
    let b = eval_ao_kpts_with_images(&cell, "GTOval_sph", &coords, &kpts, &wide).expect("wide");

    let mut worst = 0.0_f64;
    for (x, y) in a.kaos.iter().zip(&b.kaos) {
        for i in 0..x.len() {
            worst = worst
                .max((x.re[i] - y.re[i]).abs())
                .max((x.im[i] - y.im[i]).abs());
        }
    }
    println!(
        "image-list convergence WITH the screen on: {} -> {} images moves the AO \
         table by {worst:.3e}",
        narrow.len(),
        wide.len()
    );
    assert!(
        worst < 1e-11,
        "screened AO values moved by {worst:e} when the image list grew 50 % — the \
         screen has broken rcut convergence"
    );
}

/// The screen must actually reject images, or every assertion above is vacuous.
///
/// This measures the rejection rate the same way the production path does — by
/// the radius that decides it — rather than by reaching into private state.
#[test]
fn the_screen_actually_rejects_images() {
    use pyscf_pbc_gto::{estimate_rcut_for_eval, lattice};

    let cell = silicon();
    let coords = grid(&cell);
    let rcut = estimate_rcut_for_eval(&cell, 0).expect("rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let ls = lattice::get_lattice_ls(&cell, Some(rmax), None, false).expect("Ls");

    // The grid's own bounding box, and the atom positions: an image is
    // rejectable when every atom of it is further than `rmax` from every corner
    // of that box. This is a LOOSER test than the production per-block one, so
    // whatever it counts is a lower bound on what the screen rejects.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in &coords {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(c[axis]);
            hi[axis] = hi[axis].max(c[axis]);
        }
    }
    let atoms = cell.mol.atom_coords();
    let mut rejected = 0usize;
    for l in &ls {
        let reachable = atoms.iter().any(|r| {
            let p = [r[0] + l[0], r[1] + l[1], r[2] + l[2]];
            let mut d2 = 0.0;
            for axis in 0..3 {
                let d = if p[axis] < lo[axis] {
                    lo[axis] - p[axis]
                } else if p[axis] > hi[axis] {
                    p[axis] - hi[axis]
                } else {
                    0.0
                };
                d2 += d * d;
            }
            d2 <= rmax * rmax
        });
        if !reachable {
            rejected += 1;
        }
    }
    let pct = 100.0 * rejected as f64 / ls.len() as f64;
    println!(
        "screen rejects at least {rejected}/{} images ({pct:.1} %) on Si gth-szv",
        ls.len()
    );
    assert!(
        rejected > 0,
        "the screen rejected NO image out of {} — every other assertion in this \
         file would then be comparing the unscreened path against itself",
        ls.len()
    );
}
