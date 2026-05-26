//! GEOMOPT-05: HDF5 optimizer-state checkpoint round-trip + resume.
//!
//! This drives the native BFGS+RFO optimizer (07-04) through a partial run,
//! spills the live optimizer state (current geometry, trust radius, the BFGS
//! approximate Hessian, the accumulated internal-coordinate / gradient history,
//! and the step counter) to an HDF5 group via the `pyscf-chkfile` re-exported
//! `hdf5` alias (NO own `hdf5-metno` dep — the sole-owner discipline, D-07),
//! loads it back, asserts a byte-exact round-trip, and resumes the optimization
//! from the loaded state — asserting it reaches the SAME stationary point as an
//! uninterrupted run.
//!
//! This is the always-on structural/persistence gate: it needs NO SCF and NO
//! cintx grad integral (it reuses the self-contained internal-only analytic
//! harmonic `GradScanner` from `h2o_equilibrium.rs`), so it runs in every
//! `cargo test -p pyscf-geomopt`.

use pyscf_chkfile::hdf5;
use pyscf_core::{Energy, Mole};
use pyscf_geomopt::checkpoint::OptimizerState;
use pyscf_geomopt::{GeometryOptimizer, optimize, optimize_resume};
use pyscf_grad::GradScanner;
use pyscf_grad::scanner::{EnergyClosure, GradClosure};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

// ---- self-contained model (mirrors h2o_equilibrium.rs, internal-only PES) ----

const EQ_ROH: f64 = 1.81;
const EQ_THETA: f64 = 104.5; // degrees

fn perturbed_h2o() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0.0 0.0 0.0; H 1.30 1.30 0.0; H -1.30 1.30 0.0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: pyscf_core::Unit::Bohr,
        ..Default::default()
    })
    .expect("build perturbed H2O/STO-3G")
}

fn harmonic_scanner(kb: f64, ka: f64) -> GradScanner {
    let theta_eq = EQ_THETA.to_radians();
    let e_of = move |coords: &[[f64; 3]]| -> f64 {
        let r1 = dist(coords, 0, 1);
        let r2 = dist(coords, 0, 2);
        let th = angle_at(coords, 1, 0, 2);
        0.5 * kb * ((r1 - EQ_ROH).powi(2) + (r2 - EQ_ROH).powi(2)) + 0.5 * ka * (th - theta_eq).powi(2)
    };
    let e_energy = e_of;
    let energy: EnergyClosure = Box::new(move |mol: &Mole| Ok(Energy(e_energy(&mol.atom_coords()))));
    let grad: GradClosure = Box::new(move |mol: &Mole, _atmlst: Option<&[usize]>| {
        let coords = mol.atom_coords();
        let h = 1e-6;
        let mut de = vec![[0.0_f64; 3]; coords.len()];
        for a in 0..coords.len() {
            for c in 0..3 {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[a][c] += h;
                cm[a][c] -= h;
                de[a][c] = (e_of(&cp) - e_of(&cm)) / (2.0 * h);
            }
        }
        Ok(de)
    });
    GradScanner::new(energy, grad)
}

fn dist(coords: &[[f64; 3]], a: usize, b: usize) -> f64 {
    let dx = coords[a][0] - coords[b][0];
    let dy = coords[a][1] - coords[b][1];
    let dz = coords[a][2] - coords[b][2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle_at(coords: &[[f64; 3]], a: usize, apex: usize, c: usize) -> f64 {
    let u = [
        coords[a][0] - coords[apex][0],
        coords[a][1] - coords[apex][1],
        coords[a][2] - coords[apex][2],
    ];
    let v = [
        coords[c][0] - coords[apex][0],
        coords[c][1] - coords[apex][1],
        coords[c][2] - coords[apex][2],
    ];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (dot / (nu * nv)).clamp(-1.0, 1.0).acos()
}

fn bond(coords: &[[f64; 3]], a: usize, b: usize) -> f64 {
    dist(coords, a, b)
}

fn hoh_angle(coords: &[[f64; 3]]) -> f64 {
    angle_at(coords, 1, 0, 2)
}

/// A unique temp HDF5 path under the OS temp dir (no external package; just a
/// fresh file we create + remove ourselves).
fn temp_h5_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("geomopt_ckpt_{tag}_{pid}_{nanos}.h5"));
    p
}

#[test]
fn checkpoint_state_round_trips_byte_for_byte() {
    // Build a representative optimizer state directly (current geometry, trust
    // radius, the BFGS Hessian, the q/g_int history, and the step counter),
    // dump it to an HDF5 group, load it back, and assert byte-equality.
    let coords = vec![[0.1, 0.2, 0.3], [1.31, 1.29, 0.01], [-1.28, 1.30, -0.02]];
    let hessian = vec![
        1.0, 0.05, -0.02, 0.0, 0.05, 1.1, 0.03, 0.01, -0.02, 0.03, 0.9, 0.04, 0.0, 0.01, 0.04, 1.2,
    ];
    let nint = 4;
    let q = vec![1.81, 1.79, 1.82, 0.95];
    let g_int = vec![3.1e-3, -2.2e-3, 1.4e-3, -0.7e-3];
    let state = OptimizerState {
        coords: coords.clone(),
        trust: 0.1414213562373095,
        hessian: hessian.clone(),
        nint,
        n_updates: 3,
        step: 5,
        prev_q: Some(q.clone()),
        prev_g_int: Some(g_int.clone()),
        prev_e: Some(-76.0123456789),
        e_tot: -76.0234567891,
    };

    let path = temp_h5_path("roundtrip");
    {
        let file = hdf5::File::create(&path).expect("create temp HDF5 file");
        let group = file.create_group("opt_state").expect("create group");
        state.dump(&group).expect("dump optimizer state");
        file.flush().expect("flush");
    }
    let loaded = {
        let file = hdf5::File::open(&path).expect("open temp HDF5 file");
        let group = file.group("opt_state").expect("open group");
        OptimizerState::load(&group).expect("load optimizer state")
    };
    let _ = std::fs::remove_file(&path);

    // Byte-exact round-trip on every persisted field.
    assert_eq!(loaded.nint, state.nint);
    assert_eq!(loaded.n_updates, state.n_updates);
    assert_eq!(loaded.step, state.step);
    assert_eq!(loaded.coords.len(), state.coords.len());
    for (a, b) in loaded.coords.iter().zip(state.coords.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "coords must round-trip bit-exactly");
    }
    assert_eq!(loaded.trust.to_bits(), state.trust.to_bits());
    assert_eq!(loaded.e_tot.to_bits(), state.e_tot.to_bits());
    assert_eq!(loaded.hessian.len(), state.hessian.len());
    for (a, b) in loaded.hessian.iter().zip(state.hessian.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "Hessian must round-trip bit-exactly");
    }
    let lq = loaded.prev_q.expect("prev_q present");
    for (a, b) in lq.iter().zip(q.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "prev_q must round-trip bit-exactly");
    }
    let lg = loaded.prev_g_int.expect("prev_g_int present");
    for (a, b) in lg.iter().zip(g_int.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "prev_g_int must round-trip bit-exactly");
    }
    assert_eq!(
        loaded.prev_e.expect("prev_e present").to_bits(),
        state.prev_e.unwrap().to_bits()
    );
}

#[test]
fn resume_reaches_same_stationary_point_as_uninterrupted_run() {
    // 1. Reference: an uninterrupted run to convergence.
    let mol = perturbed_h2o();
    let scanner = harmonic_scanner(0.5, 0.2);
    let opt = GeometryOptimizer::new();
    let reference = optimize(&opt, &scanner, &mol).expect("uninterrupted optimize");
    assert!(reference.converged, "reference run must converge");

    // 2. Partial run: stop after a few steps, checkpoint to HDF5.
    let partial_opt = GeometryOptimizer {
        maxsteps: 3,
        ..GeometryOptimizer::new()
    };
    let path = temp_h5_path("resume");
    let partial = optimize(&partial_opt, &scanner, &mol).expect("partial optimize");
    assert!(
        !partial.converged,
        "the 3-step partial run must NOT yet be converged (so resume is exercised)"
    );

    // Spill the partial optimizer state to HDF5.
    {
        let file = hdf5::File::create(&path).expect("create resume HDF5 file");
        let group = file.create_group("opt_state").expect("create group");
        partial
            .state
            .as_ref()
            .expect("partial run must expose its optimizer state for checkpointing")
            .dump(&group)
            .expect("dump partial state");
        file.flush().expect("flush");
    }

    // 3. Resume: load the checkpoint and continue from it to convergence.
    let loaded = {
        let file = hdf5::File::open(&path).expect("open resume HDF5 file");
        let group = file.group("opt_state").expect("open group");
        OptimizerState::load(&group).expect("load resume state")
    };
    let _ = std::fs::remove_file(&path);

    let full_opt = GeometryOptimizer::new();
    let resumed = optimize_resume(&full_opt, &scanner, &mol, loaded).expect("resume optimize");
    assert!(
        resumed.converged,
        "the resumed run must reach the 5-criterion GAU convergence (got nsteps={})",
        resumed.nsteps
    );

    // 4. The resumed geometry must match the uninterrupted run's stationary
    //    point (same bond lengths + angle within chemical accuracy).
    let r_oh1 = bond(&resumed.coords, 0, 1);
    let r_oh2 = bond(&resumed.coords, 0, 2);
    let ref_oh1 = bond(&reference.coords, 0, 1);
    let ref_oh2 = bond(&reference.coords, 0, 2);
    assert!(
        (r_oh1 - ref_oh1).abs() < 1e-3 && (r_oh2 - ref_oh2).abs() < 1e-3,
        "resumed O–H bonds {r_oh1:.5},{r_oh2:.5} must match the uninterrupted run {ref_oh1:.5},{ref_oh2:.5}"
    );
    let r_angle = hoh_angle(&resumed.coords).to_degrees();
    let ref_angle = hoh_angle(&reference.coords).to_degrees();
    assert!(
        (r_angle - ref_angle).abs() < 0.1,
        "resumed H–O–H angle {r_angle:.3}° must match the uninterrupted run {ref_angle:.3}°"
    );
    // And the resumed stationary point is the physical H2O equilibrium.
    assert!(
        (r_oh1 - EQ_ROH).abs() < 0.02 && (r_angle - EQ_THETA).abs() < 1.0,
        "resumed run must land at the H2O equilibrium (O–H ~1.81 Bohr, ∠ ~104.5°)"
    );
}

#[test]
fn corrupt_checkpoint_fails_cleanly_not_silently(/* T-07-19 */) {
    // A checkpoint whose persisted nint disagrees with the Hessian length must
    // fail with a clear error on load — never resume from garbage.
    let path = temp_h5_path("corrupt");
    {
        let file = hdf5::File::create(&path).expect("create corrupt HDF5 file");
        let group = file.create_group("opt_state").expect("create group");
        // Write an internally-inconsistent state: nint=4 but a 3x3 Hessian.
        let bad = OptimizerState {
            coords: vec![[0.0, 0.0, 0.0]],
            trust: 0.1,
            hessian: vec![1.0; 9], // 3x3, inconsistent with nint=4 below
            nint: 4,
            n_updates: 0,
            step: 1,
            prev_q: None,
            prev_g_int: None,
            prev_e: None,
            e_tot: 0.0,
        };
        // dump must itself reject the inconsistency (shape validation on write).
        let dumped = bad.dump(&group);
        // Either dump rejects it, or a later load must — assert at least one does.
        if dumped.is_ok() {
            file.flush().expect("flush");
        }
    }
    if std::path::Path::new(&path).exists() {
        let file = hdf5::File::open(&path).expect("open corrupt HDF5 file");
        if let Ok(group) = file.group("opt_state") {
            let loaded = OptimizerState::load(&group);
            assert!(
                loaded.is_err(),
                "a corrupt/inconsistent checkpoint must fail cleanly on load (T-07-19), not resume from garbage"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}
