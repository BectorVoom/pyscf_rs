//! D-05 / GEOMOPT-07: the self-contained H2O-equilibrium convergence gate.
//!
//! This drives the native BFGS+RFO optimizer through its full outer loop
//! (redundant internals → Wilson B-matrix → RFO step → trust-radius →
//! back-transform → `set_geom_` → 5-criterion convergence) on a PERTURBED
//! H2O, asserting it reaches the known equilibrium geometry. NO external
//! `geometric`/`pyberny` package is importable (GEOMOPT-01 self-contained
//! discipline).
//!
//! ## Two arms (D-02 cintx-gating split, mirroring 07-03 `rhf_verify_fd`)
//!
//! 1. **Always-on (`equilibrium_via_model_scanner`):** the optimizer STRUCTURE
//!    is proven end-to-end against a self-contained ANALYTIC harmonic
//!    `GradScanner` whose `(e_tot, de)` pulls every atom toward a known
//!    H2O-equilibrium geometry. This needs NO SCF and NO cintx grad
//!    integrals, so it runs in every `cargo test -p pyscf-geomopt`. It proves
//!    the engine converges a perturbed geometry to the target stationary
//!    point within chemical accuracy and that the final gradient RMS drops
//!    below `convergence_grms` (3e-4 Eh/Bohr).
//!
//! 2. **`#[ignore]`'d (`equilibrium_via_rhf_gradient`):** the real
//!    RHF-`GradScanner`-driven arm. Per 07-01/07-03, the six gradient-integral
//!    families the RHF analytical gradient contracts (`int2e_ip1`,
//!    `int1e_ip{ovlp,kin,nuc,rinv}` + `with_rinv_at_nucleus`) are MISSING from
//!    cintx with no scheduled workstream, so `RhfGradients::kernel()` cannot
//!    yet produce a numeric gradient. This arm is `#[ignore]`'d behind that
//!    prerequisite and un-gates by dropping `#[ignore]` once the cintx
//!    grad-integral workstream lands the six families (the documented arm).

use pyscf_core::{Energy, Mole};
use pyscf_geomopt::{GeometryOptimizer, optimize};
use pyscf_grad::GradScanner;
use pyscf_grad::scanner::{EnergyClosure, GradClosure};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// The known H2O equilibrium target (Bohr): O at origin, two O–H bonds of
/// ~1.81 Bohr at the H–O–H angle ~104.5°, placed symmetrically in the xy-plane.
/// O–H = 1.81 Bohr (≈ 0.958 Å), half-angle 52.25°.
fn h2o_equilibrium() -> [[f64; 3]; 3] {
    let roh = 1.81_f64;
    let half = 52.25_f64.to_radians();
    [
        [0.0, 0.0, 0.0],                                   // O
        [roh * half.sin(), roh * half.cos(), 0.0],         // H
        [-roh * half.sin(), roh * half.cos(), 0.0],        // H
    ]
}

/// Build the perturbed-H2O `Mole` (bond/angle off equilibrium, minimal basis —
/// NO libxc). `unit = Bohr` so the optimizer's emitted geometry strings
/// round-trip through `set_geom_`.
fn perturbed_h2o() -> Mole {
    // O–H stretched to ~2.0 Bohr, angle widened — clearly off equilibrium.
    M(MoleBuildArgs {
        atom: AtomInput::String(
            "O 0.0 0.0 0.0; H 1.30 1.30 0.0; H -1.30 1.30 0.0".into(),
        ),
        basis: BasisInput::Name("sto-3g".into()),
        unit: pyscf_core::Unit::Bohr,
        ..Default::default()
    })
    .expect("build perturbed H2O/STO-3G")
}

/// The equilibrium O–H bond length (Bohr) and H–O–H angle (radians) the model
/// PES targets — derived from [`h2o_equilibrium`].
const EQ_ROH: f64 = 1.81;
const EQ_THETA: f64 = 104.5; // degrees

/// Construct a self-contained ANALYTIC harmonic `GradScanner` defined purely on
/// the INTERNAL coordinates of H2O:
///
/// ```text
///   E = ½ k_b [(r_OH1 − r_eq)² + (r_OH2 − r_eq)²] + ½ k_a (θ_HOH − θ_eq)²
/// ```
///
/// Because the PES is a function of bonds + angle only, it is exactly
/// translation- and rotation-invariant — its Cartesian gradient lives entirely
/// in the redundant-internal subspace (exactly like a real SCF gradient), so
/// the optimizer can drive the Cartesian gradient RMS to zero. Its minimum is
/// the H2O equilibrium (O–H = `EQ_ROH`, ∠ = `EQ_THETA`). This proves the engine
/// end-to-end WITHOUT any SCF or cintx grad integral. The Cartesian gradient is
/// the chain rule `∂E/∂x = Σ_q (∂E/∂q)(∂q/∂x)` computed by finite difference of
/// the internal-coordinate energy (a black-box scanner, exactly the seam the
/// optimizer consumes).
fn harmonic_scanner(kb: f64, ka: f64) -> GradScanner {
    let theta_eq = EQ_THETA.to_radians();

    // Internal-coordinate energy as a pure function of the geometry.
    let e_of = move |coords: &[[f64; 3]]| -> f64 {
        let r1 = dist(coords, 0, 1);
        let r2 = dist(coords, 0, 2);
        let th = angle_at(coords, 1, 0, 2);
        0.5 * kb * ((r1 - EQ_ROH).powi(2) + (r2 - EQ_ROH).powi(2))
            + 0.5 * ka * (th - theta_eq).powi(2)
    };

    let e_energy = e_of;
    let energy: EnergyClosure =
        Box::new(move |mol: &Mole| Ok(Energy(e_energy(&mol.atom_coords()))));

    let grad: GradClosure = Box::new(move |mol: &Mole, _atmlst: Option<&[usize]>| {
        // Central finite-difference Cartesian gradient of the internal-coord
        // energy (the analytic-reference gradient driving the optimizer).
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

#[test]
fn equilibrium_via_model_scanner() {
    // Always-on: the optimizer STRUCTURE (internals → B → RFO → back-transform
    // → set_geom → converge) driven by a self-contained analytic harmonic PES
    // whose minimum is the H2O equilibrium.
    let mol = perturbed_h2o();
    // Stiff-ish harmonic constants (Hartree per Bohr²/rad²) so the minimum is
    // well-defined; the PES is internal-only ⇒ rotation/translation invariant.
    let scanner = harmonic_scanner(0.5, 0.2);
    let opt = GeometryOptimizer::new();

    let result = optimize(&opt, &scanner, &mol).expect("optimize must not error");

    assert!(
        result.converged,
        "optimizer must reach the 5-criterion GAU convergence in {} steps (got nsteps={})",
        opt.maxsteps, result.nsteps
    );

    // The converged geometry must match the known H2O equilibrium within
    // chemical accuracy (O–H ~1.81 Bohr / ~0.958 Å, H–O–H ~104.5°).
    let target = h2o_equilibrium();
    let coords = &result.coords;
    // Compare bond lengths + angle (translation/rotation invariant), since the
    // analytic PES fixes absolute positions the geometry should match closely.
    let oh1 = bond(coords, 0, 1);
    let oh2 = bond(coords, 0, 2);
    let target_oh = 1.81_f64;
    assert!(
        (oh1 - target_oh).abs() < 0.02 && (oh2 - target_oh).abs() < 0.02,
        "O–H bonds must converge to ~1.81 Bohr, got {oh1:.4}, {oh2:.4}"
    );
    let angle = hoh_angle(coords);
    assert!(
        (angle.to_degrees() - 104.5).abs() < 1.0,
        "H–O–H angle must converge to ~104.5°, got {:.2}°",
        angle.to_degrees()
    );
    let _ = target;

    // Final gradient RMS must be below convergence_grms (3e-4 Eh/Bohr).
    let de = scanner
        .eval(&final_mol(&mol, coords), None)
        .expect("final gradient eval")
        .1;
    let grms = grad_rms(&de);
    assert!(
        grms < 3.0e-4,
        "final gradient RMS {grms:.2e} must be < convergence_grms 3e-4"
    );
}

#[test]
#[ignore = "RHF analytical gradient numeric is gated on the cintx grad-integral \
            workstream (int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} MISSING per \
            07-01/07-03); un-gate by dropping #[ignore] once cintx ships them"]
fn equilibrium_via_rhf_gradient() {
    // The real RHF-GradScanner-driven arm. Today RhfGradients::kernel() returns
    // a clean cintx-availability error (the six families are MISSING), so this
    // cannot run to a numeric optimum yet. When the cintx grad-integral
    // workstream lands, build the RHF reference + its grad scanner here and
    // assert the same H2O-equilibrium convergence as the model arm above.
    //
    // Wiring sketch (un-gates with the cintx workstream):
    //   let mol = perturbed_h2o();
    //   let rhf = RHF::new(mol.clone()); rhf.kernel()?;            // converged SCF
    //   let energy = pyscf_scf::scanner::as_scanner(&rhf);          // (Mole)->E
    //   let reference = RhfReference { mo_coeff, mo_energy, mo_occ, mol };
    //   let grad = move |m, atmlst| RhfGradients::new(reference.snapshot(m))
    //                                  .kernel(atmlst);             // (Mole,_)->de
    //   let scanner = GradScanner::new(energy, Box::new(grad));
    //   let result = optimize(&GeometryOptimizer::new(), &scanner, &mol)?;
    //   assert!(result.converged);
    let mol = perturbed_h2o();
    let _ = mol; // placeholder until the cintx grad-integral workstream lands
}

// ---- geometry helpers (self-contained; no external package) ----

fn bond(coords: &[[f64; 3]], a: usize, b: usize) -> f64 {
    let dx = coords[a][0] - coords[b][0];
    let dy = coords[a][1] - coords[b][1];
    let dz = coords[a][2] - coords[b][2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn hoh_angle(coords: &[[f64; 3]]) -> f64 {
    let u = [
        coords[1][0] - coords[0][0],
        coords[1][1] - coords[0][1],
        coords[1][2] - coords[0][2],
    ];
    let v = [
        coords[2][0] - coords[0][0],
        coords[2][1] - coords[0][1],
        coords[2][2] - coords[0][2],
    ];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (dot / (nu * nv)).clamp(-1.0, 1.0).acos()
}

fn grad_rms(de: &[[f64; 3]]) -> f64 {
    let mut s = 0.0;
    let n = de.len() * 3;
    for row in de {
        for &c in row {
            s += c * c;
        }
    }
    (s / n as f64).sqrt()
}

/// Build a Mole at the given converged geometry (Bohr) for the final gradient
/// re-evaluation.
fn final_mol(template: &Mole, coords: &[[f64; 3]]) -> Mole {
    let mut m = template.clone();
    let atom_str = template
        ._atom
        .iter()
        .zip(coords.iter())
        .map(|((s, _), c)| format!("{} {} {} {}", s, c[0], c[1], c[2]))
        .collect::<Vec<_>>()
        .join("; ");
    pyscf_gto::set_geom_(&mut m, &atom_str).expect("set_geom_ final");
    m
}
