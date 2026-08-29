//! Oracle-free gates for the periodic DFT crate (D-PBC-19).
//!
//! Nothing here runs Python. These are identities the code must satisfy on its
//! own terms, and they are the tests that actually catch defects: the upstream
//! gate in `gate.rs` tells you THAT a number moved, these tell you WHICH
//! invariant broke.
//!
//! The strongest is [`periodic_vxc_is_the_derivative_of_exc`]: `nr_rks` returns
//! `E_xc` and `V_xc` from the same density, so differentiating the returned
//! energy numerically must reproduce the returned matrix.

mod common;

use common::{diamond, he_all_electron, silicon};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::kgks::Kgks;
use pyscf_pbc_dft::kroks::Kroks;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_dft::xc::XcType;
use pyscf_pbc_gto::{make_kpts_default, Cell};
use pyscf_pbc_scf::types::KDms;
use pyscf_pbc_scf::KScfConfig;

/// A small FFT mesh: every test here is about STRUCTURE, and structure is
/// mesh-independent.
const MESH: [usize; 3] = [11, 11, 11];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

fn krks(cell: Cell, nk: [usize; 3], xc: &str) -> Krks {
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    Krks::from_df(Box::new(df), xc).expect("KRKS")
}

/// A converged closed-shell density on `cell` at `nk`, plus the pieces the
/// numint tests need.
fn converged(cell: &Cell, nk: [usize; 3], xc: &str) -> (Vec<[f64; 3]>, KDms, PeriodicGrids) {
    let mf = krks(cell.clone(), nk, xc);
    let r = mf.kernel(&tight()).expect("KRKS");
    assert!(r.converged, "fixture SCF did not converge");
    let kpts = make_kpts_default(cell, nk).expect("k-mesh");
    let grids = PeriodicGrids::uniform(cell, Some(MESH)).expect("uniform grids");
    (kpts, r.dm, grids)
}

// ---------------------------------------------------------------------------
// The derivative identity — the one that finds real defects
// ---------------------------------------------------------------------------

/// `V_xc = ∂E_xc/∂D`, at every k-point, for LDA and for GGA.
///
/// `nr_rks` returns `E_xc` and `V_xc` computed from the same density, so a
/// central difference of the returned energy along a Hermitian direction `Δ`
/// must reproduce the returned matrix contracted against `Δ`.
///
/// # The `1/N_k` asymmetry is the point
///
/// `ρ = (1/N_k) Σ_k ρ_k` carries the BZ average, but `V^k` does NOT — upstream
/// leaves the `1/N_k` on the density side alone (`numint.py:1172`). So
///
/// ```text
/// ∂E_xc/∂D^k[μν] = V^k[μν] / N_k
/// ```
///
/// and the analytic side needs that factor. Getting it wrong is exactly the
/// class of bug this test exists to catch, so it is spelled out rather than
/// absorbed into a tolerance.
fn vxc_derivative_identity(xc: &str, tol: f64) {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let (kpts, dm, grids) = converged(&cell, [2, 2, 2], xc);
    let nkpts = kpts.len();
    let ni = KNumInt::new(&kpts);

    // A fixed, reproducible Hermitian direction. A deterministic LCG keeps the
    // test independent of the `rand` crate and of thread scheduling.
    let mut seed = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let delta: Vec<CTensor> = (0..nkpts)
        .map(|_| {
            let mut re = vec![0.0; nao * nao];
            let mut im = vec![0.0; nao * nao];
            for i in 0..nao {
                for j in i..nao {
                    let (a, b) = (next(), if i == j { 0.0 } else { next() });
                    re[i * nao + j] = a;
                    re[j * nao + i] = a;
                    im[i * nao + j] = b;
                    im[j * nao + i] = -b;
                }
            }
            CTensor::from_planes(re, im)
        })
        .collect();

    let shifted = |eps: f64| -> KDms {
        vec![(0..nkpts)
            .map(|k| {
                let mut m = dm[0][k].clone();
                for i in 0..nao * nao {
                    m.re[i] += eps * delta[k].re[i];
                    m.im[i] += eps * delta[k].im[i];
                }
                m
            })
            .collect()]
    };

    let exc = |eps: f64| -> f64 {
        ni.nr_rks(&cell, &grids, xc, &shifted(eps), 1, None)
            .expect("nr_rks")
            .excsum[0]
    };

    // A step small enough that the cubic term is below the tolerance and large
    // enough that the difference is not eaten by cancellation.
    let eps = 1e-5;
    let fd = (exc(eps) - exc(-eps)) / (2.0 * eps);

    let r = ni
        .nr_rks(&cell, &grids, xc, &dm, 1, None)
        .expect("nr_rks");
    // Re Tr[V^k Δ^k], summed over k, divided by N_k.
    let mut an = 0.0;
    for k in 0..nkpts {
        let (v, d) = (&r.vmat[0][k], &delta[k]);
        for i in 0..nao {
            for j in 0..nao {
                // Tr[V D] = Σ_ij V[i,j] D[j,i]; both are Hermitian.
                an += v.re[i * nao + j] * d.re[j * nao + i]
                    - v.im[i * nao + j] * d.im[j * nao + i];
            }
        }
    }
    an /= nkpts as f64;

    let rel = (fd - an).abs() / an.abs().max(1e-12);
    println!("V_xc = dE_xc/dD  [{xc}]: fd {fd:.12e}  analytic {an:.12e}  rel {rel:.3e}");
    assert!(
        rel < tol,
        "{xc}: V_xc does not reproduce dE_xc/dD — relative error {rel:e} exceeds {tol:e}"
    );
}

#[test]
fn periodic_vxc_is_the_derivative_of_exc_lda() {
    vxc_derivative_identity("lda,vwn", 1e-6);
}

#[test]
fn periodic_vxc_is_the_derivative_of_exc_gga() {
    vxc_derivative_identity("pbe", 1e-6);
}

// ---------------------------------------------------------------------------
// Structural invariants of the numerical integration
// ---------------------------------------------------------------------------

/// `∫ ρ` over the cell is the electron count, and it CONVERGES to it as the
/// uniform mesh is refined.
///
/// Pins the AO layout, the Bloch phase, the `1/N_k` average and the quadrature
/// weight all at once — get any one of them wrong and this is not an integer at
/// any mesh. Asserting convergence rather than one tolerance at one mesh is the
/// stronger statement: a missing `1/N_k` would sit at a constant wrong value no
/// matter how fine the grid, which a single-mesh tolerance could hide behind a
/// loose bound.
#[test]
fn integrated_density_converges_to_the_electron_count() {
    let cell = diamond();
    let nelec = cell.mol.nelectron as f64;
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let mf = krks(cell.clone(), [2, 2, 2], "lda,vwn");
    let dm = mf.kernel(&tight()).expect("KRKS").dm;
    let ni = KNumInt::new(&kpts);

    let mut errs = Vec::new();
    for m in [11usize, 21, 31] {
        let grids = PeriodicGrids::uniform(&cell, Some([m, m, m])).expect("grids");
        let n = ni
            .nr_rks(&cell, &grids, "lda,vwn", &dm, 1, None)
            .expect("nr_rks")
            .nelec[0];
        let d = (n - nelec).abs();
        println!("mesh {m:>3}: integrated rho = {n:.12}  |delta| = {d:.3e}");
        errs.push(d);
    }
    // The error must COLLAPSE as the mesh is refined — by mesh 21 it is seven
    // orders below mesh 11 — and then it plateaus, because what is left is not
    // quadrature error at all but the convergence of the density matrix itself
    // (which came from a finite-mesh SCF). So the assertion is "it converges,
    // and it converges to the right integer", not strict monotonicity: a
    // plateau at the floor is the expected shape, while a MISSING `1/N_k` would
    // sit at a constant WRONG value that no refinement moves.
    assert!(
        errs[1] < errs[0] * 1e-4,
        "the quadrature error does not collapse with mesh: {errs:?} — a constant \
         offset here is a normalisation defect, not a convergence one"
    );
    assert!(
        errs[2] < 1e-9,
        "at mesh 31 the integrated density is still {:e} from {nelec}",
        errs[2]
    );
}

/// The imaginary part `eval_rho` drops is NOISE, not signal. Upstream discards
/// it silently (`numint.py:361`); this port records it so the claim is testable.
#[test]
fn periodic_density_is_real() {
    let cell = diamond();
    let (kpts, dm, grids) = converged(&cell, [2, 2, 2], "lda,vwn");
    let ni = KNumInt::new(&kpts);
    ni.nr_rks(&cell, &grids, "lda,vwn", &dm, 1, None)
        .expect("nr_rks");
    let imag = ni.last_rho_imag();
    println!("max |Im rho| = {imag:.3e}");
    assert!(imag < 1e-10, "the discarded imaginary density is {imag:e}, which is signal");
}

/// `V_xc^k` is Hermitian at every k-point — the `V + V^H` symmetrisation.
#[test]
fn vxc_is_hermitian() {
    for xc in ["lda,vwn", "pbe"] {
        let cell = diamond();
        let nao = cell.mol.nao_nr;
        let (kpts, dm, grids) = converged(&cell, [2, 2, 2], xc);
        let ni = KNumInt::new(&kpts);
        let r = ni.nr_rks(&cell, &grids, xc, &dm, 1, None).expect("nr_rks");
        for (k, v) in r.vmat[0].iter().enumerate() {
            let mut w = 0.0_f64;
            for i in 0..nao {
                for j in 0..nao {
                    w = w.max((v.re[i * nao + j] - v.re[j * nao + i]).abs());
                    w = w.max((v.im[i * nao + j] + v.im[j * nao + i]).abs());
                }
            }
            assert!(w < 1e-12, "{xc}: V_xc at k={k} is not Hermitian ({w:e})");
        }
    }
}

/// The grid block size is an IMPLEMENTATION detail. Swinging the memory budget
/// over four orders of magnitude changes the block partition completely and
/// must not move the energy.
#[test]
fn krks_energy_is_independent_of_the_grid_block_size() {
    let cell = diamond();
    let (kpts, dm, grids) = converged(&cell, [2, 2, 2], "pbe");
    let mut seen: Vec<f64> = Vec::new();
    for mb in [0.5_f64, 40.0, 2000.0] {
        let mut ni = KNumInt::new(&kpts);
        ni.max_memory = mb;
        let blocks = ni.block_ranges(grids.size(), XcType::Gga, kpts.len()).len();
        let r = ni.nr_rks(&cell, &grids, "pbe", &dm, 1, None).expect("nr_rks");
        println!("max_memory {mb:>7} MB -> {blocks:>4} blocks, E_xc = {:.15}", r.excsum[0]);
        seen.push(r.excsum[0]);
    }
    let spread = seen
        .iter()
        .map(|e| (e - seen[0]).abs())
        .fold(0.0_f64, f64::max);
    assert!(spread < 1e-10, "E_xc moved by {spread:e} across block sizes");
}

// ---------------------------------------------------------------------------
// The driver family collapses correctly
// ---------------------------------------------------------------------------

/// `KUKS` on a closed-shell cell must reproduce `KRKS`. The two share no code
/// across `nr_uks`/`nr_rks`, the full-vs-half `vk`, and the cross-spin
/// assembly, so every open-shell factor has to collapse for this to hold.
#[test]
fn kuks_reproduces_krks_on_a_closed_shell_cell() {
    let cell = diamond();
    let a = krks(cell.clone(), [2, 2, 2], "pbe")
        .kernel(&tight())
        .expect("KRKS");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    let b = Kuks::from_df(Box::new(df), "pbe")
        .expect("KUKS")
        .kernel(&tight())
        .expect("KUKS");
    let d = (a.e_tot - b.e_tot).abs();
    println!("KRKS {:.15}  KUKS {:.15}  delta {d:.3e}", a.e_tot, b.e_tot);
    assert!(d < 1e-9, "KUKS does not reproduce KRKS: delta {d:e}");
}

/// `KROKS` reduces to `KRKS` when `na == nb` — the Roothaan effective Fock.
#[test]
fn kroks_reproduces_krks_on_a_closed_shell_cell() {
    let cell = diamond();
    let a = krks(cell.clone(), [2, 2, 2], "pbe")
        .kernel(&tight())
        .expect("KRKS");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    let b = Kroks::from_df(Box::new(df), "pbe")
        .expect("KROKS")
        .kernel(&tight())
        .expect("KROKS");
    let d = (a.e_tot - b.e_tot).abs();
    println!("KRKS {:.15}  KROKS {:.15}  delta {d:.3e}", a.e_tot, b.e_tot);
    assert!(d < 1e-9, "KROKS does not reproduce KRKS: delta {d:e}");
}

/// `KGKS` in a collinear state reproduces `KRKS` — the 2-component block
/// structure and its `J` assembly.
#[test]
fn kgks_collinear_reproduces_krks() {
    let cell = diamond();
    let a = krks(cell.clone(), [2, 2, 2], "lda,vwn")
        .kernel(&tight())
        .expect("KRKS");
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    let b = Kgks::from_df(Box::new(df), "lda,vwn")
        .expect("KGKS")
        .kernel(&tight())
        .expect("KGKS");
    let d = (a.e_tot - b.e_tot).abs();
    println!("KRKS {:.15}  KGKS {:.15}  delta {d:.3e}", a.e_tot, b.e_tot);
    assert!(d < 1e-8, "KGKS does not reproduce KRKS: delta {d:e}");
}

// ---------------------------------------------------------------------------
// What the crate REFUSES, and why
// ---------------------------------------------------------------------------

/// Meta-GGA is refused rather than approximated.
///
/// The periodic AO evaluator ships value + `deriv1` only, so `tau` cannot be
/// formed, and `XcType::of` carries an explicit `Family::Mgga` arm that says so.
/// Today that arm is unreachable: the mapped xcfun corpus
/// (`xcfun_id_to_name`) tops out at GGA, so a meta-GGA NAME is rejected one
/// layer earlier, by the parser. Both are refusals — this test pins that the
/// refusal happens at all, for several meta-GGA names, rather than the string
/// one particular layer produces. If the corpus later grows a meta-GGA id, the
/// message changes and the test still holds.
#[test]
fn meta_gga_is_refused() {
    for xc in ["tpss", "scan", "m06-l", "revtpss"] {
        let e = XcType::of(xc).expect_err("meta-GGA must be refused, not approximated");
        println!("{xc:>8} -> {e}");
    }
}

/// A non-Hermitian density is refused rather than silently answered with the
/// Hermitian result — only upstream's `hermi = 1` branch of `eval_rho` is
/// implemented.
#[test]
fn non_hermitian_density_is_refused() {
    let cell = diamond();
    let (kpts, dm, grids) = converged(&cell, [2, 2, 2], "lda,vwn");
    let ni = KNumInt::new(&kpts);
    let e = ni
        .nr_rks(&cell, &grids, "lda,vwn", &dm, 0, None)
        .expect_err("hermi != 1 must be refused");
    println!("{e}");
}

// ---------------------------------------------------------------------------
// The all-electron path
// ---------------------------------------------------------------------------

/// The all-electron cell (no pseudopotential) converges and integrates to its
/// own electron count — the fixture the upstream gate leans on.
#[test]
fn he_all_electron_krks_converges_and_integrates() {
    let cell = he_all_electron();
    let nelec = cell.mol.nelectron as f64;
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let dm = krks(cell.clone(), [2, 2, 2], "pbe")
        .kernel(&tight())
        .expect("KRKS")
        .dm;
    let ni = KNumInt::new(&kpts);
    // At the gate mesh, where the all-electron cusp is actually resolved.
    let grids = PeriodicGrids::uniform(&cell, Some([31, 31, 31])).expect("grids");
    let r = ni.nr_rks(&cell, &grids, "pbe", &dm, 1, None).expect("nr_rks");
    let d = (r.nelec[0] - nelec).abs();
    println!("He-fcc integrated rho = {:.12} (expected {nelec}), delta {d:.3e}", r.nelec[0]);
    assert!(d < 1e-8, "He-fcc integrated density is {d:e} from {nelec}");
}

/// Silicon under PBE converges — the gate cell, exercised without Python.
#[test]
fn silicon_krks_pbe_converges() {
    let r = krks(silicon(), [2, 2, 2], "pbe")
        .kernel(&tight())
        .expect("KRKS");
    assert!(r.converged, "Si/PBE did not converge in {} cycles", r.cycles);
    println!("KRKS(Si,2x2x2,PBE) @ mesh 11 = {:.15}", r.e_tot);
}
