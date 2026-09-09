//! K-01 — the k-point-resolved multigrid gate (`multigrid::kpts`).
//!
//! Phase 17 shipped multigrid at the gamma point only
//! (`17-VERIFICATION.md` §10.12). This suite gates the k-point
//! generalisation, and it is built around one fact: **the forward direction
//! cannot detect a Bloch-phase sign error and the reverse direction can.**
//! The pair list runs over every ordered `(pi, pj)` and its image list is
//! closed under `L -> -L`, so `(μ, ν, L)` and `(ν, μ, -L)` contribute
//! complex conjugates whose sum is real under EITHER sign convention. A
//! density-only gate would pass with the potential silently returning `V_-k`
//! for `V_k`. So the gates here are, in order of what they can catch:
//!
//! 1. `gamma_is_bit_identical_to_the_gamma_only_route` — `to_bits()`, not a
//!    tolerance. This is what licenses replacing the gamma path.
//! 2. `nelec_matches_the_k_resolved_trace` — `∫ρ = Σ_k w_k Re Tr(D_k S_k)`,
//!    the k-general form of 17-11's `int_rho_matches_tr_dm_s`. Catches a
//!    forward phase that is wrong in MAGNITUDE (a dropped image, a wrong
//!    weight), not one that is wrong in sign.
//! 3. `vj_matches_fftdf_per_k` — per k-point, against a route with no
//!    multigrid in it at all. **This is the sign gate**, and it compares
//!    every k-point's complex matrix rather than an energy, because an
//!    energy is blind to `V_k <-> V_-k` on a symmetric k-set.
//! 4. `veff_is_hermitian_per_k` — the structural invariant the SCF's
//!    generalised eigensolve depends on.
//! 5. `band_kpts_subset_matches_the_full_evaluation` — `kpts_band` support,
//!    which is what the k-symmetric drivers actually call.

mod common;

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::multigrid::MultiGridNumInt2;
use pyscf_pbc_gto::{Cell, make_kpts_default};

/// Small enough to converge an SCF per test run, fine enough that the
/// multigrid level ladder is non-trivial.
const MESH: [usize; 3] = [15, 15, 15];

fn cell_at(mesh: [usize; 3]) -> Cell {
    let mut c = common::diamond();
    c.mesh = mesh;
    c
}

/// A converged k-resolved closed-shell density on `cell` at `nk`, through
/// the reference (non-multigrid) route — so the fixture never depends on the
/// code under test.
fn converged_dm(cell: &Cell, nk: [usize; 3], xc: &str) -> (Vec<[f64; 3]>, Vec<CTensor>) {
    let kpts = make_kpts_default(cell, nk).expect("k-mesh");
    let df = pyscf_pbc_df::Fftdf::with_mesh(cell.clone(), &kpts, cell.mesh).expect("FFTDF");
    let mf = pyscf_pbc_dft::krks::Krks::from_df(Box::new(df), xc).expect("KRKS");
    let cfg = pyscf_pbc_scf::KScfConfig {
        conv_tol: 1e-10,
        max_cycle: 60,
        ..pyscf_pbc_scf::KScfConfig::for_cell(cell)
    };
    let r = mf.kernel(&cfg).expect("KRKS kernel");
    assert!(r.converged, "fixture SCF did not converge");
    (kpts, r.dm[0].clone())
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_dev_ctensor(a: &CTensor, b: &CTensor) -> f64 {
    max_abs(&a.re, &b.re).max(max_abs(&a.im, &b.im))
}

// ---------------------------------------------------------------------------
// 1. The gamma point, bit for bit
// ---------------------------------------------------------------------------

/// The k-general route AT ONE GAMMA POINT must reproduce the gamma-only
/// route it generalises, to the bit — `w = 1`, `cos(0) = 1`, `sin(0) = 0`,
/// so `d = re·1.0 + im·0.0` is the same `f64` the gamma path multiplies, and
/// every launch downstream sees an identical `term_coef`.
///
/// A tolerance here would be a much weaker statement, and would let a
/// reordered contraction through.
#[test]
fn gamma_is_bit_identical_to_the_gamma_only_route() {
    let cell = cell_at([13, 13, 13]);
    let nao = cell.mol.nao_nr;
    let gamma = [[0.0, 0.0, 0.0]];

    let (_, dm_k) = converged_dm(&cell, [1, 1, 1], "lda,vwn");
    let dm_real = dm_k[0].re.clone();
    assert_eq!(dm_k.len(), 1);

    for xc in ["lda,vwn", "pbe,pbe"] {
        let ni = MultiGridNumInt2::new();
        let want = ni.nr_rks(&cell, xc, &dm_real).expect("gamma-only nr_rks");
        let got = ni
            .nr_rks_kpts(&cell, xc, &dm_k, &gamma, None)
            .expect("k-general nr_rks");

        assert_eq!(got.veff.len(), 1, "{xc}: one k-point in, one out");
        assert_eq!(
            got.nelec.to_bits(),
            want.nelec.to_bits(),
            "{xc}: nelec must be bit-identical"
        );
        assert_eq!(
            got.exc.to_bits(),
            want.exc.to_bits(),
            "{xc}: exc must be bit-identical"
        );
        assert_eq!(
            got.ecoul.to_bits(),
            want.ecoul.to_bits(),
            "{xc}: ecoul must be bit-identical"
        );
        for i in 0..nao * nao {
            assert_eq!(
                got.veff[0].re[i].to_bits(),
                want.veff[i].to_bits(),
                "{xc}: veff[{i}] must be bit-identical"
            );
            assert_eq!(
                got.veff[0].im[i], 0.0,
                "{xc}: veff[{i}] must be strictly real at gamma"
            );
        }
    }
}

/// The same statement for the Coulomb matrix alone, which has no XC in it
/// and so localises a failure to the collocation rather than the functional.
#[test]
fn gamma_get_j_is_bit_identical_to_the_gamma_only_route() {
    let cell = cell_at([13, 13, 13]);
    let gamma = [[0.0, 0.0, 0.0]];
    let (_, dm_k) = converged_dm(&cell, [1, 1, 1], "lda,vwn");

    let ni = MultiGridNumInt2::new();
    let want = ni.get_j(&cell, &dm_k[0].re).expect("gamma-only get_j");
    let got = ni
        .get_j_kpts(&cell, &dm_k, &gamma, None)
        .expect("k-general get_j");

    for (i, w) in want.iter().enumerate() {
        assert_eq!(got[0].re[i].to_bits(), w.to_bits(), "vj[{i}] bit-identical");
        assert_eq!(got[0].im[i], 0.0, "vj[{i}] strictly real at gamma");
    }
}

// ---------------------------------------------------------------------------
// 2. The forward direction: the k-resolved normalisation identity
// ---------------------------------------------------------------------------

/// `∫ρ = Σ_k w_k Re Tr(D_k S_k)` — exact in exact arithmetic for ANY density
/// matrix, so the residual is the collocation's own quadrature error and
/// nothing else. 17-11 established the gamma form; this is the k-general
/// one, and it is the gate that a dropped image or a wrong `1/nkpts` fails.
#[test]
fn nelec_matches_the_k_resolved_trace() {
    let cell = cell_at(MESH);
    let nao = cell.mol.nao_nr;

    for nk in [[1, 1, 2], [2, 2, 2]] {
        let (kpts, dm_k) = converged_dm(&cell, nk, "lda,vwn");
        let nkpts = kpts.len();

        // `Σ_k w_k Re Tr(D_k S_k)` through the analytic overlap.
        let ovlp = cell.pbc_intor("int1e_ovlp", &kpts, None, 0).expect("ovlp");
        let mut want = 0.0f64;
        for (s, d) in ovlp.kmats.iter().zip(&dm_k) {
            for i in 0..nao * nao {
                // Re Tr(D S) with both stored row-major: Tr(D·S) sums
                // D[i,j]·S[j,i], and S is Hermitian, so S[j,i] = conj(S[i,j]).
                want += d.re[i] * s.re[i] + d.im[i] * s.im[i];
            }
        }
        want /= nkpts as f64;

        let ni = MultiGridNumInt2::new();
        let got = ni
            .nr_rks_kpts(&cell, "lda,vwn", &dm_k, &kpts, None)
            .expect("nr_rks_kpts");

        let dev = (got.nelec - want).abs();
        println!(
            "nelec identity {nk:?}: multigrid {:.12}, Tr(DS) {want:.12}, |d| {dev:.3e}",
            got.nelec
        );
        assert!(
            dev < 1e-6,
            "{nk:?}: k-resolved ∫ρ {} vs Tr(DS) {want} — |d| {dev:.3e} is far above the \
             collocation floor, which means a phase or a weight is wrong, not a quadrature",
            got.nelec
        );
    }
}

// ---------------------------------------------------------------------------
// 3. The reverse direction: the sign gate
// ---------------------------------------------------------------------------

/// The Coulomb matrix per k-point against FFTDF's own `get_j_kpts` — a route
/// with no multigrid, no pair fusion and no lattice-image phase table in it.
///
/// **Per k-point and complex.** A k-summed or real-only comparison passes
/// with the reverse Bloch phase conjugated (which returns `V_-k` in `V_k`'s
/// slot), because the k-set is closed under `k -> -k` and the two matrices
/// are complex conjugates of one another.
/// **The k-meshes here are chosen so the matrices are actually complex.**
///
/// A Γ-centred `[2,2,2]` mesh on diamond samples nothing but time-reversal
/// invariant momenta, at which every one-electron matrix is REAL — measured
/// `|Im vj| ~ 1e-20`, i.e. zero. Gating the sign on such a mesh proves
/// nothing at all, which is exactly the trap this comment exists to record:
/// the first version of this test passed at `1e-20` in the imaginary part
/// and would have passed with the reverse phase conjugated.
///
/// An odd mesh (`[1,1,3]`, `[2,2,3]`) has `k = ±1/3` along an axis. Those
/// are not TRIMs, the matrices are genuinely complex, and the assertion
/// below refuses to run unless the reference confirms it.
const SIGN_MESHES: [[usize; 3]; 2] = [[1, 1, 3], [2, 2, 3]];

#[test]
fn vj_matches_fftdf_per_k() {
    let cell = cell_at(MESH);

    for nk in SIGN_MESHES {
        let (kpts, dm_k) = converged_dm(&cell, nk, "lda,vwn");
        let df = pyscf_pbc_df::Fftdf::with_mesh(cell.clone(), &kpts, cell.mesh).expect("FFTDF");
        let want = pyscf_pbc_df::fft_jk::get_j_kpts(
            &df,
            std::slice::from_ref(&dm_k),
            1,
            &kpts,
            None,
            None,
        )
        .expect("fftdf get_j_kpts");

        let ni = MultiGridNumInt2::new();
        let got = ni
            .get_j_kpts(&cell, &dm_k, &kpts, None)
            .expect("multigrid get_j_kpts");

        assert_eq!(got.len(), kpts.len());

        // The precondition WITHOUT which this test proves nothing: the
        // reference matrices must have a real imaginary part to compare.
        let im_scale = want[0]
            .iter()
            .flat_map(|m| m.im.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            im_scale > 1e-4,
            "{nk:?}: |Im vj|max is {im_scale:.3e} — this k-mesh samples only \
             time-reversal invariant momenta, so the sign of the reverse Bloch \
             phase is unobservable here and this gate is vacuous. Pick a mesh \
             with non-TRIM k-points."
        );
        println!("vj vs FFTDF {nk:?}: |Im vj|max in the reference = {im_scale:.3e}");

        let mut worst = 0.0f64;
        for k in 0..kpts.len() {
            let d = max_dev_ctensor(&got[k], &want[0][k]);
            worst = worst.max(d);
            // The imaginary parts are what carry the sign. Report them
            // separately so a failure says WHICH half moved.
            println!(
                "  {nk:?} k={k}: |d Re| {:.3e}, |d Im| {:.3e}",
                max_abs(&got[k].re, &want[0][k].re),
                max_abs(&got[k].im, &want[0][k].im),
            );
        }
        println!("vj vs FFTDF {nk:?}: worst |d| {worst:.3e}");
        assert!(
            worst < 1e-5,
            "{nk:?}: multigrid vj is {worst:.3e} from FFTDF's. A residual at the \
             1e-2..1e0 scale in the IMAGINARY part alone is the conjugated reverse \
             Bloch phase (V_-k returned for V_k), not a quadrature difference"
        );
    }
}

/// A conjugated reverse phase leaves `V_k` Hermitian, so this does NOT
/// replace the gate above — it catches a different class (a slot scattered
/// into the wrong `(ci, cj)` transpose), and the SCF's generalised
/// eigensolve depends on it outright.
#[test]
fn veff_is_hermitian_per_k() {
    let cell = cell_at(MESH);
    let nao = cell.mol.nao_nr;
    let (kpts, dm_k) = converged_dm(&cell, [2, 2, 3], "lda,vwn");

    let ni = MultiGridNumInt2::new();
    let got = ni
        .nr_rks_kpts(&cell, "lda,vwn", &dm_k, &kpts, None)
        .expect("nr_rks_kpts");

    let mut worst = 0.0f64;
    for (k, v) in got.veff.iter().enumerate() {
        for i in 0..nao {
            for j in 0..nao {
                let (a, b) = (i * nao + j, j * nao + i);
                worst = worst
                    .max((v.re[a] - v.re[b]).abs())
                    .max((v.im[a] + v.im[b]).abs());
            }
        }
        let _ = k;
    }
    println!("veff hermiticity: worst |d| {worst:.3e}");
    assert!(
        worst < 1e-10,
        "veff must be Hermitian per k, got {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// 4. `kpts_band` — what the k-symmetric drivers call
// ---------------------------------------------------------------------------

/// The potential at a SUBSET of the sampling k-points must be exactly the
/// same matrices the full evaluation returns for those k-points: the grid
/// integrals `I[t]` do not depend on the band list at all, only the phase
/// scatter does. Bit-exact, because it is literally the same arithmetic on
/// the same integrals.
///
/// This is the property `krks_ksymm` relies on — it hands a full-BZ density
/// and asks for an IBZ-length potential — and the reason the v2 driver's
/// `MultiGridBandUnsupported` refusal can go.
#[test]
fn band_kpts_subset_matches_the_full_evaluation() {
    let cell = cell_at(MESH);
    let (kpts, dm_k) = converged_dm(&cell, [2, 2, 3], "lda,vwn");

    let ni = MultiGridNumInt2::new();
    let full = ni
        .nr_rks_kpts(&cell, "lda,vwn", &dm_k, &kpts, None)
        .expect("full");

    // A strict subset, out of order, to prove the band list is honoured
    // positionally rather than by luck.
    let pick = [3usize, 0, 5];
    let band: Vec<[f64; 3]> = pick.iter().map(|&i| kpts[i]).collect();
    let sub = ni
        .nr_rks_kpts(&cell, "lda,vwn", &dm_k, &kpts, Some(&band))
        .expect("band");

    assert_eq!(sub.veff.len(), pick.len());
    assert_eq!(sub.nelec.to_bits(), full.nelec.to_bits());
    assert_eq!(sub.exc.to_bits(), full.exc.to_bits());
    for (b, &k) in pick.iter().enumerate() {
        for i in 0..cell.mol.nao_nr * cell.mol.nao_nr {
            assert_eq!(
                sub.veff[b].re[i].to_bits(),
                full.veff[k].re[i].to_bits(),
                "band {b} (k={k}) re[{i}]"
            );
            assert_eq!(
                sub.veff[b].im[i].to_bits(),
                full.veff[k].im[i].to_bits(),
                "band {b} (k={k}) im[{i}]"
            );
        }
    }
}
