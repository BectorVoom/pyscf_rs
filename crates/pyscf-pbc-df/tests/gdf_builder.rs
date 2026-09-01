//! `pyscf_pbc_df::gdf_builder` — the compensating-charge scheme (plan 14-02).
//!
//! Targets from `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/README.md`
//! (`params.py`). Re-run the script rather than re-deriving them.

mod common;

use pyscf_pbc_df::gdf_builder::eta::{
    ETA_MIN, estimate_eta_for_ke_cutoff, estimate_eta_min, estimate_ke_cutoff_for_eta, guess_eta,
};
use pyscf_pbc_df::gdf_builder::{auxbar, fuse_auxcell};
use pyscf_pbc_df::incore::make_modrho_basis;

/// `cell.make_kpts([n,n,n])` — the absolute Monkhorst-Pack k-points.
fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

// ---------------------------------------------------------------------------
// Task 3 — the eta estimators.
// ---------------------------------------------------------------------------

/// `_guess_eta` is called with the AUXCELL, not the orbital cell. The
/// gamma/2x2x2 split comes from its `ke_cutoff = 30 * nkpts^(-1/3)`.
#[test]
fn guess_eta_matches_upstream() {
    struct Case {
        label: &'static str,
        cell: pyscf_pbc_gto::Cell,
        km: [usize; 3],
        eta: f64,
        mesh: [usize; 3],
        ke: f64,
    }
    let cases = [
        Case {
            label: "diamond 2x2x2",
            cell: common::diamond(),
            km: [2, 2, 2],
            eta: 0.464_883_124_929_945_55,
            mesh: [11, 11, 11],
            ke: 21.721_883_440_437_864,
        },
        Case {
            label: "diamond gamma",
            cell: common::diamond(),
            km: [1, 1, 1],
            eta: 0.683_970_737_173_957_2,
            mesh: [13, 13, 13],
            ke: 31.279_512_154_230_53,
        },
        Case {
            label: "He-fcc 2x2x2",
            cell: common::he_all_electron(),
            km: [2, 2, 2],
            eta: 0.374_821_080_750_159_24,
            mesh: [9, 9, 9],
            ke: 19.653_483_258_876_75,
        },
    ];
    for c in &cases {
        let aux = make_modrho_basis(&c.cell, None, None).expect("auxcell");
        let k = kpts(&c.cell, c.km);
        let g = guess_eta(&aux.cell, &k, None).expect("guess_eta");
        assert!(
            (g.eta - c.eta).abs() < 1e-12,
            "{}: eta {} != {}",
            c.label,
            g.eta,
            c.eta
        );
        assert_eq!(g.mesh, c.mesh, "{}: mesh", c.label);
        assert!(
            (g.ke_cutoff - c.ke).abs() < 1e-10,
            "{}: ke_cutoff {} != {}",
            c.label,
            g.ke_cutoff,
            c.ke
        );
    }
}

#[test]
fn eta_estimators_match_upstream() {
    // diamond: estimate_eta_min = 0.1 (the ETA_MIN floor);
    //          estimate_ke_cutoff_for_eta(eta) = 23.868442944805754;
    //          estimate_eta_for_ke_cutoff(ke)  = 0.48784193871653286.
    let cell = common::diamond();
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");

    let emin = estimate_eta_min(&cell, None).expect("eta_min");
    assert!((emin - ETA_MIN).abs() < 1e-15, "estimate_eta_min = {emin}");

    let ke = estimate_ke_cutoff_for_eta(&cell, g.eta, None);
    assert!(
        (ke - 23.868_442_944_805_754).abs() < 1e-9,
        "estimate_ke_cutoff_for_eta = {ke}"
    );
    let e = estimate_eta_for_ke_cutoff(&cell, g.ke_cutoff, None);
    assert!(
        (e - 0.487_841_938_716_532_86).abs() < 1e-12,
        "estimate_eta_for_ke_cutoff = {e}"
    );

    // He-fcc: estimate_eta_min = 0.17166722884078006 — ABOVE the floor, which
    // is what makes it a real test of the formula rather than of the clamp.
    let he = common::he_all_electron();
    let emin = estimate_eta_min(&he, None).expect("eta_min");
    assert!(
        (emin - 0.171_667_228_840_780_06).abs() < 1e-12,
        "He estimate_eta_min = {emin}"
    );
}

/// `gdf_builder::estimate_rcut` is NOT `incore::estimate_rcut` — see the module
/// docs. Measured against the FUSED cell: 16.729034885581783 (diamond),
/// 10.750308556151602 (He-fcc).
#[test]
fn gdf_estimate_rcut_matches_upstream() {
    for (label, cell, want) in [
        ("diamond", common::diamond(), 16.729_034_885_581_783),
        ("he", common::he_all_electron(), 10.750_308_556_151_602),
    ] {
        let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
        let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");
        let fused = fuse_auxcell(&cell, None, g.eta).expect("fuse");
        let got = pyscf_pbc_df::gdf_builder::eta::estimate_rcut(&cell, &fused.fused.cell, None);
        assert!(
            (got - want).abs() < 1e-9,
            "{label}: gdf estimate_rcut = {got}, upstream {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 4 — the compensating charge.
// ---------------------------------------------------------------------------

#[test]
fn fused_cell_shape_matches_upstream() {
    for (label, cell, nao, nbas) in [
        ("diamond", common::diamond(), 126, 42),
        ("he", common::he_all_electron(), 32, 12),
    ] {
        let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
        let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");
        let f = fuse_auxcell(&cell, None, g.eta).expect("fuse");
        assert_eq!(f.nauxc(), nao, "{label}: fused_cell.nao");
        assert_eq!(f.fused.nbas(), nbas, "{label}: fused_cell.nbas");
        assert_eq!(f.naux(), aux.naux(), "{label}: naux unchanged");
        assert_eq!(f.aux_ao.len(), f.naux());
        assert_eq!(f.partner.len(), f.naux());
        // Every auxiliary AO must find a model-charge partner: `make_modchg_basis`
        // emits one shell per DISTINCT l, and every auxiliary l is distinct.
        assert!(
            f.partner.iter().all(Option::is_some),
            "{label}: some auxiliary AO has no model charge"
        );
    }
}

/// The auxiliary AOs of the fused cell must carry the SAME monopole as the
/// plain auxiliary cell — the fused cell is the auxiliary basis plus extra
/// shells, not a renormalisation of it.
#[test]
fn fusing_does_not_disturb_the_auxiliary_normalisation() {
    let cell = common::diamond();
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");
    let f = fuse_auxcell(&cell, None, g.eta).expect("fuse");
    for a in 0..f.naux() {
        let want = aux.modrho_scale[a];
        let got = f.fused.modrho_scale[f.aux_ao[a]];
        assert!(
            (got - want).abs() < 1e-12 * want.abs().max(1.0),
            "auxiliary AO {a}: fused scale {got} != auxcell scale {want}"
        );
    }
}

/// `fuse` on the identity block extracts the auxiliary rows and subtracts the
/// model-charge ones — the structural check that the index maps line up.
#[test]
fn fuse_rows_subtracts_the_model_charge() {
    let cell = common::he_all_electron();
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");
    let f = fuse_auxcell(&cell, None, g.eta).expect("fuse");
    let nauxc = f.nauxc();
    // A block whose value is the fused AO index itself.
    let block: Vec<f64> = (0..nauxc).map(|i| i as f64).collect();
    let out = f.fuse_rows(&block, 1);
    assert_eq!(out.len(), f.naux());
    for a in 0..f.naux() {
        let want = f.aux_ao[a] as f64 - f.partner[a].expect("partner") as f64;
        assert!((out[a] - want).abs() < 1e-15, "row {a}");
    }
    // `fuse_cols` must agree with `fuse_rows` on a 1-column block.
    let cols = f.fuse_cols(&block, 1);
    assert_eq!(cols, out);
}

/// `auxbar` — the background-charge interaction. Measured AFTER `fuse`:
/// 12 non-zeros / norm 0.23012787965177506 on diamond, 4 / 0.3187837520926407
/// on He-fcc.
#[test]
fn auxbar_matches_upstream() {
    for (label, cell, nnz, norm) in [
        (
            "diamond",
            common::diamond(),
            12usize,
            0.230_127_879_651_775_06,
        ),
        (
            "he",
            common::he_all_electron(),
            4usize,
            0.318_783_752_092_640_7,
        ),
    ] {
        let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
        let g = guess_eta(&aux.cell, &kpts(&cell, [2, 2, 2]), None).expect("guess_eta");
        let f = fuse_auxcell(&cell, None, g.eta).expect("fuse");
        let vbar = f.fuse_rows(&auxbar(&f.fused.cell), 1);
        assert_eq!(vbar.len(), f.naux());
        let got_nnz = vbar.iter().filter(|v| **v != 0.0).count();
        let got_norm = vbar.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_eq!(got_nnz, nnz, "{label}: auxbar non-zeros");
        assert!(
            (got_norm - norm).abs() < 1e-10,
            "{label}: |auxbar| = {got_norm}, upstream {norm}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5 — the 2-centre metric.
// ---------------------------------------------------------------------------

use pyscf_pbc_df::gdf_builder::j2c::{J2cTag, decompose_j2c, get_2c2e};

fn built(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> pyscf_pbc_df::gdf_builder::FusedCell {
    let aux = make_modrho_basis(cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &kpts(cell, km), None).expect("guess_eta");
    fuse_auxcell(cell, None, g.eta).expect("fuse")
}

/// `‖j2c(k=0)‖`, its spectrum and the decomposition route, all measured
/// upstream (`measurements/params.py`).
#[test]
fn j2c_fingerprint_matches_upstream() {
    for (label, cell, km, norm, wmin, wmax) in [
        (
            "diamond 2x2x2",
            common::diamond(),
            [2usize, 2, 2],
            9.774_955_865_744_985,
            3.171_124_433_357_167e-11,
            3.268_981_361_838_472,
        ),
        (
            "he 2x2x2",
            common::he_all_electron(),
            [2, 2, 2],
            10.064_640_251_330_108,
            6.637_682_717_009_374e-3,
            7.688_412_180_961_814,
        ),
    ] {
        let f = built(&cell, km);
        let naux = f.naux();
        let j2c = get_2c2e(&cell, &f, &[[0.0; 3]], None).expect("get_2c2e");
        assert_eq!(j2c.len(), 1);
        let m = &j2c[0];
        assert_eq!(m.re.len(), naux * naux);

        let got =
            m.re.iter()
                .zip(m.im.iter())
                .map(|(r, i)| r * r + i * i)
                .sum::<f64>()
                .sqrt();
        assert!(
            (got - norm).abs() < 1e-8 * norm,
            "{label}: |j2c| = {got}, upstream {norm}"
        );

        // At gamma the metric is REAL.
        let max_im = m.im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(max_im < 1e-10, "{label}: gamma j2c has |Im| = {max_im:e}");

        // Hermitian by construction — the symmetrisation is the last step.
        let mut asym = 0.0_f64;
        for p in 0..naux {
            for q in 0..naux {
                asym = asym.max((m.re[p * naux + q] - m.re[q * naux + p]).abs());
            }
        }
        assert!(asym < 1e-12, "{label}: j2c asymmetry {asym:e}");

        // The spectrum, and the route it sends `decompose_j2c` down.
        let cd = decompose_j2c(m, naux, false).expect("decompose");
        assert_eq!(cd.tag, J2cTag::Cd, "{label}: upstream reports j2ctag = CD");
        assert_eq!(cd.rank, naux);
        let _ = (wmin, wmax);
    }
}

/// Diamond's metric has `eig_min = 3.17e-11`, BELOW `linear_dep_threshold`, and
/// upstream still returns `CD` because Cholesky is tried first and succeeds. A
/// port that inspects the spectrum and pre-empts the eigen branch disagrees
/// with upstream on the flagship system — this test is what stops that.
#[test]
fn near_singular_metric_still_takes_the_cholesky_route() {
    let cell = common::diamond();
    let f = built(&cell, [2, 2, 2]);
    let naux = f.naux();
    let j2c = get_2c2e(&cell, &f, &[[0.0; 3]], None).expect("get_2c2e");
    let cd = decompose_j2c(&j2c[0], naux, false).expect("decompose");
    assert_eq!(cd.tag, J2cTag::Cd);
    // Forcing the eigen route must still give a usable factor.
    let eig = decompose_j2c(&j2c[0], naux, true).expect("decompose eig");
    assert_eq!(eig.tag, J2cTag::Eig);
    assert!(eig.rank <= naux);
    assert!(eig.rank > 0, "the eigen route dropped every vector");
}

/// **`V j2c Vᴴ = I` on the retained subspace** — the identity that DEFINES the
/// eigen factor, oracle-free.
///
/// `V = v[:, mask]ᴴ / sqrt(w[mask])`, so `V j2c Vᴴ = diag(w)/w = I_rank`. It
/// costs three matrix products and it is the only property of the factor that
/// matters: a transposed, permuted or mis-phased `V` still has the right
/// eigenvalues, the right rank and the right shape, and fails this.
///
/// **Plan 14-06 shipped only because this test did not exist.** `decompose_j2c`
/// read `zeigh_gen`'s COLUMN-MAJOR eigenvector buffer row-major, i.e. built the
/// factor from the transpose. `j2ctag` is `CD` on every system in
/// `measurements/params.py`, so no gate had ever reached the eigen branch;
/// MDF — `j2c_eig_always = True` (`mdf.py:365`) — was the first consumer, and
/// what it produced was **6.3e6 Ha** on He-fcc 2x2x2. This test now fails in
/// milliseconds where that took a converged SCF to notice.
#[test]
fn the_eigen_factor_inverts_the_metric_on_its_retained_subspace() {
    // The tolerance is per system and is a CONDITIONING floor, not slack:
    // `V j2c Vᴴ` is formed from `v/sqrt(w)`, so its error scales as
    // `eps / w_min(retained)`. He-fcc keeps every vector down to 6.6e-3;
    // diamond's spectrum runs to the 1e-10 `linear_dep_threshold`, seven orders
    // lower, and its residual is 3.09e-8 against He-fcc's 2.71e-14.
    for (label, cell, km, tol) in [
        ("He-fcc", common::he_all_electron(), [2usize, 2, 2], 1e-12),
        ("diamond", common::diamond(), [2, 2, 2], 1e-6),
    ] {
        let f = built(&cell, km);
        let naux = f.naux();
        let j2c = get_2c2e(&cell, &f, &[[0.0; 3]], None).expect("get_2c2e");
        let eig = decompose_j2c(&j2c[0], naux, true).expect("decompose eig");
        let r = eig.rank;
        assert!(r > 0, "{label}: the eigen route dropped every vector");

        // t = V · j2c   ->  (rank, naux)
        let mut tr = vec![0.0f64; r * naux];
        let mut ti = vec![0.0f64; r * naux];
        for a in 0..r {
            for k in 0..naux {
                let (vr, vi) = (eig.j2c.re[a * naux + k], eig.j2c.im[a * naux + k]);
                if vr == 0.0 && vi == 0.0 {
                    continue;
                }
                for q in 0..naux {
                    let (mr, mi) = (j2c[0].re[k * naux + q], j2c[0].im[k * naux + q]);
                    tr[a * naux + q] += vr * mr - vi * mi;
                    ti[a * naux + q] += vr * mi + vi * mr;
                }
            }
        }
        // out = t · Vᴴ   ->  (rank, rank), must be the identity
        let mut worst = 0.0f64;
        for a in 0..r {
            for b in 0..r {
                let (mut xr, mut xi) = (0.0f64, 0.0f64);
                for q in 0..naux {
                    let (ar, ai) = (tr[a * naux + q], ti[a * naux + q]);
                    // conj(V[b, q])
                    let (br, bi) = (eig.j2c.re[b * naux + q], -eig.j2c.im[b * naux + q]);
                    xr += ar * br - ai * bi;
                    xi += ar * bi + ai * br;
                }
                let want = f64::from(u8::from(a == b));
                worst = worst.max((xr - want).abs()).max(xi.abs());
            }
        }
        eprintln!("{label}: max|V j2c Vᴴ - I| = {worst:e} (rank {r} of {naux})");
        assert!(
            worst < tol,
            "{label}: the eigen factor does not invert the metric: {worst:e} \
             (tol {tol:e}). A transposed, permuted or mis-phased factor still \
             has the right eigenvalues, rank and shape — and fails this."
        );
    }
}

/// `j2c(k)` and `j2c(-k)` are complex conjugates — the identity
/// `gen_uniq_kpts_groups` relies on when it yields `_conj_j2c` for the
/// conjugate partner of a non-self-conjugate group.
#[test]
fn j2c_obeys_the_conjugation_identity() {
    let cell = common::he_all_electron();
    let f = built(&cell, [2, 2, 2]);
    let naux = f.naux();
    let k = [0.13_f64, -0.07, 0.21];
    let j2c = get_2c2e(&cell, &f, &[k, [-k[0], -k[1], -k[2]]], None).expect("get_2c2e");
    let mut worst = 0.0_f64;
    for p in 0..naux * naux {
        worst = worst.max((j2c[0].re[p] - j2c[1].re[p]).abs());
        worst = worst.max((j2c[0].im[p] + j2c[1].im[p]).abs());
    }
    assert!(worst < 1e-10, "j2c(k) != conj(j2c(-k)): {worst:e}");
}

// ---------------------------------------------------------------------------
// Task 6 — the 3-centre tensor and `cderi`.
// ---------------------------------------------------------------------------

use pyscf_pbc_df::gdf_builder::j3c::make_j3c;
use pyscf_pbc_df::incore::Aosym;

fn cderi_norms(b: &pyscf_pbc_df::gdf_builder::CderiBlock) -> (f64, f64) {
    (
        b.data.re.iter().map(|v| v * v).sum::<f64>().sqrt(),
        b.data.im.iter().map(|v| v * v).sum::<f64>().sqrt(),
    )
}

/// **The gate on the whole compensating-charge scheme.** `‖cderi[0,0]‖` is
/// measured upstream (`measurements/params.py`, `sr_loop(compact=False)`), so
/// it exercises `outcore_auxe2`, `add_ft_j3c`, `fuse` and `solve_cderi` at once.
fn cderi_gate(label: &str, cell: pyscf_pbc_gto::Cell, km: [usize; 3], want: f64) {
    let f = built(&cell, km);
    let k = kpts(&cell, km);
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &k, None).expect("guess_eta");
    let cderi = make_j3c(&cell, &f, &k, Aosym::S1, g.mesh, false, false, None).expect("make_j3c");
    let b = cderi.get(0, 0).expect("cderi[0,0]");
    assert_eq!(b.rank, f.naux(), "{label}: fitting rank");
    assert_eq!(b.nao_pair, cell.mol.nao_nr * cell.mol.nao_nr);
    let (r, i) = cderi_norms(b);
    assert!(
        i < 1e-10,
        "{label}: cderi[0,0] should be real, |Im| = {i:e}"
    );
    assert!(
        (r - want).abs() < 1e-8 * want,
        "{label}: |cderi[0,0]_R| = {r}, upstream {want}"
    );
}

/// The ALL-ELECTRON control, and the fast one — He-fcc has a single AO, so the
/// 3-centre lattice sum is 56k shell triples rather than diamond's 106M.
#[test]
fn cderi_fingerprint_matches_upstream_he() {
    cderi_gate(
        "he 2x2x2",
        common::he_all_electron(),
        [2, 2, 2],
        0.606_868_343_316_194_9,
    );
}

/// The flagship. **Slow**: one `aux_e2` over diamond's fused auxiliary cell is
/// minutes (plan 14-01 measured 215 s for the smaller unfused one), so this is
/// an opt-in acceptance run rather than a per-commit gate. The performance work
/// is a named carry-over in `14-02-SUMMARY.md`.
#[test]
#[ignore = "slow — one diamond aux_e2 over the fused cell is minutes"]
fn cderi_fingerprint_matches_upstream_diamond() {
    cderi_gate(
        "diamond gamma",
        common::diamond(),
        [1, 1, 1],
        1.590_025_697_491_222_6,
    );
}

/// `s2` and `s1` must give the same fit: packing is a storage choice, not a
/// numerical one. At gamma the pair block is symmetric, so the packed norm
/// relates to the square one by the triangle weight.
#[test]
fn cderi_s2_and_s1_agree() {
    let cell = common::he_all_electron();
    let km = [1usize, 1, 1];
    let f = built(&cell, km);
    let k = kpts(&cell, km);
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &k, None).expect("guess_eta");
    let s1 = make_j3c(&cell, &f, &k, Aosym::S1, g.mesh, false, false, None).expect("s1");
    let s2 = make_j3c(&cell, &f, &k, Aosym::S2, g.mesh, false, false, None).expect("s2");
    let (a, b) = (s1.get(0, 0).expect("s1"), s2.get(0, 0).expect("s2"));
    // nao = 1, so both packings are the same single pair.
    assert_eq!(a.nao_pair, 1);
    assert_eq!(b.nao_pair, 1);
    for p in 0..a.data.re.len() {
        assert!(
            (a.data.re[p] - b.data.re[p]).abs() < 1e-14,
            "s1/s2 disagree at {p}"
        );
    }
}

/// Every k-pair must keep the same fitting rank; `get_naoaux` raises otherwise,
/// because a rank that varies with `k` breaks every downstream contraction.
#[test]
fn naoaux_is_uniform_across_k_pairs() {
    let cell = common::he_all_electron();
    let km = [2usize, 2, 2];
    let f = built(&cell, km);
    let k = kpts(&cell, km);
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &k, None).expect("guess_eta");
    let cderi = make_j3c(&cell, &f, &k, Aosym::S1, g.mesh, false, false, None).expect("make_j3c");
    assert_eq!(cderi.naoaux().expect("uniform rank"), f.naux());
    // A full 2x2x2 build covers every ki-kj pair.
    assert_eq!(cderi.blocks.len(), 64);
}

// ---------------------------------------------------------------------------
// Task 7 — the exclude_dd_block seam.
// ---------------------------------------------------------------------------

use pyscf_pbc_df::gdf_builder::CcGdfBuilder;

/// **Plan 17-10 Task 3 closed this.** `exclude_dd_block = true` now BUILDS
/// and produces a correct result — see `tests/exclude_dd_block.rs` for the
/// numeric gates (both routes against their own upstream numbers, He-fcc's
/// exact 0). **This port's OWN default stays `false`**, deliberately not
/// matching upstream's `true` (see `gdf_builder`'s module docs: existing
/// oracle gates tighter than 1e-8 were built against the `false` route, and
/// this plan did not have the budget to re-verify all of them against
/// `true`'s slightly different numbers). This test pins the seam: both
/// values of the flag build without error.
#[test]
fn exclude_dd_block_both_routes_build() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let mut b = CcGdfBuilder::new(cell, &k);
    assert!(!b.exclude_dd_block, "this port's own default is false");
    assert!(b.build().is_ok(), "the false (default) branch builds");
    assert!(b.make_j3c(Aosym::S1, false).is_ok());

    let cell = common::he_all_electron();
    let mut b = CcGdfBuilder::new(cell, &k);
    b.exclude_dd_block = true;
    assert!(b.build().is_ok(), "the true (opt-in) branch also builds");
    assert!(b.make_j3c(Aosym::S1, false).is_ok());
}

/// The builder wires `guess_eta` off the AUXCELL and reproduces the standalone
/// path exactly — the seam is plumbing, not a second implementation.
#[test]
fn builder_reproduces_the_standalone_pipeline() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let mut b = CcGdfBuilder::new(cell.clone(), &k);
    b.build().expect("build");
    let eta = b.eta.expect("eta");

    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let want = guess_eta(&aux.cell, &k, None).expect("guess_eta");
    assert!((eta.eta - want.eta).abs() < 1e-15);
    assert_eq!(eta.mesh, want.mesh);
    assert_eq!(b.fused.as_ref().expect("fused").naux(), aux.naux());
}

// ---------------------------------------------------------------------------
// Task 8 — the oracle. THIS IS WHERE PLAN 14-01'S RETIRED GATE LANDS.
//
// 14-01 measured that the raw 3-centre lattice sum against a CHARGED auxiliary
// cell diverges with `rcut` and so cannot be gated. The COMPENSATED tensor is
// the one with a screening-independent value — upstream's `fuse(j3c)` is
// bit-identical at `rcut` x1.0/x1.5/x2.0 — so the 1e-11 gate is here.
// ---------------------------------------------------------------------------

const ORACLE_J3C: &str = r#"
import json
import numpy as np
import pyscf
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import gdf_builder, df as pbcdf, incore

h = 2.834589
cell = pgto.Cell()
cell.a = [[0., h, h], [h, 0., h], [h, h, 0.]]
cell.atom = [('He', (0., 0., 0.))]
cell.basis = 'sto-3g'
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()

auxcell = pbcdf.make_modrho_basis(cell, None, None)
eta, mesh, ke = gdf_builder._guess_eta(auxcell, np.zeros((1, 3)), None)
fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, eta)

kptij = np.zeros((1, 2, 3))
def j3c_at(tol):
    """`incore.aux_e2`, with upstream's Schwarz prescreen set explicitly.

    `Int3cBuilder.direct_scf_tol = None` derives
    `cell.precision / lattice_sum_factor**2 * .1` = 1.46e-11 for this system,
    which is FOUR ORDERS looser than the port's 1e-14 Gaussian-product bound.
    That difference — nothing else — is the port/upstream gap, so the gate is
    stated with the two screens equalised and the default-route deviation is
    recorded beside it.
    """
    b = incore.Int3cBuilder(cell, fused_cell, np.zeros((1, 3)))
    b.direct_scf_tol = tol
    b.build()
    kern = b.gen_int3c_kernel('int3c2e', 's1', None, True, np.arange(1),
                              return_complex=True)
    raw = np.asarray(kern()[0]).real.reshape(-1, fused_cell.nao)
    return fuse(raw, axis=1).ravel()

j3c = j3c_at(1e-14)
j3c_default_screen = j3c_at(None)

# rcut-insensitivity: the whole reason this quantity is gateable at all.
orig = incore.estimate_rcut
j3c_wide = None
try:
    incore.estimate_rcut = lambda c, a, precision=None: orig(c, a, precision) * 2.0
    j3c_wide = j3c_at(None)
finally:
    incore.estimate_rcut = orig

b = gdf_builder._CCGDFBuilder(cell, auxcell, np.zeros((1, 3)))
b.exclude_dd_block = False
b.build()
j2c = np.asarray(b.get_2c2e(np.zeros((1, 3)))[0]).real

print(json.dumps({
    'version': pyscf.__version__,
    'eta': float(eta),
    'mesh': [int(x) for x in mesh],
    'naux': int(auxcell.nao),
    'nauxc': int(fused_cell.nao),
    'j3c': j3c.tolist(),
    'j3c_default_screen': j3c_default_screen.tolist(),
    'j3c_rcut_x2_maxdiff': float(abs(j3c_default_screen - j3c_wide).max()),
    'j2c': j2c.ravel().tolist(),
}))
"#;

/// **GATE (oracle, 1e-11).** The compensated 3-centre tensor and the metric on
/// the ALL-ELECTRON control, against upstream `_CCGDFBuilder` run with
/// `exclude_dd_block = False`. He-fcc has no smooth shell, so that flag is
/// provably inert there (`measurements/ddblock.py` measures its effect as
/// exactly 0) and the gate has no escape hatch.
#[test]
#[ignore = "requires PYSCF_ORACLE_VENV"]
fn helium_fused_j3c_and_j2c_match_upstream() {
    use pyscf_pbc_df::gdf_builder::j3c::outcore_auxe2;
    use pyscf_pbc_df::incore::int3c::KptPair;

    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping the upstream oracle", common::GATE);
        return;
    };
    let want = common::run_python(&py, ORACLE_J3C, &[]);
    assert_eq!(want["version"].as_str(), Some("2.12.1"), "vendored oracle");

    // Upstream's own rcut-insensitivity, re-measured — this is the property
    // that makes the compensated tensor gateable where the raw one is not.
    let wide = want["j3c_rcut_x2_maxdiff"].as_f64().expect("rcut x2");
    assert!(
        wide < 1e-12,
        "upstream fuse(j3c) moved by {wide:e} when rcut doubled; the premise of \
         this gate (see 14-01) no longer holds"
    );

    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);
    let aux = make_modrho_basis(&cell, None, None).expect("auxcell");
    let g = guess_eta(&aux.cell, &k, None).expect("guess_eta");
    assert!((g.eta - want["eta"].as_f64().expect("eta")).abs() < 1e-12);
    let f = fuse_auxcell(&cell, None, g.eta).expect("fuse");
    assert_eq!(want["naux"].as_u64(), Some(f.naux() as u64));
    assert_eq!(want["nauxc"].as_u64(), Some(f.nauxc() as u64));

    // --- fuse(j3c) ---
    let rs = outcore_auxe2(
        &cell,
        &f,
        Aosym::S1,
        &[KptPair {
            ki: [0.0; 3],
            kj: [0.0; 3],
        }],
        None,
        None,
    )
    .expect("outcore_auxe2");
    let w: Vec<f64> = want["j3c"]
        .as_array()
        .expect("j3c")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    assert_eq!(w.len(), rs[0].re.len());
    let mut worst = 0.0_f64;
    let mut at = 0usize;
    for (p, wv) in w.iter().enumerate() {
        let d = (rs[0].re[p] - wv).abs();
        if d > worst {
            worst = d;
            at = p;
        }
    }
    eprintln!("he fuse(j3c) vs upstream @ direct_scf_tol=1e-14: max|diff| = {worst:e} at P = {at}");
    assert!(
        worst < 1e-11,
        "fuse(j3c) max|diff| = {worst:e} at P = {at} — with BOTH screens at 1e-14 \
         this is the gate on the compensating-charge algebra and has no escape hatch"
    );

    // And the deviation from upstream's DEFAULT screen, recorded as an upper
    // bound rather than a target. Upstream derives
    // `direct_scf_tol = cell.precision / lattice_sum_factor**2 * .1` = 1.46e-11
    // here, four orders looser than the port's Gaussian-product bound, so it
    // discards a P-INDEPENDENT term the port retains — 1.98e-9 on every
    // component, which is the `q_P * S_mu_nu` signature. Tightening upstream's
    // own screen collapses it to ~1e-12, which is what the assertion above
    // measures.
    let wd: Vec<f64> = want["j3c_default_screen"]
        .as_array()
        .expect("j3c_default_screen")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let mut dflt = 0.0_f64;
    for (p, wv) in wd.iter().enumerate() {
        dflt = dflt.max((rs[0].re[p] - wv).abs());
    }
    eprintln!("he fuse(j3c) vs upstream DEFAULT screen: max|diff| = {dflt:e}");
    assert!(
        dflt < 3e-9,
        "the default-screen deviation grew to {dflt:e}; it was 1.98e-9 when 14-02 \
         shipped, and it is upstream's screen, not the port's algebra"
    );

    // --- j2c ---
    let naux = f.naux();
    let j2c = get_2c2e(&cell, &f, &[[0.0; 3]], None).expect("get_2c2e");
    let w2: Vec<f64> = want["j2c"]
        .as_array()
        .expect("j2c")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    assert_eq!(w2.len(), naux * naux);
    let mut worst2 = 0.0_f64;
    for p in 0..naux * naux {
        worst2 = worst2.max((j2c[0].re[p] - w2[p]).abs());
    }
    eprintln!("he j2c: max|diff| = {worst2:e}");
    assert!(worst2 < 1e-11, "j2c max|diff| = {worst2:e}");
}
