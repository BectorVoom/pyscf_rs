//! Plan 18-03 Tasks 3+4 — `vpploc_part2_nuc_grad` and `vppnl_nuc_grad`.
//!
//! Oracle gates against vendored PySCF 2.12.1 on all five §9.2 reference
//! cells, at 18-01's measured tolerance. No ignore attribute is used anywhere in
//! this file: 18-CONTEXT §1.3 retired the deferral fallback, and every value
//! below was measured, not assumed.
//!
//! * `dm` is the Hilbert-like symmetric matrix `dm[p,q] = 1/(1+p+q)`
//!   (F-order `nao x nao`), the same closed form on both sides — no SCF, no
//!   fixtures to drift. Every entry is distinct and nonzero, so a dropped
//!   contraction term moves the answer.
//! * Upstream references: `pp_int.vpploc_part2_nuc_grad(cell, dm)` at gamma
//!   (Tasks 3) and `pp_int.vppnl_nuc_grad(cell, dm[, kpts])` at gamma and
//!   2×2×2 (Task 4). Cells are the Rust-exact §9.2 systems (Angstrom inputs ×
//!   CODATA-2014 1.8897261339213, `unit='Bohr'`, `precision=1e-8`); the
//!   `(rcut, |Ls|, nao)` preconditions pin the truncation before any gradient
//!   is compared.
//! * `vpploc_part2_nuc_grad` refuses non-gamma k-points — upstream
//!   `pp_int.py:178-179` raises `NotImplementedError` there too
//!   (`pbc/grad/rhf.py:66` and `pbc/grad/uhf.py:64` are its only callers, both
//!   gamma-only).
//!
//! # Geometry is specified in BOHR
//!
//! See `pbc_intor.rs` for the CODATA note.

use pyscf_pbc_gto::kpts_mesh::make_kpts_default;
use pyscf_pbc_gto::lattice::get_lattice_ls;
use pyscf_pbc_gto::pseudo::{vpploc_part2_nuc_grad, vppnl_nuc_grad};
use pyscf_pbc_gto::{Cell, test_systems};

mod common;
use common::systems;

/// Tolerance: 18-01's measured floors. The lattice-sum gradient residuals
/// print on every run; all five cells land at ~1e-12 or below against
/// upstream, so 1e-9 carries three decades of headroom without approaching
/// the 1e-6 `FD_TOL` gate these gradients will ultimately face.
const TOL: f64 = 1e-9;

/// `dm[p,q] = 1/(1+p+q)`, F-order — symmetric, positive definite, no zeros.
fn hilbert_dm(nao: usize) -> Vec<f64> {
    let mut dm = vec![0.0_f64; nao * nao];
    for q in 0..nao {
        for p in 0..nao {
            dm[p + q * nao] = 1.0 / (1 + p + q) as f64;
        }
    }
    dm
}

/// (nao, rcut, |Ls|) preconditions: the same cell, the same truncation.
fn assert_same_truncation(name: &str, cell: &Cell, nao: usize, rcut: f64, nls: usize) {
    assert_eq!(cell.mol.nao_nr, nao, "{name}: nao differs");
    assert!(
        (cell.rcut - rcut).abs() < 1e-9,
        "{name}: rcut differs: upstream {rcut} vs {}",
        cell.rcut
    );
    let ls = get_lattice_ls(cell, None, None, true).expect("Ls");
    assert_eq!(ls.len(), nls, "{name}: |Ls| differs");
}

/// Max |delta| between two (natm, 3) gradients in C-order ravel.
fn max_delta(got: &[[f64; 3]], want: &[f64]) -> f64 {
    assert_eq!(got.len() * 3, want.len());
    got.iter()
        .flat_map(|r| r.iter())
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max)
}

/// Momentum conservation: a lattice sum at fixed density carries no net force.
/// Upstream's own numbers satisfy this to ~1e-16; ours must too, independent
/// of any oracle.
fn assert_no_net_force(name: &str, what: &str, got: &[[f64; 3]]) {
    let mut total = [0.0_f64; 3];
    for row in got {
        for c in 0..3 {
            total[c] += row[c];
        }
    }
    let worst = total.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        worst < 1e-9,
        "{name} {what}: net force {total:?} exceeds 1e-9 — the bra/ket/aux \
         shares do not cancel"
    );
}

/// The five §9.2 cells with their (nao, rcut, |Ls|) preconditions, measured
/// from the same upstream build as the gradients below.
fn reference_cells() -> Vec<(&'static str, Cell, usize, f64, usize)> {
    vec![
        ("diamond", systems::diamond(), 8, 21.31940052177759, 767),
        ("si", systems::si(), 8, 29.960198598827567, 627),
        ("lif", systems::lif(), 6, 38.46107083110416, 3511),
        ("he_fcc", systems::he_fcc(), 1, 16.808894871965055, 429),
        ("graphene", systems::graphene(), 8, 21.31940052177759, 91),
    ]
}

/// Task 3 gate: `pp_int.vpploc_part2_nuc_grad(cell, dm)` at gamma, in Ha/Bohr.
///
/// He is all-electron (`gth-pade` carries no He potential), so both sides give
/// zeros — upstream `[-0.0; 3]`, this port `[[0.0; 3]]`.
#[test]
fn vpploc_part2_nuc_grad_matches_upstream_at_gamma() {
    // C-order ravel of the (natm, 3) upstream gradient, 17 digits.
    let want: &[&[f64]] = &[
        &[
            0.0002228613226218385,
            0.00675823931013223,
            0.01097828737244086,
            -0.0002228613226217752,
            -0.006758239310132206,
            -0.010978287372440896,
        ],
        &[
            -0.001416920235166836,
            0.0013812878824851862,
            0.003057764446220739,
            0.0014169202351668397,
            -0.0013812878824851936,
            -0.0030577644462207398,
        ],
        &[
            0.03313075808286165,
            0.02855912863819672,
            0.02234431879636954,
            -0.03313075808286165,
            -0.028559128638196735,
            -0.022344318796369514,
        ],
        &[0.0, 0.0, 0.0],
        &[
            -0.04322266751145079,
            -0.01642450320303327,
            0.01170986270312388,
            0.04322266751145074,
            0.01642450320303329,
            -0.011709862703123887,
        ],
    ];
    for ((name, cell, nao, rcut, nls), w) in reference_cells().iter().zip(want.iter()) {
        assert_same_truncation(name, cell, *nao, *rcut, *nls);
        let dm = hilbert_dm(cell.mol.nao_nr);
        let got = vpploc_part2_nuc_grad(cell, &dm, &[[0.0; 3]]).expect("vpploc_part2_nuc_grad");
        let worst = max_delta(&got, w);
        println!("{name}: vpploc_part2_nuc_grad residual = {worst:e}");
        assert!(
            worst < TOL,
            "{name}: vpploc_part2_nuc_grad differs from upstream by {worst:e}"
        );
        assert_no_net_force(name, "vpploc_part2", &got);
    }
}

/// `pp_int.py:178-179` raises `NotImplementedError("k-point sampling not
/// available")` for non-gamma k-points; this port refuses identically.
#[test]
fn vpploc_part2_nuc_grad_refuses_non_gamma_kpts() {
    let cell = test_systems::diamond();
    let dm = hilbert_dm(cell.mol.nao_nr);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let err = vpploc_part2_nuc_grad(&cell, &dm, &kpts).expect_err("non-gamma must refuse");
    assert!(
        matches!(
            err,
            pyscf_core::PyscfRsError::NotYetImplemented { phase: 18, .. }
        ),
        "unexpected error: {err}"
    );
}

/// Task 4 gate, gamma branch: `pp_int.vppnl_nuc_grad(cell, dm)`.
///
/// LiF's fluorine carries a single s projector whose gradient contributions
/// cancel to ~1e-17 for every tested density (Hilbert and identity alike) —
/// a zero-gate, not a contraction gate; diamond/si/graphene carry the weight.
#[test]
fn vppnl_nuc_grad_matches_upstream_at_gamma() {
    let want: &[&[f64]] = &[
        &[
            0.046890035887751744,
            0.02841820356833428,
            0.01953751495322964,
            -0.04689003588775176,
            -0.0284182035683343,
            -0.01953751495322964,
        ],
        &[
            0.0048612274979019255,
            -0.004366678064232294,
            -0.00989411204102772,
            -0.00486122749790196,
            0.004366678064232304,
            0.009894112041027728,
        ],
        &[
            -5.421019770079481e-18,
            4.011204204748598e-17,
            3.237834560749935e-17,
            -0.0,
            -0.0,
            -0.0,
        ],
        &[0.0, 0.0, 0.0],
        &[
            0.0826314777967509,
            0.050079683513183045,
            -0.04484582680954518,
            -0.08263147779675091,
            -0.0500796835131831,
            0.04484582680954519,
        ],
    ];
    for ((name, cell, nao, rcut, nls), w) in reference_cells().iter().zip(want.iter()) {
        assert_same_truncation(name, cell, *nao, *rcut, *nls);
        let dm = hilbert_dm(cell.mol.nao_nr);
        let got = vppnl_nuc_grad(cell, &dm, &[[0.0; 3]]).expect("vppnl_nuc_grad");
        let worst = max_delta(&got, w);
        println!("{name}: vppnl_nuc_grad (gamma) residual = {worst:e}");
        assert!(
            worst < TOL,
            "{name}: vppnl_nuc_grad (gamma) differs from upstream by {worst:e}"
        );
        assert_no_net_force(name, "vppnl_gamma", &got);
    }
}

/// Task 4 gate, k-point branch: `pp_int.vppnl_nuc_grad(cell, dm_k, kpts)` on a
/// 2×2×2 mesh (`pp_int.py:468-509`, the pure-numpy path — no C driver).
#[test]
fn vppnl_nuc_grad_matches_upstream_at_2x2x2() {
    // (name, upstream (natm,3) ravel). All five cells: the k-path is pure
    // numpy upstream, so dimension 2 is no obstacle.
    let want: &[(&str, &[f64])] = &[
        (
            "diamond",
            &[
                0.8276997816792662,
                0.7799881489895029,
                0.8101315262292421,
                -0.8276997816792662,
                -0.779988148989502,
                -0.8101315262292421,
            ],
        ),
        (
            "si",
            &[
                0.41573345580600707,
                0.3831269005266291,
                0.39838967148055926,
                -0.41573345580600696,
                -0.38312690052662957,
                -0.3983896714805589,
            ],
        ),
        (
            "lif",
            &[
                3.122502256758253e-17,
                7.979727989493313e-17,
                9.367506770274758e-17,
                0.0,
                -3.552713678800501e-15,
                0.0,
            ],
        ),
        ("he_fcc", &[0.0, 0.0, 0.0]),
        (
            "graphene",
            &[
                -0.12347573684199276,
                0.7275015142776633,
                -0.10962611438274195,
                0.12347573684199542,
                -0.7275015142776655,
                0.10962611438274283,
            ],
        ),
    ];
    for (name, w) in want.iter() {
        let cell = match *name {
            "diamond" => systems::diamond(),
            "si" => systems::si(),
            "lif" => systems::lif(),
            "he_fcc" => systems::he_fcc(),
            "graphene" => systems::graphene(),
            _ => unreachable!(),
        };
        let dm = hilbert_dm(cell.mol.nao_nr);
        let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
        let got = vppnl_nuc_grad(&cell, &dm, &kpts).expect("vppnl_nuc_grad kpts");
        let worst = max_delta(&got, w);
        println!("{name}: vppnl_nuc_grad (2x2x2) residual = {worst:e}");
        assert!(
            worst < TOL,
            "{name}: vppnl_nuc_grad (2x2x2) differs from upstream by {worst:e}\n  \
             got = {got:?}\n  want = {w:?}"
        );
        assert_no_net_force(name, "vppnl_222", &got);
    }
}
