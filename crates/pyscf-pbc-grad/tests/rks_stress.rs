//! Plan 18-12 gates — `rks_stress` CORE.
//!
//! The finite-difference form of `get_ovlp`/`get_kin` lives HERE, as the test
//! oracle and nothing else (D-PBC-31 clause 2). The production path is the
//! closed form in `pyscf_pbc_grad::stress`; each `±` cell pair is built ONCE
//! and both operators run on it (D-PBC-31 clause 11: 36 → 18 cell builds).
//!
//! Gate A tiers (`18-CONTEXT §2.3`, D-PBC-31 clause 1): A1 = 1e-9 (ovlp, kin,
//! weight, coulG, strain AO), A2 = 2e-9 (get_j, get_nuc — 18-20), A3 = 1e-8
//! (get_pp — 18-20). This file gates the A1 terms owned by 18-12 strictly at
//! upstream's own numbers; `get_vxc`/`get_j`/`get_nuc`/`get_pp`, `kernel` and
//! Gates A(assembled)/D are 18-20.

use pyscf_algebra::oracle_sum;
use pyscf_pbc_gto::{Cell, PbcIntorOpts, pbc_intor};
use pyscf_pbc_grad::stress::{
    StrainCells, coulg_strain_derivatives, eval_ao_strain_derivatives, ewald_strain,
    finite_diff_cells, ip_strain_closed_form, ip_strain_closed_form_alt, kin_strain_gamma,
    ovlp_strain_gamma, pp_nonloc_energy, pp_nonloc_strain_derivatives, strain_ao_block_comps,
    strain_block_footprint, strain_block_size, strain_tensor_displacement, to_stress,
    vpplocg_strain_derivatives, weight_strain_derivatives,
};

/// Upstream's half step for the integral strain differences
/// (`rks_stress.py:89`, `:102`); [`finite_diff_cells`] takes the FULL
/// separation, hence `2 * HALF`.
const HALF_DISP: f64 = 1e-5;
const FULL_DISP: f64 = 2e-5;

/// The FD oracle: ONE `±` pair per `(x, y)`, BOTH operators on it (clause 11).
/// Returns `(ovlp_fd, kin_fd)` F-order `nao*nao` for strain component `(x, y)`.
fn fd_ovlp_kin_pair(cell: &Cell, x: usize, y: usize) -> (Vec<f64>, Vec<f64>) {
    let gamma: [[f64; 3]; 0] = [];
    let pair: StrainCells = finite_diff_cells(cell, &gamma, x, y, FULL_DISP).expect("strain pair");
    // Trap 5: the mesh is pinned on both sides; trap 8/6.3 (fixed fractional
    // k-points) is vacuous at gamma but asserted in `verify_fd.rs`.
    assert_eq!(pair.minus.mesh, cell.mesh);
    assert_eq!(pair.plus.mesh, cell.mesh);
    let opts = PbcIntorOpts {
        hermi: 0,
        ..Default::default()
    };
    let run = |c: &Cell, intor: &str| {
        pbc_intor(c, intor, &gamma, opts)
            .expect("fd oracle intor")
            .kmats
            .remove(0)
            .re
    };
    let s_plus = run(&pair.plus, "int1e_ovlp");
    let s_minus = run(&pair.minus, "int1e_ovlp");
    let t_plus = run(&pair.plus, "int1e_kin");
    let t_minus = run(&pair.minus, "int1e_kin");
    let fd = |p: &[f64], m: &[f64]| {
        p.iter()
            .zip(m.iter())
            .map(|(a, b)| oracle_sum(&[*a, -*b]) / FULL_DISP)
            .collect()
    };
    (fd(&s_plus, &s_minus), fd(&t_plus, &t_minus))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| oracle_sum(&[*x, -*y]).abs())
        .fold(0.0, f64::max)
}

fn reference_cells() -> Vec<(&'static str, Cell)> {
    vec![
        ("diamond", pyscf_pbc_gto::test_systems::diamond()),
        ("si", pyscf_pbc_gto::test_systems::si()),
        ("lif", pyscf_pbc_gto::test_systems::lif()),
        ("he_fcc", pyscf_pbc_gto::test_systems::he_fcc()),
        ("graphene", pyscf_pbc_gto::test_systems::graphene()),
    ]
}

// ---------------------------------------------------------------------------
// Gate A1 — closed form vs FD oracle at 1e-9 (upstream's own number for
// exactly this substitution, test_rks_stress.py:65, :71, :101, :107).
// Upstream asserts components (0,0) and (0,1); so does this gate.
// ---------------------------------------------------------------------------

#[test]
fn closed_form_ovlp_matches_fd_oracle_at_1e9() {
    for (name, cell) in reference_cells() {
        let closed_a = ip_strain_closed_form(&cell, &[], "int1e_ipovlp").expect("closed ovlp");
        let closed_b =
            ip_strain_closed_form_alt(&cell, &[], "int1e_ipovlp").expect("closed ovlp alt");
        assert_eq!(closed_a.len(), 1, "{name}: gamma means one k-point");
        for (x, y) in [(0usize, 0usize), (0, 1)] {
            let (fd, _) = fd_ovlp_kin_pair(&cell, x, y);
            let idx = x * 3 + y;
            let d = max_abs_diff(&closed_a[0].re[idx], &fd);
            assert!(
                d < 1e-9,
                "{name} ovlp ({x},{y}): closed vs FD = {d:.3e} (A1 = 1e-9)"
            );
            let d_alt = max_abs_diff(&closed_b[0].re[idx], &fd);
            assert!(
                d_alt < 1e-9,
                "{name} ovlp ({x},{y}): alt vs FD = {d_alt:.3e} (A1 = 1e-9)"
            );
            // Both arrangements agree with each other (and the imaginary plane
            // is zeros at gamma).
            let ab = max_abs_diff(&closed_a[0].re[idx], &closed_b[0].re[idx]);
            assert!(
                ab < 1e-12,
                "{name} ovlp ({x},{y}): arrangement A vs B = {ab:.3e}"
            );
            assert!(
                closed_a[0].im[idx].iter().all(|v| *v == 0.0),
                "{name}: gamma imaginary plane must be exactly zero"
            );
        }
    }
}

#[test]
fn closed_form_kin_matches_fd_oracle_at_1e9() {
    for (name, cell) in reference_cells() {
        let closed_a = ip_strain_closed_form(&cell, &[], "int1e_ipkin").expect("closed kin");
        let closed_b =
            ip_strain_closed_form_alt(&cell, &[], "int1e_ipkin").expect("closed kin alt");
        for (x, y) in [(0usize, 0usize), (0, 1)] {
            let (_, fd) = fd_ovlp_kin_pair(&cell, x, y);
            let idx = x * 3 + y;
            let d = max_abs_diff(&closed_a[0].re[idx], &fd);
            assert!(
                d < 1e-9,
                "{name} kin ({x},{y}): closed vs FD = {d:.3e} (A1 = 1e-9)"
            );
            let d_alt = max_abs_diff(&closed_b[0].re[idx], &fd);
            assert!(
                d_alt < 1e-9,
                "{name} kin ({x},{y}): alt vs FD = {d_alt:.3e} (A1 = 1e-9)"
            );
            let ab = max_abs_diff(&closed_a[0].re[idx], &closed_b[0].re[idx]);
            assert!(
                ab < 1e-12,
                "{name} kin ({x},{y}): arrangement A vs B = {ab:.3e}"
            );
        }
    }
}

/// The gamma convenience wrappers agree with the k-point form.
#[test]
fn gamma_wrappers_match_kpoint_form() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let ovlp = ovlp_strain_gamma(&cell).expect("ovlp gamma");
    let kin = kin_strain_gamma(&cell).expect("kin gamma");
    let (fd_o, fd_k) = fd_ovlp_kin_pair(&cell, 1, 2);
    assert!(max_abs_diff(&ovlp.re[1 * 3 + 2], &fd_o) < 1e-9);
    assert!(max_abs_diff(&kin.re[1 * 3 + 2], &fd_k) < 1e-9);
}

// ---------------------------------------------------------------------------
// Gate A1 — weight (test_rks_stress.py:109-116) and coulG (:118-123).
// ---------------------------------------------------------------------------

#[test]
fn weight_strain_matches_vol_fd_at_1e9_and_is_diagonal() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let mesh = cell.mesh;
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let (w0, w1) = weight_strain_derivatives(cell.vol(), ngrids).expect("weight");
    assert!((w0 - cell.vol() / ngrids as f64).abs() == 0.0);
    // Exactly diagonal by construction (clause 6), not emergent-to-roundoff.
    for x in 0..3 {
        for y in 0..3 {
            if x == y {
                assert!(w1[x][y] == w0);
            } else {
                assert!(w1[x][y] == 0.0);
            }
        }
    }
    let gamma: [[f64; 3]; 0] = [];
    for (x, y) in [(0usize, 0usize), (0, 1)] {
        let pair = finite_diff_cells(&cell, &gamma, x, y, FULL_DISP).expect("strain pair");
        let fd = (pair.plus.vol() - pair.minus.vol()) / ngrids as f64 / FULL_DISP;
        let d = (w1[x][y] - fd).abs();
        assert!(d < 1e-9, "weight ({x},{y}): analytic vs FD = {d:.3e}");
    }
}

#[test]
fn coulg_strain_matches_fd_at_1e9_and_is_symmetric() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let mesh = cell.try_mesh().expect("mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(mesh)).expect("gv");
    let coulg0 = pyscf_pbc_gto::get_coulg(&cell, pyscf_pbc_gto::CoulGArgs {
        mesh: Some(mesh),
        gv: Some(&gv),
        ..pyscf_pbc_gto::CoulGArgs::new()
    })
    .expect("coulg0");
    let strain = coulg_strain_derivatives(&gv, &coulg0).expect("coulg strain");
    assert_eq!(strain.ngrids, gv.len());
    // Symmetric by construction: mirror pairs share one stored plane (clause 6).
    for x in 0..3 {
        for y in 0..3 {
            assert!(
                std::ptr::eq(strain.plane(x, y), strain.plane(y, x)),
                "coulG_1 ({x},{y}) must be the stored mirror of ({y},{x})"
            );
        }
    }
    let gamma: [[f64; 3]; 0] = [];
    for (x, y) in [(0usize, 0usize), (0, 1)] {
        let pair = finite_diff_cells(&cell, &gamma, x, y, FULL_DISP).expect("strain pair");
        // Mesh pinned; G-vectors rebuilt per displaced cell (as upstream).
        let fd_of = |c: &Cell| {
            let gv_c = pyscf_pbc_gto::get_gv(c, Some(mesh)).expect("gv displaced");
            pyscf_pbc_gto::get_coulg(c, pyscf_pbc_gto::CoulGArgs {
                mesh: Some(mesh),
                gv: Some(&gv_c),
                ..pyscf_pbc_gto::CoulGArgs::new()
            })
            .expect("coulg displaced")
        };
        let c_plus = fd_of(&pair.plus);
        let c_minus = fd_of(&pair.minus);
        let fd: Vec<f64> = c_plus
            .iter()
            .zip(c_minus.iter())
            .map(|(a, b)| oracle_sum(&[*a, -*b]) / FULL_DISP)
            .collect();
        let d = max_abs_diff(strain.plane(x, y), &fd);
        assert!(d < 1e-9, "coulG ({x},{y}): analytic vs FD = {d:.3e}");
    }
}

// ---------------------------------------------------------------------------
// Strain-AO wrapper contract (the kernel's own A1 gate lives in 18-11).
// ---------------------------------------------------------------------------

#[test]
fn strain_ao_wrapper_shape_gamma_and_refusal() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let coords = vec![[0.1, 0.2, 0.3], [1.0, 0.5, 2.0]];
    for deriv in [0u32, 1u32] {
        let table =
            eval_ao_strain_derivatives(&cell, &coords, &[], deriv).expect("strain ao wrapper");
        let comp = if deriv == 0 { 1 } else { 4 };
        assert_eq!(table.comp, comp);
        assert_eq!(table.ngrids, coords.len());
        assert_eq!(table.nao, cell.mol.nao_nr);
        assert_eq!(table.nkpts(), 1);
        assert_eq!(table.re[0].len(), 9 * comp * coords.len() * cell.mol.nao_nr);
        // Gamma: imaginary plane dropped exactly.
        assert!(table.im[0].iter().all(|v| *v == 0.0));
        // Accessor reads the documented block layout.
        let (r, _) = table.get(0, 2, 1, 0, 1, 0);
        assert!(r.is_finite());
    }
    assert!(eval_ao_strain_derivatives(&cell, &coords, &[], 2).is_err());
}

// ---------------------------------------------------------------------------
// FD-built shared symbols: finiteness, symmetry-by-construction, determinism.
// (Their analytic replacements and Gate-A3/D assertions are 18-20.)
// ---------------------------------------------------------------------------

#[test]
fn ewald_strain_is_symmetric_finite_and_deterministic() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let first = ewald_strain(&cell).expect("ewald strain");
    let second = ewald_strain(&cell).expect("ewald strain rerun");
    for x in 0..3 {
        for y in 0..3 {
            assert!(first[x][y].is_finite(), "ewald strain ({x},{y}) finite");
            assert_eq!(
                first[x][y], first[y][x],
                "ewald strain symmetric by construction"
            );
            assert_eq!(
                first[x][y], second[x][y],
                "ewald strain deterministic across reruns"
            );
        }
    }
}

#[test]
fn vpplocg_strain_is_finite_and_deterministic() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let mesh = cell.mesh;
    let first = vpplocg_strain_derivatives(&cell, mesh).expect("vpplocG strain");
    let second = vpplocg_strain_derivatives(&cell, mesh).expect("vpplocG strain rerun");
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    assert_eq!(first.ngrids, ngrids);
    assert_eq!(first.v0_re.len(), ngrids);
    for g in 0..ngrids {
        assert!(first.v0_re[g].is_finite() && first.v0_im[g].is_finite());
        for c in 0..9 {
            assert!(first.v1_re[c][g].is_finite() && first.v1_im[c][g].is_finite());
            assert_eq!(first.v1_re[c][g], second.v1_re[c][g]);
            assert_eq!(first.v1_im[c][g], second.v1_im[c][g]);
        }
    }
}

#[test]
fn pp_nonloc_strain_is_finite_and_scales_with_dm() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let nao = cell.mol.nao_nr;
    // Symmetric test density (index order immaterial on the stress path).
    let dm: Vec<f64> = (0..nao * nao)
        .map(|p| {
            let (i, j) = (p % nao, p / nao);
            if i == j { 1.0 } else { 0.25 }
        })
        .collect();
    let strain = pp_nonloc_strain_derivatives(&cell, &dm).expect("pp nonloc strain");
    for x in 0..3 {
        for y in 0..3 {
            assert!(strain[x][y].is_finite(), "pp_nonloc ({x},{y}) finite");
        }
    }
    // Linear in dm: doubling the density doubles the strain derivative.
    let dm2: Vec<f64> = dm.iter().map(|v| 2.0 * v).collect();
    let strain2 = pp_nonloc_strain_derivatives(&cell, &dm2).expect("pp nonloc strain x2");
    for x in 0..3 {
        for y in 0..3 {
            let d = (strain2[x][y] - 2.0 * strain[x][y]).abs();
            let tol = 1e-12 * (1.0 + strain[x][y].abs());
            assert!(d < tol, "pp_nonloc linear in dm ({x},{y}): {d:.3e}");
        }
    }
    // The energy underneath is the get_pp_nl contraction over vol.
    let e = pp_nonloc_energy(&cell, &dm).expect("pp nonloc energy");
    assert!(e.is_finite());
}

// ---------------------------------------------------------------------------
// Task 2 — the block loop is sized against BOTH AO arrays (clause 1).
// ---------------------------------------------------------------------------

#[test]
fn strain_blocks_cover_upstream_comps() {
    assert_eq!(strain_ao_block_comps(0), Some((4, 9)));
    assert_eq!(strain_ao_block_comps(1), Some((10, 36)));
    assert_eq!(strain_ao_block_comps(2), None);
}

#[test]
fn block_size_accounts_for_both_ao_arrays() {
    // Diamond gth-dzvp 2x2x2 scale from 18-REVIEW §3.1: nao = 26, blk = 8000.
    let (nao, blk_ref, bytes) = (26usize, 8000usize, 16usize);
    for (deriv, ao_c, st_c) in [(0u32, 4usize, 9usize), (1u32, 10usize, 36usize)] {
        let (ao_only, _) = (ao_c, st_c);
        let _ = deriv;
        // A budget that fits blk_ref of the ao array ALONE ...
        let budget_for_ao_alone_mb =
            (blk_ref * nao * ao_only * bytes) as f64 / 1e6;
        // ... must NOT fit blk_ref once the strain array is counted.
        let blk = strain_block_size(100_000, nao, ao_c, st_c, bytes, budget_for_ao_alone_mb);
        assert!(
            blk < blk_ref,
            "deriv {deriv}: budget for {blk_ref} ao-only rows must shrink to {blk} with strain counted"
        );
        // The chosen footprint accounts for both arrays and fits the budget.
        let footprint = strain_block_footprint(blk, nao, ao_c, st_c, bytes);
        assert!(
            footprint <= (budget_for_ao_alone_mb * 1e6) as u128,
            "deriv {deriv}: footprint {footprint} exceeds budget"
        );
        // Pinning a low budget shrinks blk (a budget that silently ignored the
        // larger array would fail here rather than at the OOM killer).
        let tiny = strain_block_size(100_000, nao, ao_c, st_c, bytes, 0.001);
        assert_eq!(tiny, 1, "deriv {deriv}: tiny budget must clamp to 1, got {tiny}");
        // Clamp endpoints.
        assert_eq!(strain_block_size(0, nao, ao_c, st_c, bytes, 4000.0), 0);
        assert_eq!(strain_block_size(7, nao, ao_c, st_c, bytes, 1e12), 7);
    }
}

// ---------------------------------------------------------------------------
// Units + displacement: stress is Ha/Bohr^3, vol in the expression.
// ---------------------------------------------------------------------------

#[test]
fn stress_reported_in_pressure_units_with_vol() {
    let ded_eps = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let vol = pyscf_pbc_gto::test_systems::diamond().vol();
    let sigma = to_stress(ded_eps, vol).expect("to_stress");
    for x in 0..3 {
        for y in 0..3 {
            assert_eq!(sigma[x][y], ded_eps[x][y] / vol);
        }
    }
    assert!(to_stress(ded_eps, 0.0).is_err());
    assert!(to_stress(ded_eps, f64::NAN).is_err());
    // Upstream's typo'd name maps to this symbol (rks_stress.py:59).
    let e = strain_tensor_displacement(0, 1, HALF_DISP);
    assert_eq!(e[0][1], HALF_DISP);
    assert_eq!(e[1][1], 1.0);
    assert_eq!(e[2][0], 0.0);
}
