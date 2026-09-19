//! Plan 18-12 gates — `rks_stress` CORE.
//!
//! The finite-difference form of `get_ovlp`/`get_kin` lives HERE, as the test
//! oracle and nothing else (D-PBC-31 clause 2). The production path is the
//! closed form in `pyscf_pbc_grad::stress`; each `±` cell pair is built ONCE
//! and both operators run on it (D-PBC-31 clause 11: 36 → 18 cell builds).
//!
//! Gate A tiers (`18-CONTEXT §2.3`, D-PBC-31 clause 1): A1 = 1e-9 (ovlp, kin,
//! weight, coulG, strain AO, get_vxc LDA), A2 = 2e-9 (get_j, get_nuc),
//! A3 = 1e-8 (get_pp). 18-12 gated ovlp/kin/weight/coulG; 18-20 adds the
//! strain-AO wrapper gates, the grid-response and lattice gates,
//! `get_vxc` (LDA at A1; GGA/MGGA are named refusals — the `deriv2` AO kernel
//! is Phase-4 scope and the XC surface exposes no `vtau`), get_j/get_nuc at
//! A2, get_pp at A3, and Gate D (LDA end to end at 1e-6 Ha/Bohr³ with `vol`
//! named; GGA/MGGA end to end are blocked on the same refusal).
//!
//! Every gate below names the upstream assertion line its tolerance inherits.
//! A measured floor above its tier would be a finding with a number, never a
//! licence to fall back to A3.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_core::Unit;
use pyscf_core::{ParsedBasis, ShellSpec};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::fftdf::Fftdf;
use pyscf_pbc_df::traits::{JkOpts, PeriodicDf};
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_grad::gamma_rhf::gamma_make_rdm1e;
use pyscf_pbc_grad::stress::{
    StrainCells, VxcStrainOpts, coulg_strain_derivatives, eval_ao_strain_derivatives, ewald_strain,
    finite_diff_cells, get_vxc, ip_strain_closed_form, ip_strain_closed_form_alt, kin_strain_gamma,
    ovlp_strain_gamma, pp_nonloc_energy, pp_nonloc_strain_derivatives, rks_stress_kernel,
    strain_ao_block_comps, strain_block_footprint, strain_block_size, strain_tensor_displacement,
    to_stress, vpplocg_strain_derivatives, weight_strain_derivatives,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, PbcIntorOpts, pbc_intor};

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
    let coulg0 = pyscf_pbc_gto::get_coulg(
        &cell,
        pyscf_pbc_gto::CoulGArgs {
            mesh: Some(mesh),
            gv: Some(&gv),
            ..pyscf_pbc_gto::CoulGArgs::new()
        },
    )
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
            pyscf_pbc_gto::get_coulg(
                c,
                pyscf_pbc_gto::CoulGArgs {
                    mesh: Some(mesh),
                    gv: Some(&gv_c),
                    ..pyscf_pbc_gto::CoulGArgs::new()
                },
            )
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
    // The energy underneath is the get_pp_nl contraction, with NO volume
    // divisor (18-20 probe: upstream's /vol belongs to its plane-wave
    // `vppnl` intermediate, not to the matrix trace). Contracted here in
    // einsum ij,ji order against get_pp_nl's F-order matrix — a different
    // index order than the function's own zip, so this pins both the
    // normalization and the symmetry it relies on.
    let e = pp_nonloc_energy(&cell, &dm).expect("pp nonloc energy");
    assert!(e.is_finite());
    let vnl = pyscf_pbc_gto::pseudo::get_pp_nl(&cell, &GAMMA).expect("pp nl matrix")[0]
        .re
        .clone();
    let tr = trace_dot(&dm, &vnl, nao);
    let d = (e - tr).abs();
    assert!(
        d < 1e-10 * (1.0 + e.abs()),
        "pp_nonloc energy must be the matrix trace (no /vol): {d:.3e}"
    );
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
        let budget_for_ao_alone_mb = (blk_ref * nao * ao_only * bytes) as f64 / 1e6;
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
        assert_eq!(
            tiny, 1,
            "deriv {deriv}: tiny budget must clamp to 1, got {tiny}"
        );
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

// ---------------------------------------------------------------------------
// Plan 18-20 — Gates A (assembled terms) and D (end to end).
//
// Fixture provenance: upstream's component tests build
// `a = eye(3)*5 + rand(3,3) - .5` under `np.random.seed(5)`
// (`test_rks_stress.py:28-34`), and Gate D `eye(3)*3` / `*3.5` the same way
// (`:391-393`, `:409-411`, `:427-429`). The three lattices below are those
// exact seed-5 matrices (evaluated once against the vendored PySCF), so the
// off-diagonal strain components are exercised exactly as upstream exercises
// them (D-PBC-31 clause 3: a cubic cell makes several vanish).
// Geometry is in Bohr throughout (18-CONTEXT rule 5).
// ---------------------------------------------------------------------------

/// `np.random.seed(5); eye(3)*5 + rand(3,3) - .5` — the component-test lattice.
const SEED5_A5: [[f64; 3]; 3] = [
    [4.721993171089739, 0.3707323061773764, -0.2932808446605736],
    [0.41861090793792155, 4.988411188794829, 0.11174386290264571],
    [0.26590785648031556, 0.018417987872943242, 4.796800501576222],
];

/// `np.random.seed(5); eye(3)*3 + rand(3,3) - .5` — Gate-D LDA lattice.
const SEED5_A3: [[f64; 3]; 3] = [
    [2.7219931710897396, 0.3707323061773764, -0.2932808446605736],
    [0.41861090793792155, 2.9884111887948293, 0.11174386290264571],
    [0.26590785648031556, 0.018417987872943242, 2.796800501576222],
];

const GAMMA: [[f64; 3]; 1] = [[0.0; 3]];

/// Upstream's `test_get_vxc_*` basis (`:264`, `[[0, [.5, 1]], [1, [.8, 1]]]`),
/// as an already-parsed shell list (global: both atoms are He).
fn he_sp_basis() -> BasisInput {
    BasisInput::Parsed(ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: vec![0.5],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 1,
                exponents: vec![0.8],
                coeffs: vec![vec![1.0]],
            },
        ],
    })
}

/// Upstream's `test_eval_ao_{cart,sph}` shells
/// (`:129-133`, s/p/d/f), parsed.
fn he_spdf_basis() -> BasisInput {
    BasisInput::Parsed(ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: vec![0.5],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 1,
                exponents: vec![1.5, 0.5],
                coeffs: vec![vec![1.0, 1.0]],
            },
            ShellSpec {
                l: 2,
                exponents: vec![0.8],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 3,
                exponents: vec![0.7],
                coeffs: vec![vec![1.0]],
            },
        ],
    })
}

/// All-electron He2 on the seed-5 lattice (Bohr), upstream's component fixture
/// shape (`test_rks_stress.py:28-34`, `:261-265`).
fn seed5_he_cell(basis: BasisInput, cart: bool) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("He 1 1 1; He 2 1.5 2.4".into()),
            basis,
            unit: Unit::Bohr,
            cart,
            ..Default::default()
        },
        a: ALattice::Matrix(SEED5_A5),
        mesh: None,
        precision: 1e-10,
        pseudo: None,
        ..Default::default()
    })
    .expect("seed-5 He cell builds")
}

/// Deterministic xorshift64* in `[0, 1)`: the test density's entropy source.
/// (The density needs no numpy fidelity — only symmetry — but it must be
/// fixed across runs.)
fn rng_unit(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Symmetric test density `dm = A·Aᵀ` (index order immaterial on the stress
/// path; the SCF densities `kernel` consumes are exactly symmetric).
fn sym_dm(nao: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let a: Vec<f64> = (0..nao * nao).map(|_| rng_unit(&mut s) - 0.5).collect();
    let mut dm = vec![0.0; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            let mut t = 0.0;
            for k in 0..nao {
                t += a[i * nao + k] * a[j * nao + k];
            }
            dm[i * nao + j] = t;
        }
    }
    dm
}

/// `ni.nr_rks(cell, UniformGrids(cell), xc, dm)[1]` — the Gate-A XC oracle
/// (`test_rks_stress.py:275-276` etc.). The grid follows the (possibly
/// strained) cell, exactly as upstream's fresh `UniformGrids(cell1)` does.
fn nr_rks_exc(cell: &Cell, xc: &str, dm: &[f64]) -> f64 {
    let ni = KNumInt::new(&[]);
    let grids = PeriodicGrids::uniform(cell, None).expect("uniform grids");
    let nao = cell.mol.nao_nr;
    let dm_ct = CTensor::from_planes(dm.to_vec(), vec![0.0; nao * nao]);
    ni.nr_rks(cell, &grids, xc, &vec![vec![dm_ct]], 1, None)
        .expect("nr_rks oracle")
        .excsum[0]
}

/// `FFTDF(cell1).get_jk(dm, with_k=False)[0]`, row-major.
fn fftdf_j(cell: &Cell, dm: &[f64]) -> Vec<f64> {
    let df = Fftdf::new(cell.clone(), &[]).expect("fftdf builds");
    let nao = cell.mol.nao_nr;
    let dm_ct = CTensor::from_planes(dm.to_vec(), vec![0.0; nao * nao]);
    let out = df
        .get_jk(
            &[vec![dm_ct]],
            &GAMMA,
            JkOpts {
                hermi: 1,
                kpts_band: None,
                with_j: true,
                with_k: false,
                exxdiv: None,
                omega: None,
                kk_symmetry: false,
            },
        )
        .expect("fftdf get_jk");
    out.vj.expect("vj")[0][0].re.clone()
}

/// `FFTDF(cell1).get_nuc()[0]` / `.get_pp(kpt)[0]`, row-major.
fn fftdf_nuc(cell: &Cell) -> Vec<f64> {
    let df = Fftdf::new(cell.clone(), &[]).expect("fftdf builds");
    df.get_nuc(&GAMMA).expect("fftdf get_nuc")[0].re.clone()
}

fn fftdf_pp(cell: &Cell) -> Vec<f64> {
    let df = Fftdf::new(cell.clone(), &[]).expect("fftdf builds");
    df.get_pp(&GAMMA).expect("fftdf get_pp")[0].re.clone()
}

/// `np.einsum('ij,ji', dm, dv)` — upstream's trace order (`:338`, `:361`,
/// `:386`), via `oracle_sum` (trap 9: the transposed density index is
/// upstream's, not a symmetrisation).
fn trace_dot(dm: &[f64], dv: &[f64], nao: usize) -> f64 {
    let mut terms = Vec::with_capacity(nao * nao);
    for i in 0..nao {
        for j in 0..nao {
            terms.push(dm[i * nao + j] * dv[j * nao + i]);
        }
    }
    oracle_sum(&terms)
}

const ALL_NINE: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

// ---------------------------------------------------------------------------
// Gate A1 — strain-AO wrapper vs FD of the ordinary AO (upstream :140, :158,
// :176, :194; the kernel's own A1 gate lives in 18-11's eval_ao_strain.rs).
// ---------------------------------------------------------------------------

fn eval_ao_fd_gate(cart: bool, deriv: u32, ao_name: &str) {
    let cell = seed5_he_cell(he_spdf_basis(), cart);
    let mut s = 0x9E37_79B9_7F4A_7C15;
    let coords: Vec<[f64; 3]> = (0..10)
        .map(|_| [rng_unit(&mut s), rng_unit(&mut s), rng_unit(&mut s)])
        .collect();
    let table = eval_ao_strain_derivatives(&cell, &coords, &[], deriv).expect("strain ao wrapper");
    let comp = if deriv == 0 { 1 } else { 4 };
    assert_eq!(table.comp, comp);
    // 18-11's pre-existing finding (documented in
    // `pyscf-pbc-gto/tests/eval_ao_strain.rs`): the `GTOval_cart` value path
    // is broken (periodic mismatch 2.1e2) while `GTOval_cart_deriv1` honors
    // the cart flag with an independently validated value block. The cart
    // deriv-0 gate therefore takes its FD reference from
    // `GTOval_cart_deriv1[0]` — the same quantity — and compares only comp 0.
    let (fd_name, fd_comp0_only) = if cart && deriv == 0 {
        ("GTOval_cart_deriv1", true)
    } else {
        (ao_name, false)
    };
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let a1 = pyscf_pbc_gto::eval_ao_kpts(&pair.plus, fd_name, &coords, &[])
            .expect("plus ao")
            .kaos
            .remove(0);
        let a2 = pyscf_pbc_gto::eval_ao_kpts(&pair.minus, fd_name, &coords, &[])
            .expect("minus ao")
            .kaos
            .remove(0);
        // `fd_comp0_only`: the reference table carries the value in comp 0
        // of a multi-component output; all other comps are skipped.
        for c in 0..comp {
            if fd_comp0_only && c > 0 {
                continue;
            }
            let plane = (x * 3 + y) * comp + c;
            let nb = table.ngrids * table.nao;
            let got = &table.re[0][plane * nb..(plane + 1) * nb];
            let mut fd = Vec::with_capacity(nb);
            // Ordinary-AO layouts match the strain table's per-component
            // F-order (`g + mu*ngrids`); deriv1's comp axis is the same order.
            for p in 0..nb {
                let base = if fd_comp0_only { p } else { c * nb + p };
                fd.push(oracle_sum(&[a1.re[base], -a2.re[base]]) / FULL_DISP);
            }
            let d = max_abs_diff(got, &fd);
            eprintln!("strain ao cart={cart} deriv={deriv} ({x},{y}) c={c}: {d:.3e}");
            assert!(
                d < 1e-9,
                "strain ao ({x},{y}) comp {c}: wrapper vs FD = {d:.3e} (A1 = 1e-9)"
            );
        }
    }
}

/// Gate A1 — `test_eval_ao_cart` (`test_rks_stress.py:125-140`, bound 1e-9 at
/// `:140`).
#[test]
fn gate_a1_eval_ao_cart_matches_fd() {
    eval_ao_fd_gate(true, 0, "GTOval_cart");
}

/// Gate A1 — `test_eval_ao_sph` (`:142-158`, bound 1e-9 at `:158`).
#[test]
fn gate_a1_eval_ao_sph_matches_fd() {
    eval_ao_fd_gate(false, 0, "GTOval_sph");
}

/// Gate A1 — `test_eval_ao_deriv1_cart` (`:160-176`, bound 1e-9 at `:176`).
#[test]
fn gate_a1_eval_ao_deriv1_cart_matches_fd() {
    eval_ao_fd_gate(true, 1, "GTOval_cart_deriv1");
}

/// Gate A1 — `test_eval_ao_deriv1_sph` (`:178-194`, bound 1e-9 at `:194`).
#[test]
fn gate_a1_eval_ao_deriv1_sph_matches_fd() {
    eval_ao_fd_gate(false, 1, "GTOval_sph_deriv1");
}

// ---------------------------------------------------------------------------
// Gate A1 — grid response (upstream :196-214, bound 1e-9 at :214).
//
// Only the deriv-0 half (`ao_value[:,:,0] += einsum('xgi,gy->xygi',
// ao[1:4], coords)`) is gated here: the deriv-1 half (`:226-235`) contracts
// `deriv = 2` ordinary AOs, which the molecular kernel defers (Phase-4
// scope) — the same boundary as the GGA refusal below.
// ---------------------------------------------------------------------------

/// Gate A1 — `test_eval_ao_grid_response`, deriv-0 half
/// (`test_rks_stress.py:205-214`, bound 1e-9 at `:214`).
#[test]
fn gate_a1_grid_response_matches_fd() {
    for cart in [false, true] {
        let cell = seed5_he_cell(he_spdf_basis(), cart);
        // 18-11's finding again: the cart VALUE reference is
        // `GTOval_cart_deriv1[0]`, not `GTOval_cart` (broken device path).
        let fd_name = if cart {
            "GTOval_cart_deriv1"
        } else {
            "GTOval_sph"
        };
        let mesh = [6, 5, 4];
        let grids = cell.uniform_grids(Some(mesh)).expect("uniform grids");
        let coords = &grids.coords;
        let ngrids = coords.len();
        let nao = cell.mol.nao_nr;
        let deriv1_name = if cart {
            "GTOval_cart_deriv1"
        } else {
            "GTOval_sph_deriv1"
        };
        let ao = pyscf_pbc_gto::eval_ao_kpts(&cell, deriv1_name, coords, &[])
            .expect("deriv1 ao")
            .kaos
            .remove(0);
        let strain = eval_ao_strain_derivatives(&cell, coords, &[], 0).expect("strain ao");
        for (x, y) in ALL_NINE {
            // `ao_value[:,:,0] += einsum('xgi,gy->xygi', ao[1:4], coords)`.
            let mut got = vec![0.0; ngrids * nao];
            for g in 0..ngrids {
                for mu in 0..nao {
                    let (s, _) = strain.get(0, x, y, 0, g, mu);
                    let grad = ao.re[(1 + x) * ngrids * nao + g + mu * ngrids];
                    got[g + mu * ngrids] = s + grad * coords[g][y];
                }
            }
            let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
            let g1 = pair.plus.uniform_grids(Some(mesh)).expect("plus grids");
            let g2 = pair.minus.uniform_grids(Some(mesh)).expect("minus grids");
            let a1 = pyscf_pbc_gto::eval_ao_kpts(&pair.plus, fd_name, &g1.coords, &[])
                .expect("plus ao")
                .kaos
                .remove(0);
            let a2 = pyscf_pbc_gto::eval_ao_kpts(&pair.minus, fd_name, &g2.coords, &[])
                .expect("minus ao")
                .kaos
                .remove(0);
            // Cart: value lives in comp 0 of the deriv1 table.
            let (r1, r2) = if cart {
                let nb = ngrids * nao;
                (&a1.re[..nb], &a2.re[..nb])
            } else {
                (&a1.re[..], &a2.re[..])
            };
            let fd: Vec<f64> = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| oracle_sum(&[*a, -*b]) / FULL_DISP)
                .collect();
            let d = max_abs_diff(&got, &fd);
            eprintln!("grid response cart={cart} ({x},{y}): {d:.3e}");
            assert!(
                d < 1e-9,
                "grid response cart={cart} ({x},{y}): analytic vs FD = {d:.3e} (A1 = 1e-9)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gate A1 — lattice-vector derivatives (upstream :242-258, 1e-9 at :253/:257).
// ---------------------------------------------------------------------------

/// Gate A1 — `test_lattice_vector_derivatives`
/// (`test_rks_stress.py:242-258`, bound 1e-9 at `:253` and `:257`).
#[test]
fn gate_a1_lattice_vector_derivatives_match_fd() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    // `ref[i,j,:,i] = a[:,j]` (`:248-251`): d(a·Eᵀ)/dε_xy at E = I is
    // `a[k,y]·δ[l,x]`; checked entrywise against the FD of `cell.a`.
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        for k in 0..3 {
            for l in 0..3 {
                let analytic = cell.a[k][y] * if l == x { 1.0 } else { 0.0 };
                let fd = oracle_sum(&[pair.plus.a[k][l], -pair.minus.a[k][l]]) / FULL_DISP;
                let d = (analytic - fd).abs();
                if d > 1e-12 {
                    eprintln!("lattice ({x},{y})[{k},{l}]: {d:.3e}");
                }
                assert!(
                    d < 1e-9,
                    "lattice ({x},{y})[{k},{l}]: analytic vs FD = {d:.3e} (A1 = 1e-9)"
                );
            }
        }
    }
    // Upstream also checks its own broadcast closed form against `ref`
    // (`:251-253`, `a1 = eye * a.T` vs `ref`); that identity is numpy
    // reshaping, not physics — the FD loop above is the gate, and it pins the
    // entrywise form `d(a·Eᵀ)/dε_xy[k,l] = a[k,y]·δ[l,x]`.
}

// ---------------------------------------------------------------------------
// Gate A1 — get_vxc LDA (upstream :260-277, bound 1e-9 at :277).
// 'lda,' IS Slater+VWN ("lda,vwn") — the comma is upstream's empty-token
// spelling, not a different functional.
// ---------------------------------------------------------------------------

/// Gate A1 — `test_get_vxc_lda` (`test_rks_stress.py:260-277`, bound 1e-9 at
/// `:277`). `get_vxc` without `with_j`/`with_nuc` is the pure-XC strain.
#[test]
fn gate_a1_get_vxc_lda_matches_nr_rks_fd() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dat = get_vxc(&cell, &dm, "lda,vwn", VxcStrainOpts::default()).expect("get_vxc lda");
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let e1 = nr_rks_exc(&pair.plus, "lda,vwn", &dm);
        let e2 = nr_rks_exc(&pair.minus, "lda,vwn", &dm);
        let d = (dat[x][y] - oracle_sum(&[e1, -e2]) / FULL_DISP).abs();
        eprintln!("get_vxc lda ({x},{y}): {d:.3e}");
        assert!(
            d < 1e-9,
            "get_vxc lda ({x},{y}): analytic vs FD = {d:.3e} (A1 = 1e-9)"
        );
    }
}

// ---------------------------------------------------------------------------
// get_vxc GGA/MGGA — NAMED REFUSALS (not gates).
//
// `test_get_vxc_gga` (`:279-296`) and `test_get_vxc_mgga` (`:298-315`) assert
// at A1 = 1e-9 upstream. This port cannot run them: the grid response of the
// gradient-density rows needs second-derivative AOs (`GTOval_*_deriv2`),
// which the molecular kernel defers (Phase-4 scope), and MGGA additionally
// needs the τ potential `vtau`, which the XC surface does not expose. These
// tests pin the refusal so the boundary is a gate rather than a comment; the
// measured upstream floors are in `measurements/gate-a-tiers.md`
// (A1 max stable-reference residual 9.90842297099448e-10).
// ---------------------------------------------------------------------------

/// `test_get_vxc_gga` has no port: GGA refuses by name (missing `deriv2` AO
/// kernel), never by falling back to LDA.
#[test]
fn get_vxc_gga_refuses_without_deriv2() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let dm = sym_dm(cell.mol.nao_nr, 0x51ab);
    let err = get_vxc(&cell, &dm, "pbe", VxcStrainOpts::default()).expect_err("GGA must refuse");
    assert!(
        err.to_string().contains("deriv2"),
        "GGA refusal must name the missing deriv2 kernel, got: {err}"
    );
}

/// `test_get_vxc_mgga` has no port: MGGA refuses by name (missing `tau`/`vtau`
/// supply plus the same `deriv2` kernel).
#[test]
fn get_vxc_mgga_refuses_without_tau() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let dm = sym_dm(cell.mol.nao_nr, 0x51ab);
    let err = get_vxc(&cell, &dm, "scan", VxcStrainOpts::default()).expect_err("MGGA must refuse");
    assert!(
        err.to_string().contains("tau"),
        "MGGA refusal must name the missing tau supply, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Gate A2 — with_j / with_nuc (upstream :317-340, :342-363; 2e-9 at :340/:363).
// ---------------------------------------------------------------------------

/// Gate A2 — `test_get_j` (`test_rks_stress.py:317-340`, bound 2e-9 at
/// `:340`): `with_j` strain vs FD of `0.5·Tr(dm·vj) + exc`.
#[test]
fn gate_a2_get_j_matches_fd() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dat = get_vxc(
        &cell,
        &dm,
        "lda,vwn",
        VxcStrainOpts {
            with_j: true,
            ..Default::default()
        },
    )
    .expect("get_vxc with_j");
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let vj1 = fftdf_j(&pair.plus, &dm);
        let vj2 = fftdf_j(&pair.minus, &dm);
        let mut dv = vec![0.0; nao * nao];
        for (d, (a, b)) in dv.iter_mut().zip(vj1.iter().zip(vj2.iter())) {
            *d = oracle_sum(&[*a, -*b]);
        }
        let de = oracle_sum(&[
            0.5 * trace_dot(&dm, &dv, nao),
            oracle_sum(&[
                nr_rks_exc(&pair.plus, "lda,vwn", &dm),
                -nr_rks_exc(&pair.minus, "lda,vwn", &dm),
            ]),
        ]);
        let d = (dat[x][y] - de / FULL_DISP).abs();
        eprintln!("get_j ({x},{y}): {d:.3e}");
        assert!(
            d < 2e-9,
            "get_j ({x},{y}): analytic vs FD = {d:.3e} (A2 = 2e-9)"
        );
    }
}

/// Gate A2 — `test_get_nuc` (`test_rks_stress.py:342-363`, bound 2e-9 at
/// `:363`): all-electron `with_nuc` strain vs FD of `Tr(dm·vnuc) + exc`.
#[test]
fn gate_a2_get_nuc_matches_fd() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dat = get_vxc(
        &cell,
        &dm,
        "lda,vwn",
        VxcStrainOpts {
            with_nuc: true,
            ..Default::default()
        },
    )
    .expect("get_vxc with_nuc");
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let vn1 = fftdf_nuc(&pair.plus);
        let vn2 = fftdf_nuc(&pair.minus);
        let mut dv = vec![0.0; nao * nao];
        for (d, (a, b)) in dv.iter_mut().zip(vn1.iter().zip(vn2.iter())) {
            *d = oracle_sum(&[*a, -*b]);
        }
        let de = oracle_sum(&[
            trace_dot(&dm, &dv, nao),
            oracle_sum(&[
                nr_rks_exc(&pair.plus, "lda,vwn", &dm),
                -nr_rks_exc(&pair.minus, "lda,vwn", &dm),
            ]),
        ]);
        let d = (dat[x][y] - de / FULL_DISP).abs();
        eprintln!("get_nuc ({x},{y}): {d:.3e}");
        assert!(
            d < 2e-9,
            "get_nuc ({x},{y}): analytic vs FD = {d:.3e} (A2 = 2e-9)"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate A3 — get_pp (upstream :365-388, bound 1e-8 at :388).
//
// Upstream's PP fixture is Si/C under `gth-pade`; the port's `diamond()`
// reference cell (C2, `gth-szv`/`gth-pade`) is the same physics — the gate is
// analytic-vs-own-FD, so the tier (not the fixture) is what must match.
// ---------------------------------------------------------------------------

/// Gate A3 — `test_get_pp` (`test_rks_stress.py:365-388`, bound 1e-8 at
/// `:388`): PP `with_nuc` strain vs FD of `Tr(dm·vpp) + exc`.
#[test]
fn gate_a3_get_pp_matches_fd() {
    let cell = pyscf_pbc_gto::test_systems::diamond();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x9e37);
    let dat = get_vxc(
        &cell,
        &dm,
        "lda,vwn",
        VxcStrainOpts {
            with_nuc: true,
            ..Default::default()
        },
    )
    .expect("get_vxc with_nuc (PP)");
    for (x, y) in ALL_NINE {
        let pair = finite_diff_cells(&cell, &[], x, y, FULL_DISP).expect("strain pair");
        let vp1 = fftdf_pp(&pair.plus);
        let vp2 = fftdf_pp(&pair.minus);
        let mut dv = vec![0.0; nao * nao];
        for (d, (a, b)) in dv.iter_mut().zip(vp1.iter().zip(vp2.iter())) {
            *d = oracle_sum(&[*a, -*b]);
        }
        let de = oracle_sum(&[
            trace_dot(&dm, &dv, nao),
            oracle_sum(&[
                nr_rks_exc(&pair.plus, "lda,vwn", &dm),
                -nr_rks_exc(&pair.minus, "lda,vwn", &dm),
            ]),
        ]);
        let d = (dat[x][y] - de / FULL_DISP).abs();
        eprintln!("get_pp ({x},{y}): {d:.3e}");
        assert!(
            d < 1e-8,
            "get_pp ({x},{y}): analytic vs FD = {d:.3e} (A3 = 1e-8)"
        );
    }
}

// ---------------------------------------------------------------------------
// Clause-7 determinism: the block accumulator is bit-identical at
// RAYON_NUM_THREADS = 1 and 8 (the standing 18-12 requirement, extended to
// the fused XC loop). The budget pins a multi-block partition so the
// accumulator actually runs; blocks are visited serially in grid order and
// every reduction is `oracle_sum`, so the transcript cannot depend on the
// thread count. A second assertion documents the (expected, non-bit)
// summation-order effect across partition counts, far inside A1.
// ---------------------------------------------------------------------------

/// Clause 7 — fused XC accumulation is bit-identical across thread counts.
#[test]
fn vxc_accumulator_is_bit_identical_across_rayon_threads() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let ngrids = cell.uniform_grids(None).expect("grids").coords.len();
    // ~4 blocks: the accumulator runs, without one-block-per-grid slowness.
    let mb = ((ngrids / 4).max(1) * nao * 13 * 8) as f64 / 1e6;
    let opts = VxcStrainOpts {
        max_memory_mb: Some(mb),
        ..Default::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| get_vxc(&cell, &dm, "lda,vwn", opts).expect("get_vxc"))
    };
    let one = run(1);
    let eight = run(8);
    for x in 0..3 {
        for y in 0..3 {
            assert_eq!(
                one[x][y].to_bits(),
                eight[x][y].to_bits(),
                "block accumulator ({x},{y}) differs across thread counts"
            );
        }
    }
    // Across partition counts the tree shape moves (D-PBC-17): agreement far
    // inside A1, not bit-identity.
    let whole = get_vxc(&cell, &dm, "lda,vwn", VxcStrainOpts::default()).expect("whole grid");
    for x in 0..3 {
        for y in 0..3 {
            let d = (one[x][y] - whole[x][y]).abs();
            assert!(
                d < 1e-12,
                "partition-count drift ({x},{y}) = {d:.3e}, must sit far inside A1"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gate D — end to end (upstream :390-442, 1e-6 Ha/Bohr^3 at :406/:424/:442).
//
// `|dat[i,j] − (E₊−E₋)/2h/vol|`, `h = 1e-3` (half step; `finite_diff_cells`
// takes the FULL separation 2e-3). `vol` is named in the expression — the
// result is a pressure, never an energy (18-CONTEXT §2.2).
// ---------------------------------------------------------------------------

/// Upstream's Gate-D LDA fixture (`:391-396`): H2 on the seed-5 `a = 3`
/// lattice, `svwn` (= `"lda,vwn"`), basis `[[0, [1.5, 1]], [1, [.8, 1]]]`.
fn gate_d_h_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("H 1 1 1; H 2 1.5 2.4".into()),
            basis: BasisInput::Parsed(ParsedBasis {
                shells: vec![
                    ShellSpec {
                        l: 0,
                        exponents: vec![1.5],
                        coeffs: vec![vec![1.0]],
                    },
                    ShellSpec {
                        l: 1,
                        exponents: vec![0.8],
                        coeffs: vec![vec![1.0]],
                    },
                ],
            }),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(SEED5_A3),
        mesh: None,
        precision: 1e-10,
        pseudo: None,
        ..Default::default()
    })
    .expect("Gate-D H cell builds")
}

fn krks_energy(cell: &Cell, xc: &str) -> (f64, Vec<f64>, Vec<f64>) {
    let mf = Krks::new(cell.clone(), &[], xc).expect("krks builds");
    let res = mf.run().expect("krks converges");
    assert!(res.converged, "Gate-D SCF must converge");
    let k = res.idx(0, 0);
    let nao = cell.mol.nao_nr;
    let dm0 = res.dm[0][0].re.clone();
    let dme0 = gamma_make_rdm1e(&res.mo_coeff[k].re, &res.mo_energy[k], &res.mo_occ[k], nao)
        .expect("dme0");
    (res.e_tot, dm0, dme0)
}

/// Gate D — `test_lda_vs_finite_difference`
/// (`test_rks_stress.py:390-406`, bound 1e-6 Ha/Bohr³ at `:406`).
#[test]
fn gate_d_lda_stress_matches_scf_fd_over_vol() {
    let cell = gate_d_h_cell();
    let xc = "lda,vwn";
    let (_, dm0, dme0) = krks_energy(&cell, xc);
    let dat = rks_stress_kernel(&cell, &dm0, &dme0, xc).expect("stress kernel");
    let vol = cell.vol();
    for (x, y) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)] {
        let pair = finite_diff_cells(&cell, &[], x, y, 2e-3).expect("strain pair");
        let e1 = Krks::new(pair.plus, &[], xc)
            .expect("krks plus")
            .run()
            .expect("plus converges")
            .e_tot;
        let e2 = Krks::new(pair.minus, &[], xc)
            .expect("krks minus")
            .run()
            .expect("minus converges")
            .e_tot;
        // Upstream's own acceptance, dimension included: Ha/Bohr³, `vol` named.
        let d = (dat[x][y] - oracle_sum(&[e1, -e2]) / 2e-3 / vol).abs();
        eprintln!("Gate D lda ({x},{y}): {d:.3e} Ha/Bohr^3");
        assert!(
            d < 1e-6,
            "Gate D lda ({x},{y}): kernel vs SCF FD = {d:.3e} (Gate D = 1e-6 Ha/Bohr^3)"
        );
    }
}

/// Gate D at GGA has no port: `kernel` refuses through `get_vxc` (missing
/// `deriv2` kernel). Pins the refusal; the upstream number is
/// `test_gga_vs_finite_difference` (`:408-424`, 1e-6 at `:424`).
#[test]
fn gate_d_gga_refuses_without_deriv2() {
    let cell = gate_d_h_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dme = sym_dm(nao, 0x77aa);
    let err = rks_stress_kernel(&cell, &dm, &dme, "pbe").expect_err("GGA kernel must refuse");
    assert!(
        err.to_string().contains("deriv2"),
        "GGA kernel refusal must name deriv2, got: {err}"
    );
}

/// Gate D at MGGA has no port either (missing `tau`/`vtau` + `deriv2`).
/// Upstream: `test_mgga_vs_finite_difference_high_cost` (`:426-442`, 1e-6).
#[test]
fn gate_d_mgga_refuses_without_tau() {
    let cell = gate_d_h_cell();
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dme = sym_dm(nao, 0x77aa);
    let err = rks_stress_kernel(&cell, &dm, &dme, "scan").expect_err("MGGA kernel must refuse");
    assert!(
        err.to_string().contains("tau"),
        "MGGA kernel refusal must name tau, got: {err}"
    );
}

/// `kernel` refuses hybrid functionals, mirroring upstream's
/// `NotImplementedError` (`rks_stress.py:423-424`).
#[test]
fn stress_kernel_refuses_hybrid() {
    let cell = seed5_he_cell(he_sp_basis(), false);
    let nao = cell.mol.nao_nr;
    let dm = sym_dm(nao, 0x51ab);
    let dme = sym_dm(nao, 0x77aa);
    let err = rks_stress_kernel(&cell, &dm, &dme, "pbe0").expect_err("hybrid kernel must refuse");
    assert!(
        err.to_string().contains("hybrid"),
        "hybrid refusal must name hybrid DFT, got: {err}"
    );
}
