//! Plan 18-11 — Gate A (tier A1 = 1e-9) for the strain-tensor AO kernel.
//!
//! Upstream oracle: `pyscf/pbc/grad/test/test_rks_stress.py`, which gates the
//! strain AO against the central difference of the ordinary AO over
//! `_finite_diff_cells` at `disp = 1e-5`:
//!
//! | test here | upstream method | upstream bound (all A1 = 1e-9) |
//! |---|---|---|
//! | `deriv0_cart_matches_fd` | `test_eval_ao_cart` (:125) | `:140` |
//! | `deriv0_sph_matches_fd` | `test_eval_ao_sph` (:142) | `:158` |
//! | `deriv1_cart_matches_fd` | `test_eval_ao_deriv1_cart` (:160) | `:176` |
//! | `deriv1_sph_matches_fd` | `test_eval_ao_deriv1_sph` (:178) | `:194` |
//! | `grid_response_cart_deriv0` | `test_eval_ao_grid_response` 1st half (:196) | `:214` |
//! | `grid_response_sph_deriv1` | `test_eval_ao_grid_response` 2nd half (:226) | `:240` |
//!
//! Upstream checks five `(x,y)` pairs; every test here checks all NINE, at
//! `< 1e-9` on the max over components, grid points and AOs. The fixture is
//! upstream's (He2, even-tempered s/p/d/f shells, non-symmetric lattice,
//! fixed Cartesian probe points); the reference is this port's own
//! `eval_ao_kpts`, so the gate localises a wrong strain weight to a single
//! component with no SCF in the loop (18-CONTEXT Gate A).
//!
//! Two deliberate deltas from a literal transcription:
//!
//! * `deriv0_sph_nonkpt_pinned` covers a non-Gamma k-point, which upstream's
//!   strain tests never do. The displaced k-points are re-derived at FIXED
//!   FRACTIONAL coordinates (`get_scaled_kpts` / `get_abs_kpts`): `k.L` is
//!   strain-invariant there, so the phases the kernel holds fixed stay fixed
//!   in the reference too. Reusing the Cartesian k-points would differentiate
//!   a different quantity — 18-CONTEXT §3 trap 8 (invisible at Gamma).
//! * `grid_response_sph_deriv1` builds the undisplaced deriv-2 table with
//!   `pyscf_kernels::pbc::eval_ao_deriv2` (same primitives, image list and
//!   phases as the strain driver, minus the strain weight). The identity
//!   under test still gates the hessians: a wrong hessian breaks
//!   basis-response + grid-response against the independent cell-Strain FD,
//!   since the two terms weight it by `(R+L)` inside the lattice sum versus
//!   `coords` outside it — no nonzero error cancels across all points.
//!
//! # Pre-existing finding (not 18-11 scope): `GTOval_cart` value is broken
//! for `l >= 2`
//!
//! The value-only Cartesian device path ignores its own `spherical` flag
//! (`crates/pyscf-kernels/src/eval_gto.rs`, `eval_gto_sph_into_screened`:
//! `let _ = spherical` on both the s and general launches), emitting
//! spherical-ordered data into the Cartesian layout from the d shell on.
//! Measured here: molecular `|GTOval_cart - GTOval_cart_deriv1[0]| = 0.61`
//! on this fixture's s/p/d/f basis (exact `0.0` on s/p-only `gth-szv`),
//! periodic `2.1e2`. The `GTOval_cart_deriv1` family honors the flag
//! (`eval_gto_sph_deriv1_cpu`'s documented cart branch) and its value block
//! is independently validated by `deriv1_cart_matches_fd`'s `c = 0`
//! component (strain `grad.R` from proven-independent grads vs the FD of
//! that block, at A1) — so the cart tests below take their value reference
//! from `GTOval_cart_deriv1[0]`, the same quantity. Re-point them at
//! `GTOval_cart` when the owning plan fixes the device path; the strain
//! kernel itself never calls either name.

use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, estimate_rcut_for_eval, eval_ao_kpts, get_lattice_ls,
    get_uniform_grids,
};

/// Gate A tier: A1 = 1e-9 on every component (gate-a-tiers.md; upstream
/// `:140, :158, :176, :194, :214, :240`).
const GATE_A1: f64 = 1e-9;
/// Strain finite-difference step (`rks_stress.py:88,101`, `disp = 1e-5`).
const DISP: f64 = 1e-5;

/// Upstream's even-tempered fixture basis
/// (`test_rks_stress.py:129-132`): s(0.5), p(1.5, 0.5), d(0.8), f(0.7).
/// The `:125/:142/:160/:178` halves use all four shells; the
/// `:196 test_eval_ao_grid_response` halves use s/p/d only (`:199`, `:229`)
/// — mirrored here per half, since the f shell's higher derivatives raise
/// the FD-truncation floor of the strained-grid construction (measured:
/// upstream Python itself sits at 8.7e-10 on s/p/d here).
fn he_basis(include_f: bool) -> BasisInput {
    let mut shells = vec![
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
    ];
    if include_f {
        shells.push(ShellSpec {
            l: 3,
            exponents: vec![0.7],
            coeffs: vec![vec![1.0]],
        });
    }
    BasisInput::Parsed(ParsedBasis { shells })
}

/// Fixed non-symmetric lattice (Bohr) + upstream's He positions
/// (`test_rks_stress.py:126-128`, seed-5 equivalent: non-symmetric).
fn base_lattice() -> [[f64; 3]; 3] {
    [
        [5.231_728_194, 0.123_585_041, -0.341_900_772],
        [0.087_231_550, 4.876_410_933, 0.214_407_305],
        [-0.156_203_418, 0.093_377_012, 5.102_664_589],
    ]
}

fn base_atoms() -> Vec<(String, [f64; 3])> {
    vec![
        ("He".into(), [1.0, 1.0, 1.0]),
        ("He".into(), [2.0, 1.5, 2.4]),
    ]
}

fn build_cell(
    cart: bool,
    a: [[f64; 3]; 3],
    atoms: Vec<(String, [f64; 3])>,
    include_f: bool,
) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: he_basis(include_f),
            unit: Unit::Bohr,
            cart,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        ..Default::default()
    })
    .expect("He strain fixture builds")
}

/// Ten fixed Cartesian probe points (upstream uses `np.random.rand(10, 3)`;
/// fixed here so the gate is seed-free).
fn probe_coords() -> Vec<[f64; 3]> {
    vec![
        [0.13, 0.57, 0.29],
        [0.81, 0.07, 0.63],
        [0.45, 0.91, 0.11],
        [0.67, 0.33, 0.77],
        [0.05, 0.49, 0.95],
        [0.93, 0.71, 0.37],
        [0.29, 0.15, 0.53],
        [0.55, 0.83, 0.21],
        [0.75, 0.41, 0.89],
        [0.17, 0.65, 0.47],
    ]
}

/// `rks_stress._finite_diff_cells` (`rks_stress.py:63-84`): strain the
/// lattice AND the atom positions by `E` (`E[x,y] += disp`),
/// `a1 = a.E^T`, `r1 = r.E^T`. Everything else (basis, precision, mesh
/// handling) is inherited from the undisplaced cell.
fn finite_diff_cells(cell: &Cell, x: usize, y: usize, disp: f64) -> (Cell, Cell) {
    let mut e_plus = [[0.0f64; 3]; 3];
    let mut e_minus = [[0.0f64; 3]; 3];
    for i in 0..3 {
        e_plus[i][i] = 1.0;
        e_minus[i][i] = 1.0;
    }
    e_plus[x][y] += disp;
    e_minus[x][y] -= disp;
    let apply = |e: [[f64; 3]; 3]| {
        // M.E^T for row-vector rows: out[i][j] = sum_k m[i][k]*e[j][k].
        let mat_mul_et = |m: &[[f64; 3]; 3]| {
            let mut out = [[0.0f64; 3]; 3];
            for (i, row) in out.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for k in 0..3 {
                        s += m[i][k] * e[j][k];
                    }
                    *v = s;
                }
            }
            out
        };
        let a1 = mat_mul_et(&cell.lattice_vectors());
        let atoms: Vec<(String, [f64; 3])> = cell
            .mol
            ._atom
            .iter()
            .map(|(sym, r)| {
                let alpha: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
                let mut r1 = [0.0f64; 3];
                for j in 0..3 {
                    let mut s = 0.0;
                    for k in 0..3 {
                        s += r[k] * e[j][k];
                    }
                    r1[j] = s;
                }
                (alpha, r1)
            })
            .collect();
        build_cell(cell.mol.cart, a1, atoms, cell.mol.nbas == 8)
    };
    (apply(e_plus), apply(e_minus))
}

fn strain_eval_name(spherical: bool, deriv: u32) -> &'static str {
    match (spherical, deriv) {
        (true, 0) => "GTOval_sph_deriv0_strain_tensor",
        (false, 0) => "GTOval_cart_deriv0_strain_tensor",
        (true, 1) => "GTOval_sph_deriv1_strain_tensor",
        (false, 1) => "GTOval_cart_deriv1_strain_tensor",
        _ => unreachable!("fixture only queries deriv 0/1"),
    }
}

/// Max `|strain[x,y,c] - (ao1[rb] - ao2[rb]) / 2*disp|` over the nine `(x,y)`,
/// all `comp` blocks, grid points, AOs and k-points. Complex-aware: the
/// reference FD runs on the same k-points, so phases match term by term.
///
/// `ref_block_for[c]` selects the reference table's block for strain
/// component `c` (identity for the sph names; `[0]` for the cart deriv-0
/// gate, whose value reference is `GTOval_cart_deriv1[0]` per the finding
/// above).
fn max_fd_residual(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    spherical: bool,
    deriv: u32,
    ref_name: &str,
    ref_block_for: &[usize],
) -> f64 {
    let out = eval_ao_kpts(cell, strain_eval_name(spherical, deriv), coords, kpts)
        .expect("strain collocation");
    let ncomp = if deriv == 0 { 1 } else { 4 };
    assert_eq!(out.comp, 9 * ncomp, "strain block count");
    let mut worst = 0.0f64;
    for x in 0..3 {
        for y in 0..3 {
            let (c1, c2) = finite_diff_cells(cell, x, y, DISP);
            // K-points at fixed FRACTIONAL coordinates (trap §3.8): `k.L`
            // is strain-invariant there, matching the phases the kernel
            // holds fixed. At Gamma both conventions coincide.
            let k1: Vec<[f64; 3]> = if kpts.is_empty() {
                vec![]
            } else {
                let scaled = cell.get_scaled_kpts(kpts);
                c1.get_abs_kpts(&scaled).expect("kpts map")
            };
            let k2: Vec<[f64; 3]> = if kpts.is_empty() {
                vec![]
            } else {
                let scaled = cell.get_scaled_kpts(kpts);
                c2.get_abs_kpts(&scaled).expect("kpts map")
            };
            let ao1 = eval_ao_kpts(&c1, ref_name, coords, &k1).expect("ref cell1");
            let ao2 = eval_ao_kpts(&c2, ref_name, coords, &k2).expect("ref cell2");
            for k in 0..out.nkpts() {
                for c in 0..ncomp {
                    let b = (x * 3 + y) * ncomp + c;
                    let rb = ref_block_for[c];
                    for g in 0..coords.len() {
                        for mu in 0..out.nao {
                            let (sr, si) = out.element(k, b, g, mu);
                            let (r1, i1) = ao1.element(k, rb, g, mu);
                            let (r2, i2) = ao2.element(k, rb, g, mu);
                            let dr = sr - (r1 - r2) / (2.0 * DISP);
                            let di = si - (i1 - i2) / (2.0 * DISP);
                            worst = worst.max(dr.abs()).max(di.abs());
                        }
                    }
                }
            }
        }
    }
    worst
}

#[test]
fn deriv0_sph_matches_fd() {
    // Upstream `test_eval_ao_sph` (:142), bound `:158`, A1 = 1e-9.
    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let worst = max_fd_residual(&cell, &coords, &[], true, 0, "GTOval_sph", &[0]);
    assert!(
        worst < GATE_A1,
        "deriv0 sph strain vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :158)"
    );
}

#[test]
fn deriv0_cart_matches_fd() {
    // Upstream `test_eval_ao_cart` (:125), bound `:140`, A1 = 1e-9. Value
    // reference is `GTOval_cart_deriv1[0]` (finding above).
    let cell = build_cell(true, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let worst = max_fd_residual(&cell, &coords, &[], false, 0, "GTOval_cart_deriv1", &[0]);
    assert!(
        worst < GATE_A1,
        "deriv0 cart strain vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :140)"
    );
}

#[test]
fn deriv1_sph_matches_fd() {
    // Upstream `test_eval_ao_deriv1_sph` (:178), bound `:194`, A1 = 1e-9.
    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let worst = max_fd_residual(
        &cell,
        &coords,
        &[],
        true,
        1,
        "GTOval_sph_deriv1",
        &[0, 1, 2, 3],
    );
    assert!(
        worst < GATE_A1,
        "deriv1 sph strain vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :194)"
    );
}

#[test]
fn deriv1_cart_matches_fd() {
    // Upstream `test_eval_ao_deriv1_cart` (:160), bound `:176`, A1 = 1e-9.
    let cell = build_cell(true, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let worst = max_fd_residual(
        &cell,
        &coords,
        &[],
        false,
        1,
        "GTOval_cart_deriv1",
        &[0, 1, 2, 3],
    );
    assert!(
        worst < GATE_A1,
        "deriv1 cart strain vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :176)"
    );
}

#[test]
fn deriv0_sph_nonkpt_pinned() {
    // No upstream counterpart (strain tests are Gamma-only): the kernel at a
    // non-Gamma k-point against fractionally-pinned FD (trap §3.8). A1.
    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let kpts = [[0.13, -0.07, 0.11]];
    let worst = max_fd_residual(&cell, &coords, &kpts, true, 0, "GTOval_sph", &[0]);
    assert!(
        worst < GATE_A1,
        "deriv0 sph strain at k!=0 vs pinned FD: {worst:e} >= A1 {GATE_A1:e}"
    );
}

/// Upstream `test_eval_ao_grid_response` first half (:196-224, bound `:214`,
/// A1): `strain[:,:,0] + einsum('xgi,gy->xygi', ao[1:4], coords)` against the
/// FD of the value AO on each displaced cell's OWN uniform grid (the grid
/// strains with the cell — the term under test is exactly the grid's
/// response, owned by the 18-12 caller, not the kernel).
#[test]
fn grid_response_cart_deriv0() {
    let cell = build_cell(true, base_lattice(), base_atoms(), false);
    let mesh = [6, 5, 4];
    let coords = get_uniform_grids(&cell, Some(mesh), true).expect("uniform grids");
    let out =
        eval_ao_kpts(&cell, "GTOval_cart_deriv0_strain_tensor", &coords, &[]).expect("strain");
    let ao = eval_ao_kpts(&cell, "GTOval_cart_deriv1", &coords, &[]).expect("grads");
    let mut worst = 0.0f64;
    for x in 0..3 {
        for y in 0..3 {
            let (c1, c2) = finite_diff_cells(&cell, x, y, DISP);
            let g1 = get_uniform_grids(&c1, Some(mesh), true).expect("grids1");
            let g2 = get_uniform_grids(&c2, Some(mesh), true).expect("grids2");
            // Value FD via `GTOval_cart_deriv1[0]` (finding above).
            let ao1 = eval_ao_kpts(&c1, "GTOval_cart_deriv1", &g1, &[]).expect("ao1");
            let ao2 = eval_ao_kpts(&c2, "GTOval_cart_deriv1", &g2, &[]).expect("ao2");
            for g in 0..coords.len() {
                for mu in 0..out.nao {
                    // lhs = strain[x,y,0] + sum_t grad_t[g,mu]*coords[g][y] over
                    // the gradient axis t==x picked by the einsum's x index.
                    let (sr, _) = out.element(0, x * 3 + y, g, mu);
                    let (gr, _) = ao.element(0, 1 + x, g, mu);
                    let lhs = sr + gr * coords[g][y];
                    let (r1, _) = ao1.element(0, 0, g, mu);
                    let (r2, _) = ao2.element(0, 0, g, mu);
                    worst = worst.max((lhs - (r1 - r2) / (2.0 * DISP)).abs());
                }
            }
        }
    }
    assert!(
        worst < GATE_A1,
        "deriv0 grid response vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :214)"
    );
}

/// Upstream `test_eval_ao_grid_response` second half (:226-240, bound `:240`,
/// A1): the deriv-1 grid response, whose `ao[4:9]` second-derivative terms
/// come from `pyscf_kernels::pbc::eval_ao_deriv2` (the 18-12 caller input).
#[test]
fn grid_response_sph_deriv1() {
    use pyscf_algebra::select_backend;

    let cell = build_cell(false, base_lattice(), base_atoms(), false);
    let mesh = [6, 5, 4];
    let coords = get_uniform_grids(&cell, Some(mesh), true).expect("uniform grids");
    let out = eval_ao_kpts(&cell, "GTOval_sph_deriv1_strain_tensor", &coords, &[]).expect("strain");
    assert_eq!(out.comp, 36);

    // Undisplaced deriv-2 table through the kernels driver directly, with the
    // same image list the wrapper builds (rcut at deriv+1 = 2).
    let sel = select_backend().expect("backend");
    let rcut = estimate_rcut_for_eval(&cell, 2).expect("rcut");
    let rmax = rcut.iter().copied().fold(0.0f64, f64::max);
    let ls = get_lattice_ls(&cell, Some(rmax), None, false).expect("Ls");
    let d2 = pyscf_kernels::pbc::eval_ao_deriv2(
        &sel.client,
        &coords,
        &[],
        &ls,
        &cell.mol._atm,
        &cell.mol._bas,
        &cell.mol._env,
        &cell.mol.atom_coords(),
        true,
        Some(&rcut),
        true,
    )
    .expect("deriv2 table");
    // deriv2 layout: [10, ngrids, nao], value, gx, gy, gz, hxx, hxy, hxz,
    // hyy, hyz, hzz; F-order per component.
    let d2_at =
        |c: usize, g: usize, mu: usize| d2.re[0][c * coords.len() * d2.nao + g + mu * coords.len()];

    let mut worst = 0.0f64;
    for x in 0..3 {
        for y in 0..3 {
            let (c1, c2) = finite_diff_cells(&cell, x, y, DISP);
            let g1 = get_uniform_grids(&c1, Some(mesh), true).expect("grids1");
            let g2 = get_uniform_grids(&c2, Some(mesh), true).expect("grids2");
            let ao1 = eval_ao_kpts(&c1, "GTOval_sph_deriv1", &g1, &[]).expect("ao1");
            let ao2 = eval_ao_kpts(&c2, "GTOval_sph_deriv1", &g2, &[]).expect("ao2");
            for cc in 0..4 {
                for g in 0..coords.len() {
                    for mu in 0..out.nao {
                        let b = (x * 3 + y) * 4 + cc;
                        let (sr, _) = out.element(0, b, g, mu);
                        // Upstream :230-238: c=0 reads ao[1:4], c=1 reads
                        // ao[4:7], c=2 reads the y-row (ao[5], ao[7], ao[8]),
                        // c=3 reads the z-row (ao[6], ao[8], ao[9]).
                        let grid = match cc {
                            0 => d2_at(1 + x, g, mu) * coords[g][y],
                            1 => d2_at(4 + x, g, mu) * coords[g][y],
                            2 => d2_at([5, 7, 8][x], g, mu) * coords[g][y],
                            _ => d2_at([6, 8, 9][x], g, mu) * coords[g][y],
                        };
                        let lhs = sr + grid;
                        let (r1, _) = ao1.element(0, cc, g, mu);
                        let (r2, _) = ao2.element(0, cc, g, mu);
                        worst = worst.max((lhs - (r1 - r2) / (2.0 * DISP)).abs());
                    }
                }
            }
        }
    }
    assert!(
        worst < GATE_A1,
        "deriv1 grid response vs FD: {worst:e} >= A1 {GATE_A1:e} (upstream :240)"
    );
}

/// The bounding-box screen drops only sub-precision tails: screened vs
/// unscreened agree far inside A1 (mirrors `tests/eval_ao_screen.rs`).
#[test]
fn screen_on_off_agree() {
    use pyscf_algebra::select_backend;

    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let sel = select_backend().expect("backend");
    let rcut = estimate_rcut_for_eval(&cell, 1).expect("rcut");
    let rmax = rcut.iter().copied().fold(0.0f64, f64::max);
    let ls = get_lattice_ls(&cell, Some(rmax), None, false).expect("Ls");
    let run = |screen: bool| {
        pyscf_kernels::pbc::eval_strain_ao(
            &sel.client,
            &coords,
            &[],
            &ls,
            &cell.mol._atm,
            &cell.mol._bas,
            &cell.mol._env,
            &cell.mol.atom_coords(),
            0,
            true,
            Some(&rcut),
            screen,
        )
        .expect("strain")
    };
    let on = run(true);
    let off = run(false);
    let mut worst = 0.0f64;
    for (a, b) in on.re[0].iter().zip(off.re[0].iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-11,
        "screen drops more than tails: {worst:e} (bound 1e-11, 100x inside A1)"
    );
}

/// `deriv >= 2` strain names are a named refusal, never a fall-through
/// (18-11 Task 4).
#[test]
fn strain_deriv2_is_refused() {
    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    for name in [
        "GTOval_sph_deriv2_strain_tensor",
        "GTOval_cart_deriv3_strain_tensor",
    ] {
        let err = format!(
            "{:?}",
            eval_ao_kpts(&cell, name, &coords, &[]).expect_err("must refuse")
        );
        assert!(err.contains("strain"), "refusal names the family: {err}");
    }
}

/// Output shape contract: `(3, 3, comp, ngrids, nao)` per k-point
/// (`rks_stress.py:143-146`), flat as `comp = 9*ncomp` blocks.
#[test]
fn strain_output_shape() {
    let cell = build_cell(false, base_lattice(), base_atoms(), true);
    let coords = probe_coords();
    let kpts = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]];
    let d0 = eval_ao_kpts(&cell, "GTOval_sph_deriv0_strain_tensor", &coords, &kpts).expect("d0");
    assert_eq!(d0.comp, 9);
    assert_eq!(d0.ngrids, coords.len());
    assert_eq!(d0.nao, cell.mol.nao_nr);
    assert_eq!(d0.nkpts(), 2);
    assert_eq!(d0.gamma, vec![true, false]);
    // Gamma imaginary plane is dropped (eval_gto.py:157-158).
    assert!(d0.kaos[0].im.iter().all(|&v| v == 0.0));
    let d1 = eval_ao_kpts(&cell, "GTOval_sph_deriv1_strain_tensor", &coords, &kpts).expect("d1");
    assert_eq!(d1.comp, 36);

    let cart = build_cell(true, base_lattice(), base_atoms(), true);
    let dc = eval_ao_kpts(&cart, "GTOval_cart_deriv0_strain_tensor", &coords, &[]).expect("cart");
    assert_eq!(dc.nao, cart.mol.nao_nr);
    assert!(dc.nao > d0.nao, "cart has more AOs than sph for d/f shells");
}
