//! Integration tests for `pyscf_pbc_symm::symmetry` — 17-03-PLAN.md
//! Tasks 3-6.
//!
//! # Task 3 — the AO rotation is pinned by the identity that DEFINES it
//!
//! `R(op)·S·R(op)ᴴ == S` (S = the Γ-point overlap) is checked for every op
//! on all five §9.2 reference cells, and the homomorphism identity
//! `R(op1)·R(op2) == R(op1∘op2)` over the whole group — NOT a round-trip.
//! This is the analogue of 14-05's missing `V j2c Vᴴ = I` regression test
//! (17-CONTEXT §3.2): it needs no oracle, and it catches a transposed AO
//! block, a wrong `aoslice` offset, or a wrong `l` convention.
//!
//! # Task 6 — never compare `mo_coeff` elementwise (17-CONTEXT §3.1)
//!
//! `transform_mo_coeff` is only defined up to a unitary mixing WITHIN each
//! degenerate subspace. `diamond`'s Γ-point RHF spectrum has two exactly
//! triply-degenerate levels (verified against live upstream 2.12.1:
//! `[-0.610, 0.293, 0.293, 0.293, 1.160, 1.160, 1.160, 1.526]`), so this
//! fixture is guaranteed to exercise that degeneracy. Every comparison here
//! goes through the density matrix `transform_mo_coeff` builds
//! (`make_rdm1`), never through `mo_coeff` directly, and
//! [`mo_coeff_elementwise_comparison_fails_on_a_degenerate_level`] asserts
//! the elementwise comparison DOES fail — so a later reader cannot "fix" the
//! DM comparison into an MO one.

use num_complex::Complex64;
use pyscf_algebra::CTensor;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::test_systems::{diamond, graphene, he_fcc, lif, si};
use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult, Krhf};
use pyscf_pbc_symm::space_group::SPGElement;
use pyscf_pbc_symm::symmetry::{
    self, DmatSet, Symmetry, check_mesh_symmetry, get_rotation_mat, make_dmats,
    transform_1e_operator, transform_dm, transform_mo_coeff,
};

const TOL: f64 = 1e-6;

// ---------------------------------------------------------------------
// layout helpers — see the module doc / 17-CONTEXT §3.2: `pbc_intor`'s
// output and `mo_coeff` are F-ORDER (column-major); `KMats`/`KDms` (S, dm,
// h1e as read back from a converged `KScfResult`) are ROW-MAJOR. Every
// conversion happens ONCE here, at the boundary.
// ---------------------------------------------------------------------

/// F-order (column-major) `n x n` -> row-major `Vec<Complex64>`.
fn f_order_square_to_rowmajor(ct: &CTensor, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let f_idx = i + j * n;
            out[i * n + j] = Complex64::new(ct.re[f_idx], ct.im[f_idx]);
        }
    }
    out
}

/// Row-major (C-order) `n x n` `CTensor` -> `Vec<Complex64>` (plain copy).
fn rowmajor_square(ct: &CTensor, n: usize) -> Vec<Complex64> {
    (0..n * n)
        .map(|k| Complex64::new(ct.re[k], ct.im[k]))
        .collect()
}

/// Column-major `nrows x ncols` -> row-major `Vec<Complex64>` — `mo_coeff`'s
/// layout (`pyscf-pbc-scf/src/types.rs:119`).
fn colmajor_rect_to_rowmajor(ct: &CTensor, nrows: usize, ncols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    for row in 0..nrows {
        for col in 0..ncols {
            let f_idx = row + col * nrows;
            out[row * ncols + col] = Complex64::new(ct.re[f_idx], ct.im[f_idx]);
        }
    }
    out
}

fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn identity(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i * n + i] = Complex64::new(1.0, 0.0);
    }
    out
}

fn cmatmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn cdagger(a: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j].conj();
        }
    }
    out
}

// ---------------------------------------------------------------------
// Task 3 — R S Rᴴ = S and the homomorphism identity, all five §9.2 cells
// ---------------------------------------------------------------------

fn gamma_overlap_rowmajor(cell: &Cell) -> (Vec<Complex64>, usize) {
    let nao = cell.mol.nao_nr;
    let s = pyscf_pbc_gto::hcore::get_ovlp(cell, &[]).expect("Gamma overlap");
    (f_order_square_to_rowmajor(&s[0], nao), nao)
}

/// The op list and Dmats to exercise: the FULL space group with
/// `check_mesh_symmetry = false`, so the diamond glide is exercised too —
/// the strongest version of this test. `check_mesh_symmetry = false` (rather
/// than the default `true`) is deliberate here: [`get_rotation_mat`]/`S`
/// (the Γ overlap) are ANALYTIC (`cintx`-evaluated), not FFT-mesh-derived,
/// so there is no mesh-compatibility reason to drop a non-symmorphic op —
/// diamond's default `cell.mesh = [47, 47, 47]` is not a multiple of 4 and
/// so is NOT compatible with its 1/4-fractional glide (17-CONTEXT §3.3), and
/// `check_mesh_symmetry = true` (the default) would silently reduce this to
/// the 24-op symmorphic subgroup. See
/// [`symmetry_build_check_mesh_symmetry_true_reduces_to_the_symmorphic_subgroup_on_diamonds_default_mesh`]
/// for the complementary Task 4 test that pins that reduction.
fn full_group(cell: &Cell) -> Symmetry {
    Symmetry::build(cell, true, false, false).expect("Symmetry::build")
}

fn assert_r_s_rh_equals_s(cell: &Cell) {
    let (s, nao) = gamma_overlap_rowmajor(cell);
    let sym = full_group(cell);
    for (iop, op) in sym.ops.iter().enumerate() {
        let dmats = &sym.dmats[iop];
        let r = get_rotation_mat(cell, [0.0, 0.0, 0.0], nao, op, dmats, false, TOL)
            .unwrap_or_else(|e| panic!("get_rotation_mat failed for op {iop}: {e}"));
        let rh = cdagger(&r, nao);
        let out = cmatmul(&cmatmul(&r, &s, nao), &rh, nao);
        let d = max_abs_diff(&out, &s);
        assert!(
            d < 1e-10,
            "R S Rᴴ != S for op {iop} (rot={:?} trans={:?}): max diff {d:e}",
            op.rot,
            op.trans
        );
    }
}

#[test]
fn r_s_rh_equals_s_diamond() {
    assert_r_s_rh_equals_s(&diamond());
}
#[test]
fn r_s_rh_equals_s_si() {
    assert_r_s_rh_equals_s(&si());
}
#[test]
fn r_s_rh_equals_s_lif() {
    assert_r_s_rh_equals_s(&lif());
}
#[test]
fn r_s_rh_equals_s_he_fcc() {
    assert_r_s_rh_equals_s(&he_fcc());
}
#[test]
fn r_s_rh_equals_s_graphene() {
    assert_r_s_rh_equals_s(&graphene());
}

/// `R(op1) R(op2) == R(op1 ∘ op2)` over the whole group — a representation
/// that is not a homomorphism is not a representation. Run on `diamond`
/// (non-symmorphic — exercises the fractional-translation phase too) and
/// `si` (symmorphic-equivalent rotations, different lattice constant).
fn assert_homomorphism(cell: &Cell) {
    let nao = cell.mol.nao_nr;
    let sym = full_group(cell);
    let n = sym.ops.len();
    // A full n^2 sweep is O(nop^2) `get_rotation_mat` calls (<= 48^2 = 2304
    // for these fixtures) — cheap enough to run exhaustively rather than
    // spot-check.
    for i in 0..n {
        for j in 0..n {
            let op1 = &sym.ops[i];
            let op2 = &sym.ops[j];
            let composed = op1.dot(op2);
            // `SPGElement::dot` (matching upstream `space_group.py:103-117`
            // exactly, RULE 2) does NOT reduce the resulting translation mod
            // 1 — so `composed` need not be bit-identical to (or even the
            // same REPRESENTATIVE, mod a lattice vector, as) any member of
            // `sym.ops`'s canonical `[0,1)`-reduced list. That is fine:
            // `get_rotation_mat`/`get_phase` are correct for ANY valid
            // symmetry operation, canonical-range or not (`get_phase`'s
            // `round_to_cell0` wraps the atom search into the cell
            // regardless), so this test builds `composed`'s Dmats directly
            // via `a2r` + `make_dmats` rather than requiring a
            // group-membership lookup.
            let composed_rot = composed.a2r(cell).expect("a2r").rot;
            let composed_dmats = make_dmats(cell, &[composed_rot], None)
                .0
                .into_iter()
                .next()
                .unwrap();

            let r1 = get_rotation_mat(cell, [0.0, 0.0, 0.0], nao, op1, &sym.dmats[i], false, TOL)
                .expect("R(op1)");
            let r2 = get_rotation_mat(cell, [0.0, 0.0, 0.0], nao, op2, &sym.dmats[j], false, TOL)
                .expect("R(op2)");
            let r12 = cmatmul(&r1, &r2, nao);

            let r_composed = get_rotation_mat(
                cell,
                [0.0, 0.0, 0.0],
                nao,
                &composed,
                &composed_dmats,
                false,
                TOL,
            )
            .expect("R(op1 . op2)");

            let d = max_abs_diff(&r12, &r_composed);
            assert!(
                d < 1e-8,
                "R(op{i}) R(op{j}) != R(op{i} . op{j}): max diff {d:e}"
            );
        }
    }
}

#[test]
fn homomorphism_diamond() {
    assert_homomorphism(&diamond());
}

#[test]
fn homomorphism_si() {
    assert_homomorphism(&si());
}

/// `is_eye(op) => R == I` to 1e-14.
#[test]
fn identity_op_gives_identity_rotation() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let sym = full_group(&cell);
    let (idx, _) = sym
        .ops
        .iter()
        .enumerate()
        .find(|(_, op)| op.is_eye())
        .expect("the group must contain the identity");
    let r = get_rotation_mat(
        &cell,
        [0.0, 0.0, 0.0],
        nao,
        &sym.ops[idx],
        &sym.dmats[idx],
        false,
        TOL,
    )
    .expect("R(identity)");
    let i = identity(nao);
    let d = max_abs_diff(&r, &i);
    assert!(d < 1e-14, "R(identity) != I: max diff {d:e}");
}

// ---------------------------------------------------------------------
// Task 4 — check_mesh_symmetry
// ---------------------------------------------------------------------

/// A symmorphic group (every op has `trans_is_zero() == true`) never removes
/// an op or grows the mesh, for ANY mesh.
#[test]
fn check_mesh_symmetry_is_a_noop_for_a_symmorphic_group() {
    let cell = lif();
    let sg = pyscf_pbc_symm::space_group::SpaceGroup::build(&cell, 1e-6).expect("space group");
    assert!(sg.ops.iter().all(|op| op.trans_is_zero()));
    for mesh in [[7usize, 7, 7], [8, 8, 8], [11, 13, 9]] {
        let (rm_list, mesh1) = check_mesh_symmetry(&cell, &sg.ops, Some(mesh), 1e-6, true);
        assert!(rm_list.is_empty(), "mesh {mesh:?}: rm_list must be empty");
        assert_eq!(mesh1.unwrap(), mesh);
    }
}

/// Diamond's glide (`trans = (1/4, 1/4, 1/4)`) is incompatible with a mesh
/// whose components are not all multiples of 4 (`trans * mesh` must be
/// integer). A mesh of `[6, 6, 6]` removes it; the recommended mesh1 that
/// `check_mesh_symmetry` proposes must ITSELF be compatible with every op
/// (running the check again against `mesh1` must find nothing to remove).
#[test]
fn check_mesh_symmetry_grows_a_mesh_incompatible_with_the_glide() {
    let cell = diamond();
    let sg = pyscf_pbc_symm::space_group::SpaceGroup::build(&cell, 1e-6).expect("space group");
    assert!(
        sg.ops.iter().any(|op| !op.trans_is_zero()),
        "diamond must have a glide"
    );

    let mesh = [6usize, 6, 6];
    let (rm_list, mesh1) = check_mesh_symmetry(&cell, &sg.ops, Some(mesh), 1e-6, true);
    assert!(
        !rm_list.is_empty(),
        "the glide must be incompatible with mesh {mesh:?}"
    );
    let mesh1 = mesh1.unwrap();
    assert!(
        mesh1.iter().zip(mesh.iter()).all(|(&a, &b)| a >= b),
        "mesh1 {mesh1:?} must be no smaller than mesh {mesh:?}"
    );

    // mesh1 must carry the FULL group.
    let (rm_list2, _) = check_mesh_symmetry(&cell, &sg.ops, Some(mesh1), 1e-6, true);
    assert!(
        rm_list2.is_empty(),
        "mesh1 {mesh1:?} must be compatible with every op, but rm_list2 = {rm_list2:?}"
    );
}

// ---------------------------------------------------------------------
// Task 5 — Symmetry::build
// ---------------------------------------------------------------------

#[test]
fn symmetry_build_without_space_group_symmetry_is_identity_only() {
    let cell = diamond();
    let sym = Symmetry::build(&cell, false, false, true).expect("Symmetry::build");
    assert_eq!(sym.nop, 1);
    assert!(sym.ops[0].is_eye());
    assert!(sym.spacegroup.is_none());
}

#[test]
fn symmetry_build_symmorphic_true_keeps_only_zero_translation_ops() {
    let cell = diamond();
    let sym = Symmetry::build(&cell, true, true, true).expect("Symmetry::build");
    assert_eq!(sym.nop, 24, "diamond's symmorphic subgroup has 24 ops");
    assert!(sym.ops.iter().all(|op| op.trans_is_zero()));
}

/// `check_mesh_symmetry = false` ("raise the mesh to fit the group", the
/// non-default branch) keeps the FULL non-symmorphic group regardless of
/// whether `cell.mesh` happens to carry it.
#[test]
fn symmetry_build_check_mesh_symmetry_false_keeps_the_full_group() {
    let cell = diamond();
    let sym = Symmetry::build(&cell, true, false, false).expect("Symmetry::build");
    assert_eq!(sym.nop, 48);
    assert!(sym.ops.iter().any(|op| !op.trans_is_zero()));
    assert_eq!(sym.dmats.len(), sym.nop);
}

/// `check_mesh_symmetry = true` (the default, "lower the group to fit the
/// mesh") REDUCES diamond's group to its 24-op symmorphic subgroup on this
/// fixture's default `cell.mesh = [47, 47, 47]`, which is not a multiple of
/// 4 and so cannot carry the 1/4-fractional glide (17-CONTEXT §3.3). This is
/// the complementary case to
/// [`symmetry_build_check_mesh_symmetry_false_keeps_the_full_group`] — Task
/// 4's point that "both branches ship; the flag is not a nicety".
#[test]
fn symmetry_build_check_mesh_symmetry_true_reduces_to_the_symmorphic_subgroup_on_diamonds_default_mesh()
 {
    let cell = diamond();
    assert_eq!(
        cell.mesh,
        [47, 47, 47],
        "this test pins the specific mesh the claim depends on"
    );
    let sym = Symmetry::build(&cell, true, false, true).expect("Symmetry::build");
    assert_eq!(sym.nop, 24);
    assert!(sym.ops.iter().all(|op| op.trans_is_zero()));
}

/// `m-3m` (diamond/si/lif/he_fcc) contains the inversion; `6mm` (graphene)
/// does not. Uses `check_mesh_symmetry = false` so a mesh incompatible with
/// a non-symmorphic inversion element (as on `diamond`, see the test above)
/// does not silently hide it — `has_inversion` is a property of the
/// CRYSTAL's space group, not of a particular FFT mesh's compatibility with
/// it.
#[test]
fn has_inversion_matches_point_group() {
    for (name, cell) in [
        ("diamond", diamond()),
        ("si", si()),
        ("lif", lif()),
        ("he_fcc", he_fcc()),
    ] {
        let sym = Symmetry::build(&cell, true, false, false).expect("Symmetry::build");
        assert!(sym.has_inversion, "{name} (m-3m) must have an inversion op");
    }
    let sym = Symmetry::build(&graphene(), true, false, false).expect("Symmetry::build");
    assert!(
        !sym.has_inversion,
        "graphene (6mm) must NOT have an inversion op"
    );
}

/// `build_lattice_symmetry` populates `Cell::lattice_symmetry` and — since
/// `check_mesh_symmetry = true` — never changes `cell.mesh`.
#[test]
fn build_lattice_symmetry_wires_cell_lattice_symmetry() {
    let mut cell = diamond();
    let mesh_before = cell.mesh;
    symmetry::build_lattice_symmetry(&mut cell, true).expect("build_lattice_symmetry");
    let sym = cell
        .lattice_symmetry
        .as_ref()
        .expect("lattice_symmetry must be Some");
    assert_eq!(sym.point_group_symbol, "m-3m");
    assert_eq!(
        cell.mesh, mesh_before,
        "check_mesh_symmetry=true must not change the mesh"
    );
}

// ---------------------------------------------------------------------
// Task 6 — the three transforms, against ONE converged Gamma-only KRHF
// ---------------------------------------------------------------------

struct Converged {
    cell: Cell,
    nao: usize,
    s: Vec<Complex64>,
    dm: Vec<Complex64>,
    mo_coeff: Vec<Complex64>,
    mo_occ: Vec<f64>,
    mo_energy: Vec<f64>,
}

/// Runs ONE Γ-only `KRHF(diamond)` and memoizes it for every test in this
/// file — every Task 6 test below needs the SAME converged reference (they
/// compare different transforms of it, never re-derive it), so paying for
/// the SCF five times over would be pure waste. `OnceLock` makes this safe
/// across `cargo test`'s parallel test threads (first caller blocks the
/// others until the SCF is done; every caller after that gets the cached
/// result).
fn converged_diamond_gamma() -> &'static Converged {
    static CONVERGED: std::sync::OnceLock<Converged> = std::sync::OnceLock::new();
    CONVERGED.get_or_init(|| {
        let cell = diamond();
        let df = Fftdf::new(cell.clone(), &[]).expect("FFTDF");
        let mf = Krhf::from_df(Box::new(df));
        let cfg = KScfConfig {
            conv_tol: 1e-11,
            conv_tol_grad: Some(1e-8),
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        };
        let r: KScfResult = mf.kernel(&cfg).expect("KRHF must converge");
        assert!(r.converged, "KRHF did not converge in {} cycles", r.cycles);
        let nao = cell.mol.nao_nr;
        let s = f_order_square_to_rowmajor(
            &pyscf_pbc_gto::hcore::get_ovlp(&cell, &[]).expect("ovlp")[0],
            nao,
        );
        let dm = rowmajor_square(&r.dm[0][0], nao);
        let mo_coeff = colmajor_rect_to_rowmajor(&r.mo_coeff[0], nao, nao);
        let mo_occ = r.mo_occ[0].clone();
        let mo_energy = r.mo_energy[0].clone();
        Converged {
            cell,
            nao,
            s,
            dm,
            mo_coeff,
            mo_occ,
            mo_energy,
        }
    })
}

/// Build a density matrix (row-major `nao x nao`) from a row-major
/// `mo_coeff` (`nao x nmo`) and occupations — the ONLY way `mo_coeff` is
/// ever compared in this test file (17-CONTEXT §3.1).
fn make_rdm1_rowmajor(
    mo_coeff: &[Complex64],
    occ: &[f64],
    nao: usize,
    nmo: usize,
) -> Vec<Complex64> {
    let mut dm = vec![Complex64::new(0.0, 0.0); nao * nao];
    for (i, &o) in occ.iter().enumerate() {
        if o == 0.0 {
            continue;
        }
        for mu in 0..nao {
            let a = mo_coeff[mu * nmo + i];
            for nu in 0..nao {
                let b = mo_coeff[nu * nmo + i].conj();
                dm[mu * nao + nu] += Complex64::new(o, 0.0) * a * b;
            }
        }
    }
    dm
}

fn trace_ab(a: &[Complex64], b: &[Complex64], n: usize) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            s += a[i * n + j] * b[j * n + i];
        }
    }
    s
}

/// `transform_dm(dm, op)` preserves `Tr(D S)` for every op — and, since the
/// Γ density of a closed-shell ground state is invariant under the FULL
/// point group, `transform_dm(dm, op) == dm` itself.
#[test]
fn transform_dm_preserves_trace_ds_and_is_group_invariant() {
    let c = converged_diamond_gamma();
    let want_tr = trace_ab(&c.dm, &c.s, c.nao);
    let sym = Symmetry::build(&c.cell, true, false, true).expect("Symmetry::build");
    for (iop, op) in sym.ops.iter().enumerate() {
        let dm2 = transform_dm(&c.cell, [0.0, 0.0, 0.0], &c.dm, c.nao, op, &sym.dmats[iop])
            .unwrap_or_else(|e| panic!("transform_dm failed for op {iop}: {e}"));
        let tr = trace_ab(&dm2, &c.s, c.nao);
        assert!(
            (tr - want_tr).norm() < 1e-8,
            "Tr(D S) not preserved by op {iop}: {tr:?} != {want_tr:?}"
        );
        let d = max_abs_diff(&dm2, &c.dm);
        assert!(
            d < 1e-7,
            "transform_dm(dm, op{iop}) != dm (Gamma density must be group-invariant): {d:e}"
        );
    }
}

/// `transform_1e_operator(S, op) == S` — the overlap is invariant. This is
/// the SAME identity Task 3 pins directly through [`get_rotation_mat`], now
/// exercised through the public production entry point.
#[test]
fn transform_1e_operator_leaves_overlap_invariant() {
    let c = converged_diamond_gamma();
    let sym = Symmetry::build(&c.cell, true, false, true).expect("Symmetry::build");
    for (iop, op) in sym.ops.iter().enumerate() {
        let s2 = transform_1e_operator(&c.cell, [0.0, 0.0, 0.0], &c.s, c.nao, op, &sym.dmats[iop])
            .unwrap_or_else(|e| panic!("transform_1e_operator failed for op {iop}: {e}"));
        let d = max_abs_diff(&s2, &c.s);
        assert!(d < 1e-8, "transform_1e_operator(S, op{iop}) != S: {d:e}");
    }
}

/// `transform_dm` is idempotent under `op ∘ op⁻¹`.
#[test]
fn transform_dm_idempotent_under_op_then_its_inverse() {
    let c = converged_diamond_gamma();
    let sym = Symmetry::build(&c.cell, true, false, true).expect("Symmetry::build");
    let dmats_for = |op: &SPGElement| -> DmatSet {
        let op_rot = op.a2r(&c.cell).expect("a2r").rot;
        make_dmats(&c.cell, &[op_rot], None)
            .0
            .into_iter()
            .next()
            .unwrap()
    };
    for (iop, op) in sym.ops.iter().enumerate() {
        let inv = op.inv().expect("inv");
        let dmats_inv = dmats_for(&inv);
        let dm2 = transform_dm(&c.cell, [0.0, 0.0, 0.0], &c.dm, c.nao, op, &sym.dmats[iop])
            .expect("transform_dm(op)");
        let dm3 = transform_dm(&c.cell, [0.0, 0.0, 0.0], &dm2, c.nao, &inv, &dmats_inv)
            .expect("transform_dm(op^-1)");
        let d = max_abs_diff(&dm3, &c.dm);
        assert!(
            d < 1e-7,
            "transform_dm(., op then op^-1) != dm for op {iop}: {d:e}"
        );
    }
}

/// **17-CONTEXT §3.1**: `transform_mo_coeff` is compared through the density
/// matrix it builds — NEVER elementwise. For every op, the DM built from
/// `transform_mo_coeff(mo_coeff, op)` must match `transform_dm(dm, op)`
/// (equivalently, the original `dm`, since it is group-invariant here).
#[test]
fn transform_mo_coeff_matches_transform_dm_through_the_density_matrix() {
    let c = converged_diamond_gamma();
    let sym = Symmetry::build(&c.cell, true, false, true).expect("Symmetry::build");
    for (iop, op) in sym.ops.iter().enumerate() {
        let mo2 = transform_mo_coeff(
            &c.cell,
            [0.0, 0.0, 0.0],
            &c.mo_coeff,
            c.nao,
            c.nao,
            op,
            &sym.dmats[iop],
        )
        .unwrap_or_else(|e| panic!("transform_mo_coeff failed for op {iop}: {e}"));
        let dm_from_mo = make_rdm1_rowmajor(&mo2, &c.mo_occ, c.nao, c.nao);
        let dm_from_transform_dm =
            transform_dm(&c.cell, [0.0, 0.0, 0.0], &c.dm, c.nao, op, &sym.dmats[iop])
                .expect("transform_dm");
        let d = max_abs_diff(&dm_from_mo, &dm_from_transform_dm);
        assert!(
            d < 1e-7,
            "DM(transform_mo_coeff(op{iop})) != transform_dm(dm, op{iop}): {d:e}"
        );
    }
}

/// **17-CONTEXT §3.1, the negative half.** `diamond`'s Γ spectrum has an
/// EXACTLY triply-degenerate occupied level (verified against upstream
/// 2.12.1 — see the module doc). `transform_mo_coeff` is only defined up to
/// a unitary mixing within that degenerate subspace, so comparing the
/// rotated `mo_coeff` ELEMENTWISE against a reference must FAIL for at least
/// one non-identity op that mixes it — even though the density-matrix
/// comparison above passes for every op. This test exists so nobody "fixes"
/// [`transform_mo_coeff_matches_transform_dm_through_the_density_matrix`]
/// into an elementwise `mo_coeff` comparison: doing so would make THIS test
/// fail, which is the point.
#[test]
fn mo_coeff_elementwise_comparison_fails_on_a_degenerate_level() {
    let c = converged_diamond_gamma();
    let mo_energy = &c.mo_energy;
    // Locate a degenerate manifold (adjacent MO energies within 1e-6 Ha —
    // diamond has one at the occupied/valence edge, indices 1..=3 above).
    let mut degenerate_cols: Vec<usize> = Vec::new();
    for i in 0..mo_energy.len().saturating_sub(1) {
        if (mo_energy[i + 1] - mo_energy[i]).abs() < 1e-6 {
            degenerate_cols.push(i);
            degenerate_cols.push(i + 1);
        }
    }
    degenerate_cols.sort_unstable();
    degenerate_cols.dedup();
    assert!(
        degenerate_cols.len() >= 2,
        "diamond's Gamma spectrum must contain a degenerate manifold: {mo_energy:?}"
    );

    let sym = Symmetry::build(&c.cell, true, false, true).expect("Symmetry::build");
    let mut found_a_mismatch = false;
    for (iop, op) in sym.ops.iter().enumerate() {
        if op.is_eye() {
            continue;
        }
        let mo2 = transform_mo_coeff(
            &c.cell,
            [0.0, 0.0, 0.0],
            &c.mo_coeff,
            c.nao,
            c.nao,
            op,
            &sym.dmats[iop],
        )
        .expect("transform_mo_coeff");
        for &col in &degenerate_cols {
            let mut col_diff = 0.0_f64;
            for row in 0..c.nao {
                let a = c.mo_coeff[row * c.nao + col];
                let b = mo2[row * c.nao + col];
                col_diff = col_diff.max((a - b).norm());
            }
            if col_diff > 0.05 {
                found_a_mismatch = true;
            }
        }
    }
    assert!(
        found_a_mismatch,
        "expected at least one op to visibly mix the degenerate manifold's mo_coeff columns \
         (elementwise) — if this fails, the fixture no longer exercises 17-CONTEXT §3.1's trap"
    );
}
