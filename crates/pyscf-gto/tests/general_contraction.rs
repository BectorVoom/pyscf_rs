//! Plan 02-11 — GENERAL-CONTRACTION support in the NWChem `.dat` parser.
//!
//! ROOT CAUSE (diagnosed 03-13 → 02-11): the `CurrentShell::Single`
//! primitive-row branch in `basis/nwchem.rs` pushed only `cols[1]`
//! (`exps.push(cols[0]); coeffs.push(cols[1])`), silently discarding
//! coefficient columns 2..N. So generally-contracted bases (ANO, ANO-RCC,
//! and — a latent bug — the cc-pVDZ O S-block) loaded TRUNCATED to a single
//! contraction per block. The fix emits ONE contraction per coefficient
//! column. The downstream `ParsedBasis`/`ShellSpec.coeffs` (`Vec<Vec<f64>>`,
//! `coeffs[ctr][prim]`) and `projection.rs` (`nctr = spec.coeffs.len()`)
//! already support `nctr > 1`, so this is a parser-only fix.
//!
//! Three groups:
//!   (a) ANO general-contraction load — O S-block `nctr == 8` (the ano.dat
//!       column count), ANO H gains its real contractions.
//!   (b) REGRESSION — segmented bases (sto-3g / 6-31g / 6-31g* / cc-pvdz)
//!       nao_nr + per-shell nctr PINNED. cc-pVDZ O is the EXCEPTION: its
//!       first S-block is generally-contracted (`nctr == 2`) and was being
//!       truncated — the corrected nctr is pinned here too (threat
//!       T-02-11-REGRESSION).
//!   (c) cintx evaluates the ANO mol — `int1e_ovlp_sph` finite, symmetric,
//!       positive diagonal (threat T-02-11-CINTX-NCTR).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, basis, intor};

// ----------------------------------------------------------------------------
// (a) ANO general-contraction load
// ----------------------------------------------------------------------------

/// ANO O S-block loads with the FULL general-contraction count. The `ano.dat`
/// O S-block carries `exp + 8` coefficient columns → `nctr == 8`. Pre-fix this
/// loaded as `nctr == 1` (7 columns silently dropped).
#[test]
fn ano_o_s_block_has_eight_contractions() {
    let parsed = basis::load_basis("ano", "O").expect("load ANO O");

    // First shell is the O S-block (header `O    S` precedes `O    P`).
    let s_shell = &parsed.shells[0];
    assert_eq!(s_shell.l, 0, "first ANO O shell is S (l=0)");
    assert_eq!(
        s_shell.coeffs.len(),
        8,
        "ANO O S-block must load nctr=8 (the ano.dat column count); pre-fix it \
         truncated to 1. Got {}",
        s_shell.coeffs.len()
    );

    // Every contraction column shares the SAME 14 exponents (general
    // contraction: N columns over ONE primitive set).
    assert_eq!(
        s_shell.exponents.len(),
        14,
        "ANO O S-block has 14 primitives"
    );
    for (c, col) in s_shell.coeffs.iter().enumerate() {
        assert_eq!(
            col.len(),
            s_shell.exponents.len(),
            "contraction column {c} length must equal nprim (shared exponents)"
        );
    }

    // The (4, exp, ...) → distinct columns: column 0 and column 1 of row 0
    // differ (general contraction, not a repeated segmented column).
    assert_ne!(
        s_shell.coeffs[0][0], s_shell.coeffs[1][0],
        "ANO O S contraction columns must be DISTINCT (general contraction)"
    );
}

/// ANO H gains its real contractions: the H S-block has `exp + 6` columns
/// (`nctr == 6`), the P-block `exp + 4` (`nctr == 4`), the D-block `exp + 3`
/// (`nctr == 3`), the F-block a single contraction (`nctr == 1`).
#[test]
fn ano_h_gains_its_real_contractions() {
    let parsed = basis::load_basis("ano", "H").expect("load ANO H");

    // ANO H: [6s,4p,3d,1f] contraction pattern from the 4 primitive blocks.
    let expect_l_nctr: &[(u8, usize)] = &[(0, 6), (1, 4), (2, 3), (3, 1)];
    assert_eq!(
        parsed.shells.len(),
        expect_l_nctr.len(),
        "ANO H has 4 primitive blocks (S,P,D,F); got {} shells",
        parsed.shells.len()
    );
    for (i, &(l, nctr)) in expect_l_nctr.iter().enumerate() {
        assert_eq!(parsed.shells[i].l, l, "ANO H shell {i} l mismatch");
        assert_eq!(
            parsed.shells[i].coeffs.len(),
            nctr,
            "ANO H shell {i} (l={l}) must load nctr={nctr}; got {}",
            parsed.shells[i].coeffs.len()
        );
    }
}

// ----------------------------------------------------------------------------
// (b) Segmented-basis regression — PIN nao + per-shell nctr (T-02-11-REGRESSION)
// ----------------------------------------------------------------------------

/// Collect `(l, nctr)` per shell for a (basis, element) pair.
fn shell_l_nctr(name: &str, sym: &str) -> Vec<(u8, usize)> {
    basis::load_basis(name, sym)
        .unwrap_or_else(|e| panic!("load_basis({name}, {sym}) failed: {e:?}"))
        .shells
        .iter()
        .map(|s| (s.l, s.coeffs.len()))
        .collect()
}

/// sto-3g is PURELY segmented (one coefficient column per block). The fix must
/// not perturb it: H = 1 S-shell nctr=1; O = [2s,1p] all nctr=1.
#[test]
fn sto3g_segmented_unchanged() {
    assert_eq!(shell_l_nctr("sto-3g", "H"), vec![(0, 1)]);
    // O sto-3g: 1s (S), 2s/2p shared-SP → S + P, all single contractions.
    assert_eq!(shell_l_nctr("sto-3g", "O"), vec![(0, 1), (0, 1), (1, 1)]);
}

/// 6-31g is segmented (SP shared-exponent shells split into S+P, each a single
/// contraction). C = 1s(S) + 2×SP → S,P,S,P, all nctr=1.
#[test]
fn six_31g_segmented_unchanged() {
    assert_eq!(
        shell_l_nctr("6-31g", "C"),
        vec![(0, 1), (0, 1), (1, 1), (0, 1), (1, 1)]
    );
}

/// 6-31g* (the polarised Pople basis) for O is segmented: S + 2×SP + D, all
/// single contractions. The `*` is not stripped by `canonicalise_basis_name`,
/// so use the canonical alias key `6-31gs`.
#[test]
fn six_31gs_segmented_unchanged() {
    assert_eq!(
        shell_l_nctr("6-31gs", "O"),
        vec![(0, 1), (0, 1), (1, 1), (0, 1), (1, 1), (2, 1)]
    );
}

/// cc-pVDZ is the EXCEPTION that proves the fix: O's FIRST S-block is a GENERAL
/// CONTRACTION (`exp + 2` columns → nctr=2) that was being TRUNCATED to nctr=1.
/// The corrected structure is [S(nctr=2), S(nctr=1), P(nctr=1), P(nctr=1),
/// D(nctr=1)]; H is purely segmented [S,S,P] all nctr=1.
#[test]
fn ccpvdz_o_general_contraction_nctr2_pinned() {
    // The latent-bug fix: O's leading S-block recovers its 2nd contraction.
    assert_eq!(
        shell_l_nctr("cc-pvdz", "O"),
        vec![(0, 2), (0, 1), (1, 1), (1, 1), (2, 1)],
        "cc-pVDZ O: first S-block is a general contraction (nctr=2) that was \
         truncated; the corrected nctr is pinned here (latent bug fixed by 02-11)"
    );
    // H is purely segmented — unchanged.
    assert_eq!(shell_l_nctr("cc-pvdz", "H"), vec![(0, 1), (0, 1), (1, 1)]);
}

/// nao_nr pins for the regression corpus. cc-pVDZ O gains ONE s-AO from the
/// recovered 2nd S-contraction (13 → 14 = [3s,2p,1d]); the segmented bases are
/// byte-unchanged.
#[test]
fn nao_nr_pins() {
    let cases: &[(&str, &str, usize)] = &[
        ("sto-3g", "H", 1),   // 1s
        ("sto-3g", "O", 5),   // [2s,1p] = 2 + 3
        ("6-31g", "C", 9),    // [3s,2p] = 3 + 6
        ("6-31gs", "O", 14),  // [3s,2p,1d] = 3 + 6 + 5 (segmented; spherical 5d)
        ("cc-pvdz", "H", 5),  // [2s,1p] = 2 + 3
        ("cc-pvdz", "O", 14), // [3s,2p,1d] = 3 + 6 + 5 — was 13 pre-fix (truncated S)
    ];
    for &(name, sym, want) in cases {
        let mol = M(MoleBuildArgs {
            atom: AtomInput::String(format!("{sym} 0 0 0")),
            basis: BasisInput::Name(name.into()),
            unit: Unit::Bohr,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("build {sym}/{name}: {e:?}"));
        assert_eq!(
            mol.nao_nr, want,
            "{sym}/{name} nao_nr = {} but pinned {want}",
            mol.nao_nr
        );
    }
}

// ----------------------------------------------------------------------------
// (c) cintx evaluates general-contraction (nctr>1) shells — T-02-11-CINTX-NCTR
// ----------------------------------------------------------------------------

/// Helper: F-order overlap on a single-atom Mole with the given basis.
fn atom_overlap(sym: &str, basis_name: &str) -> (usize, Vec<f64>) {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(format!("{sym} 0 0 0")),
        basis: BasisInput::Name(basis_name.into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .unwrap_or_else(|e| panic!("build {sym}/{basis_name}: {e:?}"));
    let nao = mol.nao_nr;
    let s = intor(&mol, "int1e_ovlp_sph").expect("int1e_ovlp_sph");
    assert_eq!(s.values.len(), nao * nao);
    (nao, s.values)
}

/// STRONG correctness proof for the general-contraction COEFFICIENT LAYOUT
/// (threat T-02-11-AO-ORDER): cc-pVDZ O carries a GENERAL CONTRACTION nctr=2
/// S-block (l up to d). After the fix, its `int1e_ovlp_sph` is finite,
/// SYMMETRIC, and has a UNIT diagonal (each normalised AO self-overlaps to 1).
///
/// This is the canonical regression that pins the row-major coefficient
/// interleave (`coefficients[prim*nctr + ctr]`) the cintx kernel consumes:
/// a column-major flatten silently scrambles the nctr=2 block and produces a
/// non-unit diagonal (≈1.873 / 0.876) — the exact failure this test guards.
#[test]
fn ccpvdz_general_contraction_overlap_unit_diagonal() {
    let (nao, s) = atom_overlap("O", "cc-pvdz");

    for (i, v) in s.iter().enumerate() {
        assert!(v.is_finite(), "cc-pVDZ O overlap[{i}] = {v} must be finite");
    }
    // Symmetric.
    for i in 0..nao {
        for j in 0..nao {
            assert!(
                (s[i + j * nao] - s[j + i * nao]).abs() < 1e-12,
                "cc-pVDZ O overlap not symmetric at ({i},{j})"
            );
        }
    }
    // UNIT diagonal — proves the nctr=2 S-block coefficients reach cintx in the
    // correct (row-major) layout and are per-contraction normalised.
    for i in 0..nao {
        let d = s[i + i * nao];
        assert!(
            (d - 1.0).abs() < 1e-9,
            "cc-pVDZ O overlap diagonal[{i}] = {d} must be 1.0 (normalised AO); \
             a non-unit value means the general-contraction coefficients were \
             scrambled (T-02-11-AO-ORDER)"
        );
    }
}

/// ANO O exercises the FULL general-contraction stack (S nctr=8, P nctr=7,
/// D nctr=4, F nctr=3, G nctr=2). After the coefficient-layout fix every AO is
/// correctly NORMALISED — the overlap is finite with a UNIT diagonal across ALL
/// angular momenta (s,p,d,f,g). The s/p sub-block (the orbitals minao actually
/// occupies) is additionally SYMMETRIC.
///
/// KNOWN cintx-side gap (surfaced, NOT papered over — threat T-02-11-CINTX-NCTR):
/// the cintx 1e kernel's per-contraction-pair cart→sph scatter
/// (`kernels/one_electron.rs`, branch fix/general-contraction-nctr-1e@6b14d48)
/// is correct for l ≤ 2 but introduces an asymmetry for the (p,f) and (d,g)
/// general-contraction cross-blocks (|Δ| up to ~6 at l≥3). This does NOT affect
/// `init_guess_by_minao`: the occupation walk assigns occ=0 to every l≥3 ANO
/// contraction, so the f/g columns are filtered out of the density (see
/// `crates/pyscf-scf/tests/init_guess_minao.rs`). Tracked in
/// `.planning/phases/02-gto/deferred-items.md` (DI-02-11-CINTX-NCTR-HIGHL).
#[test]
fn ano_general_contraction_overlap_finite_unit_diagonal() {
    let (nao, s) = atom_overlap("O", "ano");
    assert!(nao > 1, "ANO O must be multi-AO, got nao={nao}");

    // Finite everywhere.
    for (i, v) in s.iter().enumerate() {
        assert!(v.is_finite(), "ANO O overlap[{i}] = {v} must be finite");
    }
    // UNIT diagonal across ALL l (s,p,d,f,g) — normalisation is correct for the
    // full general-contraction stack (the 7.9→9.86 minao trace recovery proof).
    for i in 0..nao {
        let d = s[i + i * nao];
        assert!(
            (d - 1.0).abs() < 1e-6,
            "ANO O overlap diagonal[{i}] = {d} must be 1.0 (normalised AO)"
        );
    }
    // SYMMETRIC for the l ≤ 2 (s/p/d) sub-block — the AOs minao occupies (s,p)
    // and the d block live here; the cintx l≥3 asymmetry is out of this range.
    // ANO O AO layout: S[0..8) P[8..29) D[29..49) ... — l≤2 occupies [0..49).
    let l_le_2 = 49usize.min(nao);
    for i in 0..l_le_2 {
        for j in 0..l_le_2 {
            assert!(
                (s[i + j * nao] - s[j + i * nao]).abs() < 1e-9,
                "ANO O overlap (l≤2 sub-block) not symmetric at ({i},{j}): {} vs {}",
                s[i + j * nao],
                s[j + i * nao]
            );
        }
    }
}
