//! Integration tests for `pyscf_pbc_symm::group` — 17-02-PLAN.md Tasks 4/5.
//!
//! All tests here are oracle-free: group axioms, the multiplication table
//! being a Latin square, and Burnside character-table orthogonality are
//! mathematical identities that hold for ANY correct finite-group
//! implementation, independent of upstream's specific `np.random.rand` draw
//! (see `group.rs`'s module doc). `PointGroup::group_name` is cross-checked
//! against known crystallography for the five PBC-MASTER-PLAN §9.2 fixtures.

use pyscf_pbc_symm::geom::{self, SYMPREC};
use pyscf_pbc_symm::group::{FiniteGroup, PgElement, PointGroup, Representation};

fn si_point_group() -> PointGroup {
    let cell = pyscf_pbc_gto::test_systems::si();
    let rots = geom::search_point_group_ops(&cell, SYMPREC).unwrap();
    let elements: Vec<PgElement> = rots.into_iter().map(PgElement::new).collect();
    FiniteGroup::new(elements).expect("si's 48 rotations must form a group")
}

#[test]
fn si_point_group_has_order_48() {
    let pg = si_point_group();
    assert_eq!(pg.order(), 48);
}

#[test]
fn group_axioms_closure_identity_inverses() {
    let pg = si_point_group();
    let n = pg.order();
    let ht = pg.hash_table();

    // identity: exactly one element hashes to the identity's own hash.
    let eye = PgElement::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    assert!(ht.contains_key(&eye.hash_key()), "identity must be a member");
    let e_idx = ht[&eye.hash_key()];
    for (i, g) in pg.elements.iter().enumerate() {
        let ge = g.compose(&eye);
        let eg = eye.compose(g);
        assert_eq!(ge, *g, "g * e != g at index {i}");
        assert_eq!(eg, *g, "e * g != g at index {i}");
    }
    let _ = e_idx;

    // closure: product of any two elements is itself a member (hash_table
    // lookup would panic on a missing key -- exercising that IS the test).
    let mult = pg.multiplication_table();
    assert_eq!(mult.len(), n);
    for row in &mult {
        assert_eq!(row.len(), n);
    }

    // inverses: g * g^-1 == e for every g.
    let inv = pg.inverse_table();
    for (i, g) in pg.elements.iter().enumerate() {
        let ginv = &pg.elements[inv[i]];
        let prod = g.compose(ginv);
        assert_eq!(prod, eye, "g * g^-1 != e at index {i}");
    }

    // associativity, sampled: (a*b)*c == a*(b*c) for every triple drawn from
    // a fixed small sample of element indices (full 48^3 is unnecessary —
    // associativity is inherited from ordinary matrix multiplication, this
    // sample just exercises PgElement::compose's wiring).
    let sample: Vec<usize> = (0..n).step_by(7).collect();
    for &i in &sample {
        for &j in &sample {
            for &k in &sample {
                let a = &pg.elements[i];
                let b = &pg.elements[j];
                let c = &pg.elements[k];
                let ab_c = a.compose(b).compose(c);
                let a_bc = a.compose(&b.compose(c));
                assert_eq!(ab_c, a_bc, "associativity failed at ({i},{j},{k})");
            }
        }
    }
}

#[test]
fn multiplication_table_is_a_latin_square() {
    let pg = si_point_group();
    let n = pg.order();
    let mult = pg.multiplication_table();

    // every row is a permutation of 0..n
    for (i, row) in mult.iter().enumerate() {
        let mut sorted = row.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "row {i} is not a permutation");
    }
    // every column is a permutation of 0..n
    for j in 0..n {
        let mut col: Vec<usize> = mult.iter().map(|row| row[j]).collect();
        col.sort_unstable();
        assert_eq!(col, (0..n).collect::<Vec<_>>(), "column {j} is not a permutation");
    }
}

#[test]
fn character_table_burnside_orthogonality() {
    let pg = si_point_group();
    let n = pg.order() as f64;
    let (classes, _reps, _inverse) = pg.conjugacy_classes();
    let class_sizes: Vec<f64> = classes
        .iter()
        .map(|c| c.iter().filter(|&&b| b).count() as f64)
        .collect();
    let nclass = classes.len();

    let chartab = pg.character_table(false); // (nclass, nclass)
    assert_eq!(chartab.len(), nclass);

    // Sum_classes |class| * |chi_ir|^2 == |G| for every irrep (row).
    for (ir, row) in chartab.iter().enumerate() {
        let sum: f64 = row
            .iter()
            .zip(class_sizes.iter())
            .map(|(c, &sz)| c.norm_sqr() * sz)
            .sum();
        assert!(
            (sum - n).abs() < 1e-6,
            "irrep {ir}: Sum |class| |chi|^2 = {sum}, want {n}"
        );
    }

    // Row orthogonality: Sum_classes |class| * chi_a(g) * conj(chi_b(g)) == 0
    // for a != b, == |G| for a == b.
    for a in 0..nclass {
        for b in 0..nclass {
            let mut sum = num_complex::Complex64::new(0.0, 0.0);
            for c in 0..nclass {
                sum += class_sizes[c] * chartab[a][c] * chartab[b][c].conj();
            }
            let want = if a == b { n } else { 0.0 };
            assert!(
                (sum.re - want).abs() < 1e-6 && sum.im.abs() < 1e-6,
                "orthogonality failed for irreps ({a},{b}): got {sum:?}, want {want}"
            );
        }
    }
}

#[test]
fn point_group_names_match_known_crystallography() {
    let expected = [
        ("diamond", "m-3m", "Oh"),
        ("si", "m-3m", "Oh"),
        ("lif", "m-3m", "Oh"),
        ("he_fcc", "m-3m", "Oh"),
        ("graphene", "6mm", "C6v"),
    ];
    for (name, cell) in pyscf_pbc_gto::test_systems::all() {
        let rots = geom::search_point_group_ops(&cell, SYMPREC).unwrap();
        let elements: Vec<PgElement> = rots.into_iter().map(PgElement::new).collect();
        let pg = FiniteGroup::new(elements).unwrap();
        let (_, want_intl, want_scho) = expected.iter().find(|(n, _, _)| *n == name).unwrap();
        assert_eq!(pg.group_name().unwrap(), *want_intl, "{name}: international symbol");
        assert_eq!(
            pg.group_name_schoenflies().unwrap(),
            *want_scho,
            "{name}: Schoenflies symbol"
        );
    }
}

/// `group.py:412-417` — `group_index`: the position of the international
/// symbol in `SchoenfliesNotation`'s insertion order (`tables.rs`).
#[test]
fn group_index_matches_table_position() {
    let pg = si_point_group();
    let idx = pg.group_index().unwrap();
    assert_eq!(
        pyscf_pbc_symm::tables::group_index("m-3m"),
        Some(idx),
        "group_index must agree with tables::group_index"
    );
    // 'm-3m' is the LAST entry in SchoenfliesNotation (tables.py:99).
    assert_eq!(idx, pyscf_pbc_symm::tables::SCHOENFLIES_NOTATION.len() - 1);
}

/// `group.py:460-467`/`:455-458` — `chi_to_rep(rep_to_chi(r)) == r` on every
/// irrep, the identity that defines the `rep_to_chi`/`chi_to_rep` pair
/// (17-02-PLAN.md Task 5: "ship it with the identities that define it").
#[test]
fn chi_to_rep_rep_to_chi_round_trip_on_irreps() {
    let pg = si_point_group();
    let n_irrep = pg.character_table(false).len();
    for ir in 0..n_irrep {
        let mut rep = vec![0i64; n_irrep];
        rep[ir] = 1;
        let chi = Representation::rep_to_chi(&pg, &rep);
        let rep_back = Representation::chi_to_rep(&pg, &chi).unwrap();
        assert_eq!(rep_back, rep, "round trip failed for irrep {ir}");
    }
}

/// The regular representation's `chi` is `order` at the identity class and
/// `0` elsewhere; `chi_to_rep` must recover each irrep with multiplicity
/// equal to its own dimension (a standard representation-theory fact, and a
/// second, independent identity check on the same pair).
#[test]
fn regular_representation_multiplicities_equal_irrep_dimensions() {
    let pg = si_point_group();
    let chartab_full = pg.character_table(true); // (n_irrep, n)
    let n = pg.order();

    let ht = pg.hash_table();
    let eye = PgElement::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    let e_idx = ht[&eye.hash_key()];
    let mut chi = vec![num_complex::Complex64::new(0.0, 0.0); n];
    chi[e_idx] = num_complex::Complex64::new(n as f64, 0.0);

    let rep = Representation::chi_to_rep(&pg, &chi).unwrap();
    for (ir, &mult) in rep.iter().enumerate() {
        let dim = chartab_full[ir][e_idx].re.round() as i64;
        assert_eq!(mult, dim, "irrep {ir}: regular-rep multiplicity != its own dimension");
    }
    // Sum dim_i^2 == |G|.
    let sum_sq: i64 = rep
        .iter()
        .enumerate()
        .map(|(ir, &mult)| {
            let dim = chartab_full[ir][e_idx].re.round() as i64;
            debug_assert_eq!(mult, dim);
            dim * dim
        })
        .sum();
    assert_eq!(sum_sq, n as i64);
}
