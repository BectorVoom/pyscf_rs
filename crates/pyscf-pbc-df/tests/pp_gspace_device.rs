//! Device K-PP kernels against the host references (`pp_gspace` host route).
//! Needs a real cell, so these live in `pyscf-pbc-df` rather than
//! `pyscf-kernels`.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};

/// Cubic KTaO3, Bohr (copied from `tests/pp_gspace_oracle.rs`).
fn ktao3(basis: &str) -> Cell {
    let a = 7.54_f64;
    let h = 0.5 * a;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("K".into(), [0.0, 0.0, 0.0]),
                ("Ta".into(), [h, h, h]),
                ("O".into(), [h, h, 0.0]),
                ("O".into(), [h, 0.0, h]),
                ("O".into(), [0.0, h, h]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("KTaO3 cell must build")
}

#[test]
fn ftao_block_matches_the_host_transform() {
    const NK: [usize; 3] = [3, 3, 1];
    const MESH: [usize; 3] = [5, 5, 5];
    let client = pyscf_algebra::select_backend().expect("backend").client;
    for basis in ["gth-szv-molopt-sr", "gth-dzvp-molopt-sr"] {
        let cell = ktao3(basis);
        let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
        let kpt = kpts[4];
        let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
        let nb = gv.len();
        let tables = pyscf_pbc_df::pp_gspace::ftao_tables(&cell)
            .expect("tables")
            .expect("spherical l <= 3");
        let nao = cell.mol.nao_nr;
        assert_eq!(tables.nao, nao, "table AO count");
        let (d_re, d_im) = pyscf_kernels::pp_ftao_block(&client, &tables, &gv.concat(), kpt, 0, nb)
            .expect("pp_ftao_block");
        let (h_re, h_im) = pyscf_pbc_df::ft_ao::ft_ao_kpt(&cell.mol, &gv, kpt).expect("ft_ao");
        let inv_sqrt_vol = 1.0 / cell.vol().sqrt();
        let mut worst = 0.0f64;
        for q in 0..nao {
            for g in 0..nb {
                // Device `device[q*nb + g]` vs host `host[g*nao + q] / sqrt(vol)`.
                let (wr, wi) = (
                    h_re[g * nao + q] * inv_sqrt_vol,
                    h_im[g * nao + q] * inv_sqrt_vol,
                );
                let (gr, gi) = (d_re[q * nb + g], d_im[q * nb + g]);
                worst = worst
                    .max(((gr - wr) / (1.0 + wr.abs())).abs())
                    .max(((gi - wi) / (1.0 + wi.abs())).abs());
            }
        }
        println!("{basis}: ftao_block max rel dev vs host = {worst:e}");
        assert!(worst < 1e-11, "{basis}: ftao_block deviates by {worst:e}");
    }
}

#[test]
fn device_projection_matches_the_host() {
    const NK: [usize; 3] = [3, 3, 1];
    const MESH: [usize; 3] = [9, 9, 9];
    let cell = ktao3("gth-szv-molopt-sr");
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    assert_eq!(gv.len(), 729, "9^3 mesh");
    let t = pyscf_pbc_df::pp_gspace::proj_tables(&cell).expect("tables");
    let got = pyscf_pbc_df::pp_gspace::project_device(&cell, &t, &gv, &kpts, 729)
        .expect("project_device")
        .expect("device tables");
    assert_eq!(got.len(), kpts.len(), "one B per k-point");
    for (k, kpt) in kpts.iter().enumerate() {
        let (h_re, h_im) =
            pyscf_pbc_df::pp_gspace::project_host(&cell, &t, &gv, *kpt, 729).expect("host");
        let (d_re, d_im) = &got[k];
        assert_eq!(d_re.len(), h_re.len(), "k-point {k} row count");
        let mut worst = 0.0f64;
        for i in 0..h_re.len() {
            worst = worst
                .max(((d_re[i] - h_re[i]) / (1.0 + h_re[i].abs())).abs())
                .max(((d_im[i] - h_im[i]) / (1.0 + h_im[i].abs())).abs());
        }
        println!("k-point {k}: device vs host max rel dev = {worst:e}");
        assert!(worst < 1e-10, "k-point {k}: device deviates by {worst:e}");
    }
}

#[test]
fn device_projection_is_bit_identical_across_block_sizes() {
    const NK: [usize; 3] = [3, 3, 1];
    const MESH: [usize; 3] = [9, 9, 9];
    let cell = ktao3("gth-szv-molopt-sr");
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let t = pyscf_pbc_df::pp_gspace::proj_tables(&cell).expect("tables");
    let one = pyscf_pbc_df::pp_gspace::project_device(&cell, &t, &gv, &kpts, 729)
        .expect("one block")
        .expect("device tables");
    // 8 blocks, the last one ragged (29 points).
    let many = pyscf_pbc_df::pp_gspace::project_device(&cell, &t, &gv, &kpts, 100)
        .expect("many blocks")
        .expect("device tables");
    for (k, ((a_re, a_im), (b_re, b_im))) in one.iter().zip(many.iter()).enumerate() {
        assert_eq!(a_re.len(), b_re.len(), "k-point {k} len");
        for i in 0..a_re.len() {
            assert_eq!(a_re[i].to_bits(), b_re[i].to_bits(), "k-point {k} re[{i}]");
            assert_eq!(a_im[i].to_bits(), b_im[i].to_bits(), "k-point {k} im[{i}]");
        }
    }
}

/// The host reference has the same property: one accumulator per element,
/// carried across the blocks.
#[test]
fn host_projection_is_bit_identical_across_block_sizes() {
    const NK: [usize; 3] = [3, 3, 1];
    const MESH: [usize; 3] = [9, 9, 9];
    let cell = ktao3("gth-szv-molopt-sr");
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let t = pyscf_pbc_df::pp_gspace::proj_tables(&cell).expect("tables");
    let kpt = kpts[4];
    let (a_re, a_im) = pyscf_pbc_df::pp_gspace::project_host(&cell, &t, &gv, kpt, 729).expect("one block");
    let (b_re, b_im) = pyscf_pbc_df::pp_gspace::project_host(&cell, &t, &gv, kpt, 100).expect("many blocks");
    for i in 0..a_re.len() {
        assert_eq!(a_re[i].to_bits(), b_re[i].to_bits(), "re[{i}]");
        assert_eq!(a_im[i].to_bits(), b_im[i].to_bits(), "im[{i}]");
    }
}

#[test]
fn block_points_respects_the_budget() {
    use pyscf_pbc_df::pp_gspace::block_points;
    assert_eq!(block_points(1024.0, 910, 340, 249_615), 53_687);
    assert_eq!(block_points(1.0, 910, 340, 249_615), 1024);
    assert_eq!(block_points(1024.0, 27, 30, 729), 729);
}
