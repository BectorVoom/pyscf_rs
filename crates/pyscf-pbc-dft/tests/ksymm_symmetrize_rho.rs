//! S-03: the opt-in IBZ-density route agrees with the full-BZ unfold route.

use pyscf_algebra::CTensor;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::test_systems::si_precision;
use pyscf_pbc_symm::kpts::make_kpts;

fn ibz_density(nk: usize, nao: usize) -> Vec<CTensor> {
    (0..nk)
        .map(|k| {
            let mut m = CTensor::zeros(nao * nao);
            for i in 0..nao {
                for j in 0..=i {
                    let re = (((k + 1) * 31 + i * 7 + j * 13) as f64).sin() * 0.05;
                    let im = if i == j {
                        0.0
                    } else {
                        (((k + 2) * 17 + i * 11 + j * 5) as f64).cos() * 0.02
                    };
                    m.re[i * nao + j] = re;
                    m.re[j * nao + i] = re;
                    m.im[i * nao + j] = im;
                    m.im[j * nao + i] = -im;
                }
                m.re[i * nao + i] += 0.5;
            }
            m
        })
        .collect()
}

#[test]
fn symmetrized_open_shell_kuks_energy_matches_unfold() {
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, Unit};
    use pyscf_pbc_dft::krks_ksymm::KsymAdaptedKuks;
    use pyscf_pbc_gto::types::{ALattice, CellBuildArgs};
    use pyscf_pbc_scf::{KInitGuess, KScfConfig};
    use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};

    let h = 2.834589;
    let mut cell = pyscf_pbc_gto::Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("6-31g".into()),
            unit: Unit::Bohr,
            spin: 2,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("open-shell He cell");
    cell.mesh = [11, 11, 11];
    let full = make_kpts_default(&cell, [2, 2, 2]).expect("full k mesh");
    let kp = make_kpts(&cell, &full, true, false).expect("symmetric k mesh");
    basis::build_symmetry(
        &mut cell,
        &SymmAdaptedBasisInput {
            kpts_scaled_ibz: kp.kpts_scaled_ibz.clone(),
            little_cogroup_ops: kp.little_cogroup_ops.clone(),
            ops: kp.symmetry.ops.clone(),
            dmats: kp.symmetry.dmats.clone(),
        },
    )
    .expect("symmetry-adapted basis");
    let cfg = KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        init_guess: KInitGuess::Minao,
        ..Default::default()
    };

    unsafe { std::env::remove_var("PYSCF_PBC_KSYMM_RHO") };
    let unfold = KsymAdaptedKuks::new(cell.clone(), kp.clone(), "lda,vwn")
        .expect("unfold KUKS")
        .kernel(&cfg)
        .expect("unfold kernel");
    unsafe { std::env::set_var("PYSCF_PBC_KSYMM_RHO", "symmetrize") };
    let sym = KsymAdaptedKuks::new(cell, kp, "lda,vwn")
        .expect("symmetrized KUKS")
        .kernel(&cfg)
        .expect("symmetrized kernel");
    unsafe { std::env::remove_var("PYSCF_PBC_KSYMM_RHO") };

    assert!(
        unfold.converged && sym.converged,
        "both routes must converge"
    );
    assert!(
        (sym.e_tot - unfold.e_tot).abs() < 1e-11,
        "open-shell KUKS route energy delta = {:.3e}",
        (sym.e_tot - unfold.e_tot).abs()
    );
    let spin_delta = sym.dm[0]
        .iter()
        .zip(&sym.dm[1])
        .flat_map(|(a, b)| a.re.iter().zip(&b.re))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(spin_delta > 1e-6, "fixture lost its spin polarisation");
}

fn max_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn scaled_density(dms: &[CTensor], scale: f64) -> Vec<CTensor> {
    dms.iter()
        .map(|dm| {
            CTensor::from_planes(
                dm.re.iter().map(|x| x * scale).collect(),
                dm.im.iter().map(|x| x * scale).collect(),
            )
        })
        .collect()
}

#[test]
fn symmetrized_ibz_quadrature_matches_unfold_for_lda_and_gga() {
    for nk in [[2, 2, 2], [3, 3, 3]] {
        let mut cell = si_precision(1e-10);
        cell.mesh = [11, 11, 11];
        let full = make_kpts_default(&cell, nk).expect("full k mesh");
        let kp = make_kpts(&cell, &full, true, false).expect("symmetric k mesh");
        assert!(kp.nkpts_ibz() < kp.nkpts());
        let grids = PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("uniform grids");
        let dms = vec![ibz_density(kp.nkpts_ibz(), cell.mol.nao_nr)];
        let spin_dms = [dms.clone(), vec![scaled_density(&dms[0], 0.73)]];

        for xc in ["lda,vwn", "pbe,pbe"] {
            unsafe { std::env::remove_var("PYSCF_PBC_KSYMM_RHO") };
            let unfold = KNumInt::with_symmetry(&kp)
                .nr_rks(&cell, &grids, xc, &dms, 1, Some(&kp.kpts_ibz))
                .expect("unfold route");
            let unfold_u = KNumInt::with_symmetry(&kp)
                .nr_uks(&cell, &grids, xc, &spin_dms, 1, Some(&kp.kpts_ibz))
                .expect("unfold unrestricted route");
            unsafe { std::env::set_var("PYSCF_PBC_KSYMM_RHO", "symmetrize") };
            let sym = KNumInt::with_symmetry(&kp)
                .nr_rks(&cell, &grids, xc, &dms, 1, Some(&kp.kpts_ibz))
                .expect("symmetrize route");
            let sym_u = KNumInt::with_symmetry(&kp)
                .nr_uks(&cell, &grids, xc, &spin_dms, 1, Some(&kp.kpts_ibz))
                .expect("symmetrize unrestricted route");
            unsafe { std::env::remove_var("PYSCF_PBC_KSYMM_RHO") };

            assert!(
                (sym.nelec[0] - unfold.nelec[0]).abs() < 1e-11,
                "{nk:?} {xc} RKS nelec"
            );
            assert!(
                (sym.excsum[0] - unfold.excsum[0]).abs() < 1e-11,
                "{nk:?} {xc} RKS exc"
            );
            for (k, (a, b)) in sym.vmat[0].iter().zip(&unfold.vmat[0]).enumerate() {
                assert!(
                    max_delta(&a.re, &b.re) < 1e-11,
                    "{nk:?} {xc} RKS k={k} real"
                );
                assert!(
                    max_delta(&a.im, &b.im) < 1e-11,
                    "{nk:?} {xc} RKS k={k} imag"
                );
            }
            for spin in 0..2 {
                assert!(
                    (sym_u.nelec[0].0 - unfold_u.nelec[0].0).abs() < 1e-11
                        && (sym_u.nelec[0].1 - unfold_u.nelec[0].1).abs() < 1e-11,
                    "{nk:?} {xc} UKS nelec"
                );
                assert!(
                    (sym_u.excsum[0] - unfold_u.excsum[0]).abs() < 1e-11,
                    "{nk:?} {xc} UKS exc"
                );
                for (k, (a, b)) in sym_u.vmat[spin][0]
                    .iter()
                    .zip(&unfold_u.vmat[spin][0])
                    .enumerate()
                {
                    assert!(
                        max_delta(&a.re, &b.re) < 1e-11,
                        "{nk:?} {xc} UKS spin={spin} k={k} real"
                    );
                    assert!(
                        max_delta(&a.im, &b.im) < 1e-11,
                        "{nk:?} {xc} UKS spin={spin} k={k} imag"
                    );
                }
            }
        }
    }
}
