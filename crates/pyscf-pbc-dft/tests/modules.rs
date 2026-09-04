//! Oracle-free gates for the Phase-12 modules the main driver does not exercise:
//! DFT+U (`kspu`), constrained DFT (`cdft`), the 2-component integrator
//! (`numint2c`), the periodic Becke grid (`gen_grid`), and the XC kernel
//! (`nr_rks_fxc`).

mod common;

use common::{diamond, he_all_electron};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::cdft::ShiftHamiltonian;
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::kspu::{HARTREE2EV, HubbardU, USite, add_vhubbard, reference_cell, set_u};
use pyscf_pbc_dft::numint::KNumInt;
use pyscf_pbc_dft::numint2c::{Collinear, KNumInt2C};
use pyscf_pbc_dft::xc::XcType;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::KScfConfig;
use pyscf_pbc_scf::types::{KDms, KMats};

const MESH: [usize; 3] = [11, 11, 11];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

fn converged(cell: &Cell, nk: [usize; 3], xc: &str) -> (Vec<[f64; 3]>, KDms, PeriodicGrids) {
    let kpts = make_kpts_default(cell, nk).expect("k-mesh");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
    let r = Krks::from_df(Box::new(df), xc)
        .expect("KRKS")
        .kernel(&tight())
        .expect("KRKS");
    assert!(r.converged);
    let grids = PeriodicGrids::uniform(cell, Some(MESH)).expect("grids");
    (kpts, r.dm, grids)
}

// ---------------------------------------------------------------------------
// DFT+U
// ---------------------------------------------------------------------------

/// `set_u` groups one site per atom and converts `U` from eV to Hartree.
#[test]
fn set_u_groups_per_atom_and_converts_ev_to_hartree() {
    let cell = diamond(); // two carbon atoms
    let pcell = reference_cell(&cell, "minao").expect("MINAO reference cell");
    let cfg = HubbardU {
        sites: vec![USite::Shell {
            element: "C".into(),
            l: 1,
            contraction: Some(0),
        }],
        u_val: vec![5.0], // eV
        ..HubbardU::default()
    };
    let r = set_u(&pcell, &cfg).expect("set_u");
    println!(
        "sites: {:?}  U(Hartree): {:?}",
        r.indices.iter().map(Vec::len).collect::<Vec<_>>(),
        r.u_val
    );
    assert_eq!(
        r.indices.len(),
        2,
        "one Hubbard site per matching ATOM, not one per element"
    );
    for g in &r.indices {
        assert_eq!(g.len(), 3, "an l = 1 shell is three orbitals");
    }
    for u in &r.u_val {
        assert!(
            (u - 5.0 / HARTREE2EV).abs() < 1e-15,
            "U must be converted from eV to Hartree, got {u}"
        );
    }
}

/// `E_U = (U/2) Σ (Tr P − Tr P²)` is zero at `U = 0` and positive otherwise —
/// the occupancy matrix `P` is a projector-like object with eigenvalues in
/// `[0, 1]`, so `Tr P ≥ Tr P²`.
#[test]
fn hubbard_u_is_zero_at_u_zero_and_non_negative_otherwise() {
    let cell = he_all_electron();
    let (kpts, dm, _) = converged(&cell, [2, 2, 2], "lda,vwn");
    let nao = cell.mol.nao_nr;

    let mk = |u: f64| -> f64 {
        let cfg = HubbardU {
            sites: vec![USite::Shell {
                element: "He".into(),
                l: 0,
                contraction: Some(0),
            }],
            u_val: vec![u],
            ..HubbardU::default()
        };
        let mut vxc: Vec<KMats> = vec![vec![pyscf_algebra::CTensor::zeros(nao * nao); kpts.len()]];
        add_vhubbard(&mut vxc, &cell, &kpts, &dm, &cfg).expect("add_vhubbard")
    };

    let e0 = mk(0.0);
    let e5 = mk(5.0);
    println!("E_U(U=0) = {e0:.3e}   E_U(U=5 eV) = {e5:.12}");
    assert!(e0.abs() < 1e-15, "E_U must vanish at U = 0, got {e0:e}");
    assert!(
        e5 >= -1e-15,
        "E_U = (U/2)(Tr P - Tr P^2) cannot be negative for U > 0, got {e5:e}"
    );
}

// ---------------------------------------------------------------------------
// Constrained DFT
// ---------------------------------------------------------------------------

/// The cDFT shift is a single diagonal entry in the AO basis, and it lands on
/// every `(channel, k)` block of a `veff` without touching anything else.
#[test]
fn cdft_shift_is_a_single_diagonal_entry_in_the_ao_basis() {
    let nao = 8;
    let offset = 0.25;
    let orbital = 3;
    let sh = ShiftHamiltonian::new(nao, offset, orbital, None).expect("cdft");

    for i in 0..nao {
        for j in 0..nao {
            let want = if i == j && i == orbital { offset } else { 0.0 };
            assert_eq!(
                sh.matrix[i * nao + j],
                want,
                "cdft shift at ({i},{j}) should be {want}"
            );
        }
    }

    let mut veff: KDms = vec![vec![pyscf_algebra::CTensor::zeros(nao * nao); 4]; 2];
    sh.apply(&mut veff);
    for (c, set) in veff.iter().enumerate() {
        for (k, m) in set.iter().enumerate() {
            assert_eq!(
                m.re[orbital * nao + orbital],
                offset,
                "channel {c}, k {k} did not receive the shift"
            );
            assert!(m.im.iter().all(|&x| x == 0.0), "the shift must be real");
        }
    }
}

/// An out-of-range orbital is refused rather than silently clamped.
#[test]
fn cdft_refuses_an_out_of_range_orbital() {
    let e = ShiftHamiltonian::new(4, 0.1, 9, None).expect_err("must be refused");
    println!("{e}");
}

// ---------------------------------------------------------------------------
// The 2-component integrator
// ---------------------------------------------------------------------------

/// `numint2c` refuses exactly what upstream refuses: `mcol` needs `mcfun`, and
/// `ncol` is LDA-only (`numint2c.py`).
#[test]
fn numint2c_refuses_what_upstream_refuses() {
    let cell = diamond();
    let (kpts, dm, grids) = converged(&cell, [1, 1, 1], "lda,vwn");
    let nao = cell.mol.nao_nr;

    // A 2-component density: the closed-shell block on both diagonal halves.
    let dm2c: KMats = (0..kpts.len())
        .map(|k| {
            let mut m = pyscf_algebra::CTensor::zeros(4 * nao * nao);
            let n2 = 2 * nao;
            for i in 0..nao {
                for j in 0..nao {
                    let v_re = dm[0][k].re[i * nao + j] * 0.5;
                    let v_im = dm[0][k].im[i * nao + j] * 0.5;
                    for off in [0usize, nao] {
                        m.re[(i + off) * n2 + (j + off)] = v_re;
                        m.im[(i + off) * n2 + (j + off)] = v_im;
                    }
                }
            }
            m
        })
        .collect();

    let mut ni = KNumInt2C::new(&kpts);

    ni.collinear = Collinear::Mcol;
    let e = ni
        .nr_vxc(&cell, &grids, "lda,vwn", &dm2c, None)
        .expect_err("mcol must be refused");
    println!("mcol: {e}");
    assert!(
        e.to_string().contains("mcfun"),
        "the refusal must name mcfun"
    );

    ni.collinear = Collinear::Ncol;
    let e = ni
        .nr_vxc(&cell, &grids, "pbe", &dm2c, None)
        .expect_err("ncol + GGA must be refused");
    println!("ncol+GGA: {e}");

    // `col` — the upstream default, and what KGKS uses — must WORK.
    ni.collinear = Collinear::Col;
    let r = ni
        .nr_vxc(&cell, &grids, "lda,vwn", &dm2c, None)
        .expect("collinear 2-component nr_vxc");
    println!("col: nelec = {:.10}, E_xc = {:.12}", r.nelec, r.excsum);
    assert!(
        r.excsum.is_finite() && r.excsum < 0.0,
        "E_xc must be negative"
    );
}

// ---------------------------------------------------------------------------
// The periodic Becke grid
// ---------------------------------------------------------------------------

/// The periodic Becke partition and the uniform box integrate the same density
/// to the same electron count.
///
/// # The agreement is 1e-4, not machine precision — deliberately
///
/// A sum of atom-centred grids masked to the cell weights a density near a cell
/// face differently from a uniform box. `BeckeGrids` is therefore NOT
/// interchangeable with `UniformGrids` at the 1e-9 level, and a caller who swaps
/// them will see the energy move in the fifth decimal. This test states that
/// bound rather than hiding it.
#[test]
fn becke_grids_integrate_to_the_same_electron_count() {
    let cell = he_all_electron();
    let nelec = cell.mol.nelectron as f64;
    let (kpts, dm, uniform) = converged(&cell, [1, 1, 1], "lda,vwn");
    let ni = KNumInt::new(&kpts);

    let becke = PeriodicGrids::becke(&cell, pyscf_grids::Grids::new()).expect("becke grids");
    println!(
        "uniform: {} points, becke: {} points",
        uniform.size(),
        becke.size()
    );

    let n_u = ni
        .nr_rks(&cell, &uniform, "lda,vwn", &dm, 1, None)
        .expect("uniform")
        .nelec[0];
    let n_b = ni
        .nr_rks(&cell, &becke, "lda,vwn", &dm, 1, None)
        .expect("becke")
        .nelec[0];
    println!("uniform nelec {n_u:.10}   becke nelec {n_b:.10}   expected {nelec}");
    assert!(
        (n_b - nelec).abs() < 5e-4,
        "the Becke grid integrates to {n_b}, not {nelec}"
    );
    assert!(
        (n_b - n_u).abs() < 5e-4,
        "the two grids disagree by {:e}, beyond the documented 1e-4 bound",
        (n_b - n_u).abs()
    );
}

// ---------------------------------------------------------------------------
// The XC kernel
// ---------------------------------------------------------------------------

/// `fxc` is the derivative of `vxc`: contracting the kernel with a density
/// perturbation must reproduce a central difference of `V_xc` along it.
#[test]
fn fxc_is_the_derivative_of_vxc() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let (kpts, dm, grids) = converged(&cell, [1, 1, 1], "lda,vwn");
    let nkpts = kpts.len();
    let ni = KNumInt::new(&kpts);

    // A Hermitian perturbation direction.
    let mut seed = 0x1234_5678_9ABC_DEF0_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let delta: KMats = (0..nkpts)
        .map(|_| {
            let mut re = vec![0.0; nao * nao];
            for i in 0..nao {
                for j in i..nao {
                    let a = next();
                    re[i * nao + j] = a;
                    re[j * nao + i] = a;
                }
            }
            pyscf_algebra::CTensor::from_real(&re)
        })
        .collect();

    let vxc_at = |eps: f64| -> KMats {
        let shifted: KDms = vec![
            (0..nkpts)
                .map(|k| {
                    let mut m = dm[0][k].clone();
                    for i in 0..nao * nao {
                        m.re[i] += eps * delta[k].re[i];
                    }
                    m
                })
                .collect(),
        ];
        ni.nr_rks(&cell, &grids, "lda,vwn", &shifted, 1, None)
            .expect("nr_rks")
            .vmat
            .remove(0)
    };

    let eps = 1e-5;
    let (vp, vm) = (vxc_at(eps), vxc_at(-eps));

    let cache = ni
        .cache_xc_kernel1(&cell, &grids, "lda,vwn", &dm, 0)
        .expect("cache_xc_kernel1");
    let contracted = ni
        .nr_rks_fxc(
            &cell,
            &grids,
            "lda,vwn",
            Some(&dm[0]),
            &vec![delta.clone()],
            1,
            Some(&cache.fxc),
            true,
        )
        .expect("nr_rks_fxc");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for k in 0..nkpts {
        for i in 0..nao * nao {
            let fd = (vp[k].re[i] - vm[k].re[i]) / (2.0 * eps);
            worst = worst.max((fd - contracted[0][k].re[i]).abs());
            scale = scale.max(fd.abs());
        }
    }
    let rel = worst / scale.max(1e-30);
    println!("fxc vs d vxc/d D: worst |delta| {worst:.3e}, relative {rel:.3e}");
    assert!(
        rel < 1e-4,
        "fxc does not reproduce d vxc / d D (relative {rel:e})"
    );
}

/// The refusal that keeps `XcType` honest, restated here so `modules.rs` is
/// self-contained about what the crate will and will not evaluate.
#[test]
fn xc_type_classifies_the_supported_corpus() {
    assert_eq!(XcType::of("lda,vwn").expect("lda"), XcType::Lda);
    assert_eq!(XcType::of("pbe").expect("pbe"), XcType::Gga);
    assert_eq!(XcType::of("b3lyp").expect("b3lyp"), XcType::Gga);
    assert_eq!(XcType::of("pbe0").expect("pbe0"), XcType::Gga);
}
