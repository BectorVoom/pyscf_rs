//! RMP2 structural / wiring + behavior tests (always-on layer for the
//! `mp2-structural` CI job).
//!
//! All arms here are ALWAYS-ON: they exercise the closed-form kernel against a
//! SYNTHETIC `(ia|jb)` block fed through a test-only `Mp2OverrideHooks` impl
//! (bypassing `intor`), the SCS factor split, and the `default_ao2mo`
//! error-propagation shape against a real `Mole`. The numeric ENERGY against a
//! real molecule is the cintx#11-gated oracle's job (a separate CI arm).

use pyscf_core::{MOCoefficients, Mole, PyscfRsError};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_mp2::{
    ChemistsEris, Frozen, Mp2OverrideHooks, Mp2Reference, Mp2Result, NoMp2Overrides, rmp2_kernel,
    scs_energy,
};

/// Always-on: the RMP2 module surface + the error type + the MP2-08 helper
/// re-exports are reachable from the crate root.
#[test]
fn rmp2_module_surface_exists() {
    fn assert_error_type(_e: Option<pyscf_mp2::Mp2Error>) {}
    assert_error_type(None);

    // The MP2-08 helper re-exports are reachable from the crate root.
    let _: fn(&[f64], &pyscf_mp2::Frozen) -> _ = pyscf_mp2::get_nocc;
}

/// A test-only hooks impl that returns a HAND-BUILT `ChemistsEris`, bypassing
/// the `intor` / `default_ao2mo` path entirely. This lets the closed-form
/// kernel run with a known synthetic `(ia|jb)` block (no integrals needed).
struct SyntheticEris {
    eris: ChemistsEris,
}

impl Mp2OverrideHooks for SyntheticEris {
    fn ao2mo(&self, _refr: &Mp2Reference, _frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        Ok(ChemistsEris {
            ovov: self.eris.ovov.clone(),
            nocc: self.eris.nocc,
            nvir: self.eris.nvir,
        })
    }
}

/// Build a minimal Mp2Reference with the given MO energies (nao = mo count).
fn toy_reference(mo_energy: Vec<f64>, mo_occ: Vec<f64>) -> Mp2Reference {
    let nmo = mo_energy.len();
    Mp2Reference {
        mo_coeff: MOCoefficients {
            nao: nmo,
            nmo,
            data: vec![0.0; nmo * nmo],
            energies: mo_energy.clone(),
            occupations: mo_occ.clone(),
        },
        mo_energy,
        mo_occ,
        e_hf: -1.0,
        converged: true,
        mol: Mole::default(),
    }
}

/// ALWAYS-ON: the closed-form kernel returns the hand-computed e_corr from a
/// synthetic 1-occupied / 1-virtual `(ia|jb)` block.
///
/// nocc=1, nvir=1; mo_energy = [-0.5 (occ), 0.5 (vir)]; ovov = [g] = [0.5].
/// eia = εo - εv = -1.0; denom (j=0,i=0) = eia[0]+eia[0] = -2.0.
/// t2 = g/denom = 0.5/-2.0 = -0.25.
/// edi = 2 * t2 * g = 2*(-0.25)*0.5 = -0.25; exi = -t2*g = -(-0.25)*0.5 = 0.125.
/// e_ss = edi*0.5 + exi = -0.125 + 0.125 = 0.0; e_os = edi*0.5 = -0.125.
/// e_corr = e_ss + e_os = -0.125.
#[test]
fn rmp2_kernel_hand_computed_energy() {
    let refr = toy_reference(vec![-0.5, 0.5], vec![2.0, 0.0]);
    let hooks = SyntheticEris {
        eris: ChemistsEris {
            ovov: vec![0.5],
            nocc: 1,
            nvir: 1,
        },
    };
    let res: Mp2Result = rmp2_kernel(&refr, &Frozen::None, &hooks, true).expect("kernel");
    assert!(
        (res.e_corr - (-0.125)).abs() < 1e-12,
        "e_corr = {}",
        res.e_corr
    );
    assert!(res.e_ss.abs() < 1e-12, "e_ss = {}", res.e_ss);
    assert!((res.e_os - (-0.125)).abs() < 1e-12, "e_os = {}", res.e_os);
    // with_t2=true stores the amplitudes.
    let t2 = res.t2.expect("t2 stored");
    assert_eq!(t2.nocc, 1);
    assert_eq!(t2.nvir, 1);
    assert!((t2.t2[0] - (-0.25)).abs() < 1e-12, "t2 = {}", t2.t2[0]);
}

/// ALWAYS-ON: a richer 1-occupied / 2-virtual block exercises the inner
/// virtual loops. mo_energy = [-0.5 (occ), 0.3, 0.6 (vir)]; ovov laid out as
/// (i,a,j,b) with nocc=1,nvir=2: indices (0,0,0,0)=g00, (0,0,0,1)=g01,
/// (0,1,0,0)=g10, (0,1,0,1)=g11.
#[test]
fn rmp2_kernel_two_virtual() {
    // eia: a=0 -> -0.5-0.3=-0.8 ; a=1 -> -0.5-0.6=-1.1
    let eo = -0.5_f64;
    let ev = [0.3_f64, 0.6];
    let eia = [eo - ev[0], eo - ev[1]]; // [-0.8, -1.1]
    // ovov[(i=0,a,j=0,b)] = flat[a*nvir + b] for nocc=1.
    let g = [0.20_f64, 0.05, 0.05, 0.10]; // g[a*2+b]
    let refr = toy_reference(vec![eo, ev[0], ev[1]], vec![2.0, 0.0, 0.0]);
    let hooks = SyntheticEris {
        eris: ChemistsEris {
            ovov: g.to_vec(),
            nocc: 1,
            nvir: 2,
        },
    };
    let res = rmp2_kernel(&refr, &Frozen::None, &hooks, false).expect("kernel");

    // Hand reference (single i=0, single j=0): for each (a,b):
    //   t2[a,b] = g[a,b] / (eia[a] + eia[b])   [since i==j==0, denom uses both]
    //   edi += 2 * g[a,b] * t2[a,b]
    //   exi += - g[b,a] * t2[a,b]
    let mut edi = 0.0;
    let mut exi = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            let denom = eia[a] + eia[b];
            let t2 = g[a * 2 + b] / denom;
            edi += 2.0 * g[a * 2 + b] * t2;
            exi += -g[b * 2 + a] * t2;
        }
    }
    let e_ss_ref = edi * 0.5 + exi;
    let e_os_ref = edi * 0.5;
    let e_corr_ref = e_ss_ref + e_os_ref;
    assert!(
        (res.e_corr - e_corr_ref).abs() < 1e-12,
        "e_corr {} vs {}",
        res.e_corr,
        e_corr_ref
    );
    assert!(
        res.t2.is_none(),
        "with_t2=false should not store amplitudes"
    );
}

/// ALWAYS-ON: SCS factor split. With ss=os=1.0 the SCS energy equals plain
/// e_corr = e_ss + e_os. With SCS defaults (1/3, 1.2) it equals the SCS split.
#[test]
fn scs_energy_factors() {
    let e_ss = -0.30_f64;
    let e_os = -0.50_f64;

    // Plain MP2: ss=os=1.0 reproduces e_ss + e_os.
    let plain = scs_energy(e_ss, e_os, 1.0, 1.0);
    assert!((plain - (e_ss + e_os)).abs() < 1e-15, "plain = {plain}");

    // SCS defaults: e_ss/3 + e_os*1.2 (J. Chem. Phys. 118, 9095 (2003)).
    let scs = scs_energy(e_ss, e_os, 1.0 / 3.0, 1.2);
    let scs_ref = e_ss * (1.0 / 3.0) + e_os * 1.2;
    assert!((scs - scs_ref).abs() < 1e-15, "scs = {scs}");
}

/// ALWAYS-ON: `default_ao2mo` (exercised through `NoMp2Overrides`) now SUCCEEDS
/// against a REAL Mole — cintx#11 is closed (05-08), so `intor("int2e")` returns
/// a real arity-4 tensor instead of `NotYetImplemented{phase:2}`. The transform
/// runs end-to-end and yields a well-formed `ChemistsEris` of the expected
/// `(o,v,o,v)` shape. With the zero `mo_coeff` of this toy reference the block is
/// numerically zero, but the path no longer errors — the finite-energy numeric
/// proof lives in `mp2_numeric_smoke.rs`.
#[test]
fn default_ao2mo_succeeds_after_cintx11_closure() {
    // A built H2/STO-3G Mole so intor() passes its built-check and reaches the
    // arity-4 int2e NotYetImplemented branch.
    let mol = pyscf_gto::M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2");

    let nmo = mol.nao_nr;
    let refr = Mp2Reference {
        mo_coeff: MOCoefficients {
            nao: nmo,
            nmo,
            data: vec![0.0; nmo * nmo],
            energies: vec![-0.5, 0.5][..nmo.min(2)].to_vec(),
            occupations: {
                let mut o = vec![0.0; nmo];
                if !o.is_empty() {
                    o[0] = 2.0;
                }
                o
            },
        },
        mo_energy: {
            let mut e = vec![0.0; nmo];
            for (i, v) in e.iter_mut().enumerate() {
                *v = -0.5 + (i as f64) * 0.5;
            }
            e
        },
        mo_occ: {
            let mut o = vec![0.0; nmo];
            if !o.is_empty() {
                o[0] = 2.0;
            }
            o
        },
        e_hf: -1.1,
        converged: true,
        mol,
    };

    let hooks = NoMp2Overrides;
    let eris = hooks
        .ao2mo(&refr, &Frozen::None)
        .expect("default_ao2mo must succeed now that arity-4 int2e lands (cintx#11 closed)");
    // nocc=1 (one doubly-occupied), nvir = nmo-1; ovov is the (o,v,o,v) block.
    assert_eq!(eris.nocc, 1, "H2/STO-3G closed-shell has 1 occupied MO");
    assert_eq!(eris.nvir, refr.mo_coeff.nmo - 1);
    assert_eq!(
        eris.ovov.len(),
        eris.nocc * eris.nvir * eris.nocc * eris.nvir,
        "ovov block has (o,v,o,v) shape"
    );
    for v in &eris.ovov {
        assert!(v.is_finite(), "ovov entries must be finite");
    }
}
