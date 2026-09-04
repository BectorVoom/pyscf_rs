//! Plan 13-02 acceptance — the single-centre AO Fourier transform and the fake
//! nuclear cell.

mod common;

use common::{diamond, he_all_electron};
use pyscf_pbc_df::ft_ao::{fake_nuc, ft_ao_mol};

/// **Test 1** — the `s`-function closed form, `c·(π/α)^{3/2}·e^{−G²/4α}·e^{−iG·A}`.
#[test]
fn s_function_matches_the_closed_form() {
    use pyscf_core::raw_layout::{ATM_SLOTS, ATOM_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP};
    let cell = he_all_electron();
    let mol = &cell.mol;
    let gv: Vec<[f64; 3]> = vec![[0.0; 3], [0.4, -0.9, 1.3], [-2.0, 0.6, 0.2]];
    let (re, im) = ft_ao_mol(mol, &gv).expect("ft_ao");

    let nprim = mol._bas[NPRIM_OF] as usize;
    let pe = mol._bas[PTR_EXP] as usize;
    let pc = mol._bas[PTR_COEFF] as usize;
    let atom = mol._bas[ATOM_OF] as usize;
    let pcoord = mol._atm[atom * ATM_SLOTS + PTR_COORD] as usize;
    let a = [mol._env[pcoord], mol._env[pcoord + 1], mol._env[pcoord + 2]];
    let cfac = pyscf_kernels::common_fac_sp(0);
    let pi = std::f64::consts::PI;

    for (g, gvec) in gv.iter().enumerate() {
        let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
        let th = -(gvec[0] * a[0] + gvec[1] * a[1] + gvec[2] * a[2]);
        let (sn, cs) = th.sin_cos();
        let mut w = 0.0f64;
        for p in 0..nprim {
            let alpha = mol._env[pe + p];
            w += mol._env[pc + p] * cfac * (pi / alpha).powf(1.5) * (-g2 / (4.0 * alpha)).exp();
        }
        assert!((re[g] - w * cs).abs() < 1e-14, "re at G[{g}]");
        assert!((im[g] - w * sn).abs() < 1e-14, "im at G[{g}]");
    }
}

/// **Test 2** — `ft_ao[μ, G=0] == ∫ φ_μ`, which is exactly `0` for `l > 0` and
/// `c·(π/α)^{3/2}` for `l = 0`. Catches an angular factor that survives at
/// `G = 0`, which the `s`-only test cannot see.
#[test]
fn g0_is_the_plain_integral() {
    use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS};
    let cell = diamond();
    let (re, im) = ft_ao_mol(&cell.mol, &[[0.0; 3]]).expect("ft_ao");
    let mut off = 0usize;
    for ib in 0..cell.mol.nbas {
        let l = cell.mol._bas[ib * BAS_SLOTS + ANG_OF] as u32;
        let n = 2 * l as usize + 1;
        if l > 0 {
            for m in 0..n {
                assert!(
                    re[off + m].abs() < 1e-13 && im[off + m].abs() < 1e-13,
                    "shell {ib} (l={l}) component {m} should integrate to zero"
                );
            }
        } else {
            assert!(
                re[off].abs() > 1e-6,
                "an s shell must have nonzero integral"
            );
        }
        off += n;
    }
}

/// **Test 3** — `_fake_nuc` shape and the two `eta` branches.
#[test]
fn fake_nuc_has_one_steep_s_shell_per_atom() {
    use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NPRIM_OF, PTR_EXP};
    let cell = diamond();
    let fk = fake_nuc(&cell, true).expect("fake_nuc with pseudo");
    assert_eq!(fk.nbas, cell.mol.natm);
    for ia in 0..fk.nbas {
        let row = ia * BAS_SLOTS;
        assert_eq!(fk._bas[row + ANG_OF], 0, "every fake shell is s");
        assert_eq!(fk._bas[row + NPRIM_OF], 1, "and a single primitive");
        let eta = fk._env[fk._bas[row + PTR_EXP] as usize];
        // gth-pade carbon has r_loc = 0.348830, so eta = 0.5/r_loc² ≈ 4.11.
        assert!(
            (1.0..100.0).contains(&eta),
            "pseudopotential eta {eta} should come from r_loc, not be 1e16"
        );
    }
    // All-electron: eta is the numerical point charge.
    let ae = fake_nuc(&he_all_electron(), false).expect("fake_nuc all-electron");
    let eta = ae._env[ae._bas[PTR_EXP] as usize];
    assert_eq!(eta, 1e16, "an all-electron fake nucleus is a point charge");
}

/// **Test 5 — the identity that makes 13-04's two `get_nuc` branches
/// consistent.**
///
/// `_get_pp_loc_part1` builds `vpplocG` two different ways
/// (`aft.py:125-135`): with a pseudopotential it uses
/// `−Σ_i SI[i,G]·get_gth_vlocG_part1[i,G]`; without one it uses
/// `Σ_i (−Z_i)·ft_ao(fakenuc)[i,G]·coulG[G]`. Those must be the SAME function of
/// `G` when the fake nucleus is given the pseudopotential's own width
/// (`eta = 0.5/r_loc²`), because both are the Fourier transform of the same
/// smeared charge. This is the strongest available check on `_fake_nuc`'s
/// normalisation, and it is the reason 13-04 can trust either branch.
#[test]
fn fake_nuc_reproduces_the_gth_local_part() {
    let cell = diamond();
    let mesh = [11usize, 11, 11];
    let gv = pyscf_pbc_gto::gv::get_gv(&cell, Some(mesh)).expect("Gv");
    let natm = cell.mol.natm;

    // Pseudopotential branch: SI · get_gth_vlocG_part1.
    let si = pyscf_pbc_gto::gv::get_si(&cell, Some(&gv), None, None).expect("SI");
    let vloc1 = pyscf_pbc_gto::pseudo::vloc::get_gth_vlocg_part1(&cell, &gv).expect("vlocG1");

    // All-electron-shaped branch, but on the PSEUDO fake nucleus.
    let fk = fake_nuc(&cell, true).expect("fake_nuc");
    let (fre, fim) = ft_ao_mol(&fk, &gv).expect("ft_ao");
    let charges = cell.mol.atom_charges();
    let coulg = pyscf_pbc_gto::get_coulg(
        &cell,
        pyscf_pbc_gto::CoulGArgs {
            gv: Some(&gv),
            ..Default::default()
        },
    )
    .expect("coulG");

    let mut worst = 0.0f64;
    for g in 0..gv.len() {
        // G = 0 is where coulG is defined to be zero and the two forms differ by the
        // divergent piece each handles separately; skip it, as upstream does.
        if gv[g].iter().all(|c| c.abs() < 1e-12) {
            continue;
        }
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        let (mut br, mut bi) = (0.0f64, 0.0f64);
        for ia in 0..natm {
            // −Σ_i SI[i,G]·vlocG1[i,G]
            ar -= si.re[ia * gv.len() + g] * vloc1[ia * gv.len() + g];
            ai -= si.im[ia * gv.len() + g] * vloc1[ia * gv.len() + g];
            // Σ_i (−Z_i)·ft_ao[i,G]·coulG[G]
            let z = -(charges[ia] as f64);
            br += z * fre[g * natm + ia] * coulg[g];
            bi += z * fim[g * natm + ia] * coulg[g];
        }
        worst = worst.max((ar - br).abs()).max((ai - bi).abs());
    }
    assert!(
        worst < 1e-12,
        "fake_nuc/ft_ao does not reproduce the GTH local part: {worst:e}"
    );
}
