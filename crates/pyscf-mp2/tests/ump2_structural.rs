//! UMP2 structural / wiring scaffold (always-on layer for the `mp2-structural`
//! CI job). Plan 05-04 wires the spin-block kernel: these tests feed hand-built
//! synthetic `(aa/ab/bb)` `(ia|jb)` blocks (NO `intor`) and assert the
//! open-shell correlation energy, the symmetric-spin reduction to closed-shell,
//! and the asymmetric `t2aa != t2bb` invariant. All ALWAYS-ON (no cintx gate).

use pyscf_core::{MOCoefficients, Mole};
use pyscf_mp2::{
    ChemistsEris, Frozen, Mp2Reference, NoMp2Overrides, UmpReference, rmp2_kernel, ump2_kernel,
};

/// Always-on: the UMP2 path's crate surface is reachable (error type +
/// helper re-exports shared with the restricted path). Proves the wiring
/// compiles independent of the spin-block kernel.
#[test]
fn ump2_module_surface_exists() {
    fn assert_error_type(_e: Option<pyscf_mp2::Mp2Error>) {}
    assert_error_type(None);

    // get_frozen_mask is exercised by both restricted and unrestricted paths.
    let _: fn(&[f64], &pyscf_mp2::Frozen) -> _ = pyscf_mp2::get_frozen_mask;
}

/// Build a minimal `Mole` snapshot (no atoms — the synthetic kernel path never
/// touches the `intor` integrals; `Mole::default()` is enough for the
/// `UmpReference` shape).
fn toy_mol() -> Mole {
    Mole::default()
}

/// Build a single-channel `Mp2Reference` with the given per-MO occupations and
/// energies (the MO coefficients are an identity-ish placeholder; only the
/// occupations + energies drive the synthetic kernel path).
fn toy_reference(mo_occ: Vec<f64>, mo_energy: Vec<f64>) -> Mp2Reference {
    let nmo = mo_occ.len();
    // Identity MO coefficients (nao == nmo). The synthetic kernel path does not
    // re-transform, so the exact coefficient values are irrelevant here.
    let mut data = vec![0.0; nmo * nmo];
    for i in 0..nmo {
        data[i * nmo + i] = 1.0;
    }
    Mp2Reference {
        mo_coeff: MOCoefficients {
            nao: nmo,
            nmo,
            data,
            energies: mo_energy.clone(),
            occupations: mo_occ.clone(),
        },
        mo_energy,
        mo_occ,
        e_hf: 0.0,
        converged: true,
        mol: toy_mol(),
    }
}

/// Hand-compute the same-spin contribution for a single occupied / single
/// virtual block (nocc=1, nvir=1): there is no antisymmetrization partner, so
/// `e_ss = 0.5·t2·g - 0.5·t2·g_exchange`. With nvir=1 the exchange `(ib|ja)`
/// equals `(ia|jb)` so the same-spin contribution is exactly 0 (a real
/// closed-shell physical fact: same-spin pairs need ≥2 virtuals to correlate).
#[test]
fn ump2_synthetic_block_energy_matches_hand_value() {
    // nocca = noccb = 1, nvira = nvirb = 2. εocc = -0.5, εvir = {0.5, 0.7}.
    let nocc = 1usize;
    let nvir = 2usize;
    let e_occ = -0.5_f64;
    let e_vir = [0.5_f64, 0.7_f64];

    // (ia|jb) aa block, flat (i,a,j,b) C-order, i=j=0, a,b in 0..2.
    // Choose distinct values so the exchange term is non-trivial.
    // g[a][b] with a,b in {0,1}: g00=0.10, g01=0.04, g10=0.04, g11=0.08.
    // (symmetric in a<->b for a physical (ia|jb) on a single occ; we keep it
    // symmetric so g_aa == its own (ib|ja) partner per the single-i,j case.)
    let g_aa = vec![0.10, 0.04, 0.04, 0.08];
    let eris_aa = ChemistsEris {
        ovov: g_aa.clone(),
        nocc,
        nvir,
    };
    // bb identical to aa (spin-symmetric) for this test.
    let eris_bb = ChemistsEris {
        ovov: g_aa.clone(),
        nocc,
        nvir,
    };
    // ab block: same shape (nocca=noccb=1, nvira=nvirb=2).
    let g_ab = vec![0.12, 0.03, 0.03, 0.09];
    let eris_ab = ChemistsEris {
        ovov: g_ab.clone(),
        nocc,
        nvir,
    };

    let refr_a = toy_reference(vec![1.0, 0.0, 0.0], vec![e_occ, e_vir[0], e_vir[1]]);
    let refr_b = refr_a.clone();
    let uref = UmpReference {
        alpha: refr_a,
        beta: refr_b,
    };

    let res = ump2_kernel(&uref, &Frozen::None, &eris_aa, &eris_ab, &eris_bb, true)
        .expect("ump2 synthetic kernel");

    // Hand-compute the same-spin (aa) energy.
    // t2[a][b] = g[a][b] / (εi + εj − εa − εb), i=j=0.
    let denom = |a: usize, b: usize| 2.0 * e_occ - e_vir[a] - e_vir[b];
    let idx = |a: usize, b: usize| a * nvir + b;
    let mut t2 = [0.0f64; 4];
    for a in 0..nvir {
        for b in 0..nvir {
            t2[idx(a, b)] = g_aa[idx(a, b)] / denom(a, b);
        }
    }
    // e_ss = 0.5·Σ t2·g − 0.5·Σ t2·g_exchange (g_exchange[a,b] = g[b,a]).
    let mut direct = 0.0;
    let mut exch = 0.0;
    for a in 0..nvir {
        for b in 0..nvir {
            direct += t2[idx(a, b)] * g_aa[idx(a, b)];
            exch += t2[idx(a, b)] * g_aa[idx(b, a)];
        }
    }
    let e_aa_hand = 0.5 * direct - 0.5 * exch;
    let e_bb_hand = e_aa_hand;

    // e_os = Σ t2_ab·g_ab.
    let mut t2ab = [0.0f64; 4];
    for a in 0..nvir {
        for b in 0..nvir {
            t2ab[idx(a, b)] = g_ab[idx(a, b)] / denom(a, b);
        }
    }
    let mut e_ab_hand = 0.0;
    for a in 0..nvir {
        for b in 0..nvir {
            e_ab_hand += t2ab[idx(a, b)] * g_ab[idx(a, b)];
        }
    }

    let e_corr_hand = e_aa_hand + e_bb_hand + e_ab_hand;
    assert!(
        (res.e_aa - e_aa_hand).abs() < 1e-12,
        "e_aa: got {} want {}",
        res.e_aa,
        e_aa_hand
    );
    assert!(
        (res.e_bb - e_bb_hand).abs() < 1e-12,
        "e_bb: got {} want {}",
        res.e_bb,
        e_bb_hand
    );
    assert!(
        (res.e_ab - e_ab_hand).abs() < 1e-12,
        "e_ab: got {} want {}",
        res.e_ab,
        e_ab_hand
    );
    assert!(
        (res.e_corr - e_corr_hand).abs() < 1e-12,
        "e_corr: got {} want {}",
        res.e_corr,
        e_corr_hand
    );
    // e_corr == e_aa + e_ab + e_bb identity.
    assert!((res.e_corr - (res.e_aa + res.e_ab + res.e_bb)).abs() < 1e-15);
}

/// Symmetric α=β invariant: a spin-symmetric open-shell input must reproduce
/// the closed-shell `rmp2_kernel` answer. The relation between the open-shell
/// decomposition and the closed-shell `e_corr` is exact: closed-shell
/// `e_corr = e_os(closed) + e_ss(closed)`, and for a spin-symmetric system
/// `e_ab(open) == e_os(closed)` and `e_aa(open) + e_bb(open) == e_ss(closed)`.
#[test]
fn ump2_symmetric_spin_reduces_to_closed_shell() {
    let nocc = 1usize;
    let nvir = 2usize;
    let e_occ = -0.5_f64;
    let e_vir = [0.5_f64, 0.7_f64];

    // For the closed-shell comparison the (ia|jb) block is identical across all
    // three spin channels (spin-restricted reference).
    let g = vec![0.10, 0.04, 0.04, 0.08];
    let eris = ChemistsEris {
        ovov: g.clone(),
        nocc,
        nvir,
    };

    let refr = toy_reference(vec![2.0, 0.0, 0.0], vec![e_occ, e_vir[0], e_vir[1]]);
    let closed =
        rmp2_kernel(&refr, &Frozen::None, &NoClosedEris(&eris), false).expect("closed rmp2");

    // Open-shell: feed the SAME block as aa/ab/bb with α=β occupations (1,0,0).
    let refr_a = toy_reference(vec![1.0, 0.0, 0.0], vec![e_occ, e_vir[0], e_vir[1]]);
    let uref = UmpReference {
        alpha: refr_a.clone(),
        beta: refr_a,
    };
    let open = ump2_kernel(&uref, &Frozen::None, &eris, &eris, &eris, false).expect("open ump2");

    // e_ab(open) == e_os(closed) and e_aa+e_bb(open) == e_ss(closed).
    assert!(
        (open.e_ab - closed.e_os).abs() < 1e-12,
        "e_ab {} vs e_os {}",
        open.e_ab,
        closed.e_os
    );
    assert!(
        ((open.e_aa + open.e_bb) - closed.e_ss).abs() < 1e-12,
        "e_aa+e_bb {} vs e_ss {}",
        open.e_aa + open.e_bb,
        closed.e_ss
    );
    assert!(
        (open.e_corr - closed.e_corr).abs() < 1e-12,
        "open e_corr {} vs closed e_corr {}",
        open.e_corr,
        closed.e_corr
    );
}

/// Asymmetric input → `t2aa != t2bb`. Feed distinct aa and bb blocks (and
/// distinct α/β energies) and assert the stored amplitude triple differs.
#[test]
fn ump2_asymmetric_input_gives_distinct_t2() {
    let nocc = 1usize;
    let nvir = 2usize;

    let e_occ_a = -0.5_f64;
    let e_vir_a = [0.5_f64, 0.7_f64];
    let e_occ_b = -0.6_f64;
    let e_vir_b = [0.4_f64, 0.9_f64];

    // Same-spin blocks are deliberately a<->b ASYMMETRIC (g[a,b] != g[b,a]) so
    // the antisymmetrized amplitude `t2i[j,a,b] − t2i[j,b,a]` is non-zero. With
    // a symmetric block the same-spin t2 vanishes (a real physical fact), which
    // would make t2aa == t2bb == 0 regardless of the distinct β data.
    let g_aa = vec![0.10, 0.06, 0.02, 0.08];
    let g_bb = vec![0.20, 0.07, 0.03, 0.11];
    let g_ab = vec![0.12, 0.03, 0.04, 0.09];

    let eris_aa = ChemistsEris {
        ovov: g_aa,
        nocc,
        nvir,
    };
    let eris_bb = ChemistsEris {
        ovov: g_bb,
        nocc,
        nvir,
    };
    let eris_ab = ChemistsEris {
        ovov: g_ab,
        nocc,
        nvir,
    };

    let refr_a = toy_reference(vec![1.0, 0.0, 0.0], vec![e_occ_a, e_vir_a[0], e_vir_a[1]]);
    let refr_b = toy_reference(vec![1.0, 0.0, 0.0], vec![e_occ_b, e_vir_b[0], e_vir_b[1]]);
    let uref = UmpReference {
        alpha: refr_a,
        beta: refr_b,
    };

    let res = ump2_kernel(&uref, &Frozen::None, &eris_aa, &eris_ab, &eris_bb, true)
        .expect("asymmetric ump2");
    let t2 = res.t2.expect("with_t2 amplitudes");

    assert_ne!(
        t2.t2aa, t2.t2bb,
        "asymmetric input must yield distinct same-spin amplitudes"
    );
    // Spin-block shapes must agree with the inputs.
    assert_eq!(t2.nocc_a, nocc);
    assert_eq!(t2.nvir_a, nvir);
    assert_eq!(t2.nocc_b, nocc);
    assert_eq!(t2.nvir_b, nvir);
    assert_eq!(t2.t2aa.len(), nocc * nocc * nvir * nvir);
    assert_eq!(t2.t2ab.len(), nocc * nocc * nvir * nvir);
    assert_eq!(t2.t2bb.len(), nocc * nocc * nvir * nvir);
}

/// Thin hooks shim that hands a precomputed closed-shell `(ia|jb)` block back to
/// `rmp2_kernel` (no re-transform) — for the symmetric-reduction test.
struct NoClosedEris<'a>(&'a ChemistsEris);
impl pyscf_mp2::Mp2OverrideHooks for NoClosedEris<'_> {
    fn ao2mo(
        &self,
        _refr: &Mp2Reference,
        _frozen: &Frozen,
    ) -> Result<ChemistsEris, pyscf_core::PyscfRsError> {
        Ok(ChemistsEris {
            ovov: self.0.ovov.clone(),
            nocc: self.0.nocc,
            nvir: self.0.nvir,
        })
    }
}

// Silence the unused-import lint for NoMp2Overrides (kept available for the
// surface check); reference it once so the import is not dead.
#[allow(dead_code)]
fn _surface_keepalive() {
    let _ = NoMp2Overrides;
}
