//! Plan 11-02 — `get_coulG`, `madelung` and the Ewald exchange-divergence shift.
//!
//! Oracle-free structural gates plus tier-2 hard-coded numbers captured once
//! from live PySCF 2.12.1 (D-PBC-19). Regenerate with:
//!
//! ```python
//! from pyscf.pbc import gto, tools
//! c = gto.Cell(a=..., atom=..., basis='gth-szv', pseudo='gth-pade', unit='Bohr').build()
//! tools.pbc.madelung(c, c.make_kpts([2, 2, 2]))
//! tools.get_coulG(c, mesh=[11]*3, Gv=c.get_Gv([11]*3))
//! ```

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, CoulGArgs, ExxDiv, get_coulg, get_coulg_at_gv, madelung,
};

/// Diamond in BOHR — the same cell `tests/pbc_intor.rs` uses, so every number
/// here is comparable with the Phase-10 gate.
fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond builds")
}

const MESH: [usize; 3] = [11, 11, 11];

// ---------------------------------------------------------------------------
// Upstream reference numbers
// ---------------------------------------------------------------------------

/// `tools.pbc.madelung(diamond, make_kpts(nk))`, PySCF 2.12.1.
const MADELUNG: [([usize; 3], f64); 4] = [
    ([1, 1, 1], 0.6801820115172746),
    ([2, 2, 2], 0.3400910057586376),
    ([3, 3, 3], 0.22672733717242355),
    ([2, 1, 1], 0.4697199871782931),
];

#[test]
fn madelung_matches_upstream_on_diamond() {
    let cell = diamond();
    for (nk, want) in MADELUNG {
        let kpts = pyscf_pbc_gto::make_kpts_default(&cell, nk).expect("kpts");
        let got = madelung(&cell, &kpts, None).expect("madelung");
        assert!(
            (got - want).abs() < 1e-9,
            "madelung({nk:?}): got {got:.15}, want {want:.15}, delta {:e}",
            got - want
        );
    }
}

/// `tools.get_coulG(cell, mesh=[11]*3, Gv=...)` — the plain 3-D kernel.
#[test]
fn coulg_matches_upstream_at_gamma() {
    let cell = diamond();
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let coulg = get_coulg_at_gv(&cell, MESH, &gv).expect("coulG");
    assert_eq!(coulg.len(), 11 * 11 * 11);
    let sum: f64 = coulg.iter().sum();
    assert!(
        (sum - 417.21865451920996).abs() < 1e-9,
        "sum(coulG) = {sum:.12}"
    );
    assert_eq!(coulg[0], 0.0, "G = 0 must be exactly zero without exxdiv");
    assert!((coulg[1] - 4.820933479677528).abs() < 1e-12);
}

/// A finite k offset — this is the `_Gv_wrap_around` path, where a fold error
/// would show up as a completely different sum.
#[test]
fn coulg_matches_upstream_at_a_k_offset() {
    let cell = diamond();
    let kpts = pyscf_pbc_gto::make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let k = [
        kpts[1][0] - kpts[3][0],
        kpts[1][1] - kpts[3][1],
        kpts[1][2] - kpts[3][2],
    ];
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let coulg = get_coulg(
        &cell,
        CoulGArgs {
            k,
            mesh: Some(MESH),
            gv: Some(&gv),
            ..CoulGArgs::new()
        },
    )
    .expect("coulG");
    let sum: f64 = coulg.iter().sum();
    assert!((sum - 448.5039236503139).abs() < 1e-9, "sum = {sum:.12}");
    assert!((coulg[0] - 19.283733918710112).abs() < 1e-12);
    assert!((coulg[5] - 0.17910588778987718).abs() < 1e-12);
}

/// `exxdiv = 'ewald'` shifts ONLY the `G+k = 0` entry, by `Nk * vol * madelung`.
#[test]
fn ewald_exxdiv_shifts_only_g0() {
    let cell = diamond();
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let plain = get_coulg_at_gv(&cell, MESH, &gv).expect("coulG");
    let shifted = get_coulg(
        &cell,
        CoulGArgs {
            exxdiv: Some(ExxDiv::Ewald),
            mesh: Some(MESH),
            gv: Some(&gv),
            ..CoulGArgs::new()
        },
    )
    .expect("coulG ewald");
    assert!(
        (shifted[0] - 52.07970918951438).abs() < 1e-9,
        "G0 = {:.12}",
        shifted[0]
    );
    for i in 1..plain.len() {
        assert_eq!(plain[i], shifted[i], "entry {i} must be untouched");
    }
}

// ---------------------------------------------------------------------------
// Oracle-free structure
// ---------------------------------------------------------------------------

/// `coulG` is even in `G`: the kernel depends on `|k+G|` alone. At gamma the
/// `G -> -G` partner of grid index `g` is found through the integer frequencies.
#[test]
fn coulg_is_even_in_g() {
    let cell = diamond();
    let [mx, my, mz] = MESH;
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let coulg = get_coulg_at_gv(&cell, MESH, &gv).expect("coulG");
    for ix in 0..mx {
        for iy in 0..my {
            for iz in 0..mz {
                let g = (ix * my + iy) * mz + iz;
                let n = (((mx - ix) % mx) * my + (my - iy) % my) * mz + (mz - iz) % mz;
                assert!(
                    (coulg[g] - coulg[n]).abs() < 1e-14 * coulg[g].abs().max(1.0),
                    "coulG[{g}] = {} != coulG[{n}] = {}",
                    coulg[g],
                    coulg[n]
                );
            }
        }
    }
}

/// A long-range attenuated kernel is strictly below the full-range one, and the
/// long- and short-range halves add back up to it.
#[test]
fn range_separation_splits_the_kernel() {
    let cell = diamond();
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let full = get_coulg_at_gv(&cell, MESH, &gv).expect("coulG");
    let mk = |omega: f64| {
        get_coulg(
            &cell,
            CoulGArgs {
                mesh: Some(MESH),
                gv: Some(&gv),
                omega: Some(omega),
                ..CoulGArgs::new()
            },
        )
        .expect("coulG omega")
    };
    let lr = mk(0.3);
    let sr = mk(-0.3);
    for i in 0..full.len() {
        assert!(lr[i] <= full[i] + 1e-15, "LR exceeds full range at {i}");
        assert!(sr[i] >= -1e-15, "SR went negative at {i}");
        assert!(
            (lr[i] + sr[i] - full[i]).abs() < 1e-12 * full[i].abs().max(1.0),
            "LR + SR != full at {i}: {} + {} vs {}",
            lr[i],
            sr[i],
            full[i]
        );
    }
}

/// `madelung` scales like `1/Nk^{1/3}` as the probe supercell grows, which is
/// the finite-size scaling the Ewald `exxdiv` exists to remove.
#[test]
fn madelung_decreases_with_denser_k_mesh() {
    let cell = diamond();
    let mut last = f64::INFINITY;
    for nk in [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]] {
        let kpts = pyscf_pbc_gto::make_kpts_default(&cell, nk).expect("kpts");
        let m = madelung(&cell, &kpts, None).expect("madelung");
        assert!(m > 0.0, "madelung must be positive, got {m}");
        assert!(m < last, "madelung must fall with denser k: {m} !< {last}");
        last = m;
    }
}

/// D-PBC-20, plan 12-08 — the two truncated-Coulomb kernels produce a FINITE
/// kernel, including at `G + k = 0` where the untruncated `4π/G²` diverges.
/// That finiteness at the origin is the entire point of truncating.
#[test]
fn vcut_branches_produce_a_finite_kernel() {
    let cell = diamond();
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH)).expect("Gv");
    let kpts = pyscf_pbc_gto::make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    for div in [ExxDiv::VcutSph, ExxDiv::VcutWs] {
        let v = get_coulg(
            &cell,
            CoulGArgs {
                exxdiv: Some(div),
                mesh: Some(MESH),
                gv: Some(&gv),
                kpts: Some(&kpts),
                ..CoulGArgs::new()
            },
        )
        .unwrap_or_else(|e| panic!("{div:?}: {e}"));
        assert_eq!(v.len(), gv.len(), "{div:?}: one kernel value per G");
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{div:?}: the truncated kernel must be finite everywhere, including G = 0"
        );
        // `vcut_sph`'s G = 0 value is the analytic `2 pi Rc^2` with
        // `Rc = (3 Nk V / 4 pi)^(1/3)` (pbc.py:374-378) — a closed form, so it
        // is checked against the formula rather than against a recorded number.
        if matches!(div, ExxDiv::VcutSph) {
            let nk = kpts.len() as f64;
            let rc = (3.0 * nk * cell.vol() / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);
            let want = 2.0 * std::f64::consts::PI * rc * rc;
            let g0 = gv
                .iter()
                .position(|g| g[0] == 0.0 && g[1] == 0.0 && g[2] == 0.0)
                .expect("a G = 0 entry");
            assert!(
                (v[g0] - want).abs() < 1e-10 * want.abs(),
                "vcut_sph at G = 0 is {}, expected the analytic 2 pi Rc^2 = {want}",
                v[g0]
            );
        }
    }
}
