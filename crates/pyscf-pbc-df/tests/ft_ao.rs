//! Plan 13-01 / 13-03 acceptance — the analytic AO-pair Fourier transform.
//!
//! **Deviation from the plan's file placement, recorded deliberately.**
//! PBC-MASTER-PLAN §8.5 plan 13-01 puts these in
//! `crates/pyscf-kernels/tests/pbc_ft_aopair.rs`. They live here instead because
//! the kernel entry point takes pre-built flat tables: exercising it at all means
//! building those tables, which is the `pyscf-pbc-df` driver. A test in
//! `pyscf-kernels` could only re-implement the driver and check the kernel
//! against that re-implementation, which is a test of a copy.

mod common;

use common::{diamond, he_all_electron};
use pyscf_pbc_df::ft_ao::{RcutChoice, estimate_rcut, ft_aopair_kpt, ft_aopair_kpt_with_images};
use pyscf_pbc_gto::pbc_intor::{PbcIntorOpts, intor_cross_with_images, pbc_intor};

/// Largest `|ft[G=0] − S|` over a k-point's `nao × nao` block.
fn dev_vs_overlap(ft_re: &[f64], ft_im: &[f64], s: &pyscf_algebra::CTensor, nao: usize) -> f64 {
    // `pbc_intor` returns F-order per component; `ft_aopair` is row-major.
    let mut w = 0.0f64;
    for i in 0..nao {
        for j in 0..nao {
            let f = i + j * nao; // F-order
            let r = i * nao + j; // row-major
            w = w.max((ft_re[r] - s.re[f]).abs());
            w = w.max((ft_im[r] - s.im[f]).abs());
        }
    }
    w
}

/// **Gate 1a** — upstream's own screening. Upstream measures 1.554e-9 at gamma
/// on this cell (`.planning/phases/13-ft-ao-aftdf/measurements/`), so 2e-9 is
/// the honest bar; 1e-10 is NOT achievable and is not a defect.
#[test]
fn gate1a_g0_equals_overlap_at_upstream_rcut() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let gv = [[0.0, 0.0, 0.0]];
    let ft = ft_aopair_kpt(&cell, &gv, [0.0; 3], [0.0; 3], RcutChoice::Upstream)
        .expect("ft_aopair at gamma");
    let s = pbc_intor(&cell, "int1e_ovlp", &[[0.0; 3]], PbcIntorOpts::default())
        .expect("periodic overlap");
    let w = dev_vs_overlap(&ft.re, &ft.im, s.at(0), nao);
    assert!(w < 2e-9, "Gate 1a: ft[G=0] deviates from int1e_ovlp by {w:e}");
}

/// **Gate 1c** — the real gate on the McMurchie–Davidson algebra. Both sides
/// converged over ONE identical image list, so nothing but the recursion, the
/// contraction or the cart→sph transform can move this.
#[test]
fn gate1c_g0_equals_overlap_over_identical_images() {
    for (name, cell) in [("diamond", diamond()), ("he", he_all_electron())] {
        let nao = cell.mol.nao_nr;
        let rcut = 1.5 * cell.try_rcut().expect("rcut");
        let ls = pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(rcut), None, true)
            .expect("lattice images");
        let gv = [[0.0, 0.0, 0.0]];
        let ft = ft_aopair_kpt_with_images(&cell, &gv, [0.0; 3], [0.0; 3], &ls)
            .expect("ft_aopair over explicit images");
        let s = intor_cross_with_images(
            "int1e_ovlp",
            &cell,
            &cell,
            &[[0.0; 3]],
            PbcIntorOpts::default(),
            &ls,
            None,
        )
        .expect("overlap over the same images");
        let w = dev_vs_overlap(&ft.re, &ft.im, s.at(0), nao);
        assert!(w < 1e-13, "Gate 1c [{name}]: deviation {w:e}");
    }
}

/// `estimate_rcut` reproduces upstream's measured value on diamond: 20.420 Bohr,
/// which is LOOSER than `cell.rcut` = 21.319 — the fact that forces Gate 1's
/// three variants.
#[test]
fn estimate_rcut_is_looser_than_cell_rcut() {
    let cell = diamond();
    let r = estimate_rcut(&cell, None).expect("estimate_rcut");
    let rmax = r.into_iter().fold(0.0f64, f64::max);
    let cr = cell.try_rcut().expect("cell rcut");
    assert!(
        (rmax - 20.420_183_850_079_926).abs() < 1e-6,
        "estimate_rcut max = {rmax}, upstream measured 20.420183850079926"
    );
    assert!(rmax < cr, "estimate_rcut {rmax} should be < cell.rcut {cr}");
}

/// **Test 1** — the `s`-`s` closed form, no oracle and no McMurchie–Davidson
/// machinery: for two `s` primitives `E_0^{00} = K_AB` and the whole transform
/// collapses to `(π/p)^{3/2}·e^{−G²/4p}·e^{−iG·P}·K_AB`. This gates the kernel's
/// prefactor, `exp` and `sincos` plumbing independently of the recursion.
#[test]
fn ss_primitive_matches_the_closed_form() {
    use pyscf_core::raw_layout::{
        ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
    };
    let cell = he_all_electron();
    assert_eq!(cell.mol.nbas, 1, "He/sto-3g is one s shell");
    assert_eq!(cell.mol._bas[ANG_OF], 0);

    let gv: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.3, -0.7, 1.1],
        [2.0, 0.5, -1.5],
        [-1.25, 3.0, 0.75],
    ];
    let rcut = 1.5 * cell.try_rcut().expect("rcut");
    let ls =
        pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(rcut), None, true).expect("images");
    let got = ft_aopair_kpt_with_images(&cell, &gv, [0.0; 3], [0.0; 3], &ls).expect("ft");

    // Host reference: the same lattice sum, written from the closed form.
    let env = &cell.mol._env;
    let nprim = cell.mol._bas[NPRIM_OF] as usize;
    let pe = cell.mol._bas[PTR_EXP] as usize;
    let pc = cell.mol._bas[PTR_COEFF] as usize;
    let atom = cell.mol._bas[ATOM_OF] as usize;
    let pcoord = cell.mol._atm[atom * ATM_SLOTS + PTR_COORD] as usize;
    let a_c = [env[pcoord], env[pcoord + 1], env[pcoord + 2]];
    let _ = BAS_SLOTS;
    // libcint's s-shell angular prefactor, squared (bra × ket).
    let cfac = pyscf_kernels::common_fac_sp(0) * pyscf_kernels::common_fac_sp(0);
    let pi = std::f64::consts::PI;

    for (ig, g) in gv.iter().enumerate() {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        let (mut wr, mut wi) = (0.0f64, 0.0f64);
        for l in &ls {
            let b = [a_c[0] + l[0], a_c[1] + l[1], a_c[2] + l[2]];
            let d = [b[0] - a_c[0], b[1] - a_c[1], b[2] - a_c[2]];
            let ab2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            for pa in 0..nprim {
                for pb in 0..nprim {
                    let (ea, eb) = (env[pe + pa], env[pe + pb]);
                    let (ca, cb) = (env[pc + pa], env[pc + pb]);
                    let p = ea + eb;
                    let kab = (-(ea * eb / p) * ab2).exp();
                    let w = ca * cb * cfac * (pi / p).powf(1.5) * kab * (-g2 / (4.0 * p)).exp();
                    let pcen = [
                        (ea * a_c[0] + eb * b[0]) / p,
                        (ea * a_c[1] + eb * b[1]) / p,
                        (ea * a_c[2] + eb * b[2]) / p,
                    ];
                    let th = -(g[0] * pcen[0] + g[1] * pcen[1] + g[2] * pcen[2]);
                    wr += w * th.cos();
                    wi += w * th.sin();
                }
            }
        }
        let (dr, di) = ((got.re[ig] - wr).abs(), (got.im[ig] - wi).abs());
        assert!(
            dr < 1e-14 && di < 1e-14,
            "G[{ig}]: closed form {wr:e}{wi:+e}i vs kernel {:e}{:+e}i",
            got.re[ig],
            got.im[ig]
        );
    }
}

/// **Test 4** — `ft[μν, G] == conj(ft[μν, −G])`, which holds for ANY `G` at
/// `k = 0` because `φ_μ(r) φ_ν(r−L)` is real. Catches a sign error in the
/// `(−i)^{t+u+v}` rotation, which the `G = 0` gate cannot see (every odd power
/// vanishes there).
///
/// Note the index order: the TRANSPOSED identity `ft[μν,G] = conj(ft[νμ,−G])`
/// is false in general. `ft[νμ,G] = e^{−iG·L}·ft[μν,G]` under `r → r+L`, so the
/// two agree only when `G` is a reciprocal-lattice vector. The G-vectors here
/// deliberately are not.
#[test]
fn ft_is_hermitian_under_g_negation() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let gpos: Vec<[f64; 3]> = vec![[0.31, -0.62, 0.93], [1.4, 0.2, -0.8], [-2.1, 1.7, 0.4]];
    let gneg: Vec<[f64; 3]> = gpos.iter().map(|g| [-g[0], -g[1], -g[2]]).collect();
    let p = ft_aopair_kpt(&cell, &gpos, [0.0; 3], [0.0; 3], RcutChoice::Upstream).expect("ft +G");
    let m = ft_aopair_kpt(&cell, &gneg, [0.0; 3], [0.0; 3], RcutChoice::Upstream).expect("ft -G");
    let mut w = 0.0f64;
    for g in 0..gpos.len() {
        for i in 0..nao {
            for j in 0..nao {
                let a = g * nao * nao + i * nao + j;
                w = w.max((p.re[a] - m.re[a]).abs());
                w = w.max((p.im[a] + m.im[a]).abs());
            }
        }
    }
    assert!(w < 1e-13, "ft[μν,G] != conj(ft[μν,−G]) by {w:e}");
}

/// **Gate 1, k-resolved** — the Bloch phase `e^{+ik·L}` must match the one
/// `pbc_intor` uses, or the identity holds at gamma and nowhere else.
#[test]
fn gate1_holds_at_every_kpoint() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let kpts = pyscf_pbc_gto::make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let rcut = 1.5 * cell.try_rcut().expect("rcut");
    let ls =
        pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(rcut), None, true).expect("images");
    let s = intor_cross_with_images(
        "int1e_ovlp",
        &cell,
        &cell,
        &kpts,
        PbcIntorOpts::default(),
        &ls,
        None,
    )
    .expect("overlap");
    for (k, kpt) in kpts.iter().enumerate() {
        let ft = ft_aopair_kpt_with_images(&cell, &[[0.0; 3]], [0.0; 3], *kpt, &ls).expect("ft");
        let w = dev_vs_overlap(&ft.re, &ft.im, s.at(k), nao);
        assert!(w < 1e-13, "Gate 1c at k={k} {kpt:?}: deviation {w:e}");
    }
}

/// **Test 2** — the numerical Fourier transform on a dense real-space grid, no
/// oracle.
///
/// For `G` on the reciprocal lattice, `e^{−iG·r}` is cell-periodic, so
/// `Σ_L ∫ φ_μ(r) φ_ν(r−L) e^{−iG·r} dr` over all space equals
/// `∫_cell φ^per_μ(r) φ^per_ν(r) e^{−iG·r} dr`, which the uniform grid
/// approximates as `Σ_r ao_μ(r)·ao_ν(r)·e^{−iG·r}·(vol/ngrids)`.
///
/// This is exactly the AFTDF-vs-FFTDF relationship, so its residual is the
/// FFT aliasing error — it shrinks with the mesh and never reaches 1e-13.
#[test]
fn matches_a_dense_grid_numerical_ft() {
    let cell = he_all_electron();
    let nao = cell.mol.nao_nr;
    let mesh = [40usize, 40, 40];
    let coords = pyscf_pbc_gto::gv::get_uniform_grids(&cell, Some(mesh), false).expect("grid");
    let gv_all = pyscf_pbc_gto::gv::get_gv(&cell, Some(mesh)).expect("Gv");
    // A handful of low-|G| reciprocal-lattice vectors: the aliasing error grows
    // with |G|, and the identity itself is exact only on the lattice.
    let pick = [0usize, 1, 2, mesh[2], mesh[2] + 1, mesh[1] * mesh[2]];
    let gv: Vec<[f64; 3]> = pick.iter().map(|&i| gv_all[i]).collect();

    let ao = pyscf_pbc_gto::eval_ao_kpts(&cell, "GTOval_sph", &coords, &[[0.0; 3]]).expect("ao");
    let ngrids = coords.len();
    let vol = cell.vol();
    let w = vol / ngrids as f64;

    let ft = ft_aopair_kpt(&cell, &gv, [0.0; 3], [0.0; 3], RcutChoice::Upstream).expect("ft");

    let mut worst = 0.0f64;
    for (ig, g) in gv.iter().enumerate() {
        for i in 0..nao {
            for j in 0..nao {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for r in 0..ngrids {
                    // AO block is F-order `(ngrids, nao)`; gamma → imag dropped.
                    let a = ao.kaos[0].re[r + i * ngrids];
                    let b = ao.kaos[0].re[r + j * ngrids];
                    let th =
                        -(g[0] * coords[r][0] + g[1] * coords[r][1] + g[2] * coords[r][2]);
                    sr += a * b * th.cos();
                    si += a * b * th.sin();
                }
                let (sr, si) = (sr * w, si * w);
                let p = ig * nao * nao + i * nao + j;
                worst = worst.max((ft.re[p] - sr).abs()).max((ft.im[p] - si).abs());
            }
        }
    }
    assert!(worst < 1e-6, "numerical FT differs by {worst:e}");
}

/// **Gate 3 (oracle)** — match upstream `pyscf.pbc.df.ft_ao.ft_aopair`.
///
/// Runs at [`RcutChoice::Upstream`], which is the whole point: this is the ONLY
/// setting at which the two screens agree, and 13-01's boxed note in
/// PBC-MASTER-PLAN §8.5 explains why sharpening `rcut` would make this WORSE
/// while making Gate 1 better.
///
/// # The bar is 1e-9, not the plan's 1e-10, and the residual is understood
///
/// Measured progression on diamond as upstream's truncation was reproduced:
///
/// | what this port screened with | worst vs upstream |
/// |---|---|
/// | one max `rcut`, no per-pair screen | 1.553e-9 |
/// | + `strip_basis` + `get_ovlp_mask` at shell level | 5.733e-10 |
/// | + the `_RangeSeparatedCell` per-primitive grouping | 5.733e-10 |
/// | + libcint's `PTR_EXPCUTOFF` (`K_AB < precision·1e-4`) | 5.121e-10 |
///
/// The worst element is at `G = 0` on a `p`-`p` diagonal, where this port is
/// the MORE accurate side: Gate 1c pins it against a matching-`Ls` overlap at
/// **1e-13**, and the screens are self-consistent (the `Upstream` result is
/// bit-identical whether the image list is built at 20.4, 32.0 or 42.6 Bohr).
/// What is left is upstream's `ExtendedMole.from_cell` image construction, which
/// D-PBC-21 declines to port. It is a truncation difference, not an algebra one.
///
/// The measured cost downstream is small — screening differences of this size
/// moved the KRHF energy by 2.607e-11 Ha in the pre-implementation study — so
/// this stays at 1e-9 unless a downstream energy gate demonstrably binds on it.
#[test]
fn matches_upstream_ft_aopair() {
    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping the upstream oracle", common::GATE);
        return;
    };
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let gv: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.31, -0.62, 0.93],
        [1.4, 0.2, -0.8],
        [-2.1, 1.7, 0.4],
        [0.5, 0.5, 0.5],
    ];
    let script = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import ft_ao

a_json, xyz_json, sym_json, basis, pseudo, gv_json, kpt_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
Gv = np.asarray(json.loads(gv_json))
kpt = np.asarray(json.loads(kpt_json))
val = ft_ao.ft_aopair(c, Gv, kpti_kptj=np.array([kpt, kpt]))
val = np.asarray(val)
out = {'nao': int(c.nao_nr()), 'ng': int(Gv.shape[0]),
       'version': __import__('pyscf').__version__,
       're': np.real(val).ravel().tolist(),
       'im': np.imag(val).ravel().tolist()}
print(json.dumps(out))
"#;
    let gv_json = serde_json::to_string(
        &gv.iter().map(|g| g.to_vec()).collect::<Vec<_>>(),
    )
    .expect("json");
    let mut args = common::cell_args(&cell, &[]);
    args.insert(3, "gth-szv".into());
    args.insert(4, "gth-pade".into());
    args.push(gv_json);
    args.push("[0.0,0.0,0.0]".into());
    let want = common::run_python(&py, script, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );

    let got = ft_aopair_kpt(&cell, &gv, [0.0; 3], [0.0; 3], RcutChoice::Upstream).expect("ft");
    let re: Vec<f64> = want["re"]
        .as_array()
        .expect("re")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let im: Vec<f64> = want["im"]
        .as_array()
        .expect("im")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    assert_eq!(re.len(), gv.len() * nao * nao, "upstream shape");
    let mut w = 0.0f64;
    let mut at = 0usize;
    for p in 0..re.len() {
        let d = (got.re[p] - re[p]).abs().max((got.im[p] - im[p]).abs());
        if d > w {
            w = d;
            at = p;
        }
    }
    let (g, rem) = (at / (nao * nao), at % (nao * nao));
    eprintln!(
        "worst at G[{g}] ({:?}) elem ({},{}) got {:.17e}{:+.17e}i want {:.17e}{:+.17e}i",
        gv[g], rem / nao, rem % nao, got.re[at], got.im[at], re[at], im[at]
    );
    assert!(w < 1e-9, "ft_aopair deviates from upstream by {w:e}");
}

/// The `_RangeSeparatedCell` split, pinned against upstream.
///
/// `ft_ao.estimate_rcut(_RangeSeparatedCell.from_cell(cell, 10.0, 1.0))` on
/// diamond/`gth-szv`. Each `s` shell splits into a 3-primitive local block and a
/// 1-primitive smooth block with materially different radii; the `p` shell does
/// not split. This is the subtlest step in plan 13-01 and it gets its own gate.
#[test]
fn rs_group_rcuts_match_upstream() {
    let got = pyscf_pbc_df::ft_ao::rs_group_rcuts(&diamond()).expect("rs groups");
    let want = [
        16.123_457_658_142_51,
        20.113_957_729_325_875,
        20.420_183_850_079_926,
        16.123_457_658_142_51,
        20.113_957_729_325_875,
        20.420_183_850_079_926,
    ];
    assert_eq!(got.len(), want.len(), "RS group count");
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-9, "RS rcut {g} vs upstream {w}");
    }
}

/// **Precision tuning, pinned end-to-end.**
///
/// Building the cell with a tighter `precision` converges BOTH sides of the
/// `G = 0` identity — `ft_aopair`'s own lattice sum and `pbc_intor`'s — so the
/// Gate-1b floor that `RcutChoice::Scaled` runs into (1.472e-10, which belongs
/// to `pbc_intor`) moves with it:
///
/// | `precision` | `cell.rcut` | residual |
/// |---|---|---|
/// | 1e-8 (default) | 21.319 | 1.189e-9 |
/// | 1e-10 | 23.193 | 1.142e-11 |
/// | 1e-12 | 24.910 | 8.416e-14 |
///
/// It must be set at BUILD time: `cell.rcut` is cached during `Cell::build`, so
/// a post-hoc `cell.precision = p` tightens only this module's call-time
/// `estimate_rcut` and leaves the reference loose.
#[test]
fn build_time_precision_converges_both_sides_of_gate1() {
    let mut devs = Vec::new();
    for prec in [1e-8f64, 1e-10, 1e-12] {
        let cell = common::diamond_prec(prec);
        let nao = cell.mol.nao_nr;
        let ft = ft_aopair_kpt(&cell, &[[0.0; 3]], [0.0; 3], [0.0; 3], RcutChoice::Upstream)
            .expect("ft_aopair");
        let s = pbc_intor(&cell, "int1e_ovlp", &[[0.0; 3]], PbcIntorOpts::default())
            .expect("periodic overlap");
        let w = dev_vs_overlap(&ft.re, &ft.im, s.at(0), nao);
        eprintln!(
            "cell.precision {prec:.0e}  cell.rcut {:.3}  |ft[G=0] − int1e_ovlp| = {w:e}",
            cell.try_rcut().expect("rcut")
        );
        devs.push(w);
    }
    assert!(
        devs[1] < devs[0] * 0.01,
        "1e-10 must improve on 1e-8 by >100x: {:e} vs {:e}",
        devs[1],
        devs[0]
    );
    assert!(
        devs[2] < 1e-12,
        "cell.precision = 1e-12 should reach ~1e-13, got {:e}",
        devs[2]
    );
}
