//! Plan 16-06's oracle gates — `KUCCSD` against vendored PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_kuccsd -- --ignored --nocapture
//! ```
//!
//! # The fixture is the ONLY genuinely open-shell one in Phase 16
//!
//! `pbc/cc/test/test_kuccsd_openshell.py`'s three-hydrogen cell on a two-`s`
//! basis, with `cell.spin` multiplied by `nkpts` as upstream's test does. At
//! `[1,1,2]` that gives `nocca = 2`, `noccb = 1`, `nmo = 6` — the alpha and
//! beta occupations DIFFER, which is what makes the `ab` channel and the
//! `BbAa` ERI pass reachable at all. A closed-shell fixture would run every
//! line of this port and prove almost nothing, because `nocca == noccb` makes
//! several genuinely-different shapes coincide.
//!
//! # Every gate is driven from UPSTREAM's own mean field
//!
//! For the reason `oracle_phase16.rs`'s `Upstream` doc gives, and one more
//! that is specific to the unrestricted case: two UHF solvers need not find
//! the same SCF solution at all, so a correlation energy compared across two
//! independently converged KUHFs is not a measurement of anything.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_cc::kccsd_rhf::KrccsdOpts;
use pyscf_pbc_cc::kccsd_uhf::{
    add_vvvv_for_test, energy, fock_terms_for_test, init_amps, kernel, update_amps,
    woooo_terms_for_test, wovvo_terms_for_test,
};
use pyscf_pbc_cc::kintermediates_uhf::{
    UT1, UT2, cc_foo, cc_fov, cc_fvv, cc_woooo, cc_wovvo, cc_wvvvv_half, make_tau, make_tau2,
};
use pyscf_pbc_cc::kuccsd_rdm::{Gamma1, make_rdm1, make_rdm1_from_gamma1};
use pyscf_pbc_cc::kueris::{KuEris, UBlk, UKind, UPass};
use pyscf_pbc_cc::{KErisOpts, ZArr};
use pyscf_pbc_df::{Fftdf, MoCoeff};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;
use pyscf_runtime::ZWorkspacePool;

const GATE: &str = "PYSCF_ORACLE_VENV";
const EMITTER: &str = ".planning/phases/16-periodic-cc-ci/measurements/oracle_phase16.py";

/// The ERI-block gate, MEASURED on this fixture and not inherited.
///
/// All twenty-six blocks land between `1.16e-10` and `8.07e-10` at the primary
/// `[31,31,31]` mesh, flat across `oooo` … `vvvv`. `1e-9` sits just above the
/// largest measured residual — four orders TIGHTER than the restricted phase's
/// `1e-6`, which is possible only because this fixture's mesh actually resolves
/// its own basis; see [`the_eri_residual_is_the_mesh_and_not_the_port`] for the
/// measurement that establishes both facts.
const ERI_BLOCK: f64 = 1e-9;

/// The same table at upstream's pinned `[13,13,13]`, where it is NOT flat:
/// `2.7e-9` on `oooo` rising monotonically to `1.2e-5` on `vvvv`. This bound
/// exists so the coarse half of the mesh measurement still fails on a real
/// defect, which would be `O(1)`.
const ERI_BLOCK_COARSE: f64 = 1e-4;

/// The refinement `[13,13,13] -> [31,31,31]` must buy at least this factor on
/// the worst block. Measured: `1.2012e-5 / 8.073e-10 = 14 880`.
const MESH_REFINEMENT_FACTOR: f64 = 1e3;

/// `update_amps` and every intermediate, on a FIXED synthetic amplitude
/// quintuple. MEASURED: the whole set — nine `tau`s (bit-identical), six Fock
/// intermediates, twelve `W`s, four equation stages and fifteen amplitude
/// arrays — lands between `0` and `2.3e-10`, so `1e-9` sits just above it and
/// inherits the ERI floor rather than being loosened past it.
const AMPS_BLOCK: f64 = 1e-9;

/// `e_corr` against upstream, the same `1e-7` `measurements/README.md §1` sets
/// for G1 (`KRCCSD e_corr` vs upstream, FFTDF).
const E_CORR: f64 = 1e-7;

/// `emp2` from `init_amps`. `1e-9`, as the restricted `emp2` gate is — both
/// sides are limited by the SCF's own `conv_tol`.
const EMP2: f64 = 1e-9;

/// The 1-RDM on FIXED synthetic amplitudes — the gate on the equations
/// themselves, with no convergence in it. MEASURED at `6.9e-18`, i.e. the
/// two implementations are bit-identical up to summation order, so this is
/// `1e-15` and not a round number chosen for comfort.
const RDM_SYNTHETIC: f64 = 1e-15;

/// The 1-RDM on the CONVERGED amplitudes. This one does NOT measure the
/// density-matrix code: `dm1` is a direct function of `t1`/`t2`, and the two
/// sides' converged amplitudes agree only to whatever `conv_tol` bought. The
/// test prints `max|Δt1|` beside the result so the two numbers can be compared,
/// and the gate is set from the amplitude spread rather than from the RDM.
const RDM_CONVERGED: f64 = 1e-7;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    Some(if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        root().join(".venv/bin/python")
    } else {
        let path = PathBuf::from(&raw);
        if path.is_dir() {
            path.join("bin/python")
        } else {
            path
        }
    })
}

fn emit(section: &str) -> Option<String> {
    let Some(python) = python() else {
        eprintln!("skip: set {GATE} to arm the PySCF oracle");
        return None;
    };
    let out = Command::new(python)
        .args(["-u", EMITTER, section])
        .current_dir(root())
        .env("PYTHONPATH", root())
        .output()
        .expect("run the Phase-16 oracle emitter");
    assert!(
        out.status.success(),
        "oracle section {section} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8(out.stdout).expect("UTF-8 oracle output");
    assert!(
        s.contains("pyscf.__version__=2.12.1"),
        "the oracle is not PySCF 2.12.1"
    );
    Some(s)
}

fn block(out: &str, name: &str) -> Vec<f64> {
    let head = format!("BEGIN {name} n=");
    let start = out
        .find(&head)
        .unwrap_or_else(|| panic!("oracle emitted no block {name}"));
    let rest = &out[start + head.len()..];
    let nl = rest.find('\n').expect("block header");
    let n: usize = rest[..nl].trim().parse().expect("block count");
    let body = &rest[nl + 1..];
    let end = body
        .find(&format!("END {name}"))
        .unwrap_or_else(|| panic!("block {name} is unterminated"));
    let v: Vec<f64> = body[..end]
        .split_ascii_whitespace()
        .map(|t| t.parse().expect("f64"))
        .collect();
    assert_eq!(v.len(), n, "block {name}: declared {n}, got {}", v.len());
    v
}

fn cblock(out: &str, name: &str) -> CTensor {
    let v = block(out, name);
    assert_eq!(v.len() % 2, 0, "block {name} is not complex");
    CTensor::from_planes(
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

fn scalar(out: &str, key: &str) -> f64 {
    let head = format!("{key}=");
    for line in out.lines() {
        if let Some(v) = line.trim().strip_prefix(&head) {
            return v.trim().parse().expect("scalar");
        }
    }
    panic!("oracle emitted no scalar {key}");
}

fn maxdiff(got: &ZArr, want: &CTensor, name: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.re.len(),
        "{name}: {} elements here, {} upstream",
        got.len(),
        want.re.len()
    );
    let mut m = 0.0_f64;
    for i in 0..got.len() {
        m = m
            .max((got.data().re[i] - want.re[i]).abs())
            .max((got.data().im[i] - want.im[i]).abs());
    }
    m
}

/// `test_kuccsd_openshell.py:10-24`, built here rather than in `common` because
/// it is the ONLY open-shell cell in the phase and nothing else wants it.
fn h3_openshell(mesh: [usize; 3]) -> Cell {
    let h = 6.740_274_66_f64;
    let d = 1.685_068_66_f64;
    let basis = "H    S\n      1.0   1.0\nH    S\n      0.5   1.0\n";
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("H".into(), [0.0, 0.0, 0.0]),
                ("H".into(), [d, d, d]),
                ("H".into(), [2.0 * d, 2.0 * d, 2.0 * d]),
            ]),
            basis: BasisInput::NwchemText(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        mesh: Some(mesh),
        ..Default::default()
    })
    .expect("H3 open-shell cell")
}

/// One spin's padded MO set, read straight out of the oracle stream.
fn padded(out: &str, spin: &str, nocc: usize, nmo: usize, nkpts: usize, nao: usize) -> PaddedMos {
    let c = cblock(out, &format!("mo_coeff_{spin}"));
    assert_eq!(c.re.len(), nkpts * nao * nmo, "mo_coeff_{spin} shape");
    let mo_coeff: Vec<MoCoeff> = (0..nkpts)
        .map(|k| {
            let off = k * nao * nmo;
            MoCoeff::new(
                nao,
                nmo,
                CTensor {
                    re: c.re[off..off + nao * nmo].to_vec(),
                    im: c.im[off..off + nao * nmo].to_vec(),
                },
            )
        })
        .collect();
    let me = block(out, &format!("mo_energy_{spin}"));
    let mo_energy: Vec<Vec<f64>> = (0..nkpts)
        .map(|k| me[k * nmo..(k + 1) * nmo].to_vec())
        .collect();
    let nocc_per_kpt: Vec<usize> = block(out, &format!("nocc_per_kpt_{spin}"))
        .iter()
        .map(|v| *v as usize)
        .collect();
    PaddedMos {
        mo_coeff,
        mo_energy,
        nmo_per_kpt: vec![nmo; nkpts],
        nocc_per_kpt,
        nmo,
        nocc,
    }
}

/// The 26 blocks, paired with the name the oracle emitted them under.
fn all_blocks() -> Vec<(UBlk, &'static str)> {
    UBlk::all().into_iter().map(|b| (b, b.name())).collect()
}

/// The same SplitMix64 stream the emitter's `draw` uses, drawn in
/// `amplitudes_to_vector`'s order.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
    fn draw(&mut self, shape: &[usize]) -> ZArr {
        let n: usize = shape.iter().product();
        let mut a = ZArr::zeros(shape);
        for i in 0..n {
            a.data_mut().re[i] = 0.05 * self.unit();
            a.data_mut().im[i] = 0.05 * self.unit();
        }
        a
    }
}

struct Ctx {
    eris: KuEris,
    pa: PaddedMos,
    pb: PaddedMos,
    khelper: KptsHelper,
    out: String,
}

fn build(section: &str) -> Option<Ctx> {
    let out = emit(section)?;
    let nkpts = scalar(&out, "nkpts") as usize;
    let nao = scalar(&out, "nao") as usize;
    let (nocca, noccb) = (
        scalar(&out, "nocca") as usize,
        scalar(&out, "noccb") as usize,
    );
    let (nmoa, nmob) = (scalar(&out, "nmoa") as usize, scalar(&out, "nmob") as usize);
    assert_ne!(
        nocca, noccb,
        "the fixture must be open shell or this test proves little"
    );

    // The mesh comes from the oracle stream, not from a literal here: the two
    // sides must agree on it exactly or the comparison measures the mesh.
    let mv = block(&out, "mesh");
    assert_eq!(mv.len(), 3, "mesh shape");
    let mesh = [mv[0] as usize, mv[1] as usize, mv[2] as usize];
    let cell = h3_openshell(mesh);
    let kv = block(&out, "kpts");
    assert_eq!(kv.len(), 3 * nkpts, "kpts shape");
    let kpts: Vec<[f64; 3]> = (0..nkpts)
        .map(|k| [kv[3 * k], kv[3 * k + 1], kv[3 * k + 2]])
        .collect();
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let khelper = KptsHelper::without_symm_map(&cell.a, &kpts);

    let pa = padded(&out, "a", nocca, nmoa, nkpts, nao);
    let pb = padded(&out, "b", noccb, nmob, nkpts, nao);
    let fock = (
        ZArr::from_ctensor(&[nkpts, nmoa, nmoa], cblock(&out, "focka")).expect("focka"),
        ZArr::from_ctensor(&[nkpts, nmob, nmob], cblock(&out, "fockb")).expect("fockb"),
    );
    let mo_energy = (pa.mo_energy.clone(), pb.mo_energy.clone());
    let eris = KuEris::from_parts(
        &df,
        &khelper.kconserv,
        (&pa, &pb),
        fock,
        mo_energy,
        scalar(&out, "madelung"),
        KErisOpts::default(),
    )
    .expect("_ChemistsERIs on upstream MOs");
    println!("fixture: mesh {mesh:?}, nkpts {nkpts}, nocc ({nocca},{noccb}), nmo ({nmoa},{nmob})");
    Some(Ctx {
        eris,
        pa,
        pb,
        khelper,
        out,
    })
}

/// **16-06 test 1 — the twenty-six ERI blocks, element-wise.**
///
/// This is the gate that pins the `[kp,kq,kr]` index convention, the four
/// `oppp` passes and the `conj().transpose(1,0,3,2)` at `[kq,kp,ks]` that
/// `voov`/`vovv` and their five spin siblings are stored under. A block put at
/// the wrong address here is `O(1)` wrong; the FFT transform floor is `~1e-8`.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kueris_blocks_match_upstream() {
    let Some(ctx) = build("kuccsd") else { return };
    let (worst, failures) = compare_blocks(&ctx, ERI_BLOCK);
    println!(
        "worst block: {} at {:e} (gate {ERI_BLOCK:e})",
        worst.1, worst.0
    );
    assert!(failures.is_empty(), "blocks above the gate: {failures:?}");
}

/// **16-06 test 1b — the measurement that turns test 1's gate into a number.**
///
/// Test 1 gates at `1e-9`. That is only defensible if the residual it measures
/// is the port's and not the mesh's, and the way to know is to move the mesh.
///
/// At upstream's pinned `[13,13,13]` the residuals run from `2.7e-9` on `oooo`
/// to `1.2e-5` on `vvvv` — MONOTONE in the number of virtual indices, which is
/// the signature of the plane-wave grid: this fixture's virtuals are the
/// antibonding combinations of an all-electron `0.5`-exponent `s` function in a
/// 6.74-Bohr cell, the most diffuse objects in the calculation. A misplaced
/// block or a wrong transpose is `O(1)` and does not care how many virtual
/// indices it has.
///
/// Refining to `[31,31,31]` flattens the whole table to `~5e-10`, and `vvvv`
/// becomes TIGHTER than `oooo`. That inversion is the proof: the coarse
/// residual was the grid, and the port reproduces upstream's transform to the
/// floor wherever the grid allows it.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn the_eri_residual_is_the_mesh_and_not_the_port() {
    let Some(coarse) = build("kuccsd_coarse") else {
        return;
    };
    let (worst_coarse, failures) = compare_blocks(&coarse, ERI_BLOCK_COARSE);
    assert!(
        failures.is_empty(),
        "even at the coarse mesh these are beyond the grid's reach: {failures:?}"
    );
    drop(coarse);

    let Some(fine) = build("kuccsd") else { return };
    let (worst_fine, failures) = compare_blocks(&fine, ERI_BLOCK);
    assert!(failures.is_empty(), "blocks above the gate: {failures:?}");

    let factor = worst_coarse.0 / worst_fine.0;
    println!(
        "worst block: [13,13,13] {} {:e}  ->  [31,31,31] {} {:e}   refinement x{factor:.0}",
        worst_coarse.1, worst_coarse.0, worst_fine.1, worst_fine.0
    );
    assert!(
        factor > MESH_REFINEMENT_FACTOR,
        "refining the mesh bought only x{factor:.1}, so the residual is NOT the mesh"
    );
    // The inversion, stated as an assertion rather than left in the printout:
    // at the coarse mesh the worst block carries four virtual indices; at the
    // fine mesh it does not.
    assert_eq!(worst_coarse.1, "vvvv", "the coarse-mesh worst block moved");
    assert_ne!(
        worst_fine.1, "vvvv",
        "vvvv is still the worst at the fine mesh, so this is not a grid effect"
    );
}

/// Element-wise-compare all twenty-six blocks; returns `((worst, name), failures)`.
fn compare_blocks(ctx: &Ctx, gate: f64) -> ((f64, &'static str), Vec<String>) {
    let mut worst = (0.0_f64, "");
    let mut failures: Vec<String> = Vec::new();
    for (b, name) in all_blocks() {
        let want = cblock(&ctx.out, name);
        let nk = ctx.eris.nkpts;
        let dims = b.dims(ctx.eris.nocc, ctx.eris.nvir);
        let per: usize = dims.iter().product();
        assert_eq!(
            want.re.len(),
            nk * nk * nk * per,
            "{name}: upstream block has an unexpected size"
        );
        let mut got = ZArr::zeros(&[nk, nk, nk, dims[0], dims[1], dims[2], dims[3]]);
        for k0 in 0..nk {
            for k1 in 0..nk {
                for k2 in 0..nk {
                    got.set_leading(&[k0, k1, k2], &ctx.eris.blk(b, k0, k1, k2).expect("blk"))
                        .expect("set");
                }
            }
        }
        let d = maxdiff(&got, &want, name);
        println!("  {name:6} max|Δ| {d:e}");
        if d > worst.0 {
            worst = (d, name);
        }
        if !(d < gate) {
            failures.push(format!("{name} {d:e}"));
        }
    }
    (worst, failures)
}

/// **16-06 test 2 — `init_amps`'s MP2 energy.**
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_init_amps_matches_upstream() {
    let Some(ctx) = build("kuccsd") else { return };
    let (emp2, _, _) =
        init_amps(&ctx.eris, (&ctx.pa, &ctx.pb), &ctx.khelper.kconserv).expect("init_amps");
    let want = scalar(&ctx.out, "emp2");
    let d = (emp2 - want).abs();
    println!("emp2 {emp2} vs upstream {want}  |Δ| {d:e}  (gate {EMP2:e})");
    assert!(d < EMP2, "emp2 differs by {d:e}");
}

/// **16-06 test 3 — `energy` and `update_amps` on a FIXED synthetic amplitude
/// quintuple.**
///
/// The point of the synthetic amplitudes is that they isolate the equations
/// from the iteration: a sign error in one `Ht2ab` term that the converged
/// energy happens to be insensitive to still shows up here, at `O(1)`.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_update_amps_matches_upstream() {
    let Some(ctx) = build("kuccsd") else { return };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );

    let e = energy(&t1, &t2, &ctx.eris).expect("energy");
    let want = scalar(&ctx.out, "energy_synth");
    let d = (e - want).abs();
    println!("energy(synthetic) {e} vs upstream {want}  |Δ| {d:e}");
    assert!(d < 1e-9, "the synthetic energy differs by {d:e}");

    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let opts = KrccsdOpts::default();

    let mut failures: Vec<String> = Vec::new();
    let run = |tag: &str, t1: &UT1, t2: &UT2, failures: &mut Vec<String>| {
        let (t1n, t2n) = update_amps(
            &pool,
            t1,
            t2,
            &ctx.eris,
            (&ctx.pa, &ctx.pb),
            &ctx.khelper.kconserv,
            &opts,
        )
        .expect("update_amps");
        for (got, name) in [
            (&t1n.0, "st1anew"),
            (&t1n.1, "st1bnew"),
            (&t2n.0, "st2aanew"),
            (&t2n.1, "st2abnew"),
            (&t2n.2, "st2bbnew"),
        ] {
            let key = format!("{name}{tag}");
            let d = maxdiff(got, &cblock(&ctx.out, &key), &key);
            println!("  {key:14} max|Δ| {d:e}");
            if !(d < AMPS_BLOCK) {
                failures.push(format!("{key} {d:e}"));
            }
        }
    };

    // The three runs bisect a doubles mismatch: `_t1z` keeps only what is
    // independent of the singles, `_t2z` only what is independent of the
    // doubles, and the full run adds the cross terms.
    let z1: UT1 = (ZArr::zeros(t1.0.shape()), ZArr::zeros(t1.1.shape()));
    let z2: UT2 = (
        ZArr::zeros(t2.0.shape()),
        ZArr::zeros(t2.1.shape()),
        ZArr::zeros(t2.2.shape()),
    );
    run("_t1z", &z1, &t2, &mut failures);
    run("_t2z", &t1, &z2, &mut failures);
    run("", &t1, &t2, &mut failures);
    assert!(
        failures.is_empty(),
        "amplitudes above the gate: {failures:?}"
    );
}

/// **16-06 test 4 — the converged `e_corr`.**
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_e_corr_matches_upstream() {
    let Some(ctx) = build("kuccsd") else { return };
    assert_eq!(
        scalar(&ctx.out, "converged_cc"),
        1.0,
        "upstream did not converge"
    );
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let opts = KrccsdOpts::default();
    let res = kernel(
        &pool,
        &ctx.eris,
        (&ctx.pa, &ctx.pb),
        &ctx.khelper.kconserv,
        &opts,
    )
    .expect("KUCCSD kernel");
    let want = scalar(&ctx.out, "e_corr");
    let d = (res.e_corr - want).abs();
    println!(
        "e_corr {} ({} cycles, converged {}) vs upstream {want}  |Δ| {d:e}  (gate {E_CORR:e})",
        res.e_corr, res.cycles, res.converged
    );
    assert!(
        res.converged,
        "KUCCSD did not converge in {} cycles",
        res.cycles
    );
    assert!(d < E_CORR, "e_corr differs by {d:e}");
}

/// **16-06 test 5 — `_make_df_eris` refuses `cell.dimension == 2`, and the
/// refusal names the upstream line.**
#[test]
fn direct_df_route_refuses_a_slab_and_says_where() {
    assert!(KuEris::check_dimension_for_direct_df(3).is_ok());
    assert!(KuEris::check_dimension_for_direct_df(0).is_ok());
    let e = KuEris::check_dimension_for_direct_df(2).expect_err("dimension 2 must refuse");
    let msg = e.to_string();
    assert!(msg.contains("kccsd_uhf.py:1022"), "{msg}");
    assert!(msg.contains("NotImplementedError"), "{msg}");
}

/// **16-06 test 6 — the block table is the twenty-six upstream builds, and
/// `OOoo` / `VVvv` are absent BY NAME.**
///
/// `kccsd_uhf.py:830` and `:1013` set those two to `None` and nothing reads
/// them; a port that built them anyway would allocate `nkpts³ · noccb²nocca²`
/// and `nkpts³ · nvirb²nvira²` for nothing. This pins the omission so it stays
/// a decision rather than an accident.
#[test]
fn the_block_table_is_upstreams() {
    let all = UBlk::all();
    assert_eq!(all.len(), 26, "4x6 - OOoo + 3 quads");
    let names: Vec<&str> = all.iter().map(|b| b.name()).collect();
    assert!(!names.contains(&"OOoo"), "OOoo is None upstream (:830)");
    assert!(!names.contains(&"VVvv"), "VVvv is None upstream (:1013)");
    for n in ["oooo", "OVOV", "ooVV", "VOvv", "vvvv", "VVVV", "vvVV"] {
        assert!(names.contains(&n), "missing {n}");
    }
    // The two transposed kinds are stored with their first two axes swapped,
    // which is what makes `voOV` `[nvira, nocca, noccb, nvirb]` and not
    // `[nocca, nvira, nvirb, noccb]`.
    let d = UBlk::Pair(UPass::AaBb, UKind::Voov).dims((2, 1), (4, 5));
    assert_eq!(d, [4, 2, 1, 5], "voOV dims");
    let d = UBlk::Pair(UPass::BbAa, UKind::Vovv).dims((2, 1), (4, 5));
    assert_eq!(d, [5, 1, 4, 4], "VOvv dims");
}

/// **16-06 test 3b — every ground-state intermediate, one by one.**
///
/// Test 3 compares `update_amps`'s OUTPUT. When that fails it says only "the
/// doubles are wrong", which on a five-array, three-spin-channel update is not
/// a diagnosis. This runs the same fixed synthetic amplitudes through every
/// intermediate `kintermediates_uhf` builds and reports each separately, so a
/// failure names the function.
///
/// It is not redundant once test 3 passes: an error in one intermediate that
/// cancels in the assembled `t2new` is exactly the kind of thing that survives
/// an end-to-end gate and then breaks EOM-KUCCSD, which reuses these.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_intermediates_match_upstream() {
    let Some(ctx) = build("kuccsd_imds") else {
        return;
    };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let kc = &ctx.khelper.kconserv;
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let budget = 4_000_000_000_usize;

    let mut failures: Vec<String> = Vec::new();
    let check = |name: &str, got: &ZArr, failures: &mut Vec<String>| {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:9} max|Δ| {d:e}");
        if !(d < AMPS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    };

    let tau = make_tau(&t2, &t1, &t1, 1.0).expect("make_tau");
    check("tau_aa", &tau.0, &mut failures);
    check("tau_ab", &tau.1, &mut failures);
    check("tau_bb", &tau.2, &mut failures);
    let tauh = make_tau(&t2, &t1, &t1, 0.5).expect("make_tau fac=0.5");
    check("tauh_aa", &tauh.0, &mut failures);
    check("tauh_ab", &tauh.1, &mut failures);
    check("tauh_bb", &tauh.2, &mut failures);
    let tau2 = make_tau2(&t2, &t1, &t1, 2.0).expect("make_tau2");
    check("tau2_aa", &tau2.0, &mut failures);
    check("tau2_ab", &tau2.1, &mut failures);
    check("tau2_bb", &tau2.2, &mut failures);

    let (fvv_a, fvv_b) = cc_fvv(&t1, &t2, &ctx.eris, kc).expect("cc_Fvv");
    check("Fvv_a", &fvv_a, &mut failures);
    check("Fvv_b", &fvv_b, &mut failures);
    let (foo_a, foo_b) = cc_foo(&t1, &t2, &ctx.eris, kc).expect("cc_Foo");
    check("Foo_a", &foo_a, &mut failures);
    check("Foo_b", &foo_b, &mut failures);
    let (fov_a, fov_b) = cc_fov(&t1, &ctx.eris).expect("cc_Fov");
    check("Fov_a", &fov_a, &mut failures);
    check("Fov_b", &fov_b, &mut failures);

    let w = cc_woooo(&pool, budget, &t1, &t2, &ctx.eris, kc).expect("cc_Woooo");
    for (blocks, name) in [(&w.0, "Woooo"), (&w.1, "WooOO"), (&w.2, "WOOOO")] {
        check(name, &gather(blocks, nk), &mut failures);
    }
    let wv = cc_wvvvv_half(&pool, budget, &t1, &ctx.eris, kc).expect("cc_Wvvvv_half");
    for (blocks, name) in [(&wv.0, "Wvvvv"), (&wv.1, "WvvVV"), (&wv.2, "WVVVV")] {
        check(name, &gather(blocks, nk), &mut failures);
    }
    let wo = cc_wovvo(&pool, budget, &t1, &t2, &ctx.eris, kc).expect("cc_Wovvo");
    for (blocks, name) in [
        (&wo.aa, "Wovvo"),
        (&wo.ab, "WovVO"),
        (&wo.ba, "WOVvo"),
        (&wo.bb, "WOVVO"),
        (&wo.abba, "WoVVo"),
        (&wo.baab, "WOvvO"),
    ] {
        check(name, &gather(blocks, nk), &mut failures);
    }

    // `add_vvvv_` in isolation, on zeroed `Ht2`, exactly as the emitter runs it.
    let mut ht2: UT2 = (
        ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]),
        ZArr::zeros(&[nk, nk, nk, oa, ob, va, vb]),
        ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    add_vvvv_for_test(&mut ht2, &t1, &t2, &wv, nk, kc).expect("add_vvvv_");
    check("vvvv_aa", &ht2.0, &mut failures);
    check("vvvv_ab", &ht2.1, &mut failures);
    check("vvvv_bb", &ht2.2, &mut failures);

    assert!(
        failures.is_empty(),
        "intermediates above the gate: {failures:?}"
    );
}

/// `KBlocks` -> one `[nkpts, nkpts, nkpts, ...]` array, in upstream's layout.
fn gather(blocks: &pyscf_pbc_cc::KBlocks, nk: usize) -> ZArr {
    let bs = blocks.block_shape().to_vec();
    let mut shape = vec![nk, nk, nk];
    shape.extend_from_slice(&bs);
    let mut out = ZArr::zeros(&shape);
    for k0 in 0..nk {
        for k1 in 0..nk {
            for k2 in 0..nk {
                out.set_leading(&[k0, k1, k2], &blocks.get([k0, k1, k2]).expect("block"))
                    .expect("set");
            }
        }
    }
    out
}

/// **16-06 test 3c — `kccsd_uhf.py:230-386` standalone.**
///
/// The `Wovvo` block is the largest single piece of the doubles equation:
/// fifteen k-loops, six intermediates, four mirrored write addresses and two
/// antisymmetrisations. The oracle side runs upstream's own lines VERBATIM on
/// zeroed `Ht2` (see `section_kuccsd_wovvo`), so this compares one block
/// against one block instead of one 400-line function against another.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_wovvo_block_matches_upstream() {
    let Some(ctx) = build("kuccsd_wovvo") else {
        return;
    };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let kc = &ctx.khelper.kconserv;
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let w = cc_wovvo(&pool, 4_000_000_000, &t1, &t2, &ctx.eris, kc).expect("cc_Wovvo");
    let mut ht2: UT2 = (
        ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]),
        ZArr::zeros(&[nk, nk, nk, oa, ob, va, vb]),
        ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    wovvo_terms_for_test(&mut ht2, &t1, &t2, &ctx.eris, &w, kc).expect("wovvo_terms");

    let mut failures: Vec<String> = Vec::new();
    for (got, name) in [
        (&ht2.0, "wovvo_aa"),
        (&ht2.1, "wovvo_ab"),
        (&ht2.2, "wovvo_bb"),
    ] {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:9} max|Δ| {d:e}");
        if !(d < AMPS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "the Wovvo block is above the gate: {failures:?}"
    );
}

/// **16-06 test 3d — `kccsd_uhf.py:205-226` standalone.**
///
/// The bare `ovov` driver plus the `Woooo` stage, against upstream's own lines
/// run verbatim on zeroed `Ht2`. With 3c this covers the whole doubles
/// equation except the `Fvv`/`Foo` driving loop, which the singles already
/// exercise.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_woooo_block_matches_upstream() {
    let Some(ctx) = build("kuccsd_woooo") else {
        return;
    };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let kc = &ctx.khelper.kconserv;
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let mut ht2: UT2 = (
        ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]),
        ZArr::zeros(&[nk, nk, nk, oa, ob, va, vb]),
        ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    woooo_terms_for_test(&mut ht2, &t1, &t2, &ctx.eris, &pool, 4_000_000_000, kc)
        .expect("woooo_terms");

    let mut failures: Vec<String> = Vec::new();
    for (got, name) in [
        (&ht2.0, "woooo_aa"),
        (&ht2.1, "woooo_ab"),
        (&ht2.2, "woooo_bb"),
    ] {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:9} max|Δ| {d:e}");
        if !(d < AMPS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "the Woooo block is above the gate: {failures:?}"
    );
}

/// **16-06 test 3e — `kccsd_uhf.py:65-202` standalone.**
///
/// The intermediates, the singles equation and the `Fvv`/`Foo` doubles driving
/// loop. With 3c and 3d this covers every one of `update_amps`'s five stages
/// against upstream's own lines, so a mismatch in the assembled `t2new` is
/// attributable to a named stage or to the denominators, and to nothing else.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_fock_block_matches_upstream() {
    let Some(ctx) = build("kuccsd_fock") else {
        return;
    };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let mut ht1: UT1 = (ZArr::zeros(&[nk, oa, va]), ZArr::zeros(&[nk, ob, vb]));
    let mut ht2: UT2 = (
        ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]),
        ZArr::zeros(&[nk, nk, nk, oa, ob, va, vb]),
        ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    fock_terms_for_test(
        &mut ht1,
        &mut ht2,
        &t1,
        &t2,
        &ctx.eris,
        &ctx.khelper.kconserv,
        &KrccsdOpts::default(),
    )
    .expect("fock_terms");

    let mut failures: Vec<String> = Vec::new();
    for (got, name) in [
        (&ht2.0, "fock_aa"),
        (&ht2.1, "fock_ab"),
        (&ht2.2, "fock_bb"),
        (&ht1.0, "fock_t1a"),
        (&ht1.1, "fock_t1b"),
    ] {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:9} max|Δ| {d:e}");
        if !(d < AMPS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "the Fock block is above the gate: {failures:?}"
    );
}

/// **16-12 test 1 — the `KUCCSD` one-particle density matrix.**
///
/// Both on the CONVERGED amplitudes and on the fixed synthetic quintuple. The
/// synthetic pair is the one that exercises the equations: at convergence `t1`
/// is small, so a wrong `t1`-linear term barely moves `dm1` and an end-to-end
/// comparison would not see it.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_rdm1_matches_upstream() {
    let Some(ctx) = build("kuccsd") else { return };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let kc = &ctx.khelper.kconserv;
    let mut failures: Vec<String> = Vec::new();

    // --- the converged amplitudes
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let res = kernel(
        &pool,
        &ctx.eris,
        (&ctx.pa, &ctx.pb),
        kc,
        &KrccsdOpts::default(),
    )
    .expect("KUCCSD kernel");
    assert!(res.converged);
    // The amplitude spread FIRST, so the RDM residual below is explained by a
    // measured number and not by an assertion about convergence.
    let dt1 = maxdiff(&res.t1.0, &cblock(&ctx.out, "t1a"), "t1a").max(maxdiff(
        &res.t1.1,
        &cblock(&ctx.out, "t1b"),
        "t1b",
    ));
    println!("  converged max|Δt1| {dt1:e}  (this is what the converged RDM inherits)");

    let (a, b) = make_rdm1(&res.t1, &res.t2, None, None, kc, false).expect("make_rdm1");
    for (got, name) in [(&a, "rdm1a"), (&b, "rdm1b")] {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:7} max|Δ| {d:e}  (gate {RDM_CONVERGED:e})");
        if !(d < RDM_CONVERGED) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // --- the synthetic amplitudes
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let (a, b) = make_rdm1(&t1, &t2, None, None, kc, false).expect("make_rdm1");
    for (got, name) in [(&a, "srdm1a"), (&b, "srdm1b")] {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:7} max|Δ| {d:e}  (gate {RDM_SYNTHETIC:e})");
        if !(d < RDM_SYNTHETIC) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    // Hermiticity is structural (`:123` writes the `vo` block as the `ov`
    // block's conjugate transpose), so it holds for ANY amplitudes and is
    // asserted rather than compared.
    for (dm, name) in [(&a, "srdm1a"), (&b, "srdm1b")] {
        let n = dm.shape()[1];
        let mut worst = 0.0_f64;
        for k in 0..nk {
            let blk = dm.slice_leading(&[k]).expect("block");
            for p in 0..n {
                for q in 0..n {
                    let (re, im) = blk.at(&[p, q]).expect("elem");
                    let (re2, im2) = blk.at(&[q, p]).expect("elem");
                    worst = worst.max((re - re2).abs()).max((im + im2).abs());
                }
            }
        }
        println!("  {name:7} non-Hermiticity {worst:e}");
        assert!(worst < 1e-15, "{name} is not Hermitian: {worst:e}");
    }
    assert!(
        failures.is_empty(),
        "the 1-RDM is above the gate: {failures:?}"
    );
}

/// **16-12 test 2 — the frozen-core branch refuses, and names the line.**
#[test]
fn rdm1_frozen_core_refuses_and_says_where() {
    let d1 = Gamma1 {
        doo: (ZArr::zeros(&[1, 1, 1]), ZArr::zeros(&[1, 1, 1])),
        dov: (ZArr::zeros(&[1, 1, 1]), ZArr::zeros(&[1, 1, 1])),
        dvo: (ZArr::zeros(&[1, 1, 1]), ZArr::zeros(&[1, 1, 1])),
        dvv: (ZArr::zeros(&[1, 1, 1]), ZArr::zeros(&[1, 1, 1])),
    };
    assert!(make_rdm1_from_gamma1(&d1, false).is_ok());
    let e = make_rdm1_from_gamma1(&d1, true).expect_err("frozen core must refuse");
    let msg = e.to_string();
    assert!(msg.contains("kuccsd_rdm.py:137"), "{msg}");
    assert!(msg.contains("NotImplementedError"), "{msg}");
}

/// **16-11 Task 1 — the UHF EOM intermediates.**
///
/// `eom_kccsd_uhf._IMDS` builds `Foo`/`Fvv`/`Fov`, `Wovvo`, `Woovv` (shared),
/// `Woooo`/`Wooov`/`Woovo` (IP) and `Wvvov`/`Wvvvv`/`Wvvvo` (EA). Every one
/// returns THREE or FOUR spin blocks that are NOT related by any symmetry —
/// `WooVO` and `WOOvo` have different shapes, not transposes of one another —
/// so each block is compared on its own.
///
/// `W1oovv` and `W2oovv` are gated separately from their sum for the reason
/// 16-10 gives about `W1ovvo`/`W2ovvo`: upstream builds the halves.
#[test]
#[ignore = "opt-in: needs PYSCF_ORACLE_VENV"]
fn kuccsd_eom_intermediates_match_upstream() {
    let Some(ctx) = build("kuccsd_eom") else {
        return;
    };
    let nk = ctx.eris.nkpts;
    let (oa, ob) = ctx.eris.nocc;
    let (va, vb) = ctx.eris.nvir;
    let mut r = SplitMix64(20260906);
    let t1: UT1 = (r.draw(&[nk, oa, va]), r.draw(&[nk, ob, vb]));
    let t2: UT2 = (
        r.draw(&[nk, nk, nk, oa, oa, va, va]),
        r.draw(&[nk, nk, nk, oa, ob, va, vb]),
        r.draw(&[nk, nk, nk, ob, ob, vb, vb]),
    );
    let kc = &ctx.khelper.kconserv;
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let budget = 4_000_000_000_usize;

    let mut failures: Vec<String> = Vec::new();
    let check = |name: &str, got: &ZArr, failures: &mut Vec<String>| {
        let d = maxdiff(got, &cblock(&ctx.out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < AMPS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    };

    use pyscf_pbc_cc::kintermediates_uhf as uimd;
    let (a, b) = uimd::foo(&t1, &t2, &ctx.eris, kc).expect("Foo");
    check("u_Foo", &a, &mut failures);
    check("u_FOO", &b, &mut failures);
    let (a, b) = uimd::fvv(&t1, &t2, &ctx.eris, kc).expect("Fvv");
    check("u_Fvv", &a, &mut failures);
    check("u_FVV", &b, &mut failures);
    let (a, b) = uimd::fov(&t1, &ctx.eris).expect("Fov");
    check("u_Fov", &a, &mut failures);
    check("u_FOV", &b, &mut failures);

    let q = uimd::wooov(&t1, &ctx.eris).expect("Wooov");
    for (got, name) in [
        (&q.0, "u_Wooov"),
        (&q.1, "u_WooOV"),
        (&q.2, "u_WOOov"),
        (&q.3, "u_WOOOV"),
    ] {
        check(name, got, &mut failures);
    }
    let q = uimd::wovvo(&pool, budget, &t1, &t2, &ctx.eris, kc).expect("Wovvo");
    for (got, name) in [
        (&q.0, "u_Wovvo"),
        (&q.1, "u_WovVO"),
        (&q.2, "u_WOVvo"),
        (&q.3, "u_WOVVO"),
    ] {
        check(name, got, &mut failures);
    }
    let q = uimd::w1oovv(&t2, &ctx.eris, kc).expect("W1oovv");
    for (got, name) in [
        (&q.0, "u_W1oovv"),
        (&q.1, "u_W1ooVV"),
        (&q.2, "u_W1OOvv"),
        (&q.3, "u_W1OOVV"),
    ] {
        check(name, got, &mut failures);
    }
    let q = uimd::w2oovv(&t1, &ctx.eris, kc).expect("W2oovv");
    for (got, name) in [
        (&q.0, "u_W2oovv"),
        (&q.1, "u_W2ooVV"),
        (&q.2, "u_W2OOvv"),
        (&q.3, "u_W2OOVV"),
    ] {
        check(name, got, &mut failures);
    }
    let q = uimd::woovv(&t1, &t2, &ctx.eris, kc).expect("Woovv");
    for (got, name) in [
        (&q.0, "u_Woovv"),
        (&q.1, "u_WooVV"),
        (&q.2, "u_WOOvv"),
        (&q.3, "u_WOOVV"),
    ] {
        check(name, got, &mut failures);
    }
    let t = uimd::eom_woooo(&t1, &t2, &ctx.eris, kc).expect("Woooo");
    for (got, name) in [(&t.0, "u_Woooo"), (&t.1, "u_WooOO"), (&t.2, "u_WOOOO")] {
        check(name, got, &mut failures);
    }
    let t = uimd::eom_wvvvv(&pool, budget, &t1, &t2, &ctx.eris, kc).expect("Wvvvv");
    for (got, name) in [(&t.0, "u_Wvvvv"), (&t.1, "u_WvvVV"), (&t.2, "u_WVVVV")] {
        check(name, got, &mut failures);
    }

    let q = uimd::wvvov(&t1, &ctx.eris, kc).expect("Wvvov");
    for (got, name) in [
        (&q.0, "u_Wvvov"),
        (&q.1, "u_WvvOV"),
        (&q.2, "u_WVVov"),
        (&q.3, "u_WVVOV"),
    ] {
        check(name, got, &mut failures);
    }

    // `get_Wvvvv` at one k-triple — a DIFFERENT function from `Wvvvv`, and the
    // one `eaccsd_matvec` calls per triple (`eom_kccsd_uhf.py:1123`).
    let kb_ = 1 % nk;
    let t = uimd::get_wvvvv(&t1, &t2, &ctx.eris, kc, 0, kb_, kb_).expect("get_Wvvvv");
    for (got, name) in [(&t.0, "u_gvvvv"), (&t.1, "u_gvvVV"), (&t.2, "u_gVVVV")] {
        check(name, got, &mut failures);
    }

    let q = uimd::woovo(&t1, &t2, &ctx.eris, kc).expect("Woovo");
    for (got, name) in [
        (&q.0, "u_Woovo"),
        (&q.1, "u_WooVO"),
        (&q.2, "u_WOOvo"),
        (&q.3, "u_WOOVO"),
    ] {
        check(name, got, &mut failures);
    }

    let q = uimd::wvvvo(&t1, &t2, &ctx.eris, kc).expect("Wvvvo");
    for (got, name) in [
        (&q.0, "u_Wvvvo"),
        (&q.1, "u_WvvVO"),
        (&q.2, "u_WVVvo"),
        (&q.3, "u_WVVVO"),
    ] {
        check(name, got, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "UHF EOM intermediates above the gate: {failures:?}"
    );
}
