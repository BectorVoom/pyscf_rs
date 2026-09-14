//! Phase 18-09 kernel gates — multigrid-v2 gradient entry points.
//!
//! Ported from `pyscf/pbc/dft/multigrid/multigrid_pair.py:748-921` and
//! `pp.py:135-201`. Acceptance here is KERNEL-LEVEL (18-09-PLAN Task 4):
//! 17-12 shipped v2 with 5/5 green kernel gates and an UNVERIFIED host side
//! (every SCF-bearing host test SIGKILLed at exit 137), so no test here
//! runs an SCF. Every gate holds the density matrix FIXED and
//! finite-differences the corresponding energy contraction — no convergence
//! path, no second-solution noise.
//!
//! # The (a)/(b) split (read before "simplifying" a gate)
//!
//! A nuclear derivative of `Tr(dm·V)` has TWO pieces: (a) the AO product
//! moves with the nuclei (`ip1` matrices — the eval_mat-derivative piece,
//! contracted bra-sliced times 2), and (b) the POTENTIAL moves with the
//! nuclei (the G-space `(natm, 3)` contraction). Upstream's `grad/rhf.py`
//! adds exactly these two (`de = ni.get_nuc_nuc_grad(...)` then
//! `de += _contract_vhf_dm(h1ao -= ni.get_nuc_ip1(...), ...)`), and the same
//! split holds for the part-1 PP (`vpploc_part1_nuc_grad` + contracted
//! `get_vpploc_part1_ip1`). For `veff` at fixed dm there is NO (b) piece
//! (`v = δE/δρ` is the functional derivative — the chain rule closes), so
//! `get_veff_ip1` gates directly against FD of `nr_rks`.
//!
//! Each piece gates against its own EXACT oracle at machine precision:
//! (a) against FD of the frozen-potential grid energy, (b) against FD of
//! the frozen-density field energy. A naive "(b) vs FD of the full energy"
//! comparison measures the missing (a), not an error — both isolation
//! oracles are built inline below so the association is auditable.
//!
//! Gates (all gamma point, closed shell):
//!
//! * the three refusals (non-gamma k-list = the `KPoints` refusal,
//!   meta-GGA = the `deriv > 1` refusal, `deriv = 2`, bad `atm_id`);
//! * (b)_nuc vs frozen-density FD — 1e-9 (he + diamond);
//! * (b)_nuc batched vs a literal per-atom transcription of upstream's
//!   `:916-919` loop — BIT-identical;
//! * (a)_nuc vs frozen-potential FD — 1e-9 (diamond);
//! * `get_veff_ip1` (bra-sliced `2·Tr` assembly) vs FD of `nr_rks`'s
//!   `exc + ecoul`, LDA, GGA and HF(Coulomb-only) — 5e-9 at
//!   `precision = 1e-10` (at the default 1e-8 the pair-screening floor sits
//!   at ~1e-8 — measured, and the reason for the tighter precision here;
//!   the remaining ~1e-9 is the optimal-FD floor, documented at `H`);
//! * (b)_core vs frozen-density FD and (a)_core vs frozen-potential FD —
//!   1e-9 (diamond); cached-`rhoG` vs recomputed — bit-identical;
//!   explicit-`atm_id` rows agree with the full call;
//! * the collocation adjoint identity `Tr(dm·V) = Σ ρ·v·w` at 1e-12;
//! * repeat-call bit-identity (the RAYON 1-vs-8 proof for the shared
//!   reduction lives with the primitive itself,
//!   `pyscf-kernels/tests/atom_grid_contract.rs`).
//!
//! Every gate runs on the smallest §9.2 cell first (`he_fcc`, `nao = 1`)
//! on a coarse matched mesh and escalates to `diamond` only for the gates
//! that need non-trivial forces (a single atom's gradient is translation
//! noise, gated ~0-vs-0). Diamond FD gates work around a cell with atom 0
//! pushed `WORK_SHIFT` off equilibrium (at equilibrium the forces vanish by
//! symmetry and the gate proves nothing). Displaced cells REBUILD with
//! `mesh` pinned (`displaced.mesh = cell.mesh`), per 18-CONTEXT trap 5.
//! Geometry is in **Bohr** throughout.
//!
//! Log lines name which cells completed: look for `18-09 gate ... ok`.

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::multigrid::pair::MultiGridNumInt2;
use pyscf_pbc_gto::test_systems::he_fcc;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

mod common;

/// Coarse matched mesh for the kernel gates — speed, not physics: both
/// sides of every comparison below use the same mesh, so mesh error
/// cancels. (Upstream's Gate-E meshes are 35³–81³; a full density build
/// there is seconds per call and these gates make dozens.)
const HE_MESH: [usize; 3] = [12, 12, 12];
const DIA_MESH: [usize; 3] = [10, 10, 10];
/// Central-difference half-step (Bohr). Chosen BELOW the truncation knee:
/// at `1e-4` the residual scales ∝ h² (measured 16× drop quartering h —
/// pure `h²E'''/6` truncation), at `1e-5` the remainder is the h-independent
/// optimal-FD floor (`E`-rounding/`h` vs truncation, ~1e-9 for `veff`).
/// Upstream's `disp` is the FULL step; `verify_fd`'s is the half step —
/// here `h` is the half step and the quotient is `(E⁺−E⁻)/2h`
/// (18-CONTEXT §2.1).
const H: f64 = 1e-5;
/// Off-equilibrium working displacement (Bohr): diamond at equilibrium has
/// vanishing forces by symmetry, so the FD gates work around a cell with
/// atom 0 pushed off-site. He (one atom) is translation-invariant and only
/// ever gates ~0-vs-0 plus code paths.
const WORK_SHIFT: f64 = 0.03;
/// Tight integral precision for the `veff` gates: at the default 1e-8 the
/// pair-screening floor sits at ~1e-8 in the FD comparison (measured —
/// the screening decision flips between the two FD sides); at 1e-10 the
/// comparison is screening-free down to 1e-11.
const TIGHT_PRECISION: f64 = 1e-10;

fn he_cell() -> Cell {
    let mut cell = he_fcc();
    cell.mesh = HE_MESH;
    cell
}

fn dia_cell() -> Cell {
    let mut cell = common::diamond();
    cell.mesh = DIA_MESH;
    cell
}

fn dia_tight_cell() -> Cell {
    let mut cell = pyscf_pbc_gto::test_systems::diamond_precision(TIGHT_PRECISION);
    cell.mesh = DIA_MESH;
    cell
}

/// Fixed density: uniform occupancy, right electron count, no SCF.
/// `dm = (nelec/nao)·I`, row-major.
fn fixed_dm(cell: &Cell) -> Vec<f64> {
    let nao = cell.mol.nao_nr;
    let nelec = cell.tot_electrons(1) as f64;
    assert!(nao > 0);
    let mut dm = vec![0.0f64; nao * nao];
    for i in 0..nao {
        dm[i * nao + i] = nelec / nao as f64;
    }
    dm
}

/// Rebuild `cell` with atom `ia` shifted by `h` along `x`, everything else
/// (lattice, pseudo, precision, ke_cutoff, rcut, dimension) carried over
/// and `mesh` PINNED (18-CONTEXT trap 5: an FD whose two sides land on
/// different meshes measures the mesh, not the gradient).
///
/// `basis` is the `BasisInput::Name` the fixture was built with: the built
/// `mol.basis` is normalised (`name("gthszv")`) and does NOT round-trip
/// through the alias lookup, so the caller — which knows its fixture —
/// passes it explicitly. A rebuild (rather than mutating `_atom` in place)
/// is REQUIRED: `cell_fingerprint` hashes `mol._env`, so an in-place edit
/// would silently reuse stale pair tables.
fn displaced(cell: &Cell, ia: usize, x: usize, h: f64, basis: &str) -> Cell {
    let mut atoms: Vec<(String, [f64; 3])> = cell
        .mol
        ._atom
        .iter()
        .map(|(s, c)| (s.clone(), *c))
        .collect();
    atoms[ia].1[x] += h;
    let rebuilt = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name(basis.into()),
            unit: pyscf_core::Unit::Bohr,
            spin: cell.mol.spin,
            charge: cell.mol.charge,
            ..Default::default()
        },
        a: ALattice::Matrix(cell.a),
        mesh: Some(cell.mesh),
        ke_cutoff: cell.ke_cutoff,
        rcut: Some(cell.rcut),
        precision: cell.precision,
        dimension: cell.dimension,
        pseudo: cell.pseudo_name.clone(),
        ..Default::default()
    })
    .expect("displaced cell must build");
    assert_eq!(rebuilt.mesh, cell.mesh, "mesh must stay pinned across FD");
    rebuilt
}

/// `2·Σ_{mu in atom ia, nu} dm[nu,mu]·ip1[x,mu,nu]` — the `_contract_vhf_dm`
/// assembly for a bra-derivative `(3,nao,nao)` matrix (bra-sliced, times 2;
/// the ket side is accounted by symmetry — the frozen-potential FD gates
/// pin this).
fn contract_bra(ip1: &[f64], dm: &[f64], nao: usize, p0: usize, p1: usize, x: usize) -> f64 {
    let mut acc = 0.0f64;
    for mu in p0..p1 {
        for nu in 0..nao {
            acc += dm[nu * nao + mu] * ip1[(x * nao + mu) * nao + nu];
        }
    }
    2.0 * acc
}

fn aoslices(cell: &Cell) -> Vec<(usize, usize, usize, usize)> {
    pyscf_gto::aoslice_by_atom(&cell.mol).expect("aoslice")
}

fn ni() -> MultiGridNumInt2 {
    MultiGridNumInt2::new()
}

fn ngrids_of(cell: &Cell) -> usize {
    cell.mesh[0] * cell.mesh[1] * cell.mesh[2]
}

// ---------------------------------------------------------------------------
// The three refusals (18-09-PLAN Task 1 — each a named error).
// ---------------------------------------------------------------------------

#[test]
fn refusals_are_named() {
    let cell = he_cell();
    let ni = ni();
    let dm = fixed_dm(&cell);
    let off_gamma = [[0.1, 0.0, 0.0]];
    let gamma: [[f64; 3]; 0] = [];

    // Task 1 refusal 1 + §1.1 refusal: KPoints (any non-gamma k-list —
    // Rust has no KPoints type per 17-09's deferral, so the refusal lands
    // on the k-list itself, at the same call sites).
    for r in [
        ni.get_veff_ip1(&cell, "lda,vwn", &dm, &off_gamma)
            .map(|_| ()),
        ni.get_nuc_ip1(&cell, &off_gamma).map(|_| ()),
        ni.get_nuc_nuc_grad(&cell, &dm, &off_gamma, None)
            .map(|_| ()),
        ni.get_vpploc_part1_ip1(&cell, &off_gamma).map(|_| ()),
        ni.vpploc_part1_nuc_grad(&cell, &dm, &off_gamma, None, None)
            .map(|_| ()),
    ] {
        let e = r.expect_err("non-gamma k-list must be refused");
        assert!(
            format!("{e}").contains("exactly one gamma point"),
            "unexpected refusal: {e}"
        );
    }

    // Task 1 refusal 2 (multigrid_pair.py:766): anything above GGA raises.
    let e = ni
        .get_veff_ip1(&cell, "SCAN", &dm, &gamma)
        .expect_err("meta-GGA must raise");
    assert!(
        format!("{e}").contains("meta-GGA"),
        "unexpected refusal: {e}"
    );

    // The deriv>1 refusal, directly addressable through the deriv-taking
    // entry points.
    for (deriv, r) in [
        (2u32, ni.get_nuc_with_deriv(&cell, &gamma, 2).map(|_| ())),
        (3u32, ni.get_nuc_with_deriv(&cell, &gamma, 3).map(|_| ())),
    ] {
        let e = r.expect_err("deriv > 1 must raise");
        assert!(
            format!("{e}").contains(&format!("derivative order {deriv}")),
            "unexpected refusal: {e}"
        );
    }
    let e = ni
        .rho_g_with_deriv(&cell, &dm, 2)
        .expect_err("deriv > 1 must raise");
    assert!(format!("{e}").contains("derivative order 2"));

    // Out-of-range atm_id is refused, not clipped.
    let e = ni
        .vpploc_part1_nuc_grad(&cell, &dm, &gamma, None, Some(&[7]))
        .expect_err("bad atm_id must raise");
    assert!(format!("{e}").contains("out of range"), "odd refusal: {e}");

    println!("18-09 gate refusals ok (he_fcc)");
}

// ---------------------------------------------------------------------------
// Shared exact oracles: the frozen-density field energy (b) and the
// frozen-potential grid energy (a), for the nuclear and part-1 potentials.
// ---------------------------------------------------------------------------

/// Point-charge nuclear G-space field `rhoG_nuc·coulG` (bare Fourier
/// components — the same spelling `nuclear_vg` uses).
fn nuclear_vg(cell: &Cell) -> CTensor {
    let mesh = cell.mesh;
    let ngrids = ngrids_of(cell);
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh)).expect("gv");
    let coulg = pyscf_pbc_gto::get_coulg_at_gv(cell, mesh, &gv).expect("coulg");
    let coords = cell.mol.atom_coords();
    let charges = cell.atom_charges();
    let mut re = vec![0.0f64; ngrids];
    let mut im = vec![0.0f64; ngrids];
    for (a, q) in coords.iter().zip(charges.iter()) {
        let qi = -f64::from(*q);
        for (g, gv_g) in gv.iter().enumerate() {
            let theta = gv_g[0] * a[0] + gv_g[1] * a[1] + gv_g[2] * a[2];
            re[g] += qi * theta.cos();
            im[g] += -qi * theta.sin();
        }
    }
    for g in 0..ngrids {
        re[g] *= coulg[g];
        im[g] *= coulg[g];
    }
    CTensor::from_planes(re, im)
}

/// (b)_nuc oracle: field energy with the DENSITY frozen at the working
/// geometry — `ReΣ F(R)·D_frozen/vol`, no conjugation (matches the
/// clause-10 primitive's association exactly).
fn nuc_field_energy(cell: &Cell, rho_frozen: &CTensor) -> f64 {
    let mesh = cell.mesh;
    let ngrids = ngrids_of(cell);
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh)).expect("gv");
    let coulg = pyscf_pbc_gto::get_coulg_at_gv(cell, mesh, &gv).expect("coulg");
    let coords = cell.mol.atom_coords();
    let charges = cell.atom_charges();
    let mut re = vec![0.0f64; ngrids];
    let mut im = vec![0.0f64; ngrids];
    for (a, q) in coords.iter().zip(charges.iter()) {
        let qi = -f64::from(*q);
        for (g, gv_g) in gv.iter().enumerate() {
            let theta = gv_g[0] * a[0] + gv_g[1] * a[1] + gv_g[2] * a[2];
            re[g] += qi * theta.cos();
            im[g] += -qi * theta.sin();
        }
    }
    let terms: Vec<f64> = (0..ngrids)
        .map(|g| re[g] * coulg[g] * rho_frozen.re[g] + im[g] * coulg[g] * rho_frozen.im[g])
        .collect();
    oracle_sum(&terms) / cell.vol()
}

/// (a) oracle: grid energy with the POTENTIAL frozen at the working
/// geometry — `w·Σ_g X[g]·X[g]·v_frozen[g]`, real-space physical potential.
fn frozen_grid_energy(cell: &Cell, dm: &[f64], v_frozen: &[f64]) -> f64 {
    let mesh = cell.mesh;
    let ngrids = ngrids_of(cell);
    let grids = pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(cell, Some(mesh)).expect("grids");
    let coords = grids.coords().expect("coords").to_vec();
    let ao =
        pyscf_pbc_gto::eval_gto::eval_ao_kpts(cell, "GTOval_sph_deriv1", &coords, &[]).expect("ao");
    let blk = &ao.kaos[0].re;
    let nao = cell.mol.nao_nr;
    let w = cell.vol() / ngrids as f64;
    let mut e = 0.0f64;
    for g in 0..ngrids {
        for mu in 0..nao {
            for nu in 0..nao {
                e += dm[mu * nao + nu] * blk[g + mu * ngrids] * blk[g + nu * ngrids] * v_frozen[g];
            }
        }
    }
    e * w
}

fn physical_potential(v_g: &CTensor, cell: &Cell) -> Vec<f64> {
    // Bare Fourier components (nuclear_vg, vpploc_g_part1): ifft IS the
    // physical potential. WEIGHTED fields (mg_xc_parts rows) must be
    // unweighted by the caller first — see pass2_ip1's weight note.
    pyscf_pbc_tools::ifft(v_g, cell.mesh).expect("ifft").re
}

// ---------------------------------------------------------------------------
// (b)_nuc vs frozen-density FD — machine precision (he + diamond).
// ---------------------------------------------------------------------------

#[test]
fn nuc_nuc_grad_matches_frozen_fd() {
    // He first: single atom, gradient is translation noise (~0 vs ~0),
    // which is exactly what the frozen oracle reproduces.
    let cell = he_cell();
    let ni = ni();
    let dm = fixed_dm(&cell);
    let rho = ni.eval_rho_g(&cell, &dm).expect("rhoG");
    let grad = ni
        .get_nuc_nuc_grad(&cell, &dm, &[], Some(&rho))
        .expect("nuc_nuc_grad");
    let mut worst = 0.0f64;
    for x in 0..3 {
        let ep = nuc_field_energy(&displaced(&cell, 0, x, H, "gth-szv"), &rho);
        let em = nuc_field_energy(&displaced(&cell, 0, x, -H, "gth-szv"), &rho);
        worst = worst.max((grad[0][x] - (ep - em) / (2.0 * H)).abs());
    }
    println!("18-09 gate nuc_nuc_grad he_fcc worst |analytic-FD| = {worst:.3e}");
    assert!(worst < 1e-9, "he_fcc nuc-nuc FD residual {worst:.3e}");

    // Diamond, worked off-equilibrium so the forces are non-trivial.
    let base = dia_cell();
    let work = displaced(&base, 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let rho = ni.eval_rho_g(&work, &dm).expect("rhoG");
    let grad = ni
        .get_nuc_nuc_grad(&work, &dm, &[], Some(&rho))
        .expect("nuc_nuc_grad");
    assert_eq!(grad.len(), 2);
    let mut worst = 0.0f64;
    for ia in 0..2 {
        for x in 0..3 {
            let ep = nuc_field_energy(&displaced(&work, ia, x, H, "gth-szv"), &rho);
            let em = nuc_field_energy(&displaced(&work, ia, x, -H, "gth-szv"), &rho);
            worst = worst.max((grad[ia][x] - (ep - em) / (2.0 * H)).abs());
        }
    }
    println!("18-09 gate nuc_nuc_grad diamond worst |analytic-FD| = {worst:.3e}");
    assert!(worst < 1e-9, "diamond nuc-nuc FD residual {worst:.3e}");
    println!("18-09 gate nuc_nuc_grad ok (he_fcc, diamond)");
}

// ---------------------------------------------------------------------------
// Batched vs a literal per-atom transcription of upstream :916-919.
// ---------------------------------------------------------------------------

#[test]
fn nuc_nuc_grad_batched_matches_literal_bitwise() {
    let base = dia_cell();
    let work = displaced(&base, 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let ni = ni();
    // Same density both sides — the comparison is loop structure only.
    let rho = ni.eval_rho_g(&work, &dm).expect("rhoG");
    let batched = ni
        .get_nuc_nuc_grad(&work, &dm, &[], Some(&rho))
        .expect("batched");

    // The literal transcription: one (3, ngrids) complex array per atom,
    // `vG = 1j·e^{iG·A}·q·coulG·G`, dotted against rhoG — same
    // transcendental spelling (`sin_cos`), same product association, same
    // oracle_sum order, so the gate is bitwise rather than epsilon-close.
    let mesh = work.mesh;
    let ngrids = ngrids_of(&work);
    let gv = pyscf_pbc_gto::get_gv(&work, Some(mesh)).expect("gv");
    let coulg = pyscf_pbc_gto::get_coulg_at_gv(&work, mesh, &gv).expect("coulg");
    let coords = work.mol.atom_coords();
    let charges = work.atom_charges();
    let vol = work.vol();
    for (ia, (a, q)) in coords.iter().zip(charges.iter()).enumerate() {
        let qi = -f64::from(*q);
        for x in 0..3 {
            let terms: Vec<f64> = (0..ngrids)
                .map(|g| {
                    let theta = gv[g][0] * a[0] + gv[g][1] * a[1] + gv[g][2] * a[2];
                    let (s, c) = theta.sin_cos();
                    let cr = coulg[g] * qi;
                    let vgr = -cr * gv[g][x] * s;
                    let vgi = cr * gv[g][x] * c;
                    vgr * rho.re[g] - vgi * rho.im[g]
                })
                .collect();
            let literal = oracle_sum(&terms) / vol;
            assert_eq!(
                batched[ia][x].to_bits(),
                literal.to_bits(),
                "batched != literal at atom {ia} comp {x}"
            );
        }
    }
    println!("18-09 gate batched-vs-literal bit-identical ok (diamond)");
}

// ---------------------------------------------------------------------------
// (a)_nuc vs frozen-potential FD — machine precision (diamond).
// ---------------------------------------------------------------------------

#[test]
fn nuc_ip1_matches_frozen_fd() {
    let base = dia_cell();
    let work = displaced(&base, 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let ni = ni();
    let nao = work.mol.nao_nr;
    let ip1 = ni.get_nuc_ip1(&work, &[]).expect("nuc_ip1");
    assert_eq!(ip1.len(), 3 * nao * nao);
    // The potential is frozen at the working geometry; only the AOs move.
    let v_frozen = physical_potential(&nuclear_vg(&work), &work);
    let slices = aoslices(&work);
    let mut worst = 0.0f64;
    for (ia, &(_, _, p0, p1)) in slices.iter().enumerate() {
        for x in 0..3 {
            let ep = frozen_grid_energy(&displaced(&work, ia, x, H, "gth-szv"), &dm, &v_frozen);
            let em = frozen_grid_energy(&displaced(&work, ia, x, -H, "gth-szv"), &dm, &v_frozen);
            worst =
                worst.max((contract_bra(&ip1, &dm, nao, p0, p1, x) - (ep - em) / (2.0 * H)).abs());
        }
    }
    println!("18-09 gate nuc_ip1 diamond worst |analytic-FD| = {worst:.3e}");
    assert!(worst < 1e-9, "nuc_ip1 FD residual {worst:.3e}");
    println!("18-09 gate nuc_ip1 ok (diamond)");
}

// ---------------------------------------------------------------------------
// get_veff_ip1 vs FD of nr_rks energy — LDA, GGA and HF (diamond, tight
// precision). No (b) piece exists at fixed dm, so the full FD applies.
// ---------------------------------------------------------------------------

fn rks_energy(cell: &Cell, xc: &str, dm: &[f64]) -> f64 {
    let r = ni().nr_rks(cell, xc, dm).expect("nr_rks");
    r.ecoul + r.exc
}

fn veff_ip1_fd_gate(xc: &str, hf: bool) {
    // Tolerance 5e-9 (measured ~9e-10): the optimal-FD floor, NOT a
    // formula error — `E`-rounding/`h` and truncation cross at ~1e-9 here
    // (see H's doc). Any wrong sign, weight, or convention misses by
    // 1e-3..1e-1 (all of them did during development), so 5e-9 still gates
    // correctness with orders of margin. Compare Gate E (18-01 floor for
    // the full SCF gradient on fine meshes): diamond 7.5e-10.
    const TOL: f64 = 5e-9;
    let base = dia_tight_cell();
    let work = displaced(&base, 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let ni = ni();
    let got = ni.get_veff_ip1(&work, xc, &dm, &[]).expect("get_veff_ip1");
    let nao = work.mol.nao_nr;
    assert_eq!(got.veff_ip1.len(), 3 * nao * nao);
    let slices = aoslices(&work);
    let mut worst = 0.0f64;
    for (ia, &(_, _, p0, p1)) in slices.iter().enumerate() {
        for x in 0..3 {
            let (ep, em) = if hf {
                // Coulomb-only oracle: the same ecoul formula mg_xc_parts
                // uses, recomputed per displaced geometry.
                (
                    j_energy(&displaced(&work, ia, x, H, "gth-szv"), &dm),
                    j_energy(&displaced(&work, ia, x, -H, "gth-szv"), &dm),
                )
            } else {
                (
                    rks_energy(&displaced(&work, ia, x, H, "gth-szv"), xc, &dm),
                    rks_energy(&displaced(&work, ia, x, -H, "gth-szv"), xc, &dm),
                )
            };
            let fd = (ep - em) / (2.0 * H);
            let analytic = contract_bra(&got.veff_ip1, &dm, nao, p0, p1, x);
            worst = worst.max((analytic - fd).abs());
        }
    }
    println!("18-09 gate veff_ip1 {xc} diamond worst |analytic-FD| = {worst:.3e}");
    assert!(worst < TOL, "{xc} veff_ip1 FD residual {worst:.3e}");
}

/// `0.5·ReΣ conj(rhoG)·(rhoG·coulG)/vol` — the Coulomb energy mg_xc_parts
/// reports as `ecoul`, rebuilt per geometry for the HF FD oracle.
fn j_energy(cell: &Cell, dm: &[f64]) -> f64 {
    let rho = ni().eval_rho_g(cell, dm).expect("rhoG");
    let mesh = cell.mesh;
    let gv = pyscf_pbc_gto::get_gv(cell, Some(mesh)).expect("gv");
    let coulg = pyscf_pbc_gto::get_coulg_at_gv(cell, mesh, &gv).expect("coulg");
    let terms: Vec<f64> = (0..rho.re.len())
        .map(|g| rho.re[g] * rho.re[g] * coulg[g] + rho.im[g] * rho.im[g] * coulg[g])
        .collect();
    0.5 * oracle_sum(&terms) / cell.vol()
}

#[test]
fn veff_ip1_matches_fd_lda() {
    veff_ip1_fd_gate("lda,vwn", false);
    println!("18-09 gate veff_ip1 LDA ok (diamond)");
}

#[test]
fn veff_ip1_matches_fd_gga() {
    veff_ip1_fd_gate("pbe", false);
    println!("18-09 gate veff_ip1 GGA ok (diamond)");
}

#[test]
fn veff_ip1_matches_fd_hf() {
    veff_ip1_fd_gate("HF", true);
    println!("18-09 gate veff_ip1 HF ok (diamond)");
}

/// Adjoint identity: `Tr(dm·V) = Σ_g ρ[g]·v[g]·w` at association noise.
/// `V` is the plain pass2 matrix (deriv 0) rebuilt here through the same
/// eval_ao table the ip1 route uses, so the identity gates the table, the
/// weight convention and the contraction association in one number.
#[test]
fn collocation_adjoint_identity() {
    let cell = he_cell();
    let ni = ni();
    let dm = fixed_dm(&cell);
    let mesh = cell.mesh;
    let ngrids = ngrids_of(&cell);
    let nao = cell.mol.nao_nr;
    let rho_g = ni.eval_rho_g(&cell, &dm).expect("rhoG");
    // Any smooth G-space field will do — the identity is about the table,
    // the weight and the association, not about XC. Reuse the density
    // itself (physical scale, right smoothness).
    let parts = rho_g.clone();
    let v_r = pyscf_pbc_tools::ifft(&parts, mesh).expect("ifft");
    let weight = cell.vol() / ngrids as f64;
    let grids = pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(mesh)).expect("grids");
    let grids = grids.coords().expect("coords").to_vec();
    let ao =
        pyscf_pbc_gto::eval_gto::eval_ao_kpts(&cell, "GTOval_sph_deriv1", &grids, &[]).expect("ao");
    let blk = &ao.kaos[0].re;
    let at = |c: usize, g: usize, mu: usize| blk[c * ngrids * nao + g + mu * ngrids];
    // Matrix route: V[mu,nu] = w·Σ_g X·X·v, then Tr(dm·V).
    let mut e_mat_terms = Vec::with_capacity(nao * nao);
    for mu in 0..nao {
        for nu in 0..nao {
            let vterms: Vec<f64> = (0..ngrids)
                .map(|g| at(0, g, mu) * at(0, g, nu) * v_r.re[g])
                .collect();
            e_mat_terms.push(dm[mu * nao + nu] * weight * oracle_sum(&vterms));
        }
    }
    let e_mat = oracle_sum(&e_mat_terms);
    // Grid route: ρ[g] = Σ dm·X·X, then Σ ρ·v·w.
    let rho_terms: Vec<f64> = (0..ngrids)
        .map(|g| {
            let mut rg = Vec::with_capacity(nao * nao);
            for mu in 0..nao {
                for nu in 0..nao {
                    rg.push(dm[mu * nao + nu] * at(0, g, mu) * at(0, g, nu));
                }
            }
            oracle_sum(&rg) * v_r.re[g] * weight
        })
        .collect();
    let e_grid = oracle_sum(&rho_terms);
    let denom = e_mat.abs().max(e_grid.abs()).max(1e-30);
    let rel = (e_mat - e_grid).abs() / denom;
    println!("18-09 gate adjoint identity rel = {rel:.3e}");
    assert!(rel < 1e-12, "adjoint identity {rel:.3e}");
    println!("18-09 gate adjoint identity ok (he_fcc)");
}

// ---------------------------------------------------------------------------
// vpploc part-1: (b)_core vs frozen-density FD, (a)_core vs frozen-potential
// FD, cached-vs-recomputed bit identity, atm_id consistency.
// ---------------------------------------------------------------------------

/// (b)_core oracle: part-1 field energy with the density frozen —
/// `ReΣ vpplocG(R)·rho_frozen/vol`.
fn part1_field_energy(cell: &Cell, rho_frozen: &CTensor) -> f64 {
    let vpp = ni().vpploc_g_part1(cell).expect("vpplocG");
    assert_eq!(vpp.re.len(), rho_frozen.re.len());
    let terms: Vec<f64> = (0..vpp.re.len())
        .map(|g| vpp.re[g] * rho_frozen.re[g] + vpp.im[g] * rho_frozen.im[g])
        .collect();
    oracle_sum(&terms) / cell.vol()
}

#[test]
fn vpploc_pieces_match_frozen_fd() {
    let base = dia_cell();
    let work = displaced(&base, 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let ni = ni();
    let nao = work.mol.nao_nr;
    let slices = aoslices(&work);

    // (b)_core vs frozen-density FD.
    let rho = ni.eval_rho_g(&work, &dm).expect("rhoG");
    let grad = ni
        .vpploc_part1_nuc_grad(&work, &dm, &[], Some(&rho), None)
        .expect("vpploc grad");
    assert_eq!(grad.len(), 2);
    let mut worst = 0.0f64;
    for ia in 0..2 {
        for x in 0..3 {
            let ep = part1_field_energy(&displaced(&work, ia, x, H, "gth-szv"), &rho);
            let em = part1_field_energy(&displaced(&work, ia, x, -H, "gth-szv"), &rho);
            worst = worst.max((grad[ia][x] - (ep - em) / (2.0 * H)).abs());
        }
    }
    println!("18-09 gate vpploc_part1_nuc_grad diamond worst |analytic-FD| = {worst:.3e}");
    assert!(worst < 1e-9, "vpploc (b) FD residual {worst:.3e}");

    // (a)_core vs frozen-potential FD (smooth Gaussians — no cusp).
    let vpp = ni.vpploc_g_part1(&work).expect("vpplocG");
    let v_frozen = physical_potential(&vpp, &work);
    let ip1 = ni.get_vpploc_part1_ip1(&work, &[]).expect("vpploc ip1");
    assert_eq!(ip1.len(), 3 * nao * nao);
    let mut worst = 0.0f64;
    for (ia, &(_, _, p0, p1)) in slices.iter().enumerate() {
        for x in 0..3 {
            let ep = frozen_grid_energy(&displaced(&work, ia, x, H, "gth-szv"), &dm, &v_frozen);
            let em = frozen_grid_energy(&displaced(&work, ia, x, -H, "gth-szv"), &dm, &v_frozen);
            worst =
                worst.max((contract_bra(&ip1, &dm, nao, p0, p1, x) - (ep - em) / (2.0 * H)).abs());
        }
    }
    println!("18-09 gate vpploc_part1_ip1 diamond worst |analytic-FD| = {worst:.3e}");
    assert!(worst < 1e-9, "vpploc (a) FD residual {worst:.3e}");

    // Cached rhoG (the get_veff_ip1 handle) vs recomputed — bit-identical.
    let cached = ni
        .get_veff_ip1(&work, "lda,vwn", &dm, &[])
        .expect("veff for rhoG handle");
    let from_cached = ni
        .vpploc_part1_nuc_grad(&work, &dm, &[], Some(&cached.rho_g), None)
        .expect("cached");
    let recomputed = ni
        .vpploc_part1_nuc_grad(&work, &dm, &[], None, None)
        .expect("recomputed");
    for ia in 0..from_cached.len() {
        for x in 0..3 {
            assert_eq!(
                from_cached[ia][x].to_bits(),
                recomputed[ia][x].to_bits(),
                "cached != recomputed at atom {ia} comp {x}"
            );
        }
    }
    // Explicit-atm_id path recomputes and restricts: selected rows agree
    // with the full call, the rest are zero.
    let subset = ni
        .vpploc_part1_nuc_grad(&work, &dm, &[], None, Some(&[1]))
        .expect("subset");
    for x in 0..3 {
        assert_eq!(subset[1][x].to_bits(), recomputed[1][x].to_bits());
        assert_eq!(subset[0][x].to_bits(), 0.0f64.to_bits());
    }
    println!("18-09 gate vpploc_part1 ok (diamond)");
}

#[test]
fn repeat_calls_are_bit_identical() {
    let work = displaced(&dia_cell(), 0, 0, WORK_SHIFT, "gth-szv");
    let dm = fixed_dm(&work);
    let ni = ni();
    let a = ni.get_nuc_nuc_grad(&work, &dm, &[], None).expect("first");
    let b = ni.get_nuc_nuc_grad(&work, &dm, &[], None).expect("second");
    for ia in 0..a.len() {
        for x in 0..3 {
            assert_eq!(a[ia][x].to_bits(), b[ia][x].to_bits());
        }
    }
    let c = ni.get_veff_ip1(&work, "pbe", &dm, &[]).expect("veff");
    let d = ni.get_veff_ip1(&work, "pbe", &dm, &[]).expect("veff");
    assert_eq!(c.veff_ip1.len(), d.veff_ip1.len());
    for (x, y) in c.veff_ip1.iter().zip(d.veff_ip1.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    println!("18-09 gate repeat-call bit-identity ok (diamond)");
}
