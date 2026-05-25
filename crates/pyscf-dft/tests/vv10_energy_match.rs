//! DFT-06 (plan 04-07): VV10 non-local-correlation energy match
//! (`mf.nlc='VV10'` + a coarser `nlcgrids`) vs upstream PySCF, plus an
//! always-on structural layer proving the ported pure-Python `_vv10nlc`
//! double-loop runs end-to-end over a coarser second grid.
//!
//! ## Two-layer design (the 04-04/04-05/04-06 convention)
//!   - **Always-on structural layer** (default xcfun backend, no libpython):
//!     `nr_nlc_vxc` over a coarser `nlcgrids` produces a finite NLC energy +
//!     a symmetric NLC potential matrix from the ported double-loop; the
//!     coefficients resolve to the bare-VV10 default (A1); the `nlcgrids` is
//!     a SEPARATE, coarser `Grids` instance than the main grid.
//!   - **CI-only bit-exact layer**: the VV10-corrected RKS total energy
//!     matches upstream within 1 µHartree. Blocked by the same Phase-2 ERI /
//!     init-guess gap as the 04-06 DFT-01 oracle (a converged RKS run needs
//!     working arity-3/4 ERIs); CI-gated `--features python`.
//!
//! Upstream reference: `pyscf/dft/numint.py` `_vv10nlc` (471-555, the
//! commented pure-Python block 526-538 — NOT C `VXC_vv10nlc`, Pitfall 4) +
//! `nr_nlc_vxc` (1347-1416).

use pyscf_core::Unit;
use pyscf_dft::{NlcCoeffs, NumInt, nr_nlc_vxc};
use pyscf_grids::Grids;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

fn h2o_mol() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0.0 0.0 0.0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("H2O build")
}

/// Structural: `nr_nlc_vxc` over a coarser `nlcgrids` runs the ported VV10
/// double-loop end-to-end, producing a finite NLC energy and a symmetric
/// `nao × nao` NLC potential matrix. This is the `_vv10nlc` pure-Python port
/// exercised on a real grid (Pitfall 4: NOT the C VXC_vv10nlc).
#[test]
fn vv10_nlc_runs_end_to_end_over_coarser_nlcgrids() {
    let ni = NumInt::new();
    let mut mol = h2o_mol();
    let nao = mol.nao_nr;

    // A coarser nlcgrids — a SEPARATE Grids instance at a lower level than a
    // typical main grid (upstream nlcgrids defaults to a coarser level).
    let mut nlcgrids = Grids::new();
    nlcgrids.level = 1; // coarser than the class-default level 3 main grid.
    let (coords, weights) = nlcgrids.build(&mol);
    nlcgrids.coords = Some(coords);
    nlcgrids.weights = Some(weights.clone());
    assert!(!weights.is_empty(), "nlcgrids built with grid points");

    // A real, normalized-ish density (1e-shaped: identity-scaled) so ρ is
    // above the VV10 threshold somewhere on the grid.
    let mut data = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        data[i * nao + i] = 1.0;
    }
    let dm = pyscf_core::Density { nao, data };

    let res = nr_nlc_vxc(&ni, &mol, &nlcgrids, "VV10", &dm).expect("nr_nlc_vxc VV10");

    // Finite NLC energy + potential (the double-loop produced real numbers).
    assert!(
        res.excsum.is_finite(),
        "NLC excsum must be finite, got {}",
        res.excsum
    );
    assert!(
        res.nelec.is_finite() && res.nelec > 0.0,
        "NLC nelec finite + positive"
    );
    assert_eq!(res.vmat.nao, nao, "NLC Vxc is nao × nao");
    assert!(
        res.vmat.data.iter().all(|v| v.is_finite()),
        "NLC Vxc finite"
    );

    // The NLC potential matrix is symmetric (vmat += V + Vᵀ, numint.py:1415).
    for mu in 0..nao {
        for nu in 0..nao {
            let a = res.vmat.data[mu * nao + nu];
            let b = res.vmat.data[nu * nao + mu];
            assert!(
                (a - b).abs() < 1e-12,
                "NLC Vxc must be symmetric: V[{mu},{nu}]={a} vs V[{nu},{mu}]={b}"
            );
        }
    }

    // The shared mol is untouched by the NLC build (no env mutation here).
    let _ = &mut mol;
}

/// Structural: the bare 'VV10' coefficients resolve to the A1 default
/// (Bvv=5.9, Cvv=0.0093); per-functional codes need libxc (A1).
#[test]
fn vv10_default_coefficients_a1() {
    let c = NlcCoeffs::for_nlc_code("VV10").expect("bare VV10");
    assert_eq!(c.bvv, 5.9, "Bvv default = 5.9 (A1)");
    assert_eq!(c.cvv, 0.0093, "Cvv default = 0.0093 (A1)");
    assert!(
        NlcCoeffs::for_nlc_code("wB97X-V").is_err(),
        "per-functional nlc requires the libxc backend (A1)"
    );
}

/// Source assertions (Pitfall 4 + plan acceptance criteria): vv10.rs ports
/// the pure-Python double-loop, does NOT reference `VXC_vv10nlc`, uses
/// `oracle_sum` for the inner reductions, and the nlcgrids is a separate
/// (coarser) Grids instance.
#[test]
fn vv10_source_uses_pure_python_port_and_oracle_sum() {
    let src = include_str!("../src/vv10.rs");
    // Pitfall 4: no reference to the C kernel symbol as a call.
    assert!(
        !src.contains("VXC_vv10nlc("),
        "Pitfall 4: must port the pure-Python _vv10nlc, NOT call C VXC_vv10nlc"
    );
    // Inner-grid reductions go through oracle_sum (T-04-07b, bit-exact).
    assert!(
        src.contains("oracle_sum"),
        "VV10 inner reductions must use pyscf_algebra::oracle_sum (bit-exact)"
    );
    // The nlcgrids is a separate Grids instance (the orchestrator takes it
    // as a distinct argument).
    assert!(
        src.contains("nlcgrids: &Grids"),
        "nr_nlc_vxc must take a separate (coarser) nlcgrids Grids instance"
    );
}

/// DFT-06 bit-exact arm (CI-only): the VV10-corrected RKS total energy
/// matches upstream within 1 µHartree. Blocked by the Phase-2 ERI /
/// init-guess gap (a converged RKS run needs working arity-3/4 ERIs), same
/// as the 04-06 DFT-01 rks_uks_bitexact oracle. CI-gated `--features python`.
#[test]
#[ignore = "DFT-06 VV10 energy match: needs a converged RKS run (Phase-2 ERI/init-guess gap) + live PySCF — CI-only"]
fn vv10_energy_match() {
    // When the Phase-2 ERIs + init guess land:
    //   let mol = h2o_mol_ccpvdz();
    //   let mut mf = RKS::new(mol); mf.xc="pbe".into(); mf.nlc="VV10".into();
    //   mf.nlcgrids.level = 1; // coarser
    //   let e = mf.kernel()?;
    //   assert ≤ 1e-6 vs upstream dft.RKS(mol,'pbe'); mf.nlc='VV10'.
    unimplemented!("Phase-2 ERI/init-guess gap + live PySCF oracle (CI-only)");
}
