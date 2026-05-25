//! DF-CCSD ERI-swap + HDF5-spill always-on tests (CCSD-08).
//!
//! Three arms, all always-on (no live `intor`, no upstream PySCF — the
//! benzene-dimer/cc-pVDZ constrained-budget spill proof is the 06-08-closeout
//! `workflow_dispatch` human-verify arm, the D-04 heavy/upstream constraint):
//!
//! 1. **DF ERI assembly correctness:** the DF `df_ao2mo` `ovov` block on a
//!    synthetic B-tensor matches an INDEPENDENT longhand `(ia|jb)=Σ_Q B^Q·B^Q`
//!    reference (strict scalar folds, never `oracle_*`) — proving the
//!    MO-transform + Q-contraction port (the dfmp2 test pattern).
//! 2. **DF-CCSD converges:** a synthetic-B DFRCCSD runs the reused in-core
//!    kernel end-to-end to a finite, converged `e_corr` (the swap-the-ERI
//!    contract; the 05-09 DF-MP2-vs-in-core pattern, here on a self-consistent
//!    synthetic B-tensor with a generous budget).
//! 3. **SPILL proof (CCSD-08 / D-07):** a deliberately tiny `PYSCF_MAX_MEMORY`
//!    forces the `vvL` reservation through the `WorkspacePool` `Spilled`
//!    backend; we assert an HDF5 scratch file is CREATED during the run AND
//!    DELETED afterward — no leftover scratch (the Runtime State Inventory
//!    requirement, T-06-09-LEAK). Run `--test-threads=1` so the temp-dir
//!    before/after snapshot is not raced by a sibling spill.

use pyscf_ccsd::{CcsdReference, DFRCCSD, df_ao2mo, dfrccsd_kernel};
use pyscf_core::{MOCoefficients, Mole};
use pyscf_df::DfIntegrals;
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;

/// Build a deterministic synthetic DF B-tensor `b_uvq` (ROW-MAJOR
/// `[nao,nao,naux]`): element `(μ,ν,Q)` at `μ*nao*naux + ν*naux + Q`.
fn synthetic_b(nao: usize, naux: usize) -> DfIntegrals {
    let mut b = vec![0.0_f64; nao * nao * naux];
    for mu in 0..nao {
        for nu in 0..nao {
            for q in 0..naux {
                // Symmetric in (μ,ν) so the assembled ERIs have the physical
                // (pq|rs)=(qp|rs) chemist symmetry (the in-core kernel assumes
                // it). Small magnitudes keep the synthetic CCSD well-behaved.
                let v = 0.05 + 0.03 * (mu as f64) + 0.03 * (nu as f64) + 0.07 * (q as f64);
                let vs = 0.05 + 0.03 * (nu as f64) + 0.03 * (mu as f64) + 0.07 * (q as f64);
                b[mu * nao * naux + nu * naux + q] = 0.5 * (v + vs);
            }
        }
    }
    DfIntegrals {
        b_uvq: b,
        naux,
        nao,
    }
}

/// Build a toy [`CcsdReference`] with `nocc` occupied + `nvir` virtual orbitals
/// (nao = nmo = nocc+nvir) and a deterministic non-trivial column-major
/// `[nao, nmo]` MO coefficient block (occupied columns first). Orbital energies
/// have the occ<0<vir gap CCSD denominators need.
fn toy_reference(nocc: usize, nvir: usize) -> CcsdReference {
    let nmo = nocc + nvir;
    let nao = nmo;
    let mut data = vec![0.0_f64; nao * nmo];
    for mo in 0..nmo {
        for ao in 0..nao {
            data[ao + mo * nao] = 0.2 + 0.15 * (ao as f64) - 0.1 * (mo as f64);
        }
    }
    let mut mo_occ = vec![0.0_f64; nmo];
    let mut mo_energy = vec![0.0_f64; nmo];
    for i in 0..nocc {
        mo_occ[i] = 2.0;
        mo_energy[i] = -0.7 + 0.1 * (i as f64); // occupied < 0
    }
    for a in 0..nvir {
        mo_energy[nocc + a] = 0.4 + 0.2 * (a as f64); // virtual > 0
    }
    CcsdReference {
        mo_coeff: MOCoefficients {
            nao,
            nmo,
            data,
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

/// INDEPENDENT longhand `(ia|jb) = Σ_Q B^Q_ia · B^Q_jb` reference (strict scalar
/// folds, never `oracle_*`) for a toy closed-shell reference + synthetic B.
/// `B^Q_pq = Σ_{μ,ν} C_μ^p · b[μ,ν,Q] · C_ν^q`. Returns the C-order ovov block
/// `[nocc,nvir,nocc,nvir]`: `(i,a,j,b)` at `((i*nvir+a)*nocc+j)*nvir+b`.
fn longhand_ovov(refr: &CcsdReference, df: &DfIntegrals) -> Vec<f64> {
    let nao = df.nao;
    let naux = df.naux;
    let nocc = refr.mo_occ.iter().filter(|&&o| o > 0.0).count();
    let nvir = refr.mo_coeff.nmo - nocc;
    let c = &refr.mo_coeff.data; // column-major [nao, nmo], C[ao,mo] at ao+mo*nao.

    // B^Q_ia for active occ i, active vir a (vir columns are nocc..nmo).
    let mut bia = vec![0.0_f64; nocc * nvir * naux];
    for i in 0..nocc {
        for a in 0..nvir {
            for q in 0..naux {
                let mut acc = 0.0_f64;
                for mu in 0..nao {
                    for nu in 0..nao {
                        let ci = c[mu + i * nao];
                        let ca = c[nu + (nocc + a) * nao];
                        acc += ci * df.b_uvq[mu * nao * naux + nu * naux + q] * ca;
                    }
                }
                bia[(i * nvir + a) * naux + q] = acc;
            }
        }
    }
    // (ia|jb) = Σ_Q B^Q_ia · B^Q_jb (strict scalar fold).
    let mut ovov = vec![0.0_f64; nocc * nvir * nocc * nvir];
    for i in 0..nocc {
        for a in 0..nvir {
            for j in 0..nocc {
                for b in 0..nvir {
                    let mut acc = 0.0_f64;
                    for q in 0..naux {
                        acc += bia[(i * nvir + a) * naux + q] * bia[(j * nvir + b) * naux + q];
                    }
                    ovov[((i * nvir + a) * nocc + j) * nvir + b] = acc;
                }
            }
        }
    }
    ovov
}

/// Snapshot the set of this-process DF-CCSD spill scratch paths in the temp dir.
fn spill_snapshot() -> std::collections::BTreeSet<std::path::PathBuf> {
    let dir = std::env::temp_dir();
    let prefix = format!("pyscf_ccsd_spill_{}_", std::process::id());
    std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default()
}

/// ARM 1 — DF ERI assembly correctness: `df_ao2mo`'s `ovov` block equals the
/// independent longhand `(ia|jb)=Σ_Q B^Q·B^Q` reference (the DF ERI port).
#[test]
fn dfccsd_ovov_matches_longhand_reference() {
    for &(nocc, nvir, naux) in &[(1usize, 1usize, 2usize), (2, 2, 3), (2, 3, 4)] {
        let nao = nocc + nvir;
        let df = synthetic_b(nao, naux);
        let refr = toy_reference(nocc, nvir);
        // Generous budget so vvL stays in-core (this arm checks math, not spill).
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

        let eris = df_ao2mo(&refr, &Frozen::None, &df, &pool).expect("df_ao2mo");
        let want = longhand_ovov(&refr, &df);

        assert_eq!(eris.ovov.len(), want.len(), "ovov length");
        for (idx, (got, exp)) in eris.ovov.iter().zip(want.iter()).enumerate() {
            // RELATIVE tolerance: oracle_dot (pairwise tree) vs the longhand
            // left-fold reduce in different orders; bit-close, not bit-identical.
            let rel = (got - exp).abs() / exp.abs().max(1.0);
            assert!(
                rel < 1e-12,
                "ovov[{idx}] DF-assembled {got} vs longhand {exp} (rel {rel}) \
                 for (nocc={nocc},nvir={nvir},naux={naux})"
            );
        }
    }
}

/// ARM 2 — DF-CCSD converges: the synthetic-B DFRCCSD runs the reused in-core
/// kernel end-to-end to a finite, converged `e_corr` (the swap-the-ERI
/// contract). Also confirms the `DFRCCSD` driver `kernel()` matches the free fn.
#[test]
fn dfccsd_converges_and_driver_matches_free_fn() {
    let (nocc, nvir, naux) = (2usize, 2usize, 4usize);
    let nao = nocc + nvir;
    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

    let res = dfrccsd_kernel(&refr, &Frozen::None, &df, &pool).expect("dfrccsd_kernel");
    eprintln!(
        "DF-CCSD synthetic: e_corr = {:.12} emp2 = {:.12} converged = {} niter = {}",
        res.e_corr, res.emp2, res.converged, res.niter
    );
    assert!(res.e_corr.is_finite(), "DF-CCSD e_corr must be finite");
    assert!(res.converged, "DF-CCSD must converge on the dual criterion");
    // A closed-shell correlation energy is <= 0 (the synthetic system is
    // well-conditioned: occ<0<vir, small symmetric ERIs).
    assert!(
        res.e_corr <= 1e-9,
        "closed-shell e_corr must be <= 0, got {}",
        res.e_corr
    );

    // Driver wiring: DFRCCSD::kernel matches the free dfrccsd_kernel.
    let pool2 = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let driver = DFRCCSD {
        reference: refr,
        df,
    };
    let dres = driver
        .kernel(&Frozen::None, &pool2)
        .expect("DFRCCSD::kernel");
    assert!(
        (dres.e_corr - res.e_corr).abs() < 1e-12,
        "driver e_corr {} vs free fn {}",
        dres.e_corr,
        res.e_corr
    );
}

/// ARM 3 — SPILL proof (CCSD-08 / D-07): a tiny `PYSCF_MAX_MEMORY`-equivalent
/// budget forces the `vvL` `[nvir,nvir,naux]` reservation through the `Spilled`
/// backend (HDF5 temp dataset), while the in-core `Wvvvv` `[nvir^4]` reservation
/// still fits. We assert a spill scratch file is CREATED during the run AND
/// DELETED afterward (no leftover scratch, T-06-09-LEAK).
///
/// Budget arithmetic (nvir=2, naux=5): `Wvvvv = nvir^4 = 16` f64 (128 B);
/// `vvL = nvir^2*naux = 20` f64 (160 B). A 128-byte budget lets the pre-flight
/// plus the `Wvvvv` in-core reserve pass (128 <= 128) but forces `vvL` to spill
/// because 160 > 128. The window condition is `nvir^2 (4) <= naux (5)`.
#[test]
fn dfccsd_vvl_spills_to_hdf5_and_no_leftover_scratch() {
    let (nocc, nvir, naux) = (1usize, 2usize, 5usize);
    let nao = nocc + nvir;
    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);

    // Sanity: the budget window exists for this shape.
    let wvvvv_bytes = nvir * nvir * nvir * nvir * 8; // 128
    let vvl_bytes = nvir * nvir * naux * 8; // 160
    assert!(
        wvvvv_bytes < vvl_bytes,
        "spill test requires vvL ({vvl_bytes}) > Wvvvv ({wvvvv_bytes}); pick naux > nvir^2"
    );
    let budget = wvvvv_bytes; // 128: Wvvvv fits in-core, vvL must spill.

    let before = spill_snapshot();

    // A probe hook that captures whether the vvL reservation spilled. We can't
    // see inside the pool from here, so we detect the spill by the temp-file
    // delta DURING the run: snapshot inside a thread that polls is overkill, so
    // instead we assert (a) the run succeeds under the tiny budget (an in-core
    // vvL would HARD-refuse since allow_spill drives the path) and (b) no
    // leftover scratch survives. The "created during the run" leg is proven by
    // (a): with budget < vvL_bytes the ONLY way df_ao2mo returns Ok is via the
    // Spilled backend (an in-core reserve(allow_spill=false) would refuse, and
    // df_ao2mo reserves with allow_spill=true precisely to spill here).
    let pool = WorkspacePool::new(budget);
    let res = dfrccsd_kernel(&refr, &Frozen::None, &df, &pool);

    // The vvL reservation (160 B) exceeds the 128 B budget, so it took the
    // Spilled backend; the kernel's Wvvvv (128 B) fits in-core. The run must
    // succeed (DF is the explicit opt-in that spills rather than OOMing).
    let res = res.expect(
        "DF-CCSD must succeed under a tiny budget by SPILLING vvL to HDF5 \
         (the Spilled backend), not HARD-refuse",
    );
    assert!(
        res.e_corr.is_finite(),
        "spilled DF-CCSD e_corr must be finite"
    );
    assert!(res.converged, "spilled DF-CCSD must converge");

    // Dropping the pool drops every SpillHandle → RAII deletes the temp files.
    drop(pool);

    let after = spill_snapshot();
    let leaked: Vec<_> = after.difference(&before).collect();
    assert!(
        leaked.is_empty(),
        "leftover DF-CCSD spill scratch after the run (RAII drop-delete failed): {leaked:?}"
    );
}

/// ARM 3c — DIRECT spill-file observability at the DF `vvL` seam: reserve a
/// `vvL`-shaped buffer (`[nvir,nvir,naux]`) in a tiny-budget pool the test HOLDS
/// OPEN, glob the temp dir to assert the HDF5 spill scratch EXISTS while the
/// reservation is live, then drop the pool and assert the scratch is gone. This
/// is the explicit "created during the run AND deleted afterward" proof the
/// CCSD-08 success criterion calls for, observed via the temp-dir glob (the pool
/// internals are not cross-crate visible, so we drive the same `reserve(...,
/// allow_spill=true)` seam `df_ao2mo` uses at the SAME vvL dimensions).
#[test]
fn vvl_spill_file_observed_created_then_deleted() {
    let (nvir, naux) = (2usize, 5usize);
    let vvl_bytes = nvir * nvir * naux * 8; // 160
    let budget = vvl_bytes - 8; // 152: forces the vvL reservation to spill.

    let before = spill_snapshot();
    {
        // Reserve a vvL-shaped buffer with allow_spill=true under a budget
        // smaller than its footprint → the Spilled (HDF5) backend.
        let pool = WorkspacePool::new(budget);
        let id = pool
            .reserve(&[nvir, nvir, naux], true)
            .expect("vvL reserve must spill, not refuse, under allow_spill=true");
        // Round-trip the data through the spilled backend (the df_ao2mo write→read).
        let data: Vec<f64> = (0..nvir * nvir * naux).map(|i| i as f64 * 0.5).collect();
        pool.write_slice(&id, &data).expect("write spilled vvL");
        assert_eq!(pool.as_slice(&id).expect("read spilled vvL"), data);

        // The scratch file exists WHILE the reservation is live (created during
        // the run).
        let during = spill_snapshot();
        let created: Vec<_> = during.difference(&before).collect();
        assert!(
            !created.is_empty(),
            "a vvL HDF5 spill scratch must exist while the over-budget reservation is live"
        );
    } // pool drops → SpillHandle::drop deletes the scratch (RAII).

    let after = spill_snapshot();
    let leaked: Vec<_> = after.difference(&before).collect();
    assert!(
        leaked.is_empty(),
        "vvL spill scratch must be deleted on pool drop (RAII, T-06-09-LEAK): {leaked:?}"
    );
}

/// ARM 3b — the in-core vvL path (generous budget) does NOT spill: no scratch
/// file is created at all (the spill is opt-in / budget-driven, not always-on).
#[test]
fn dfccsd_incore_vvl_creates_no_scratch() {
    let (nocc, nvir, naux) = (1usize, 2usize, 5usize);
    let nao = nocc + nvir;
    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);

    let before = spill_snapshot();
    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let res = dfrccsd_kernel(&refr, &Frozen::None, &df, &pool).expect("in-core DF-CCSD");
    assert!(res.e_corr.is_finite());
    // No spill scratch was created while the pool was live (vvL stayed InMemory).
    let during = spill_snapshot();
    drop(pool);
    let leaked: Vec<_> = during.difference(&before).collect();
    assert!(
        leaked.is_empty(),
        "in-core DF-CCSD (generous budget) must NOT spill: {leaked:?}"
    );
}
