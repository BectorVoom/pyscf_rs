//! K-03 — the forward multigrid scatter, on the device.
//!
//! Before K-03 the batched forward route read `npoints · 8` B back **per
//! chunk** and then ran a host loop `rho[point_global[p]] = out[p]` over
//! every padded point of every chunk of every level, on every SCF cycle.
//! K-03 keeps the level's density on the device across its chunks
//! (`PairOutScratch::mesh`), scatters into it with `mg_scatter_kernel`, and
//! reads `ngrids · 8` B back ONCE per level.
//!
//! # The claim is bit-identity, and this file is what makes it a claim
//!
//! `grid_blocks`' partition gives each real grid index exactly one owning
//! block, so the scatter is a plain STORE, not an accumulate — there is no
//! summation whose order could change, and the device route writes the same
//! `f64`s to the same indices as the host loop. That is an argument. The
//! test below is the evidence: both routes run **in one process** (the
//! `PYSCF_MG_PAIR_DEVICE_SCATTER` seam, M-03's `use_batch` seam applied to
//! the other direction) and are compared with `to_bits()`.
//!
//! # Serial, and deliberately so
//!
//! The seam is an environment variable, which is process-global. These
//! tests are marked `#[serial]`-by-construction: they live in their own
//! integration binary and the runner is given `--test-threads=1` by the
//! gate. Running them concurrently with anything that collocates would let
//! one test's setting leak into another's measurement.

mod common;

use pyscf_pbc_dft::multigrid::MultiGridNumInt2;
use pyscf_pbc_gto::{Cell, make_kpts_default};

const MESH: [usize; 3] = [15, 15, 15];

fn cell() -> Cell {
    let mut c = common::diamond();
    c.mesh = MESH;
    c
}

fn converged_dm(cell: &Cell, nk: [usize; 3]) -> (Vec<[f64; 3]>, Vec<pyscf_algebra::CTensor>) {
    let kpts = make_kpts_default(cell, nk).expect("k-mesh");
    let df = pyscf_pbc_df::Fftdf::with_mesh(cell.clone(), &kpts, cell.mesh).expect("FFTDF");
    let mf = pyscf_pbc_dft::krks::Krks::from_df(Box::new(df), "lda,vwn").expect("KRKS");
    let r = mf
        .kernel(&pyscf_pbc_scf::KScfConfig {
            conv_tol: 1e-10,
            max_cycle: 60,
            ..pyscf_pbc_scf::KScfConfig::for_cell(cell)
        })
        .expect("KRKS kernel");
    assert!(r.converged, "fixture SCF did not converge");
    (kpts, r.dm[0].clone())
}

/// Run `f` with the device scatter forced on, then forced off. The numint is
/// rebuilt for each so no cached device buffer is shared between the two.
fn both_routes<T>(mut f: impl FnMut() -> T) -> (T, T) {
    // SAFETY (`set_var`): this test binary is single-threaded — the gate
    // runs it with `--test-threads=1` and nothing here spawns a thread that
    // reads the environment concurrently.
    unsafe { std::env::set_var("PYSCF_MG_PAIR_DEVICE_SCATTER", "1") };
    let on = f();
    unsafe { std::env::set_var("PYSCF_MG_PAIR_DEVICE_SCATTER", "0") };
    let off = f();
    unsafe { std::env::remove_var("PYSCF_MG_PAIR_DEVICE_SCATTER") };
    (on, off)
}

/// The gamma-point closed-shell forward+reverse sweep, both routes,
/// `to_bits()`.
#[test]
fn device_scatter_is_bit_identical_to_the_host_scatter_at_gamma() {
    let cell = cell();
    let (_, dm_k) = converged_dm(&cell, [1, 1, 1]);
    let dm = dm_k[0].re.clone();

    for xc in ["lda,vwn", "pbe,pbe"] {
        let (on, off) = both_routes(|| {
            let ni = MultiGridNumInt2::new();
            ni.nr_rks(&cell, xc, &dm).expect("nr_rks")
        });

        assert_eq!(on.nelec.to_bits(), off.nelec.to_bits(), "{xc}: nelec");
        assert_eq!(on.exc.to_bits(), off.exc.to_bits(), "{xc}: exc");
        assert_eq!(on.ecoul.to_bits(), off.ecoul.to_bits(), "{xc}: ecoul");
        assert_eq!(on.veff.len(), off.veff.len());
        for i in 0..on.veff.len() {
            assert_eq!(
                on.veff[i].to_bits(),
                off.veff[i].to_bits(),
                "{xc}: veff[{i}] — the device scatter must WRITE the same f64s to \
                 the same grid indices the host loop wrote, not merely agree to a \
                 tolerance"
            );
        }
    }
}

/// The k-resolved sweep, both routes. The scatter knows nothing about
/// k-points — it runs once per level per sweep whatever `nkpts` is — so this
/// mainly proves the two changes compose.
#[test]
fn device_scatter_is_bit_identical_under_kpoints() {
    let cell = cell();
    let (kpts, dm_k) = converged_dm(&cell, [2, 2, 3]);

    let (on, off) = both_routes(|| {
        let ni = MultiGridNumInt2::new();
        ni.nr_rks_kpts(&cell, "lda,vwn", &dm_k, &kpts, None)
            .expect("nr_rks_kpts")
    });

    assert_eq!(on.nelec.to_bits(), off.nelec.to_bits(), "nelec");
    assert_eq!(on.exc.to_bits(), off.exc.to_bits(), "exc");
    assert_eq!(on.ecoul.to_bits(), off.ecoul.to_bits(), "ecoul");
    for k in 0..on.veff.len() {
        for i in 0..on.veff[k].re.len() {
            assert_eq!(
                on.veff[k].re[i].to_bits(),
                off.veff[k].re[i].to_bits(),
                "k={k} re[{i}]"
            );
            assert_eq!(
                on.veff[k].im[i].to_bits(),
                off.veff[k].im[i].to_bits(),
                "k={k} im[{i}]"
            );
        }
    }
}

/// The A/B wall clock of the two routes, on THIS machine's backend.
///
/// **Reported, not asserted.** D-PBC-26 point 6 and the `zgemm_dense`
/// precedent both say the same thing: a route that is structurally better
/// is not automatically faster, and a speed claim is worth what its
/// measurement is worth. What K-03 removes is unconditional — one read-back
/// per level instead of one per chunk, and no host scatter loop over every
/// padded point — but on the CPU runtime, which is this workspace's default
/// backend, "device memory" IS host memory and a read-back is a memcpy, so
/// the win there is bounded by the host loop it deletes rather than by any
/// bus. On a discrete GPU the same change removes real PCIe traffic.
///
/// This test therefore prints both times and passes either way; the number
/// belongs in the ledger, not in an assertion that would fail on a machine
/// whose backend makes it moot.
#[test]
fn device_scatter_ab_wall_clock() {
    use std::time::Instant;

    let cell = cell();
    let (_, dm_k) = converged_dm(&cell, [1, 1, 1]);
    let dm = dm_k[0].re.clone();
    const REPS: u32 = 3;

    let time_it = || {
        // Warm the backend and the level cache first — otherwise the first
        // route measured pays for both.
        let ni = MultiGridNumInt2::new();
        let _ = ni.nr_rks(&cell, "lda,vwn", &dm).expect("warm");
        let t = Instant::now();
        for _ in 0..REPS {
            let _ = ni.nr_rks(&cell, "lda,vwn", &dm).expect("nr_rks");
        }
        t.elapsed().as_secs_f64() / f64::from(REPS)
    };
    let (on, off) = both_routes(time_it);

    println!(
        "K-03 A/B on {} (mesh {MESH:?}, {REPS} reps after a warm-up):\n  \
         device scatter ON  : {on:.4} s / nr_rks\n  \
         device scatter OFF : {off:.4} s / nr_rks\n  \
         ratio (off/on)     : {:.3}x",
        pyscf_algebra::select_backend()
            .map(|s| s.client.kind().name().to_string())
            .unwrap_or_else(|_| "unknown backend".to_string()),
        off / on,
    );
}

/// The open-shell sweep, which scatters TWO meshes per chunk and reads both
/// back in one batched call — the arm most likely to cross its channels.
#[test]
fn device_scatter_is_bit_identical_open_shell() {
    let mut c = common::li_atom_spin1();
    c.mesh = [13, 13, 13];
    let nao = c.mol.nao_nr;

    // A pair of distinct, symmetric, positive-definite spin densities. The
    // two channels must NOT be equal, or a crossed scatter would pass.
    let mk = |seed: u64| {
        let mut st = seed;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64 / (1u64 << 53) as f64) * 0.05
        };
        let mut d = vec![0.0f64; nao * nao];
        for i in 0..nao {
            for j in 0..nao {
                d[i * nao + j] = next();
            }
        }
        for i in 0..nao {
            for j in 0..nao {
                let v = 0.5 * (d[i * nao + j] + d[j * nao + i]);
                d[i * nao + j] = v;
                d[j * nao + i] = v;
            }
            d[i * nao + i] += 0.5;
        }
        d
    };
    let (da, db) = (mk(11), mk(4242));
    assert!(
        da.iter().zip(&db).any(|(x, y)| x != y),
        "the two channels must differ, or a crossed scatter passes"
    );

    let (on, off) = both_routes(|| {
        let ni = MultiGridNumInt2::new();
        ni.nr_uks(&c, "lda,vwn", &[&da, &db]).expect("nr_uks")
    });

    assert_eq!(on.nelec.0.to_bits(), off.nelec.0.to_bits(), "nelec alpha");
    assert_eq!(on.nelec.1.to_bits(), off.nelec.1.to_bits(), "nelec beta");
    assert_eq!(on.exc.to_bits(), off.exc.to_bits(), "exc");
    assert_eq!(on.ecoul.to_bits(), off.ecoul.to_bits(), "ecoul");
    for s in 0..2 {
        for i in 0..on.veff[s].len() {
            assert_eq!(
                on.veff[s][i].to_bits(),
                off.veff[s][i].to_bits(),
                "spin {s} veff[{i}]"
            );
        }
    }
}
