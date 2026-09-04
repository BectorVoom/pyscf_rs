//! **M-02 — the multigrid geometry cache is bit-exact and cannot go stale.**
//!
//! Plan item M-02 of
//! `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`. Both multigrid
//! drivers used to rebuild their entire geometry on every call — the
//! decontraction, the level task list, and (v2) every level's pair table,
//! block partition and per-block reach list — none of which depends on the
//! density. v1 additionally collocated every level TWICE per call, once for
//! the density and once for `pass2`.
//!
//! Caching a pure function is only safe if two things hold, and this file
//! asserts both rather than assuming either:
//!
//! 1. **The cached path returns exactly what the uncached path returned.** A
//!    geometry cache that changed a number would be a defect, not an
//!    optimisation, so the assertion is `to_bits()` equality and not a
//!    tolerance.
//! 2. **A different cell is never served another cell's geometry.** This is
//!    the failure mode a one-entry cache invites, it would produce a
//!    plausible-looking wrong energy rather than a crash, and it is exactly
//!    what `cell_fingerprint` exists to prevent.
//!
//! The third property — that the cache is actually *used* — is asserted
//! indirectly by (1) plus the fact that a stale-geometry bug would show up
//! there; timing is reported by the plan's GATE S ledger, not gated here.

mod common;

use pyscf_pbc_dft::multigrid::{MultiGridNumInt, MultiGridNumInt2};
use pyscf_pbc_gto::Cell;

const MESH: [usize; 3] = [25, 25, 25];
/// A second, genuinely different mesh — the cheapest way to make a cell whose
/// geometry differs while everything else about it is identical, which is the
/// hardest case for a fingerprint to get right.
const MESH_OTHER: [usize; 3] = [21, 21, 21];

fn silicon_at(mesh: [usize; 3]) -> Cell {
    let mut c = common::silicon();
    c.mesh = mesh;
    c
}

fn diamond_at(mesh: [usize; 3]) -> Cell {
    let mut c = common::diamond();
    c.mesh = mesh;
    c
}

fn random_symmetric_dm(nao: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 0.2
    };
    let mut dm = vec![0.0f64; nao * nao];
    for i in 0..nao {
        for j in 0..nao {
            dm[i * nao + j] = next();
        }
    }
    for i in 0..nao {
        for j in 0..nao {
            let v = 0.5 * (dm[i * nao + j] + dm[j * nao + i]);
            dm[i * nao + j] = v;
            dm[j * nao + i] = v;
        }
        dm[i * nao + i] += 1.0;
    }
    dm
}

fn same_bits_slice(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}[{i}] differs: {x:.17e} vs {y:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// v1
// ---------------------------------------------------------------------------

#[test]
fn v1_cached_and_uncached_agree_bit_for_bit() {
    let cell = silicon_at(MESH);
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x0BAD_F00D);

    // A driver whose cache is cold on every call — the pre-M-02 behaviour,
    // reproduced by resetting between calls rather than by keeping a second
    // copy of the old code.
    let cold = MultiGridNumInt::new();
    cold.reset();
    let a = cold.nr_rks(&cell, "PBE", &dm).expect("uncached nr_rks");
    cold.reset();
    let a2 = cold
        .nr_rks(&cell, "PBE", &dm)
        .expect("uncached nr_rks again");

    // A driver that keeps its geometry.
    let warm = MultiGridNumInt::new();
    let _ = warm.nr_rks(&cell, "PBE", &dm).expect("warming call");
    let b = warm.nr_rks(&cell, "PBE", &dm).expect("cached nr_rks");

    for (tag, x) in [("uncached-twice", &a2), ("cached", &b)] {
        assert_eq!(
            a.nelec.to_bits(),
            x.nelec.to_bits(),
            "v1 {tag}: nelec moved ({} vs {})",
            a.nelec,
            x.nelec
        );
        assert_eq!(a.exc.to_bits(), x.exc.to_bits(), "v1 {tag}: exc moved");
        assert_eq!(
            a.ecoul.to_bits(),
            x.ecoul.to_bits(),
            "v1 {tag}: ecoul moved"
        );
        same_bits_slice(&format!("v1 {tag} veff"), &a.veff, &x.veff);
    }
}

/// The stale-geometry trap: one driver, two cells. If the fingerprint missed
/// anything the second call would silently use the first cell's task list and
/// return a plausible, wrong number.
#[test]
fn v1_does_not_serve_one_cell_the_geometry_of_another() {
    let ni = MultiGridNumInt::new();

    let si = silicon_at(MESH);
    let dm_si = random_symmetric_dm(si.mol.nao_nr, 0x11);
    let c = diamond_at(MESH);
    let dm_c = random_symmetric_dm(c.mol.nao_nr, 0x22);
    let si_other = silicon_at(MESH_OTHER);

    // Reference values, each from a driver that has seen nothing else.
    let want_si = MultiGridNumInt::new().get_j(&si, &dm_si).expect("si");
    let want_c = MultiGridNumInt::new().get_j(&c, &dm_c).expect("diamond");
    let want_si_other = MultiGridNumInt::new()
        .get_j(&si_other, &dm_si)
        .expect("si at the other mesh");

    // The same driver, interleaved. A different CELL and — the harder case —
    // the same cell at a different MESH must both miss the cache.
    same_bits_slice("v1 si", &want_si, &ni.get_j(&si, &dm_si).expect("si"));
    same_bits_slice(
        "v1 diamond",
        &want_c,
        &ni.get_j(&c, &dm_c).expect("diamond"),
    );
    same_bits_slice("v1 si", &want_si, &ni.get_j(&si, &dm_si).expect("si again"));
    same_bits_slice(
        "v1 si at a different mesh",
        &want_si_other,
        &ni.get_j(&si_other, &dm_si).expect("si, other mesh"),
    );
}

// ---------------------------------------------------------------------------
// v2
// ---------------------------------------------------------------------------

#[test]
fn v2_cached_and_uncached_agree_bit_for_bit() {
    let cell = silicon_at(MESH);
    let dm = random_symmetric_dm(cell.mol.nao_nr, 0x0BAD_F00D);

    let cold = MultiGridNumInt2::new();
    let a = cold.nr_rks(&cell, "LDA,VWN", &dm).expect("uncached nr_rks");
    cold.reset();
    let a2 = cold.nr_rks(&cell, "LDA,VWN", &dm).expect("uncached again");

    let warm = MultiGridNumInt2::new();
    let _ = warm.nr_rks(&cell, "LDA,VWN", &dm).expect("warming call");
    let b = warm.nr_rks(&cell, "LDA,VWN", &dm).expect("cached nr_rks");

    for (tag, x) in [("uncached-twice", &a2), ("cached", &b)] {
        assert_eq!(
            a.nelec.to_bits(),
            x.nelec.to_bits(),
            "v2 {tag}: nelec moved"
        );
        assert_eq!(a.exc.to_bits(), x.exc.to_bits(), "v2 {tag}: exc moved");
        assert_eq!(
            a.ecoul.to_bits(),
            x.ecoul.to_bits(),
            "v2 {tag}: ecoul moved"
        );
        same_bits_slice(&format!("v2 {tag} veff"), &a.veff, &x.veff);
    }
}

#[test]
fn v2_does_not_serve_one_cell_the_geometry_of_another() {
    let ni = MultiGridNumInt2::new();
    let si = silicon_at(MESH);
    let dm_si = random_symmetric_dm(si.mol.nao_nr, 0x11);
    let c = diamond_at(MESH);
    let dm_c = random_symmetric_dm(c.mol.nao_nr, 0x22);

    let want_si = MultiGridNumInt2::new().get_j(&si, &dm_si).expect("si");
    let want_c = MultiGridNumInt2::new().get_j(&c, &dm_c).expect("diamond");

    same_bits_slice("v2 si", &want_si, &ni.get_j(&si, &dm_si).expect("si"));
    same_bits_slice(
        "v2 diamond",
        &want_c,
        &ni.get_j(&c, &dm_c).expect("diamond"),
    );
    same_bits_slice("v2 si", &want_si, &ni.get_j(&si, &dm_si).expect("si again"));
}
