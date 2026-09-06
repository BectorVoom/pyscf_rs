//! M-11 (session 3): the v1 multigrid driver keeps its collocated level
//! values alive with the task list, so the second and later SCF cycles do not
//! re-run the collocation kernel.
//!
//! Bit-exactness is asserted three ways against the per-call collocation the
//! kill switch `PYSCF_PBC_MULTIGRID_LEVEL_CACHE=0` restores: a second call on
//! the SAME numint (a cache hit) equals the first (a miss), and both equal
//! the uncached route. Everything is `to_bits()`. One test function, because
//! the kill switch is a process-wide environment variable.

use pyscf_pbc_dft::multigrid::MultiGridNumInt;
use pyscf_pbc_gto::test_systems::si;

fn random_symmetric_dm(nao: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut dm = vec![0.0; nao * nao];
    for i in 0..nao {
        for j in 0..=i {
            let v = next();
            dm[i * nao + j] = v;
            dm[j * nao + i] = v;
        }
    }
    dm
}

fn same_bits(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
    }
}

#[test]
fn cached_level_values_are_bit_identical_to_per_call_collocation() {
    let mut cell = si();
    cell.mesh = [15, 15, 15];
    let nao = cell.mol.nao_nr;
    let dm_a = random_symmetric_dm(nao, 7);
    let dm_b = random_symmetric_dm(nao, 11);

    // Cache ON (default): miss, then hit, on one numint.
    let ni = MultiGridNumInt::new();
    let miss = ni.nr_rks(&cell, "pbe", &dm_a).expect("nr_rks miss");
    let hit = ni.nr_rks(&cell, "pbe", &dm_a).expect("nr_rks hit");
    let hit_b = ni
        .nr_rks(&cell, "pbe", &dm_b)
        .expect("nr_rks hit, other dm");
    let j_hit = ni.get_j(&cell, &dm_a).expect("get_j hit");
    let u_hit = ni
        .nr_uks(&cell, "pbe", &[&dm_a, &dm_b])
        .expect("nr_uks hit");

    // Cache OFF: every call collocates afresh.
    // SAFETY: single-threaded test body.
    unsafe { std::env::set_var("PYSCF_PBC_MULTIGRID_LEVEL_CACHE", "0") };
    let ni_off = MultiGridNumInt::new();
    let off = ni_off.nr_rks(&cell, "pbe", &dm_a);
    let off_b = ni_off.nr_rks(&cell, "pbe", &dm_b);
    let j_off = ni_off.get_j(&cell, &dm_a);
    let u_off = ni_off.nr_uks(&cell, "pbe", &[&dm_a, &dm_b]);
    unsafe { std::env::remove_var("PYSCF_PBC_MULTIGRID_LEVEL_CACHE") };
    let off = off.expect("nr_rks uncached");
    let off_b = off_b.expect("nr_rks uncached, other dm");
    let j_off = j_off.expect("get_j uncached");
    let u_off = u_off.expect("nr_uks uncached");

    for (what, a, b) in [
        ("miss vs hit veff", &miss.veff, &hit.veff),
        ("hit vs uncached veff", &hit.veff, &off.veff),
        ("hit vs uncached veff (dm b)", &hit_b.veff, &off_b.veff),
        ("get_j hit vs uncached", &j_hit, &j_off),
        (
            "nr_uks alpha hit vs uncached",
            &u_hit.veff[0],
            &u_off.veff[0],
        ),
        (
            "nr_uks beta hit vs uncached",
            &u_hit.veff[1],
            &u_off.veff[1],
        ),
    ] {
        same_bits(what, a, b);
    }
    for (what, a, b) in [
        ("nelec", miss.nelec, off.nelec),
        ("exc", miss.exc, off.exc),
        ("ecoul", miss.ecoul, off.ecoul),
        ("hit nelec", hit.nelec, off.nelec),
        ("hit exc", hit.exc, off.exc),
        ("uks exc", u_hit.exc, u_off.exc),
    ] {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}: {a:e} vs {b:e}");
    }

    // A different cell must not be served from the old table.
    let mut other = si();
    other.mesh = [17, 17, 17];
    let fresh = MultiGridNumInt::new()
        .nr_rks(&other, "pbe", &dm_a)
        .expect("fresh cell");
    let via_cache = ni
        .nr_rks(&other, "pbe", &dm_a)
        .expect("other cell on the cached numint");
    same_bits("other cell veff", &fresh.veff, &via_cache.veff);
    println!("v1 level-value cache: miss/hit/uncached bit-identical on two cells");
}
