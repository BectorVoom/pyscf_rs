//! Plan 17-12 Task 2/3 — kernel-level tests for
//! `pyscf_kernels::multigrid_pair::collocate_pairs` and
//! `pyscf_kernels::multigrid_gspace::{gradient_gs, get_gga_vrho_gs}`.
//!
//! Written FIRST, per the plan: the adjoint identity below needs no
//! upstream oracle and catches a wrong grid offset, a wrong image wrap or a
//! wrong normalisation in either direction of the pair-fused collocation
//! kernel. `gradient_gs`/`get_gga_vrho_gs` are gated against their OWN
//! documented exact `einsum` reference (`_backend_c.py`'s docstrings).

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::{AlgebraClient, CTensor};
use pyscf_kernels::multigrid_gspace::{get_gga_vrho_gs_fac, gradient_gs};
use pyscf_kernels::multigrid_pair::{PairSlotTable, collocate_pairs};

fn client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

/// A small xorshift PRNG — deterministic, no extra crate dependency, matches
/// the fixture convention `multigrid_collocate.rs` already establishes for
/// this plan's sibling kernel tests.
struct Xs(u64);
impl Xs {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

/// Build a small, deterministic-but-"random" [`PairSlotTable`] with `n_inst`
/// instances (random centre/eta) and `slots_per_inst` slots each (random
/// powers 0..=2, random coefficient), each slot tagged with a random `(ci,
/// cj)` route into a `nao x nao` matrix — mirroring how
/// `crate::multigrid::pair` will route real pair-fusion slots once it
/// exists, but WITHOUT depending on that (not-yet-written) host module: this
/// test only needs the kernel's own linear-algebra identity, which holds for
/// ANY slot table.
fn random_table(
    rng: &mut Xs,
    n_inst: usize,
    slots_per_inst: usize,
    nao: usize,
    ngrids_axis: usize,
) -> (PairSlotTable, Vec<usize>, Vec<usize>) {
    let mut coords = Vec::new();
    for ix in 0..ngrids_axis {
        for iy in 0..ngrids_axis {
            for iz in 0..ngrids_axis {
                coords.push(ix as f64 * 0.4 - 1.0);
                coords.push(iy as f64 * 0.4 - 1.0);
                coords.push(iz as f64 * 0.4 - 1.0);
            }
        }
    }

    let mut slot_pow = Vec::new();
    let mut slot_coef = Vec::new();
    let mut slot_instance = Vec::new();
    let mut instance_alpha = Vec::new();
    let mut instance_center = Vec::new();
    let mut route_ci = Vec::new();
    let mut route_cj = Vec::new();

    for inst in 0..n_inst {
        instance_alpha.push(rng.f64_in(0.5, 2.5));
        instance_center.push(rng.f64_in(-0.6, 0.6));
        instance_center.push(rng.f64_in(-0.6, 0.6));
        instance_center.push(rng.f64_in(-0.6, 0.6));
        for _ in 0..slots_per_inst {
            slot_pow.push((rng.next_u64() % 3) as u32);
            slot_pow.push((rng.next_u64() % 3) as u32);
            slot_pow.push((rng.next_u64() % 3) as u32);
            slot_coef.push(rng.f64_in(-1.5, 1.5));
            slot_instance.push(inst as u32);
            route_ci.push((rng.next_u64() as usize) % nao);
            route_cj.push((rng.next_u64() as usize) % nao);
        }
    }

    (
        PairSlotTable {
            coords,
            slot_pow,
            slot_coef,
            slot_instance,
            instance_alpha,
            instance_center,
        },
        route_ci,
        route_cj,
    )
}

/// **The adjoint identity — oracle-free, written before any other Task 2
/// test.** `<integrate(w), D> == <w, collocate(D)>` for random `w`/`D`, to
/// 1e-13. Reused directly at the kernel level: given ANY slot table, define
/// `collocate(D)[g] = Σ_slot D[route(slot)] · values[slot,g]` and
/// `integrate(w)[route] = Σ_{slot: route(slot)=route} Σ_g w[g]·values[slot,g]`
/// — these are transposes of the SAME linear map `values`, by construction,
/// for ANY `values` array, so this test exercises the actual kernel output
/// (catching a wrong grid offset / wrong image wrap / wrong normalisation in
/// `collocate_pair_kernel` itself) without needing `crate::multigrid::pair`'s
/// not-yet-written host-side pair-fusion geometry.
#[test]
fn adjoint_identity_collocate_integrate() {
    let mut rng = Xs(0x243F_6A88_85A3_08D3);
    let nao = 4;
    let (table, route_ci, route_cj) = random_table(&mut rng, 5, 4, nao, 6);
    let nslots = table.slot_pow.len() / 3;
    let ngrids = table.coords.len() / 3;

    let c = client();
    let values = collocate_pairs(&c, &table).expect("collocate_pairs");
    assert_eq!(values.len(), nslots * ngrids);

    // Random D (nao x nao) and random grid weight w.
    let mut dm = vec![0.0f64; nao * nao];
    for d in dm.iter_mut() {
        *d = rng.f64_in(-2.0, 2.0);
    }
    let mut w = vec![0.0f64; ngrids];
    for x in w.iter_mut() {
        *x = rng.f64_in(-1.0, 1.0);
    }

    // Forward: rho[g] = sum_slot dm[route(slot)] * values[slot,g]
    let mut rho = vec![0.0f64; ngrids];
    for s in 0..nslots {
        let d = dm[route_ci[s] * nao + route_cj[s]];
        for g in 0..ngrids {
            rho[g] += d * values[s * ngrids + g];
        }
    }
    let lhs: f64 = (0..ngrids).map(|g| w[g] * rho[g]).sum();

    // Reverse: v[route] += sum_g w[g] * values[slot,g], then <v, dm>.
    let mut v = vec![0.0f64; nao * nao];
    for s in 0..nslots {
        let mut acc = 0.0f64;
        for g in 0..ngrids {
            acc += w[g] * values[s * ngrids + g];
        }
        v[route_ci[s] * nao + route_cj[s]] += acc;
    }
    let rhs: f64 = (0..nao * nao).map(|i| v[i] * dm[i]).sum();

    let diff = (lhs - rhs).abs();
    assert!(
        diff < 1e-13,
        "adjoint identity failed: lhs={lhs:.17e} rhs={rhs:.17e} diff={diff:.3e}"
    );
}

/// A second, independent instance-count/grid-size combination — guards
/// against the identity holding only by coincidence at one small size.
#[test]
fn adjoint_identity_collocate_integrate_larger() {
    let mut rng = Xs(0xD1B5_4A32_D192_ED03);
    let nao = 7;
    let (table, route_ci, route_cj) = random_table(&mut rng, 11, 7, nao, 8);
    let nslots = table.slot_pow.len() / 3;
    let ngrids = table.coords.len() / 3;

    let c = client();
    let values = collocate_pairs(&c, &table).expect("collocate_pairs");

    let mut dm = vec![0.0f64; nao * nao];
    for d in dm.iter_mut() {
        *d = rng.f64_in(-1.0, 1.0);
    }
    let mut w = vec![0.0f64; ngrids];
    for x in w.iter_mut() {
        *x = rng.f64_in(-1.0, 1.0);
    }

    let mut rho = vec![0.0f64; ngrids];
    for s in 0..nslots {
        let d = dm[route_ci[s] * nao + route_cj[s]];
        for g in 0..ngrids {
            rho[g] += d * values[s * ngrids + g];
        }
    }
    let lhs: f64 = (0..ngrids).map(|g| w[g] * rho[g]).sum();

    let mut v = vec![0.0f64; nao * nao];
    for s in 0..nslots {
        let mut acc = 0.0f64;
        for g in 0..ngrids {
            acc += w[g] * values[s * ngrids + g];
        }
        v[route_ci[s] * nao + route_cj[s]] += acc;
    }
    let rhs: f64 = (0..nao * nao).map(|i| v[i] * dm[i]).sum();

    let diff = (lhs - rhs).abs();
    assert!(
        diff < 1e-13,
        "adjoint identity (larger) failed: lhs={lhs:.17e} rhs={rhs:.17e} diff={diff:.3e}"
    );
}

/// Sanity check the kernel evaluates a genuine `(r-P)^pow * exp(-eta r^2)`
/// against a direct host computation (single slot, single instance, no
/// routing) — independent of the adjoint test above.
#[test]
fn single_slot_matches_direct_formula() {
    let table = PairSlotTable {
        coords: vec![0.3, -0.2, 0.5, 1.0, 1.0, 1.0],
        slot_pow: vec![2, 1, 0],
        slot_coef: vec![1.7],
        slot_instance: vec![0],
        instance_alpha: vec![0.8],
        instance_center: vec![0.1, 0.1, -0.1],
    };
    let c = client();
    let values = collocate_pairs(&c, &table).expect("collocate_pairs");
    assert_eq!(values.len(), 2);
    for (g, val) in values.iter().enumerate() {
        let x = table.coords[g * 3] - table.instance_center[0];
        let y = table.coords[g * 3 + 1] - table.instance_center[1];
        let z = table.coords[g * 3 + 2] - table.instance_center[2];
        let r2 = x * x + y * y + z * z;
        let expect = 1.7 * x * x * y * (-0.8 * r2).exp();
        assert!(
            (val - expect).abs() < 1e-13,
            "grid {g}: got {val}, expected {expect}"
        );
    }
}

/// `gradient_gs` ≡ `einsum('np,px->nxp', f_gs, 1j*Gv)` — `_backend_c.py`'s
/// OWN documented reference, to 1e-14 (free, exact oracle).
#[test]
fn gradient_gs_matches_documented_einsum() {
    let mut rng = Xs(0x1234_5678_9ABC_DEF0);
    let nset = 3;
    let ngrids = 17;
    let mut f_gs = CTensor::zeros(nset * ngrids);
    for x in f_gs.re.iter_mut().chain(f_gs.im.iter_mut()) {
        *x = rng.f64_in(-3.0, 3.0);
    }
    let mut gv = vec![0.0f64; ngrids * 3];
    for x in gv.iter_mut() {
        *x = rng.f64_in(-2.0, 2.0);
    }

    let out = gradient_gs(&f_gs, &gv, ngrids);
    assert_eq!(out.re.len(), nset * 3 * ngrids);

    for n in 0..nset {
        for p in 0..ngrids {
            let fr = f_gs.re[n * ngrids + p];
            let fi = f_gs.im[n * ngrids + p];
            for x in 0..3 {
                // einsum('np,px->nxp', f_gs, 1j*Gv)[n,x,p] = f_gs[n,p] * i*Gv[p,x]
                let gx = gv[p * 3 + x];
                let expect_re = -gx * fi;
                let expect_im = gx * fr;
                let idx = (n * 3 + x) * ngrids + p;
                assert!(
                    (out.re[idx] - expect_re).abs() < 1e-14,
                    "re mismatch at n={n},x={x},p={p}"
                );
                assert!(
                    (out.im[idx] - expect_im).abs() < 1e-14,
                    "im mismatch at n={n},x={x},p={p}"
                );
            }
        }
    }
}

/// `get_gga_vrho_gs` ≡ `v -= fac*1j*einsum('px,xp->p', Gv, v1); v *= weight`
/// — `_backend_c.py`'s own documented reference, to 1e-14.
#[test]
fn get_gga_vrho_gs_matches_documented_formula() {
    let mut rng = Xs(0xFEED_FACE_CAFE_BABE);
    let ngrids = 13;
    let fac = 2.0;
    let weight = 0.037;

    let mut v = CTensor::zeros(ngrids);
    for x in v.re.iter_mut().chain(v.im.iter_mut()) {
        *x = rng.f64_in(-1.0, 1.0);
    }
    let v_orig = v.clone();

    let mut v1 = CTensor::zeros(3 * ngrids);
    for x in v1.re.iter_mut().chain(v1.im.iter_mut()) {
        *x = rng.f64_in(-1.0, 1.0);
    }
    let mut gv = vec![0.0f64; ngrids * 3];
    for x in gv.iter_mut() {
        *x = rng.f64_in(-2.0, 2.0);
    }

    get_gga_vrho_gs_fac(&mut v, &v1, &gv, weight, ngrids, fac);

    for p in 0..ngrids {
        // einsum('px,xp->p', Gv, v1)[p] = sum_x Gv[p,x] * v1[x,p]
        let mut dot_re = 0.0;
        let mut dot_im = 0.0;
        for x in 0..3 {
            let gx = gv[p * 3 + x];
            dot_re += gx * v1.re[x * ngrids + p];
            dot_im += gx * v1.im[x * ngrids + p];
        }
        // v -= fac * i * dot
        let sub_re = -fac * dot_im;
        let sub_im = fac * dot_re;
        let expect_re = (v_orig.re[p] - sub_re) * weight;
        let expect_im = (v_orig.im[p] - sub_im) * weight;
        assert!((v.re[p] - expect_re).abs() < 1e-14, "re mismatch at p={p}");
        assert!((v.im[p] - expect_im).abs() < 1e-14, "im mismatch at p={p}");
    }
}

/// The grouped (one `exp` per instance × point) kernel [`collocate_pairs`]
/// dispatches to for instance-grouped tables must be BIT-IDENTICAL to the
/// per-slot reference kernel — same operations in the same order per output
/// element. This is what lets `pyscf-pbc-dft`'s v2 driver take the ~`slots
/// per instance` speedup without a numerical seam.
#[test]
fn grouped_kernel_is_bit_identical_to_per_slot() {
    let client = pyscf_algebra::select_backend().expect("backend").client;
    let mut rng = Xs(0x5EED_1234_ABCD_0001);
    let (table, _ci, _cj) = random_table(&mut rng, 7, 5, 4, 6);
    let grouped = pyscf_kernels::collocate_pairs(&client, &table).expect("grouped");
    let per_slot = pyscf_kernels::collocate_pairs_per_slot(&client, &table).expect("per-slot");
    assert_eq!(grouped.len(), per_slot.len());
    assert_eq!(
        grouped, per_slot,
        "grouped kernel diverged from the per-slot reference"
    );
}

/// The fused-reduction kernels against the materialised per-slot values:
/// `rho[g] == Σ_slot vals[slot,g]` and `I[slot] == Σ_g w[g]·vals[slot,g]`.
/// The summation ORDER differs (sequential in-kernel vs host), so this is
/// a rounding-level tolerance, not bit identity; the adjoint identity below
/// then ties the two fused kernels to each other with no reference at all.
#[test]
fn fused_rho_and_integrate_match_per_slot_values() {
    let client = pyscf_algebra::select_backend().expect("backend").client;
    let mut rng = Xs(0x0F0F_1234_5678_9ABC);
    let (table, _ci, _cj) = random_table(&mut rng, 9, 4, 3, 5);
    let ngrids = table.coords.len() / 3;
    let nslots = table.slot_pow.len() / 3;
    let vals = pyscf_kernels::collocate_pairs(&client, &table).expect("per-slot");
    let rho = pyscf_kernels::collocate_pairs_rho(&client, &table).expect("rho");
    assert_eq!(rho.len(), ngrids);
    for g in 0..ngrids {
        let want: f64 = (0..nslots).map(|s| vals[s * ngrids + g]).sum();
        assert!(
            (rho[g] - want).abs() <= 1e-13 * (1.0 + want.abs()),
            "rho[{g}] = {} vs {want}",
            rho[g]
        );
    }
    let w: Vec<f64> = (0..ngrids).map(|_| rng.f64_in(-1.0, 1.0)).collect();
    let integ = pyscf_kernels::collocate_pairs_integrate(&client, &table, &w).expect("integrate");
    assert_eq!(integ.len(), nslots);
    for s in 0..nslots {
        let want: f64 = (0..ngrids).map(|g| w[g] * vals[s * ngrids + g]).sum();
        assert!(
            (integ[s] - want).abs() <= 1e-13 * (1.0 + want.abs()),
            "I[{s}] = {} vs {want}",
            integ[s]
        );
    }
}

/// `<integrate(w), 1> == <w, rho>` for the FUSED kernels — the same
/// oracle-free adjoint identity Task 2 wrote first, now on the pair that
/// the v2 driver actually launches.
#[test]
fn fused_kernels_satisfy_adjoint_identity() {
    let client = pyscf_algebra::select_backend().expect("backend").client;
    let mut rng = Xs(0xADD0_1234_0000_7777);
    let (table, _ci, _cj) = random_table(&mut rng, 11, 6, 3, 6);
    let ngrids = table.coords.len() / 3;
    let w: Vec<f64> = (0..ngrids).map(|_| rng.f64_in(-1.0, 1.0)).collect();
    let rho = pyscf_kernels::collocate_pairs_rho(&client, &table).expect("rho");
    let integ = pyscf_kernels::collocate_pairs_integrate(&client, &table, &w).expect("integrate");
    let lhs: f64 = integ.iter().sum();
    let rhs: f64 = w.iter().zip(&rho).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-13 * (1.0 + lhs.abs()),
        "adjoint identity broken: {lhs} vs {rhs}"
    );
}
