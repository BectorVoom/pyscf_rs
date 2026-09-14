//! Plan 18-11 — strain-tensor scale kernel tests (no Cell, no physics).
//!
//! * [`block_map_matches_c_switch`]: [`strain_block_map`] against the
//!   `grid_ao.c` `case 1` / `case 4` switch, entry by entry, plus the
//!   `deriv >= 2` named refusal (18-11 Task 4).
//! * [`scale_kernel_is_exact_multiply`]: [`strain_scale`] against a host
//!   multiply loop — exact equality, since IEEE `f64` multiplication is
//!   correctly rounded on every backend and the kernel performs nothing
//!   else (this is also why `check-no-fma` has nothing to fuse here).
//! * [`driver_rejects_deriv2_without_tables`] / [`empty_grid_shapes`]:
//!   driver validation order and degenerate shapes.
//!
//! The physics gate (Gate A, tier A1 = 1e-9 vs finite differences, upstream
//! `test_rks_stress.py:140,158,176,194`) lives in
//! `crates/pyscf-pbc-gto/tests/eval_ao_strain.rs`, which drives the full
//! [`eval_strain_ao`] image loop through a built `Cell`.

use pyscf_algebra::select_backend;
use pyscf_kernels::pbc::{eval_strain_ao, strain_block_map, strain_comp, strain_scale};

/// Deterministic pseudo-random fill; a fixed LCG so a failure reproduces.
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// `strain_block_map` is `grid_ao.c`'s strain switch transcribed:
/// `deriv = 0` → 9 blocks `src = 1+x`, axis `y`;
/// `deriv = 1` → 36 blocks, value-strain reads the gradient (`1+x`),
/// x/y/z-gradient strains read `xx,xy,xz` / `xy,yy,yz` / `xz,yz,zz`.
#[test]
fn block_map_matches_c_switch() {
    let (src, axis) = strain_block_map(0).expect("deriv 0 maps");
    assert_eq!(src.len(), 9);
    assert_eq!(axis.len(), 9);
    for x in 0..3 {
        for y in 0..3 {
            let b = x * 3 + y;
            assert_eq!(src[b], (1 + x) as u32, "deriv0 src at b={b}");
            assert_eq!(axis[b], y as u32, "deriv0 axis at b={b}");
        }
    }

    let (src, axis) = strain_block_map(1).expect("deriv 1 maps");
    assert_eq!(src.len(), 36);
    assert_eq!(axis.len(), 36);
    // Derivative-table order: value, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz.
    let row_c1 = [4u32, 5, 6]; // xx, xy, xz
    let row_c2 = [5u32, 7, 8]; // xy, yy, yz
    let row_c3 = [6u32, 8, 9]; // xz, yz, zz
    for x in 0..3 {
        for y in 0..3 {
            let s = x * 3 + y;
            assert_eq!(src[s * 4], (1 + x) as u32, "deriv1 c=0 src at s={s}");
            assert_eq!(src[s * 4 + 1], row_c1[x], "deriv1 c=1 src at s={s}");
            assert_eq!(src[s * 4 + 2], row_c2[x], "deriv1 c=2 src at s={s}");
            assert_eq!(src[s * 4 + 3], row_c3[x], "deriv1 c=3 src at s={s}");
            for c in 0..4 {
                assert_eq!(axis[s * 4 + c], y as u32, "deriv1 axis at s={s},c={c}");
            }
        }
    }
}

/// `deriv >= 2` has no upstream caller and no oracle: named refusal, never a
/// fall-through to the non-strain family (18-11 Task 4, the R-14 shape).
#[test]
fn deriv2_is_a_named_refusal() {
    for deriv in [2, 3, 9] {
        let err = format!(
            "{:?}",
            strain_comp(deriv).expect_err("deriv>=2 must refuse")
        );
        assert!(err.contains("deriv"), "refusal names the order, got: {err}");
        let err = format!(
            "{:?}",
            strain_block_map(deriv).expect_err("deriv>=2 must refuse")
        );
        assert!(err.contains("deriv"), "map refusal names the order");
    }
}

/// The device kernel performs exactly one multiply per lane: compare against
/// the host loop with zero tolerance, at both orders and at odd shapes that
/// defeat vector-width alignment.
#[test]
fn scale_kernel_is_exact_multiply() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    for (deriv, ngrids, nao) in [(0u32, 5usize, 3usize), (0, 1, 1), (1, 7, 4), (1, 3, 1)] {
        let d = if deriv == 0 { 4 } else { 10 };
        let n = ngrids * nao;
        let deriv_block = fill(d * n, 0x51ed_270b + deriv as u64);
        let rvec = fill(3 * nao, 0x9e37_79b9 + ngrids as u64);
        let got =
            strain_scale(client, &deriv_block, &rvec, ngrids, nao, deriv).expect("scale launches");
        let (src, axis) = strain_block_map(deriv).expect("map");
        assert_eq!(got.len(), 9 * (if deriv == 0 { 1 } else { 4 }) * n);
        for (b, (&s, &ax)) in src.iter().zip(axis.iter()).enumerate() {
            for q in 0..n {
                let a = q / ngrids;
                let want = deriv_block[s as usize * n + q] * rvec[a * 3 + ax as usize];
                let lane = b * n + q;
                assert_eq!(
                    got[lane], want,
                    "deriv={deriv} block={b} q={q}: kernel must be the exact multiply"
                );
            }
        }
    }
}

/// Short slices and bad orders refuse before any launch.
#[test]
fn scale_rejects_bad_shapes() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    // deriv=0, ngrids=1, nao=2 needs deriv_block 4*2=8 and rvec 3*2=6.
    assert!(strain_scale(client, &[0.0; 7], &[0.0; 6], 1, 2, 0).is_err());
    assert!(strain_scale(client, &[0.0; 8], &[0.0; 5], 1, 2, 0).is_err());
    assert!(strain_scale(client, &[0.0; 8], &[0.0; 6], 1, 2, 2).is_err());
    // Empty table: empty out, no launch.
    let got = strain_scale(client, &[], &[], 0, 0, 0).expect("empty is fine");
    assert!(got.is_empty());
}

/// The driver's `deriv >= 2` refusal fires before any table is touched
/// (empty tables would otherwise fail validation first and mask it).
#[test]
fn driver_rejects_deriv2_without_tables() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    let err = format!(
        "{:?}",
        eval_strain_ao(
            client,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            2,
            true,
            None,
            true
        )
        .expect_err("deriv=2 must refuse")
    );
    assert!(err.contains("deriv"), "named refusal, got: {err}");
}

/// Degenerate grid: correctly-shaped empty planes, one per k-point.
#[test]
fn empty_grid_shapes() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    let kpts = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let out = eval_strain_ao(
        client,
        &[],
        &kpts,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        true,
        None,
        true,
    )
    .expect("empty basis and grid");
    assert_eq!(out.re.len(), 2);
    assert_eq!(out.im.len(), 2);
    assert!(out.re.iter().all(|p| p.is_empty()));
    assert_eq!(out.ngrids, 0);
    assert_eq!(out.nao, 0);
    assert_eq!(out.deriv, 1);
}
