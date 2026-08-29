//! GTO-07 + D-04: AO-on-grid kernel (`eval_gto`).
//!
//! Source: pyscf/gto/eval_gto.py (Apache-2.0). The reference algorithm:
//! per grid point g, walk every shell s, compute the contracted radial
//! `Σ_p coeff[c,p] * exp(-α_p * r²)`, optionally apply the cart→sph
//! harmonic transform, and write to `out[g, ao_idx]` in F-order.
//!
//! Phase 2 shipped the **s-shell only** implementation. Phase 4 plan
//! 04-03 (this commit) lands the deferred `l ≥ 1` path: per shell with
//! angular momentum `l`, evaluate the `ncart(l) = (l+1)(l+2)/2`
//! cartesian monomials `x^lx y^ly z^lz · R(r)` (upstream loop order
//! `lx=l..0, ly=l-lx..0, lz=l-lx-ly`), then apply the libcint
//! cart→sph transform (`g_trans_cart2sph[]`, the same Condon-Shortley
//! coefficients libcint's `CINTc2s_ket_sph` uses) to produce the
//! `2l+1` real spherical-harmonic AO components. For `l = 0, 1` the
//! cart→sph transform is the identity and the angular factor lives in
//! `CINTcommon_fac_sp(l)`; for `l ≥ 2` the factor is folded into the
//! c2s matrix. The contracted-radial sum (> 2 prims) is routed through
//! `pyscf_algebra::oracle_sum` for FMA-free, thread-order-independent
//! reduction (Pitfall 3 / FOUND-06 — byte-exact discipline).
//!
//! Reference algorithm: `pyscf/lib/gto/deriv1.c GTOshell_eval_grid_cart`
//! (cartesian monomial eval) + `pyscf/gto/mole.py cart2sph` →
//! `CINTc2s_ket_sph` (libcint `cart2sph.c g_trans_cart2sph[]`). The c2s
//! coefficient tables below are byte-identical to cintx-cubecl
//! `transform::c2s::C2S_L{0..4}` (libcint provenance).
//!
//! The `GTOval_sph_deriv1` variant (value + 3 gradient components) is the
//! GGA grid-loop input; it is implemented alongside the value path here
//! (plan 04-03 Task 2). `GTOval_sph_deriv2` / `GTOval_ip*` / `GTOval_ig*`
//! remain dispatched at the user-facing wrapper (`pyscf-gto::eval_gto`)
//! and return clean `NotYetImplemented{phase:4|7}` — no kernel cost.
//!
//! ALG-06 algebra-wall: this module imports `cubecl-*` directly via the
//! Wave 0 W0-T4 allowlist update. The PUBLIC function `eval_gto_sph`
//! takes ONLY `pyscf-algebra` types (`AlgebraClient`) so that
//! `pyscf-gto`'s wrapper (the next layer up) imports this without ever
//! naming a cubecl type. `xtask::check-dependency-wall` enforces the
//! containment.
//!
//! ### Plan deviation (Rule 3): cubecl macro deferred to Phase 4
//!
//! The plan's draft kernel (`#[cube(launch_unchecked)] fn
//! eval_gto_sph_kernel(..., #[comptime] _spherical: bool)`) hits multiple
//! cubecl 0.10.0 macro-expansion issues that don't appear in the plan's
//! spec:
//!
//!   - `ScalarArg::new` is not a public type in cubecl 0.10.0 (the
//!     replacement is `InputScalar` in `cubecl::frontend::scalar`)
//!   - `ArrayArg::from_raw_parts` takes `(handle, length)` — no turbofish
//!     for element type
//!   - `ABSOLUTE_POS` returns `usize`, not `u32` — silent type mismatch
//!     in the plan's draft
//!   - `let bas_slots: u32 = 8u32;` triggers `from_lit` on
//!     `NativeExpand<u32>` which is not satisfied by `From<NativeExpand<u32>>
//!     for ConstantValue` in cubecl 0.10.0
//!   - inlined `f64::exp` works inside `#[cube]` (verified via
//!     `impl_unary_func!(Exp, exp, …, f64)`) but only after fixing all
//!     of the above syntax issues
//!
//! Wave 0 (`tests/wave0_cubecl_smoke.rs`) already proved cubecl-cpu can
//! launch a `#[cube(launch_unchecked)]` kernel from this crate. Plan
//! 02-06 (this commit) preserves that proof and ships a host CPU
//! implementation behind the same `AlgebraClient`-typed public surface
//! (`eval_gto_sph(&AlgebraClient, …)`). The host path is in lockstep
//! with `pyscf-algebra::host_fallback::{eigh, cholesky, qr, svd}` (which
//! also routes the eigh family to `faer 0.24` on host per ALG-05); the
//! algebra wall is preserved without forcing this Phase-2 plan to land a
//! production-ready cubecl macro that the upstream API hasn't fully
//! frozen.
//!
//! Phase 4 DFT (or a dedicated Phase 8 GPU-enable plan) extends this
//! file with the actual `#[cube(launch_unchecked)]` kernel for l ≥ 1
//! cart2sph transforms + deriv1/deriv2 stencils — that's also when the
//! cubecl-macro surface really earns its keep (large grids on GPU).
//! The host CPU path stays as a fallback for the algebra-wall and as
//! the FMA-free oracle target (FOUND-05).

use pyscf_algebra::{AlgebraClient, oracle_sum};
// quick-260530-ljv: `dispatch_backend!` is now exported from pyscf-algebra
// (`#[macro_export]`), so this downstream crate fans the s-shell cube launch out
// over every backend without re-deriving the cfg-gated `match client { … }`. The
// bare runtime paths inside the macro (`cubecl_cpu::CpuRuntime`, …) resolve in
// THIS crate's namespace — pyscf-kernels carries the cfg-aligned cubecl-* deps.
use pyscf_algebra::dispatch_backend;

// `cubecl` is reachable from this crate per the ALG-06 carve-out
// (`xtask/check_dependency_wall.rs:47` lists `pyscf-kernels` in
// `ALLOWED_CRATES`). The Wave 0 smoke test
// (`tests/wave0_cubecl_smoke.rs`) keeps the launch path warm — see
// the file-level doc comment for the deferral rationale.
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
#[allow(unused_imports)]
use cubecl::prelude::*;

use pyscf_core::PyscfRsError;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};

// ── cart→sph angular machinery (libcint provenance) ─────────────────────
//
// Source: libcint `cart2sph.c g_trans_cart2sph[]` (the matrices
// `CINTc2s_ket_sph` applies; `pyscf/gto/mole.py cart2sph` uses the same
// routine). FROZEN f64 — L0..L4 are byte-identical to cintx-cubecl
// `transform::c2s::C2S_L{0..4}`; L5 (h) and L6 (i) are extracted verbatim
// from the same libcint array (offsets 245/476) and additionally
// cross-checked against the Schlegel–Frisch analytical formula. Rows =
// m = -l..+l, cols = libcint cartesian order (the `GTOshell_eval_grid_cart`
// monomial order: lx=l..0, ly=l-lx..0, lz=l-lx-ly). Changing any value
// breaks bit-exact agreement with upstream PySCF.

/// libcint `CINTcommon_fac_sp` (g1e.c:566). l=0,1 carry the angular
/// prefactor in the radial part; l≥2 fold it into the c2s matrix.
///
/// `pub` since plan 13-01: `ft_aopair` builds the SAME cartesian AO convention
/// and must multiply by `common_fac_sp(li)·common_fac_sp(lj)` before the c2s
/// transform, or its `G=0` limit will not equal `int1e_ovlp`. One definition,
/// one place to be wrong.
#[inline]
pub fn common_fac_sp(l: u32) -> f64 {
    match l {
        0 => 0.282_094_791_773_878_14,
        1 => 0.488_602_511_902_919_9,
        _ => 1.0,
    }
}

/// Number of cartesian components for angular momentum `l`.
#[inline]
fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Number of spherical components for angular momentum `l` (`2l+1`).
#[inline]
fn nsph(l: u32) -> usize {
    2 * l as usize + 1
}

/// Cartesian monomial powers `(lx, ly, lz)` for cart column `c`, in the
/// upstream `GTOshell_eval_grid_cart` loop order
/// (`for lx=l..0 { for ly=l-lx..0 { lz=l-lx-ly }}`). This ordering is
/// what `ao_loc_nr` + the c2s columns assume — Pitfall 8/17 lives here.
pub fn cart_powers(l: u32) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::with_capacity(ncart(l));
    let li = l as i32;
    let mut lx = li;
    while lx >= 0 {
        let mut ly = li - lx;
        while ly >= 0 {
            v.push((lx as u32, ly as u32, (li - lx - ly) as u32));
            ly -= 1;
        }
        lx -= 1;
    }
    v
}

/// libcint `g_trans_cart2sph` coefficient `T[l][m_row][cart_col]`.
/// Returns the FROZEN Condon-Shortley value. `l ≤ 6` supported (g-shells g,
/// h-shells, i-shells — covers cc-pV5Z h and cc-pV6Z i). For `l > 6`
/// (k-shells and above) this returns
/// `Err(PyscfRsError::NotYetImplemented{phase:4,..})` rather than panicking,
/// so a user-supplied basis fails loudly through the PyO3 boundary with a
/// Python exception instead of aborting the process (BLOCKER CR-03 /
/// FOUND-07 never-panic policy). The l=5/l=6 tables are libcint-verbatim and
/// independently cross-checked against the Schlegel–Frisch analytical formula.
fn c2s_coeff(l: u32, m_row: usize, cart_col: usize) -> Result<f64, PyscfRsError> {
    // s (l=0): 1×1 identity.
    const L0: [[f64; 1]; 1] = [[1.0]];
    // p (l=1): identity (px,py,pz); the 0.4886 prefactor is in fac1.
    const L1: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // d (l=2): 5×6. cols: xx,xy,xz,yy,yz,zz. rows: m=-2..+2.
    const L2: [[f64; 6]; 5] = [
        [0.0, 1.092_548_430_592_079_2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.092_548_430_592_079_2, 0.0],
        [
            -0.315_391_565_252_52,
            0.0,
            0.0,
            -0.315_391_565_252_52,
            0.0,
            0.630_783_130_505_04,
        ],
        [0.0, 0.0, 1.092_548_430_592_079_2, 0.0, 0.0, 0.0],
        [
            0.546_274_215_296_039_6,
            0.0,
            0.0,
            -0.546_274_215_296_039_6,
            0.0,
            0.0,
        ],
    ];
    // f (l=3): 7×10. cols: xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz.
    const L3: [[f64; 10]; 7] = [
        [
            0.0,
            1.770_130_769_779_930_4,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.590_043_589_926_643_5,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            2.890_611_442_640_554_3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -0.457_045_799_464_465_7,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.457_045_799_464_465_7,
            0.0,
            1.828_183_197_857_862_9,
            0.0,
        ],
        [
            0.0,
            0.0,
            -1.119_528_997_770_346_2,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.119_528_997_770_346_2,
            0.0,
            0.746_352_665_180_230_8,
        ],
        [
            -0.457_045_799_464_465_7,
            0.0,
            0.0,
            -0.457_045_799_464_465_7,
            0.0,
            1.828_183_197_857_862_9,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.445_305_721_320_277_1,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.445_305_721_320_277_1,
            0.0,
            0.0,
        ],
        [
            0.590_043_589_926_643_5,
            0.0,
            0.0,
            -1.770_130_769_779_930_4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    // g (l=4): 9×15. cols: xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,
    // xzzz,yyyy,yyyz,yyzz,yzzz,zzzz.
    const L4: [[f64; 15]; 9] = [
        [
            0.0,
            2.503_342_941_796_704_6,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.503_342_941_796_704_6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            5.310_392_309_339_791,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.770_130_769_779_930_4,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -0.946_174_695_757_56,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.946_174_695_757_56,
            0.0,
            5.677_048_174_545_360_5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -2.007_139_630_671_867_6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.007_139_630_671_867_6,
            0.0,
            2.676_186_174_229_157,
            0.0,
        ],
        [
            0.317_356_640_745_612_93,
            0.0,
            0.0,
            0.634_713_281_491_225_9,
            0.0,
            -2.538_853_125_964_903_4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.317_356_640_745_612_93,
            0.0,
            -2.538_853_125_964_903_4,
            0.0,
            0.846_284_375_321_634_5,
        ],
        [
            0.0,
            0.0,
            -2.007_139_630_671_867_6,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.007_139_630_671_867_6,
            0.0,
            2.676_186_174_229_157,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.473_087_347_878_78,
            0.0,
            0.0,
            0.0,
            0.0,
            2.838_524_087_272_680_2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.473_087_347_878_78,
            0.0,
            -2.838_524_087_272_680_2,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.770_130_769_779_930_4,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.310_392_309_339_791,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.625_835_735_449_176_1,
            0.0,
            0.0,
            -3.755_014_412_695_057,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.625_835_735_449_176_1,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    // h (l=5): 11×21 and i (l=6): 13×28. Same FROZEN libcint provenance as
    // L0..L4 (`g_trans_cart2sph` offsets 245 and 476). Extracted verbatim from
    // the libcint source array and independently cross-validated against the
    // Schlegel–Frisch `xyz2sph_real` analytical formula (libcint
    // `scripts/cart2sph.py`) to a ratio of exactly 1.0 — see the
    // `c2s_coeff_l5_l6_*` regression tests. These cover cc-pV5Z (h) and
    // cc-pV6Z (i) basis sets through the generic CPU eval path; the device
    // (`#[cube]`) path still routes l>4 shells to the CPU.
    // L5 (l=5): 11×21. rows m=-5..+5; libcint `g_trans_cart2sph` offset 245.
    // cols (lx=l..0, ly=l-lx..0): xxxxx xxxxy xxxxz xxxyy xxxyz xxxzz xxyyy
    // xxyyz xxyzz xxzzz xyyyy xyyyz xyyzz xyzzz xzzzz yyyyy yyyyz yyyzz yyzzz
    // yzzzz zzzzz.
    const L5: [[f64; 21]; 11] = [
        [
            0.0,
            3.281_910_284_200_850_7,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.563_820_568_401_701,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.656_382_056_840_170_1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            8.302_649_259_524_165,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -8.302_649_259_524_165,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -1.467_714_898_305_751,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.978_476_598_870_500_8,
            0.0,
            11.741_719_186_446_009,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.489_238_299_435_250_4,
            0.0,
            -3.913_906_395_482_003,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -4.793_536_784_973_324,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.793_536_784_973_324,
            0.0,
            9.587_073_569_946_648,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.452_946_651_195_696_94,
            0.0,
            0.0,
            0.0,
            0.0,
            0.905_893_302_391_393_9,
            0.0,
            -5.435_359_814_348_363,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.452_946_651_195_696_94,
            0.0,
            -5.435_359_814_348_363,
            0.0,
            3.623_573_209_565_575_5,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.754_254_836_801_354,
            0.0,
            0.0,
            0.0,
            0.0,
            3.508_509_673_602_708,
            0.0,
            -4.678_012_898_136_944,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.754_254_836_801_354,
            0.0,
            -4.678_012_898_136_944,
            0.0,
            0.935_602_579_627_388_8,
        ],
        [
            0.452_946_651_195_696_94,
            0.0,
            0.0,
            0.905_893_302_391_393_9,
            0.0,
            -5.435_359_814_348_363,
            0.0,
            0.0,
            0.0,
            0.0,
            0.452_946_651_195_696_94,
            0.0,
            -5.435_359_814_348_363,
            0.0,
            3.623_573_209_565_575_5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            -2.396_768_392_486_662,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.793_536_784_973_324,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.396_768_392_486_662,
            0.0,
            -4.793_536_784_973_324,
            0.0,
            0.0,
        ],
        [
            -0.489_238_299_435_250_4,
            0.0,
            0.0,
            0.978_476_598_870_500_8,
            0.0,
            3.913_906_395_482_003,
            0.0,
            0.0,
            0.0,
            0.0,
            1.467_714_898_305_751,
            0.0,
            -11.741_719_186_446_009,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            2.075_662_314_881_041,
            0.0,
            0.0,
            0.0,
            0.0,
            -12.453_973_889_286_248,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.075_662_314_881_041,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.656_382_056_840_170_1,
            0.0,
            0.0,
            -6.563_820_568_401_701,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.281_910_284_200_850_7,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    // L6 (l=6): 13×28. rows m=-6..+6; libcint `g_trans_cart2sph` offset 476.
    // cols (lx=l..0, ly=l-lx..0): xxxxxx xxxxxy xxxxxz xxxxyy xxxxyz xxxxzz
    // xxxyyy xxxyyz xxxyzz xxxzzz xxyyyy xxyyyz xxyyzz xxyzzz xxzzzz xyyyyy
    // xyyyyz xyyyzz xyyzzz xyzzzz xzzzzz yyyyyy yyyyyz yyyyzz yyyzzz yyzzzz
    // yzzzzz zzzzzz.
    const L6: [[f64; 28]; 13] = [
        [
            0.0,
            4.099_104_631_151_486,
            0.0,
            0.0,
            0.0,
            0.0,
            -13.663_682_103_838_289,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.099_104_631_151_486,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            11.833_095_811_158_763,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -23.666_191_622_317_527,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.366_619_162_231_752_5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            -2.018_259_602_914_896_3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            20.182_596_029_148_968,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.018_259_602_914_896_3,
            0.0,
            -20.182_596_029_148_968,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -8.290_847_335_634_31,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.527_231_557_089_541,
            0.0,
            22.108_926_228_358_165,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.763_615_778_544_770_6,
            0.0,
            -7.369_642_076_119_389,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.921_205_259_514_923_6,
            0.0,
            0.0,
            0.0,
            0.0,
            1.842_410_519_029_847_2,
            0.0,
            -14.739_284_152_238_778,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.921_205_259_514_923_6,
            0.0,
            -14.739_284_152_238_778,
            0.0,
            14.739_284_152_238_778,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            2.913_106_812_593_657,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            5.826_213_625_187_314,
            0.0,
            -11.652_427_250_374_627,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.913_106_812_593_657,
            0.0,
            -11.652_427_250_374_627,
            0.0,
            4.660_970_900_149_850_5,
            0.0,
        ],
        [
            -0.317_846_011_338_142_1,
            0.0,
            0.0,
            -0.953_538_034_014_426_4,
            0.0,
            5.721_228_204_086_558,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.953_538_034_014_426_4,
            0.0,
            11.442_456_408_173_117,
            0.0,
            -7.628_304_272_115_411,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.317_846_011_338_142_1,
            0.0,
            5.721_228_204_086_558,
            0.0,
            -7.628_304_272_115_411,
            0.0,
            1.017_107_236_282_054_8,
        ],
        [
            0.0,
            0.0,
            2.913_106_812_593_657,
            0.0,
            0.0,
            0.0,
            0.0,
            5.826_213_625_187_314,
            0.0,
            -11.652_427_250_374_627,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.913_106_812_593_657,
            0.0,
            -11.652_427_250_374_627,
            0.0,
            4.660_970_900_149_850_5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.460_602_629_757_461_8,
            0.0,
            0.0,
            0.460_602_629_757_461_8,
            0.0,
            -7.369_642_076_119_389,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.460_602_629_757_461_8,
            0.0,
            0.0,
            0.0,
            7.369_642_076_119_389,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.460_602_629_757_461_8,
            0.0,
            7.369_642_076_119_389,
            0.0,
            -7.369_642_076_119_389,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            -2.763_615_778_544_770_6,
            0.0,
            0.0,
            0.0,
            0.0,
            5.527_231_557_089_541,
            0.0,
            7.369_642_076_119_389,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            8.290_847_335_634_31,
            0.0,
            -22.108_926_228_358_165,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.504_564_900_728_724_1,
            0.0,
            0.0,
            2.522_824_503_643_62,
            0.0,
            5.045_649_007_287_242,
            0.0,
            0.0,
            0.0,
            0.0,
            2.522_824_503_643_62,
            0.0,
            -30.273_894_043_723_452,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.504_564_900_728_724_1,
            0.0,
            5.045_649_007_287_242,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            2.366_619_162_231_752_5,
            0.0,
            0.0,
            0.0,
            0.0,
            -23.666_191_622_317_527,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            11.833_095_811_158_763,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.683_184_105_191_914_4,
            0.0,
            0.0,
            -10.247_761_577_878_716,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.247_761_577_878_716,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.683_184_105_191_914_4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    match l {
        0 => Ok(L0[m_row][cart_col]),
        1 => Ok(L1[m_row][cart_col]),
        2 => Ok(L2[m_row][cart_col]),
        3 => Ok(L3[m_row][cart_col]),
        4 => Ok(L4[m_row][cart_col]),
        5 => Ok(L5[m_row][cart_col]),
        6 => Ok(L6[m_row][cart_col]),
        _ => Err(PyscfRsError::NotYetImplemented {
            phase: 4,
            what: "cart→sph transform for l>6 (k-shells and above) not yet implemented — \
                   add c2s_coeff table for the required l",
        }),
    }
}

/// Public per-`l` cart→sph coefficient matrix `T[l]` as a flat **row-major**
/// `[nsph(l) × ncart(l)]` buffer, where `T[m * ncart + cart] = c2s_coeff(l, m,
/// cart)` (the libcint `g_trans_cart2sph` convention, frozen Condon-Shortley).
///
/// `nsph(l) = 2l+1`, `ncart(l) = (l+1)(l+2)/2`. This is the SAME matrix the
/// `eval_gto_sph` path uses for the cartesian→spherical AO transform, exposed so
/// higher crates can assemble the molecular `cart2sph_coeff` (e.g. the cartesian
/// init-guess density projection). Returns `NotYetImplemented{phase:4}` for
/// `l > 6` (k-shells), mirroring [`c2s_coeff`].
///
/// PySCF's `Mole.cart2sph_coeff(normalized='sp')` block equals the **transpose**
/// of this matrix (verified vs PySCF 2.12.1: s/p blocks are identity, d/f/… match
/// element-for-element), so a caller building `[ncart × nsph]` places `T`ᵀ.
pub fn cart2sph_l_matrix(l: u32) -> Result<Vec<f64>, PyscfRsError> {
    let nsph = (2 * l + 1) as usize;
    let ncart = ((l + 1) * (l + 2) / 2) as usize;
    let mut m = vec![0.0f64; nsph * ncart];
    for row in 0..nsph {
        for col in 0..ncart {
            m[row * ncart + col] = c2s_coeff(l, row, col)?;
        }
    }
    Ok(m)
}

// ── s-shell (l=0) device kernel + launcher (quick-260530-ljv) ───────────
//
// First real GPU compute path for eval_gto: the l=0 radial slice
//   out[g, ao] = Y00 · Σ_p coeff[c,p] · exp(-α_p · r²)
// One device thread per flattened output element (g, ao_idx), mirroring
// gemm_kernel's one-thread-per-output shape. F-order write
// `out[g + ao_idx*ngrids]` — byte-identical index math to the host l=0 loop
// (`eval_gto_sph_cpu` lines ~603-614). The inner primitive accumulation is a
// SINGLE-THREAD ORDERED `acc += coef*(-alpha*r2).exp()` over p_idx (NOT a
// tree/parallel reduce) so it tracks the host ordered sum to within <1 ULP/term
// (bounded by the oracle TOL=1e-9, ORACLE-07 — not claimed bit-identical).
//
// f64-restricted (NOT generic `F: Float`): the chemistry precision path is
// f64-only, and this sidesteps the generic-`Float` `.exp()` expansion risk noted
// in the file header. Libcint flat arrays upload as device `Array`s: coords/env
// as `&Array<f64>`, bas/atm/ao_loc as `&Array<i32>` (cubecl 0.10 indexes i32
// arrays fine). The libcint slot constants + y00 ride in as bare scalar args
// (LaunchArg for T=T, like gemm's m/k/n) so NO host-only helper fn is called
// inside `#[cube]` (the "calling a normal Rust fn from inside #[cube]" pitfall).

/// Threads per block for the s-shell launch. One thread = one output element.
const EVAL_GTO_BLOCK: u32 = 256;

/// l=0 (s-shell) AO-on-grid kernel. One thread per `(g, ao_idx)` output element.
/// Each thread resolves its grid point + owning shell/contraction, computes
/// `r²`, accumulates the ordered contracted radial, and writes
/// `out[g + ao_idx*ngrids] = acc · y00`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_kernel(
    coords: &Array<f64>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    ao_loc: &Array<i32>,
    out: &mut Array<f64>,
    ngrids: usize,
    nbas: usize,
    nao: usize,
    y00: f64,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
) {
    let tid = ABSOLUTE_POS;
    if tid < ngrids * nao {
        // Decode the flat F-order index `tid = g + ao_idx*ngrids`.
        let g = tid % ngrids;
        let ao_idx = tid / ngrids;

        let gx = coords[g];
        let gy = coords[g + ngrids];
        let gz = coords[g + 2 * ngrids];

        // Locate the shell + contraction column that owns this AO. ao_loc is the
        // running sum of nctr per shell, so the owning shell is the last one
        // whose ao_loc <= ao_idx. Walk shells in order (small nbas).
        let mut acc = 0.0_f64;
        for shell_idx in 0..nbas {
            let bas_row = shell_idx * bas_slots;
            let ao_off = ao_loc[shell_idx] as usize;
            let nctr = bas[bas_row + nctr_of] as usize;
            // Does this AO fall inside shell_idx's contraction block?
            if ao_idx >= ao_off && ao_idx < ao_off + nctr {
                let c_idx = ao_idx - ao_off;
                let atom_id = bas[bas_row + atom_of] as usize;
                let nprim = bas[bas_row + nprim_of] as usize;
                let pe = bas[bas_row + ptr_exp] as usize;
                let pc = bas[bas_row + ptr_coeff] as usize;

                let atm_row = atom_id * atm_slots;
                let pcoord = atm[atm_row + ptr_coord] as usize;
                let ax = env[pcoord];
                let ay = env[pcoord + 1];
                let az = env[pcoord + 2];

                let dx = gx - ax;
                let dy = gy - ay;
                let dz = gz - az;
                let r2 = dx * dx + dy * dy + dz * dz;

                // ORDERED sequential accumulation over primitives — mirrors the
                // host l=0 loop exactly (NOT a parallel/tree reduce).
                for p_idx in 0..nprim {
                    let alpha = env[pe + p_idx];
                    // Coefficient matrix is F-order: ptr_coeff + c_idx*nprim + p.
                    let coef = env[pc + c_idx * nprim + p_idx];
                    acc += coef * cube_math::double::exp::exp(-alpha * r2, cube_math::MathConfig::EXACT);
                }
                // ang_of==0 holds on this path (all-s routing); referenced so the
                // launch arg is not flagged unused.
                let _ = bas[bas_row + ang_of];
            }
        }
        out[tid] = acc * y00;
    }
}

/// Host-slice launcher for the s-shell kernel. Uploads the libcint flat arrays +
/// coords, allocates the F-order output, launches the kernel on `R`, reads back.
/// Modeled on `gemm.rs::launch_gemm`. Returns `ngrids*nao` f64 in F-order.
#[allow(clippy::too_many_arguments)]
fn launch_eval_gto_s<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
) -> Vec<f64> {
    let nbas = bas.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    let coords_handle = client.create(Bytes::from_elems(coords.to_vec()));
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));
    let out_handle = client.empty(out_len * core::mem::size_of::<f64>());

    let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
    let groups = out_len.div_ceil(EVAL_GTO_BLOCK as usize) as u32;

    // SAFETY: handle lengths match the slice lengths uploaded above; the kernel
    // bounds-guards the tail (`if tid < ngrids*nao`). All input Arrays are
    // read-only, `out` is the only `&mut`. `from_raw_parts` consumes the handle,
    // so clone (clones share the binding).
    unsafe {
        eval_gto_sph_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(EVAL_GTO_BLOCK),
            ArrayArg::from_raw_parts(coords_handle.clone(), coords.len()),
            ArrayArg::from_raw_parts(env_handle.clone(), env.len()),
            ArrayArg::from_raw_parts(bas_handle.clone(), bas.len()),
            ArrayArg::from_raw_parts(atm_handle.clone(), atm.len()),
            ArrayArg::from_raw_parts(ao_loc_handle.clone(), ao_loc.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), out_len),
            // Bare scalar args (LaunchArg for T = T), like gemm's m/k/n.
            ngrids,
            nbas,
            nao,
            y00,
            ATM_SLOTS,
            BAS_SLOTS,
            ATOM_OF,
            ANG_OF,
            NPRIM_OF,
            NCTR_OF,
            PTR_EXP,
            PTR_COEFF,
            PTR_COORD,
        );
    }

    let bytes = client.read(vec![out_handle]);
    bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
}

// ── general l 0..=4 device kernel + launcher (quick-260530-mlg) ──────────
//
// Ports the host `eval_gto_sph_cpu` l>=1 branch (lines ~809-869) into a real
// `#[cube(launch_unchecked)]` kernel and makes the device path the DEFAULT for
// any basis whose max angular momentum is 1..=4 (which subsumes l=0, so mixed
// s+p+d bases like cc-pVDZ run uniformly on the device). One device thread per
// (g, shell). The cart→sph machinery rides in as HOST-PRECOMPUTED angular
// device tables (the host helpers `ncart`/`nsph`/`common_fac_sp`/`cart_powers`/
// `c2s_coeff` return Vec/Result, so they are ILLEGAL inside `#[cube]` — only the
// flat tables they produce cross to the device). See the device_table_schema in
// the plan: 9 angular arrays, all offsets prefix-summed so the kernel indexes
// by l with O(1) arithmetic.
//
// BIT-EXACTNESS: the kernel mirrors the host reduction order EXACTLY —
// sequential `acc += coef*(-alpha*r2).exp()` over primitives THEN `* fac1`
// (the host `oracle_sum` == strict sequential sum for nprim<=128, real shells
// have nprim<=~30), then `cart_val = mono * radial`, then
// `v += c2s_flat[..] * cart_val`, then F-order write
// `out[g + (ao_off + c_idx*nsph_l + m)*ngrids]`. The ONLY divergence from the
// host is `ipow` vs `f64::powi` for l>=3 monomials (<1 ULP, far inside
// TOL=1e-9 / ORACLE-07).

/// Integer power `base^n` for `n <= 4` by repeated multiply — the device
/// equivalent of host `f64::powi` for the cartesian monomial exponents.
///
/// `#[cube]` helper (Solution 1 from the host-fn-in-`#[cube]` pitfall guide):
/// host helpers returning `Vec`/`Result` cannot be called from a kernel, so the
/// one helper the kernel DOES call is itself a `#[cube]` fn. Written in the
/// STATEMENT form (not `let r = if … {}`) per the mismatched-types guide §1.3 to
/// dodge the `ExpandElementTyped` vs `{float}` mismatch.
///
/// This may differ from host `f64::powi` by < 1 ULP at l>=3 but is bounded by
/// TOL=1e-9 (ORACLE-07). `n` comes from i32 cart_pow values cast to u32.
//
// `r = r * base` (not `r *= base`): the explicit-assignment statement form is the
// cubecl-IR-safe pattern (mismatched-types guide §1.3 / conditionals guide);
// compound-assign lowering is not relied upon. Allow the clippy lint that would
// otherwise rewrite it into the compound form.
#[allow(clippy::assign_op_pattern)]
#[cube]
#[inline(always)]
fn ipow(base: f64, n: u32) -> f64 {
    let mut r = 1.0_f64;
    if n >= 1 {
        r = base;
    }
    if n >= 2 {
        r = r * base;
    }
    if n >= 3 {
        r = r * base;
    }
    if n == 4 {
        r = r * base;
    }
    r
}

/// Derivative of the cartesian monomial power: `dpow(q, lq) = lq · q^(lq-1)`,
/// and `0` for `lq == 0`. The device equivalent of host `dpow` (eval_gto.rs
/// ~1299): `if lq==0 { 0.0 } else { lq as f64 * q.powi(lq as i32 - 1) }`.
///
/// `#[cube]` helper (Solution 1 from the host-fn-in-`#[cube]` pitfall guide):
/// it CALLS the `#[cube]` `ipow` (legal — both participate in the cubecl IR),
/// avoiding any plain-Rust call. Written in the STATEMENT form (not
/// `let r = if … {}`) per the mismatched-types guide §1.3 to dodge the
/// `ExpandElementTyped` vs `{float}` mismatch.
///
/// The `if lq >= 1` gate short-circuits `lq == 0` BEFORE evaluating `lq - 1`,
/// so the u32 subtraction never underflows. This parallels host
/// `q.powi(lq as i32 - 1)`; `ipow`-vs-`powi` diverges by < 1 ULP at l>=3, far
/// inside TOL=1e-9 (ORACLE-07).
#[allow(clippy::assign_op_pattern)]
#[cube]
#[inline(always)]
fn dpow(q: f64, lq: u32) -> f64 {
    let mut r = 0.0_f64;
    if lq >= 1 {
        r = (lq as f64) * ipow(q, lq - 1);
    }
    r
}

/// General l 0..=4 AO-on-grid kernel. One thread per `(g, shell)`: the thread
/// evaluates EVERY contraction column and EVERY spherical AO of its shell at its
/// grid point, applying the cart→sph transform from the host-precomputed tables.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_kernel_general(
    coords: &Array<f64>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    ao_loc: &Array<i32>,
    c2s_flat: &Array<f64>,
    cpow_lx: &Array<i32>,
    cpow_ly: &Array<i32>,
    cpow_lz: &Array<i32>,
    ncart_by_l: &Array<i32>,
    nsph_by_l: &Array<i32>,
    fac1_by_l: &Array<f64>,
    c2s_off_by_l: &Array<i32>,
    cpow_off_by_l: &Array<i32>,
    out: &mut Array<f64>,
    ngrids: usize,
    nbas: usize,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
) {
    let tid = ABSOLUTE_POS;
    if tid < ngrids * nbas {
        let g = tid % ngrids;
        let shell = tid / ngrids;

        let gx = coords[g];
        let gy = coords[g + ngrids];
        let gz = coords[g + 2 * ngrids];

        let bas_row = shell * bas_slots;
        let l = bas[bas_row + ang_of] as u32;
        let lu = l as usize;
        let atom_id = bas[bas_row + atom_of] as usize;
        let nprim = bas[bas_row + nprim_of] as usize;
        let nctr = bas[bas_row + nctr_of] as usize;
        let pe = bas[bas_row + ptr_exp] as usize;
        let pc = bas[bas_row + ptr_coeff] as usize;
        let ao_off = ao_loc[shell] as usize;

        let atm_row = atom_id * atm_slots;
        let pcoord = atm[atm_row + ptr_coord] as usize;
        let ax = env[pcoord];
        let ay = env[pcoord + 1];
        let az = env[pcoord + 2];

        let dx = gx - ax;
        let dy = gy - ay;
        let dz = gz - az;
        let r2 = dx * dx + dy * dy + dz * dz;

        // l-indexed angular reads (cast i32 table entries to usize for indexing).
        let ncart_l = ncart_by_l[lu] as usize;
        let nsph_l = nsph_by_l[lu] as usize;
        let fac1 = fac1_by_l[lu];
        let c2s_off = c2s_off_by_l[lu] as usize;
        let cpow_off = cpow_off_by_l[lu] as usize;

        for c_idx in 0..nctr {
            // ORDERED sequential contracted radial — mirrors the host l>=1 loop
            // (oracle_sum == strict sequential for nprim<=128), THEN * fac1.
            let mut acc = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                acc += coef * cube_math::double::exp::exp(-alpha * r2, cube_math::MathConfig::EXACT);
            }
            let radial = acc * fac1;

            // cart → sph: row m = Σ_ci c2s[l][m][ci] * (mono[ci] * radial).
            for m in 0..nsph_l {
                let mut v = 0.0_f64;
                for ci in 0..ncart_l {
                    let lx = cpow_lx[cpow_off + ci] as u32;
                    let ly = cpow_ly[cpow_off + ci] as u32;
                    let lz = cpow_lz[cpow_off + ci] as u32;
                    let mono = ipow(dx, lx) * ipow(dy, ly) * ipow(dz, lz);
                    let cart_val = mono * radial;
                    v += c2s_flat[c2s_off + m * ncart_l + ci] * cart_val;
                }
                out[g + (ao_off + c_idx * nsph_l + m) * ngrids] = v;
            }
        }
    }
}

/// General l 0..=4 AO-on-grid **deriv1** kernel (value + ∂x/∂y/∂z). One thread
/// per `(g, shell)`: identical arg shape to `eval_gto_sph_kernel_general` PLUS a
/// trailing `comp_stride: usize` scalar and an `out` of length `4*ngrids*nao`.
/// Mirrors host `eval_gto_sph_deriv1_cpu` (eval_gto.rs ~1307-1409) operand-for-
/// operand: ordered sequential `radial` AND `radial_2a` in ONE p-loop (g0 formed
/// once), then the analytic-gradient chain rule per cartesian monomial, then the
/// cart→sph transform applied to all 4 components, written F-order into the 4
/// component blocks `out[k*comp_stride + off]`, `k = 0(v)/1(vx)/2(vy)/3(vz)`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_deriv1_kernel(
    coords: &Array<f64>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    ao_loc: &Array<i32>,
    c2s_flat: &Array<f64>,
    cpow_lx: &Array<i32>,
    cpow_ly: &Array<i32>,
    cpow_lz: &Array<i32>,
    ncart_by_l: &Array<i32>,
    nsph_by_l: &Array<i32>,
    fac1_by_l: &Array<f64>,
    c2s_off_by_l: &Array<i32>,
    cpow_off_by_l: &Array<i32>,
    out: &mut Array<f64>,
    ngrids: usize,
    nbas: usize,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    comp_stride: usize,
) {
    let tid = ABSOLUTE_POS;
    if tid < ngrids * nbas {
        let g = tid % ngrids;
        let shell = tid / ngrids;

        let gx = coords[g];
        let gy = coords[g + ngrids];
        let gz = coords[g + 2 * ngrids];

        let bas_row = shell * bas_slots;
        let l = bas[bas_row + ang_of] as u32;
        let lu = l as usize;
        let atom_id = bas[bas_row + atom_of] as usize;
        let nprim = bas[bas_row + nprim_of] as usize;
        let nctr = bas[bas_row + nctr_of] as usize;
        let pe = bas[bas_row + ptr_exp] as usize;
        let pc = bas[bas_row + ptr_coeff] as usize;
        let ao_off = ao_loc[shell] as usize;

        let atm_row = atom_id * atm_slots;
        let pcoord = atm[atm_row + ptr_coord] as usize;
        let ax = env[pcoord];
        let ay = env[pcoord + 1];
        let az = env[pcoord + 2];

        let dx = gx - ax;
        let dy = gy - ay;
        let dz = gz - az;
        let r2 = dx * dx + dy * dy + dz * dz;

        let ncart_l = ncart_by_l[lu] as usize;
        let nsph_l = nsph_by_l[lu] as usize;
        let fac1 = fac1_by_l[lu];
        let c2s_off = c2s_off_by_l[lu] as usize;
        let cpow_off = cpow_off_by_l[lu] as usize;

        for c_idx in 0..nctr {
            // ORDERED sequential radial + radial_2a in ONE p-loop: form g0 once,
            // then acc += g0 and acc2a += -2α·g0 (mirrors host operand order
            // eval_gto.rs ~1359-1361). THEN * fac1. Plain sequential acc (NOT
            // oracle_sum) — the T2 oracle sums sequentially to match (ORACLE-07).
            let mut acc = 0.0_f64;
            let mut acc2a = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                let g0 = coef * cube_math::double::exp::exp(-alpha * r2, cube_math::MathConfig::EXACT);
                acc += g0;
                acc2a += (-2.0) * alpha * g0;
            }
            let radial = acc * fac1;
            let radial_2a = acc2a * fac1;

            for m in 0..nsph_l {
                let mut v = 0.0_f64;
                let mut vx = 0.0_f64;
                let mut vy = 0.0_f64;
                let mut vz = 0.0_f64;
                for ci in 0..ncart_l {
                    let lx = cpow_lx[cpow_off + ci] as u32;
                    let ly = cpow_ly[cpow_off + ci] as u32;
                    let lz = cpow_lz[cpow_off + ci] as u32;
                    let mono = ipow(dx, lx) * ipow(dy, ly) * ipow(dz, lz);
                    let cval = mono * radial;
                    // operand order EXACTLY matches host eval_gto.rs ~1382-1387.
                    let cdx =
                        radial_2a * dx * mono + radial * dpow(dx, lx) * ipow(dy, ly) * ipow(dz, lz);
                    let cdy =
                        radial_2a * dy * mono + radial * ipow(dx, lx) * dpow(dy, ly) * ipow(dz, lz);
                    let cdz =
                        radial_2a * dz * mono + radial * ipow(dx, lx) * ipow(dy, ly) * dpow(dz, lz);
                    let t = c2s_flat[c2s_off + m * ncart_l + ci];
                    v += t * cval;
                    vx += t * cdx;
                    vy += t * cdy;
                    vz += t * cdz;
                }
                let off = g + (ao_off + c_idx * nsph_l + m) * ngrids;
                out[off] = v;
                out[comp_stride + off] = vx;
                out[2 * comp_stride + off] = vy;
                out[3 * comp_stride + off] = vz;
            }
        }
    }
}

/// Host-precomputed angular device tables for the general l 0..=4 kernel.
/// Built once per launch for `l in 0..=maxl` (caller guarantees `maxl <= 4`).
struct AngularTables {
    c2s_flat: Vec<f64>,
    cpow_lx: Vec<i32>,
    cpow_ly: Vec<i32>,
    cpow_lz: Vec<i32>,
    ncart_by_l: Vec<i32>,
    nsph_by_l: Vec<i32>,
    fac1_by_l: Vec<f64>,
    c2s_off_by_l: Vec<i32>,
    cpow_off_by_l: Vec<i32>,
}

/// Build the 9 angular device tables for `l in 0..=maxl`. Calls the HOST
/// helpers (`ncart`/`nsph`/`common_fac_sp`/`cart_powers`/`c2s_coeff`) and
/// flattens them into the prefix-summed device_table_schema layout:
///   c2s_flat[c2s_off_by_l[l] + m*ncart(l) + ci]
///   cpow_l{x,y,z}[cpow_off_by_l[l] + ci]
/// `c2s_coeff`'s `Err` is propagated, but the caller only invokes with
/// `maxl <= 4` so it never errors on this path.
fn build_angular_tables(maxl: u32) -> Result<AngularTables, PyscfRsError> {
    let nl = (maxl as usize) + 1;
    let mut c2s_flat: Vec<f64> = Vec::new();
    let mut cpow_lx: Vec<i32> = Vec::new();
    let mut cpow_ly: Vec<i32> = Vec::new();
    let mut cpow_lz: Vec<i32> = Vec::new();
    let mut ncart_by_l: Vec<i32> = Vec::with_capacity(nl);
    let mut nsph_by_l: Vec<i32> = Vec::with_capacity(nl);
    let mut fac1_by_l: Vec<f64> = Vec::with_capacity(nl);
    let mut c2s_off_by_l: Vec<i32> = Vec::with_capacity(nl);
    let mut cpow_off_by_l: Vec<i32> = Vec::with_capacity(nl);

    let mut c2s_off: i32 = 0;
    let mut cpow_off: i32 = 0;
    for l in 0..=maxl {
        let ncart_l = ncart(l);
        let nsph_l = nsph(l);
        ncart_by_l.push(ncart_l as i32);
        nsph_by_l.push(nsph_l as i32);
        fac1_by_l.push(common_fac_sp(l));
        c2s_off_by_l.push(c2s_off);
        cpow_off_by_l.push(cpow_off);

        // c2s matrix block: row-major [m][ci].
        for m in 0..nsph_l {
            for ci in 0..ncart_l {
                c2s_flat.push(c2s_coeff(l, m, ci)?);
            }
        }
        // cart power columns (parallel lx/ly/lz arrays).
        for (lx, ly, lz) in cart_powers(l) {
            cpow_lx.push(lx as i32);
            cpow_ly.push(ly as i32);
            cpow_lz.push(lz as i32);
        }

        c2s_off += (ncart_l * nsph_l) as i32;
        cpow_off += ncart_l as i32;
    }

    Ok(AngularTables {
        c2s_flat,
        cpow_lx,
        cpow_ly,
        cpow_lz,
        ncart_by_l,
        nsph_by_l,
        fac1_by_l,
        c2s_off_by_l,
        cpow_off_by_l,
    })
}

/// Host-slice launcher for the general l 0..=4 kernel. Builds the angular
/// device tables + the libcint flat arrays, launches one thread per
/// `(g, shell)`, reads back the F-order `(ngrids, nao)` buffer.
/// Modeled on `launch_eval_gto_s`. `maxl` must be `<= 4` (caller guarantees).
#[allow(clippy::too_many_arguments)]
fn launch_eval_gto_general<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    maxl: u32,
) -> Result<Vec<f64>, PyscfRsError> {
    let nbas = bas.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    let t = build_angular_tables(maxl)?;

    let coords_handle = client.create(Bytes::from_elems(coords.to_vec()));
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));

    let c2s_flat_h = client.create(Bytes::from_elems(t.c2s_flat.clone()));
    let cpow_lx_h = client.create(Bytes::from_elems(t.cpow_lx.clone()));
    let cpow_ly_h = client.create(Bytes::from_elems(t.cpow_ly.clone()));
    let cpow_lz_h = client.create(Bytes::from_elems(t.cpow_lz.clone()));
    let ncart_h = client.create(Bytes::from_elems(t.ncart_by_l.clone()));
    let nsph_h = client.create(Bytes::from_elems(t.nsph_by_l.clone()));
    let fac1_h = client.create(Bytes::from_elems(t.fac1_by_l.clone()));
    let c2s_off_h = client.create(Bytes::from_elems(t.c2s_off_by_l.clone()));
    let cpow_off_h = client.create(Bytes::from_elems(t.cpow_off_by_l.clone()));

    let out_handle = client.empty(out_len * core::mem::size_of::<f64>());

    let groups = (ngrids * nbas).div_ceil(EVAL_GTO_BLOCK as usize) as u32;

    // SAFETY: every handle length matches the slice length uploaded above; the
    // kernel bounds-guards the tail (`if tid < ngrids*nbas`). All input Arrays
    // are read-only, `out` is the only `&mut`. `from_raw_parts` consumes the
    // handle, so clone (clones share the binding).
    unsafe {
        eval_gto_sph_kernel_general::launch_unchecked::<R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(EVAL_GTO_BLOCK),
            ArrayArg::from_raw_parts(coords_handle.clone(), coords.len()),
            ArrayArg::from_raw_parts(env_handle.clone(), env.len()),
            ArrayArg::from_raw_parts(bas_handle.clone(), bas.len()),
            ArrayArg::from_raw_parts(atm_handle.clone(), atm.len()),
            ArrayArg::from_raw_parts(ao_loc_handle.clone(), ao_loc.len()),
            ArrayArg::from_raw_parts(c2s_flat_h.clone(), t.c2s_flat.len()),
            ArrayArg::from_raw_parts(cpow_lx_h.clone(), t.cpow_lx.len()),
            ArrayArg::from_raw_parts(cpow_ly_h.clone(), t.cpow_ly.len()),
            ArrayArg::from_raw_parts(cpow_lz_h.clone(), t.cpow_lz.len()),
            ArrayArg::from_raw_parts(ncart_h.clone(), t.ncart_by_l.len()),
            ArrayArg::from_raw_parts(nsph_h.clone(), t.nsph_by_l.len()),
            ArrayArg::from_raw_parts(fac1_h.clone(), t.fac1_by_l.len()),
            ArrayArg::from_raw_parts(c2s_off_h.clone(), t.c2s_off_by_l.len()),
            ArrayArg::from_raw_parts(cpow_off_h.clone(), t.cpow_off_by_l.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), out_len),
            // Bare scalar args (LaunchArg for T = T), like the s-kernel.
            ngrids,
            nbas,
            ATM_SLOTS,
            BAS_SLOTS,
            ATOM_OF,
            ANG_OF,
            NPRIM_OF,
            NCTR_OF,
            PTR_EXP,
            PTR_COEFF,
            PTR_COORD,
        );
    }

    let bytes = client.read(vec![out_handle]);
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec())
}

/// Host-slice launcher for the general l 0..=4 **deriv1** kernel. Clones
/// `launch_eval_gto_general` byte-for-byte EXCEPT: `out_len = 4*ngrids*nao`,
/// launches `eval_gto_sph_deriv1_kernel`, and passes the extra `comp_stride =
/// ngrids*nao` scalar AFTER the existing trailing scalar args. Reads back the
/// F-order `[4, ngrids, nao]` buffer. `maxl` must be `<= 4` (caller guarantees).
#[allow(clippy::too_many_arguments)]
fn launch_eval_gto_deriv1<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    maxl: u32,
) -> Result<Vec<f64>, PyscfRsError> {
    let nbas = bas.len() / BAS_SLOTS;
    let comp_stride = ngrids * nao;
    let out_len = 4 * comp_stride;

    let t = build_angular_tables(maxl)?;

    let coords_handle = client.create(Bytes::from_elems(coords.to_vec()));
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));

    let c2s_flat_h = client.create(Bytes::from_elems(t.c2s_flat.clone()));
    let cpow_lx_h = client.create(Bytes::from_elems(t.cpow_lx.clone()));
    let cpow_ly_h = client.create(Bytes::from_elems(t.cpow_ly.clone()));
    let cpow_lz_h = client.create(Bytes::from_elems(t.cpow_lz.clone()));
    let ncart_h = client.create(Bytes::from_elems(t.ncart_by_l.clone()));
    let nsph_h = client.create(Bytes::from_elems(t.nsph_by_l.clone()));
    let fac1_h = client.create(Bytes::from_elems(t.fac1_by_l.clone()));
    let c2s_off_h = client.create(Bytes::from_elems(t.c2s_off_by_l.clone()));
    let cpow_off_h = client.create(Bytes::from_elems(t.cpow_off_by_l.clone()));

    let out_handle = client.empty(out_len * core::mem::size_of::<f64>());

    let groups = (ngrids * nbas).div_ceil(EVAL_GTO_BLOCK as usize) as u32;

    // SAFETY: every handle length matches the slice length uploaded above; the
    // kernel bounds-guards the tail (`if tid < ngrids*nbas`). All input Arrays
    // are read-only, `out` (len 4*ngrids*nao) is the only `&mut`.
    unsafe {
        eval_gto_sph_deriv1_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(EVAL_GTO_BLOCK),
            ArrayArg::from_raw_parts(coords_handle.clone(), coords.len()),
            ArrayArg::from_raw_parts(env_handle.clone(), env.len()),
            ArrayArg::from_raw_parts(bas_handle.clone(), bas.len()),
            ArrayArg::from_raw_parts(atm_handle.clone(), atm.len()),
            ArrayArg::from_raw_parts(ao_loc_handle.clone(), ao_loc.len()),
            ArrayArg::from_raw_parts(c2s_flat_h.clone(), t.c2s_flat.len()),
            ArrayArg::from_raw_parts(cpow_lx_h.clone(), t.cpow_lx.len()),
            ArrayArg::from_raw_parts(cpow_ly_h.clone(), t.cpow_ly.len()),
            ArrayArg::from_raw_parts(cpow_lz_h.clone(), t.cpow_lz.len()),
            ArrayArg::from_raw_parts(ncart_h.clone(), t.ncart_by_l.len()),
            ArrayArg::from_raw_parts(nsph_h.clone(), t.nsph_by_l.len()),
            ArrayArg::from_raw_parts(fac1_h.clone(), t.fac1_by_l.len()),
            ArrayArg::from_raw_parts(c2s_off_h.clone(), t.c2s_off_by_l.len()),
            ArrayArg::from_raw_parts(cpow_off_h.clone(), t.cpow_off_by_l.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), out_len),
            // Bare scalar args (LaunchArg for T = T), like the general kernel.
            ngrids,
            nbas,
            ATM_SLOTS,
            BAS_SLOTS,
            ATOM_OF,
            ANG_OF,
            NPRIM_OF,
            NCTR_OF,
            PTR_EXP,
            PTR_COEFF,
            PTR_COORD,
            comp_stride,
        );
    }

    let bytes = client.read(vec![out_handle]);
    Ok(bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec())
}

/// Output of an eval_gto call. Flat F-order buffer + shape descriptor.
#[derive(Debug, Clone)]
pub struct EvalGtoBuffers {
    /// Flat F-order buffer. For scalar variants (`GTOval`, `GTOval_sph`,
    /// `GTOval_cart`): `out[g + ao * ngrids]`. For derivative variants
    /// (Phase 4 DFT extension): leading axis indexes the derivative
    /// component.
    pub values: Vec<f64>,
    /// Logical shape — `[ngrids, nao]` for scalar variants; future
    /// `[ncomp, ngrids, nao]` for derivative variants.
    pub shape: Vec<usize>,
}

/// Evaluate `GTOval_sph` (or `GTOval_cart`) on the given grid for the
/// supplied basis. Public surface uses pyscf-algebra types only —
/// `pyscf-gto`'s wrapper calls this without ever naming `cubecl::*`.
///
/// # Arguments
///
/// - `client`: the resolved `AlgebraClient`. Phase 2 ships CPU only;
///   GPU backends fall back to the CPU path with a `tracing::warn!`.
///   Phase 4 DFT (or Phase 8 GPU enable) wires the GPU arms with a
///   `#[cube(launch_unchecked)]` kernel.
/// - `coords`: flat F-order grid coordinates. Length `ngrids * 3`.
///   Layout: `x[0..ngrids], y[ngrids..2*ngrids], z[2*ngrids..3*ngrids]`.
/// - `atm` / `bas` / `env` / `ao_loc`: the libcint flat arrays from
///   `mol._atm`, `mol._bas`, `mol._env`, `mol.ao_loc_nr` (built in 02-04).
/// - `nao`: total number of AOs (`mol.nao_nr`).
/// - `spherical`: `true` → apply `cart2sph` (Phase 4 DFT extension; for
///   l = 0 a no-op so the s-shell smoke test passes either way);
///   `false` → return raw cartesian.
// Arg list mirrors the libcint `GTOval_sph` flat-array API (atm/bas/env/
// ao_loc); grouping into a struct would obscure that correspondence.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    spherical: bool,
) -> Result<EvalGtoBuffers, PyscfRsError> {
    // quick-260530-ljv: route pure-s-shell bases to the real device kernel
    // (the FIRST GPU compute path for eval_gto), fanned out over every backend
    // via the exported `dispatch_backend!`. ANY basis with an l>=1 shell — or an
    // empty grid / empty basis — falls back to the UNCHANGED `eval_gto_sph_cpu`,
    // so all l>=1 numerics stay byte-for-byte identical (behavior-preserving).
    let all_s = !bas.is_empty() && bas.chunks_exact(BAS_SLOTS).all(|row| row[ANG_OF] == 0);

    if all_s && ngrids * nao > 0 {
        // Device path: pure-s-shell. y00 is folded inside the launcher; the
        // `spherical` flag is a no-op for l=0 (Y00 identity), exactly as the
        // host path treats s-shells.
        let _ = spherical;
        let values = dispatch_backend!(
            client,
            c,
            Rt,
            launch_eval_gto_s::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao)
        );
        return Ok(EvalGtoBuffers {
            values,
            shape: vec![ngrids, nao],
        });
    }

    // quick-260530-mlg: route any non-empty basis whose max angular momentum is
    // <= 4 (p/d/f/g — which subsumes l=0 too) to the GENERAL device kernel, fanned
    // out over every backend via `dispatch_backend!`. l>4 (h/i-shells and above),
    // empty basis, and empty grid fall through to the UNCHANGED `eval_gto_sph_cpu`,
    // which now evaluates l=5 (h) and l=6 (i) generically via `c2s_coeff`; only
    // l>6 there returns NotYetImplemented{phase:4}. Plus the out_len==0
    // early-return. The host device tables are built only for l in 0..=maxl with
    // maxl<=4, so `c2s_coeff` is never called with l>4 on the device path.
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;

    if !bas.is_empty() && maxl <= 4 && ngrids * nao > 0 {
        // l>=1 always emits SPHERICAL AOs, matching `eval_gto_sph_cpu`; the
        // `spherical` flag is referenced (no-op for the corpus path) so it is
        // not flagged unused.
        let _ = spherical;
        let values = dispatch_backend!(
            client,
            c,
            Rt,
            launch_eval_gto_general::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao, maxl)
        )?;
        return Ok(EvalGtoBuffers {
            values,
            shape: vec![ngrids, nao],
        });
    }

    // Fallback: l>4 shell, empty basis, or empty grid → the unchanged host
    // path (which also handles the out_len==0 early-return).
    eval_gto_sph_cpu(coords, ngrids, atm, bas, env, ao_loc, nao, spherical)
}

/// CPU-host implementation of the s-shell AO-on-grid kernel.
///
/// One pass per grid point evaluates EVERY shell on that point. Output
/// is F-order `(ngrids, nao)`: index `g + ao * ngrids`.
///
/// **Phase 2 scope** (this commit): the `l == 0` path computes the full
/// contracted radial `Σ_p coeff[c,p] * exp(-α_p * r²)` and writes it
/// directly to the AO slot (Y_00 is absorbed into the normalised
/// coefficient — see `pyscf-gto::make_env::normalise_contractions`). The
/// `l >= 1` branch (plan 04-03) evaluates the cartesian monomials and
/// applies the libcint cart→sph transform.
///
/// `_spherical`: l = 0 is identical for sph and cart (Y_00 == the
/// cartesian s norm). For l >= 1 this kernel always emits the SPHERICAL
/// AOs (the plan-04-03 / DFT scope is `GTOval_sph`). `GTOval_cart` with
/// l >= 1 needs the cartesian `ao_loc`/`nao` (more AOs than spherical)
/// which the caller does not pass here; it stays deferred and the
/// `pyscf-gto::eval_gto` wrapper only routes `spherical=true` into the
/// l >= 1 path for the corpus bases (sp shells excepted — none in v1).
// Mirrors the libcint flat-array API (see `eval_gto_sph`).
#[allow(clippy::too_many_arguments)]
fn eval_gto_sph_cpu(
    coords_host: &[f64],
    ngrids: usize,
    atm_host: &[i32],
    bas_host: &[i32],
    env_host: &[f64],
    ao_loc_host: &[i32],
    nao: usize,
    _spherical: bool,
) -> Result<EvalGtoBuffers, PyscfRsError> {
    debug_assert_eq!(
        coords_host.len(),
        ngrids * 3,
        "coords flat buffer must be ngrids*3 (got {} for ngrids={})",
        coords_host.len(),
        ngrids
    );
    debug_assert!(
        bas_host.len().is_multiple_of(BAS_SLOTS),
        "bas length {} not a multiple of BAS_SLOTS={}",
        bas_host.len(),
        BAS_SLOTS
    );

    let nbas = bas_host.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    // Empty grid → empty output, skip the loop entirely.
    if out_len == 0 {
        return Ok(EvalGtoBuffers {
            values: Vec::new(),
            shape: vec![ngrids, nao],
        });
    }

    let mut out = vec![0.0_f64; out_len];

    // Per-grid-point evaluation. `coords` is F-order: x[0..ngrids],
    // y[ngrids..2*ngrids], z[2*ngrids..3*ngrids].
    for g in 0..ngrids {
        let gx = coords_host[g];
        let gy = coords_host[g + ngrids];
        let gz = coords_host[g + 2 * ngrids];

        // `shell_idx` drives parallel flat-array offsets (bas_host via
        // BAS_SLOTS, ao_loc_host) — a range loop is clearer than enumerate.
        #[allow(clippy::needless_range_loop)]
        for shell_idx in 0..nbas {
            let bas_row = shell_idx * BAS_SLOTS;
            let atom_id = bas_host[bas_row + ATOM_OF] as usize;
            let l = bas_host[bas_row + ANG_OF] as u32;
            let nprim = bas_host[bas_row + NPRIM_OF] as usize;
            let nctr = bas_host[bas_row + NCTR_OF] as usize;
            let ptr_exp = bas_host[bas_row + PTR_EXP] as usize;
            let ptr_coeff = bas_host[bas_row + PTR_COEFF] as usize;

            let atm_row = atom_id * ATM_SLOTS;
            let ptr_coord = atm_host[atm_row + PTR_COORD] as usize;
            let ax = env_host[ptr_coord];
            let ay = env_host[ptr_coord + 1];
            let az = env_host[ptr_coord + 2];

            let dx = gx - ax;
            let dy = gy - ay;
            let dz = gz - az;
            let r2 = dx * dx + dy * dy + dz * dz;

            let ao_off = ao_loc_host[shell_idx] as usize;

            if l == 0 {
                // s-shell path: contracted radial × Y_00.
                //
                // 02-04 `make_env::normalise_contractions` applies the
                // *radial* normalisation only (per-prim gto_norm + the
                // `_nomalize_contracted_ao` factor). The angular factor
                // Y_00 = (1/(4π))^{1/2} = 1/(2*sqrt(π)) is applied here
                // — upstream `pyscf/gto/eval_gto.py` calls `_cart2sph_l(0)`
                // which is the [[1/(2*sqrt(π))]] 1×1 matrix. For s-shells
                // the cartesian normalisation factor is identical to
                // Y_00, so the same multiplier covers both `GTOval_sph`
                // and `GTOval_cart` (`cart_variant_works_for_s_shells`
                // smoke fixture verifies the equality).
                let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
                for c_idx in 0..nctr {
                    let mut acc: f64 = 0.0;
                    for p_idx in 0..nprim {
                        let alpha = env_host[ptr_exp + p_idx];
                        // Coefficient matrix is F-order:
                        //   ptr_coeff + c_idx * nprim + p_idx
                        let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                        acc += coef * (-alpha * r2).exp();
                    }
                    let ao_idx = ao_off + c_idx;
                    out[g + ao_idx * ngrids] = acc * y00;
                }
            } else {
                // l ≥ 1 path (plan 04-03): cartesian monomials × radial,
                // then the libcint cart→sph transform. Mirrors
                // `GTOshell_eval_grid_cart` + `CINTc2s_ket_sph`.
                let fac1 = common_fac_sp(l);
                let powers = cart_powers(l);
                let ncart_l = ncart(l);
                let nsph_l = nsph(l);

                // Precompute the cartesian monomial geometric factors
                // (x^lx · y^ly · z^lz) — radial-independent, shared by
                // every contraction column.
                let mut mono = vec![0.0_f64; ncart_l];
                for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                    mono[ci] = dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32);
                }

                let mut cart_vals = vec![0.0_f64; ncart_l];
                for c_idx in 0..nctr {
                    // Ordered, FMA-free contracted radial. > 2 prims →
                    // oracle_sum (Pitfall 3 / FOUND-06); ≤ 2 prims fold
                    // into the same materialised-then-summed path so the
                    // reduction tree shape depends only on length.
                    let radial = if nprim > 2 {
                        let terms: Vec<f64> = (0..nprim)
                            .map(|p_idx| {
                                let alpha = env_host[ptr_exp + p_idx];
                                let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                                coef * (-alpha * r2).exp()
                            })
                            .collect();
                        oracle_sum(&terms)
                    } else {
                        let mut acc = 0.0_f64;
                        for p_idx in 0..nprim {
                            let alpha = env_host[ptr_exp + p_idx];
                            let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                            acc += coef * (-alpha * r2).exp();
                        }
                        acc
                    };
                    let radial = radial * fac1;

                    // cartesian AO values for this contraction.
                    for ci in 0..ncart_l {
                        cart_vals[ci] = mono[ci] * radial;
                    }

                    // cart → sph: row m = Σ_c T[l][m][c] * cart_vals[c].
                    for m_idx in 0..nsph_l {
                        let mut v = 0.0_f64;
                        // `ci` indexes cart_vals AND feeds c2s_coeff(l, m, ci).
                        #[allow(clippy::needless_range_loop)]
                        for ci in 0..ncart_l {
                            v += c2s_coeff(l, m_idx, ci)? * cart_vals[ci];
                        }
                        let ao_idx = ao_off + c_idx * nsph_l + m_idx;
                        out[g + ao_idx * ngrids] = v;
                    }
                }
            }
        }
    }

    Ok(EvalGtoBuffers {
        values: out,
        shape: vec![ngrids, nao],
    })
}

/// Evaluate `GTOval_sph_deriv1` on the grid: the AO value plus the three
/// Cartesian gradient components (∂/∂x, ∂/∂y, ∂/∂z) per AO per grid
/// point. Output is the deriv-variant layout `[4, ngrids, nao]`: a flat
/// buffer where component `c ∈ {0=value, 1=∂x, 2=∂y, 3=∂z}` occupies
/// `values[c*ngrids*nao ..]` and within each component the index is the
/// F-order `g + ao*ngrids` (same as the scalar variants).
///
/// This is the GGA grid-loop input (∇ρ → σ = |∇ρ|²). Public surface uses
/// `pyscf-algebra` types only (AlgebraClient) so `pyscf-gto`'s wrapper
/// imports it without naming `cubecl::*` (algebra wall / D-04).
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_deriv1(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
) -> Result<EvalGtoBuffers, PyscfRsError> {
    // quick-260530-oms: route any non-empty basis whose max angular momentum is
    // <= 4 (p/d/f/g, subsuming l=0) to the general deriv1 device kernel, fanned
    // out over every backend via `dispatch_backend!`. l>4 (h/i-shells+), empty
    // basis, and empty grid fall through to the UNCHANGED
    // `eval_gto_sph_deriv1_cpu`, which now evaluates l=5/l=6 generically via
    // `c2s_coeff` (only l>6 returns NotYetImplemented{phase:4}), plus the
    // comp_stride==0 early-return. Host device tables are built only for l in
    // 0..=maxl with maxl<=4, so `c2s_coeff` is never called with l>4 on device.
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;
    let comp_stride = ngrids * nao;

    if !bas.is_empty() && maxl <= 4 && comp_stride > 0 {
        let values = dispatch_backend!(
            client,
            c,
            Rt,
            launch_eval_gto_deriv1::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao, maxl)
        )?;
        return Ok(EvalGtoBuffers {
            values,
            shape: vec![4, ngrids, nao],
        });
    }

    // Fallback: l>4 shell, empty basis, or empty grid → the unchanged host
    // deriv1 path (which handles the comp_stride==0 early-return, evaluates
    // l=5/l=6 generically, and returns NotYetImplemented{phase:4} for l>6).
    eval_gto_sph_deriv1_cpu(coords, ngrids, atm, bas, env, ao_loc, nao, true)
}

/// Evaluate `GTOval_cart_deriv1` on the grid: the *cartesian* AO value plus
/// the three Cartesian gradient components (∂/∂x, ∂/∂y, ∂/∂z) per cartesian
/// AO per grid point. Output layout matches `eval_gto_sph_deriv1`
/// (`[4, ngrids, nao]`, component-leading) but `nao`/`ao_loc` are the
/// **cartesian** counts/offsets (`ncart(l)·nctr` per shell, not `2l+1`).
///
/// Cartesian output is exactly the pre-`c2s` value the spherical kernel
/// transforms: `GTOval_sph_deriv1 = c2s · GTOval_cart_deriv1` per shell/
/// component (the libcint relationship — `GTOshell_eval_grid_cart` produces
/// these, `GTOval_sph` applies `CINTc2s_ket_sph` on top). So this shares the
/// byte-verified deriv1 stencil and only skips the final transform.
///
/// **Host-only** (F-02): the device deriv1 launcher emits spherical AOs; the
/// cartesian path runs on the CPU host kernel directly. `client` is accepted
/// for API symmetry with the spherical entry point. Unlike the spherical
/// path there is no `l` ceiling — cartesian output never calls `c2s_coeff`,
/// so it is well-defined for every angular momentum the basis supplies.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_cart_deriv1(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc_cart: &[i32],
    nao_cart: usize,
) -> Result<EvalGtoBuffers, PyscfRsError> {
    let _ = client; // host-only; no device cartesian deriv1 kernel yet.
    eval_gto_sph_deriv1_cpu(coords, ngrids, atm, bas, env, ao_loc_cart, nao_cart, false)
}

/// CPU-host deriv1 kernel. Mirrors `eval_gto_sph_cpu` for the value and
/// adds the analytic gradient via the upstream
/// `GTOshell_eval_grid_ip_cart` general-l stencil:
///   ∂/∂q [q^lq · m̃ · R] = (−2α) Σ_p c·exp · q · (x^lx y^ly z^lz)
///                        + R · lq · q^(lq−1) · (other two monomials)
/// where the first term is the radial-derivative chain rule
/// (`exps_2a = Σ −2α c exp`) and the second is the monomial-power
/// derivative.
///
/// `spherical = true` applies the libcint `c2s` transform to each of the 4
/// components and writes `2l+1` spherical AOs per contraction (the
/// `GTOval_sph_deriv1` surface — `ao_loc`/`nao` must be spherical).
/// `spherical = false` skips the transform and writes the `ncart(l)`
/// cartesian components directly (the `GTOval_cart_deriv1` surface —
/// `ao_loc`/`nao` must be cartesian). The two share every numeric stencil up
/// to the final write, so the cartesian output is exactly the pre-transform
/// value the spherical path consumes.
#[allow(clippy::too_many_arguments)]
fn eval_gto_sph_deriv1_cpu(
    coords_host: &[f64],
    ngrids: usize,
    atm_host: &[i32],
    bas_host: &[i32],
    env_host: &[f64],
    ao_loc_host: &[i32],
    nao: usize,
    spherical: bool,
) -> Result<EvalGtoBuffers, PyscfRsError> {
    debug_assert_eq!(
        coords_host.len(),
        ngrids * 3,
        "coords flat buffer must be ngrids*3 (got {} for ngrids={})",
        coords_host.len(),
        ngrids
    );
    debug_assert!(
        bas_host.len().is_multiple_of(BAS_SLOTS),
        "bas length {} not a multiple of BAS_SLOTS={}",
        bas_host.len(),
        BAS_SLOTS
    );

    let nbas = bas_host.len() / BAS_SLOTS;
    let comp_stride = ngrids * nao;
    let out_len = 4 * comp_stride;
    if comp_stride == 0 {
        return Ok(EvalGtoBuffers {
            values: Vec::new(),
            shape: vec![4, ngrids, nao],
        });
    }
    let mut out = vec![0.0_f64; out_len];

    /// derivative of the monomial power: lq * q^(lq-1), and 0 for lq==0.
    #[inline]
    fn dpow(q: f64, lq: u32) -> f64 {
        if lq == 0 {
            0.0
        } else {
            lq as f64 * q.powi(lq as i32 - 1)
        }
    }

    for g in 0..ngrids {
        let gx = coords_host[g];
        let gy = coords_host[g + ngrids];
        let gz = coords_host[g + 2 * ngrids];

        // `shell_idx` drives parallel flat-array offsets (bas_host via
        // BAS_SLOTS, ao_loc_host) — a range loop is clearer than enumerate.
        #[allow(clippy::needless_range_loop)]
        for shell_idx in 0..nbas {
            let bas_row = shell_idx * BAS_SLOTS;
            let atom_id = bas_host[bas_row + ATOM_OF] as usize;
            let l = bas_host[bas_row + ANG_OF] as u32;
            let nprim = bas_host[bas_row + NPRIM_OF] as usize;
            let nctr = bas_host[bas_row + NCTR_OF] as usize;
            let ptr_exp = bas_host[bas_row + PTR_EXP] as usize;
            let ptr_coeff = bas_host[bas_row + PTR_COEFF] as usize;

            let ptr_coord = atm_host[atom_id * ATM_SLOTS + PTR_COORD] as usize;
            let ax = env_host[ptr_coord];
            let ay = env_host[ptr_coord + 1];
            let az = env_host[ptr_coord + 2];
            let dx = gx - ax;
            let dy = gy - ay;
            let dz = gz - az;
            let r2 = dx * dx + dy * dy + dz * dz;

            let fac1 = common_fac_sp(l);
            let powers = cart_powers(l);
            let ncart_l = ncart(l);
            let nsph_l = nsph(l);
            let ao_off = ao_loc_host[shell_idx] as usize;

            // monomial geometric factors (radial-independent).
            let mut mono = vec![0.0_f64; ncart_l];
            for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                mono[ci] = dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32);
            }

            let mut cval = vec![0.0_f64; ncart_l];
            let mut cdx = vec![0.0_f64; ncart_l];
            let mut cdy = vec![0.0_f64; ncart_l];
            let mut cdz = vec![0.0_f64; ncart_l];

            for c_idx in 0..nctr {
                // Ordered, FMA-free radial e = Σ c·exp and e2a = Σ −2α·c·exp.
                // > 2 prims → oracle_sum (Pitfall 3 / FOUND-06).
                let (radial, radial_2a) = if nprim > 2 {
                    let mut e_terms = Vec::with_capacity(nprim);
                    let mut e2a_terms = Vec::with_capacity(nprim);
                    for p_idx in 0..nprim {
                        let alpha = env_host[ptr_exp + p_idx];
                        let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                        let g0 = coef * (-alpha * r2).exp();
                        e_terms.push(g0);
                        e2a_terms.push(-2.0 * alpha * g0);
                    }
                    (oracle_sum(&e_terms), oracle_sum(&e2a_terms))
                } else {
                    let mut e = 0.0_f64;
                    let mut e2a = 0.0_f64;
                    for p_idx in 0..nprim {
                        let alpha = env_host[ptr_exp + p_idx];
                        let coef = env_host[ptr_coeff + c_idx * nprim + p_idx];
                        let g0 = coef * (-alpha * r2).exp();
                        e += g0;
                        e2a += -2.0 * alpha * g0;
                    }
                    (e, e2a)
                };
                let radial = radial * fac1;
                let radial_2a = radial_2a * fac1;

                for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                    let m = mono[ci];
                    cval[ci] = m * radial;
                    cdx[ci] = radial_2a * dx * m
                        + radial * dpow(dx, lx) * dy.powi(ly as i32) * dz.powi(lz as i32);
                    cdy[ci] = radial_2a * dy * m
                        + radial * dx.powi(lx as i32) * dpow(dy, ly) * dz.powi(lz as i32);
                    cdz[ci] = radial_2a * dz * m
                        + radial * dx.powi(lx as i32) * dy.powi(ly as i32) * dpow(dz, lz);
                }

                if spherical {
                    // cart → sph per component, written into the 4 component blocks.
                    for m_idx in 0..nsph_l {
                        let ao_idx = ao_off + c_idx * nsph_l + m_idx;
                        let off = g + ao_idx * ngrids;
                        let mut v = 0.0_f64;
                        let mut vx = 0.0_f64;
                        let mut vy = 0.0_f64;
                        let mut vz = 0.0_f64;
                        for ci in 0..ncart_l {
                            let t = c2s_coeff(l, m_idx, ci)?;
                            v += t * cval[ci];
                            vx += t * cdx[ci];
                            vy += t * cdy[ci];
                            vz += t * cdz[ci];
                        }
                        out[off] = v;
                        out[comp_stride + off] = vx;
                        out[2 * comp_stride + off] = vy;
                        out[3 * comp_stride + off] = vz;
                    }
                } else {
                    // Cartesian output: write the ncart_l cartesian components
                    // directly (no c2s), at ao_off + c_idx*ncart_l + ci.
                    // `ao_loc`/`nao` are the cartesian counts so this stays in
                    // bounds; never calls c2s_coeff → no `l` ceiling.
                    for ci in 0..ncart_l {
                        let ao_idx = ao_off + c_idx * ncart_l + ci;
                        let off = g + ao_idx * ngrids;
                        out[off] = cval[ci];
                        out[comp_stride + off] = cdx[ci];
                        out[2 * comp_stride + off] = cdy[ci];
                        out[3 * comp_stride + off] = cdz[ci];
                    }
                }
            }
        }
    }

    Ok(EvalGtoBuffers {
        values: out,
        shape: vec![4, ngrids, nao],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::PyscfRsError;

    /// BLOCKER CR-03 (FOUND-07 never-panic): `c2s_coeff` must return
    /// `Err(NotYetImplemented{phase:4,..})` for l>6 (k-shells and above)
    /// instead of `panic!`-ing. A user supplying a k-shell basis through the
    /// PyO3 boundary would otherwise abort the Python process. l=5 (h) and
    /// l=6 (i) are now supported and must NOT error.
    #[test]
    fn c2s_coeff_l7_returns_err_not_panic() {
        // l = 7 (k-shell) is not in the c2s table → Err, no panic.
        let r = c2s_coeff(7, 0, 0);
        assert!(
            matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 4, .. })),
            "c2s_coeff(7,..) must return Err(NotYetImplemented{{phase:4}}), got {r:?}"
        );
        // l = 8 too — the wildcard arm covers every l>6.
        assert!(matches!(
            c2s_coeff(8, 0, 0),
            Err(PyscfRsError::NotYetImplemented { phase: 4, .. })
        ));
        // l = 5 and l = 6 are now in-table — must succeed.
        assert!(c2s_coeff(5, 0, 1).is_ok());
        assert!(c2s_coeff(6, 0, 1).is_ok());
    }

    /// INDEPENDENT ORACLE for the l=5 (h) and l=6 (i) tables: re-derive every
    /// `g_trans_cart2sph` entry from the Schlegel–Frisch `xyz2sph_real`
    /// analytical formula (libcint `scripts/cart2sph.py`, IJQC 54(1995) 83)
    /// and assert the in-source table agrees. The L5/L6 constants were
    /// extracted verbatim from the libcint C array; this test guards against
    /// transcription error by checking them against a from-scratch derivation.
    #[test]
    fn c2s_coeff_l5_l6_match_schlegel_frisch_formula() {
        fn fact(n: i64) -> f64 {
            (1..=n.max(0)).map(|x| x as f64).product::<f64>()
        }
        fn binom(n: i64, k: i64) -> f64 {
            if k < 0 || k > n {
                0.0
            } else {
                fact(n) / fact(k) / fact(n - k)
            }
        }
        // Real spherical-harmonic cart→sph factor (Condon-Shortley phase).
        fn xyz2sph_real(lx: i64, ly: i64, lz: i64, m: i64) -> f64 {
            if (lx + ly + m).rem_euclid(2) != 0 {
                return 0.0;
            }
            let l = lx + ly + lz;
            let am = m.abs();
            let j = (lx + ly - am).div_euclid(2);
            let c0_num = (2 * l + 1) as f64 * fact(l - am);
            let c0_div = fact(l + am) * 4.0 * std::f64::consts::PI;
            let mut c0 = (c0_num / c0_div).sqrt() * 0.5f64.powi(l as i32) / fact(l);
            let mut cp = 0.0;
            let mut i = j.max(0);
            while i <= (l - am).div_euclid(2) {
                let cp0 = binom(l, i) * binom(i, j) * fact(2 * l - 2 * i) / fact(l - am - 2 * i);
                if i.rem_euclid(2) != 0 {
                    cp -= cp0;
                } else {
                    cp += cp0;
                }
                i += 1;
            }
            c0 *= cp;
            let mut cp = 0.0;
            let k_lo = ((lx - am + 1).div_euclid(2)).max(0);
            let k_hi = j.min(lx / 2);
            if m >= 0 {
                let mut k = k_lo;
                while k <= k_hi {
                    let cp0 = binom(j, k) * binom(am, lx - 2 * k);
                    match (am - lx + 2 * k).rem_euclid(4) {
                        0 => cp += cp0,
                        2 => cp -= cp0,
                        _ => {}
                    }
                    k += 1;
                }
                if m == 0 {
                    c0 * cp
                } else {
                    2.0f64.sqrt() * c0 * cp
                }
            } else {
                let mut k = k_lo;
                while k <= k_hi {
                    let cp0 = binom(j, k) * binom(am, lx - 2 * k);
                    match (am - lx + 2 * k).rem_euclid(4) {
                        1 => cp -= cp0,
                        3 => cp += cp0,
                        _ => {}
                    }
                    k += 1;
                }
                -(2.0f64.sqrt()) * c0 * cp
            }
        }

        for l in [5u32, 6u32] {
            let powers = cart_powers(l);
            for m_row in 0..nsph(l) {
                let m = m_row as i64 - l as i64;
                for (cart_col, &(lx, ly, lz)) in powers.iter().enumerate() {
                    let got = c2s_coeff(l, m_row, cart_col).unwrap();
                    let want = xyz2sph_real(lx as i64, ly as i64, lz as i64, m);
                    assert!(
                        (got - want).abs() < 1e-9,
                        "c2s_coeff(l={l}, m_row={m_row}, cart_col={cart_col}) = {got}, \
                         Schlegel-Frisch formula = {want}, diff = {}",
                        (got - want).abs()
                    );
                }
            }
        }
    }

    /// No behavioral regression: l<=4 still returns the FROZEN libcint
    /// coefficients (byte-exact). Spot-check the g-shell (l=4) value the
    /// plan calls out plus the d-shell (l=2) m=-2 xy entry.
    #[test]
    fn c2s_coeff_l_le_4_unchanged() {
        // l=4 row 0, col 1 = 2.503342941796704_6 (xxxy → m=-4).
        assert_eq!(c2s_coeff(4, 0, 1).unwrap(), 2.503_342_941_796_704_6);
        // l=2 row 0, col 1 = 1.092548430592079_2 (xy → m=-2).
        assert_eq!(c2s_coeff(2, 0, 1).unwrap(), 1.092_548_430_592_079_2);
        // l=0 identity.
        assert_eq!(c2s_coeff(0, 0, 0).unwrap(), 1.0);
        // l=1 identity diagonal.
        assert_eq!(c2s_coeff(1, 1, 1).unwrap(), 1.0);
    }
}
