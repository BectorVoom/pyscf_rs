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

// launch_1d CPU work estimates.  One primitive exponential is costed as about
// 100 scalar flops.  The reference GTH basis has up to four primitives per
// contraction; the general kernels additionally form/transform the angular
// components, and deriv1 writes four component blocks.
const EVAL_GTO_S_WORK_PER_LANE: usize = 4 * 100;

/// A-04 — the per-shell squared cutoff table the three device kernels read.
///
/// `None` (every molecular caller, and the periodic driver with its point
/// screen off) becomes `+inf` per shell: `r2 <= +inf` is a constant `true`,
/// so the kernel performs exactly the operations it performed before the
/// test existed and stays bit-identical to it. `Some(rcut2)` is the periodic
/// driver's per-shell `rcut²` (`estimate_rcut_for_eval`), the SAME radius
/// its W-09 block screen already applies per 128-point block; here it is
/// applied per point, which is what the block screen's kept blocks still pay
/// for in full. A table of the wrong length is a caller bug and is treated
/// as "no screen" rather than indexed past.
fn rcut2_table(rcut2: Option<&[f64]>, nbas: usize) -> Vec<f64> {
    // Measurement knob ONLY (`PYSCF_PBC_AO_RCUT2_OVERRIDE=<f64>`): every shell
    // gets this squared radius, so `0` makes every lane take the zero-fill arm
    // and times the launch + index + store floor of the kernel. Garbage output.
    if let Some(v) = rcut2_override() {
        return vec![v; nbas.max(1)];
    }
    match rcut2 {
        Some(r) if r.len() == nbas => r.to_vec(),
        _ => vec![f64::INFINITY; nbas.max(1)],
    }
}

fn rcut2_override() -> Option<f64> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<f64>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PYSCF_PBC_AO_RCUT2_OVERRIDE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
    })
}

/// The primitive exponential of the three device kernels, with the accuracy
/// policy resolved at expansion (`cube_math`'s own `exp` is written the same
/// way): `fast == false` is the glibc bit-exact schedule every gate was
/// measured with; `fast == true` is `cube_math`'s table-free series, NOT
/// bit-exact to the host `f64::exp`, and exists as a MEASUREMENT arm only —
/// `PYSCF_PBC_AO_EXP=fast` (session 4, "is the software exp the lane cost?").
/// Two policies are two kernels with two `KernelId`s; no runtime branch.
#[cube]
#[inline(always)]
fn ao_exp(x: f64, #[comptime] mode: u32) -> f64 {
    if comptime!(mode == 1) {
        cube_math::double::exp::exp(x, cube_math::MathConfig::FAST)
    } else if comptime!(mode == 2) {
        // Measurement arm ONLY (`PYSCF_PBC_AO_EXP=none`): no exponential at
        // all, so the rest of the lane can be timed. The output is garbage.
        x
    } else {
        cube_math::double::exp::exp(x, cube_math::MathConfig::EXACT)
    }
}

/// `PYSCF_PBC_AO_EXP`, read once: `fast` → 1, `none` → 2, anything else → 0
/// (the exact schedule, the only shippable value).
fn ao_exp_mode() -> u32 {
    use std::sync::OnceLock;
    static MODE: OnceLock<u32> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("PYSCF_PBC_AO_EXP") {
        Ok(v) if v.trim().eq_ignore_ascii_case("fast") => 1,
        Ok(v) if v.trim().eq_ignore_ascii_case("none") => 2,
        _ => 0,
    })
}
const EVAL_GTO_GENERAL_WORK_PER_LANE: usize = 4 * 100 + 128;
const EVAL_GTO_DERIV1_WORK_PER_LANE: usize = 4 * 100 + 4 * 128;

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
    rcut2: &Array<f64>,
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
    #[comptime] exp_mode: u32,
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

                // A-04: the per-point reach test. `rcut2[shell]` is the squared
                // cutoff radius of this shell (or +inf, which makes the test a
                // constant `true` and the kernel bit-identical to the unscreened
                // one). Past the cutoff every primitive of the shell is below the
                // precision that sized it, so the contracted radial is dropped
                // whole and `acc` stays 0.
                if r2 <= rcut2[shell_idx] {
                    // ORDERED sequential accumulation over primitives — mirrors the
                    // host l=0 loop exactly (NOT a parallel/tree reduce).
                    for p_idx in 0..nprim {
                        let alpha = env[pe + p_idx];
                        // Coefficient matrix is F-order: ptr_coeff + c_idx*nprim + p.
                        let coef = env[pc + c_idx * nprim + p_idx];
                        acc += coef * ao_exp(-alpha * r2, exp_mode);
                    }
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
fn launch_eval_gto_s_into<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    rcut2: Option<&[f64]>,
    out_handle: &cubecl::server::Handle,
) {
    let nbas = bas.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    let coords_handle = pyscf_algebra::launch::upload::<R, f64>(client, coords);
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));
    let rcut2_host = rcut2_table(rcut2, nbas);
    let rcut2_handle = pyscf_algebra::launch::upload::<R, f64>(client, &rcut2_host);
    let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
    let (cube_count, cube_dim) =
        pyscf_algebra::launch::launch_1d(client, out_len, EVAL_GTO_S_WORK_PER_LANE);

    // SAFETY: handle lengths match the slice lengths uploaded above; the kernel
    // bounds-guards the tail (`if tid < ngrids*nao`). All input Arrays are
    // read-only, `out` is the only `&mut`. `from_raw_parts` consumes the handle,
    // so clone (clones share the binding).
    unsafe {
        eval_gto_sph_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(coords_handle.clone(), coords.len()),
            ArrayArg::from_raw_parts(env_handle.clone(), env.len()),
            ArrayArg::from_raw_parts(bas_handle.clone(), bas.len()),
            ArrayArg::from_raw_parts(atm_handle.clone(), atm.len()),
            ArrayArg::from_raw_parts(ao_loc_handle.clone(), ao_loc.len()),
            ArrayArg::from_raw_parts(rcut2_handle.clone(), rcut2_host.len()),
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
            ao_exp_mode(),
        );
    }
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
    rcut2: &Array<f64>,
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
    #[comptime] exp_mode: u32,
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

        // A-04: per-point reach test (see `rcut2_table`). Both arms write every
        // output element this lane owns — `out` is uninitialised device memory.
        if r2 <= rcut2[shell] {
            for c_idx in 0..nctr {
                // ORDERED sequential contracted radial — mirrors the host l>=1 loop
                // (oracle_sum == strict sequential for nprim<=128), THEN * fac1.
                let mut acc = 0.0_f64;
                for p_idx in 0..nprim {
                    let alpha = env[pe + p_idx];
                    let coef = env[pc + c_idx * nprim + p_idx];
                    acc += coef * ao_exp(-alpha * r2, exp_mode);
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
        } else {
            for c_idx in 0..nctr {
                for m in 0..nsph_l {
                    out[g + (ao_off + c_idx * nsph_l + m) * ngrids] = 0.0_f64;
                }
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
    rcut2: &Array<f64>,
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
    #[comptime] exp_mode: u32,
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

        // A-04: per-point reach test (see `rcut2_table`); the radius was sized
        // for this derivative order by the caller. Both arms write all four
        // component blocks of every output element this lane owns.
        if r2 <= rcut2[shell] {
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
                    let g0 = coef * ao_exp(-alpha * r2, exp_mode);
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
                        let cdx = radial_2a * dx * mono
                            + radial * dpow(dx, lx) * ipow(dy, ly) * ipow(dz, lz);
                        let cdy = radial_2a * dy * mono
                            + radial * ipow(dx, lx) * dpow(dy, ly) * ipow(dz, lz);
                        let cdz = radial_2a * dz * mono
                            + radial * ipow(dx, lx) * ipow(dy, ly) * dpow(dz, lz);
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
        } else {
            for c_idx in 0..nctr {
                for m in 0..nsph_l {
                    let off = g + (ao_off + c_idx * nsph_l + m) * ngrids;
                    out[off] = 0.0_f64;
                    out[comp_stride + off] = 0.0_f64;
                    out[2 * comp_stride + off] = 0.0_f64;
                    out[3 * comp_stride + off] = 0.0_f64;
                }
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
fn launch_eval_gto_general_into<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    maxl: u32,
    rcut2: Option<&[f64]>,
    out_handle: &cubecl::server::Handle,
) -> Result<(), PyscfRsError> {
    let nbas = bas.len() / BAS_SLOTS;
    let out_len = ngrids * nao;

    let t = build_angular_tables(maxl)?;

    // `upload` stages the coordinate block from the slice directly; the
    // `Bytes::from_elems(x.to_vec())` idiom copied it once more on the host
    // first, and coords is the one operand here that scales with the grid.
    let coords_handle = pyscf_algebra::launch::upload::<R, f64>(client, coords);
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));
    let rcut2_host = rcut2_table(rcut2, nbas);
    let rcut2_handle = pyscf_algebra::launch::upload::<R, f64>(client, &rcut2_host);

    let c2s_flat_h = client.create(Bytes::from_elems(t.c2s_flat.clone()));
    let cpow_lx_h = client.create(Bytes::from_elems(t.cpow_lx.clone()));
    let cpow_ly_h = client.create(Bytes::from_elems(t.cpow_ly.clone()));
    let cpow_lz_h = client.create(Bytes::from_elems(t.cpow_lz.clone()));
    let ncart_h = client.create(Bytes::from_elems(t.ncart_by_l.clone()));
    let nsph_h = client.create(Bytes::from_elems(t.nsph_by_l.clone()));
    let fac1_h = client.create(Bytes::from_elems(t.fac1_by_l.clone()));
    let c2s_off_h = client.create(Bytes::from_elems(t.c2s_off_by_l.clone()));
    let cpow_off_h = client.create(Bytes::from_elems(t.cpow_off_by_l.clone()));

    let lanes = ngrids * nbas;
    let (cube_count, cube_dim) =
        pyscf_algebra::launch::launch_1d(client, lanes, EVAL_GTO_GENERAL_WORK_PER_LANE);

    // SAFETY: every handle length matches the slice length uploaded above; the
    // kernel bounds-guards the tail (`if tid < ngrids*nbas`). All input Arrays
    // are read-only, `out` is the only `&mut`. `from_raw_parts` consumes the
    // handle, so clone (clones share the binding).
    unsafe {
        eval_gto_sph_kernel_general::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
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
            ArrayArg::from_raw_parts(rcut2_handle.clone(), rcut2_host.len()),
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
            ao_exp_mode(),
        );
    }

    Ok(())
}

/// Host-slice launcher for the general l 0..=4 **deriv1** kernel. Clones
/// `launch_eval_gto_general` byte-for-byte EXCEPT: `out_len = 4*ngrids*nao`,
/// launches `eval_gto_sph_deriv1_kernel`, and passes the extra `comp_stride =
/// ngrids*nao` scalar AFTER the existing trailing scalar args. Reads back the
/// F-order `[4, ngrids, nao]` buffer. `maxl` must be `<= 4` (caller guarantees).
#[allow(clippy::too_many_arguments)]
fn launch_eval_gto_deriv1_into<R: Runtime>(
    client: &ComputeClient<R>,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    maxl: u32,
    rcut2: Option<&[f64]>,
    out_handle: &cubecl::server::Handle,
) -> Result<(), PyscfRsError> {
    let nbas = bas.len() / BAS_SLOTS;
    let comp_stride = ngrids * nao;
    let out_len = 4 * comp_stride;

    let t = build_angular_tables(maxl)?;

    // `upload` stages the coordinate block from the slice directly; the
    // `Bytes::from_elems(x.to_vec())` idiom copied it once more on the host
    // first, and coords is the one operand here that scales with the grid.
    let coords_handle = pyscf_algebra::launch::upload::<R, f64>(client, coords);
    let env_handle = client.create(Bytes::from_elems(env.to_vec()));
    let bas_handle = client.create(Bytes::from_elems(bas.to_vec()));
    let atm_handle = client.create(Bytes::from_elems(atm.to_vec()));
    let ao_loc_handle = client.create(Bytes::from_elems(ao_loc.to_vec()));
    let rcut2_host = rcut2_table(rcut2, nbas);
    let rcut2_handle = pyscf_algebra::launch::upload::<R, f64>(client, &rcut2_host);

    let c2s_flat_h = client.create(Bytes::from_elems(t.c2s_flat.clone()));
    let cpow_lx_h = client.create(Bytes::from_elems(t.cpow_lx.clone()));
    let cpow_ly_h = client.create(Bytes::from_elems(t.cpow_ly.clone()));
    let cpow_lz_h = client.create(Bytes::from_elems(t.cpow_lz.clone()));
    let ncart_h = client.create(Bytes::from_elems(t.ncart_by_l.clone()));
    let nsph_h = client.create(Bytes::from_elems(t.nsph_by_l.clone()));
    let fac1_h = client.create(Bytes::from_elems(t.fac1_by_l.clone()));
    let c2s_off_h = client.create(Bytes::from_elems(t.c2s_off_by_l.clone()));
    let cpow_off_h = client.create(Bytes::from_elems(t.cpow_off_by_l.clone()));

    let lanes = ngrids * nbas;
    let (cube_count, cube_dim) =
        pyscf_algebra::launch::launch_1d(client, lanes, EVAL_GTO_DERIV1_WORK_PER_LANE);

    // SAFETY: every handle length matches the slice length uploaded above; the
    // kernel bounds-guards the tail (`if tid < ngrids*nbas`). All input Arrays
    // are read-only, `out` (len 4*ngrids*nao) is the only `&mut`.
    unsafe {
        eval_gto_sph_deriv1_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
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
            ArrayArg::from_raw_parts(rcut2_handle.clone(), rcut2_host.len()),
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
            ao_exp_mode(),
        );
    }

    Ok(())
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

/// Opaque, device-resident AO block shared with the periodic Bloch accumulator.
/// The CubeCL handle stays private so method crates cannot cross the ALG-06 wall.
pub struct AoBlockDevice {
    handle: cubecl::server::Handle,
    len: usize,
    shape: Vec<usize>,
}

impl core::fmt::Debug for AoBlockDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AoBlockDevice")
            .field("len", &self.len)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl AoBlockDevice {
    /// A block over an existing (possibly offset) device handle — K-09's
    /// image-batch slots. Crate-private: the handle never crosses ALG-06.
    pub(crate) fn from_handle(
        handle: cubecl::server::Handle,
        len: usize,
        shape: Vec<usize>,
    ) -> Self {
        Self { handle, len, shape }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub(crate) fn handle(&self) -> &cubecl::server::Handle {
        &self.handle
    }

    /// Upload a host AO block for uncommon evaluator fallbacks while preserving
    /// the opaque-handle dependency wall.
    pub fn from_values(client: &AlgebraClient, values: &[f64], shape: Vec<usize>) -> Self {
        upload_ao_block(client, values, shape)
    }

    pub fn into_values(self, client: &AlgebraClient) -> Vec<f64> {
        if self.len == 0 {
            return Vec::new();
        }
        dispatch_backend!(client, c, Rt, {
            let bytes = c.read(vec![self.handle.clone()]);
            bytemuck::cast_slice::<u8, f64>(&bytes[0]).to_vec()
        })
    }
}

fn upload_ao_block(client: &AlgebraClient, values: &[f64], shape: Vec<usize>) -> AoBlockDevice {
    let handle = dispatch_backend!(
        client,
        c,
        Rt,
        pyscf_algebra::launch::upload::<Rt, f64>(c, values)
    );
    AoBlockDevice {
        handle,
        len: values.len(),
        shape,
    }
}

/// Evaluate a scalar AO block without reading the device result back to the host.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_into(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    spherical: bool,
) -> Result<AoBlockDevice, PyscfRsError> {
    eval_gto_sph_into_screened(
        client, coords, ngrids, atm, bas, env, ao_loc, nao, spherical, None,
    )
}

/// [`eval_gto_sph_into`] with A-04's per-point, per-shell reach test.
///
/// `rcut2` is one SQUARED cutoff radius per shell (`bas.len() / BAS_SLOTS`
/// entries). A grid point farther than that from the shell's centre gets an
/// exact `0.0` for every AO of the shell instead of a contracted radial that
/// the radius was chosen to bound below the caller's precision. **This drops
/// terms and so changes the result** — by construction by less than the same
/// bound the periodic block screen (W-09) already accepts, but it is not
/// bit-exact against `None`, which is the unscreened kernel and the only form
/// the molecular callers use. The periodic driver owns the radius
/// (`estimate_rcut_for_eval`) and the kill switch.
///
/// The host fallback (no device kernel for the basis) ignores `rcut2`: it is
/// the reference path and stays unscreened.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_into_screened(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    spherical: bool,
    rcut2: Option<&[f64]>,
) -> Result<AoBlockDevice, PyscfRsError> {
    let all_s = !bas.is_empty() && bas.chunks_exact(BAS_SLOTS).all(|row| row[ANG_OF] == 0);
    let out_len = ngrids * nao;
    if all_s && out_len > 0 {
        let _ = spherical;
        return Ok(dispatch_backend!(client, c, Rt, {
            let out = c.empty(out_len * core::mem::size_of::<f64>());
            launch_eval_gto_s_into::<Rt>(
                c, coords, ngrids, atm, bas, env, ao_loc, nao, rcut2, &out,
            );
            AoBlockDevice {
                handle: out,
                len: out_len,
                shape: vec![ngrids, nao],
            }
        }));
    }
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;
    if !bas.is_empty() && maxl <= 4 && out_len > 0 {
        let _ = spherical;
        return dispatch_backend!(client, c, Rt, {
            let out = c.empty(out_len * core::mem::size_of::<f64>());
            launch_eval_gto_general_into::<Rt>(
                c, coords, ngrids, atm, bas, env, ao_loc, nao, maxl, rcut2, &out,
            )?;
            Ok(AoBlockDevice {
                handle: out,
                len: out_len,
                shape: vec![ngrids, nao],
            })
        });
    }
    let host = eval_gto_sph_cpu(coords, ngrids, atm, bas, env, ao_loc, nao, spherical)?;
    Ok(upload_ao_block(client, &host.values, host.shape))
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
    let shape = vec![ngrids, nao];
    let values = eval_gto_sph_into(
        client, coords, ngrids, atm, bas, env, ao_loc, nao, spherical,
    )?
    .into_values(client);
    Ok(EvalGtoBuffers { values, shape })
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
pub fn eval_gto_sph_deriv1_into(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
) -> Result<AoBlockDevice, PyscfRsError> {
    eval_gto_sph_deriv1_into_screened(client, coords, ngrids, atm, bas, env, ao_loc, nao, None)
}

// ---------------------------------------------------------------------------
// Session 5 — A-05 / A-06: the image-invariant operands uploaded ONCE per
// periodic AO evaluation, and one launch per image BATCH.
//
// Session 4 measured (`PYSCF_PBC_AO_SKIP_K08=1` + `RCUT2_OVERRIDE=0`) that a
// launch whose every lane does nothing but its zero-fill store still costs
// 0.9 ms (deriv 0) / 2.2 ms (deriv 1) on the CPU runtime — 404 / 978 ms over
// 454 images — and that a lane's arithmetic costs ~0 on top of that. So what
// the AO stage pays for is per LAUNCH: fourteen small uploads of tables that
// never change between images (`env`, `bas`, `atm`, `ao_loc`, `rcut2` and the
// nine angular tables), the output allocation, and the runtime's dispatch.
// `EvalGtoDeviceContext` hoists the uploads (`11_launch_overhead_and_transfers.md`
// §2); the `*_batched` kernels below collapse the launches (§5): one launch
// evaluates every image of a K-09 batch straight into that batch's slots.
//
// The batched kernels' lane bodies are COPIES of the per-image kernels above,
// operand for operand, with the output index rebased on the image's slot
// (`obase + g + ao·npts`). The per-image kernels are deliberately left
// untouched so that `PYSCF_PBC_AO_IMAGE_BATCH=1` stays an independent
// reference and `tests/eval_ao_image_batch.rs` compares two code paths, not
// one path with itself.
// ---------------------------------------------------------------------------

/// The nine angular tables of the general/deriv1 kernels, resident.
struct AngularDevice {
    c2s_flat: cubecl::server::Handle,
    cpow_lx: cubecl::server::Handle,
    cpow_ly: cubecl::server::Handle,
    cpow_lz: cubecl::server::Handle,
    ncart_by_l: cubecl::server::Handle,
    nsph_by_l: cubecl::server::Handle,
    fac1_by_l: cubecl::server::Handle,
    c2s_off_by_l: cubecl::server::Handle,
    cpow_off_by_l: cubecl::server::Handle,
    lens: [usize; 9],
}

/// The image-invariant device operands of the three AO kernels, uploaded once
/// per caller loop (A-05). Handles stay private (ALG-06).
pub struct EvalGtoDeviceContext {
    env: cubecl::server::Handle,
    bas: cubecl::server::Handle,
    atm: cubecl::server::Handle,
    ao_loc: cubecl::server::Handle,
    rcut2: cubecl::server::Handle,
    lens: [usize; 5],
    angular: Option<AngularDevice>,
    nbas: usize,
    nao: usize,
    all_s: bool,
    exp_mode: u32,
}

impl core::fmt::Debug for EvalGtoDeviceContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EvalGtoDeviceContext")
            .field("nbas", &self.nbas)
            .field("nao", &self.nao)
            .field("all_s", &self.all_s)
            .finish_non_exhaustive()
    }
}

impl EvalGtoDeviceContext {
    /// Upload the basis once. Requires [`eval_gto_device_capable`].
    ///
    /// # Errors
    /// [`PyscfRsError::Core`] when the basis has no device kernel.
    pub fn new(
        client: &AlgebraClient,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        ao_loc: &[i32],
        nao: usize,
        rcut2: Option<&[f64]>,
    ) -> Result<Self, PyscfRsError> {
        if !eval_gto_device_capable(bas) {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                "EvalGtoDeviceContext: basis has no device kernel (l > 4 or empty)".into(),
            )));
        }
        let nbas = bas.len() / BAS_SLOTS;
        let all_s = bas.chunks_exact(BAS_SLOTS).all(|row| row[ANG_OF] == 0);
        let maxl = bas
            .chunks_exact(BAS_SLOTS)
            .map(|row| row[ANG_OF])
            .max()
            .unwrap_or(0) as u32;
        let rcut2_host = rcut2_table(rcut2, nbas);
        // Built for every basis, all-s included (then l = 0 only, a few
        // words). The s-kernel covers an all-s basis at deriv 0 only; at deriv
        // 1 it runs the general kernels, which need these tables. Leaving them
        // out made the K-10 fused path refuse He/STO-3G under PBE
        // (`he_all_electron_krks_converges_and_integrates`), and the K-09 path
        // had been rebuilding the same l = 0 tables on every call to cover it.
        let angular_host = Some(build_angular_tables(maxl)?);
        Ok(dispatch_backend!(client, c, Rt, {
            let angular = angular_host.as_ref().map(|t| AngularDevice {
                c2s_flat: pyscf_algebra::launch::upload::<Rt, f64>(c, &t.c2s_flat),
                cpow_lx: c.create_from_slice(bytemuck::cast_slice(&t.cpow_lx)),
                cpow_ly: c.create_from_slice(bytemuck::cast_slice(&t.cpow_ly)),
                cpow_lz: c.create_from_slice(bytemuck::cast_slice(&t.cpow_lz)),
                ncart_by_l: c.create_from_slice(bytemuck::cast_slice(&t.ncart_by_l)),
                nsph_by_l: c.create_from_slice(bytemuck::cast_slice(&t.nsph_by_l)),
                fac1_by_l: pyscf_algebra::launch::upload::<Rt, f64>(c, &t.fac1_by_l),
                c2s_off_by_l: c.create_from_slice(bytemuck::cast_slice(&t.c2s_off_by_l)),
                cpow_off_by_l: c.create_from_slice(bytemuck::cast_slice(&t.cpow_off_by_l)),
                lens: [
                    t.c2s_flat.len(),
                    t.cpow_lx.len(),
                    t.cpow_ly.len(),
                    t.cpow_lz.len(),
                    t.ncart_by_l.len(),
                    t.nsph_by_l.len(),
                    t.fac1_by_l.len(),
                    t.c2s_off_by_l.len(),
                    t.cpow_off_by_l.len(),
                ],
            });
            Self {
                env: pyscf_algebra::launch::upload::<Rt, f64>(c, env),
                bas: c.create_from_slice(bytemuck::cast_slice(bas)),
                atm: c.create_from_slice(bytemuck::cast_slice(atm)),
                ao_loc: c.create_from_slice(bytemuck::cast_slice(ao_loc)),
                rcut2: pyscf_algebra::launch::upload::<Rt, f64>(c, &rcut2_host),
                lens: [
                    env.len(),
                    bas.len(),
                    atm.len(),
                    ao_loc.len(),
                    rcut2_host.len(),
                ],
                angular,
                nbas,
                nao,
                all_s,
                exp_mode: ao_exp_mode(),
            }
        }))
    }

    pub fn nao(&self) -> usize {
        self.nao
    }
    pub fn nbas(&self) -> usize {
        self.nbas
    }
}

/// One image of a batched AO launch: how many grid points it covers.
/// Its coordinates are the next `3 · npts` reals of the concatenated F-order
/// coordinate buffer; its output is the next K-09 slot.
#[derive(Debug, Clone, Copy)]
pub struct EvalGtoImage {
    pub npts: usize,
}

/// s-shell lane — the body of `eval_gto_sph_kernel`, rebased on `obase`.
#[allow(clippy::too_many_arguments)]
#[cube]
fn eval_gto_s_lane(
    gx: f64,
    gy: f64,
    gz: f64,
    ao_idx: usize,
    g: usize,
    stride: usize,
    obase: usize,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    ao_loc: &Array<i32>,
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    nbas: usize,
    y00: f64,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let mut acc = 0.0_f64;
    for shell_idx in 0..nbas {
        let bas_row = shell_idx * bas_slots;
        let ao_off = ao_loc[shell_idx] as usize;
        let nctr = bas[bas_row + nctr_of] as usize;
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

            if r2 <= rcut2[shell_idx] {
                for p_idx in 0..nprim {
                    let alpha = env[pe + p_idx];
                    let coef = env[pc + c_idx * nprim + p_idx];
                    acc += coef * ao_exp(-alpha * r2, exp_mode);
                }
            }
        }
    }
    out[obase + g + ao_idx * stride] = acc * y00;
}

/// General l 0..=4 lane — the body of `eval_gto_sph_kernel_general`, rebased.
#[allow(clippy::too_many_arguments)]
#[cube]
fn eval_gto_general_lane(
    gx: f64,
    gy: f64,
    gz: f64,
    shell: usize,
    g: usize,
    stride: usize,
    obase: usize,
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
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
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

    if r2 <= rcut2[shell] {
        for c_idx in 0..nctr {
            let mut acc = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                acc += coef * ao_exp(-alpha * r2, exp_mode);
            }
            let radial = acc * fac1;
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
                out[obase + g + (ao_off + c_idx * nsph_l + m) * stride] = v;
            }
        }
    } else {
        for c_idx in 0..nctr {
            for m in 0..nsph_l {
                out[obase + g + (ao_off + c_idx * nsph_l + m) * stride] = 0.0_f64;
            }
        }
    }
}

/// deriv1 lane — the body of `eval_gto_sph_deriv1_kernel`, rebased. The four
/// component blocks are `comp_stride` (= `npts · nao` of THIS image) apart.
#[allow(clippy::too_many_arguments)]
#[cube]
fn eval_gto_deriv1_lane(
    gx: f64,
    gy: f64,
    gz: f64,
    shell: usize,
    g: usize,
    stride: usize,
    obase: usize,
    comp_stride: usize,
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
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
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

    if r2 <= rcut2[shell] {
        for c_idx in 0..nctr {
            let mut acc = 0.0_f64;
            let mut acc2a = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                let g0 = coef * ao_exp(-alpha * r2, exp_mode);
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
                let off = obase + g + (ao_off + c_idx * nsph_l + m) * stride;
                out[off] = v;
                out[comp_stride + off] = vx;
                out[2 * comp_stride + off] = vy;
                out[3 * comp_stride + off] = vz;
            }
        }
    } else {
        for c_idx in 0..nctr {
            for m in 0..nsph_l {
                let off = obase + g + (ao_off + c_idx * nsph_l + m) * stride;
                out[off] = 0.0_f64;
                out[comp_stride + off] = 0.0_f64;
                out[2 * comp_stride + off] = 0.0_f64;
                out[3 * comp_stride + off] = 0.0_f64;
            }
        }
    }
}

/// Which image owns lane `tid`: the largest `m` with `lane0[m] <= tid`.
/// `lane0` has `nimg + 1` entries, `lane0[0] == 0`, and `tid < lane0[nimg]`.
///
/// A range `for` with a single `if` and no `else`, on purpose: the first
/// version was a `while` binary search with an `if/else` body and the CPU
/// runtime's MLIR lowering rejected it at run time ("operation with block
/// successors must terminate its parent block", `cubecl-cpu module.rs:94`);
/// `Cubecl_loop_control.md` prefers range loops, and this shape is the one
/// every other kernel in this crate already compiles. At most 32 iterations
/// of one compare, against the lane's hundreds of operations; adjacent lanes
/// agree on `m`, so a plane does not diverge on it.
#[cube]
fn eval_gto_image_of(lane0: &Array<u32>, nimg: usize, tid: usize) -> usize {
    let mut m = 0usize;
    for i in 1..nimg {
        if lane0[i] as usize <= tid {
            m = i;
        }
    }
    m
}

/// A-06: every image of a batch in ONE launch, s-shell basis. Lane = `(image,
/// g, ao)`; each image's block lands F-order in its own slot.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_kernel_batched(
    coords: &Array<f64>,
    img_lane0: &Array<u32>,
    img_npts: &Array<u32>,
    img_coord_off: &Array<u32>,
    img_out_off: &Array<u32>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    ao_loc: &Array<i32>,
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    nimg: usize,
    nlanes: usize,
    nbas: usize,
    nao: usize,
    y00: f64,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nlanes {
        let m = eval_gto_image_of(img_lane0, nimg, tid);
        let local = tid - img_lane0[m] as usize;
        let npts = img_npts[m] as usize;
        let g = local % npts;
        let ao_idx = local / npts;
        let cbase = img_coord_off[m] as usize;
        let gx = coords[cbase + g];
        let gy = coords[cbase + npts + g];
        let gz = coords[cbase + 2 * npts + g];
        let _ = nao;
        eval_gto_s_lane(
            gx,
            gy,
            gz,
            ao_idx,
            g,
            npts,
            img_out_off[m] as usize,
            env,
            bas,
            atm,
            ao_loc,
            rcut2,
            out,
            nbas,
            y00,
            atm_slots,
            bas_slots,
            atom_of,
            nprim_of,
            nctr_of,
            ptr_exp,
            ptr_coeff,
            ptr_coord,
            exp_mode,
        );
    }
}

/// A-06: every image of a batch in ONE launch, general l 0..=4. Lane =
/// `(image, g, shell)`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_kernel_general_batched(
    coords: &Array<f64>,
    img_lane0: &Array<u32>,
    img_npts: &Array<u32>,
    img_coord_off: &Array<u32>,
    img_out_off: &Array<u32>,
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
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    nimg: usize,
    nlanes: usize,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nlanes {
        let m = eval_gto_image_of(img_lane0, nimg, tid);
        let local = tid - img_lane0[m] as usize;
        let npts = img_npts[m] as usize;
        let g = local % npts;
        let shell = local / npts;
        let cbase = img_coord_off[m] as usize;
        let gx = coords[cbase + g];
        let gy = coords[cbase + npts + g];
        let gz = coords[cbase + 2 * npts + g];
        eval_gto_general_lane(
            gx,
            gy,
            gz,
            shell,
            g,
            npts,
            img_out_off[m] as usize,
            env,
            bas,
            atm,
            ao_loc,
            c2s_flat,
            cpow_lx,
            cpow_ly,
            cpow_lz,
            ncart_by_l,
            nsph_by_l,
            fac1_by_l,
            c2s_off_by_l,
            cpow_off_by_l,
            rcut2,
            out,
            atm_slots,
            bas_slots,
            atom_of,
            ang_of,
            nprim_of,
            nctr_of,
            ptr_exp,
            ptr_coeff,
            ptr_coord,
            exp_mode,
        );
    }
}

/// A-06: every image of a batch in ONE launch, deriv1. Lane = `(image, g,
/// shell)`; `comp_stride = npts · nao` per image.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_gto_sph_deriv1_kernel_batched(
    coords: &Array<f64>,
    img_lane0: &Array<u32>,
    img_npts: &Array<u32>,
    img_coord_off: &Array<u32>,
    img_out_off: &Array<u32>,
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
    rcut2: &Array<f64>,
    out: &mut Array<f64>,
    nimg: usize,
    nlanes: usize,
    nao: usize,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < nlanes {
        let m = eval_gto_image_of(img_lane0, nimg, tid);
        let local = tid - img_lane0[m] as usize;
        let npts = img_npts[m] as usize;
        let g = local % npts;
        let shell = local / npts;
        let cbase = img_coord_off[m] as usize;
        let gx = coords[cbase + g];
        let gy = coords[cbase + npts + g];
        let gz = coords[cbase + 2 * npts + g];
        eval_gto_deriv1_lane(
            gx,
            gy,
            gz,
            shell,
            g,
            npts,
            img_out_off[m] as usize,
            npts * nao,
            env,
            bas,
            atm,
            ao_loc,
            c2s_flat,
            cpow_lx,
            cpow_ly,
            cpow_lz,
            ncart_by_l,
            nsph_by_l,
            fac1_by_l,
            c2s_off_by_l,
            cpow_off_by_l,
            rcut2,
            out,
            atm_slots,
            bas_slots,
            atom_of,
            ang_of,
            nprim_of,
            nctr_of,
            ptr_exp,
            ptr_coeff,
            ptr_coord,
            exp_mode,
        );
    }
}

/// A-06: evaluate every image of `images` in one launch, straight into the
/// batch's slots `first_slot..first_slot + images.len()`, in order.
///
/// `coords` is the images' shifted grids concatenated, each F-order
/// (`x[0..npts], y[..], z[..]`); `deriv1` selects the four-component kernel
/// (the slot layout is then `[4, npts, nao]`). Bit-identical to evaluating
/// each image alone: every lane computes the per-image kernel's expression
/// on the same operands and writes it to the same slot position.
///
/// # Errors
/// [`PyscfRsError::Core`] on a shape disagreement (coordinate count, slot
/// capacity, an image too large for its slot, or offsets past `u32`).
pub fn eval_gto_batch_into_image_batch(
    client: &AlgebraClient,
    ctx: &EvalGtoDeviceContext,
    deriv1: bool,
    coords: &[f64],
    images: &[EvalGtoImage],
    first_slot: usize,
    batch: &crate::pbc::AoImageBatch,
) -> Result<(), PyscfRsError> {
    let nimg = images.len();
    let comp = if deriv1 { 4 } else { 1 };
    let per_lane_shells = if ctx.all_s && !deriv1 {
        ctx.nao
    } else {
        ctx.nbas
    };
    if nimg == 0 {
        return Ok(());
    }
    if first_slot + nimg > batch.capacity() {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "eval_gto_batch: slots {first_slot}..{} exceed the batch capacity {}",
                first_slot + nimg,
                batch.capacity()
            ),
        )));
    }
    let mut lane0 = Vec::with_capacity(nimg + 1);
    let mut npts_v = Vec::with_capacity(nimg);
    let mut coord_off = Vec::with_capacity(nimg);
    let mut out_off = Vec::with_capacity(nimg);
    let (mut lanes, mut coff) = (0usize, 0usize);
    lane0.push(0u32);
    for (m, img) in images.iter().enumerate() {
        let block = comp * img.npts * ctx.nao;
        if block > batch.block_len() {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!(
                    "eval_gto_batch: image {m} needs {block} reals, slot holds {}",
                    batch.block_len()
                ),
            )));
        }
        npts_v.push(img.npts as u32);
        coord_off.push(coff as u32);
        out_off.push(((first_slot + m) * batch.block_len()) as u32);
        coff += 3 * img.npts;
        lanes += img.npts * per_lane_shells;
        lane0.push(lanes as u32);
    }
    if coff != coords.len() || coff > u32::MAX as usize || lanes > u32::MAX as usize {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "eval_gto_batch: coordinate buffer {} reals for {coff} expected, {lanes} lanes",
                coords.len()
            ),
        )));
    }
    if (first_slot + nimg) * batch.block_len() > u32::MAX as usize {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "eval_gto_batch: slot offsets exceed u32".into(),
        )));
    }
    if lanes == 0 {
        return Ok(());
    }
    let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
    let total_out = batch.capacity() * batch.block_len();
    dispatch_backend!(client, c, Rt, {
        let coords_h = pyscf_algebra::launch::upload::<Rt, f64>(c, coords);
        let lane0_h = c.create_from_slice(bytemuck::cast_slice(&lane0));
        let npts_h = c.create_from_slice(bytemuck::cast_slice(&npts_v));
        let coff_h = c.create_from_slice(bytemuck::cast_slice(&coord_off));
        let ooff_h = c.create_from_slice(bytemuck::cast_slice(&out_off));
        let out_h = batch.buffer().clone();
        let [env_len, bas_len, atm_len, ao_loc_len, rcut2_len] = ctx.lens;
        // SAFETY: every handle length is the length of the slice it was
        // created from; the kernels guard `tid < nlanes`; `out` is the only
        // `&mut` and every lane's slot offset was bounds-checked above.
        if ctx.all_s && !deriv1 {
            let (count, dim) = pyscf_algebra::launch::launch_1d(c, lanes, EVAL_GTO_S_WORK_PER_LANE);
            unsafe {
                eval_gto_sph_kernel_batched::launch_unchecked::<Rt>(
                    c,
                    count,
                    dim,
                    ArrayArg::from_raw_parts(coords_h, coords.len()),
                    ArrayArg::from_raw_parts(lane0_h, nimg + 1),
                    ArrayArg::from_raw_parts(npts_h, nimg),
                    ArrayArg::from_raw_parts(coff_h, nimg),
                    ArrayArg::from_raw_parts(ooff_h, nimg),
                    ArrayArg::from_raw_parts(ctx.env.clone(), env_len),
                    ArrayArg::from_raw_parts(ctx.bas.clone(), bas_len),
                    ArrayArg::from_raw_parts(ctx.atm.clone(), atm_len),
                    ArrayArg::from_raw_parts(ctx.ao_loc.clone(), ao_loc_len),
                    ArrayArg::from_raw_parts(ctx.rcut2.clone(), rcut2_len),
                    ArrayArg::from_raw_parts(out_h, total_out),
                    nimg,
                    lanes,
                    ctx.nbas,
                    ctx.nao,
                    y00,
                    ATM_SLOTS,
                    BAS_SLOTS,
                    ATOM_OF,
                    NPRIM_OF,
                    NCTR_OF,
                    PTR_EXP,
                    PTR_COEFF,
                    PTR_COORD,
                    ctx.exp_mode,
                );
            }
            return Ok(());
        }
        let Some(ang) = ctx.angular.as_ref() else {
            // `EvalGtoDeviceContext::new` builds the tables for every basis.
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                "K-09: no angular tables in the context".into(),
            )));
        };
        launch_batched_angular::<Rt>(
            c,
            ctx,
            ang,
            deriv1,
            coords_h,
            coords.len(),
            lane0_h,
            npts_h,
            coff_h,
            ooff_h,
            out_h,
            total_out,
            nimg,
            lanes,
        );
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_batched_angular<R: Runtime>(
    c: &ComputeClient<R>,
    ctx: &EvalGtoDeviceContext,
    ang: &AngularDevice,
    deriv1: bool,
    coords_h: cubecl::server::Handle,
    coords_len: usize,
    lane0_h: cubecl::server::Handle,
    npts_h: cubecl::server::Handle,
    coff_h: cubecl::server::Handle,
    ooff_h: cubecl::server::Handle,
    out_h: cubecl::server::Handle,
    total_out: usize,
    nimg: usize,
    lanes: usize,
) {
    let [env_len, bas_len, atm_len, ao_loc_len, rcut2_len] = ctx.lens;
    let al = ang.lens;
    // SAFETY: as in `eval_gto_batch_into_image_batch`.
    if deriv1 {
        let (count, dim) =
            pyscf_algebra::launch::launch_1d(c, lanes, EVAL_GTO_DERIV1_WORK_PER_LANE);
        unsafe {
            eval_gto_sph_deriv1_kernel_batched::launch_unchecked::<R>(
                c,
                count,
                dim,
                ArrayArg::from_raw_parts(coords_h, coords_len),
                ArrayArg::from_raw_parts(lane0_h, nimg + 1),
                ArrayArg::from_raw_parts(npts_h, nimg),
                ArrayArg::from_raw_parts(coff_h, nimg),
                ArrayArg::from_raw_parts(ooff_h, nimg),
                ArrayArg::from_raw_parts(ctx.env.clone(), env_len),
                ArrayArg::from_raw_parts(ctx.bas.clone(), bas_len),
                ArrayArg::from_raw_parts(ctx.atm.clone(), atm_len),
                ArrayArg::from_raw_parts(ctx.ao_loc.clone(), ao_loc_len),
                ArrayArg::from_raw_parts(ang.c2s_flat.clone(), al[0]),
                ArrayArg::from_raw_parts(ang.cpow_lx.clone(), al[1]),
                ArrayArg::from_raw_parts(ang.cpow_ly.clone(), al[2]),
                ArrayArg::from_raw_parts(ang.cpow_lz.clone(), al[3]),
                ArrayArg::from_raw_parts(ang.ncart_by_l.clone(), al[4]),
                ArrayArg::from_raw_parts(ang.nsph_by_l.clone(), al[5]),
                ArrayArg::from_raw_parts(ang.fac1_by_l.clone(), al[6]),
                ArrayArg::from_raw_parts(ang.c2s_off_by_l.clone(), al[7]),
                ArrayArg::from_raw_parts(ang.cpow_off_by_l.clone(), al[8]),
                ArrayArg::from_raw_parts(ctx.rcut2.clone(), rcut2_len),
                ArrayArg::from_raw_parts(out_h, total_out),
                nimg,
                lanes,
                ctx.nao,
                ATM_SLOTS,
                BAS_SLOTS,
                ATOM_OF,
                ANG_OF,
                NPRIM_OF,
                NCTR_OF,
                PTR_EXP,
                PTR_COEFF,
                PTR_COORD,
                ctx.exp_mode,
            );
        }
    } else {
        let (count, dim) =
            pyscf_algebra::launch::launch_1d(c, lanes, EVAL_GTO_GENERAL_WORK_PER_LANE);
        unsafe {
            eval_gto_sph_kernel_general_batched::launch_unchecked::<R>(
                c,
                count,
                dim,
                ArrayArg::from_raw_parts(coords_h, coords_len),
                ArrayArg::from_raw_parts(lane0_h, nimg + 1),
                ArrayArg::from_raw_parts(npts_h, nimg),
                ArrayArg::from_raw_parts(coff_h, nimg),
                ArrayArg::from_raw_parts(ooff_h, nimg),
                ArrayArg::from_raw_parts(ctx.env.clone(), env_len),
                ArrayArg::from_raw_parts(ctx.bas.clone(), bas_len),
                ArrayArg::from_raw_parts(ctx.atm.clone(), atm_len),
                ArrayArg::from_raw_parts(ctx.ao_loc.clone(), ao_loc_len),
                ArrayArg::from_raw_parts(ang.c2s_flat.clone(), al[0]),
                ArrayArg::from_raw_parts(ang.cpow_lx.clone(), al[1]),
                ArrayArg::from_raw_parts(ang.cpow_ly.clone(), al[2]),
                ArrayArg::from_raw_parts(ang.cpow_lz.clone(), al[3]),
                ArrayArg::from_raw_parts(ang.ncart_by_l.clone(), al[4]),
                ArrayArg::from_raw_parts(ang.nsph_by_l.clone(), al[5]),
                ArrayArg::from_raw_parts(ang.fac1_by_l.clone(), al[6]),
                ArrayArg::from_raw_parts(ang.c2s_off_by_l.clone(), al[7]),
                ArrayArg::from_raw_parts(ang.cpow_off_by_l.clone(), al[8]),
                ArrayArg::from_raw_parts(ctx.rcut2.clone(), rcut2_len),
                ArrayArg::from_raw_parts(out_h, total_out),
                nimg,
                lanes,
                ATM_SLOTS,
                BAS_SLOTS,
                ATOM_OF,
                ANG_OF,
                NPRIM_OF,
                NCTR_OF,
                PTR_EXP,
                PTR_COEFF,
                PTR_COORD,
                ctx.exp_mode,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// K-10 — the fused periodic AO evaluation (session 5, user-authorised over
// PBC-MASTER-PLAN plan 10-04's "do not write a new AO evaluator").
//
// After K-09 and A-06 the cold pass was one third AO evaluation and two
// thirds accumulate, and the accumulate's remaining traffic per image was
// the AO block itself: `n` reals written by the evaluation kernel and read
// back by the accumulate. K-10 never materialises it. One lane per
// `(g, shell)` over the UNSHIFTED grid: for each image `m` of the batch it
// forms `r_g − L_m` in-kernel (the same IEEE subtraction the host did), runs
// the per-image kernel's lane body into a local value array, and then, per
// output `(c, ao)` of the shell and per k-point, adds `pr[m,k]·v_m` into the
// resident planes IN IMAGE ORDER. Each `(k, p)` therefore receives exactly
// the additions the per-image path performed, in the same order — a point a
// screened image does not keep receives none from it (`keep[m][block(g)]`,
// the W-09 decision) — so the planes are bit-identical. What crosses to the
// device per batch: `3·B` lattice vectors, `B·nkpts` phases and `B·nblocks`
// keep flags; per call: the grid once and the basis tables once (A-05).
//
// The per-lane value array is `FUSED_VALS_CAP` reals; the host sizes the
// batch so `B · Q_max` fits, `Q_max = comp · max(nctr·nsph)` over shells.
// ---------------------------------------------------------------------------

/// Reals of per-image AO values one fused lane may hold (`B · Q_max`).
///
/// 512, not more, because of a MEASURED CPU-runtime limit whose mechanism is
/// UNVERIFIED: with 2048 the worker threads overflowed their 64 MB stacks
/// (`CUBECL_CPU_STACK_MB=128` ran, `80` did not), with 512 they run at the
/// default and at 32 MB — i.e. the kernel's stack frame grows ~50 KB per
/// element of this local array, three orders more than its 8 B. Whatever the
/// lowering does with a large `LocalArray`, the budget is `B · Q_max <= 512`:
/// 32 images at gth-szv (`Q = 12` at deriv 1), 12 at gth-dzvp (`Q = 40`).
pub const FUSED_VALS_CAP: usize = 512;

/// The device-resident unshifted grid, F-order (`x[0..ngrids], y, z`),
/// uploaded once per periodic AO evaluation. Opaque (ALG-06).
pub struct AoGridDevice {
    handle: cubecl::server::Handle,
    ngrids: usize,
}

impl core::fmt::Debug for AoGridDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AoGridDevice")
            .field("ngrids", &self.ngrids)
            .finish_non_exhaustive()
    }
}

impl AoGridDevice {
    /// `coords` is `3 · ngrids` reals, F-order.
    pub fn new(client: &AlgebraClient, coords: &[f64], ngrids: usize) -> Self {
        let handle = dispatch_backend!(
            client,
            c,
            Rt,
            pyscf_algebra::launch::upload::<Rt, f64>(c, coords)
        );
        Self { handle, ngrids }
    }
    pub fn ngrids(&self) -> usize {
        self.ngrids
    }
}

/// One image of a fused batch.
#[derive(Debug, Clone)]
pub struct FusedImage {
    /// The lattice vector `L`; the lane evaluates at `r_g − L`.
    pub l: [f64; 3],
    /// Per screening block: kept (`!= 0`) or not. Empty = dense (every point).
    pub keep_blocks: Vec<u32>,
}

/// The general-kernel lane body writing its `nctr · nsph` values to
/// `vals[vbase + c_idx·nsph + m]` instead of the output buffer. Same
/// operands, same order as `eval_gto_general_lane`.
#[allow(clippy::too_many_arguments)]
#[cube]
fn eval_gto_general_values(
    gx: f64,
    gy: f64,
    gz: f64,
    shell: usize,
    vbase: usize,
    vals: &mut Array<f64>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    c2s_flat: &Array<f64>,
    cpow_lx: &Array<i32>,
    cpow_ly: &Array<i32>,
    cpow_lz: &Array<i32>,
    ncart_by_l: &Array<i32>,
    nsph_by_l: &Array<i32>,
    fac1_by_l: &Array<f64>,
    c2s_off_by_l: &Array<i32>,
    cpow_off_by_l: &Array<i32>,
    rcut2: &Array<f64>,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let bas_row = shell * bas_slots;
    let l = bas[bas_row + ang_of] as u32;
    let lu = l as usize;
    let atom_id = bas[bas_row + atom_of] as usize;
    let nprim = bas[bas_row + nprim_of] as usize;
    let nctr = bas[bas_row + nctr_of] as usize;
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

    let ncart_l = ncart_by_l[lu] as usize;
    let nsph_l = nsph_by_l[lu] as usize;
    let fac1 = fac1_by_l[lu];
    let c2s_off = c2s_off_by_l[lu] as usize;
    let cpow_off = cpow_off_by_l[lu] as usize;

    if r2 <= rcut2[shell] {
        for c_idx in 0..nctr {
            let mut acc = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                acc += coef * ao_exp(-alpha * r2, exp_mode);
            }
            let radial = acc * fac1;
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
                vals[vbase + c_idx * nsph_l + m] = v;
            }
        }
    } else {
        for c_idx in 0..nctr {
            for m in 0..nsph_l {
                vals[vbase + c_idx * nsph_l + m] = 0.0_f64;
            }
        }
    }
}

/// The deriv1 lane body writing its `4 · nctr · nsph` values to
/// `vals[vbase + c·qn + c_idx·nsph + m]`, `qn = nctr·nsph`. Same operands,
/// same order as `eval_gto_deriv1_lane`.
#[allow(clippy::too_many_arguments)]
#[cube]
fn eval_gto_deriv1_values(
    gx: f64,
    gy: f64,
    gz: f64,
    shell: usize,
    vbase: usize,
    vals: &mut Array<f64>,
    env: &Array<f64>,
    bas: &Array<i32>,
    atm: &Array<i32>,
    c2s_flat: &Array<f64>,
    cpow_lx: &Array<i32>,
    cpow_ly: &Array<i32>,
    cpow_lz: &Array<i32>,
    ncart_by_l: &Array<i32>,
    nsph_by_l: &Array<i32>,
    fac1_by_l: &Array<f64>,
    c2s_off_by_l: &Array<i32>,
    cpow_off_by_l: &Array<i32>,
    rcut2: &Array<f64>,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    #[comptime] exp_mode: u32,
) {
    let bas_row = shell * bas_slots;
    let l = bas[bas_row + ang_of] as u32;
    let lu = l as usize;
    let atom_id = bas[bas_row + atom_of] as usize;
    let nprim = bas[bas_row + nprim_of] as usize;
    let nctr = bas[bas_row + nctr_of] as usize;
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

    let ncart_l = ncart_by_l[lu] as usize;
    let nsph_l = nsph_by_l[lu] as usize;
    let fac1 = fac1_by_l[lu];
    let c2s_off = c2s_off_by_l[lu] as usize;
    let cpow_off = cpow_off_by_l[lu] as usize;
    let qn = nctr * nsph_l;

    if r2 <= rcut2[shell] {
        for c_idx in 0..nctr {
            let mut acc = 0.0_f64;
            let mut acc2a = 0.0_f64;
            for p_idx in 0..nprim {
                let alpha = env[pe + p_idx];
                let coef = env[pc + c_idx * nprim + p_idx];
                let g0 = coef * ao_exp(-alpha * r2, exp_mode);
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
                let q = c_idx * nsph_l + m;
                vals[vbase + q] = v;
                vals[vbase + qn + q] = vx;
                vals[vbase + 2 * qn + q] = vy;
                vals[vbase + 3 * qn + q] = vz;
            }
        }
    } else {
        for c_idx in 0..nctr {
            for m in 0..nsph_l {
                let q = c_idx * nsph_l + m;
                vals[vbase + q] = 0.0_f64;
                vals[vbase + qn + q] = 0.0_f64;
                vals[vbase + 2 * qn + q] = 0.0_f64;
                vals[vbase + 3 * qn + q] = 0.0_f64;
            }
        }
    }
}

/// K-10: one lane per `(g, shell)`; evaluates `nimg` images and folds them
/// into the `nkpts` planes. `comp` is 1 or 4 (`deriv1`); `keep[m·nblocks +
/// g / blk]` says whether image `m` covers the point.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn eval_ao_k_fused_kernel<N: Size>(
    coords: &Array<f64>,
    lvec: &Array<f64>,
    keep: &Array<u32>,
    pr: &Array<Vector<f64, N>>,
    pi: &Array<Vector<f64, N>>,
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
    rcut2: &Array<f64>,
    out_re: &mut Array<Vector<f64, N>>,
    out_im: &mut Array<Vector<f64, N>>,
    ngrids: usize,
    nbas: usize,
    nao: usize,
    nkv: usize,
    nimg: usize,
    nblocks: usize,
    blk: usize,
    atm_slots: usize,
    bas_slots: usize,
    atom_of: usize,
    ang_of: usize,
    nprim_of: usize,
    nctr_of: usize,
    ptr_exp: usize,
    ptr_coeff: usize,
    ptr_coord: usize,
    lane0: usize,
    #[comptime] deriv1: bool,
    #[comptime] exp_mode: u32,
) {
    // `lane0`: chunked on the CPU runtime — `vals` and `present` are stack per
    // iteration there (`launch_1d_chunked`).
    let tid = ABSOLUTE_POS + lane0;
    if tid < ngrids * nbas {
        let g = tid % ngrids;
        let shell = tid / ngrids;
        let x = coords[g];
        let y = coords[g + ngrids];
        let z = coords[g + 2 * ngrids];
        let block = g / blk;

        let bas_row = shell * bas_slots;
        let lu = bas[bas_row + ang_of] as usize;
        let nctr = bas[bas_row + nctr_of] as usize;
        let nsph_l = nsph_by_l[lu] as usize;
        let ao_off = ao_loc[shell] as usize;
        let qn = nctr * nsph_l;
        let mut comp = 1usize;
        if comptime!(deriv1) {
            comp = 4usize;
        }
        let qtot = comp * qn;

        let mut vals = Array::<f64>::new(FUSED_VALS_CAP);
        for m in 0..nimg {
            let hit = keep[m * nblocks + block];
            if hit == 0u32 {
                // Not kept by this image: contributes nothing. The value slots
                // are zeroed so the unconditional multiply-add below adds an
                // exact `±0.0`, which leaves an accumulator that is never
                // `-0.0` (it starts at `+0.0` and `+0.0 + -0.0 == +0.0`)
                // bit-for-bit unchanged — the same as the per-image path,
                // which skipped the point.
                for q in 0..qtot {
                    vals[m * qtot + q] = 0.0_f64;
                }
            } else {
                // The host's `coords[g][axis] - l[axis]`, in-kernel.
                let gx = x - lvec[m * 3];
                let gy = y - lvec[m * 3 + 1];
                let gz = z - lvec[m * 3 + 2];
                if comptime!(deriv1) {
                    eval_gto_deriv1_values(
                        gx,
                        gy,
                        gz,
                        shell,
                        m * qtot,
                        &mut vals,
                        env,
                        bas,
                        atm,
                        c2s_flat,
                        cpow_lx,
                        cpow_ly,
                        cpow_lz,
                        ncart_by_l,
                        nsph_by_l,
                        fac1_by_l,
                        c2s_off_by_l,
                        cpow_off_by_l,
                        rcut2,
                        atm_slots,
                        bas_slots,
                        atom_of,
                        ang_of,
                        nprim_of,
                        nctr_of,
                        ptr_exp,
                        ptr_coeff,
                        ptr_coord,
                        exp_mode,
                    );
                } else {
                    eval_gto_general_values(
                        gx,
                        gy,
                        gz,
                        shell,
                        m * qtot,
                        &mut vals,
                        env,
                        bas,
                        atm,
                        c2s_flat,
                        cpow_lx,
                        cpow_ly,
                        cpow_lz,
                        ncart_by_l,
                        nsph_by_l,
                        fac1_by_l,
                        c2s_off_by_l,
                        cpow_off_by_l,
                        rcut2,
                        atm_slots,
                        bas_slots,
                        atom_of,
                        ang_of,
                        nprim_of,
                        nctr_of,
                        ptr_exp,
                        ptr_coeff,
                        ptr_coord,
                        exp_mode,
                    );
                }
            }
        }
        // The accumulate: per output of this lane, per k-VECTOR (N adjacent
        // k-points of the point-major planes, `out[p·nkpts + k]`), the images
        // in order. K-10v: one `Vector<f64, N>` multiply-add per image covers
        // N k-points; the phases `pr[m·nkpts + k..+N]` are contiguous, the
        // value broadcasts. Per `(k, p)` the sequence of additions is the
        // per-image path's, so the planes stay bit-identical.
        for c in 0..comp {
            for c_idx in 0..nctr {
                for msph in 0..nsph_l {
                    let q = c * qn + c_idx * nsph_l + msph;
                    let p = c * ngrids * nao + g + (ao_off + c_idx * nsph_l + msph) * ngrids;
                    for kv in 0..nkv {
                        let idx = p * nkv + kv;
                        let mut re = out_re[idx];
                        let mut im = out_im[idx];
                        for m in 0..nimg {
                            let v = Vector::<f64, N>::new(vals[m * qtot + q]);
                            re += pr[m * nkv + kv] * v;
                            im += pi[m * nkv + kv] * v;
                        }
                        out_re[idx] = re;
                        out_im[idx] = im;
                    }
                }
            }
        }
    }
}

/// The most images one fused batch may hold.
pub const AO_FUSED_BATCH_MAX: usize = 32;

/// `Q_max = comp · max_shell(nctr · nsph)` — the per-image value count of the
/// widest shell, which sizes the fused batch (`B · Q_max <= FUSED_VALS_CAP`).
pub fn fused_values_per_image(bas: &[i32], deriv1: bool) -> usize {
    let comp = if deriv1 { 4 } else { 1 };
    bas.chunks_exact(BAS_SLOTS)
        .map(|row| {
            let l = row[ANG_OF].max(0) as u32;
            let nctr = row[NCTR_OF].max(0) as usize;
            nctr * nsph(l)
        })
        .max()
        .unwrap_or(0)
        * comp
}

/// K-10: fold `images` into `acc`'s planes in ONE launch, evaluating the AO
/// values in-kernel. `ctx` carries the basis (its `rcut2` is the per-point
/// screen, `+inf` when off); `grid` the unshifted coordinates;
/// `pr`/`pi` are `images.len() · nkpts`, image-major; `blk` is the
/// screening block size the `keep_blocks` were built with.
///
/// Refused (as an error, never silently) when the basis has no device kernel,
/// when it is all-s at `deriv 0` (that path's s-kernel arithmetic is not the
/// general kernel's), or when `images.len() · Q_max` exceeds
/// [`FUSED_VALS_CAP`] — the caller sizes its batches with
/// [`fused_values_per_image`].
///
/// # Errors
/// [`PyscfRsError::Core`] on any of the above or a shape disagreement.
#[allow(clippy::too_many_arguments)]
pub fn eval_ao_k_fused_batch(
    client: &AlgebraClient,
    ctx: &EvalGtoDeviceContext,
    grid: &AoGridDevice,
    deriv1: bool,
    images: &[FusedImage],
    pr: &[f64],
    pi: &[f64],
    acc: &mut crate::pbc::AoKAccumulator,
    nkpts: usize,
    blk: usize,
) -> Result<(), PyscfRsError> {
    let err = |msg: String| PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg));
    let nimg = images.len();
    if nimg == 0 {
        return Ok(());
    }
    if ctx.all_s && !deriv1 {
        return Err(err(
            "K-10: all-s basis at deriv 0 keeps the s-kernel path".into()
        ));
    }
    let ang = ctx
        .angular
        .as_ref()
        .ok_or_else(|| err("K-10: no angular tables in the context".into()))?;
    let ngrids = grid.ngrids;
    let comp = if deriv1 { 4 } else { 1 };
    let n = comp * ngrids * ctx.nao;
    let (acc_nkpts, acc_n) = acc.shape();
    if !acc.is_point_major() {
        return Err(err(
            "K-10: the fused kernel needs a point-major accumulator (`AoKAccumulator::zeros_point_major`)"
                .into(),
        ));
    }
    if acc_nkpts != nkpts || acc_n != n || pr.len() != nimg * nkpts || pi.len() != nimg * nkpts {
        return Err(err(format!(
            "K-10: accumulator ({acc_nkpts}, {acc_n}) vs ({nkpts}, {n}); pr {} pi {} for {nimg} images",
            pr.len(),
            pi.len()
        )));
    }
    if nimg > AO_FUSED_BATCH_MAX {
        return Err(err(format!(
            "K-10: {nimg} images exceed {AO_FUSED_BATCH_MAX}"
        )));
    }
    let nblocks = ngrids.div_ceil(blk.max(1)).max(1);
    let mut keep: Vec<u32> = Vec::with_capacity(nimg * nblocks);
    let mut lvec: Vec<f64> = Vec::with_capacity(3 * nimg);
    for (m, img) in images.iter().enumerate() {
        lvec.extend_from_slice(&img.l);
        if img.keep_blocks.is_empty() {
            keep.extend(std::iter::repeat_n(1u32, nblocks));
        } else if img.keep_blocks.len() == nblocks {
            keep.extend_from_slice(&img.keep_blocks);
        } else {
            return Err(err(format!(
                "K-10: image {m} has {} keep flags for {nblocks} blocks",
                img.keep_blocks.len()
            )));
        }
    }
    let lanes = ngrids * ctx.nbas;
    if lanes == 0 || nkpts == 0 {
        return Ok(());
    }
    let (re_h, im_h) = acc.planes();
    let [env_len, bas_len, atm_len, ao_loc_len, rcut2_len] = ctx.lens;
    let al = ang.lens;
    // Per lane: the images' evaluations plus `2 · Q · nkpts · nimg` multiply-adds.
    let per_lane =
        nimg * (if deriv1 {
            EVAL_GTO_DERIV1_WORK_PER_LANE
        } else {
            EVAL_GTO_GENERAL_WORK_PER_LANE
        }) + 2 * comp * 9 * nkpts * nimg;
    dispatch_backend!(client, c, Rt, {
        let lvec_h = pyscf_algebra::launch::upload::<Rt, f64>(c, &lvec);
        let keep_h = c.create_from_slice(bytemuck::cast_slice(&keep));
        let pr_h = pyscf_algebra::launch::upload::<Rt, f64>(c, pr);
        let pi_h = pyscf_algebra::launch::upload::<Rt, f64>(c, pi);
        let local_bytes = FUSED_VALS_CAP * core::mem::size_of::<f64>();
        // K-10v: the widest vector the device likes for f64 that divides nkpts.
        let line = pyscf_algebra::launch::line_size_for::<Rt, f64>(c, nkpts);
        let nkv = nkpts / line;
        // SAFETY: every handle length is its slice's length; the kernel
        // guards `tid < ngrids·nbas`; the two planes are the only `&mut`.
        for chunk in pyscf_algebra::launch::launch_1d_chunked(c, lanes, per_lane, local_bytes) {
            unsafe {
                eval_ao_k_fused_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::Static(chunk.count_x, 1, 1),
                    chunk.dim,
                    line,
                    ArrayArg::from_raw_parts(grid.handle.clone(), 3 * ngrids),
                    ArrayArg::from_raw_parts(lvec_h.clone(), 3 * nimg),
                    ArrayArg::from_raw_parts(keep_h.clone(), nimg * nblocks),
                    ArrayArg::from_raw_parts(pr_h.clone(), nimg * nkpts),
                    ArrayArg::from_raw_parts(pi_h.clone(), nimg * nkpts),
                    ArrayArg::from_raw_parts(ctx.env.clone(), env_len),
                    ArrayArg::from_raw_parts(ctx.bas.clone(), bas_len),
                    ArrayArg::from_raw_parts(ctx.atm.clone(), atm_len),
                    ArrayArg::from_raw_parts(ctx.ao_loc.clone(), ao_loc_len),
                    ArrayArg::from_raw_parts(ang.c2s_flat.clone(), al[0]),
                    ArrayArg::from_raw_parts(ang.cpow_lx.clone(), al[1]),
                    ArrayArg::from_raw_parts(ang.cpow_ly.clone(), al[2]),
                    ArrayArg::from_raw_parts(ang.cpow_lz.clone(), al[3]),
                    ArrayArg::from_raw_parts(ang.ncart_by_l.clone(), al[4]),
                    ArrayArg::from_raw_parts(ang.nsph_by_l.clone(), al[5]),
                    ArrayArg::from_raw_parts(ang.fac1_by_l.clone(), al[6]),
                    ArrayArg::from_raw_parts(ang.c2s_off_by_l.clone(), al[7]),
                    ArrayArg::from_raw_parts(ang.cpow_off_by_l.clone(), al[8]),
                    ArrayArg::from_raw_parts(ctx.rcut2.clone(), rcut2_len),
                    ArrayArg::from_raw_parts(re_h.clone(), nkpts * n),
                    ArrayArg::from_raw_parts(im_h.clone(), nkpts * n),
                    ngrids,
                    ctx.nbas,
                    ctx.nao,
                    nkv,
                    nimg,
                    nblocks,
                    blk.max(1),
                    ATM_SLOTS,
                    BAS_SLOTS,
                    ATOM_OF,
                    ANG_OF,
                    NPRIM_OF,
                    NCTR_OF,
                    PTR_EXP,
                    PTR_COEFF,
                    PTR_COORD,
                    chunk.lane0,
                    deriv1,
                    ctx.exp_mode,
                );
            }
        }
    });
    Ok(())
}

/// Whether [`eval_gto_sph_into_target`] / [`eval_gto_sph_deriv1_into_target`]
/// can serve this basis on the device: every shell `l <= 4`, at least one
/// shell. Bases outside that take the host fallback, which allocates its own
/// block and so cannot write into a caller-owned slot.
pub fn eval_gto_device_capable(bas: &[i32]) -> bool {
    !bas.is_empty()
        && bas
            .chunks_exact(BAS_SLOTS)
            .all(|row| (0..=4).contains(&row[ANG_OF]))
}

/// [`eval_gto_sph_into_screened`], writing into `target` — a caller-owned
/// device block of exactly `ngrids * nao` reals (K-09's image-batch slot)
/// instead of a fresh allocation. Requires [`eval_gto_device_capable`].
///
/// # Errors
/// [`PyscfRsError::Core`] when `target` has the wrong length or the basis has
/// no device kernel.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_into_target(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    rcut2: Option<&[f64]>,
    target: &AoBlockDevice,
) -> Result<(), PyscfRsError> {
    let out_len = ngrids * nao;
    if target.len() != out_len || !eval_gto_device_capable(bas) {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "eval_gto_sph_into_target: target holds {} reals, need {out_len}; \
                 device-capable basis: {}",
                target.len(),
                eval_gto_device_capable(bas)
            ),
        )));
    }
    if out_len == 0 {
        return Ok(());
    }
    let all_s = bas.chunks_exact(BAS_SLOTS).all(|row| row[ANG_OF] == 0);
    if all_s {
        dispatch_backend!(client, c, Rt, {
            launch_eval_gto_s_into::<Rt>(
                c,
                coords,
                ngrids,
                atm,
                bas,
                env,
                ao_loc,
                nao,
                rcut2,
                target.handle(),
            );
        });
        return Ok(());
    }
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;
    dispatch_backend!(client, c, Rt, {
        launch_eval_gto_general_into::<Rt>(
            c,
            coords,
            ngrids,
            atm,
            bas,
            env,
            ao_loc,
            nao,
            maxl,
            rcut2,
            target.handle(),
        )
    })
}

/// [`eval_gto_sph_deriv1_into_screened`], writing into `target` — exactly
/// `4 * ngrids * nao` reals. See [`eval_gto_sph_into_target`].
///
/// # Errors
/// As [`eval_gto_sph_into_target`].
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_deriv1_into_target(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    rcut2: Option<&[f64]>,
    target: &AoBlockDevice,
) -> Result<(), PyscfRsError> {
    let out_len = 4 * ngrids * nao;
    if target.len() != out_len || !eval_gto_device_capable(bas) {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "eval_gto_sph_deriv1_into_target: target holds {} reals, need {out_len}; \
                 device-capable basis: {}",
                target.len(),
                eval_gto_device_capable(bas)
            ),
        )));
    }
    if out_len == 0 {
        return Ok(());
    }
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;
    dispatch_backend!(client, c, Rt, {
        launch_eval_gto_deriv1_into::<Rt>(
            c,
            coords,
            ngrids,
            atm,
            bas,
            env,
            ao_loc,
            nao,
            maxl,
            rcut2,
            target.handle(),
        )
    })
}

/// [`eval_gto_sph_deriv1_into`] with A-04's per-point reach test — see
/// [`eval_gto_sph_into_screened`]. The caller sizes `rcut2` for `deriv = 1`.
#[allow(clippy::too_many_arguments)]
pub fn eval_gto_sph_deriv1_into_screened(
    client: &AlgebraClient,
    coords: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
    rcut2: Option<&[f64]>,
) -> Result<AoBlockDevice, PyscfRsError> {
    let maxl = bas
        .chunks_exact(BAS_SLOTS)
        .map(|row| row[ANG_OF])
        .max()
        .unwrap_or(0) as u32;
    let out_len = 4 * ngrids * nao;
    if !bas.is_empty() && maxl <= 4 && out_len > 0 {
        return dispatch_backend!(client, c, Rt, {
            let out = c.empty(out_len * core::mem::size_of::<f64>());
            launch_eval_gto_deriv1_into::<Rt>(
                c, coords, ngrids, atm, bas, env, ao_loc, nao, maxl, rcut2, &out,
            )?;
            Ok(AoBlockDevice {
                handle: out,
                len: out_len,
                shape: vec![4, ngrids, nao],
            })
        });
    }
    let host = eval_gto_sph_deriv1_cpu(coords, ngrids, atm, bas, env, ao_loc, nao, true)?;
    Ok(upload_ao_block(client, &host.values, host.shape))
}

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
    let shape = vec![4, ngrids, nao];
    let values = eval_gto_sph_deriv1_into(client, coords, ngrids, atm, bas, env, ao_loc, nao)?
        .into_values(client);
    Ok(EvalGtoBuffers { values, shape })
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
