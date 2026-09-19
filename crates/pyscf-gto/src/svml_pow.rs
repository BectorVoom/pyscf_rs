//! Scalar port of Intel SVML's `__svml_pow8` (`svml_z0_pow_d_la.s`) — the
//! AVX-512 `np.power(x, y)` kernel numpy 1.26.4 dispatches on this host.
//!
//! # Source
//!
//! numpy v1.26.4 ships the SVML sources under
//! `numpy/core/src/umath/svml/linux/avx512/svml_z0_pow_d_la.s` (Intel
//! Copyright 2023, BSD-3-Clause). The dispatch proof: `np.power(x_array, e)`
//! differs from glibc `pow` in ~28% of cases on AVX-512, and zeroes out under
//! `NPY_DISABLE_CPU_FEATURES=AVX512F…`; the wheel exports both `__svml_pow8`
//! and `__svml_pow8_ha`, and `loops_umath_fp.dispatch.c.src` calls the plain
//! `__svml_pow8` (the `_la` variant). All probes are in
//! `target/probes-task6/`.
//!
//! # What is ported
//!
//! The FAST PATH (`__svml_pow8` lines 86-295) for **positive finite `x`** and
//! **finite `y`**. Those inputs never enter the `.LBL_1_3` rare branch (its
//! trigger, `edx != 0`, needs `x <= 0`, `x`/`y` Inf or NaN, or
//! `|y·log2(x)| >= 1021.5` — asserted with `debug_assert!`), and the `_env`
//! normalisation calls it only with `x = 2·α > 0` and `y = l + 1.5`.
//!
//! The algorithm (`log2|x|` via a 5-bit-rounded reciprocal + three correction
//! tables, then `2^(y·log2|x|)` via a 16-entry `2^(j/16)` table and a degree-5
//! polynomial) is transcribed instruction for instruction. Rounding modes:
//!
//! * every `vfmadd*` → [`f64::mul_add`];
//! * `vgetmantpd`/`vgetexppd` → explicit bit manipulation ([`get_mantissa`],
//!   [`get_exponent`]);
//! * `vrndscalepd $88` (round to 5 fraction bits, RN ties-even) →
//!   [`round_5bits`];
//! * `vreducepd $65` (round DOWN to 4 fraction bits) → [`floor_4bits`];
//! * the `1.5·2^48 + 1023` magic-constant index trick is reproduced exactly;
//! * `vrcp14pd` — the ONE hardware-only primitive — calls the AVX-512
//!   instruction through `core::arch` (not FFI; it is the reference host's own
//!   arithmetic, like `f64::mul_add`'s FMA). A probe over 3.2M mantissas shows
//!   `round5(vrcp14pd(x)) != round5(1/x)` in 898 cases, so `1/x` is NOT a
//!   substitute.
//!
//! The `{rz}` (round-toward-zero) roundings on the log2 low-part corrections
//! are reproduced with plain `mul_add`: the values they perturb are ~2^-50
//! relative to the result, far below the final ulp, so RN and RZ give the same
//! final bits (verified by the oracle test).
//!
//! # Tables
//!
//! Copied as decimal literals from `svml_z0_pow_d_la.s`'s
//! `__svml_dpow_data_internal_avx512` (byte offsets given per table); the
//! `vpermt2pd` 16-entry gathers (`idx & 0xF`, bit 3 selects the second half)
//! are direct array reads.

#![cfg(target_arch = "x86_64")]

// --- the `__svml_dpow_data_internal_avx512` tables (16 entries each) ---

/// `log2(1/Rcp)` corrections for `DblRcp >= 0.75` (`data+0/+64`).
const TAB0: [f64; 16] = [
    0.0,
    -0.04439411935800308,
    -0.0874628412502716,
    -0.12928301694500988,
    -0.16992500144260703,
    -0.20945336562908778,
    -0.24792751344375574,
    -0.2854022188621457,
    -0.32192809488697094,
    -0.3575520046179008,
    -0.39231742277843296,
    -0.4262647547020606,
    -0.45943161863760906,
    -0.49185309632957797,
    -0.5235619560571649,
    -0.5545888516780906,
];

/// `log2(1/Rcp)` corrections for `DblRcp < 0.75` (`data+128/+192`).
const TAB1: [f64; 16] = [
    0.4150374992786965,
    0.3852901558848316,
    0.3561438102251486,
    0.3275746580284249,
    0.29956028185915784,
    0.2720795454370091,
    0.24511249783608946,
    0.21864028647542,
    0.19264507794196106,
    0.16710998583494074,
    0.1420190048720542,
    0.1173569506381682,
    0.09310940439172555,
    0.06926266243681312,
    0.04580368961342174,
    0.02272007650026353,
];

/// Low parts for `DblRcp >= 0.75` (`data+256/+320`).
const TAB2: [f64; 16] = [
    0.0,
    -4.5035636491329477e-13,
    -6.781363655020842e-14,
    4.3421195976986995e-14,
    2.9466363971563933e-13,
    1.3800246976244608e-13,
    1.7024883460791358e-13,
    -1.02668366941445e-13,
    -3.914087556296025e-13,
    -1.82872716976671e-13,
    -3.273329303085191e-13,
    -3.733907276649932e-14,
    3.1180651700181383e-13,
    -9.67451159139632e-14,
    1.520621696507556e-13,
    4.532053086411405e-13,
];

/// Low parts for `DblRcp < 0.75` (`data+384/+448`).
const TAB3: [f64; 16] = [
    1.4733181985781966e-13,
    -3.983479855974919e-14,
    1.266771905137232e-13,
    7.951818330761124e-14,
    -2.500001867992647e-13,
    -2.0825307466611057e-13,
    4.4199545957345896e-13,
    -7.960223862778872e-14,
    4.348299516065895e-13,
    3.1758065446573327e-13,
    3.736684162192024e-13,
    -9.459192232399996e-15,
    -2.4407693577178287e-13,
    3.0060745418937855e-13,
    -2.9694741952048217e-13,
    -1.800011104506994e-13,
];

/// `2^(j/16)` for `j = 0..15` (`data+512/+576`).
const EXP2: [f64; 16] = [
    1.0,
    1.0442737824274138,
    1.0905077326652577,
    1.1387886347566916,
    1.189207115002721,
    1.241857812073484,
    1.2968395546510096,
    1.3542555469368927,
    1.4142135623730951,
    1.4768261459394993,
    1.5422108254079407,
    1.6104903319492543,
    1.681792830507429,
    1.7562521603732995,
    1.8340080864093424,
    1.9152065613971474,
];

/// The log2 conversion constant `1/ln(2)` (`data+768`).
const C1H: f64 = 1.4426950408889634;
/// `-1/(2·ln2)` (`data+832`).
const C2H: f64 = -0.7213475204444817;
/// A low-order `1/ln2` correction (`data+1408`).
const C1L: f64 = 2.0355736058257225e-17;
/// `1.5·2^48 + 1023` — the magic constant that encodes `N + j/16` in the
/// mantissa's low bits (`data+1472`).
const MAGIC: f64 = 422212465067007.0;

// exp2 polynomial coefficients (`data+1600..+1920`).
const P56_CO: f64 = 0.00015741120548571732;
const P56: f64 = 0.0013330645434141127;
const P34_CO: f64 = 0.009618141545265708;
const P34: f64 = 0.0555041083923959;
const P12_CO: f64 = 0.24022650696194106;
const P12: f64 = 0.6931471805599342;

// log2 polynomial coefficients (`data+896..+1408`).
const P89_CO: f64 = 0.16038822625577315;
const P89: f64 = -0.18041677509176682;
const P67_CO: f64 = 0.20609927020561977;
const P67: f64 = -0.24044915873623204;
const P45_CO: f64 = 0.2885390081799307;
const P45: f64 = -0.3606737602232487;
const P23_CO: f64 = 0.4808983469629877;
const P23: f64 = 8.421822774076385e-18;

/// `vrcp14pd(x)` — the AVX-512 14-bit reciprocal, via
/// [`pyscf_algebra::arch::vrcp14`] (the one hardware primitive; its exact
/// result is not reproducible with `1/x`).

/// `vgetmantpd $10` (= `_MM_MANT_NORM_p5_1 | _MM_MANT_SIGN_nan`): the mantissa
/// in `[0.5, 1)`, via bit manipulation (normal positive input). The `[0.5, 1)`
/// interval (not `[1, 2)`) is what keeps `DblRcp = 1/mantissa` in `(1, 2]` and
/// the table indices aligned to the reciprocal's own mantissa bits.
fn get_mantissa(x: f64) -> f64 {
    f64::from_bits((x.to_bits() & 0x000F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000)
}

/// `vgetexppd`: `floor(log2(x))` as a double (normal positive input).
fn get_exponent(x: f64) -> f64 {
    ((x.to_bits() >> 52) as f64) - 1023.0
}

/// `vrndscalepd $88`: round to 5 fraction bits, RN (ties-to-even).
fn round_5bits(x: f64) -> f64 {
    (x * 32.0).round_ties_even() / 32.0
}

/// `vreducepd $65`'s rounding half: round DOWN to 4 fraction bits.
fn floor_4bits(x: f64) -> f64 {
    (x * 16.0).floor() / 16.0
}

/// `vaddpd {rd-sae}`: `a + b` rounded TOWARD -inf. `a` is always `MAGIC` here
/// (large, positive), so `s = RN(a+b)` is positive; the two-sum residual `e`
/// decides whether to step one ulp down.
fn add_round_down(a: f64, b: f64) -> f64 {
    let (s, e) = two_sum(a, b);
    if e >= 0.0 {
        s
    } else {
        f64::from_bits(s.to_bits() - 1)
    }
}

/// `vmulpd {rz-sae}`: the exact product `a*b` rounded TOWARD ZERO.
///
/// `hi = RN(a*b)` and `lo = fma(a, b, -hi)` give `a*b = hi + lo` exactly with
/// `|lo| <= ulp(hi)/2`. The exact product lies between `hi` and the next double
/// away from zero when `lo` shares `hi`'s sign; otherwise it is between `hi`
/// and the next double toward zero, so step `hi` one ulp toward zero.
fn mul_rz(a: f64, b: f64) -> f64 {
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    if hi == 0.0 || !hi.is_finite() {
        return hi;
    }
    if (hi > 0.0 && lo < 0.0) || (hi < 0.0 && lo > 0.0) {
        f64::from_bits(hi.to_bits() - 1)
    } else {
        hi
    }
}

/// Knuth two-sum: `a + b = (s, t)` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let t = (a - (s - bb)) + (b - bb);
    (s, t)
}

/// Round the exact `s + e` (`|e| <= ulp(s)/2`) toward zero.
fn rtz(s: f64, e: f64) -> f64 {
    if e == 0.0 {
        return s;
    }
    if s == 0.0 {
        return if e > 0.0 { 0.0 } else { -0.0 };
    }
    if (e > 0.0) == (s > 0.0) {
        s
    } else {
        f64::from_bits(s.to_bits() - 1)
    }
}

/// `vfmadd213pd {rz-sae}`: `a*b + c` rounded TOWARD ZERO, via exact
/// two-product/two-sum arithmetic. The three-term exact sum is reduced to a
/// normalised double-double `(s, e)` (a final `two_sum` absorbs the residual so
/// `|e| <= ulp(s)/2`), then rounded toward zero by [`rtz`].
fn fma_rz(a: f64, b: f64, c: f64) -> f64 {
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    let (s1, t1) = two_sum(hi, c);
    let (s2, t2) = two_sum(s1, lo);
    let (s, e) = two_sum(s2, t1 + t2);
    rtz(s, e)
}

/// `__svml_pow8`'s fast path for `x > 0` finite, `y` finite. Scalar per-lane
/// transcription of the vector kernel.
pub fn svml_pow8(x: f64, y: f64) -> f64 {
    assert!(x.is_finite() && x > 0.0, "svml_pow8 fast path needs positive finite x");
    assert!(y.is_finite(), "svml_pow8 fast path needs finite y");
    debug_assert!(
        is_x86_feature_detected!("avx512f"),
        "svml_pow8 needs AVX-512 (the numpy reference host has it)"
    );

    // log2|x|: getmant/getexp, then the 5-bit-rounded reciprocal.
    let mant = get_mantissa(x);
    let expo = get_exponent(x);
    let dblrcp_raw = pyscf_algebra::arch::vrcp14(mant);
    let dblrcp = round_5bits(dblrcp_raw);
    // R = DblRcp*X1 - 1 (vfmsub213: zmm13*zmm10 - zmm15).
    let r = dblrcp.mul_add(mant, -1.0);
    // Table index from the rounded reciprocal's mantissa bits.
    let db = dblrcp.to_bits();
    let k = ((db >> 47) & 0xF) as usize;
    // `vblendmpd` AT&T `[src1, src2, dst{k}]` is `dst = k ? src1 : src2`, so
    // `k2 = bit 51` (DblRcp >= 1.5, i.e. mantissa <= 2/3) picks TAB1/TAB3.
    let upper = ((db >> 51) & 1) == 1;
    // `vaddpd {rn} zmm15, zmm9, zmm9{k1}` — add 1 to Expon when DblRcp < 1.5
    // (k1 = `vcmppd $17`, `1.5 < DblRcp` inverted).
    let e = expo + if dblrcp < 1.5 { 1.0 } else { 0.0 };
    let th = if upper { TAB1[k] } else { TAB0[k] };
    let tl = if upper { TAB3[k] } else { TAB2[k] };
    let r2 = r * r;

    // log2 polynomial (P8_9 -> P6_9 -> P4_9 -> P2_9) and the low parts.
    let p = P89_CO.mul_add(r, P89);
    let p1 = P67_CO.mul_add(r, P67);
    let p = r2.mul_add(p, p1);
    let p2 = P45_CO.mul_add(r, P45);
    let p = r2.mul_add(p, p2);
    let p3 = P23_CO.mul_add(r, P23);
    let p = r2.mul_add(p, p3);
    let r2l = r.mul_add(r, -r2);
    let tlr = C2H.mul_add(r2l, tl);
    let tlr = r2.mul_add(p, tlr);
    let high = e + th;
    let high_r = r.mul_add(C1H, high);
    let high_r2c2 = r2.mul_add(C2H, high_r);
    let r_c1h_h = high_r - high;
    let r_c1h_l = C1H.mul_add(r, -r_c1h_h);
    let rc1 = C1L.mul_add(r, r_c1h_l);
    let high2 = tlr + high_r2c2;
    let r2c2_h = high_r2c2 - high_r;
    let r2c2_l = C2H.mul_add(r2, -r2c2_h);
    let tll = high2 - high_r2c2;
    let tl2 = tlr - tll;
    let rc1r2 = r2c2_l + rc1;
    let tll2 = rc1r2 + tl2;

    // y*High, split into N + j/16 (the magic-constant trick) and the fraction.
    // `y_high` is the RZ (truncated) product — the split rounding is observable
    // in the final ulp through the cancellation in `zl`.
    let y_high = mul_rz(y, high2);
    // `vaddpd {rd} MAGIC, y_high` — the index/scale encoding, rounded down.
    let z2 = add_round_down(MAGIC, y_high);
    // `vreducepd $65` on `y_high`: `round_down(y_high - floor_4bits(y_high))`
    // (the subtraction rounds DOWN, not to-nearest — observable one ulp off
    // for tiny `|y_high|`, the x≈1 cases).
    let f4 = floor_4bits(y_high);
    let z_frac = add_round_down(y_high, -f4);
    // `vfmsub213pd {rz}` and `vfmadd213pd {rz}` — both fused, both RZ.
    let y_high_low = fma_rz(y, high2, -y_high);
    let zl = fma_rz(y, tll2, y_high_low);
    let r_z = zl + z_frac;
    let z2b = z2.to_bits();
    let the = EXP2[(z2b & 0xF) as usize];
    let scale2 = f64::from_bits((z2b << 48) & 0x7FF0_0000_0000_0000);

    // exp2 polynomial.
    let r_z2 = r_z * r_z;
    let the_r = the * r_z;
    let p5 = P56_CO.mul_add(r_z, P56);
    let p3 = P34_CO.mul_add(r_z, P34);
    let p1 = P12_CO.mul_add(r_z, P12);
    let p = r_z2.mul_add(p5, p3);
    let p = r_z2.mul_add(p, p1);
    let p = the_r.mul_add(p, the);

    // Bound check from the assembly: the rare path also needs
    // `|y*log2(x)| >= 1021.5`, impossible for the `_env` inputs.
    let masked = (y_high.to_bits() & 0x7FF8_0000_0000_0000) as f64;
    debug_assert!(
        !(masked >= 1021.5),
        "svml_pow8: |y*log2(x)| too large for the fast path"
    );

    scale2 * p
}