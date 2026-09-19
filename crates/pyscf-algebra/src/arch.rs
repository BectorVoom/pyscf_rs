//! AVX-512 hardware primitives that the SVML `pow` port
//! (`pyscf_gto::svml_pow`) needs — `vrcp14pd`, the one instruction that has no
//! bit-manipulation equivalent (a probe over 3.2M mantissas shows
//! `round5(vrcp14pd(x)) != round5(1/x)` in 898 cases).
//!
//! Not FFI: `core::arch` wraps the CPU instruction itself, exactly like
//! `f64::mul_add`'s FMA. The numpy reference host (AVX-512) always has it;
//! anything else panics rather than silently approximating.
#![cfg(target_arch = "x86_64")]

/// `vrcp14pd(x)` for one double, via the AVX-512 vector instruction.
#[target_feature(enable = "avx512f")]
unsafe fn vrcp14_512(x: f64) -> f64 {
    use core::arch::x86_64::*;
    unsafe { _mm512_cvtsd_f64(_mm512_rcp14_pd(_mm512_set1_pd(x))) }
}

/// `vrcp14pd(x)` — the 14-bit approximate reciprocal, exactly as the hardware
/// computes it.
///
/// # Panics
/// On a CPU without AVX-512 (the SVML reference host always has it).
pub fn vrcp14(x: f64) -> f64 {
    assert!(
        is_x86_feature_detected!("avx512f"),
        "vrcp14 needs AVX-512 (the numpy reference host has it)"
    );
    unsafe { vrcp14_512(x) }
}