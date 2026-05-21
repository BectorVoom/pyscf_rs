//! Kernel-side precision seam (quick-260522-b06).
//!
//! `pyscf-kernels` is on the algebra-wall allowlist (ALG-06), so it MAY name
//! `cubecl`. This module establishes the precision seam for kernels WITHOUT
//! generalizing any kernel body yet.
//!
//! TRADE-OFF (explicit): kernels stay `f64` by default. Generalizing the
//! existing `eval_gto` cubecl kernel to `<T>` now would double cubecl
//! monomorphization for zero current call site (a compile-time blowup with no
//! payoff). f32 device kernels are an OPT-IN future surface; when they land,
//! they bound their element type on [`DeviceScalar`] (re-exported here) rather
//! than on the host-only `pyscf_core::Scalar`, keeping `cubecl::Float` inside
//! the wall. f32 is a speed/GPU path only and is NOT bit-exact with PySCF
//! parity — oracle/regression kernels stay on f64.

/// The device float bound for cubecl kernels. Re-exported from
/// `pyscf-algebra` so future generic kernels can write `<T: DeviceScalar>`
/// without each kernel crate redeclaring the `cubecl::Float` supertrait.
pub use pyscf_algebra::DeviceScalar;

/// Default kernel element type. Records that the current kernels (e.g.
/// `eval_gto_sph`) are f64; a future f32 kernel is generic over a
/// `T: DeviceScalar` instead of using this alias.
pub type KernelScalar = f64;
