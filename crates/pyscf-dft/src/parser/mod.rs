//! XC-string parsers (DFT-02, D-01).
//!
//! Two resolvers with the *same* return shape but different functional-id
//! namespaces and routing target:
//!
//! - [`libxc`] — the **default** resolver (D-01). Ports
//!   `pyscf/dft/libxc.py:parse_xc` (lines 491-718) + `XC_CODES` (154) +
//!   `XC_ALIAS` (217). Emits libxc functional IDs; routes to `libxc_rs`
//!   (gated). `b3lyp` resolves through this path to match upstream numbers.
//! - [`xcfun`] — the **alternate** resolver. Ports
//!   `pyscf/dft/xcfun.py:parse_xc` (lines 416-569) + its `XC_CODES` (94) +
//!   `XC_ALIAS` (240). Emits xcfun functional IDs (0..77); routes to
//!   `xcfun_rs`.
//!
//! License: Apache-2.0 (matching upstream PySCF, which these are direct ports
//! of). Tables are inline `const` slices — NO codegen, NO build.rs (mirrors
//! the `pyscf-df` `DEFAULT_AUXBASIS` convention).
//!
//! Security (threat T-04-05a/b): the XC string is untrusted user input. Both
//! parsers are pure string→spec mappings: they never `eval` arbitrary code,
//! and the compound-expansion path (`0.5*b3lyp` unit scaling) carries an
//! explicit recursion-depth bound so adversarial / cyclic alias chains return
//! a clean `Err` instead of overflowing the stack.

pub mod libxc;
pub mod xcfun;

/// Range-separated-hybrid / exact-exchange coefficients: `(hyb, alpha, omega)`.
///
/// Mirrors upstream `parse_xc`'s `hyb = [hybrid, alpha, omega]` triple
/// (`libxc.py:539`):
/// - `hyb` (`hyb[0]`): short-range / full HF-exchange mixing coefficient.
/// - `alpha` (`hyb[1]`): long-range HF-exchange (LR_HF) coefficient.
/// - `omega` (`hyb[2]`): range-separation parameter (0 ⇒ non-RSH).
pub type HybCoeffs = (f64, f64, f64);

/// One parsed functional component: `(functional_id, factor)`.
///
/// For the libxc parser the id is a libxc functional number (e.g. B88 = 106);
/// for the xcfun parser it is an xcfun functional id (0..77). Negative/positive
/// factors and de-duplication (`remove_dup`) match upstream exactly.
pub type Component = (u32, f64);

/// The decoded XC description: `((hyb, alpha, omega), [(id, fac), ...])`.
///
/// Exactly the upstream `parse_xc` return shape
/// (`(hybrid, alpha, omega), ((libxc-Id, fac), ...)`, `libxc.py:534-536`).
#[derive(Debug, Clone, PartialEq)]
pub struct XcSpec {
    /// `(hyb, alpha, omega)` — HF-exchange / range-separation coefficients.
    pub hyb: HybCoeffs,
    /// `[(functional_id, factor), ...]`, duplicate-merged in encounter order.
    pub components: Vec<Component>,
}

impl XcSpec {
    /// Convenience accessor for the `(hyb, alpha, omega)` triple.
    pub fn hyb(&self) -> HybCoeffs {
        self.hyb
    }

    /// Convenience accessor for the `[(id, fac), ...]` component list.
    pub fn components(&self) -> &[Component] {
        &self.components
    }
}

/// Maximum depth for compound-functional expansion (`0.5*b3lyp` → its parts,
/// which may themselves be compound names). Bounds the recursion against
/// adversarial / cyclic alias chains (threat T-04-05b). PySCF's real alias
/// graph nests at most ~2 levels (e.g. `MPW3PW` → `MPW3PW5` → primitives), so
/// 32 is comfortably above any legitimate input while still O(1)-ish.
pub(crate) const MAX_EXPANSION_DEPTH: usize = 32;

/// `remove_dup` port (`pyscf/dft/xc/utils.py:19`): merge duplicate functional
/// IDs by summing their factors, preserving first-encounter order. Shared by
/// both parsers.
pub(crate) fn remove_dup(fn_facs: &[Component]) -> Vec<Component> {
    let mut ids: Vec<u32> = Vec::with_capacity(fn_facs.len());
    let mut facs: Vec<f64> = Vec::with_capacity(fn_facs.len());
    for &(key, val) in fn_facs {
        if let Some(pos) = ids.iter().position(|&k| k == key) {
            facs[pos] += val;
        } else {
            ids.push(key);
            facs.push(val);
        }
    }
    ids.into_iter().zip(facs).collect()
}
