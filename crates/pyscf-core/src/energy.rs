//! Energy newtype — Hartree units. The newtype prevents callers from
//! accidentally passing a raw `f64` (e.g., a count or a coordinate)
//! where an energy is expected.

/// Energy value in atomic units (Hartree). Use `.0` to extract the raw
/// `f64`; use `Energy(x)` to wrap.
///
/// Per FOUND-02 this is a proper newtype, not a type alias — the
/// distinct type is the entire point.
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct Energy(pub f64);

impl Energy {
    /// Hartree → kcal/mol conversion factor (CODATA 2018: 627.5094740631).
    pub const HARTREE_TO_KCAL_MOL: f64 = 627.509_474_063_1;

    /// Hartree → eV conversion factor (CODATA 2018: 27.211386245988).
    pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

    /// Construct from raw f64 in Hartree.
    #[inline]
    pub const fn hartree(v: f64) -> Self {
        Self(v)
    }

    /// Extract the raw value in Hartree.
    #[inline]
    pub const fn to_hartree(self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for Energy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.10} Eh", self.0)
    }
}
