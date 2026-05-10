//! CPU probe — trivially true. CPU is always available (FOUND-03
//! default-on).

use crate::backend::DType;

/// CPU probe — always returns true, ignoring DType (CPU supports both
/// f32 and f64 unconditionally).
pub fn cpu_available(_dtype: DType) -> bool {
    true
}
