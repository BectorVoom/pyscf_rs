//! Planar (split) complex host type — PBC-MASTER-PLAN §5.1 / D-PBC-02.
//!
//! RULE 8 of the PBC master plan: complex numbers NEVER cross the algebra wall
//! (ALG-06) as a `Complex<f64>` element type. Every complex quantity in the
//! periodic code path is carried as [`CTensor`] — two independent `Vec<f64>`
//! planes of equal length — so that every device operation reduces to the
//! EXISTING real cubecl primitives (`gemm_dense`, `axpy_dense`, …) and no new
//! numeric type is introduced on the device side.
//!
//! Interleaved `[re0, im0, re1, im1, …]` is the NumPy/PyO3 wire format only; it
//! appears at the `pyscf-py` boundary and nowhere else. [`CTensor::from_interleaved`]
//! and [`CTensor::to_interleaved`] are the only sanctioned conversions, and they
//! are exact (pure element moves — no arithmetic, so no rounding).

/// Planar (split) complex matrix/vector. `re` and `im` always have equal length.
/// Row-major unless a function explicitly documents F-order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CTensor {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl CTensor {
    /// All-zero tensor of logical complex length `n`.
    pub fn zeros(n: usize) -> Self {
        let out = Self {
            re: vec![0.0; n],
            im: vec![0.0; n],
        };
        debug_assert_eq!(out.re.len(), out.im.len(), "CTensor::zeros: plane mismatch");
        out
    }

    /// Build from the interleaved wire format `[re0, im0, re1, im1, …]`.
    ///
    /// # Panics
    /// Debug-asserts that `z.len()` is even. In release an odd trailing real
    /// component is dropped rather than producing unequal planes.
    pub fn from_interleaved(z: &[f64]) -> Self {
        debug_assert_eq!(
            z.len() % 2,
            0,
            "CTensor::from_interleaved: interleaved buffer length {} is odd",
            z.len()
        );
        let n = z.len() / 2;
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for c in z.chunks_exact(2) {
            re.push(c[0]);
            im.push(c[1]);
        }
        let out = Self { re, im };
        debug_assert_eq!(
            out.re.len(),
            out.im.len(),
            "CTensor::from_interleaved: plane mismatch"
        );
        out
    }

    /// Emit the interleaved wire format `[re0, im0, re1, im1, …]`.
    /// Exactly inverts [`CTensor::from_interleaved`] — no arithmetic is done, so
    /// the round-trip is bit-exact for every finite and non-finite value.
    pub fn to_interleaved(&self) -> Vec<f64> {
        debug_assert_eq!(
            self.re.len(),
            self.im.len(),
            "CTensor::to_interleaved: plane mismatch"
        );
        let mut out = Vec::with_capacity(2 * self.re.len());
        for (r, i) in self.re.iter().zip(self.im.iter()) {
            out.push(*r);
            out.push(*i);
        }
        out
    }

    /// Lift a purely real buffer: `im` is all zeros.
    pub fn from_real(re: &[f64]) -> Self {
        let out = Self {
            re: re.to_vec(),
            im: vec![0.0; re.len()],
        };
        debug_assert_eq!(
            out.re.len(),
            out.im.len(),
            "CTensor::from_real: plane mismatch"
        );
        out
    }

    /// Build directly from two planes.
    ///
    /// Not part of the §5.1 surface but used pervasively by `zgemm`/`zblas`,
    /// which assemble the two planes independently from real primitives.
    pub fn from_planes(re: Vec<f64>, im: Vec<f64>) -> Self {
        debug_assert_eq!(
            re.len(),
            im.len(),
            "CTensor::from_planes: plane mismatch (re {} vs im {})",
            re.len(),
            im.len()
        );
        Self { re, im }
    }

    /// Logical complex length (`== re.len() == im.len()`).
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.re.len(), self.im.len(), "CTensor::len: plane mismatch");
        self.re.len()
    }

    /// `true` when the tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Complex conjugate: `im` negated, `re` untouched.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: self.im.iter().map(|v| -v).collect(),
        }
    }

    /// `true` when `max|im| < tol` (PBC-MASTER-PLAN §5.1). The comparison is
    /// STRICT, so `tol = 0.0` is always `false` for a non-empty tensor; an empty
    /// tensor is vacuously real.
    pub fn is_real(&self, tol: f64) -> bool {
        self.im.iter().all(|v| v.abs() < tol)
    }
}
