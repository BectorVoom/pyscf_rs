//! `ZArr` — a shaped, row-major planar-complex host array, and the deterministic
//! `einsum` every Phase-16 contraction is written in.
//!
//! # Why an einsum at all
//!
//! `pyscf/pbc/cc/*.py` is written almost entirely in `lib.einsum`. Transcribing
//! several thousand such lines into hand-rolled index loops is where a
//! transposition slips in unnoticed — the class of defect this project has
//! already paid **+6 306 866.73 Ha** for once (14-05's `decompose_j2c`,
//! `16-CONTEXT §3.4`). Keeping the subscript string in the Rust source next to
//! the upstream line it came from makes the transcription checkable by reading.
//!
//! # D-PBC-29 clause 2 — the primitive, named once, here
//!
//! Every contraction in this phase is a **host loop over k-point triples with
//! `oracle_*` accumulators, not `zgemm_dense`**. The standing measurement
//! `zgemm-dense-loses-to-host-rayon` records `zgemm_dense` at 6-12× slower on
//! the CPU backend — the default here — and 1.35e-10 off, outside this
//! project's 1e-11 gate.
//!
//! **The primitive is [`oracle_zsum`], the UNCONJUGATED ordered complex sum,
//! and `16-CONTEXT §3.2` is satisfied at the source rather than site by site:**
//! `numpy.einsum` / `lib.einsum` never conjugate an operand. Wherever upstream
//! wants a conjugate it writes `.conj()` explicitly (`kccsd_rhf.py:47-66`'s
//! `energy`, `transform_symm`'s operations 2 and 3), and this port applies
//! [`ZArr::conj`] at exactly those points. So an einsum-transcribed line is
//! unconjugated **by construction**, which is the property `15-REVIEW.md
//! D-15-R-02` found a plan could not state safely in prose: there, "route
//! through `oracle_dot`" silently produced `Σ x·x` instead of `Σ conj(x)·y`.
//! Here the subscript string carries the meaning and the conjugation is
//! visible as a separate call.
//!
//! Determinism (§9.3): each output element is one [`oracle_zsum`] over a
//! fixed-length product buffer, whose pairwise recursion tree depends only on
//! that length — never on `RAYON_NUM_THREADS`, the scheduler or the partition.
//! Output elements never mix. So a `ZArr` einsum is bit-identical at any thread
//! count, by construction rather than by test — though 16-05 test 7 gates it
//! anyway.

use pyscf_algebra::{CTensor, oracle_zsum};
use rayon::prelude::*;

use crate::error::PbcCcError;

/// A row-major, shaped, planar-complex array.
#[derive(Debug, Clone, PartialEq)]
pub struct ZArr {
    shape: Vec<usize>,
    data: CTensor,
}

impl ZArr {
    /// All-zero array of `shape`.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: CTensor::zeros(n),
        }
    }

    /// Wrap an existing planar buffer.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the buffer length does not match `shape`.
    pub fn from_ctensor(shape: &[usize], data: CTensor) -> Result<Self, PbcCcError> {
        let n: usize = shape.iter().product();
        if data.re.len() != n || data.im.len() != n {
            return Err(PbcCcError::Shape(format!(
                "ZArr::from_ctensor: shape {shape:?} needs {n} elements, got {}",
                data.re.len()
            )));
        }
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Lift a purely real buffer.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a length mismatch.
    pub fn from_real(shape: &[usize], re: &[f64]) -> Result<Self, PbcCcError> {
        Self::from_ctensor(shape, CTensor::from_real(re))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.re.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.re.is_empty()
    }

    pub fn data(&self) -> &CTensor {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut CTensor {
        &mut self.data
    }

    pub fn into_ctensor(self) -> CTensor {
        self.data
    }

    /// Row-major strides.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1_usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    /// Element at a multi-index.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a rank or bound violation.
    pub fn at(&self, idx: &[usize]) -> Result<(f64, f64), PbcCcError> {
        let f = self.flat(idx)?;
        Ok((self.data.re[f], self.data.im[f]))
    }

    fn flat(&self, idx: &[usize]) -> Result<usize, PbcCcError> {
        if idx.len() != self.shape.len() {
            return Err(PbcCcError::Shape(format!(
                "ZArr index rank {} for shape {:?}",
                idx.len(),
                self.shape
            )));
        }
        let mut f = 0_usize;
        for (i, (&x, &d)) in idx.iter().zip(self.shape.iter()).enumerate() {
            if x >= d {
                return Err(PbcCcError::Shape(format!(
                    "ZArr index {x} out of bounds at axis {i} of shape {:?}",
                    self.shape
                )));
            }
            f = f * d + x;
        }
        Ok(f)
    }

    /// `numpy.transpose(axes)` — `out[i0,i1,…] = self[i_{axes[0]}, …]` in
    /// numpy's sense: `axes[k]` is the SOURCE axis that becomes output axis `k`.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if `axes` is not a permutation of this array's axes.
    pub fn transpose(&self, axes: &[usize]) -> Result<Self, PbcCcError> {
        let nd = self.shape.len();
        if axes.len() != nd {
            return Err(PbcCcError::Shape(format!(
                "transpose: {} axes for a rank-{nd} array",
                axes.len()
            )));
        }
        let mut seen = vec![false; nd];
        for &a in axes {
            if a >= nd || seen[a] {
                return Err(PbcCcError::Shape(format!(
                    "transpose: {axes:?} is not a permutation of 0..{nd}"
                )));
            }
            seen[a] = true;
        }
        let out_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let src_strides = self.strides();
        let mut out = Self::zeros(&out_shape);
        let n = self.len();
        let mut idx = vec![0_usize; nd];
        for dst in 0..n {
            let mut src = 0_usize;
            for k in 0..nd {
                src += idx[k] * src_strides[axes[k]];
            }
            out.data.re[dst] = self.data.re[src];
            out.data.im[dst] = self.data.im[src];
            // odometer over the OUTPUT shape
            for k in (0..nd).rev() {
                idx[k] += 1;
                if idx[k] < out_shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Ok(out)
    }

    /// Element-wise complex conjugate — the ONLY place a conjugation enters a
    /// contraction in this crate (see the module doc).
    pub fn conj(&self) -> Self {
        let mut out = self.clone();
        for v in out.data.im.iter_mut() {
            *v = -*v;
        }
        out
    }

    /// `self += other`, element-wise.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a shape mismatch.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), PbcCcError> {
        self.zip_assign(other, 1.0)
    }

    /// `self -= other`, element-wise.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a shape mismatch.
    pub fn sub_assign(&mut self, other: &Self) -> Result<(), PbcCcError> {
        self.zip_assign(other, -1.0)
    }

    /// `self += factor * other`, element-wise (real factor).
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a shape mismatch.
    pub fn zip_assign(&mut self, other: &Self, factor: f64) -> Result<(), PbcCcError> {
        if self.shape != other.shape {
            return Err(PbcCcError::Shape(format!(
                "element-wise op on shapes {:?} and {:?}",
                self.shape, other.shape
            )));
        }
        for i in 0..self.data.re.len() {
            self.data.re[i] += factor * other.data.re[i];
            self.data.im[i] += factor * other.data.im[i];
        }
        Ok(())
    }

    /// `self *= s` (real scalar).
    pub fn scale(&mut self, s: f64) {
        for i in 0..self.data.re.len() {
            self.data.re[i] *= s;
            self.data.im[i] *= s;
        }
    }

    /// `self *= (re, im)` (complex scalar).
    pub fn scale_complex(&mut self, re: f64, im: f64) {
        for i in 0..self.data.re.len() {
            let (a, b) = (self.data.re[i], self.data.im[i]);
            self.data.re[i] = a * re - b * im;
            self.data.im[i] = a * im + b * re;
        }
    }

    /// Reinterpret with a new shape of the same total size.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the element count differs.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, PbcCcError> {
        let n: usize = shape.iter().product();
        if n != self.len() {
            return Err(PbcCcError::Shape(format!(
                "reshape {:?} -> {shape:?} changes the element count",
                self.shape
            )));
        }
        Ok(Self {
            shape: shape.to_vec(),
            data: self.data.clone(),
        })
    }

    /// The sub-array at a leading multi-index — `x[i, j]` for a rank-4 `x`
    /// gives the rank-2 remainder. This is the operation upstream writes as
    /// `eris.oovv[kk,kl,kc]`.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the leading index is too long or out of bounds.
    pub fn slice_leading(&self, lead: &[usize]) -> Result<Self, PbcCcError> {
        if lead.len() > self.shape.len() {
            return Err(PbcCcError::Shape(format!(
                "slice_leading {lead:?} on shape {:?}",
                self.shape
            )));
        }
        let rest: Vec<usize> = self.shape[lead.len()..].to_vec();
        let block: usize = rest.iter().product();
        let mut off = 0_usize;
        for (i, &x) in lead.iter().enumerate() {
            if x >= self.shape[i] {
                return Err(PbcCcError::Shape(format!(
                    "slice_leading index {x} out of bounds at axis {i} of {:?}",
                    self.shape
                )));
            }
            off = off * self.shape[i] + x;
        }
        off *= block;
        Ok(Self {
            shape: rest,
            data: CTensor {
                re: self.data.re[off..off + block].to_vec(),
                im: self.data.im[off..off + block].to_vec(),
            },
        })
    }

    /// Write a sub-array back at a leading multi-index — the assignment form of
    /// [`ZArr::slice_leading`].
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a bound or size mismatch.
    pub fn set_leading(&mut self, lead: &[usize], value: &Self) -> Result<(), PbcCcError> {
        if lead.len() > self.shape.len() {
            return Err(PbcCcError::Shape(format!(
                "set_leading {lead:?} on shape {:?}",
                self.shape
            )));
        }
        let rest = &self.shape[lead.len()..];
        if rest != value.shape.as_slice() {
            return Err(PbcCcError::Shape(format!(
                "set_leading: block shape {:?} does not match {:?}",
                value.shape, rest
            )));
        }
        let block: usize = rest.iter().product();
        let mut off = 0_usize;
        for (i, &x) in lead.iter().enumerate() {
            if x >= self.shape[i] {
                return Err(PbcCcError::Shape(format!(
                    "set_leading index {x} out of bounds at axis {i} of {:?}",
                    self.shape
                )));
            }
            off = off * self.shape[i] + x;
        }
        off *= block;
        self.data.re[off..off + block].copy_from_slice(&value.data.re);
        self.data.im[off..off + block].copy_from_slice(&value.data.im);
        Ok(())
    }
}

/// One parsed einsum operand.
struct Operand {
    letters: Vec<u8>,
}

/// `lib.einsum(spec, a, b, …)` over [`ZArr`] operands.
///
/// `spec` is the ordinary subscript string, e.g. `"klcd,ilcd->ki"`. Repeated
/// letters within one operand (a diagonal) are NOT supported — upstream never
/// writes one in `pbc/cc` — and an ellipsis is not supported either; both are
/// rejected rather than silently mis-computed.
///
/// **The contraction is unconjugated**, exactly as `numpy.einsum` is; see the
/// module doc for why that is the right default here and where the conjugations
/// live instead.
///
/// # Errors
/// [`PbcCcError::Shape`] on a malformed spec, an operand-count mismatch, an
/// index whose extent disagrees between operands, or an output letter that
/// appears in no operand.
pub fn einsum(spec: &str, ops: &[&ZArr]) -> Result<ZArr, PbcCcError> {
    let (lhs, out_letters) = match spec.split_once("->") {
        Some((l, r)) => (l, r.as_bytes().to_vec()),
        None => {
            return Err(PbcCcError::Shape(format!(
                "einsum spec {spec:?} must contain '->'"
            )));
        }
    };
    let operands: Vec<Operand> = lhs
        .split(',')
        .map(|s| Operand {
            letters: s.trim().as_bytes().to_vec(),
        })
        .collect();
    if operands.len() != ops.len() {
        return Err(PbcCcError::Shape(format!(
            "einsum {spec:?}: {} subscript groups for {} arrays",
            operands.len(),
            ops.len()
        )));
    }
    for (i, (o, a)) in operands.iter().zip(ops.iter()).enumerate() {
        if o.letters.len() != a.shape().len() {
            return Err(PbcCcError::Shape(format!(
                "einsum {spec:?}: operand {i} has {} subscripts for shape {:?}",
                o.letters.len(),
                a.shape()
            )));
        }
        let mut seen = o.letters.clone();
        seen.sort_unstable();
        let n = seen.len();
        seen.dedup();
        if seen.len() != n {
            return Err(PbcCcError::Shape(format!(
                "einsum {spec:?}: operand {i} repeats a subscript (diagonals are \
                 not supported; upstream pbc/cc never writes one)"
            )));
        }
    }

    // Extents, checked for consistency across operands.
    let mut extent: [usize; 256] = [0; 256];
    for (o, a) in operands.iter().zip(ops.iter()) {
        for (&c, &d) in o.letters.iter().zip(a.shape().iter()) {
            let e = &mut extent[c as usize];
            if *e == 0 {
                *e = d;
            } else if *e != d {
                return Err(PbcCcError::Shape(format!(
                    "einsum {spec:?}: subscript '{}' has extents {} and {d}",
                    c as char, *e
                )));
            }
        }
    }
    for &c in &out_letters {
        if extent[c as usize] == 0 {
            return Err(PbcCcError::Shape(format!(
                "einsum {spec:?}: output subscript '{}' appears in no operand",
                c as char
            )));
        }
    }

    // Contracted letters, in first-appearance order so the summation order is
    // deterministic and reproducible from the spec alone.
    let mut con_letters: Vec<u8> = Vec::new();
    for o in &operands {
        for &c in &o.letters {
            if !out_letters.contains(&c) && !con_letters.contains(&c) {
                con_letters.push(c);
            }
        }
    }

    let out_shape: Vec<usize> = out_letters.iter().map(|&c| extent[c as usize]).collect();
    let con_shape: Vec<usize> = con_letters.iter().map(|&c| extent[c as usize]).collect();
    let out_size: usize = out_shape.iter().product();
    let con_size: usize = con_shape.iter().product();

    // Per-operand stride for each output / contracted letter (0 when absent).
    let mut out_strides = vec![vec![0_usize; out_letters.len()]; ops.len()];
    let mut con_strides = vec![vec![0_usize; con_letters.len()]; ops.len()];
    for (i, (o, a)) in operands.iter().zip(ops.iter()).enumerate() {
        let st = a.strides();
        for (pos, &c) in o.letters.iter().enumerate() {
            if let Some(k) = out_letters.iter().position(|&x| x == c) {
                out_strides[i][k] = st[pos];
            } else if let Some(k) = con_letters.iter().position(|&x| x == c) {
                con_strides[i][k] = st[pos];
            }
        }
    }

    let mut out = ZArr::zeros(&out_shape);
    if out_size == 0 {
        return Ok(out);
    }

    // One `oracle_zsum` per output element over a fixed-length product buffer:
    // the recursion tree depends only on `con_size`, so the result is
    // bit-identical at any thread count (§9.3).
    let planes: Vec<(f64, f64)> = (0..out_size)
        .into_par_iter()
        .map(|o_flat| {
            let mut o_idx = vec![0_usize; out_letters.len()];
            let mut rem = o_flat;
            for k in (0..out_shape.len()).rev() {
                o_idx[k] = rem % out_shape[k];
                rem /= out_shape[k];
            }
            let mut base = vec![0_usize; ops.len()];
            for (i, b) in base.iter_mut().enumerate() {
                let mut acc = 0_usize;
                for (k, &x) in o_idx.iter().enumerate() {
                    acc += x * out_strides[i][k];
                }
                *b = acc;
            }
            let mut buf = CTensor::zeros(con_size);
            let mut c_idx = vec![0_usize; con_letters.len()];
            for c_flat in 0..con_size {
                let mut pr = 1.0_f64;
                let mut pi = 0.0_f64;
                for (i, a) in ops.iter().enumerate() {
                    let mut off = base[i];
                    for (k, &x) in c_idx.iter().enumerate() {
                        off += x * con_strides[i][k];
                    }
                    let (ar, ai) = (a.data.re[off], a.data.im[off]);
                    let nr = pr * ar - pi * ai;
                    let ni = pr * ai + pi * ar;
                    pr = nr;
                    pi = ni;
                }
                buf.re[c_flat] = pr;
                buf.im[c_flat] = pi;
                for k in (0..con_letters.len()).rev() {
                    c_idx[k] += 1;
                    if c_idx[k] < con_shape[k] {
                        break;
                    }
                    c_idx[k] = 0;
                }
            }
            oracle_zsum(&buf)
        })
        .collect();

    for (i, (re, im)) in planes.into_iter().enumerate() {
        out.data.re[i] = re;
        out.data.im[i] = im;
    }
    Ok(out)
}

/// `einsum` with a real prefactor applied to the result — `factor * einsum(…)`,
/// the shape `Wklij += 0.5 * einsum(...)` takes upstream.
///
/// # Errors
/// As [`einsum`].
pub fn einsum_scaled(spec: &str, ops: &[&ZArr], factor: f64) -> Result<ZArr, PbcCcError> {
    let mut out = einsum(spec, ops)?;
    out.scale(factor);
    Ok(out)
}
