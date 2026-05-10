//! D-05: opaque Tensor / BufferId surface.

use pyscf_runtime::DType;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) u64);

impl BufferId {
    #[doc(hidden)]
    pub(crate) fn from_raw(id: u64) -> Self { Self(id) }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: BufferId,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn elem_size(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
    pub fn nbytes(&self) -> usize {
        self.numel().saturating_mul(self.elem_size())
    }

    /// Phase-1 placeholder constructor for integration tests. The id is
    /// a sentinel never dereferenced by Phase 1 primitives. Phase 2's
    /// allocator replaces this with a real id from BufferId::from_raw.
    #[doc(hidden)]
    pub fn placeholder(shape: Vec<usize>, dtype: DType) -> Self {
        Self { id: BufferId(u64::MAX), shape, dtype }
    }
}
