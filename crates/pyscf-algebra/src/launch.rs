//! Device-aware launch geometry and host->device staging.
//!
//! Every engine in this crate used to hard-code `CubeDim::new_1d(256)` (or a
//! 16x16 / 32x8 two-dimensional cube) and a hand-rolled `div_ceil` cube count.
//! That is a reasonable shape for a discrete GPU and a pathological one for
//! CubeCL's CPU runtime, which is the DEFAULT backend here (`default = ["cpu"]`
//! per ALG-03): a "unit" there is an operating-system thread, and the runtime
//! dispatches one task per unit in the cube and blocks until all of them report
//! back. Asking for 256 units to add 256 floats spends far more time in the
//! thread pool than in the arithmetic.
//!
//! This module centralizes the two decisions that fix it:
//!
//!   * [`launch_1d`] derives the cube dimension FROM THE DEVICE — a whole number
//!     of hardware planes where planes exist, and a work-proportional thread
//!     count capped at the core count where they do not.
//!   * [`line_size_for`] picks the widest vector width the device likes for `F`
//!     that divides the element count exactly, so the kernels can read and write
//!     `Vector<F, N>` (one SIMD load) instead of one scalar per unit.
//!
//! Both are crate-private: the cubecl `Runtime` bound stays inside the ALG-06
//! wall exactly as it does in the engines that call them.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::scalar::DeviceScalar;

/// Element operations one unit should be worth before another unit is asked for,
/// on runtimes where a "unit" is an operating-system thread.
///
/// CubeCL's CPU runtime dispatches one task per unit in the cube to a pool of
/// worker threads and blocks until all of them report back, so each extra unit
/// costs roughly a microsecond of pure dispatch. A microsecond buys a few tens
/// of thousands of element operations, which is where this threshold comes from:
/// below it, a second thread is a loss.
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// Ceiling on the units per cube [`launch_1d`] will ask a CPU-like runtime for.
///
/// The width it picks is already bounded by the core count; this only guards
/// against a runtime that reports an implausible one.
const CPU_CUBE_DIM_MAX: u32 = 64;

/// Whether this device has real hardware planes (a GPU-like runtime).
///
/// The CPU runtime reports `plane_size_max == 1`: it has no warps/wavefronts, so
/// `sync_cube` is not a hardware instruction and shared memory is just more
/// ordinary memory. Kernels that stage tiles through shared memory and
/// synchronize are the right shape when this is true and the wrong shape when it
/// is false, which is the choice [`crate::gemm`] makes with it.
pub fn has_planes<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// Launch geometry for a kernel that assigns one unit to each of `lanes` items
/// and does roughly `work_per_lane` element operations per lane.
///
/// The cube dimension is derived from the device rather than fixed, because the
/// two runtime families want opposite things:
///
///   * GPU-like runtimes (`plane_size_max > 1`) want a cube that is a whole
///     number of planes wide; [`CubeDim::new`] sizes that from the hardware.
///   * CubeCL's CPU runtime has no hardware planes — every unit is a thread and
///     the cube *count* is a serial loop inside each thread. There, extra units
///     are pure overhead until the kernel has enough work to amortise them, so
///     the width grows with the total work and stops at the core count.
pub fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    work_per_lane: usize,
) -> (CubeCount, CubeDim) {
    let hardware = &client.properties().hardware;
    let cube_dim = if hardware.plane_size_max > 1 {
        CubeDim::new(client, lanes)
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let total = lanes.saturating_mul(work_per_lane.max(1));
        let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes.max(1)));
        CubeDim::new_1d((units as u32).min(CPU_CUBE_DIM_MAX))
    };
    (
        cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim),
        cube_dim,
    )
}

/// The widest vector width the device likes for `F` that divides `num_elems`.
///
/// Kernels over flat, contiguous buffers read and write [`Vector<F, N>`]s of this
/// many elements, which is what lets one unit issue a full SIMD load instead of
/// a scalar one. A width of `1` means "no vectorization" and the kernel degrades
/// to the scalar shape without a separate code path. The count must divide
/// exactly: a partial trailing vector would read past the buffer.
pub fn line_size_for<R: Runtime, F: DeviceScalar>(
    client: &ComputeClient<R>,
    num_elems: usize,
) -> usize {
    if num_elems == 0 {
        return 1;
    }
    client
        .io_optimized_vector_sizes(core::mem::size_of::<F>())
        .find(|width| num_elems.is_multiple_of(*width))
        .unwrap_or(1)
}

/// Stage a host slice into a fresh device buffer.
///
/// `Bytes::from_elems` takes ownership of a `Vec`, so every launcher here used to
/// write `client.create(Bytes::from_elems(x.to_vec()))` — a full host-side copy
/// of the operand before the copy that actually reaches the device.
/// `create_from_slice` skips the first one.
pub fn upload<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, data: &[F]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(data))
}

/// The total number of units a `(CubeCount, CubeDim)` pair dispatches.
///
/// Grid-stride reduction kernels need this twice: to size the partials buffer
/// they write one entry each into, and as the stride itself (`ABSOLUTE_POS`
/// spans the whole grid, so `CUBE_COUNT_X * CUBE_DIM_X` is only the right stride
/// when the count is one-dimensional — this is right regardless).
pub fn total_units(count: &CubeCount, dim: CubeDim) -> usize {
    let cubes = match count {
        CubeCount::Static(x, y, z) => (*x as usize) * (*y as usize) * (*z as usize),
        // Dynamic counts come from a device-side binding, so the host cannot know
        // the extent. Nothing in this crate launches one; be conservative.
        _ => 1,
    };
    cubes * dim.num_elems() as usize
}

/// How many grid-stride lanes a reduction over `n_lines` vectors should use.
///
/// Reductions cannot size their grid the way an element-wise kernel does: one
/// unit per element would produce one partial per element, which is not a
/// reduction at all — every partial has to be read back and summed on the host.
/// So the lane count is capped, and each lane walks the input in grid strides.
///
/// On a GPU the cap is high enough to keep every compute unit busy while keeping
/// the read-back to tens of kilobytes. On CubeCL's CPU runtime a lane is a
/// thread, so the useful count is the core count, scaled down when there is not
/// enough work to amortise the dispatch.
/// `line` is the vector width, so that `n_lines * line` is the number of scalar
/// element operations the reduction performs — the unit [`WORK_PER_CPU_UNIT`] is
/// expressed in. Passing the vector count alone would under-count the work by
/// that width and leave a wide reduction running on a fraction of the cores.
pub fn reduction_lanes<R: Runtime>(
    client: &ComputeClient<R>,
    n_lines: usize,
    line: usize,
) -> usize {
    /// Grid-stride lanes a GPU-like backend is asked for at most.
    const GPU_LANES_MAX: usize = 8192;

    let hardware = &client.properties().hardware;
    if hardware.plane_size_max > 1 {
        n_lines.clamp(1, GPU_LANES_MAX)
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let work = n_lines.saturating_mul(line.max(1));
        (work / WORK_PER_CPU_UNIT).clamp(1, cores.min(n_lines.max(1)))
    }
}
