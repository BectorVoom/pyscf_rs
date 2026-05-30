
```rust
use anyhow::Result;
use cubecl_core::{self as cubecl, prelude::{*, Cos, Sin}};
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_runtime::client::ComputeClient;
use std::f64::consts::PI;

const NX: usize = 4;
const NY: usize = 4;
const NZ: usize = 4;
const N: usize = NX * NY * NZ;

#[inline(always)]
fn idx3d(x: usize, y: usize, z: usize) -> usize {
    (z * NY + y) * NX + x
}

#[cube(launch_unchecked)]
pub fn fft_x(
    in_re: &Array<f64>,
    in_im: &Array<f64>,
    out_re: &mut Array<f64>,
    out_im: &mut Array<f64>,
    #[comptime] nx: usize,
    #[comptime] ny: usize,
    #[comptime] nz: usize,
) {
    let idx = ABSOLUTE_POS;
    let total = (nx * ny * nz) as u32;
    if idx >= total {
        terminate!();
    }

    let nx_u = nx as u32;
    let ny_u = ny as u32;

    let kx = idx % nx_u;
    let tmp = idx / nx_u;
    let y = tmp % ny_u;
    let z = tmp / ny_u;

    let base = (z * ny_u + y) * nx_u;
    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let two_pi = 2.0f64 * PI;

    let kx_f = kx as f64;
    let nx_f = nx_u as f64;

    let mut x = 0u32;
    while x < nx_u {
        let theta = -two_pi * kx_f * (x as f64) / nx_f;
        let cos_t = f64::cos(theta);
        let sin_t = f64::sin(theta);
        let src = base + x;
        let a_re = in_re[src];
        let a_im = in_im[src];
        sum_re += a_re * cos_t - a_im * sin_t;
        sum_im += a_re * sin_t + a_im * cos_t;
        x += 1;
    }

    out_re[idx] = sum_re;
    out_im[idx] = sum_im;
}

#[cube(launch_unchecked)]
pub fn fft_y(
    in_re: &Array<f64>,
    in_im: &Array<f64>,
    out_re: &mut Array<f64>,
    out_im: &mut Array<f64>,
    #[comptime] nx: usize,
    #[comptime] ny: usize,
    #[comptime] nz: usize,
) {
    let idx = ABSOLUTE_POS;
    let total = (nx * ny * nz) as u32;
    if idx >= total {
        terminate!();
    }

    let nx_u = nx as u32;
    let ny_u = ny as u32;

    let kx = idx % nx_u;
    let tmp = idx / nx_u;
    let ky = tmp % ny_u;
    let z = tmp / ny_u;

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let two_pi = 2.0f64 * PI;
    let ky_f = ky as f64;
    let ny_f = ny_u as f64;

    let mut y = 0u32;
    while y < ny_u {
        let theta = -two_pi * ky_f * (y as f64) / ny_f;
        let cos_t = f64::cos(theta);
        let sin_t = f64::sin(theta);
        let src = (z * ny_u + y) * nx_u + kx;
        let a_re = in_re[src];
        let a_im = in_im[src];
        sum_re += a_re * cos_t - a_im * sin_t;
        sum_im += a_re * sin_t + a_im * cos_t;
        y += 1;
    }

    out_re[idx] = sum_re;
    out_im[idx] = sum_im;
}

#[cube(launch_unchecked)]
pub fn fft_z(
    in_re: &Array<f64>,
    in_im: &Array<f64>,
    out_re: &mut Array<f64>,
    out_im: &mut Array<f64>,
    #[comptime] nx: usize,
    #[comptime] ny: usize,
    #[comptime] nz: usize,
) {
    let idx = ABSOLUTE_POS;
    let total = (nx * ny * nz) as u32;
    if idx >= total {
        terminate!();
    }

    let nx_u = nx as u32;
    let ny_u = ny as u32;
    let nz_u = nz as u32;

    let kx = idx % nx_u;
    let tmp = idx / nx_u;
    let ky = tmp % ny_u;
    let kz = tmp / ny_u;

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let two_pi = 2.0f64 * PI;
    let kz_f = kz as f64;
    let nz_f = nz_u as f64;

    let mut z = 0u32;
    while z < nz_u {
        let theta = -two_pi * kz_f * (z as f64) / nz_f;
        let cos_t = f64::cos(theta);
        let sin_t = f64::sin(theta);
        let src = (z * ny_u + ky) * nx_u + kx;
        let a_re = in_re[src];
        let a_im = in_im[src];
        sum_re += a_re * cos_t - a_im * sin_t;
        sum_im += a_re * sin_t + a_im * cos_t;
        z += 1;
    }

    out_re[idx] = sum_re;
    out_im[idx] = sum_im;
}

fn cpu_dft3d(in_re: &[f64], in_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut out_re = vec![0.0f64; N];
    let mut out_im = vec![0.0f64; N];
    let two_pi = 2.0f64 * PI;

    for kz in 0..NZ {
        for ky in 0..NY {
            for kx in 0..NX {
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;
                let kx_f = kx as f64;
                let ky_f = ky as f64;
                let kz_f = kz as f64;

                for z in 0..NZ {
                    for y in 0..NY {
                        for x in 0..NX {
                            let idx = idx3d(x, y, z);
                            let angle = -two_pi
                                * ((kx_f * x as f64) / NX as f64
                                    + (ky_f * y as f64) / NY as f64
                                    + (kz_f * z as f64) / NZ as f64);
                            let cos_t = angle.cos();
                            let sin_t = angle.sin();
                            let a_re = in_re[idx];
                            let a_im = in_im[idx];
                            sum_re += a_re * cos_t - a_im * sin_t;
                            sum_im += a_re * sin_t + a_im * cos_t;
                        }
                    }
                }

                let out_idx = idx3d(kx, ky, kz);
                out_re[out_idx] = sum_re;
                out_im[out_idx] = sum_im;
            }
        }
    }

    (out_re, out_im)
}

fn main() -> Result<()> {
    // 1) Set up CPU runtime and compute client
    let device = CpuDevice::default();
    let client: ComputeClient<_> = ComputeClient::load(&device);

    // 2) Prepare host data (split-complex)
    let mut input_re = vec![0.0f64; N];
    let mut input_im = vec![0.0f64; N];
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let idx = idx3d(x, y, z);
                input_re[idx] = (x as f64) + 10.0 * (y as f64) + 100.0 * (z as f64);
                input_im[idx] = 0.0;
            }
        }
    }

    // 3) Allocate device buffers
    let in_re_handle = client.create(f64::as_bytes(&input_re));
    let in_im_handle = client.create(f64::as_bytes(&input_im));
    let tmp1_re_handle = client.empty(N * core::mem::size_of::<f64>());
    let tmp1_im_handle = client.empty(N * core::mem::size_of::<f64>());
    let tmp2_re_handle = client.empty(N * core::mem::size_of::<f64>());
    let tmp2_im_handle = client.empty(N * core::mem::size_of::<f64>());
    let out_re_handle = client.empty(N * core::mem::size_of::<f64>());
    let out_im_handle = client.empty(N * core::mem::size_of::<f64>());

    // Launch configuration
    let cube_dim = CubeDim::new_1d(N as u32); // one cube covers all elements (64 threads)
    let cube_count = CubeCount::new_1d(1);

    unsafe {
        fft_x::launch_unchecked::<CpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&in_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&in_im_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp1_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp1_im_handle, N, 1),
            NX,
            NY,
            NZ,
        );

        fft_y::launch_unchecked::<CpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&tmp1_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp1_im_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp2_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp2_im_handle, N, 1),
            NX,
            NY,
            NZ,
        );

        fft_z::launch_unchecked::<CpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&tmp2_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&tmp2_im_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&out_re_handle, N, 1),
            ArrayArg::from_raw_parts::<f64>(&out_im_handle, N, 1),
            NX,
            NY,
            NZ,
        );
    }

    // 4) Read back results
    let out_re_bytes = client.read_one(out_re_handle);
    let out_im_bytes = client.read_one(out_im_handle);
    let out_re = f64::from_bytes(&out_re_bytes);
    let out_im = f64::from_bytes(&out_im_bytes);

    // 5) CPU reference for validation
    let (ref_re, ref_im) = cpu_dft3d(&input_re, &input_im);

    let mut max_err = 0.0f64;
    for i in 0..N {
        let dr = (out_re[i] - ref_re[i]).abs();
        let di = (out_im[i] - ref_im[i]).abs();
        max_err = max_err.max(dr.max(di));
    }

    println!("Computed 3D FFT on {} points ({}x{}x{}).", N, NX, NY, NZ);
    println!("Max abs error vs CPU reference: {:.3e}", max_err);
    println!("First few frequency bins (real, imag):");
    for i in 0..usize::min(8, N) {
        println!("  k{:02}: {:.4}, {:.4}", i, out_re[i], out_im[i]);
    }

    Ok(())
}

```