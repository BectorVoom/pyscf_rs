//! Benchmark suite for CubeCL ROCm GPU kernels and PySCF-RS quantum chemistry methods.

use cubecl::Runtime;
use std::time::{Duration, Instant};

use pyscf_algebra::client::AlgebraClient;
use pyscf_algebra::{
    axpy_dense, dot_dense, gemm_dense, reduce_sum_dense, scal_dense, transpose_dense,
};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, Unit};

fn time_op<F: FnMut()>(mut f: F, warm: usize, reps: usize) -> (Duration, Duration) {
    for _ in 0..warm {
        f();
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];
    (median, min)
}

fn bench_algebra(client: &AlgebraClient, backend_name: &str) {
    println!("\n{}", "=".repeat(80));
    println!(" LINEAR ALGEBRA BENCHMARKS ON [{backend_name}]");
    println!("{}", "=".repeat(80));
    println!(
        "{:<24} | {:<16} | {:<16} | {:<16}",
        "Operation", "Size", "Median Time", "Min Time"
    );
    println!("{}", "-".repeat(80));

    // 1. GEMM
    let gemm_sizes = if matches!(client, AlgebraClient::Cpu(_)) {
        vec![256]
    } else {
        vec![256, 512, 1024, 2048]
    };
    for &n in &gemm_sizes {
        let a = vec![1.001_f64; n * n];
        let b = vec![0.999_f64; n * n];
        let (med, min) = time_op(
            || {
                let _ = gemm_dense(client, &a, &b, n, n, n).unwrap();
            },
            2,
            5,
        );
        let gflops = 2.0 * (n as f64).powi(3) / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GFLOPS)",
            "GEMM (f64)",
            format!("{n}x{n}"),
            med,
            min,
            gflops
        );
    }

    let vec_sizes = if matches!(client, AlgebraClient::Cpu(_)) {
        vec![100_000]
    } else {
        vec![100_000, 1_000_000, 10_000_000]
    };

    // 2. DOT
    for &n in &vec_sizes {
        let x = vec![1.001_f64; n];
        let y = vec![0.999_f64; n];
        let (med, min) = time_op(
            || {
                let _ = dot_dense(client, &x, &y).unwrap();
            },
            3,
            10,
        );
        let gb_s = 2.0 * (n * 8) as f64 / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GB/s)",
            "DOT (f64)",
            format!("{n} elem"),
            med,
            min,
            gb_s
        );
    }

    // 3. AXPY
    for &n in &vec_sizes {
        let x = vec![1.001_f64; n];
        let mut y = vec![0.999_f64; n];
        let (med, min) = time_op(
            || {
                axpy_dense(client, 2.5, &x, &mut y).unwrap();
            },
            3,
            10,
        );
        let gb_s = 3.0 * (n * 8) as f64 / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GB/s)",
            "AXPY (f64)",
            format!("{n} elem"),
            med,
            min,
            gb_s
        );
    }

    // 4. SCAL
    for &n in &vec_sizes {
        let mut x = vec![1.001_f64; n];
        let (med, min) = time_op(
            || {
                scal_dense(client, 1.01, &mut x).unwrap();
            },
            3,
            10,
        );
        let gb_s = 2.0 * (n * 8) as f64 / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GB/s)",
            "SCAL (f64)",
            format!("{n} elem"),
            med,
            min,
            gb_s
        );
    }

    // 5. REDUCE SUM
    for &n in &vec_sizes {
        let x = vec![1.001_f64; n];
        let (med, min) = time_op(
            || {
                let _ = reduce_sum_dense(client, &x).unwrap();
            },
            3,
            10,
        );
        let gb_s = (n * 8) as f64 / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GB/s)",
            "REDUCE SUM (f64)",
            format!("{n} elem"),
            med,
            min,
            gb_s
        );
    }

    // 6. TRANSPOSE
    let transpose_sizes = if matches!(client, AlgebraClient::Cpu(_)) {
        vec![512]
    } else {
        vec![512, 1024, 2048, 4096]
    };
    for &n in &transpose_sizes {
        let x = vec![1.001_f64; n * n];
        let (med, min) = time_op(
            || {
                let _ = transpose_dense(client, &x, n, n).unwrap();
            },
            2,
            5,
        );
        let gb_s = 2.0 * (n * n * 8) as f64 / min.as_secs_f64() / 1e9;
        println!(
            "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.1} GB/s)",
            "TRANSPOSE (f64)",
            format!("{n}x{n}"),
            med,
            min,
            gb_s
        );
    }
}

fn bench_eval_gto() {
    println!("\n{}", "=".repeat(80));
    println!(" GTO ON GRID EVALUATION (pyscf_gto::eval_gto)");
    println!("{}", "=".repeat(80));
    println!(
        "{:<24} | {:<16} | {:<16} | {:<16}",
        "System", "Grid Points", "Median Time", "Throughput"
    );
    println!("{}", "-".repeat(80));

    let h2o = pyscf_gto::M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0".to_string()),
        basis: BasisInput::Name("cc-pvdz".to_string()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .unwrap();

    let benzene = pyscf_gto::M(MoleBuildArgs {
        atom: AtomInput::String(
            "
            C  0.0000  1.3970  0.0000
            C  1.2099  0.6985  0.0000
            C  1.2099 -0.6985  0.0000
            C  0.0000 -1.3970  0.0000
            C -1.2099 -0.6985  0.0000
            C -1.2099  0.6985  0.0000
            H  0.0000  2.4810  0.0000
            H  2.1486  1.2405  0.0000
            H  2.1486 -1.2405  0.0000
            H  0.0000 -2.4810  0.0000
            H -2.1486 -1.2405  0.0000
            H -2.1486  1.2405  0.0000
        "
            .to_string(),
        ),
        basis: BasisInput::Name("6-31g*".to_string()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .unwrap();

    for (name, mol) in &[
        ("H2O/cc-pVDZ (24 AO)", &h2o),
        ("Benzene/6-31G* (102 AO)", &benzene),
    ] {
        for &ngrids in &[5_000, 20_000, 100_000, 500_000] {
            let coords: Vec<[f64; 3]> = (0..ngrids)
                .map(|i| {
                    let f = i as f64;
                    [f.sin() * 3.0, f.cos() * 3.0, (f * 0.5).sin() * 3.0]
                })
                .collect();

            let (med, min) = time_op(
                || {
                    let _ = pyscf_gto::eval_gto(mol, "GTOval_sph", &coords).unwrap();
                },
                2,
                5,
            );
            let mpts_s = (ngrids as f64 * mol.nao_nr as f64) / min.as_secs_f64() / 1e6;
            println!(
                "{:<24} | {:<16} | {:<16.2?} | {:<10.2?} ({:.2} M AO*pts/s)",
                name,
                format!("{ngrids} pts"),
                med,
                min,
                mpts_s
            );
        }
    }
}

fn main() {
    println!("{}", "#".repeat(80));
    println!(" PYSCF-RS PERFORMANCE & ROCm GPU BENCHMARK SUITE");
    println!("{}", "#".repeat(80));

    #[cfg(feature = "rocm")]
    {
        println!("\n>>> Initializing AMD ROCm HIP GPU Device...");
        let rocm_device = cubecl_hip::AmdDevice::default();
        let rocm_client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&rocm_device));
        bench_algebra(&rocm_client, "ROCm GPU (AMD HIP)");
    }

    #[cfg(all(feature = "cpu", not(feature = "rocm")))]
    {
        let cpu_client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(
            &cubecl_cpu::CpuDevice::default(),
        ));
        bench_algebra(&cpu_client, "CPU (CubeCL / Multi-core)");
    }

    bench_eval_gto();
}
