#!/usr/bin/env python3
"""
Comprehensive performance benchmark: PySCF vs pyscf-rs (Rust + CubeCL ROCm).
Measures execution times, speedups, and numerical accuracy across:
1. GTO on Grid evaluation (eval_gto)
2. Hartree-Fock SCF (RHF)
3. Density Functional Theory (RKS: B3LYP, PBE)
4. Møller-Plesset Perturbation Theory (MP2)
5. Coupled Cluster (CCSD)
"""

import os
import sys
import time
import numpy as np

# Ensure clean imports
from pyscf import gto as gto_up, scf as scf_up, mp as mp_up, cc as cc_up, dft as dft_up
import pyscf._native as native


def time_func(fn, n_warmup=1, n_runs=3):
    """Time a function with warm-up and multiple iterations."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times)), float(np.std(times)), res


def run_gto_benchmarks():
    print("\n" + "=" * 80)
    print(" 1. GTO ON GRID EVALUATION (eval_gto / GTOval_sph)")
    print("=" * 80)
    print(f"{'System':<12} | {'Grid Points':<12} | {'PySCF (ms)':<14} | {'pyscf-rs (ms)':<14} | {'Speedup':<10} | {'Max |Δ|':<12}")
    print("-" * 80)

    systems = [
        ("H2O/STO-3G", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g"),
        ("H2O/cc-pVDZ", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "cc-pvdz"),
        ("Benzene/6-31G*", """
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
        """, "6-31g*"),
    ]

    grid_sizes = [5000, 20000, 100000]

    for sys_name, atom, basis in systems:
        mol_up = gto_up.M(atom=atom, basis=basis, verbose=0)
        mol_rs = native.gto.M(atom=atom, basis=basis)

        for ngrids in grid_sizes:
            np.random.seed(42)
            coords = np.random.uniform(-4.0, 4.0, size=(ngrids, 3)).astype(np.float64)

            # PySCF upstream
            t_up, _, out_up = time_func(lambda: mol_up.eval_gto("GTOval_sph", coords), n_warmup=1, n_runs=3)

            try:
                t_rs, _, out_rs = time_func(lambda: mol_rs.eval_gto("GTOval_sph", coords), n_warmup=1, n_runs=3)
                max_diff = float(np.max(np.abs(out_up - out_rs)))
                speedup_str = f"{t_up / t_rs:.2f}x"
                t_rs_str = f"{t_rs * 1000:.2f}"
            except Exception as e:
                t_rs_str = f"Err: {e}"
                speedup_str = "N/A"
                max_diff = 0.0

            print(f"{sys_name:<12} | {ngrids:<12} | {t_up * 1000:<14.2f} | {t_rs_str:<14} | {speedup_str:<10} | {max_diff:<12.2e}")


def run_scf_benchmarks():
    print("\n" + "=" * 80)
    print(" 2. HARTREE-FOCK SCF (RHF)")
    print("=" * 80)
    print(f"{'System':<16} | {'AOs':<5} | {'PySCF (ms)':<14} | {'pyscf-rs (ms)':<14} | {'Speedup':<10} | {'|ΔE| (Ha)':<12}")
    print("-" * 80)

    systems = [
        ("H2O/STO-3G", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g"),
        ("H2O/6-31G*", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "6-31g*"),
        ("H2O/cc-pVDZ", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "cc-pvdz"),
        ("CH4/STO-3G", "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63", "sto-3g"),
    ]

    for sys_name, atom, basis in systems:
        mol_up = gto_up.M(atom=atom, basis=basis, verbose=0)
        mol_rs = native.gto.M(atom=atom, basis=basis)
        nao = mol_up.nao_nr()

        t_up, _, mf_up = time_func(lambda: scf_up.RHF(mol_up).run(), n_warmup=1, n_runs=2)
        t_rs, _, mf_rs = time_func(lambda: native.scf.RHF(mol_rs).run(), n_warmup=1, n_runs=2)

        diff = abs(mf_up.e_tot - mf_rs.e_tot)
        speedup = t_up / t_rs
        print(f"{sys_name:<16} | {nao:<5} | {t_up * 1000:<14.2f} | {t_rs * 1000:<14.2f} | {speedup:<10.2f} | {diff:<12.2e}")


def run_post_hf_benchmarks():
    print("\n" + "=" * 80)
    print(" 3. POST-HARTREE-FOCK: MP2 & CCSD CORRELATION")
    print("=" * 80)
    print(f"{'Method':<8} | {'System':<14} | {'PySCF (ms)':<14} | {'pyscf-rs (ms)':<14} | {'Speedup':<10} | {'|ΔE_corr| (Ha)':<14}")
    print("-" * 80)

    systems = [
        ("H2O/STO-3G", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g"),
        ("H2O/6-31G*", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "6-31g*"),
        ("H2O/cc-pVDZ", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "cc-pvdz"),
    ]

    for sys_name, atom, basis in systems:
        mol_up = gto_up.M(atom=atom, basis=basis, verbose=0)
        mol_rs = native.gto.M(atom=atom, basis=basis)

        mf_up = scf_up.RHF(mol_up).run()
        mf_rs = native.scf.RHF(mol_rs).run()

        # MP2
        t_mp_up, _, res_mp_up = time_func(lambda: mp_up.MP2(mf_up).run(), n_warmup=1, n_runs=3)
        t_mp_rs, _, res_mp_rs = time_func(lambda: native.mp.RMP2(mf_rs).run(), n_warmup=1, n_runs=3)
        d_mp = abs(res_mp_up.e_corr - res_mp_rs.e_corr)
        print(f"{'MP2':<8} | {sys_name:<14} | {t_mp_up * 1000:<14.2f} | {t_mp_rs * 1000:<14.2f} | {t_mp_up / t_mp_rs:<10.2f} | {d_mp:<14.2e}")

        # CCSD
        t_cc_up, _, res_cc_up = time_func(lambda: cc_up.CCSD(mf_up).run(), n_warmup=1, n_runs=3)
        t_cc_rs, _, res_cc_rs = time_func(lambda: native.cc.RCCSD(mf_rs).run(), n_warmup=1, n_runs=3)
        d_cc = abs(res_cc_up.e_corr - res_cc_rs.e_corr)
        print(f"{'CCSD':<8} | {sys_name:<14} | {t_cc_up * 1000:<14.2f} | {t_cc_rs * 1000:<14.2f} | {t_cc_up / t_cc_rs:<10.2f} | {d_cc:<14.2e}")


def run_dft_benchmarks():
    print("\n" + "=" * 80)
    print(" 4. DENSITY FUNCTIONAL THEORY (DFT - RKS)")
    print("=" * 80)
    print(f"{'Functional':<10} | {'System':<14} | {'PySCF (ms)':<14} | {'pyscf-rs (ms)':<14} | {'Speedup':<10} | {'|ΔE| (Ha)':<12}")
    print("-" * 80)

    systems = [
        ("H2O/STO-3G", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g"),
        ("H2O/6-31G*", "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "6-31g*"),
    ]

    for xc in ["b3lyp", "pbe"]:
        for sys_name, atom, basis in systems:
            mol_up = gto_up.M(atom=atom, basis=basis, verbose=0)
            mol_rs = native.gto.M(atom=atom, basis=basis)

            t_up, _, dft_up_res = time_func(lambda: dft_up.RKS(mol_up, xc=xc).run(), n_warmup=1, n_runs=2)
            t_rs, _, dft_rs_res = time_func(lambda: native.dft.RKS(mol_rs, xc=xc).run(), n_warmup=1, n_runs=2)

            diff = abs(dft_up_res.e_tot - dft_rs_res.e_tot)
            print(f"{xc.upper():<10} | {sys_name:<14} | {t_up * 1000:<14.2f} | {t_rs * 1000:<14.2f} | {t_up / t_rs:<10.2f} | {diff:<12.2e}")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print(" PYSCF VS PYSCF-RS BENCHMARK SUITE")
    print(" Backend: AMD ROCm GPU / CPU")
    print("#" * 80)

    run_scf_benchmarks()
    run_post_hf_benchmarks()
    run_dft_benchmarks()
    run_gto_benchmarks()
    print("\n" + "=" * 80)
    print(" Benchmark completed successfully.")
    print("=" * 80)
