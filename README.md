<div align="left">
  <img src="https://github.com/pyscf/pyscf-doc/blob/master/logo/pyscf-logo.png" height="80px"/>
</div>

Python-based Simulations of Chemistry Framework
-----------------------------------------------
[![Build Status](https://github.com/pyscf/pyscf/workflows/CI/badge.svg)](https://github.com/pyscf/pyscf/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/pyscf/pyscf/branch/master/graph/badge.svg)](https://codecov.io/gh/pyscf/pyscf)

2026-01-27

* [Stable release 2.12.1](https://github.com/pyscf/pyscf/releases/tag/v2.12.1)
* [Changelog](../master/CHANGELOG)
* [Documentation](http://www.pyscf.org)
* [Installation](#installation)
* [Features](../master/FEATURES)
* [News](https://pyscf.org/news.html): **3rd PySCF Developers Meeting!**


# Installation

* Install stable release:

        pip install pyscf

* New features developed in recent years are available in the pyscf-forge package:

        pip install pyscf-forge

* Certain modules are maintained as extensions of PySCF, such as dispersion,
  dmrgscf, fciqmc, icmpspt, properties, semiempirical, shciscf ... (more on
  https://github.com/pyscf) can be installed using pip:

        pip install pyscf[all]

  An individual extension can be installed:

        pip install pyscf[dispersion]

* More details of custom installation can be found in
  [installation manual](http://pyscf.org/user/install.html#build-from-source)


# Citing PySCF

## Base PySCF
The following paper should be cited in publications utilizing the PySCF program package:

[Recent developments in the PySCF program package](https://doi.org/10.1063/5.0006074),
Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan,
*J. Chem. Phys.*, **153**, 024109 (2020). doi:[10.1063/5.0006074](https://doi.org/10.1063/5.0006074)

## Density functional calculations

As PySCF does not implement density functionals, instead employing external libraries to handle their evaluation, these libraries should also be cited in publications employing PySCF for density functional calculations.

If your calculation employed Libxc, cite

[Recent developments in libxc — A comprehensive library of functionals for density functional theory](https://doi.org/10.1016/j.softx.2017.11.002),
Susi Lehtola, Conrad Steigemann, Micael J.T. Oliveira, and Miguel A.L. Marques,
*SoftwareX* **7**, 1 (2018). doi:[10.1016/j.softx.2017.11.002](https://doi.org/10.1016/j.softx.2017.11.002)

If your calculation employed XCFun, cite

[Arbitrary-order density functional response theory from automatic differentiation](https://doi.org/10.1021/ct100117s),
Ulf Ekström, Lucas Visscher, Radovan Bast, Andreas J. Thorvaldsen, and Kenneth Ruud,
*J. Chem. Theory Comput.* **6**, 1971 (2010). doi:[10.1021/ct100117s](https://doi.org/10.1021/ct100117s)

# Bug reports and feature requests

Please submit tickets on the [issues](https://github.com/pyscf/pyscf/issues) page.



---

## pyscf-rs (Rust port)

> **Status:** Phase 1 (Foundation) — workspace skeleton + algebra surface
> shipped. Methods (HF, DFT, MP2, CCSD, gradients) land in Phases 2-7.
> See `.planning/ROADMAP.md` for the full phase plan.

pyscf-rs is the pure-Rust port of PySCF, designed to be a drop-in
replacement for `import pyscf` with bit-exact agreement on regression
tests and 2–5× speedup vs PySCF + C extensions. Built on
[cubecl](https://github.com/tracel-ai/cubecl) (single kernel source for
CPU SIMD / CUDA / WGPU / ROCm), [PyO3](https://pyo3.rs) for Python
bindings, and the sibling crates `cintx` (libcint replacement),
`libxc_rs`, `xcfun_rs`.

### Quickstart (developer)

```bash
# Clone alongside the upstream PySCF Python tree (already in this repo).
git clone https://github.com/BectorVoom/pyscf_rs
cd pyscf_rs

# CPU-only build (default — fast, no GPU drivers needed):
cargo build --workspace --locked

# GPU-enabled build (compiles cuda + wgpu probe arms; runtime requires drivers):
cargo build --workspace --locked --features gpu

# Run the test suite:
cargo test --workspace --locked -- --test-threads=1
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local sibling-crate
development recipe and the CI gate cheatsheet.

### Backend selection at runtime

Two environment variables drive the algebra layer:

| Env var          | Default | Allowed values                                       |
|------------------|---------|------------------------------------------------------|
| `PYSCF_BACKEND`  | `cpu`   | `cpu`, `cuda`, `wgpu`, `rocm`, `metal`, `auto`       |
| `PYSCF_DTYPE`    | `f64`   | `f32`, `f64`                                         |

The workspace `gpu` feature is **OFF by default** (CPU-only builds are
fast and reproducible). Enable a single backend via `--features cuda`
or the host-portable umbrella `--features gpu` (= `cuda + wgpu`).

`PYSCF_BACKEND=auto` walks the priority chain
`cuda → rocm → metal → wgpu → cpu`, selecting the first backend that
is both compiled in AND has a usable device. Unrecognised values fall
back to CPU with a `tracing::warn!`. Setting `PYSCF_BACKEND=wgpu` with
`PYSCF_DTYPE=f64` on an adapter without the `shader-f64` Vulkan
extension returns a hard error rather than silently downgrading.

Every PyO3 entry point emits a `tracing::info!` line on backend
resolution: `pyscf-algebra: backend=cpu (env=unset, dtype=f64)`.

### Workspace structure

| Crate                  | Phase | Role                                                              |
|------------------------|-------|-------------------------------------------------------------------|
| `pyscf-rs` (façade)    | 1     | Top-level re-exports for `cargo add pyscf-rs`                      |
| `pyscf-core`           | 1     | Universal types (Mole, Density, Energy) and method traits         |
| `pyscf-runtime`        | 1     | BackendKind, per-backend probes, WorkspacePool, tracing init      |
| `pyscf-algebra`        | 1     | Sole cubecl-* consumer; gemm/reduce/axpy/eigh public surface      |
| `pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt}` | 2-7 | Method crates (under construction)                |
| `pyscf-py`             | 3     | PyO3 abi3-py310 wheel (`pip install pyscf-rs`)                    |
| `pyscf-oracle`         | 3     | PySCF live oracle (dev-deps only; release wheels never link Python) |
| `pyscf-bench`          | 8     | Criterion benchmark suite                                         |

### Cubecl pin

cubecl 0.10.0 is exact-pinned across pyscf-rs AND the three sibling
crates (cintx, libxc_rs, xcfun_rs). Bumping cubecl is a four-crate
operation; see [docs/upgrade-cubecl.md](docs/upgrade-cubecl.md) for
the documented ritual.
