# Pitfalls Research

**Domain:** Quantum-chemistry library (PySCF rewrite) in Rust + cubecl + PyO3
**Researched:** 2026-05-09
**Confidence:** HIGH for items verified against PyO3/cubecl/PySCF issue trackers and official docs; MEDIUM where extrapolated from sibling-crate (cintx, xcfun_rs) lessons; LOW flagged inline where only training-data folklore.

> Severity legend: **SHOWSTOPPER** = will block v1 release / break drop-in promise. **MAJOR** = will cost weeks if hit late. **MINOR** = annoying but bounded.
>
> Phases referenced (canonical roadmap names): `gto`, `scf`, `dft`, `mp2`, `ccsd`, `grad`, `geomopt`, `bindings`, `oracle`, `distribution`. A meta-phase `infra` covers cross-cutting (CI, math abstractions, error types) before `gto`.

---

## Critical Pitfalls

### Pitfall 1: Silent FMA contraction breaks bit-exact agreement with PySCF

**Severity:** SHOWSTOPPER (the bit-exact contract is the "Core Value")

**What goes wrong:**
LLVM, given `a*b + c*d`, may emit one fused-multiply-add (`fmuladd` / `vfmadd231sd`) on AVX2/AVX-512/aarch64 targets. PySCF's C extensions, compiled by GCC at `-O2` (PySCF default `setup.py` flags), typically do *not* contract because GCC needs `-ffp-contract=fast` or `__builtin_fma`. Rust + LLVM under `cargo build --release` may emit FMA aggressively when the target features include `+fma`. Same source, different rounding, ~1 ulp drift per multiply-add — compounds across an SCF cycle into mHartree-scale divergence after DIIS amplification.

**Why it happens:**
LLVM IR `fadd` and `fmul` with `contract` fast-math flag (default off) become `llvm.fmuladd.f64`. The Rust compiler sets `contract` for explicit `f64::mul_add` calls only — but autovectorizers and target-cpu=native still produce `vfmadd*` pairs from separate operations on some chip families. Sources confirm "compiler might optimize a float formula to use an FMA instruction on one machine but not on another."

**How to avoid:**
1. **Lock `RUSTFLAGS` for the numerical crates:** `-C target-feature=-fma` for the oracle-comparison build. Add a `[profile.release-oracle]` profile in workspace `Cargo.toml`. CI uses this profile when running parity tests.
2. Provide an `#[inline(always)] fn fma_or_mulasd(a, b, c)` abstraction. The cintx-style discipline of "one named operation per arithmetic step" applies — it makes contraction grep-able.
3. Use `core::hint::black_box` between mul and add in critical reductions to prevent the optimizer from fusing.
4. Document: the CPU SIMD backend is the *bit-exact* backend. CUDA/HIP/WGPU are *chemical-accuracy* backends (because their FMA is non-negotiable hardware behavior).

**Warning signs (CI assertion):**
- Parity test asserts `assert_eq!(rust_energy.to_bits(), pyscf_energy.to_bits())` for a small-molecule fixture (e.g., H2O / STO-3G HF). Drift on a single architecture upgrade = FMA contraction got introduced.
- Build a `cargo-llvm-ir` xtask that greps for `llvm.fmuladd` in oracle-mode object files; CI fails if any appear in arithmetic-critical crates.

**Phase to address:** `infra` (set the profile + abstraction) before any numerical kernel lands. Re-verify in `oracle` phase.

**Reference:** [Rust users forum: mul_add accuracy](https://users.rust-lang.org/t/why-does-the-mul-add-method-produce-a-more-accurate-result-with-better-performance/1626), [packed_simd FMA notes](https://rust-lang.github.io/packed_simd/perf-guide/float-math/fma.html), [KDAB FMA woes](https://www.kdab.com/fma-woes/).

---

### Pitfall 2: Parallel-reduction order non-determinism

**Severity:** SHOWSTOPPER for bit-exact, MAJOR for chemical-accuracy

**What goes wrong:**
Most QC hot loops are reductions: density-matrix builds, Fock matrix accumulation, `(ij|kl) * P_kl` contractions, DFT grid integrations. Floating-point addition is non-associative. `rayon::par_iter().sum()` distributes work across threads, and the merge tree depends on `RAYON_NUM_THREADS`, work-stealing schedule, and CPU topology. Two CI runs on the same machine produce different last bits.

**Why it happens:**
Multi-threaded sum is the canonical non-deterministic reduction. The Rust ecosystem inherits this from anywhere using rayon's reduce / fold. `cubecl` reductions, when not configured with deterministic mode, are likewise non-deterministic — multiple GPU threads atomically add into a shared accumulator (or shuffle-reduce in unspecified pair order).

**How to avoid:**
1. **Deterministic reduction order in `oracle` profile.** Define a single canonical reduction tree: chunk by 256 elements, sum each chunk in index order, then sum chunks in index order. cubecl supports comptime constants; bake the chunk size in. For CPU SIMD, accumulate per-lane then horizontal-add in fixed lane order (SSE/AVX `hadd` is order-defined).
2. **Pin OpenMP threads in oracle CI.** Both Rust side (`RAYON_NUM_THREADS=1` for parity tests) *and* PySCF side (`mol.lib.num_threads(1)` per the PySCF documented API — env var alone is insufficient because `OMP_NUM_THREADS` is read at module import).
3. Provide a `kahan_sum` for the high-bar fixtures. Kahan + ordered chunk reduction yields the same bits across thread counts at ~6× cost; acceptable for fixtures but not hot path.
4. Document deviation: GPU backends are not bit-deterministic by default; energy agreement is bounded to 1e-10 hartree, gradients to 1e-8.

**Warning signs:**
- Run the same parity test twice with `RAYON_NUM_THREADS=1` and `=8`; assert identical bits. Drift = a non-deterministic reduction snuck in.
- Fingerprint test (à la `pyscf.lib.finger`) on every accumulator output; flake = order issue.

**Phase to address:** `infra` (define `oracle_sum`, `oracle_dot`, `oracle_einsum` primitives), enforced from `gto` onwards.

**Reference:** [PySCF lib.num_threads issue](https://github.com/pyscf/pyscf/issues/1102), [PySCF threading issue 1138](https://github.com/pyscf/pyscf/issues/1138), [Gaffer on Games: Floating Point Determinism](https://gafferongames.com/post/floating_point_determinism/).

---

### Pitfall 3: cubecl pre-1.0 churn — pinning, breaking changes, f64 holes

**Severity:** MAJOR (a single cubecl minor bump can break four sibling crates simultaneously)

**What goes wrong:**
1. **Version churn**: cubecl is at v0.10.0-pre.4 as of April 2025; cintx is on `cubecl = "0.10.0"`, xcfun on workspace cubecl. Pre-1.0 minor bumps are routinely breaking — v0.9 → v0.10 reorganized memory pools, channel APIs, and reduction implementations.
2. **f64 backend holes**: There are *open* cubecl issues confirming f64 problems on the SPIR-V (WGPU) backend specifically: `f64 ln/exp emit invalid GLSL.std.450 OpExtInst` (#1316) and `f64 to i64 cast` issues (#1317, closed). WebGPU itself does not yet require f64 in the spec — see [gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805). QC needs f64 *everywhere*; on WGPU/Vulkan it may be unavailable on consumer GPUs, requiring an `f64-emulation` software path or backend gating.
3. **Default backend changed silently**: cintx CHANGELOG records that `Builder::default()` is changing from `Wgpu` → `Cpu` between releases. A pyscf_rs unit test that used the default backend would silently flip target.

**Why it happens:**
Pre-1.0 software has no semver guarantee. The "all four backends in v1" decision in PROJECT.md inherits the ceiling of cubecl's least-stable backend (currently WGPU for f64).

**How to avoid:**
1. **Pin cubecl to an exact version** (`= "0.10.0-pre.4"` not `"0.10"`) in `Cargo.toml`, including `cubecl-cuda`, `cubecl-hip`, `cubecl-wgpu`, `cubecl-cpu`. Match the sibling crates' pin exactly — diverging puts pyscf_rs and cintx in cubecl-version conflict.
2. **Maintain a compatibility matrix** as `docs/CUBECL_VERSION.md`: what cubecl version, what backends are tested, what f64 limitations apply per backend.
3. **Feature-gate WGPU f64**: `wgpu` feature requires the `shader-f64` Vulkan extension; otherwise the WGPU backend is a build-time error with a clear message, not silently degraded f32.
4. **Test the f64-restricted code paths early** — write a Phase-`infra` smoke test that does `cubecl::launch!(elementwise add f64 array)` on every backend; fail fast, raise a tracking issue against cubecl, gate the backend behind `cfg(feature = ...)` until upstream fixes it.
5. **Subscribe to the cubecl issue feed** for the four backend crates; encode this in a `CONTRIBUTING.md` upgrade checklist.

**Warning signs:**
- cintx releases a new version pinning a different cubecl — pyscf_rs CI fails because its workspace can't pick both. Treat this as a SHOWSTOPPER for the affected sprint.
- A WGPU CI job emits `OpExtInst` link-time error on a kernel that was working last week — backend f64 regression.

**Phase to address:** `infra` (pinning + matrix doc + smoke test), enforced in every `gto`/`scf`/`dft` kernel landing.

**Reference:** [tracel-ai/cubecl issues](https://github.com/tracel-ai/cubecl/issues) (#1316, #1317, #1318), [cintx CHANGELOG](file:///home/user/Documents/workspace/cintx/CHANGELOG.md), [WebGPU f64 tracking issue](https://github.com/gpuweb/gpuweb/issues/2805).

---

### Pitfall 4: Eigenvector sign / degenerate-subspace ordering ambiguity

**Severity:** SHOWSTOPPER for bit-exact MO coefficients; MINOR for energies

**What goes wrong:**
Eigenvectors are defined up to ±1 (real symmetric eigenproblems) or up to a unitary rotation in degenerate subspaces. PySCF's documented behavior: for symmetric systems, "multiple AO coefficients have the same magnitude but opposite sign, and which one's idx gets selected by argmax is non-deterministic." Any downstream test that compares MO coefficient matrices directly (chkfile re-load, MP2 integral transformation, CC initial amplitudes) will fail flakily even when energies match.

**Why it happens:**
LAPACK `dsyev` / `dsyevd` make no sign promise. PySCF tries to enforce a sign convention by argmax-coefficient sign, but argmax with ties (degenerate orbitals, symmetric molecules like benzene D6h) is implementation-defined. Different LAPACK vendors (OpenBLAS vs MKL vs Apple Accelerate) produce different signs even on the *same* machine.

**How to avoid:**
1. **Adopt PySCF's sign convention explicitly** and apply it deterministically: after diagonalization, for each MO, find the index of largest absolute coefficient with the lowest tie-broken-by-index entry, and flip the column to make that coefficient positive. Implement once in `pyscf_rs::lib::canonicalize_signs`.
2. **For degenerate subspaces**, do not rely on diagonalization output. Project onto symmetry-adapted basis (deferred — pyscf.symm is out of v1 scope) OR adopt the convention that within a degenerate subspace (eigenvalue tolerance 1e-8 Hartree), apply a deterministic Gram–Schmidt with a fixed pivot order.
3. **Tests compare invariants** (density matrix `D = C C^T`, energies, dipole moments) by default; only compare MO coefficient matrices via `|<C_rust|C_pyscf>|^2 == 1` (overlap test), never element-wise.

**Warning signs:**
- A test passes locally but fails on CI macOS-arm64 (Accelerate vs OpenBLAS). Sign drift, not numerical drift.
- CCSD initial amplitude `t1, t2` magnitudes match but signs flip — eigenvector phase propagating.

**Phase to address:** `scf` for the canonicalization helper. `oracle` for the comparison protocol. `mp2`/`ccsd` rely on the helper.

**Reference:** [PySCF issue #1196](https://github.com/pyscf/pyscf/issues/1196), [PySCF SCF docs](https://pyscf.org/user/scf.html).

---

### Pitfall 5: NumPy zero-copy buffer hazards across the PyO3 boundary

**Severity:** MAJOR (silent wrong answers if non-contiguous, segfaults if lifetime mismanaged)

**What goes wrong:**
PySCF users heavily pass numpy arrays to functions and read back numpy results. Naive `PyArray::as_slice()` assumes C-contiguous; if the user passed a view (`a[::2]`, `a.T`, `a[:, 1:5]`), the slice is *non*-contiguous and either Rust panics (`as_slice` returns `Err`) or — if you used `as_slice_unchecked` — silently reads wrong data. Documented rust-numpy issue #114 (slicing + `to_pyarray` produces incorrect results) is precisely this trap.

PySCF's Fortran heritage: many internal arrays are F-contiguous (`order='F'`), e.g., MO coefficient matrices. Rust's `ndarray::Array2` defaults to row-major. Code that does `Array2::from_shape_vec((n, m), pyarray.to_vec())` reading from an F-order numpy array transposes silently.

**Why it happens:**
- Contiguity flags can be wrong even in numpy itself ([numpy#14627](https://github.com/numpy/numpy/issues/14627) — incorrect contiguous flag on non-contiguous views).
- PyO3 lifetime: a `&PyArray2<f64>` borrows from Python; if the user runs GC between Rust spawning a thread and the thread reading, the buffer is freed — segfault.

**How to avoid:**
1. **Strict input normalization at the FFI boundary.** Every public PyO3 function on a numpy input does:
   ```rust
   let arr = arr.readonly();              // Bound<PyReadonlyArrayN>
   let arr = arr.as_array();              // ArrayView with strides
   if !arr.is_standard_layout() { let arr = arr.to_owned(); /* defensive copy */ }
   ```
   Never `as_slice_unchecked` on caller-supplied input. Document the copy as the contract; users wanting zero-copy must pre-flatten.
2. **Fortran-order awareness:** when PySCF compatibility requires F-order returns (MO coefficients), produce them via `Array2::from_shape_vec(shape.f(), data)` and return as `PyArray2::from_owned_array`. Add a doctest pinning the layout.
3. **No threads outside the GIL with borrowed arrays.** If you `Python::detach` to do CPU work, you must own the array (`to_owned` on entry).
4. **Run `abi3audit` in CI** to catch ABI surface mistakes that cause crashes on Python upgrades.

**Warning signs:**
- Test passes when called as `f(np.eye(N))` but fails as `f(np.eye(N).T)` — non-contiguity bug.
- Random-seeded fuzz test that passes views of various stride patterns; any divergence vs the same input passed `.copy()` indicates a bug.

**Phase to address:** `bindings` phase establishes the FFI boundary discipline; every subsequent module reuses the helpers.

**Reference:** [rust-numpy#114](https://github.com/PyO3/rust-numpy/issues/114), [numpy#14627](https://github.com/numpy/numpy/issues/14627), [Trail of Bits abi3audit](https://blog.trailofbits.com/2022/11/15/python-wheels-abi-abi3audit/).

---

### Pitfall 6: GIL deadlock when Rust calls back into Python (or releases GIL holding state)

**Severity:** MAJOR (intermittent hangs in production)

**What goes wrong:**
PySCF users heavily subclass `gto.Mole` and `scf.RHF`, overriding methods (`get_jk`, `get_veff`). The Rust SCF kernel will need to invoke these Python overrides — which means re-acquiring the GIL inside a `Python::detach` block. PyO3 deadlock patterns documented:
- Inside `tokio::spawn` multi-threaded, calling `Python::with_gil` deadlocks (PyO3 discussion #3045).
- `OnceLock` initialization inside `with_gil` that internally releases the GIL via `Python::import` deadlocks (PyO3 #4738).
- Free-threaded Python 3.13 introduces a *new* class of deadlocks (PyO3 #4738) — the GIL is gone but `pyo3::sync::GILOnceCell` semantics shift.

**Why it happens:**
The GIL is reentrant on the same thread but blocks across threads. Mixed threading models (rayon Rust threads + Python's threading module + numpy's OpenMP threads via MKL) overlap their lock requirements unpredictably.

**How to avoid:**
1. **Single-threaded callback model:** Python overrides are only called on the thread that holds the GIL (the "main" PyO3 entry point). All Rust-side parallelism happens *between* Python callbacks, never spawning threads that themselves call back to Python.
2. **Drop the GIL only at clearly marked seam functions** (`Python::detach` wrapping a single fn that does pure-Rust work). Document each such site.
3. **Use `pyo3::sync::OnceLock` and `OnceExt` traits**, not `std::sync::OnceLock` + `with_gil`, for any cross-call cached Python objects.
4. **Test under Python 3.13 free-threaded build (`python3.13t`) in CI** — catches deadlocks the regular GIL hides.

**Warning signs:**
- A test hangs on CI but passes locally — almost always a GIL ordering issue.
- `py-spy dump` of the hung test shows two threads in `take_gil` / `pthread_cond_wait`.

**Phase to address:** `bindings` (establish the callback model and the documented release sites). Re-validated in every method-class binding (HF, KS, MP2, CCSD).

**Reference:** [PyO3 discussion #3045](https://github.com/PyO3/pyo3/discussions/3045), [PyO3 discussion #4738](https://github.com/PyO3/pyo3/discussions/4738), [PyO3 FAQ](https://pyo3.rs/v0.23.4/faq.html), [Free-Threading Porting Guide](https://py-free-threading.github.io/porting-extensions/).

---

### Pitfall 7: PyO3 subclassing — Python-side subclass of Rust class breaks polymorphism

**Severity:** SHOWSTOPPER for drop-in promise (PySCF users heavily subclass `Mole`/`RHF`)

**What goes wrong:**
A core upstream-PySCF idiom: `class MyHF(scf.RHF): def get_veff(self, dm): return super().get_veff(dm) + correction(dm)`. If `pyscf.scf.RHF` is a PyO3 `#[pyclass]` and its Rust kernel does `self.get_veff(dm)`, Rust dispatches via Rust method resolution — *bypasses* the Python override. The user's correction term silently disappears.

**Why it happens:**
"`self.greeting()` still follows Rust method resolution, and Rust isn't aware of the class hierarchy defined in pyo3 annotations" (PyO3 discussion #4164). Polymorphism via Python's MRO requires explicit `call_method0(py, "greeting")`.

**How to avoid:**
1. **Mark every overridable method with `subclass`** and dispatch via `slf.call_method1(py, "get_veff", (dm,))` whenever the SCF driver calls a method a user might override. Maintain a documented list of overridable hooks.
2. **Audit upstream PySCF for the override surface.** Run `grep -rn "def get_jk\|def get_veff\|def get_hcore\|def get_init_guess" pyscf/` to enumerate what users monkey-patch. Each becomes a `call_method1` site.
3. **Provide a Python-side compatibility shim:** keep a thin `pyscf/scf/hf.py` in the wheel that subclasses the Rust class and re-exposes the override hooks as Python methods that delegate to Rust. This lets users keep `class MyHF(scf.RHF)` working without learning new APIs.
4. **Test with PySCF-in-the-wild scripts** — pull representative user notebooks (NCI, dispersion-corrected DFT scripts) through CI; they exercise subclass paths the unit tests miss.

**Warning signs:**
- A user reports "my subclass override is being ignored." Detected late = furious users.
- CI test where Rust class is subclassed in Python and an override is asserted to be called — if not asserted, you don't know.

**Phase to address:** `bindings`. The override-dispatch contract must be in place from the first `pyclass`. Each method module (`scf`, `dft`, `mp2`, `ccsd`) revalidates.

**Reference:** [PyO3 discussion #4164](https://github.com/PyO3/pyo3/discussions/4164), [PyO3 issue #947 (subclass new returns wrong type)](https://github.com/PyO3/pyo3/issues/947).

---

### Pitfall 8: Loop-ordering / memory-layout mismatch between Rust default and PySCF/Fortran

**Severity:** MAJOR (10–100× perf hit; may not be detected until benchmarking)

**What goes wrong:**
1. PySCF stores MO coefficients, density matrices, and ERI tensors in F-order (column-major) because libcint, BLAS, and the Fortran heritage all assume it. Rust's `ndarray::Array2` defaults to C-order. Naive translation: `for i in 0..n { for j in 0..m { c[i, j] }}` is row-major iteration on a column-major array → cache miss every element → 50× slow.
2. **8-fold ERI symmetry** (`(ij|kl)`): PySCF stores the unique `npair*(npair+1)/2` quartets where `npair = nao*(nao+1)/2`. Naive Rust loop over `i,j,k,l` independently inflates work 8×. `pyscf.ao2mo` makes this explicit; a Rust port that ignores the indexing convention quadruples memory and CPU.

**Why it happens:**
Rust idioms (slices, iterators, `Array2::from_shape_vec((n,m), data)`) default row-major; programmers porting Fortran-style code don't realize that the `s8` packing in PySCF represents a specific layout.

**How to avoid:**
1. **Layout convention is part of the type signature.** Define `type ERI8 = ArrayD<f64> /* layout: s8 packed, npair*(npair+1)/2 */;` aliases. A function signature taking `ERI8` is a contract; mismatched layout is a compile error if you push it into a newtype.
2. **All BLAS-bound matrices use F-order by default** in this crate (matches libcint, matches PySCF). Document explicitly. `Array2::default()` is banned in favor of `Array2::default(shape.f())`.
3. **Loop-order audit checklist** for every kernel: which axis is fastest-varying? Match it to memory layout. Add a `cargo bench` with `--bench` smoke that flags >2× regression vs reference.
4. **ERI access must go through a `Eri8 { data, npair, ... }` view type** with `.get(i, j, k, l)` that handles the unique-quartet index math once.

**Warning signs:**
- Benchmark on H2O/cc-pvdz HF is >5× slower than PySCF — either loop order or ERI symmetry not being exploited.
- `perf stat` shows >10% LLC miss rate on the Fock-build kernel.

**Phase to address:** `gto` (define ERI types). `scf` (Fock build is the canonical perf-critical loop). `mp2`/`ccsd` exploit symmetry of MO-basis amplitudes.

**Reference:** [PySCF ao2mo docs](https://pyscf.org/pyscf_api_docs/pyscf.ao2mo.html), [Permutational symmetries notes (Sherrill)](http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf).

---

### Pitfall 9: DIIS divergence-path drift after the first divergent step

**Severity:** MAJOR (tests pass on H2O, fail on a transition-metal complex)

**What goes wrong:**
DIIS extrapolates Fock matrices from a history. When SCF nearly diverges (open-shell, small HOMO-LUMO gap, transition metals, dissociation curves), tiny numerical differences in the residual at iteration `k` choose a different DIIS coefficient at iteration `k+1`, which produces a measurably different Fock at iteration `k+2`, and the convergence path bifurcates — eventually converging to the *same* energy (chemical accuracy preserved) but on a different number of iterations and with a different last-step density. Bit-exact comparison of `mf.mo_coeff` after `kernel()` fails, even though `mf.e_tot` matches PySCF to 1e-12.

**Why it happens:**
DIIS is fundamentally chaotic in the divergent regime: the linear-extrapolation system for coefficients has a tiny condition number, and PySCF's `diis.subspace` history maintenance chooses to drop the oldest vs. the most-residual entry based on a settings-dependent heuristic. Q-Chem's manual notes "the minimization of the orbital rotation gradient does not always lead to a lower energy."

**How to avoid:**
1. **Mirror PySCF's DIIS implementation byte-for-byte at the algorithm level**: same vector-storage convention, same residual definition (B(i,j) = <r_i | r_j>), same tie-breaking rule for which history vector to evict. Read `pyscf/scf/diis.py` and port logically — not a paper port.
2. **Bit-exact contract is on energy and density, not on convergence path.** Document this. Test fixtures use molecules that converge in <15 iterations from `init_guess='minao'` (well-behaved cases). Hard cases (Cr2, Fe(CO)5) test chemical accuracy only.
3. **Capture PySCF's DIIS state per iteration as a fixture**: if the user reports drift on a hard molecule, replay against a pickled DIIS state. cintx's oracle pattern (Cargo feature `cpu`/`rocm` with fixtures) generalizes here.

**Warning signs:**
- Identical energy (to 1e-12) but different MO coefficients between Rust and PySCF, on iteration count >10 — DIIS path drift, not eigenvector sign (sign would flip a column, not all of them).
- A second-row transition-metal complex test that bisects on iteration count.

**Phase to address:** `scf` (DIIS impl), `oracle` (define test fixtures: easy-converger only for bit-exact, hard cases for chemical accuracy).

**Reference:** [DIIS Wikipedia](https://en.wikipedia.org/wiki/DIIS), [DIIS comments (Sherrill)](https://vergil.chemistry.gatech.edu/static/content/diis.pdf), [Q-Chem manual sect 4.5](https://manual.q-chem.com/5.1/sect-convergence.html).

---

### Pitfall 10: DFT grid weighting (Becke / Stratmann / Treutler) — wrong-by-default

**Severity:** MAJOR (every DFT energy off by ~1e-5 hartree if mismatched)

**What goes wrong:**
PySCF's `gen_grid.py` defaults to `treutler` radial + Becke partition with NWChem-style atomic radii. A Rust port that picks the "obvious" Becke (J. Chem. Phys. 88:2547) without the *exact* atomic-radii table, the *exact* prune scheme, and the *exact* ζ-parameter values produces different grid weights → different XC energies → ~1e-5 hartree drift everywhere DFT shows up.

**Why it happens:**
The Becke paper specifies the algorithm but not the parameters. Each QC code picks an atomic-radii convention. PySCF specifically uses NWChem-derived parameters in `pyscf/dft/gen_grid.py`. Documentation in the literature is vague; the only ground truth is the source.

**How to avoid:**
1. **Port `pyscf/dft/gen_grid.py` byte-for-byte at the grid-generation level.** Atomic radii, Lebedev quadrature points, prune scheme — all from the upstream source, with file references in the Rust comments.
2. **Oracle test on grid weights, not just energies:** for a fixed molecule + grid level, assert Rust grid (positions, weights) matches PySCF's `mol.gen_grid()` output bit-exact (after sort canonicalization).
3. **Expose `grids.scheme = 'becke' | 'stratmann'`** as a kwarg the way PySCF does, not as a build-time choice. Default = PySCF default (`treutler`, `Becke partition`, `prune=nwchem_prune`).

**Warning signs:**
- Every RKS energy off by ~1e-5 hartree from PySCF — grid weights are wrong.
- Grid atom-count vs PySCF differs at any prune level.

**Phase to address:** `dft` — first-thing-after-grid-stub.

**Reference:** [PySCF gen_grid example](https://github.com/pyscf/pyscf/blob/master/examples/dft/11-grid_scheme.py), [NASA TR on weight scheme effect](https://ntrs.nasa.gov/citations/20040084670).

---

### Pitfall 11: chkfile (HDF5) byte-for-byte compatibility with upstream PySCF

**Severity:** SHOWSTOPPER for drop-in promise (users restart calculations from chkfiles)

**What goes wrong:**
PySCF chkfiles are `h5py`-written HDF5 with conventions: paths like `/scf/mo_coeff`, `/scf/e_tot`, dataset attributes for symmetry irreps, F-order arrays for MO coefficients (because numpy was instructed), pickle blobs for `mol` reconstruction. A user does `mf = scf.RHF(mol).run(); mf.dump_chk()` then later `mf2 = scf.RHF(mol).update_from_chk('foo.chk')`. If pyscf-rs writes the chkfile, both write *and* read paths must match upstream byte-for-byte.

**Why it happens:**
HDF5 is flexible. The `mol` object includes pickled basis-set data (PySCF's basis-set dictionaries are Python dicts, pickled into HDF5 attrs). Pickle is Python-version-sensitive; `protocol=2` is forward-compatible but PySCF has used different protocols across versions. h5py library version determines the HDF5 dataset chunking.

**How to avoid:**
1. **Use `hdf5-rs` (the native-lib binding) not `hdf5-metno`** — chkfile compatibility requires real HDF5 library, not a Rust-only reimplementation.
2. **Mol pickling: don't.** Re-derive the chkfile schema so the pickled-Python pieces are reduced to JSON-able dicts (already half-true in PySCF — `mol.dumps()` returns JSON). Where pickle is unavoidable (basis-set dict with custom GTO objects), call into Python via PyO3 to do the pickling — round-trip through CPython's pickle, never reimplement in Rust.
3. **Round-trip oracle test:** PySCF writes a chkfile, Rust reads it and asserts every field; Rust writes a chkfile, PySCF reads it and runs a downstream calc to verify.
4. **Pin h5py version range** in the test environment (`h5py>=3.8,<4`); document the supported chkfile producer versions.

**Warning signs:**
- `h5dump foo.chk` differs structurally between Rust-written and PySCF-written for the same input.
- `mf.update_from_chk` raises `KeyError: 'scf/mo_coeff'` — schema mismatch.

**Phase to address:** `scf` (RHF chkfile). Each subsequent method (DFT, MP2, CCSD) revalidates its own chkfile schema.

**Reference:** [pyscf/lib/chkfile.py](https://github.com/pyscf/pyscf/blob/master/pyscf/lib/chkfile.py), [pyscf/scf/chkfile.py](https://github.com/pyscf/pyscf/blob/master/pyscf/scf/chkfile.py), [INTEGRATIONS.md upstream](file:///home/user/Documents/workspace/pyscf_rs/.planning/codebase/INTEGRATIONS.md).

---

### Pitfall 12: Cross-platform numerical drift (x86_64 vs aarch64)

**Severity:** MAJOR (CI green on x86 / red on M-series macOS)

**What goes wrong:**
`sin`, `cos`, `exp`, `erf`, `erfc` (Boys function leans on `erf`) — the platform `libm` implementation is *not* IEEE-correctly-rounded. glibc's libm and Apple's libm differ in the last ulp on transcendentals. Rust's `f64::sin` defers to platform libm. Cross-platform parity tests will flake on transcendental-heavy paths (DFT XC functionals, ECP integrals, Boys function evaluation).

**Why it happens:**
IEEE 754 doesn't mandate correctly-rounded transcendentals. ARM's official microsite acknowledges floating-point behavior differences. PySCF dodged this by being CPython, which uses the same libm everywhere — but the C extensions inherit platform variation; PySCF's CI matrix includes ubuntu/macos/aarch64 with non-bit-exact tests (they use `assertAlmostEqual` with 6 decimal places).

**How to avoid:**
1. **Use a portable correctly-rounded math library** for the Boys function and DFT functionals: depend on a Rust port of `libmcr` or use `libm::sin` (the `libm` crate, which is cross-platform-deterministic, separate from the platform libm). Pay the perf cost — they are 2–3× slower than native libm but are reproducible.
2. **Adopt the cintx oracle convention**: bit-exact tests gated to `x86_64-unknown-linux-gnu`; on other platforms, the tests run with looser tolerance (`atol=1e-12`). Documented in `oracle` README.
3. **Don't depend on libm's accuracy for chemistry-correctness** — wrap `f64::exp` in `pyscf_rs::math::exp` so future replacement is one place.

**Warning signs:**
- Test suite green on Linux x86, red on macOS-arm64 with `2 ulp` drift in MP2 energy.
- DFT energy drift exactly = ~1e-15 hartree per grid point — transcendental-evaluation drift integrated.

**Phase to address:** `infra` (math abstraction crate). `oracle` (test gating).

**Reference:** [Arm Learning: floating-point cross-platform](https://learn.arm.com/learning-paths/cross-platform/floating-point-rounding-errors/), [PySCF macOS-arm64 issue](https://github.com/pyscf/pyscf/issues/1015).

---

### Pitfall 13: PyPI wheel size limits and CUDA distribution

**Severity:** MAJOR (a built-fine wheel can't be uploaded)

**What goes wrong:**
PyPI default file size limit is 60 MB. PyTorch wheels are 776 MB (CUDA bundled). A pyscf-rs wheel that statically links cubecl-cuda's PTX cubins or bundles HIP runtime hits this immediately. Filing a PyPI exemption is documented to take weeks.

**Why it happens:**
`maturin --features cuda` will produce a wheel that includes the CUDA compute kernels. cubecl JIT-compiles, so the binary is *smaller* than torch's bundled cubin — but adding HIP + CUDA + WGPU together blows past 60 MB.

**How to avoid:**
1. **Split wheels by backend:** `pyscf_rs` (CPU SIMD only, default), `pyscf_rs[cuda]` (extra dep `pyscf_rs_cuda`), `pyscf_rs[rocm]`, `pyscf_rs[wgpu]`. Each wheel is one backend, kept small. Mirrors the PyTorch approach (`torch`, `torch+cuXXX`).
2. **Pre-emptively request PyPI size exemption** in the `distribution` phase, before the first GPU-bundled upload.
3. **Use manylinux_2_28** (current modern baseline; manylinux2014 = glibc 2.17 = 7 years old). Rust 1.64+ requires glibc 2.17 minimum.
4. **HDF5 link strategy**: `hdf5-rs` requires libhdf5 at runtime. Either bundle (wheel grows) or require user-installed (`pip install h5py` first). Document the choice.

**Warning signs:**
- `twine upload` fails with `400 File too large`.
- `auditwheel show` reports unbundled libs that won't exist on user machines.

**Phase to address:** `distribution`.

**Reference:** [Maturin distribution guide](https://www.maturin.rs/distribution.html), [Quansight 2021 packaging post](https://labs.quansight.org/blog/2021/01/python-packaging-brainstorm).

---

### Pitfall 14: Rust panic across FFI = undefined behavior

**Severity:** SHOWSTOPPER (UB is UB; intermittent crashes)

**What goes wrong:**
Rust panics in PyO3 functions are caught by PyO3's `extern "C-unwind"` shim on supported toolchains — but only if the Rust function is `#[pyfunction]`. Internal Rust functions called from PyO3 callbacks (e.g., a numint callback for DFT) may panic. If a numint callback Rust function is called from C (libcint? cubecl runtime?) and panics, it crosses an `extern "C"` boundary → UB.

**Why it happens:**
"All Rust stack frames which have FFI stack frames directly under them should be guarded by a `catch_unwind`" — Rustonomicon. cubecl runtime and any cintx C-ABI seam are exactly such boundaries.

**How to avoid:**
1. **Wrap every `extern "C"` callback** (cubecl host callbacks, FFI callbacks into cintx-capi) in `std::panic::catch_unwind`. On panic, log + return error code; never let unwinding cross the boundary.
2. **Use `extern "C-unwind"` ABI** for explicit Rust-to-Rust FFI seams where unwinding is allowed and tested.
3. **Replace `unwrap()` / `panic!()` in numerical code with typed errors.** A divergent SCF should return `Err(ScfError::DivergedAfter(n_iter))`, never panic. Code review checklist enforces.
4. **Build with `panic = "abort"` in production wheels** to make panic-induced UB into a clean abort; debug builds keep `panic = "unwind"` for tests.

**Warning signs:**
- Random segfaults on user machines that Valgrind traces to past an FFI boundary.
- `RUST_BACKTRACE=1` in the panicking pod shows the Rust stack but C frames are mangled — panic crossed FFI.

**Phase to address:** `infra` (panic policy + lint), enforced everywhere.

**Reference:** [Rust nomicon: FFI](https://doc.rust-lang.org/nomicon/ffi.html), [PyO3 issue #492](https://github.com/PyO3/pyo3/issues/492), [RFC 2945](https://rust-lang.github.io/rfcs/2945-c-unwind-abi.html).

---

### Pitfall 15: Sibling-crate ABI / cubecl-version drift between cintx, libxc_rs, xcfun_rs, pyscf_rs

**Severity:** SHOWSTOPPER if it lands; MAJOR while preventing

**What goes wrong:**
Four crates, all using cubecl. cintx pins `cubecl = "0.10.0"` (workspace `Cargo.toml`); xcfun_rs uses workspace cubecl with backend-specific feature flags; libxc_rs analogous. When pyscf_rs depends on all four, Cargo's resolver must find a single cubecl version. If cintx upgrades to `0.11` while xcfun_rs stays on `0.10`, pyscf_rs cannot build until both align.

cintx's CHANGELOG documents an in-flight breaking change (`Builder::default()` backend default). Any consumer that had `BackendIntent::default()` is silently broken across release.

**Why it happens:**
Pre-1.0 dep sprawl. The four-crate family is a single de facto product but ships as four crates without a synchronized release process.

**How to avoid:**
1. **Workspace-level cubecl pin in `pyscf_rs/Cargo.toml` `[workspace.dependencies]`**, with all four sibling crates using path overrides during dev (`[patch.crates-io]`) so they all resolve to the same cubecl. Document the upgrade ritual: bump cubecl in pyscf_rs, update siblings in lockstep.
2. **Single-developer asset**: solo dev means all four crates have the same maintainer. A pre-merge CI matrix builds pyscf_rs against the latest cintx/libxc_rs/xcfun_rs main branches every night to catch drift early.
3. **Semver-violation tracking**: cintx's pre-1.0 means every release is potentially breaking. pyscf_rs's `Cargo.toml` pins to `=cintx-version`, not `^cintx-version`. Upgrade is a deliberate, reviewed change.
4. **Functional-name compatibility:** libxc_rs / xcfun_rs must accept the same functional name strings PySCF accepts (`b3lyp`, `LDA,VWN`, `0.5*HF + 0.5*B88`, etc.) — write a parity table from `pyscf/dft/libxc.py` and gate on it.

**Warning signs:**
- `cargo update -p cubecl` cascades into a broken build.
- `pip install pyscf-rs` succeeds, but `import pyscf` fails because the bundled libxc functional name doesn't parse.

**Phase to address:** `infra` (workspace pin), `dft` (functional-name parity test), `oracle` (nightly cross-crate matrix).

**Reference:** [cintx CHANGELOG](file:///home/user/Documents/workspace/cintx/CHANGELOG.md), [PySCF libxc.py functional registry](https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py).

---

### Pitfall 16: Test-oracle global-state contamination

**Severity:** MAJOR (flaky tests; missing regressions)

**What goes wrong:**
PySCF's `KnownValues` test class pattern (TESTING.md): module-level globals (`global mol, mf`) populated in `setUpModule()`, mutated by tests, torn down in `tearDownModule()`. PySCF's SCF kernels mutate `mf` in-place (last density, DIIS history, chkfile path). When pyscf-rs runs PySCF as oracle, sequential tests share state — test A leaves `mf.mo_coeff` modified; test B asserts on it and gets stale data.

PySCF additionally has process-wide state: `pyscf.lib.num_threads()`, the global `__config__` module, `tempfile` / `PYSCF_TMPDIR` paths. Two tests setting different thread counts race.

**Why it happens:**
PySCF, like NumPy, is not designed for inter-test isolation; its assumption is one-script-one-process.

**How to avoid:**
1. **Subprocess-per-fixture** for oracle tests. Each Rust test that calls PySCF spawns a fresh `python -c "..."` (or persistent worker per test class). cintx-oracle's pattern of `bindgen` + C stubs in a separate compilation unit is a good model.
2. **Use `pyo3::prepare_freethreaded_python()` once** at process start, then `Python::with_gil` per test — but test isolation requires *clean* `pyscf` state. Reload module between tests via `importlib.reload(pyscf)` or, simpler, subprocess.
3. **Pin `mol.verbose = 0; mol.output = None`** in every fixture to suppress log file races.
4. **Treat test runtime budget seriously.** Pre-merge: HF/STO-3G water, RKS/B3LYP/cc-pvdz water, MP2/cc-pvdz water, all <5s. Nightly: CCSD/aug-cc-pvdz benzene, geometry optimization runs.

**Warning signs:**
- Test passes alone, fails in suite — order dependence, shared `mol` mutated.
- `pytest -p no:randomly` vs `-p randomly` give different pass rates.

**Phase to address:** `oracle` (the test harness pattern is the deliverable). Each method phase reuses the harness.

**Reference:** Upstream PySCF [TESTING.md](file:///home/user/Documents/workspace/pyscf_rs/.planning/codebase/TESTING.md), [cintx oracle Cargo.toml](file:///home/user/Documents/workspace/cintx/crates/cintx-oracle/Cargo.toml).

---

### Pitfall 17: Off-by-one and convention-drift in basis-set indexing

**Severity:** MAJOR (silent wrong integrals)

**What goes wrong:**
Three indexing systems coexist:
- Fortran libcint internal: 1-based, `bas[ATOM_OF + bas_id*BAS_SLOTS]` packed.
- libcint C API: 0-based atom and shell indices.
- Python PySCF `mol._bas`: 0-based numpy `(nbas, 8)` table.

A Rust port that ports Python's `_bas` access to libcint's C-API expectations may forget that `mol._atm` is 0-indexed in Python but the libcint atomic-coordinate offset in `env` is 1-indexed in some legacy code paths. Off-by-one yields integrals computed about a wrong nucleus.

ECP convention: PySCF's ECP module references several papers with subtly different sign and normalization conventions (J. Chem. Phys. 65:3826 vs J. Comput. Phys. 44:289). Mismatched ECP integrals are wrong-by-default and not caught by HF energy alone (the wrong sign cancels symmetrically) but break gradients.

Spherical vs Cartesian basis: libcint's "sp" normalization is unusual — Cartesian s,p are normalized but d,f,g are not. Forgetting the d-shell normalization correction during sph→cart transforms produces ~2× wrong integrals on d functions.

**Why it happens:**
Quantum chemistry has 30 years of conflicting conventions baked into legacy codes. PySCF made specific choices documented only in source.

**How to avoid:**
1. **Leverage cintx exhaustively.** cintx-rs is already the libcint replacement; pyscf-rs builds *on* cintx, doesn't re-implement integral indexing. Anywhere pyscf-rs touches `bas[]` directly, it's a bug.
2. **Round-trip oracle test on every basis set:** `mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis=name)` for `name in EVERY_BASIS_PYSCF_KNOWS` — assert pyscf-rs and PySCF compute the same `intor('int1e_kin')` to bit-exact. Catches sph/cart, normalization, indexing in one shot. Heavy but necessary; can be nightly.
3. **ECP**: do not implement ECP in pyscf-rs v1 if cintx doesn't yet support it. Defer with a clear `NotImplementedError` message until cintx supports the ECPscalar integral.

**Warning signs:**
- HF energy correct, dipole moment differs in last 4 digits — basis-function origin off by one.
- Energy correct, gradient wrong — likely ECP normalization or sph/cart d-shell.

**Phase to address:** `gto` (basis-set ingestion + sph/cart paths), validated by the per-basis oracle test.

**Reference:** [Sun 2015 libcint paper](https://arxiv.org/pdf/1412.0649), [libcint sph/cart issue #22](https://github.com/sunqm/libcint/issues/22), [pyscf/gto/ecp source](https://pyscf.org/_modules/pyscf/gto/ecp.html).

---

### Pitfall 18: Boys-function evaluation accuracy / speed tradeoff

**Severity:** MAJOR (a sloppy Boys eats the integral accuracy budget)

**What goes wrong:**
`F_n(T)` (Boys function) appears at the bottom of every ERI evaluation. It must be accurate to ~1e-13 across `T ∈ [0, ~30]` and `n ∈ [0, 16]` — implementations use a piecewise scheme: Taylor series at small `T`, downward-recursion + table lookup at moderate `T`, asymptotic `√(π/(4T))` at large `T`. PySCF/libcint use a specific table interpolation (Chebyshev-coefficient table, polynomial degree 6). Re-deriving "your own" Boys with `0.5*sqrt(pi/T)*erf(sqrt(T))` and recursion is correct in theory, but accumulates ~5 ulp of error per recursion step. After 6 steps, you've lost 5 digits.

**Why it happens:**
The naive textbook formula is numerically unstable downward. The numerically stable form requires a precomputed table, which is a 50 KB data blob hidden in a header. Re-implementing without the table is wrong; with the table, you've copied PySCF's data file and need to maintain it.

**How to avoid:**
1. **cintx-rs already does Boys.** Use it. pyscf-rs should never call a Boys function directly; all integrals go through cintx.
2. If a custom Boys is unavoidable (e.g., a specialized RI integral cintx doesn't expose), **adopt cintx's Chebyshev table verbatim**, with a build-time integrity check (`sha256` of the data blob).
3. **Test the Boys function in isolation** before any ERI test: `F_n(T)` for `n ∈ [0,16], T ∈ {0.001, 0.1, 1.0, 5.0, 25.0}` against PySCF's `pyscf.gto.mole.gto_norm` Boys path (or a high-precision reference like `mpmath.gammainc`).

**Warning signs:**
- ERI accuracy degrades for high-angular-momentum (g-functions) shells but is fine for s,p,d.
- A specific T range (`T ∈ [3, 8]`, the recursion-dominated regime) is where errors concentrate.

**Phase to address:** `gto` if anything, but ideally `infra` validates that cintx-rs is sufficient.

**Reference:** [Evaluation of the Boys Function (Mamedov 2004)](https://www.researchgate.net/publication/226566421_Evaluation_of_the_Boys_Function_using_Analytical_Relations), [Optimized Boys evaluation strategy](https://www.researchgate.net/publication/276295692).

---

### Pitfall 19: GPU kernel-launch overhead dominates for small molecules

**Severity:** MAJOR (perf regression vs CPU on small jobs)

**What goes wrong:**
A user runs `mf = scf.RHF(mol).run()` on H2O / cc-pvdz (24 basis functions). On GPU, the Fock build kernel launch overhead (~50 μs per launch) × 60 SCF iterations × 100s of small kernel calls per Fock build = orders of magnitude slower than CPU. User sees `pyscf-rs[cuda]` is *slower* than vanilla PySCF on their tutorial-sized system. Trust shattered.

**Why it happens:**
GPU kernels assume amortized launch cost over large work. cubecl autotune picks the largest cubin variant when the input is too small to fill the device, but the launch is still 50 μs.

**How to avoid:**
1. **Adaptive backend selection.** Heuristic: if `nao < 200` and `niter * nshell^4 < 1e10`, use CPU even if GPU is selected. A `BackendDispatch::auto(work_estimate)` helper that pyscf-rs uses, returning CPU for small jobs.
2. **Document explicit `backend = 'cpu' | 'cuda' | 'auto'` kwarg** on `mol.run()` (but not on `gto.M()` — we don't ship a gto.M() change). Default `'auto'`.
3. **Benchmark suite includes small molecules** (H2/sto-3g, H2O/sto-3g, H2O/cc-pvdz) and asserts GPU is no slower than 1.5× CPU.

**Warning signs:**
- User reports `pyscf-rs[cuda]` regressed performance on `examples/scf/00-simple_hf.py`.
- nsys trace: 90% of wall time in `cuLaunchKernel`.

**Phase to address:** `scf` (first kernel that hits this; sets the dispatch convention). `dft`, `mp2`, `ccsd` apply same heuristic.

**Reference:** [Hacker News on cubecl](https://news.ycombinator.com/item?id=43777731), CubeCL [autotuning docs](https://docs.rs/cubecl/0.9.0/).

---

### Pitfall 20: CCSD intermediate-tensor heap thrash and OOM at modest size

**Severity:** MAJOR (OOM at the molecule size users will routinely try)

**What goes wrong:**
CCSD has O(o²v⁴) memory in the unrelaxed-T2 form. The `Wabef` intermediate alone is `nv^4` = (basis-occupied)^4. For benzene (cc-pvdz: ~114 basis, 21 occupied, 93 virtual): `93^4 * 8 bytes = 600 MB`. For caffeine (cc-pvdz: ~190 basis, 41 occ, 149 virt): `149^4 * 8 = 4 GB`. Repeated allocation per iteration (every CCSD iteration freshens the intermediates) thrashes the allocator — 10+ s/iter purely in `malloc/free`.

Density-fitted CCSD reduces memory per iteration but the auxiliary-basis tensors (~3-index) are still 100s of MB and reallocated.

**Why it happens:**
Idiomatic Rust ownership encourages "create new tensor, return owned, drop old" — fine for small tensors, catastrophic for 4 GB blocks. Allocation latency scales with size.

**How to avoid:**
1. **Tensor arena / scratchpad pattern.** Allocate a maximally-sized scratch buffer once at CCSD start; intermediate tensors are views into the scratch. cubecl exposes memory pools; use them. Mirror PySCF's `lib.frompointer`/`lib.numpy_helper` patterns.
2. **Pre-flight memory check:** before CCSD starts, compute peak memory required, compare to available (`PYSCF_MAX_MEMORY` env var honored), error out cleanly if insufficient. Match PySCF's `mf.max_memory` semantics.
3. **Implement DF-CCSD before in-core CCSD** if the goal is "useful CCSD" — DF is the path to caffeine-sized molecules. Document in-core CCSD as a "reference / small-molecule" implementation.

**Warning signs:**
- `time` shows >50% in `malloc` for caffeine CCSD.
- OOM at 8 GB for a molecule PySCF runs in 6 GB — Rust allocation overhead.

**Phase to address:** `ccsd` (this is the first phase where memory becomes fundamental).

**Reference:** [Psi4 CCENERGY caching levels](https://psicode.org/psi4manual/master/autodir_options_c/module__ccenergy.html), [Reduced-cost CCSD(T)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912216/).

---

### Pitfall 21: Scope creep — TDDFT/CCSD(T)/MCSCF sneaking in via "while we're here"

**Severity:** MAJOR (pushes v1 release by months)

**What goes wrong:**
The roadmap says "v1 = HF/DFT/MP2/CCSD/grad/geomopt." Predictable creep paths:
- A user files an issue asking for CCSD(T); a maintainer adds the perturbative triples correction "because it's just a noniterative postprocessor" — but it pulls in storage of `(ai|bj)` integrals at full size, requires DF integration for non-toy molecules, then needs gradients...
- TDDFT looks like "linear response of the SCF Fock"; you justify the work as "we already have SCF, this is 10% extra." It's 10× the test surface (singlet/triplet, RPA vs TDA, N-state convergence, gauge issues).
- Symmetry adaptation — a single symmetry-aware unit test gets added "for completeness," and now `pyscf.symm` is on the dependency surface.

**Why it happens:**
QC modules are highly entangled. Each method has an obvious "next thing." Solo dev + Claude shipping at 0% capacity-headroom = no slack to absorb scope.

**How to avoid:**
1. **PROJECT.md `Out of Scope` section is binding.** Every PR that touches a file outside the in-scope module list (`pyscf/cc/` ✓, `pyscf/cc/eom_*` ✗) requires explicit milestone-promotion review. Document the lint: `forbidden-paths = ["pyscf/tdscf/", "pyscf/mcscf/", "pyscf/x2c/", "pyscf/pbc/"]`.
2. **A "scope-trap" review checklist** at every phase boundary (`gsd-transition`):
   - Did this phase add any TODO/FIXME pointing to a deferred method?
   - Any new dependency on a code path that requires symm/pbc/relativistic?
   - Any "tiny" features added because they were "easy"?
3. **Validate v1 on representative real-world scripts** before declaring done. If `examples/scf/00-simple_hf.py` works but a typical research notebook doesn't, the scope is wrong, not the implementation.
4. **Weekly scope audit**: `git log --since='1 week' --name-only | xargs -I{} grep -l "tdscf\|mcscf\|pbc" {} 2>/dev/null` — anything matching is a creep alarm.

**Warning signs:**
- A PR titled "Add minor feature X" touches >10 files in modules outside its phase.
- `Out of Scope` items get re-discussed in design notes.

**Phase to address:** Cross-cutting; enforced at every phase transition, especially `mp2` → `ccsd` (post-SCF where (T)/EOM-CC look "close enough") and `dft` → `mp2` (where TDDFT/CIS look tempting).

**Reference:** PySCF code organization in CONCERNS.md / STRUCTURE.md.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Calling `unwrap()` on numerical primitives | Faster prototyping | UB if FFI panics; user-facing crashes | Never on hot path. OK in `xtask/` build scripts. |
| Single-thread reductions for now | Bit-exact tests pass | Performance ceiling at 1 core | Only for `oracle` test profile, never in `release`. |
| Bundling f64 emulation on WGPU | "All four backends" claim met | 10× slowdown vs native CUDA | Only as a fallback with explicit `--features wgpu-soft-f64`. |
| Re-implementing ERIs instead of going through cintx | "Independence" from sibling crate | Bug-for-bug compat with PySCF impossible; doubles maintenance | Never. cintx is the integral engine. |
| Skipping `Python::detach` for "fast" pure-Rust calls | Simpler code | GIL contention, blocks Python multiprocessing users | When Rust call < 100 μs reliably. |
| Static `lazy_static` of Python objects | "Cache the basis-set dict" | Deadlock on free-threaded Python | Use `pyo3::sync::GILOnceCell`, never `lazy_static` for `Py<...>` types. |
| Hardcoded `f32` "for prototyping" | Compiles on every cubecl backend today | Chemistry needs f64; rewriting later is cubecl-version-coupled | Never. Use `f64` from line one; if a backend can't, gate with cfg. |
| Using `assert_eq!` instead of `assert_relative_eq!` in numeric tests | Tests "pass" if you ignore them | Cross-platform CI flake hell | Only for non-FP integer assertions. |
| Skipping the chkfile round-trip test "for now" | Faster green CI | Users discover incompatible chkfiles after release | Never; chkfile is part of the drop-in promise. |
| Python-only fallback for one method | Ships v1 faster | "Pure Rust" claim is now false; users notice | Only with explicit, prominently-documented exclusion list. |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| cubecl autotune | Assuming the cache is shared across processes | Set `CUBECL_CACHE_DIR` to a project-stable path; ship pre-tuned cache for known hardware. |
| cintx | Calling cintx functions from cubecl device code | cintx is the host-side integral engine; device kernels receive precomputed integral tensors. Don't cross. |
| libxc_rs / xcfun_rs | Passing a non-PySCF functional name | Validate functional name against an authoritative list pulled from `pyscf/dft/libxc.py`; reject unknown with a clear message. |
| h5py / hdf5 | Using `hdf5-metno` (Rust-only) for chkfile compat | Use `hdf5-rs` with system libhdf5; chkfiles must be readable by `h5py`. |
| numpy.ndarray PyO3 boundary | Trusting input contiguity | `to_owned()` on entry if `!is_standard_layout()`; document the copy. |
| OpenMP (PySCF) + rayon (Rust) | Both threading at once | At test time, set both to 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`). Production: rayon+OpenMP is fine if their thread counts sum to physical cores. |
| PySCF subclass calling Rust super | Rust dispatch bypasses Python override | All overridable methods use `slf.call_method1(py, "name", ...)`. |
| basis-set-exchange (optional dep) | Assuming all PySCF basis names are BSE-known | Embed PySCF's bundled basis dict as the canonical source; BSE is a fallback path. |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Allocating `Array2` inside SCF inner loop | Heap thrash, 2× slower than expected | Allocate scratchpad once, reuse | Already at 50 basis functions, if hot path. |
| C-order ndarray fed to F-order BLAS | 50× slowdown in DGEMM | Force F-order at type level; lint with newtype | Anywhere BLAS is called. |
| GPU launch for 24-basis molecule | Slower than CPU | Auto-dispatch to CPU below threshold | Tutorial-sized systems (~all PySCF examples). |
| ERI naive 4-index loop | 8× too much work | Use `s8`-packed ERI type; iterate unique quartets | Anywhere shell quartets appear (ERI build, MP2 transform). |
| DIIS subspace too large | Memory blow-up; convergence may stall | Cap at 12 vectors (PySCF default = 6); document override | Long SCF runs >50 iterations. |
| Boys table interpolation degree-4 instead of degree-6 | ~5 ulp error per ERI | Use cintx's table verbatim | All g-shell integrals. |
| f64 → f32 cast on WGPU "to make it work" | Energies wrong by 1e-3 hartree | Gate WGPU at compile time if no f64 extension | Consumer GPUs without `shader-f64`. |
| Reallocating CCSD t2 amplitudes per iteration | Allocator bottleneck | Memory pool with reusable slabs | Caffeine-sized CCSD onwards. |
| Calling `Python::with_gil` inside a rayon parallel iter | Serialization → no parallelism | Hoist GIL acquisition above the loop | Any callback path. |
| Running PySCF oracle in-process with shared `mol` | Cross-test contamination | Subprocess-per-fixture | Test count > ~50. |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Loading user-supplied basis-set Python files via `eval` | Arbitrary code execution | PySCF's basis-set format is data; use a parser, not `exec`. Already a documented PySCF concern (CONCERNS.md). |
| Reading chkfiles from untrusted sources | HDF5 + pickle = RCE if pickle is involved | Refuse to unpickle on chkfile read; require explicit `--allow-pickle` for legacy chkfiles. |
| Linking `hdf5` with too-old version | CVE-2021-46243 / CVE-2021-37501 | Pin minimum HDF5 to 1.12+; document required system version. |
| Logging molecular geometries verbatim | Geometries can leak proprietary structures | Default verbose=0 in libraries; only print on user opt-in. |
| Storing tempfiles in `/tmp` predictably | TOCTOU on shared HPC systems | Use `tempfile::TempDir` with random suffixes; honor `PYSCF_TMPDIR`. |
| Bundling unvetted PTX cubins in wheel | Malicious cubin = RCE | All cubins built from cubecl source at build time, never bundled prebuilt. |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Rust panic surfacing as `RuntimeError: panicked` to Python | User confusion ("what panicked?") | Convert all panics to `PyException` subclasses with chemistry-meaningful messages: `ScfDivergedError`, `BasisSetNotFoundError`, etc. |
| `import pyscf` slow because of basis-set parse | Notebook startup feels broken | Lazy-load basis sets — match PySCF's existing `mol.build()` semantics. |
| GPU backend chosen by env var with no log message | "Why is my CPU run on GPU?" | At first kernel, log `pyscf_rs: using cubecl backend = cuda (auto-selected)` at INFO level. |
| Mismatched chkfile gives `KeyError` | Looks like a bug, not a version mismatch | Detect missing keys, raise `ChkfileVersionError("written by pyscf X.Y, requires pyscf-rs >= ...")`. |
| Subclass override silently ignored | Wrong physics, no error | Phase 7 prevention; if not done, the failure mode is silent — worst possible. |
| `pyscf-rs` and `pyscf` both installed | Import order matters; user can't tell which is loaded | Ship a runtime check: `pyscf.__version__` includes "+rs" suffix; expose `pyscf.is_rs()`. |
| Long-running CCSD with no progress | "Is it stuck?" | Per-iteration callback prints `iter N: dE=..., max|t1|=..., walltime=...s` at INFO. Match PySCF's verbose=4 output format. |
| Cubecl autotune first run takes 30 s | "First import of pyscf-rs is broken" | Pre-tune for common shapes at install (post-install hook) OR display "Tuning kernels for your GPU (one-time, ~30s)..." |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **HF energy matches PySCF:** Often missing chkfile compat, MO coefficient sign convention, gradient. Verify chkfile round-trip + sign canonicalization + gradient match.
- [ ] **DFT works on water:** Often missing exact PySCF grid weight scheme (Becke/Treutler/Stratmann), correct functional-name parser, correct UKS spin-density path. Verify against `pyscf.dft.RKS(water).kernel()` for at least 5 functionals (LDA, BLYP, B3LYP, M06-2X, ωB97X) bit-exact.
- [ ] **MP2 energy matches:** Often missing density-fitted variant, frozen-core handling (default in PySCF is freeze nothing; some codes default freeze-core), open-shell `UMP2`. Verify all of `RMP2`, `UMP2`, `DFMP2` variants.
- [ ] **CCSD converges on H2O/cc-pvdz:** Often missing memory pre-flight, DIIS for amplitudes (separate DIIS instance), restart from chkfile. Verify suite of 5 molecules including a transition metal.
- [ ] **Gradient matches PySCF:** Often missing response-density-matrix terms (Z-vector), Pulay forces from basis-set derivatives, ECP gradient. Verify `numerical_gradient(mf) ≈ analytical_gradient(mf)` for HF, RKS, MP2.
- [ ] **Geomopt converges:** Often missing convergence tolerance match with PySCF (Rgeom, Egeom, gradient norm), step constraint (trust radius), backtracking. Verify against `pyscf.geomopt.optimize(mf)` final geometry.
- [ ] **PyO3 bindings drop-in:** Often missing subclass override dispatch, kwargs that PySCF accepts, chkfile compat, error message styles. Run a curated set of upstream PySCF unit tests against pyscf-rs as the import target — minimum 80% pass.
- [ ] **Oracle CI green:** Often missing cross-platform (aarch64), cross-Python-version (3.10/3.13/3.13t), GPU-backend (CUDA/HIP/WGPU). Matrix has all of these or document gaps.
- [ ] **Wheels published:** Often missing the `pyscf` import-shim package (so old scripts work without rename), HDF5 system-lib documentation, GPU-backend extras. Verify `pip install pyscf-rs && python -c "from pyscf import gto, scf; gto.M(atom='H 0 0 0').build()"` works on a fresh container.
- [ ] **Benchmarks meet 2-5× target:** Often measured on cherry-picked molecules. Verify on the documented benchmark suite (small organics + protein fragment), all backends, both vs PySCF current release and PySCF + libxsmm.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Bit-exact contract broken (FMA / reduction order) | LOW–MEDIUM | Add `RUSTFLAGS=-C target-feature=-fma` to oracle profile; rerun parity tests; identify the specific kernel via bisection. |
| cubecl pre-1.0 breaking change | MEDIUM | Pin to last-known-good cubecl version; file upstream issue; coordinate with cintx/libxc_rs/xcfun_rs to upgrade together. |
| Eigenvector sign flake | LOW | Apply canonicalization helper post-diagonalization; update tests to compare invariants not raw matrices. |
| NumPy non-contiguous input wrong-answer | LOW (after detection) | Patch the FFI seam to `to_owned()`; add fuzz test for stride patterns. |
| GIL deadlock in production | HIGH | py-spy dump to identify lock cycle; refactor callbacks to single-threaded model; add CI test under `python3.13t`. |
| Subclass override ignored | HIGH | Audit all `pyclass` methods, replace `self.method()` with `slf.call_method1(py, "method", ...)`; release patch. |
| Chkfile incompatibility | HIGH | Schema migration tool: read old chkfile, rewrite in new format; ship for one release cycle; document. |
| Cross-platform numerical drift | MEDIUM | Switch transcendentals to `libm` crate; gate bit-exact tests to x86_64-linux; widen tolerance on aarch64. |
| Wheel size > PyPI limit | MEDIUM | Split into per-backend extras; request size exemption preemptively. |
| Panic across FFI (UB symptom) | HIGH | Wrap `extern "C"` boundaries in `catch_unwind`; replace panics with `Result` in numerical code. |
| Sibling-crate dep conflict | MEDIUM | `[patch.crates-io]` to point all to local paths; release coordinated bump. |
| DIIS path drift on hard molecule | LOW–MEDIUM | Document as "chemical accuracy regime"; widen tolerance for that fixture; only require energy match. |
| DFT grid weights wrong | LOW | Re-port `gen_grid.py` with line-by-line cross-reference; oracle test on grid points. |
| GPU launch overhead on small molecule | LOW | Add work-estimate dispatch heuristic; benchmark; ship CPU as default for small. |
| CCSD OOM | MEDIUM | Implement scratchpad pattern; add memory pre-flight; recommend DF-CCSD for users who hit it. |
| Off-by-one in basis indexing | HIGH | Per-basis nightly oracle test catches it; revert to cintx as ground truth. |
| Boys function inaccuracy | LOW | Replace with cintx's evaluation; cintx already uses correct table. |
| Test global-state contamination | MEDIUM | Subprocess-per-fixture refactor; reuse cintx-oracle test pattern. |
| Scope creep | HIGH (releases delayed) | Strict phase-boundary review; revert out-of-scope code; promote to next milestone explicitly. |
| Sibling functional-name incompatibility | LOW | Build name-translation table from PySCF source; gate via parity test. |

---

## Pitfall-to-Phase Mapping

| # | Pitfall | Severity | Prevention Phase | Verification |
|---|---------|----------|------------------|--------------|
| 1 | FMA contraction breaks bit-exact | SHOWSTOPPER | `infra` | LLVM-IR grep for `fmuladd`; bit-exact H2O fixture |
| 2 | Parallel-reduction order non-determinism | SHOWSTOPPER | `infra` | Same-bits-on-1-vs-N-threads test |
| 3 | cubecl pre-1.0 churn / f64 holes | MAJOR | `infra` | f64 smoke test on every backend; pinned versions |
| 4 | Eigenvector sign / degenerate ordering | SHOWSTOPPER | `scf` | Canonicalization helper; invariant-only comparisons |
| 5 | NumPy zero-copy hazards | MAJOR | `bindings` | Stride-fuzz test on every PyO3 entry point |
| 6 | GIL deadlock | MAJOR | `bindings` | `python3.13t` CI; py-spy on hangs |
| 7 | PyO3 subclassing breaks polymorphism | SHOWSTOPPER | `bindings` | Subclass-override-called assertion test |
| 8 | Loop order / Fortran layout | MAJOR | `gto` + `scf` | F-order newtype; perf bench vs PySCF |
| 9 | DIIS path drift | MAJOR | `scf` + `oracle` | Energy-only assertion for hard fixtures |
| 10 | DFT grid weighting wrong | MAJOR | `dft` | Grid-points + weights bit-exact vs PySCF |
| 11 | chkfile compatibility | SHOWSTOPPER | `scf` (then per method) | Round-trip oracle test |
| 12 | Cross-platform drift | MAJOR | `infra` | Bit-exact gated to x86_64-linux only |
| 13 | Wheel size / CUDA distribution | MAJOR | `distribution` | PyPI test upload; per-backend extras |
| 14 | Rust panic across FFI | SHOWSTOPPER | `infra` | `catch_unwind` lint; no `unwrap()` in numerical code |
| 15 | Sibling-crate ABI / cubecl drift | SHOWSTOPPER | `infra` + `oracle` | Nightly cross-crate matrix; functional-name parity |
| 16 | Test-oracle global-state | MAJOR | `oracle` | Subprocess-per-fixture pattern |
| 17 | Off-by-one basis indexing | MAJOR | `gto` | Per-basis nightly oracle |
| 18 | Boys-function accuracy | MAJOR | `gto` (via cintx) | Boys-only isolated test |
| 19 | GPU launch overhead small mol | MAJOR | `scf` (sets pattern) | Small-molecule benchmark CPU vs GPU |
| 20 | CCSD memory thrash | MAJOR | `ccsd` | Pre-flight memory check; OOM-budget test |
| 21 | Scope creep | MAJOR | All transitions | Forbidden-paths lint; phase-boundary audit |

---

## Phase Research Flags

Phases that carry concentrated risk and likely need deeper Phase research before execution:

- **`infra`** — high stakes (the FMA/reduction/panic/cubecl-pin foundation) but fairly mechanical. Research-light, design-heavy.
- **`scf`** — DIIS, eigenvector canonicalization, chkfile schema. Will need PySCF-source-level deep-dive. *Flagged for phase research.*
- **`dft`** — grid generation parity, functional-name parser, libxc/xcfun bridging. *Flagged for phase research.*
- **`ccsd`** — memory architecture, DF integration. *Flagged for phase research.*
- **`bindings`** — subclass-override audit, GIL-release seam map, numpy contract. *Flagged for phase research.*
- **`oracle`** — test isolation, fixture organization, nightly vs pre-merge split. *Flagged for phase research.*
- **`distribution`** — wheel split, PyPI exemption, system-lib link strategy. *Flagged for phase research.*

Phases unlikely to need deep research (mostly mechanical translation):
- **`gto`** — cintx already does the work; pyscf-rs gto is a thin wrapper.
- **`mp2`** — once `gto` and `scf` are right, MP2 is a tensor contraction; standard patterns.
- **`grad`** — analytical-gradient formulas are textbook; testing against numerical gradients is mechanical.
- **`geomopt`** — BFGS/RFO from `argmin` or similar; well-trodden.

---

## Sources

### Primary (authoritative, HIGH confidence)

- [PROJECT.md](file:///home/user/Documents/workspace/pyscf_rs/.planning/PROJECT.md) — scope, constraints, decisions
- [CONCERNS.md](file:///home/user/Documents/workspace/pyscf_rs/.planning/codebase/CONCERNS.md) — upstream PySCF debt and fragile areas
- [TESTING.md](file:///home/user/Documents/workspace/pyscf_rs/.planning/codebase/TESTING.md) — upstream test conventions, fixtures, threading caveats
- [INTEGRATIONS.md](file:///home/user/Documents/workspace/pyscf_rs/.planning/codebase/INTEGRATIONS.md) — h5py, libcint, libxc, FFTW dependency surface
- [cintx CHANGELOG](file:///home/user/Documents/workspace/cintx/CHANGELOG.md) — sibling-crate breaking-change patterns
- [cintx-oracle Cargo.toml](file:///home/user/Documents/workspace/cintx/crates/cintx-oracle/Cargo.toml) — oracle-build conventions
- [xcfun_rs validation harness](file:///home/user/Documents/workspace/xcfun_rs/validation/) — sibling-crate parity-harness pattern

### PyO3 / rust-numpy

- [PyO3 user guide](https://pyo3.rs/main/) — Bound API, GIL, subclassing
- [PyO3 FAQ on GIL deadlocks](https://pyo3.rs/v0.23.4/faq.html)
- [PyO3 discussion #3045 — tokio + GIL deadlock](https://github.com/PyO3/pyo3/discussions/3045)
- [PyO3 discussion #4738 — Python 3.13 free-threaded deadlock](https://github.com/PyO3/pyo3/discussions/4738)
- [PyO3 discussion #4164 — overriding methods on subclasses](https://github.com/PyO3/pyo3/discussions/4164)
- [PyO3 issue #492 — panics across FFI](https://github.com/PyO3/pyo3/issues/492)
- [rust-numpy issue #114 — slicing + to_pyarray incorrect](https://github.com/PyO3/rust-numpy/issues/114)
- [numpy issue #14627 — wrong stride/contiguous calculations](https://github.com/numpy/numpy/issues/14627)
- [Free-Threading Porting Guide](https://py-free-threading.github.io/porting-extensions/)
- [Trail of Bits: Python wheel ABI](https://blog.trailofbits.com/2022/11/15/python-wheels-abi-abi3audit/)
- [Maturin distribution guide](https://www.maturin.rs/distribution.html)
- [Quansight: Python packaging in 2021](https://labs.quansight.org/blog/2021/01/python-packaging-brainstorm)

### cubecl

- [cubecl GitHub](https://github.com/tracel-ai/cubecl)
- [cubecl releases](https://github.com/tracel-ai/cubecl/releases) — v0.10.0-pre.4 latest as of April 2025
- [cubecl issue #1316 — f64 ln/exp invalid GLSL.std.450](https://github.com/tracel-ai/cubecl/issues/1316)
- [cubecl issue #1317 — f64 to i64 cast on SPIR-V](https://github.com/tracel-ai/cubecl/issues/1317)
- [cubecl issue #1318 — atomic u64 panic](https://github.com/tracel-ai/cubecl/issues/1318)
- [WebGPU spec issue #2805 — f64 IEEE 754 binary64](https://github.com/gpuweb/gpuweb/issues/2805)

### Floating-point determinism

- [Rust users forum — mul_add accuracy and FMA](https://users.rust-lang.org/t/why-does-the-mul-add-method-produce-a-more-accurate-result-with-better-performance/1626)
- [packed_simd FMA guide](https://rust-lang.github.io/packed_simd/perf-guide/float-math/fma.html)
- [KDAB FMA woes](https://www.kdab.com/fma-woes/)
- [Gaffer on Games — Floating Point Determinism](https://gafferongames.com/post/floating_point_determinism/)
- [Random ASCII — Floating-Point Determinism](https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/)
- [Arm Learning Path — floating-point cross-platform](https://learn.arm.com/learning-paths/cross-platform/floating-point-rounding-errors/)

### Rust FFI / panic safety

- [Rust nomicon — FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [Rust reference — panic](https://doc.rust-lang.org/stable/reference/panic.html)
- [RFC 2945 — c-unwind ABI](https://rust-lang.github.io/rfcs/2945-c-unwind-abi.html)

### PySCF specifics

- [PySCF SCF user guide](https://pyscf.org/user/scf.html)
- [PySCF threading issue #1102](https://github.com/pyscf/pyscf/issues/1102)
- [PySCF threading issue #1138](https://github.com/pyscf/pyscf/issues/1138)
- [PySCF macOS-arm64 install issue #1015](https://github.com/pyscf/pyscf/issues/1015)
- [PySCF make_natural_orbitals + symmetry issue #1196](https://github.com/pyscf/pyscf/issues/1196)
- [PySCF flipped-signs ERI issue #1935](https://github.com/pyscf/pyscf/issues/1935)
- [PySCF chkfile source — pyscf/lib/chkfile.py](https://github.com/pyscf/pyscf/blob/master/pyscf/lib/chkfile.py)
- [PySCF gen_grid example](https://github.com/pyscf/pyscf/blob/master/examples/dft/11-grid_scheme.py)
- [PySCF ao2mo docs](https://pyscf.org/pyscf_api_docs/pyscf.ao2mo.html)
- [PySCF ECP source](https://pyscf.org/_modules/pyscf/gto/ecp.html)

### Quantum-chemistry methods

- [Sun 2015 — libcint paper](https://arxiv.org/pdf/1412.0649)
- [libcint sph/cart issue #22](https://github.com/sunqm/libcint/issues/22)
- [DIIS Wikipedia](https://en.wikipedia.org/wiki/DIIS)
- [Sherrill notes on DIIS](https://vergil.chemistry.gatech.edu/static/content/diis.pdf)
- [Q-Chem 5.1 manual — converging SCF](https://manual.q-chem.com/5.1/sect-convergence.html)
- [TRAH-SCF paper](https://pubs.aip.org/aip/jcp/article/154/16/164104/317501/A-trust-region-augmented-Hessian-implementation)
- [Mamedov 2004 — Boys function](https://www.researchgate.net/publication/226566421)
- [NASA TR — DFT weight-scheme effect on frequencies](https://ntrs.nasa.gov/citations/20040084670)
- [Permutational symmetries notes (Sherrill)](http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf)
- [Reduced-cost CCSD(T) implementation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912216/)
- [Psi4 CCENERGY caching levels](https://psicode.org/psi4manual/master/autodir_options_c/module__ccenergy.html)

---

*Pitfalls research for: pyscf_rs (pure-Rust PySCF rewrite, cubecl + PyO3, drop-in `pyscf.*` surface)*
*Researched: 2026-05-09*
