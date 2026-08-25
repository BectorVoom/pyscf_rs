---
phase: 09-pbc-foundation
plan: 05
subsystem: pbc-gto / kernels
tags: [pbc, gv, structure-factor, uniform-grids, fftfreq, cubecl, K-01, K-02]

# Dependency graph
requires:
  - phase: 09-pbc-foundation
    plan: 01
    provides: "the pyscf-pbc-gto crate scaffold + path-scoped lint exemptions"
  - phase: 09-pbc-foundation
    plan: 02
    provides: "the planar CTensor complex contract (D-PBC-02 / RULE 8) and the pyscf-kernels/src/pbc module seeded by K-04"
  - phase: 09-pbc-foundation
    plan: 03
    provides: "Cell, lattice_vectors / vol / reciprocal_vectors, atom_coords through Deref"
  - phase: 09-pbc-foundation
    plan: 04
    provides: "Cell::try_mesh — the default mesh get_Gv / get_uniform_grids fall back to"
provides:
  - "K-01 pyscf_kernels::gv — the (ngrids,3) G-vector table on the device"
  - "K-02 pyscf_kernels::struct_factor — planar SI[a,g] = exp(-i Gv.R_a) on the device"
  - "pyscf_pbc_gto::gv — fftfreq_scaled, fftfreq, get_gv, get_gv_weights, get_si, get_uniform_grids (+ the four Cell methods)"
  - "GvWeights { gv, gvbase, weights, mesh } — upstream's (Gv, Gvbase, weights) tuple"
affects: [09-06, 09-07, 09-08, 09-09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "a pyscf-pbc-* crate reaches the device ONLY through pyscf-kernels (RULE 6); it never names cubecl-* itself"
    - "kernel-level tests live with the kernel (pyscf-kernels/tests/) and assert bit-identity with a host reference; the upstream-numeric gate lives with the caller (pyscf-pbc-gto/tests/)"
    - "a second reference cell built DIRECTLY in Bohr sidesteps the Unit::Ang gap, so an upstream comparison can be asserted at its intended tolerance instead of a loosened one"
    - "a large hard-coded reference table lives under tests/common/ (include!d), not tests/, so cargo does not make it an integration-test target"

key-files:
  created:
    - crates/pyscf-kernels/src/pbc/gv.rs
    - crates/pyscf-kernels/src/pbc/struct_factor.rs
    - crates/pyscf-kernels/tests/pbc_gv.rs
    - crates/pyscf-pbc-gto/src/gv.rs
    - crates/pyscf-pbc-gto/tests/gv.rs
    - crates/pyscf-pbc-gto/tests/common/gv_reference.rs
  modified:
    - crates/pyscf-kernels/src/pbc/mod.rs
    - crates/pyscf-kernels/src/lib.rs
    - crates/pyscf-pbc-gto/src/lib.rs
    - crates/pyscf-pbc-gto/Cargo.toml
    - crates/pyscf-runtime/src/probe/wgpu.rs   # pre-existing clippy blocker

key-decisions:
  - "The K-01 body transcribes `lib/pbc/cell.c:133-141` including its ACCUMULATION ORDER (rx term, then += ry, then += rz); FP addition is not associative, so reordering would move the last bits of every planewave integral."
  - "The 1D launch inverts the C-order flat index per thread (x = g/(my*mz), y = (g/mz)%my, z = g%mz) rather than using a 3D grid, because the plan mandates CubeCount::Static(ceil(ngrids/256),1,1) / CubeDim{x:256}."
  - "get_SI ports BOTH upstream branches: the separable Gvbase product (default, natm*(mx+my+mz) transcendentals) and the direct K-02 device form (when Gv is given). They agree to 1.3e-15, matching upstream's own 1.8e-15 spread."
  - "SI is a planar CTensor, never interleaved (D-PBC-02 / RULE 8)."
  - "The inf_vacuum / non-uniform Gv branches return NotYetImplemented { phase: 12 } (D-PBC-20) — they need Gauss-Chebyshev quadrature and turn `weights` from a scalar into a per-grid array."
  - "pyscf-pbc-gto gained a pyscf-kernels dependency (RULE 6). check-dependency-wall still PASSes: the wall forbids cubecl-* in method crates, and pyscf-kernels is the carve-out kernel home."

patterns-established:
  - "Meshes [6,6,7] and [6,7,6] as a transposed-index probe: same grid count, different shape, so a swapped my/mz cannot pass"
  - "The defining reciprocal-grid identity Gv[g].a[i] == 2*pi*r_i as a tier-1 test that pins row order, axis order and normalisation at once"

requirements-completed: [PBC-GTO-04]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 05: G-vectors, Structure Factors, Uniform Grids

**Two new cubecl kernels — K-01 `gv` and K-02 `struct_factor`, both generic over
`F: Float` and launched through `dispatch_backend!` — plus the `pyscf-pbc-gto` host
wrappers `get_Gv` / `get_Gv_weights` / `get_SI` / `get_uniform_grids` and the
`fftfreq` helper every downstream FFT depends on. The plan's headline gate (the full
125x3 diamond `Gv` array against upstream at 1e-12) passes with ~1 ULP to spare.**

## What Shipped

### K-01 — `crates/pyscf-kernels/src/pbc/gv.rs`

`Gv[g] = rx[x]*b[0] + ry[y]*b[1] + rz[z]*b[2]`, one thread per grid point. The body is
the nine-line inner loop of `pyscf/lib/pbc/cell.c:133-141` transcribed verbatim,
**including its accumulation order** — floating-point addition is not associative, so
reordering the three terms would move the last bits of every G-vector and therefore of
every planewave integral downstream. The flat index `g = x*my*mz + y*mz + z` is inverted
per thread; launch geometry is the plan's `CubeCount::Static(ceil(ngrids/256), 1, 1)` /
`CubeDim { x: 256, y: 1, z: 1 }`.

### K-02 — `crates/pyscf-kernels/src/pbc/struct_factor.rs`

`theta = -(Gv[g] . R_a)`, `si_re = cos(theta)`, `si_im = sin(theta)`, one thread per
`(a, g)`. Outputs are two flat planes (PLANAR, D-PBC-02 / RULE 8), row-major
`(natm, ngrids)`.

### `crates/pyscf-pbc-gto/src/gv.rs`

| upstream | Rust |
|---|---|
| `cell.py:525-537` `get_Gv` | `get_gv` (+ `Cell::get_gv`) |
| `cell.py:539-604` `get_Gv_weights` | `get_gv_weights` -> `GvWeights` (+ `Cell::get_gv_weights`) |
| `cell.py:606-613` `_non_uniform_Gv_base` | **deferred**, `NotYetImplemented { phase: 12 }` |
| `cell.py:615-646` `get_SI` | `get_si` — both branches (+ `Cell::get_si`) |
| `cell.py:886-911` `get_uniform_grids` | `get_uniform_grids` (+ `Cell::get_uniform_grids`) |
| `np.fft.fftfreq(n, 1./n)` | `fftfreq_scaled` |
| `np.fft.fftfreq(n)` | `fftfreq` |

`GvWeights` carries upstream's `(Gv, Gvbase, weights)` tuple plus the mesh actually used.
`weights = |det(b)|/(2*pi)^3`, which upstream's own comment (`cell.py:600`) notes equals
`1/cell.vol`.

`get_si` ports BOTH upstream branches. With `Gv = None` (upstream's default) it uses the
SEPARABLE form of `cell.py:626-635` — one complex exponential per
`(atom, axis, frequency)` then the outer product, i.e. `natm*(mx+my+mz)` transcendental
calls instead of `natm*ngrids`. With `Gv = Some(...)` it runs K-02 on the device. The two
agree to 1.3e-15 (upstream's own two branches differ by 1.8e-15 on the same cell).

## Verification Results

```
cargo test -p pyscf-pbc-gto --test gv                                        ✅ 19 passed / 0 failed
cargo test -p pyscf-kernels --test pbc_gv                                    ✅  6 passed / 0 failed
cargo test -p pyscf-pbc-gto --all-features                                   ✅ 71 passed / 0 failed
cargo test -p pyscf-kernels                                                  ✅ 21 passed / 0 failed
cargo test -p pyscf-pbc-tools                                                ✅ 12 passed / 0 failed
cargo clippy -p pyscf-pbc-gto -p pyscf-kernels -p pyscf-pbc-tools --all-targets --all-features -- -D warnings   ✅ clean
cargo build --workspace                                                      ✅ clean
cargo run -p xtask --bin check-dependency-wall                               ✅ PASS (ALG-06 intact)
cargo run -p xtask --bin check-forbidden-paths                               ✅ PASS (350 files)
rustfmt --edition 2024 --check <all touched files>                           ✅ clean
```

`--all-features` on `pyscf-kernels` enables `cpu` + `cuda` + `wgpu` + `rocm`, so the
clippy run above compiles **all four `dispatch_backend!` arms** of both new kernels, not
just the CPU one.

### Tier 2 — upstream PySCF 2.12.1 (D-PBC-19)

| check | result |
|---|---|
| `fftfreq_scaled(n)` / `fftfreq(n)`, n = 1..8 | **exact** (`assert_eq!` on the raw tables) |
| diamond `Gv`, mesh `[5,5,5]`, full 125x3 | max abs deviation **< 1e-15** (gate is 1e-12) |
| `weights` at mesh `[5,5,5]` | 0.013062524449620905, relative < 1e-14 |
| `SI[0, :4]`, `SI[1, :6]` | absolute < 1e-14 |
| `get_uniform_grids`, both `wrap_around`, rows 0/1/7/31/124 + `abs().sum()` | < 1e-13 |

The 125x3 table lives in `tests/common/gv_reference.rs`; the generating snippet is
recorded in `tests/gv.rs`'s `UPSTREAM_SNIPPET`.

### Tier 1 — invariants (no upstream needed)

`fftfreq_scaled(n)` for **n = 1..=32**: every value integral, congruent to its index mod
`n`, inside numpy's `[-n/2, n/2)` window, and all `n` residues distinct — the plan warns
that getting this fold wrong "silently corrupts every FFT downstream", so it is
characterised twice rather than only table-matched. `fftfreq(n) == fftfreq_scaled(n)/n`
and lies in `[-0.5, 0.5)`.

`Gv[g] . a[i] == 2*pi * r_i` — the defining reciprocal-grid identity, checked on all five
§9.2 systems; it pins the row order, the axis assignment and the `2*pi` normalisation at
once, so a transposed `b`, a swapped axis or a missing `2*pi` all fail without any
reference value. `Gv[0] == [0,0,0]`; the G-vector set is closed under negation on an odd
mesh; `weights == 1/vol` and is mesh-independent. `|SI[a,g]| == 1` to 1e-14 and
`SI[a,0] == 1+0i` on both branches and all five systems; an atom at the origin gives
`1+0i` everywhere; `SI[a,-g] == conj(SI[a,g])`. The uniform grid is exactly the
lattice-fraction product grid, sums to zero for `wrap_around = true`, and the two variants
differ only by whole lattice translations. `atmlst` selects and reorders rows and rejects
out-of-range indices; a zero mesh axis errors; `mesh = None` falls back to `cell.mesh`
(`[47,47,47]`, 103823 points).

At the kernel level (`pyscf-kernels/tests/pbc_gv.rs`), both kernels match a naive host
reference **bit-for-bit** (`assert_eq!` on `Vec<f64>`, not a tolerance) across seven mesh
shapes — including `[6,6,7]` vs `[6,7,6]`, which have the same grid count so a swapped
`my`/`mz` cannot hide, and `[8,8,4]` = 256 exactly plus `[9,9,9]` = 729 for the tail
guard — with a deliberately non-symmetric `b` so a transposed matrix cannot pass.

## DEVIATIONS from the plan

**1. The `weights` formula in the plan text is right but incomplete.** §8.1 step 3 says
`weights = |det(b)| / (2π)³` for the uniform 3D case. That is what is implemented; the
port also records upstream's own comment that this equals `1/cell.vol`, and asserts it.

**2. `get_SI` ports the SEPARABLE branch too.** The plan's step 4 describes only the
direct K-02 form. Upstream's DEFAULT call (`Gv = None`) uses the separable Gvbase product,
which is `natm*(mx+my+mz)` transcendentals instead of `natm*ngrids` — on diamond's real
`[47,47,47]` mesh that is 282 versus 207 646 per atom. Shipping only the direct form would
have made every downstream default call an order of magnitude slower than upstream and
would not have reproduced its rounding. Both branches ship; a test asserts they agree.

**3. `get_uniform_grids` is host-side.** The plan names only K-01 and K-02 as kernels;
step 5 gives `get_uniform_grids` as a host formula. It is a strided AXPY over three
precomputed fractional axes, so the device round-trip would cost more than the arithmetic.

**4. Snake-case names.** The plan writes `get_Gv`, `get_Gv_weights`, `get_SI`. Those trip
`non_snake_case` under the `-D warnings` gate, so the Rust spellings are `get_gv`,
`get_gv_weights`, `get_si` — the same convention plan 09-03 used for `get_abs_kpts`. Each
doc comment names the upstream symbol.

**5. The plan's 1e-12 `Gv` tolerance needed a second reference cell to be reachable.**
The §9.2 `systems::diamond()` builds its lattice from Angstrom and so carries the 4.95e-9
`pyscf_core::Unit::Ang` gap documented in 09-03 — `|Gv| ~ 1.86`, so the absolute deviation
is ~9e-9 and a 1e-12 absolute check is unreachable no matter how correct the port is. The
test therefore adds `diamond_bohr()`, the SAME cell with the lattice given directly in
Bohr using upstream's own `cell.lattice_vectors()` literals. Against that cell the port
matches upstream to **< 1e-15** (~1 ULP — the only residual is closed-form `inv3` versus
numpy's LU inverse), satisfying the plan's stated tolerance. The Angstrom cell is checked
separately at a relative 1e-7 with the residual pinned below 2e-8, so the loosened bound
cannot hide a real error.

**6. `tests/common/gv_reference.rs`, not `tests/gv_reference.rs`.** Cargo makes every
`tests/*.rs` its own integration-test target, which turned the 125-row table into a target
with an unused constant (`-D dead-code`). Moving it one level down and `include!`ing it
keeps it a plain data file.

## Out-of-scope fixes made to unblock the verification gate

`cargo clippy -p pyscf-kernels --all-targets --all-features -- -D warnings` lints the
whole dependency graph, and `--all-features` newly pulls in `pyscf-runtime`'s `wgpu`
module. One PRE-EXISTING finding there failed the gate; it is a doc-formatting no-op:

* `crates/pyscf-runtime/src/probe/wgpu.rs:27` — `doc_lazy_continuation`: a blank `///`
  line inserted between the D-09 bullet list and the paragraph that follows it.

## Carry-overs

- **The `inf_vacuum` / non-uniform `Gv` base is DEFERRED to Phase 12** (D-PBC-20,
  `NotYetImplemented { phase: 12 }`). It needs `pyscf.dft.radi.gauss_chebyshev` and turns
  `GvWeights::weights` from an `f64` into a per-grid `Vec<f64>` — a signature change, so
  Phase 12 should widen the field rather than add a parallel API. `get_si`'s separable
  branch inherits the deferral because it calls `get_gv_weights`.
- **`get_Gv_weights` ignores the deprecated `gs=` kwarg** (`cell.py:549-552`), which
  upstream only keeps to emit a `DeprecationWarning`. `pyscf_pbc_tools::mesh::gs_to_cutoff`
  already covers the `gs` spelling where it still matters.
- **No real-GPU differential test for K-01/K-02.** Both kernels are exercised through the
  cubecl `CpuRuntime` (the genuine device path) and all four backend arms compile under
  `--all-features`, but the ROCm gfx1152 oracle that quick-260529-* runs for the algebra
  kernels has no analogue here yet. The bit-for-bit host reference in
  `tests/pbc_gv.rs` is the current contract.
- **`pyscf_core::Unit::Ang.length_in_au()` still disagrees with upstream by 4.95e-9
  relative** (09-03 carry-over, unchanged). It is why `angstrom_diamond_gv_matches_...`
  uses a relative bound; `diamond_bohr()` shows what the port does once that constant is
  fixed.
- **`get_uniform_grids`'s non-`wrap_around` branch is untested against a 2D cell's
  `get_lattice_Ls`** — upstream's comment says the extra image layer that branch needs is
  a `get_lattice_Ls` concern, which is plan 09-06.
