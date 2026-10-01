# Band structure on-device: KTaO3 stand-in, BAND-08, and the T4 plan (2026-09-23)

Request: make the YTa7O19 band-structure path (`KRKS` + `get_bands`) fast and
memory-lean on a discrete GPU (Colab T4), with the same CubeCL kernels serving
the CPU runtime, and show the GPU beating the CPU. YTa7O19 itself (54 atoms,
312 AOs, mesh 31×31×95, 1 486 s for three Γ-only SCF cycles on 16 cores) is
too heavy to iterate on, so a stand-in was chosen and profiled first.

## 1. The stand-in: KTaO3

`krks_profile bands --cell ktao3 --ke 100 --lattice cubic` — cubic perovskite
(Pm-3m, a = 3.9885 Å, 5 atoms), `gth-szv-molopt-sr` / `gth-pbe`, PBE. It keeps
the parts of YTa7O19 that shape the kernels — Ta with the GTH-PBE q13
pseudopotential (l = 2 projectors, d shell) and O q6 — and drops what made
YTa7O19 unusable as a fixture: it is an insulator (an independent SCF is a
sound reference, unlike the metallically degenerate CsCl YTa of
`krhf_bands_oracle_yta.rs`), and it is small enough that one profile is a
minute. `--ke 100` is mandatory: the basis-derived default cutoff gives a
307 200-point mesh and a 400 s AO pass per SCF cycle. `--super a,b,c` scales
the cell toward the real `nao` when a heavier point is wanted.

| quantity | value |
|---|---|
| nao / nelectron | 27 / 40 |
| mesh (ke 100 Ha) | 35³ = 42 875 points |
| SCF k-mesh / band path | 2×2×2 / Γ-X-M-Γ-R-X, 33 points |
| images in the deriv-1 AO lattice sum | 1 237 |

## 2. Profile on the 16-core CPU runtime (before BAND-08)

`krks_profile bands` after BAND-01..07, three warm `get_bands` reps:

| stage | ms |
|---|---|
| SCF (to 1e-9) | 19 622 |
| `get_bands` (33 k) | 13 277 / 13 112 / 13 268 |
| stages, warm: hcore / ovlp / nr_xc / get_j / eig | 7 955 / 45 / 14 205 / 3 761 / 52 |

Span totals over the three reps plus the stage calls, with the stage spans
added to `fused_band_fock` this session (`band_xc_weights`, `band_coulomb`,
`band_vmats`, `band_hcore_nl`, `band_vloc`, `band_ovlp`, `band_eig`):

| span | calls | ms |
|---|---|---|
| `band_vmats` | 3 | 38 823 |
| `pbc_eval_ao_readback` | 34 | 53 516 |
| `pbc_eval_ao_kpts` (host side of the accumulate) | 34 | 3 644 |
| `band_xc_weights` | 3 | 301 |
| `band_eig` / `band_hcore_nl` / `band_ovlp` | 3 each | 175 / 156 / 148 |

So `band_vmats` is 13.0 s of each 13.3 s call. The "readback" number is a
trap this session fell into for an hour: launches are lazy, and on the CPU
runtime the AO kernels EXECUTE inside whichever span performs the next
`read`, which was the read-back. Adding a `sync_device` barrier
(`pbc_eval_ao_exec`) is what separates kernel execution from the copy —
see [[lazy-launches-blur-stage-spans]].

## 3. BAND-08 — the band contraction on the device

`band_vmats` used to evaluate the band AO table (`nk · 4 · ngrids · nao`
complex, 2.4 GB per call here), read all of it back and contract it on the
host (`vxc_mat_one`, rayon over rows, pairwise `oracle_sum`). BAND-08 adds:

* `pyscf_kernels::pbc::band_vmat` / `band_vmat_resident`
  (`crates/pyscf-kernels/src/pbc/band_vmat.rs`): the K-14f `local_vmat`
  kernel generalised to `comp <= 4` components and `nvar` grid weights —
  `v[k][p,q] = Σ_g conj(ao⁰[k,p,g]) · Σ_n wv[n,g] · aoⁿ[k,q,g]`, one lane per
  `(k, p, q)`, both accumulator layouts (point-major gathered per k), a
  `#[comptime] im_zero` variant for the Γ point instead of zeroing the
  borrowed accumulator in place.
* `pyscf_pbc_gto::eval_ao_kpts_band_vmat`: same lattice sum and image list
  as `eval_ao_kpts`, contraction on the resident accumulator, only
  `nset · nk · nao²` comes home.
* `KsNumInt::band_vmats` route switch `PYSCF_PBC_BAND_VMAT_DEVICE`
  (`0` host, `1` device, unset = device iff the backend has hardware planes —
  so the CPU runtime's default is UNCHANGED).
* `AlgebraClient::has_planes()` and `sync_device()` (pyscf-algebra owns the
  `cubecl_*` runtime crates the dispatch macro names).
* `PYSCF_PBC_AO_RESIDENT_NKT` — a k-tile cap on the resident AO kernel, a
  GPU-shape dial that is bit-identical for any value.

Gates: `pyscf-kernels/tests/pbc_band_vmat.rs` (bitwise against a serial
reference, `nvar < comp`, shape errors) and
`pyscf-pbc-dft/tests/band_vmat_device.rs` (Si/PBE and Si/LDA `get_bands`,
host vs device, worst 2.5e-14 and 1.5e-14 Ha against a 1e-9 gate).

Measured on the 16-core CPU runtime, same binary, KTaO3 as above:

| route | `get_bands` ms (3 reps) | worst band diff vs host | peak RSS |
|---|---|---|---|
| host (`=0`) | 13 277 / 13 112 / 13 268 | — | 7.51 GB |
| device (`=1`) | 14 048 / 13 792 / 13 872 | 6.5e-14 Ha | 7.51 GB |

Read-back span 53.5 s → 16.7 s (the 4 remaining reads are `nr_xc`/`hcore`
stage calls, not the band path), `pbc_band_vmat` 1.9 s for 30 blocks, but
wall time is unchanged. With the `sync_device` barrier in place the device
route attributes honestly (same binary, three reps, 30 grid blocks):

| span (device route) | calls | ms | per `get_bands` |
|---|---|---|---|
| `band_vmats` | 3 | 39 498 | 13 166 |
| `pbc_eval_ao_exec` — the fused resident AO kernel executing | 30 | 34 959 | 11 653 |
| `pbc_eval_ao_kpts` — host side of the accumulate (uploads, staging) | 34 | 3 665 | ~1 100 |
| `pbc_band_vmat` — BAND-08 contraction incl. its read | 30 | 1 899 | 633 |

The AO kernel is 88 % of the stage on this runtime; the host contraction it
replaced was ~1.5 s per call, and the device contraction's per-`(k,p,q)`
lanes are slightly slower than the rayon route here. Peak RSS is set elsewhere (the SCF's cached AO table /
`get_hcore`, see [[hcore-peak-is-the-ao-evaluation-not-the-readback]]), so
neither route moves it on this fixture. **Decision:** CPU default stays on
the host route; the device route is the GPU default, where the 2.4 GB table
would otherwise cross PCIe and be reduced by a two-core host.

## 4. GPU — Kaggle T4, measured 2026-09-24

Colab never allocated a T4 (every request 503 for 3+ hours; L4 not
entitled). The run moved to Kaggle, account `yensen2`, notebook
`yensen2/ktao3-bands-t4` (version 4), dataset `yensen2/pyscf-rs-ktao3-runner`.
VM: Tesla T4 (driver 580, CUDA 12.8 NVRTC), 4 vCPU Xeon 2.0 GHz, 31 GB.
Binary: `krks_profile`, `cpu,cuda` features, cubecl-cpp patched to register
F64. Every arm passed `--require-backend`. Sources and outputs:
`.yta_bundle/kaggle_t4/` (`make_nb.py`, `out4/summary.json`, per-arm logs).

Correctness was isolated with `--dm-out` / `--dm-in` (bit-exact density
dump, added to `krks_profile bands`): the CPU arm's converged density drives
both CUDA band arms.

| arm | density | max band diff vs CPU host | 
|---|---|---|
| CUDA, BAND-08 device contraction | CPU's | 5.8e-14 Ha |
| CUDA, host contraction | CPU's | 1.1e-14 Ha |
| CPU, device contraction | CPU's | 6.5e-14 Ha |
| CUDA, own SCF | its own | 1.03e-6 Ha |

The CUDA SCF lands on the same `e_tot` to 15 digits (-134.6216686224864 Ha,
9 cycles, both backends), but its density differs at the `conv_tol = 1e-9`
level and band energies are first-order in the density. That 1e-6 is SCF
tolerance, not a kernel error — the same lesson as
[[yta-fixture-is-metallically-degenerate]]: gate bands on a shared density.

Timings (warm reps; the first `--dm-in` rep includes JIT compilation of
kernels the skipped SCF would have warmed, ~57 s):

| | `get_bands` warm | SCF | host peak RSS | GPU mem peak |
|---|---|---|---|---|
| T4, BAND-08 device route | **7.5 s** | 87.6 s | 8.54 GB | 5.97 GB |
| T4, host contraction | 10.1 s | — | 8.77 GB | 5.97 GB |
| Kaggle VM CPU (4 vCPU), host route | 84.4 s | 131.1 s | 7.07 GB | — |
| local 16-core CPU, host route | 13.2 s | 19.6 s | 7.33 GB | — |

* BAND-08 on the GPU: 10.1 s → 7.5 s (1.35×) and −230 MB host RSS — the
  2.4 GB table no longer crosses PCIe (`pbc_eval_ao_readback` 34 → 4-5 calls).
* The T4 beats the local 16-core CPU on the band step (1.76×) and the same
  VM's CPU by 11×; it does NOT beat the 16-core CPU on the SCF (87.6 s vs
  19.6 s) — the SCF's J/K, FFT and eigensolve stay host-heavy.
* Per T4 `get_bands` call: fused AO kernel `pbc_eval_ao_exec` ~4.2 s,
  BAND-08 contraction `pbc_band_vmat` ~2.1 s. The contraction launches only
  `nk · nao² = 24 057` lanes, each walking 42 875 points serially — low
  occupancy on a 2 560-core GPU. Next GPU lever: split the grid across
  lanes (blocked partial sums reduced in a second pass, on the GPU arm only,
  so the CPU arm keeps its serial order) and then the AO kernel's per-lane
  locals.

## 4b. BAND-09 — contraction occupancy (Kaggle T4, 2026-09-24, notebook v5)

On the GPU the BAND-08 contraction had two occupancy defects: every output
element's 42 875-point sum ran serially on one thread, and because the
resident accumulator is point-major (and Γ is on the path) it launched once
per k-point with only `nao² = 729` threads. BAND-09
(`crates/pyscf-kernels/src/pbc/band_vmat.rs`), GPU only:

* the point-major table is transposed to k-major in ONE launch;
* each output's grid sum is split over `nsplit` lanes, lane `s` summing
  `g = s, s + nsplit, …`, so adjacent lanes load adjacent grid points;
* a second kernel folds the partials in a fixed order (deterministic).

`nsplit` is `1` on the CPU runtime (the kernel compiles to the old serial
shape: local checksums unchanged to the last bit), a measured default on a
GPU, and pinnable with `PYSCF_PBC_BAND_VMAT_SPLIT`. Gates:
`pyscf-kernels/tests/pbc_band_vmat_split.rs` (split 2..4096 vs unsplit,
bitwise-deterministic reruns, Γ and non-Γ) and the split arm of
`band_vmat_device.rs` (2.1e-15 Ha vs host, PBE and LDA).

All CUDA arms read the CPU arm's converged density:

| arm | `get_bands` warm | contraction (30 blocks) | band diff vs CPU |
|---|---|---|---|
| CUDA, BAND-08 unsplit | 7.24-7.31 s | 6 376 ms | 5.8e-14 Ha |
| CUDA, split 256 (first auto) | 5.46 s | 1 339 ms | 1.3e-14 Ha |
| CUDA, split 64 | 5.45 s | 1 024 ms | 9.4e-15 Ha |
| CUDA, split 32 | 5.41-5.46 s | 1 128 ms | 1.4e-14 Ha |
| CUDA, host contraction | 9.29-9.50 s | — | 1.1e-14 Ha |
| Kaggle 4-vCPU, host | 72.6-74.4 s | — | 0 |

Contraction 6.2× faster, `get_bands` 1.33× (7.3 → 5.45 s), 1.7× over the
host contraction, 2.4× over the local 16-core CPU (13.2 s). GPU memory
unchanged (5.97 GB). The default was retuned from this sweep to pick 64 for
this shape (`SPLIT_TARGET_LANES = 1 << 16`, floor 32); the non-Γ path
(one launch over every k) is reasoned from the same rule, not measured.

The T4 band step is now ~4.2 s fused AO kernel (`pbc_eval_ao_exec`) + ~0.34 s
contraction + ~0.9 s host stages: the AO kernel is the next GPU lever.

## 4c. BAND-10 — the AO kernel on the T4 (notebooks v6-v8, 2026-09-24)

Shape of the resident AO launch in a `get_bands` block (new
`PYSCF_PBC_AO_STATS=1` diagnostic): ~365 of the 1 237 lattice images staged
after the W-09 screen, 78 % of image × block pairs kept, `Q = 20` values for
the widest shell (Ta d, with gradients), 33 k-points.

**Refuted — pairing `L` with `-L`** (`e^{ik·(-L)} = conj(e^{ik·L})`, half
the fold multiply-adds). Only 57 % of staged images have their partner
staged (the image list is not symmetric), and on the T4 the doubled value
buffer and branching made the AO kernel SLOWER: 12.7 s → 17.2 s per 30
blocks. Removed. On the 16-core CPU it changed nothing (14.7 vs 15.0 s).

**Refuted as the bottleneck — accumulator size.** Shrinking the k-tile
(`PYSCF_PBC_AO_RESIDENT_NKT`) made the kernel much slower (tile 12: 21.2 s,
tile 4: 44.2 s): each tile re-evaluates every image, ~3.9 s per extra tile.
Image evaluation dominates, not the accumulator's local-memory footprint.
The batched fused kernel (`PYSCF_PBC_AO_RESIDENT=0`) was 23.5 s.

**Kept — one k-tile on a GPU.** At the CPU's 512-slot accumulator the
`Q · nk = 660` fold needs two tiles, so every image was evaluated twice.
The accumulator cap is now a compile-time kernel parameter: 512 on the CPU
runtime (unchanged), and on a GPU the smallest power of two ≥ `Q · nkpts`
within [512, 2048] (`resident_acc_cap`, pin with
`PYSCF_PBC_AO_RESIDENT_ACC`). Tiling never changes a `(q, k)` chain: the
bands are bitwise identical across caps (gate in `band_vmat_device.rs`:
one tile vs four tiles, `to_bits`).

| T4, CPU's density | AO kernel (30 blocks) | `get_bands` warm | GPU mem |
|---|---|---|---|
| cap 512 (two tiles, previous) | 12.6-12.8 s | 5.56-5.69 s | 5.97 GB |
| cap 1024 (one tile) = adaptive default | 10.25-10.6 s | 4.77-4.92 s | ~6.0-6.6 GB |
| cap 2048 (one tile) | 10.24 s | 4.79 s | 7.87 GB |

End to end on the T4 (own SCF, then bands): SCF 73-77 s, `get_bands`
4.7-5.4 s. Running total for this workload, warm `get_bands`:
host contraction 9.4 s → BAND-08 7.3 s → BAND-09 5.45 s → BAND-10 4.9 s
(1.9× overall on the GPU; 2.7× faster than the local 16-core CPU's 13.2 s).

Two observations for the next pass. The CUDA SCF lands on one of two
densities run to run (bands 1.3e-14 or 1.03e-6 from the CPU's, identical
`e_tot` to 15 digits) — some GPU reduction in the SCF is not
order-deterministic; bands must keep being gated on a shared density. And
the AO kernel is still ~3.5 s of the 4.9 s call, now almost all image
evaluation: the next GPU lever is evaluating each image's primitives once
per grid point for all shells on an atom (shared exponentials), or a
shared-memory cooperative layout, not the fold.

## 4d. BAND-11/12 — image evaluation and the fold (T4 notebooks v9-v13, 2026-09-24)

**BAND-11 (kept, neutral on the T4).** The value routines behind the fused
and resident AO kernels (`eval_gto_{general,deriv1}_values`) recomputed every
Cartesian monomial and its derivatives once per spherical component (5× for
d, 3× for p) and multiplied by every zero cart→sph coefficient. Now each
Cartesian term is evaluated once and scattered, zero coefficients skipped.
Per output slot the sum is still `ci` ascending from `+0.0` (adding `±0.0`
never changes such a sum), so it is bit-identical: the fused-vs-reference
gate `eval_ao_image_batch` passes and the KTaO3 band checksums are unchanged
to the last digit. T4: AO kernel 10.6 → 10.9-11.2 s, i.e. no gain — the
arithmetic it removed was not the GPU's cost.

**Attribution** (`PYSCF_PBC_AO_EXP=none`, timing-only): the exponentials
are 15-23 % of the AO kernel; the rest was the k-fold — ~170 GFLOP per
`get_bands` at about a quarter of T4 FP64 peak, because each multiply-add
pair loaded one value and two phases.

**BAND-12 (kept).** `resident_fold_blocked`: a `4 × 4` block of `(q, k)`
accumulators in unrolled (register) arrays across the staged images, so a
loaded value feeds four k-points and a loaded phase four components. Each
accumulator still adds image by image in ascending order from the same
start value — bit-identical (gates pass; KTaO3 bands byte-identical).

| run (same density) | AO kernel, 30 blocks | `get_bands` warm | bands vs CPU |
|---|---|---|---|
| T4 before (BAND-11) | 11.0-11.2 s | 4.98-5.14 s | 9.4e-15 Ha |
| **T4 BAND-12** | **6.4-6.8 s** | **3.27-3.49 s** | 9.4e-15 Ha |
| T4 BAND-12, no exp (timing only) | 5.6 s | 3.06-3.16 s | — |
| 16-core CPU, host route, before → after | — | 12.2 → 9.2 s | bitwise same |

The blocked fold also speeds the SCF-side AO passes (`nr_xc` stage 8.3 →
6.5 s on the T4) and the CPU runtime (−25 %). T4 end to end: SCF 74 s,
`get_bands` 3.6 s.

Running total, warm `get_bands` on the T4 for this workload:
host contraction 9.4 s → BAND-08 7.3 → BAND-09 5.45 → BAND-10 4.9 →
BAND-12 **3.3-3.5 s** (2.8× over the host-contraction start; 2.7× faster
than the 16-core CPU's own new 9.2 s).

Process note: twice a kernel pushed after `kaggle datasets status` said
"ready" mounted the PREVIOUS dataset version, once after a 503 on the
upload. The notebook now asserts the expected SHA-256 of the binary.

## 4e. SCF on the T4 (notebooks v14-v16, 2026-09-24)

SCF stage spans were added (`scf_*` in `pyscf-pbc-scf/src/kscf.rs`,
`veff_nr_rks`/`veff_get_jk` in `krks.rs`, `nr_eval_ao/rho/xc/vxc_mat` in
`numint.rs`) and `krks_profile bands` now records `scf_spans`.

Where the SCF goes (KTaO3, 9 cycles, e_tot −134.6216686224864 Ha on every
arm and backend):

| stage | 16-core CPU | T4 (host routes) |
|---|---|---|
| total SCF | 20.1 s | 77.7 s |
| `get_ovlp` (once) | 4.7 s | 23.9 s |
| `get_hcore` (once) | 6.5 s | 25.2 s |
| `get_veff` (10×) | 8.4 s | 27.4 s |
| └ XC grid loop | 5.7 s | 14.5 s (rho 5.9, vxc 7.4) |
| └ Coulomb `get_jk` | 2.7 s | 12.8 s |
| `eig` (9×) | 0.36 s | 1.1 s |

The T4's SCF is host-bound: the Kaggle VM has 4 slow vCPUs and these
stages' contractions run there.

What was tried (all gated: `pyscf-kernels/tests/pbc_rho_k.rs` bitwise,
`pyscf-pbc-dft/tests/numint_device.rs` Si PBE/LDA SCF host vs device
|Δe_tot| ≤ 8.9e-16, bands on one density ≤ 1.2e-14):

* **Coulomb build on the device — kept.** `get_j_kpts`'s density
  (`pyscf_kernels::pbc::rho_k`, new) and `Σ_g conj(ao) vR ao` (BAND-08's
  `band_vmat`), Hermitian case, `PYSCF_PBC_FFTJK_DEVICE` (unset = on with
  hardware planes). T4: **12.8 → 5.0 s**.
* **XC density and `_vxc_mat` on the device — opt-in only.** T4: XC loop
  14.5 → 37.4 s. The SCF's AO table is cached on the HOST, so every call
  re-uploads the ~600 MB deriv-1 table per k-point. Default off
  (`PYSCF_PBC_NUMINT_DEVICE=1` to force) until the table is device-resident.
* **Compiled-kernel cache — kept in the notebook, small effect.** A
  per-binary-hash `cubecl.toml` (`cache = "local"`) works (1.6 MB of PTX
  reused), but warm vs cold moved `get_ovlp` only 26.7 → 20.9 s: the one-time
  cost is not CUDA JIT.
* **Integrals on CUDA (`CINTX_BACKEND=cuda`, cintx-cubecl `cuda` feature
  forwarded) — no effect.** `get_ovlp`/`get_hcore` unchanged (21.2 vs 21.5 s,
  22.8 vs 22.7 s): the lattice-summed 1e integrals do not go through the path
  that variable selects; their cost is host CPU (4.7 s / 6.5 s on 16 cores).

Net for the T4 default after this pass: Coulomb on device, XC on host —
estimated SCF ≈ 70 s (host-route arm minus the measured Coulomb saving; this
combination was not run as its own arm). The GPU still loses the SCF to the
16-core CPU; the levers left are a device-resident SCF AO table (then the XC
routes win too) and moving the 1e lattice sums off the host.

## 5. Review

The opencode `muse-spark-1.3-contributor-free` plan agent reviewed the diff
twice (`.yta_bundle/band08_review.md`, `band08_review2.md`). First pass
found four real defects, all fixed: case-sensitive env parsing ("False"
enabled the device route), in-place zeroing of a borrowed accumulator,
silent `zip` truncation on an empty band list, unchecked weight lengths.
