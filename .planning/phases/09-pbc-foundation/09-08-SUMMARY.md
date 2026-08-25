---
phase: 09-pbc-foundation
plan: 08
type: summary
wave: 4
status: complete
completed: 2026-08-26
requirements: [PBC-GTO-06]
---

# 09-08 SUMMARY — Ewald summation (K-05, K-06)

## What shipped

### `crates/pyscf-kernels/src/pbc/ewald.rs` — K-05 and K-06

Both `#[cube(launch_unchecked)]`, generic over `F: Float`, launched through
`dispatch_backend!` (RULE 5 / AGENTS.md §3). Registered in
`pbc/mod.rs` and re-exported from `pyscf_kernels`.

| kernel | upstream | what runs on device |
|---|---|---|
| **K-05** `ewald_rlij` | `cell.py:729-732` | the `(nL, natm, natm)` C-order table of pair distances `r = \|R_i - R_j + L\|` |
| **K-06** `ewald_gs_terms` | `cell.py:753-770` | `term[g] = \|ZSI[g]\|² · exp(-absG2/4η²) · (4π/absG2) · weights` |

Both reduce on the HOST with `oracle_sum`, so the answer is bit-deterministic
(§9.3) rather than depending on a device reduction tree.

### `crates/pyscf-pbc-gto/src/ewald.rs`

* `get_ewald_params(cell, precision, mesh)` — `cell.py:650-694`, **all four
  branches** (they are pure parameter algebra with no grid behind them).
* `ewald(cell, ew_eta, ew_cut)` — `cell.py:696-822`, the `dimension == 3` path.
* `ewald_real_space`, `ewald_self`, `ewald_g_space` — the three terms, public so
  plan 18 (periodic gradients) can reuse them.
* `Cell::get_ewald_params`, `Cell::ewald`, `Cell::energy_nuc` (`cell.py:824`).

### `crates/pyscf-pbc-gto/src/ewald_pme.rs`

* `bspline_value` / `bspline_grad` / `bspline` — `ewald_methods.py:32-78`, in
  full, including the Euler exponential-spline coefficients (planar `b_re` /
  `b_im`, RULE 8) and the odd-order/even-grid Nyquist zeroing.
* `get_ewald_direct` — `ewald_methods.py:80-99`, i.e. the C loop
  `pyscf/lib/pbc/cell.c:get_ewald_direct` (the SCREENED real-space sum).
* `pme_charge_mesh` — the `Q` mesh of `ewald_methods.py:155-159`.
* `particle_mesh_ewald` — everything up to the FFT, then a typed deferral.

### `crates/pyscf-pbc-gto/tests/ewald.rs` + `tests/common/ewald_reference.rs`

22 tests: 16 tier-1 invariants and 6 tier-2 upstream-pinned. Every literal lives
in `ewald_reference.rs` beside the exact snippet that produced it (D-PBC-19).

## Green test command

```
cargo test -p pyscf-pbc-gto --test ewald      # 22 passed
cargo test -p pyscf-pbc-tools                 # 30 passed
cargo clippy -p pyscf-pbc-gto -p pyscf-kernels --all-targets -- -D warnings   # clean
cargo build --workspace                       # clean
cargo run -p xtask --bin check-dependency-wall # PASS
cargo run -p xtask --bin check-forbidden-paths # PASS
```

## Numeric acceptance

`cell.ewald()` vs upstream PySCF 2.12.1, Bohr-specified §9.2 systems,
**tolerance 1e-9 Ha**:

| system | upstream `ewald()` (Ha) | status |
|---|---|---|
| diamond | -28.771040577654524 | PASS |
| si | -102.88216217333321 | PASS |
| lif | -30.95510482656236 | PASS |
| he_fcc | -1.6174696832216189 | PASS |
| graphene | -44.57202102404764 | deferred (dimension = 2 → plan 12-08) |

`ew_eta` / `ew_cut` match on all five to 1e-12 / 1e-10; `len(get_lattice_Ls(rcut=ew_cut))`
and the internal `cutoff_to_mesh` match EXACTLY on all five.

`ew_eta`-invariance over `[0.5η₀, 2η₀]`: observed spread **< 1e-13 Ha** against a
1e-8 gate.

## Deviations from the plan text (RULE 2 — corrected against the Python)

1. **The plan's `ew_eta`-invariance test is under-specified and fails as
   literally written.** Holding `ew_cut` fixed while scaling `ew_eta` is not a
   physical invariance: weaker screening needs a longer real-space tail.
   Upstream ITSELF drifts **8.1e-7 Ha** at `0.5·η₀` under that recipe, against
   the plan's 1e-8 gate. The test therefore recomputes
   `ew_cut = _estimate_rcut(η², 0, 1., precision)` for each `η`, exactly as
   `get_ewald_params` derives it — which is what "invariant to `ew_eta`" has to
   mean. Both upstream and this port then agree to 3.4e-13 / < 1e-13.

2. **The plan's reference snippet uses `pseudo='gth-pade'`; the committed
   references do not.** Upstream `cell.atom_charges()` returns the VALENCE charge
   when `pseudo=` is set (C → 4), while this port records the pseudopotential as
   a name only until plan 10-01 (D-PBC-11) and still returns the all-electron
   `Z`. Using the plan's snippet verbatim would have pinned -12.787129 against a
   port that legitimately computes -28.771041. References are generated WITHOUT
   `pseudo=`; the `gth-pade` numbers are recorded in
   `ewald_reference::PSEUDISED_EWALD` as plan 10-01's target.

3. **Tier-2 cells are specified in BOHR, not Angstrom.** `pyscf_core::Unit::Ang`
   is CODATA-2014 and upstream is CODATA-2010 — the 4.951e-9 relative lattice gap
   of plan 09-03. Ewald scales as `1/length`, so an Angstrom-specified diamond
   is 1.4e-7 Ha from upstream, two orders above this plan's 1e-9 Ha gate. Same
   resolution plan 09-07 used for `make_kpts`. The §9.2 Angstrom systems are
   swept in a separate test at the tolerance the unit gap implies, so the
   conversion path stays covered.

4. **`erfc` runs on the host.** The plan offered a choice and marked this
   "preferred"; recording it as required. cubecl's `Float` has no `erfc`, and the
   Abramowitz-Stegun 7.1.26 rational form the plan mentions as a fallback is
   ~1.5e-7 accurate — two orders too coarse for a 1e-9 Ha gate. `libm::erfc`
   (FDLIBM, < 1 ulp) is used instead. **New workspace dependency: `libm = "0.2"`**
   — no-std, no transitive deps, already present in `Cargo.lock`; declared in
   `pyscf-pbc-gto` only. `check-dependency-wall` passes.

5. **`4π` and the `1e200` sentinel ride in an `Array<F>`, not as kernel
   literals.** cubecl's `F::new` takes an `f32`: `1e200` would become `inf` and
   `4π` would lose 29 mantissa bits — on its own enough to blow the 1e-9 Ha gate.
   `ScalarArg::new` is not public in cubecl 0.10.0 (same finding as
   `eval_gto.rs`), so the scalars are uploaded as a 3- or 4-element buffer, the
   idiom `gv_kernel` already uses for its 3×3 `b` matrix.

6. **`_bspline` uses `powf`, not `powi`.** numpy's `**` is a correctly-rounded
   `pow`; `powi`'s repeated multiplication loses ~3 ulp at order 10, visible as a
   5.1e-12 partition-of-unity drift versus upstream's 1.7e-12.

7. **`cell.a is None` has no analogue in this port.** `Cell::a` is a plain
   `[[f64; 3]; 3]`, so `ewald` falls back to `Mole::enuc()` on a DEGENERATE
   lattice (`det(a) == 0`) instead. Pinned by
   `degenerate_lattice_falls_back_to_the_molecular_nuclear_repulsion`.

## Deferred branches (D-PBC-20 — typed errors, never a wrong number)

| branch | upstream | error |
|---|---|---|
| `dimension == 2` truncated Coulomb | `cell.py:773-800` | `NotYetImplemented { phase: 12 }` → plan 12-08 |
| `inf_vacuum` G-space (`dimension <= 2`) | `cell.py:558-578` via `get_Gv_weights` | `NotYetImplemented { phase: 12 }` |
| `particle_mesh_ewald` G-space | `ewald_methods.py:171-173` | `NotYetImplemented { phase: 11 }` — needs the 3-D FFT of plan 11-01 |
| `dimension == 0` truncated Coulomb | `cell.py:802-808` | `InvalidMolecule` — upstream raises here too |

Each has a test asserting the exact phase number.

## Carry-overs

* **Plan 10-01** — once the GTH parser lands, `Cell::atom_charges()` returns
  valence charges and every Ewald number shifts to
  `ewald_reference::PSEUDISED_EWALD`. Those tests will need re-pinning, not
  re-deriving; the targets are already committed.
* **Plan 11-01** — `particle_mesh_ewald` needs only `fft`/`ifft`; `Q`, `B`, `C`,
  `ewovrl` and `ewself` are already computed inside it.
* **Plan 12-08** — the `dimension == 2` branch. Its target
  (-44.57202102404764 for graphene) is committed in `EWALD_REFERENCES`.
* **Plan 18** — periodic gradients need `ewald_nuc_grad` /
  `get_ewald_direct_nuc_grad` (`ewald_methods.py:178-292`), not in this plan's
  scope. `ewald_real_space` / `ewald_self` / `ewald_g_space` are public so that
  plan can build on them.
