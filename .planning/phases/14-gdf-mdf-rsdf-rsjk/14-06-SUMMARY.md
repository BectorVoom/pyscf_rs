# 14-06 — MDF. **Gate 2 met, and the plan's own two premises were both wrong.**

**Status:** shipped, green. 5 tests in `crates/pyscf-pbc-df/tests/mdf.rs`,
3 more (Gate 2, the structural check, the oracle) in
`crates/pyscf-pbc-scf/tests/df_swap.rs` — they drive a converged `KRHF` and
`pyscf-pbc-scf` depends on `pyscf-pbc-df`, so they live there rather than behind
a dev-dependency cycle.

## THE DEFECT: `decompose_j2c` read `zeigh_gen`'s eigenvectors TRANSPOSED, and nothing had ever exercised that branch

`pyscf_algebra::zeigh_gen` returns its eigenvector matrix **COLUMN-MAJOR
(F-order)** — its module docs say so. `gdf_builder::j2c::eigenvalue_decomposed_metric`
read it row-major:

```rust
re[r * n + q] = vectors.re[q * n + i] * s;   // WRONG — F-order needs q + i*n
```

The transpose of an orthogonal matrix is still orthogonal, so the factor had the
right shape, the right rank and the right eigenvalues, and nothing crashed. It
simply built the fitted tensor in the wrong basis.

**No gate in the phase had ever reached the eigen branch.** `j2ctag` is `CD` on
every system in `measurements/params.py` — including diamond, whose metric has
`eig_min = 3.17e-11`, below `linear_dep_threshold`, and which upstream still
decomposes by Cholesky because Cholesky is tried first and succeeds. MDF is the
first consumer of the eigen route (`j2c_eig_always = True`, `mdf.py:365`), and
what it produced was:

| | He-fcc 2×2×2 `E_KRHF` |
|---|---|
| with the transposed factor | **+6 306 866.73** |
| after the fix | **−2.808 485 113** |
| upstream | −2.808 485 114 |

The regression test is
`tests/gdf_builder.rs::the_eigen_factor_inverts_the_metric_on_its_retained_subspace`:
`V j2c Vᴴ = I` on the retained subspace — three matrix products, the identity
that DEFINES the factor, and the only property of it that matters. It runs in
12 s and fails on a transposed, permuted or mis-phased `V`, all of which pass
the shape/rank/tag assertions the previous test made. Measured **2.709e-14**
(He-fcc, rank 23/23) and **3.094e-08** (diamond, rank 105/108, where the
retained spectrum reaches the 1e-10 threshold and the residual is a conditioning
floor, not slack).

## Two more upstream devices that only the eigen route needs

3. **`if self_conj: j2c = j2c.real`** (`rsdf_builder.py:866-868`). For a
   self-conjugate k-difference the metric is real in exact arithmetic and
   carries ~1e-16 of imaginary part in practice. A complex Hermitian
   eigensolver is then free to return each eigenvector with an arbitrary phase
   `e^{iθ}` — and `cderi` is contracted as `Σ_L c_L c_L` with **no conjugate**
   (`df_ao2mo.py:74`, `zdotNN`), so the phase survives as `e^{2iθ}` rather than
   cancelling. Cholesky has no such freedom: its factor is unique with a
   positive real diagonal. Now applied.
4. **The conjugate pass.** `gen_uniq_kpts_groups` (`rsdf_builder.py:851-871`)
   yields TWO entries per non-self-conjugate group — the group at `+kpt`, and
   its time-reverse at `−kpt` with the pairs swapped and the SAME decomposition
   conjugated (`_conj_j2c`) — rather than decomposing `j2c[−k]` independently.
   Upstream says why in a comment: the two decompositions can land on different
   `cderi` dimensions. `make_j3c` now synthesises that second pass. On a 2×2×2
   mesh every k-difference is self-conjugate so this was latent, but it is a
   real gap for any mesh that is not (3×3×3 and up).

## `get_naoaux` was STRICTER than upstream, and wrongly so

Plan 14-03 made `Cderi::naoaux` raise when the per-k-pair ranks disagreed, on
the stated reasoning that "upstream raises rather than silently truncating".
Upstream does no such thing: it opens the file, takes `next(iter(...))` — one
arbitrary block — and returns its leading dimension (`df.py:592-597`). And the
ranks legitimately DO differ per k-difference on the eigen route: **MDF on
He-fcc 2×2×2 at mesh 15 keeps 10 vectors for one group and 11 for another**, and
that is correct, because the auxiliary index is only comparable *within* a
group.

`naoaux` now returns the rank of the **diagonal `(0,0)`** block. That is what
its one real consumer needs: `df_jk::get_j_kpts`'s `rho` accumulator sums over
the diagonal pairs, and every `(k, k)` has `q = 0`, so they share one metric.
The consumers that contract *across* groups — `df_ao2mo`'s two-block branches —
now check the pair they actually use and refuse with an actionable message.

## Both of the plan's premises were wrong, and both are corrected in the tests

### 1. MDF's default mesh is not `[7,7,7]`

`14-06-PLAN.md` Task 4 states it as `[7,7,7]` "measured upstream in
`measurements/params.py`". Measured directly:

| system | MDF default mesh |
|---|---|
| diamond 2×2×2 | **[11,11,11]** |
| diamond gamma | [13,13,13] |
| He-fcc 2×2×2 | **[9,9,9]** |
| He-fcc gamma | [11,11,11] |

Mesh 7 is simply `mdfladder.py`'s lowest rung. Asserted in
`mdf_default_mesh_is_the_builders_own_estimate`.

### 2. `mdfladder.out` measures the WRONG BUILDER

Every row of it was recorded with `df.MDF`'s default, and **`MDF._prefer_ccdf`
is `False`** (`mdf.py:79`) — so the whole table is `_RSMDFBuilder`, which plan
14-07 owns and could not ship. This plan ports `_CCMDFBuilder`, exactly as 14-02
ported `_CCGDFBuilder`. `measurements/mdfladder_cc.py` / `.out` were added and
are what Gate 2 asserts against.

## GATE 2 — MET

He-fcc/`sto-3g` 2×2×2, against `E_KRHF(FFTDF, mesh 31)`:

| builder | upstream (CC) | **the port** |
|---|---|---|
| GDF (no plane waves) | 6.002e-05 | **6.002e-05** |
| MDF mesh 7 | 1.695e-06 | **1.695e-06** |
| MDF mesh 9 (default) | 5.476e-08 | — |
| MDF mesh 11 | 6.684e-09 | **3.433e-09** |
| MDF mesh 15 | 3.216e-08 | **3.245e-08** |

He-fcc gamma, the structural check:

| | upstream (CC) | **the port** |
|---|---|---|
| \|GDF − FFTDF\| | 1.476e-05 | **1.476e-05** |
| \|MDF − FFTDF\| | 8.788e-08 | **8.694e-08** |

MDF beats GDF by **170×** at gamma and by **17 000×** at the 2×2×2 plateau.
That is the phase's whole case for a cross-builder gate: GDF is an
approximation whose error is a property of the auxiliary basis, and MDF is the
builder that legitimately closes it.

**The ladder is not monotone and a monotone gate would fail a correct
implementation.** Two independent floors are in play — MDF's own auxiliary fit
and the mesh-31 truncation of the FFTDF *reference* — and past the crossover the
comparison measures the reference. Upstream bounces the same way (He-fcc: 6.684e-09
at mesh 11, 3.216e-08 at 15, 3.318e-08 at 21) and **so does the port, to within
1 %** (3.245e-08 at mesh 15 against upstream's 3.216e-08) — the bounce is the
FFTDF reference's own mesh-31 truncation showing through, and reproducing it is
stronger evidence than reproducing the descent alone. Gate 2 is therefore stated as:
beat GDF by an order at mesh 7, fall two more orders by mesh 11, beat GDF by
three orders at the plateau, and stay within an order of the plateau afterwards.

## The oracle

`KRHF` on MDF vs upstream `mdf.MDF` with `_prefer_ccdf = True` and the three
screening substitutions 14-05 established:

| system | port | upstream | \|dE\| |
|---|---|---|---|
| He-fcc gamma, mesh 11 | −3.20863596869297 | −3.20863596509184 | **3.601e-09** |
| He-fcc 2×2×2, mesh 9 | −2.80848516472422 | −2.80848516444156 | **2.827e-10** |

Gated at 1e-8 rather than 1e-11, and the reason is measured: MDF's metric is
`<g|g> − <g|G><G|g>` and is deliberately near-singular — smallest RETAINED
eigenvalue **2.464e-08** against a largest of 1.168 on He-fcc at gamma — so
`solve_cderi`'s pseudo-inverse amplifies any `j3c` residual by up to **4.1e7**.
Upstream states the same fragility in its own words at `mdf.py:362-365` ("small
integral errors can lead to a difference in the total energy … around 4th
decimal place"), which is why it abandons Cholesky for MDF. 3.601e-09 is four
orders inside that.

## Composition, not a second implementation

`make_j3c` is ONE driver with a `Scheme` tag (`CompensatedCharge` | `Mixed`),
mirroring upstream's `_CCMDFBuilder(\_CCGDFBuilder)` subclass that overrides four
methods. `Mdf` holds an inner `Gdf` carrying the MDF `cderi` (so 14-04's `df_jk`
and 14-05's `df_ao2mo` are reused unchanged) and an inner `Aftdf` at MDF's mesh
with `mdf_pw_edge_screen` set (so Phase 13's `aft_jk` and `pbc_ao2mo` are
reused unchanged). Nothing here re-implements a contraction.

`Aftdf::mdf_pw_edge_screen` is new and is upstream's: `MDF.weighted_coulG`
(`mdf.py:143-172`) zeroes the plane waves at `±Gmax ± 0.5` at a half-integer
scaled k-point, a screen that was removed from `tools.pbc.get_coulG` (it broke
supercell / k-point consistency) and re-applied inside MDF. On a 2×2×2
Monkhorst-Pack mesh **every** k-difference is half a reciprocal vector, so it is
not an edge case here.
