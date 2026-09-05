# Phase 15 measurements (vendored PySCF 2.12.1)

Every script is run from the workspace root with the vendored tree first on
`PYTHONPATH`:

```bash
PYTHONPATH=$PWD .venv/bin/python -u \
  .planning/phases/15-periodic-ao2mo-kmp2/measurements/<script>.py
```

Each script asserts `pyscf.__version__ == "2.12.1"`. Raw numerical output is
committed beside the generator as `<script>.out`.

## Committed diamond anchor

`anchor.py` reproduces `pyscf/pbc/mp/kmp2.py:795-821`: Bohr diamond,
`gth-szv`/`gth-pade`, `[1,1,2]`, FFTDF, `exxdiv=None`, and `conv_tol=1e-11`.
The observed correlation energy is `-0.20472143304034024`; its residual from
the source-tree constant `-0.204721432828996` is **2.1134424765811843e-10 Ha**.
The separately tagged components are SS `-0.034594521893337379` and OS
`-0.17012691114700285`.

Five fresh single-threaded processes produced
`-0.20472143304034063 Ha` bit-identically (`anchor_repeat.out`, spread 0).

The acceptance tolerance is **2e-6 Ha**. This is deliberately above the
independent Rust/Python SCF-path floor and far below the ~0.1 Ha failures caused
by a wrong exchange-divergence setting or missing k-point normalisation. A
naive `1e-14` gate would reject the upstream program against its own committed
constant.

## K-dependent padding fixture

`padding.py` evaluates upstream's documented ragged example
`nmo=(6,6,5)`, `nocc=(2,3,2)`, including uniform count, uniform list and per-k
frozen forms. `padding.out` records every mask and both split/joint index list.
The dense dimension is `max(nocc)+max(nvir)=7`, not `max(nmo)=6`.

## Two route measurement and speed floor

`routes.py` uses all-electron He/6-31g, `[1,1,2]`, mesh 9 and `exxdiv=None`.
Three kernel-only repetitions measured:

| route | e_corr (Ha) | SS | OS | median kernel (s) | run spread (Ha) |
|---|---:|---:|---:|---:|---:|
| FFTDF / MO-first | -0.033241446759957924 | -1.1861279878419215e-6 | -0.033240260631970081 | 0.019187749 | 1.388e-17 |
| GDF / Lov | -0.016989369077568279 | -3.2102396847681451e-6 | -0.01698615883788351 | 0.003406988 | 0 |

The measured FFTDF/GDF median ratio is **5.6313x** on this deliberately small
fixture. Forcing GDF through its four-index AO2MO path gives
`-0.016989369077568282`, a **3.469e-18 Ha** route residual. This is a
performance measurement, not a claim that FFTDF and GDF should have equal
correlation energies: they are different integral approximations and are gated
against their own upstream values.

## Measurement scope

The committed measurements cover the source-tree anchor, the complete padding
fixture, both KMP2 routes, SS/OS decomposition, route equivalence on one GDF
mean field, and a kernel timing ratio. The much larger five-process matrix over
two systems, two SCF tolerances, two thread counts, four DF builders and four
FFT meshes specified by the planning survey was not used to invent tighter
tolerances; the implemented tests retain the conservative measured gate above.

## Staggered-mesh oracle (`stagger.py`, added 2026-09-05)

`stagger.py` reproduces the fixture committed in `kmp2_stagger.py:353-420` — an
H2 dimer, `gth-szv`/`gth-pade`, `ke_cutoff = 100` (mesh `[29,29,29]`), `2x2x2`
k-points, `exxdiv='ewald'`, `conv_tol = 1e-11` — and prints every number
`crates/pyscf-pbc-mp/tests/kmp2_stagger.rs` gates against. Output in
`stagger.out`.

| quantity | live 2.12.1 | constant in `kmp2_stagger.py` | drift |
|---|---:|---:|---:|
| KRHF `e_tot` | `-1.1004620466064836` | — | — |
| stagger, submesh, FFTDF | `-0.016089900380356827` | `-0.0160902544091997` (`:385`) | `3.540e-7` |
| stagger, full mesh, FFTDF | `-0.014028716824109303` | `-0.0140289970302513` (`:390`) | `2.802e-7` |
| standard KMP2, FFTDF | `-0.014390203713094872` | `-0.0143904878990777` (`:395`) | `2.842e-7` |

**Upstream's own committed constants no longer reproduce**, by `2.8e-7`-`3.5e-7`
— the same shape as the diamond anchor's `2.1e-10` from `kmp2.py:820`. The
tests therefore gate on the live values above, not on the source constants.
`15-VERIFICATION.md §7` makes that a standing caveat.

The script also prints, on a GDF mean field, `stagger_submesh_gdf`
(`-0.015836452346190341`, the `Lov` route) against `stagger_submesh_gdf_ao2mo`
(`-0.015886558478292751`). Those are **`5.01e-5 Ha` apart on purpose**:
`kmp2_stagger.py:73-75` builds a fresh `FFTDF` for the four-index path even on a
GDF mean field, unlike plain `KMP2` (`kmp2.py:92`). The port had to be corrected
to match — see `15-VERIFICATION.md §3.2`.

## Rollup oracle emitter (`oracle_rollup.py`, added 2026-09-05)

One section per invocation, driving `crates/pyscf-pbc-mp/tests/oracle_phase15.rs`:

```bash
PYTHONPATH=$PWD .venv/bin/python -u \
  .planning/phases/15-periodic-ao2mo-kmp2/measurements/oracle_rollup.py <section>
```

`symm_map`, `padding`, `ao2mo7d`, `lov`, `kmp2`, `t2rdm`, `mofirst`. Numeric
blocks are `%.17g` (an exact f64 round-trip) between `BEGIN <name> n=<count>`
and `END <name>`, complex interleaved re/im. Sections that need MO coefficients
emit **upstream's own** alongside the reference values, so a diff measures the
transform rather than two independent SCFs.

Deviations from `15-07-PLAN.md` Task 1, both forced by the fixtures:

* the plan names `he_fcc` `2x2x2` for the `Lov` and MO-first parts; `he_fcc` is
  `gth-szv` on a single He atom, i.e. **one AO**, so `nvir = 0` and the block is
  empty. Those parts use diamond `gth-szv` `[1,1,2]` and He/6-31g instead.
* the MO-first part covers FFTDF over every conserving quadruple and leaves
  AFTDF to the `ao2mo7d` section: one AFT quadruple at diamond's 47^3 mesh ran
  past 19 CPU-minutes without finishing, on both sides.
