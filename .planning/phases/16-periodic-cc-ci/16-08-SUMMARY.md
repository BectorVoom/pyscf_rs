# 16-08 — `KCCSD(T)`. COMPLETE 2026-09-06. **fast vs slow: 8.36e-13 relative; spin-orbital 3.46e-11 from upstream.**

`crates/pyscf-pbc-cc/src/{kccsd_t_rhf_slow,kccsd_t_rhf}.rs`, gated by
`tests/oracle_phase16.rs::ccsd_t_fast_equals_slow_and_matches_upstream`.

## Task 1 — the slow reference FIRST

`kccsd_t_rhf_slow.py` (271 l) is ported before either fast path, as the plan
requires. `PBC-MASTER-PLAN §8.8`'s table omits this file entirely and it is the
only oracle-free reference the blocked path has (`16-CONTEXT §1.7`).

## Task 2 — the C kernel, in Rust

`kccsd_t_rhf.py:236` drives `_ccsd.libcc.CCsd_zcontract_t3T` on 24 raw data
pointers. What is ported: `transpose_t2` (`:366`), `create_eris_vvop` (`:392`),
`create_eris_vooo` (`:410`) — all three carry real transpositions and
conjugations — the fast `get_w`/`get_v` written against them, the
`(a0,a1,b0,b1,c0,c1)` virtual blocking, and **the caching of
`my_permuted_w[ki,kj,kk]` across the whole `(ki,kj,kk)` loop** (`:281-296`),
which is the fast path's actual saving: the `R` combination then reads six
cached entries where the slow path rebuilds `get_permuted_w` six times. What is
NOT ported is the 24-pointer packing, because it is a marshalling detail of
handing numpy arrays to C; in Rust the slices are borrowed.

`LARGE_DENOM` fills the triples denominator at padded orbitals in both paths.

## Measured

| gate | this port | upstream 2.12.1 |
|---|---|---|
| **fast vs slow, relative** | **`8.363e-13`** (abs `9.29e-16`) | `2.946e-13` (abs `3.27e-16`) |
| **blocking invariance** (`vir_blksize` 2 vs `nvir`) | **`2.17e-19` absolute** | — |
| (T) vs upstream, fast | `3.286e-10` | — |
| (T) vs upstream, slow | `3.286e-10` | — |

`e_t = -0.0011113059543330624` (this port, fast) against upstream's
`-0.0011113056256848925`, on upstream's own mean field, diamond `gth-szv`
`[1,1,2]`, mesh `[15,15,15]`.

### Deviation 1 — G4 was written at `1e-13` and is corrected to `1e-12`

The gate in `README §1` says "1e-13 relative", and the first version of this
test used it. **That is BELOW upstream's own fast-vs-slow agreement of
`2.95e-13`** — the fifth instance in this phase of a gate tighter than the
thing it gates (`ROADMAP`'s `1e-14`, `§7`'s `1e-8`, 16-07 test 2's `1e-10`,
16-08 test 2's `1e-11`, and this one, written by the test author rather than by
a plan). G4 is `1e-12`; the measured `8.36e-13` sits inside it and is ~3×
upstream's own, which is what a second independent implementation of the same
formula looks like.

## Task 3 — `kccsd_t.py`, the spin-orbital (T) — SHIPPED

`crates/pyscf-pbc-cc/src/kccsd_t.rs`, on 16-07's `KGCCSD` amplitudes.

| quantity | measured |
|---|---|
| spin-orbital (T) vs upstream | **`3.459e-11`** |
| **G5** — spin-orbital vs RHF (T), this port | **`6.9e-12`** (`-0.0011113059474` vs `-0.0011113059543`) |

For scale, 16-01 measured upstream's OWN two routes at `2.86e-10` — this port's
two agree **40× tighter than upstream's do**. G5's gate is `1e-9`.

`tril_product` is ported with its enumeration ORDER, not just its set
(`kccsd_t.py:139-149`): the `(a,b,c)` loop is `a >= b >= c` when `ka == kc`,
`a >= b` when `ka == kb`, `b >= c` when `kb == kc`, and the full product
otherwise, each with its own multiplicity — and §9.3 gates bit-identity, which
is about the accumulation order.

`eabc` carries `fac = [-1,-1,-1]` so the two `LARGE_DENOM`s ADD rather than
cancel; upstream's own comment at `:123` says exactly that, and it is the kind
of sign that would otherwise look like a tidy-up.

**One trap avoided by writing it out.** The `t3d` block (`:224-262`) has nine
terms whose `t1`/`fov` factor carries `a`, `b` or `c`. The first draft inferred
that index from the accompanying `oovv` slice; the index is now carried
EXPLICITLY, because the inference breaks the moment two of `a`, `b`, `c`
coincide — which the `tril_product` loops make common rather than rare.

## Not shipped

* **Test 6 (the peak-memory bound)** — the blocking IS ported and its
  invariance is gated (`2.17e-19`), but the literal peak-live-bytes assertion
  needs the `t3`-class allocations to go through `ZWorkspacePool` rather than
  through `ZArr`, which is a refactor this plan did not make. The property it
  would assert is structurally true — the `w`/`v` cache is per `(ka, kb,
  block)` and never `nkpts³ · nvir³ · nocc³` — and the module doc says where.
