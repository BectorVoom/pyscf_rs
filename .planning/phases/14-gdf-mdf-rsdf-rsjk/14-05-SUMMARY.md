# 14-05 — `df_ao2mo` + `outcore`. **Phase 13's `ao2mo_7d` carry-over is CLOSED.**

**Status:** shipped, green. 15 tests in `crates/pyscf-pbc-df/tests/df_ao2mo.rs`
(11 unconditional, 4 oracle/acceptance).

## The contract Phase 15 was blocked on, written out

```text
eri[ki, kj, kk][i, j, k, l]      shape (nkpts, nkpts, nkpts, nmoi, nmoj, nmok, nmol)
kl = kconserv[ki, kj, kk]        pyscf_pbc_lib::kpts_helper::get_kconserv
```

Element `[ki,kj,kk][i,j,k,l]` is `(i^{ki} j^{kj} | k^{kk} l^{kl})` in **chemists'
notation**: the first index of each pair is conjugated, the second is not.
`k_i − k_j + k_k − k_l = 0`, i.e. exactly the quadruple `get_eri`'s
momentum-conservation test accepts.

**Phase 15's KMP2 reads it as `eri[ki, ka, kj][i, a, j, b]` = `(ia|jb)` with
`kb = kconserv[ki, ka, kj]`** (`PBC-MASTER-PLAN.md` §8.7). That is the same
table under two index namings — no re-ordering, no transpose. The master plan's
`kconserv[ki, ka, kj]` and this module's `kconserv[ki, kj, kk]` are the same
call.

Settled against upstream, which writes `kl = kconserv[ki,kj,kk]` in all three of
`df_ao2mo.py:210-275`, `fft_ao2mo.py:344-428` and `aft_ao2mo.py:294-…`;
asserted in `ao2mo_7d_index_order_is_the_phase_15_contract` with **four
different `nmo`s** (2/3/1/4) so no index permutation can pass by accident, over
every `(ki,kj,kk)` of a 2×2×2 mesh, at **< 1e-13**.

## What shipped

* `crates/pyscf-pbc-df/src/df_ao2mo.rs` — `get_eri` (all four upstream
  branches), `general` (all four), `ao2mo_7d`, `r_e2`, `warn_pbc2d_eri`,
  `MoCoeff` / `PairDims` / `Eri` / `Eri7d`.
* `crates/pyscf-pbc-df/src/pbc_ao2mo.rs` — **the 13-06 carry-over**:
  `general` and `ao2mo_7d` for FFTDF and AFTDF, `get_mo_pairs_g`,
  `transform_ao_eri`.
* `crates/pyscf-pbc-df/src/outcore.rs` — `balance_segs`, `aux_e1`, `aux_e2`,
  `Aux3cFile`, `Blocking`.
* `Gdf::with_cderi` / `Gdf::load_cderi` — upstream's `mydf._cderi = <…>`, which
  `df.py:253-289` honours by skipping the build.

## `r_e2` conjugates the BRA only, and that had to be measured

`pyscf/lib/ao2mo/r_ao2mo.c:AO2MOmmm_r_iltj` carries **two comments that both say
`^*`** — `C_pi^* (pq| = (iq|` and `C_qj^* (iq| = (ij|` — but its arithmetic
conjugates the bra alone. Measured directly against `_ao2mo.r_e2` on random
complex data:

| convention | max deviation |
|---|---|
| **conj(bra) only** | **2.512e-15** |
| conj(both) | 12.227 |
| conj(neither) | 10.593 |

Every MO transform in the phase goes through one function so the convention is
stated once.

## Gate 1 — the oracle, and what it took to make it a real 1e-11

| quantity | vs upstream |
|---|---|
| `df_ao2mo.get_eri`, He-fcc `[2,1,1]`, three quadruples | **1.667e-12** |
| `df_ao2mo.ao2mo_7d`, He-fcc `[2,1,1]`, complex MOs | **1.984e-12** |
| the same `get_eri` vs upstream's DEFAULT screening | 2.750e-09 |
| **port `get_eri` vs UPSTREAM `get_eri`, both over the PORT's own `cderi`** | **1.110e-16** |

The last row is the attribution device, and it is the one worth keeping: it runs
upstream's actual `df_ao2mo.get_eri` over a stub `mydf` whose `sr_loop` yields
this port's blocks. **1.110e-16 is one ulp at `|eri| ~ 0.5`** — the two sides
run the same contraction over the same inputs and differ only in SUMMATION
ORDER (this port reduces sequentially over `L`; upstream calls BLAS `ddot`).
So "is the contraction upstream's?" and "is the `cderi` upstream's?" are
separated permanently, and a future `cderi` regression cannot be mistaken for a
`df_ao2mo` regression.

## THE FINDING: upstream's `ExtendedMole.strip_basis` is worth 1.054e-09 in `j3c`, and 14-02's gate could not see it

The first oracle run came in at **2.750e-09**, 275× the 1e-11 target. Localising
it took six measurements and every one of them mattered:

| stage | port vs upstream |
|---|---|
| `weighted_ft_ao` (model-charge FT × coulG) | **6.939e-18** |
| `auxbar` | 2.776e-17 |
| `int1e_ovlp` | 2.887e-15 |
| `ft_aopair` at the GDF mesh | 2.001e-11 (predicted `j3c` contribution 2.255e-13) |
| Cholesky factor `L` | 1.068e-13 |
| real-space `fuse(j3c)` at BOTH `incore` and `gdf_builder` `rcut` | 1.448e-12 / 1.451e-12 |
| **fused `j3c` AFTER `add_ft_j3c`** | **1.054e-09** ← |
| `cderi` | 6.745e-09 |

Every input matched and the assembly did not. The cause is not in this port at
all: `_CCGDFBuilder.build` (`gdf_builder.py:116-121`) takes a **per-shell-pair**
radius array from `estimate_rcut` and hands it to
`ft_ao.ExtendedMole.strip_basis(rcut)`, which then drops images pair by pair.
This port has no `ExtendedMole` (D-PBC-21, extended by D-PBC-23) and evaluates
every shell pair out to the maximum radius. Flattening upstream's radius array
to its own maximum — which is what this port does — collapses the gap:

| upstream screening | max\|Δ j3c\| |
|---|---|
| `strip_basis`, per-pair (upstream default) | **1.054e-09** |
| uniform at `rcut.max()` (this port) | **7.333e-13** |

**The port is the MORE converged of the two**: it keeps images upstream
discards. 14-02's own `fuse(j3c)` gate could not have seen this — it compared
against a standalone `incore.Int3cBuilder`, which strips nothing.

With all three substitutions in place (`exclude_dd_block = False`,
`direct_scf_tol = 1e-14`, uniform `estimate_rcut`) the gate is a real 1e-11 and
passes at 1.667e-12. The default-screening number is recorded beside it as a
priced upper bound. All three substitutions live in the test's oracle script, so
the attribution cannot rot.

## Stated deviations

1. **`outcore` blocks over the k-point pair, not the auxiliary shell.**
   Upstream partitions auxiliary shell ranges and calls `wrap_int3c` with a
   `shls_slice`. This port's `aux_e2` builds its cintx `BasisSet` from
   `cell.mol._atom` / `_basis` — the per-ELEMENT parsed basis — so an arbitrary
   auxiliary shell range is not expressible without synthesising per-atom basis
   entries, and 14-02 measured what that costs when it goes wrong (`‖j2c‖` 4495
   against 251.96). `kptij_lst` is already a first-class parameter, it is the
   axis that dominates GDF's footprint (`nkpts²`), and `balance_segs` is ported
   faithfully so the axis is a one-line change at the call site.
2. **FFTDF's and AFTDF's `general` / `ao2mo_7d` transform the ASSEMBLED AO
   block** rather than folding the MO transform inside `pw_loop`. Same
   contraction, different association order; what is given up is memory, and the
   builder that cares about memory is GDF, whose `general` IS factorised through
   `cderi`.

## Numbers for the record

* `ao2mo_7d(FFTDF)` vs `ao2mo_7d(AFTDF)`, diamond `[2,1,1]`, meshes 11/15/21 —
  falls monotonically, stated as a ladder rather than a fixed bound for the same
  reason `tests/pbc_ao2mo.rs::aft_and_fft_eri_converge` is.
  On He-fcc at mesh 15 the same comparison is **1.011e-01**, which is FFTDF's
  aliasing on a steep all-electron 1s and would have been mis-attributed to this
  plan by a naive fixed gate.
* `outcore` `aux_e1` / `aux_e2` reproduce `incore::aux_e2` **bit-identically**
  (0.0) with one k-pair per block.

## For 14-09

* Phase 15's two prerequisites are met: the `ao2mo_7d` index order (this plan)
  and a GDF that KMP2 can read (`sr_loop`, 14-03).
* The `strip_basis` deferral is a NEW carry-over and belongs with D-PBC-23: it
  is the same `_RangeSeparatedCell` / `ExtendedMole` work, and it is now priced
  at 1.054e-09 in `j3c` / 2.750e-09 in the ERI, in addition to D-PBC-23's
  1.835e-08 on the energy.
