# 16-13 — `KCIS`. COMPLETE 2026-09-06. **Davidson roots match upstream to 5.5e-10.**

`crates/pyscf-pbc-ci/src/kcis_rhf.rs`, gated by `tests/oracle_kcis.rs`.

## Task 1 — the deferral, written FIRST

`pbc/ci/cisd.py` is **DEFERRED EXPLICITLY**, recorded in three places so it
cannot be lost: `crates/pyscf-pbc-ci/src/lib.rs`'s module doc, this summary,
and `ROADMAP.md`'s carry-over list. The reason, from `16-CONTEXT §1.6`:
`§8.8` pairs `KCIS` with `pbc/ci/cisd.py`, but they are unrelated modules —
`kcis_rhf.py` is a real k-point CIS while `pbc/ci/cisd.py` (116 l) is a
**Γ-only shim** (`:24`, `:47` refuse `kpt != 0`) over molecular
`cisd.RCISD`/`ucisd.UCISD`/`gcisd`, and **this port has no molecular CI crate
at all**. Porting it means porting molecular RCISD/UCISD/GCISD first, which is
a phase, not a task. This is 17-09's discipline applied before the work rather
than after it.

## Measured (diamond `gth-szv` `[1,1,2]`, mesh `[15,15,15]`, both `kshift`)

| quantity | kshift 0 | kshift 1 |
|---|---|---|
| CIS diagonal vs upstream | `1.434e-9` | `3.055e-9` |
| **DENSE roots vs upstream's dense** | `1.838e-9` | `2.372e-9` |
| **DAVIDSON roots vs upstream's Davidson (G7)** | **`5.484e-10`** | `2.372e-9` |
| Davidson-vs-dense spread, this port | `2.5058421861e-3` | `9.99e-16` |
| Davidson-vs-dense spread, UPSTREAM | `2.5058408969e-3` | `1.05e-15` |
| the two spreads agree to | **`1.289e-9`** | `5.55e-17` |

### The gate is on the DAVIDSON-vs-DAVIDSON comparison, and the spread is asserted separately

At `kshift = 0` **upstream's own Davidson and dense paths differ by
`2.51e-3`** on the third root: the Davidson converges to a different state.
A tighter gate on a Davidson root would be measuring the solver's luck. What IS
assertable — and is a far stronger statement than either root comparison alone
— is that **this port reproduces the spread to `1.29e-9`**: the two
implementations agree on WHICH state the Davidson finds, not merely on the ones
the dense solve finds. Both are asserted.

## Task 2 — what shipped

`get_kconserv_r` (`kcis_rhf.py:428-450`, `kconserv[:, kshift, 0]`),
`vector_size`, `get_init_guess`, `cis_diag`, `cis_matvec` (the
`cis.direct == False` branch), `cis_h_from_matvec` and `kernel_at_kshift`.

**`_CIS_ERIS` is not reimplemented.** Upstream's own TODO at `:455` says
"Merge this with `kccsd_rhf._ERIS`", and its `ovov`/`voov` come from the same
`symm_map` orbit loop with the same `[kp, kr, kq]` indexing and the same
`transpose(0,2,1,3)`; this port takes a `pyscf_pbc_cc::KEris` and reads those
two blocks, so the exxdiv/Madelung treatment is shared rather than duplicated.
`_adjust_occ`, the padding surface, `kconserv` and the Davidson likewise come
from where they already live.

**`epsilons` is the FOCK DIAGONAL, not `mo_energy`** (`:158`, `:300`). They
differ by the Madelung shift `_adjust_occ` puts on the occupied block; using
`mo_energy` would move every root by the Madelung constant and nothing but an
oracle would catch it. The module doc says so at the top.

## Task 3 — both solver paths

The Davidson (16-03's `davidson_nosym1` with upstream's own preconditioner
`r / (e0 - diag + 1e-12)` and the default `pick_real_eigs`) and the DENSE
`np.linalg.eig` fallback on the explicitly built `H`. `cis.davidson` defaults
to `true` and the knob is kept. The dense path is what the Davidson is gated
against, exactly as `kccsd_t_rhf_slow` is for the blocked (T).

The Davidson's `aop` cannot return a `Result`, so a matvec failure is captured
and re-raised after the solve rather than swallowed into a zero vector — a
silent zero would be a wrong number.

### Deviation — the emitted diagonal is COMPLEX

`cis_diag` returns a complex array upstream (`dtype = eris.dtype`, `:302`) even
though every entry is real, so the emitted block is INTERLEAVED. Reading it as
real is a silent factor-of-two index shift; it cost one debug cycle and the
test now says so where the read happens.

## Not shipped

* **`pbc/ci/cisd.py`** — deferred, above.
* **Task 4 test 5, the `dimension == 2` refusal on `graphene`.**
  `check_dimension_for_direct_df` ships with the upstream line
  (`kcis_rhf.py:637`) and upstream's own reason in the payload, but it is only
  reachable on the `cis.direct = True` + GDF path, which this port does not
  ship (the `direct` branch needs `Lpq_mo`, `_init_cis_df_eris`). The refusal
  and the missing branch are one item, not two.
* **Task 4 test 3, the `nkpts = 1` reduction against a molecular CIS** — this
  port has no molecular CIS.
* **The `cis.direct = True` (GDF three-centre) matvec** (`:174-186`,
  `:311-315`) — the four-centre path ships and is what every `§9.2` fixture
  uses; the direct one is a memory optimisation for large cells.
