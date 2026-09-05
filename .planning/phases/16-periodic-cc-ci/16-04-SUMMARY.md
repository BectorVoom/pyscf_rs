# 16-04 — `kintermediates_rhf`. COMPLETE 2026-09-06.

`crates/pyscf-pbc-cc/src/kintermediates_rhf.rs`, gated by
`tests/oracle_phase16.rs::intermediates_and_update_amps_match_upstream`.

## Task 1 — the block is CLEAR

`crates/pyscf-pbc-mp` exports `padding_k_idx` (`src/padding.rs:35`),
`padded_mo_energy` (`:66`), `padded_mo_coeff` (`:96`), `get_nocc`
(`src/frozen_k.rs:86`), `get_nmo` (`:140`) and `get_frozen_mask` (`:192`), all
shipped by Phase 15 and closed 2026-09-05. **No second padding implementation
was written**, per `16-CONTEXT §1.1` — the convention is occupied-bottom /
virtual-**TOP** aligned (`kmp2.py:262-263`) and two of them is how a plausible
wrong number ships. `PBC-MASTER-PLAN §8.8`'s HARD-BLOCK paragraph is updated to
say the block cleared.

## Tasks 2-3 — what shipped

`cc_Foo`, `cc_Fvv`, `cc_Fov`, `Loo`, `Lvv` (Eqs. 37-41) and `cc_Woooo`,
`cc_Wvvvv`, `cc_Wvoov`, `cc_Wvovo` (Eqs. 42-45). The four `W`s come back as
`KBlocks`, whose tier is chosen from an exact byte count against the caller's
budget — upstream's `kccsd_rhf.py:132-137` / `:179-192` / `:423-455` three-way
branch with `_mem_usage` replaced (D-PBC-29 clause 4).

`cc_Woooo` and `cc_Wvvvv` keep upstream's **two-pass** structure: the mirror
`W[kl,kk,kj] = W[kk,kl,ki].transpose(1,0,3,2)` is written in a SECOND loop
after every member of the first is made (upstream's own comment at `:136`).
Fusing them reads the mirror before it is written.

## The primitive, named once at the source rather than per site

`16-CONTEXT §3.2` requires every contraction to name its primitive. This module
satisfies it structurally: every contraction is an [`einsum`] subscript string
transcribed from the upstream line above it, and **`numpy.einsum` never
conjugates**, so an einsum-transcribed line is `oracle_zsum(Π operands)` — the
unconjugated ordered complex sum — BY CONSTRUCTION. `tests/zarr.rs`'s
`einsum_does_not_conjugate` pins the direction with an assertion
(`Σ x·x = -14` for a purely imaginary `x`, not `+14`), and the places that DO
conjugate are explicit `.conj()` calls at the upstream `.conj()` they came
from. That is a stronger guarantee than a per-site comment, and it is what
`15-REVIEW.md D-15-R-02`'s finding actually asks for.

## Measured against upstream (`measurements/README.md §10`)

Driven from upstream's own mean field, on a FIXED synthetic `t1`/`t2` from a
shared SplitMix64 stream — synthetic rather than converged on purpose, so a
failure names one function instead of "the energy is wrong":

| | vs upstream |
|---|---|
| `cc_Foo` / `cc_Fvv` / `cc_Fov` | `1.13e-8` / `2.52e-8` / `3.51e-9` |
| `Loo` / `Lvv` | `1.21e-8` / `2.52e-8` |
| `cc_Woooo` / `cc_Wvvvv` | `1.55e-8` / `2.28e-7` |
| `cc_Wvoov` / `cc_Wvovo` | `1.44e-7` / `1.13e-7` |

All inside the `1e-6` block gate, and all consistent with the FFT
integral-transform floor the ERI blocks themselves carry (`1.2e-8 … 2.3e-7` at
the pinned `[15,15,15]` mesh). `cc_Wvvvv` is the largest because it inherits
`vvvv`'s.

### Deviation 1 — Task 4's tests 1-5 are not all shipped as specified

Test 1 (the Γ reduction against `pyscf-ccsd`'s molecular `rintermediates`) needs
a molecular ERI object built from a periodic Γ mean field, which is 16-12's
`pbc/cc/ccsd.py` shim and does not exist yet; it is deferred to 16-12 and
listed in `16-VERIFICATION.md`. Tests 2-3 (momentum conservation, the
`check_antisymm_*` symmetries) are subsumed by the element-wise oracle
comparison above, which is strictly stronger. Tests 4-5 (tier equivalence,
determinism) ship in `tests/kccsd_rhf.rs` at the `_ERIS` level, where the tier
branch actually lives.
