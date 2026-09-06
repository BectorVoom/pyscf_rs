# 16-06 — `KUCCSD`: unrestricted k-point coupled cluster

**Status: SHIPPED.** `pyscf/pbc/cc/kccsd_uhf.py` (1116 l) and the ground-state
half of `pyscf/pbc/cc/kintermediates_uhf.py` (`:26-588`) are ported.
`16-VERIFICATION.md §6.1` recorded this as *"not reached — the largest single
remaining item, three spin channels (`aa`/`ab`/`bb`) throughout"*.

## What shipped

| File | Contents |
|---|---|
| `crates/pyscf-pbc-cc/src/kueris.rs` (~690 l) | `KuEris`, `UBlk`/`UPass`/`UKind`, `UFock`, the four `oppp` `ao2mo` passes, the three `ao2mo_7d` quads, the unrestricted Fock build |
| `crates/pyscf-pbc-cc/src/kintermediates_uhf.rs` (~880 l) | `make_tau`, `make_tau2`, `cc_fvv`, `cc_foo`, `cc_fov`, `cc_woooo`, `cc_wvvvv_half`, `cc_wovvo`, `kconserv_mat`, and the `b::` block-name table |
| `crates/pyscf-pbc-cc/src/kccsd_uhf.rs` (~1600 l) | `energy`, `init_amps`, the five `update_amps` stages, `kernel`, `Kuccsd` |
| `crates/pyscf-pbc-cc/tests/oracle_kuccsd.rs` | 11 tests (9 oracle-gated) |
| `crates/pyscf-pbc-cc/tests/kuccsd.rs` | the oracle-free zero-amplitude identity |

## Two DEFECTS in already-shipped code, found by this plan

Neither is in the new module. Both were latent in `pyscf-runtime`'s complex
arena and invisible to every earlier fixture.

### D1 — a recycled buffer was handed back with the previous tenant's data

`ZWorkspacePool::reserve` reuses a free-listed buffer "of sufficient capacity"
rather than allocating; `fresh` returns `ZBuffer::zeros`. The two paths
disagreed, so `KTensor::zeros` — whose name is a promise — returned zeros only
when it happened to allocate.

`update_amps` builds `cc_Woooo`, releases it, builds `cc_Wvvvv_half`, releases
it, then builds `cc_Wovvo`, which **accumulates** across many scattered
k-addresses. `Wovvo` came back carrying `Wvvvv`. Every doubles amplitude was
`~1e-2` wrong while all five equation stages were independently exact against
upstream to `1e-11`.

The restricted path never saw it: `kintermediates_rhf` happens to `set` every
block it fills rather than accumulating, so it never read what it was given.

Fixed in `ZWorkspacePool::reserve` (zero on reuse — one memset, the same write
the avoided allocation would have done, so the allocate-once guarantee is
untouched). Regression: `zworkspace_pool.rs::a_recycled_buffer_comes_back_zeroed`.

### D2 — a block read returned the whole buffer, not the block

Because a recycled buffer is `>= block_len`, `KTensor::block` /`with_block` /
`with_block_mut` returned the buffer's full length. Diamond `gth-szv` has
`nocc == nvir == 4`, so `Woooo`'s `nocc⁴` and `Wvvvv`'s `nvir⁴` are both 256 and
the recycled buffer was accidentally exactly right. The first fixture with four
distinct extents (`nocca 2`, `noccb 1`, `nvira 4`, `nvirb 5`) failed on the
first read with a shape error.

Fixed by truncating to `block_len`. Regression:
`ktensor.rs::a_recycled_buffer_is_truncated_to_the_new_block_shape`.

### D3 — spill filenames collided between pools in one process

`pyscf_kcc_zspill_<pid>_<buffer id>.h5`, and the buffer id restarts at 0 for
every `ZWorkspacePool`. Two pools in one process (two `#[test]`s on two
threads) both asked for `..._0.h5` and HDF5 refused the second. This is the
intermittent failure `16-VERIFICATION §7.1` recorded and attributed to a full
`/tmp`; that was wrong. A process-global counter now makes the name unique.

## The fixture, and a deliberate deviation from upstream's pinned mesh

`test_kuccsd_openshell.py`'s three-hydrogen cell on a two-`s` basis at `[1,1,2]`:
`nocca = 2`, `noccb = 1`, `nmo = 6`. **The only genuinely open-shell fixture in
Phase 16**, and the reason both arena defects surfaced — with `nocca == noccb`
several genuinely different shapes coincide.

The mesh is `[31,31,31]`, not upstream's pinned `[13,13,13]`. **Measured, not
assumed:** at `[13,13,13]` the twenty-six ERI blocks agree with upstream from
`2.7e-9` on `oooo` to `1.2e-5` on `vvvv`, monotone in the number of VIRTUAL
indices — this fixture's virtuals are the antibonding combinations of an
all-electron `0.5`-exponent `s` function in a 6.74-Bohr cell, which a 13-point
FFT does not resolve. At `[31,31,31]` the table is flat at `~5e-10` and `vvvv`
is TIGHTER than `oooo`. Refinement factor on the worst block: **×14 881**.

Gating on the coarse mesh would have meant a `1e-4` gate hiding four orders of
real agreement. `the_eri_residual_is_the_mesh_and_not_the_port` runs BOTH meshes
and asserts the factor, so the choice stays a measurement.

## Gates (all measured; `measurements/README.md` is the authority)

| Gate | Measured | Set | Test |
|---|---|---|---|
| 26 ERI blocks, element-wise | `1.16e-10 … 8.07e-10` | `1e-9` | `kueris_blocks_match_upstream` |
| the same at `[13,13,13]` | `2.7e-9 … 1.20e-5` | `1e-4` | `the_eri_residual_is_the_mesh_and_not_the_port` |
| mesh refinement factor | `×14 881` | `>1e3` | same |
| `init_amps` `emp2` | `6.44e-12` | `1e-9` | `kuccsd_init_amps_matches_upstream` |
| `energy` on synthetic amps | `8.47e-12` | `1e-9` | `kuccsd_update_amps_matches_upstream` |
| 9 `tau`s | `0` (bit-identical) | `1e-9` | `kuccsd_intermediates_match_upstream` |
| 6 Fock intermediates | `5.1e-12 … 2.5e-11` | `1e-9` | same |
| `Woooo`/`WooOO`/`WOOOO` | `2.4e-10 … 8.1e-10` | `1e-9` | same |
| `Wvvvv`/`WvvVV`/`WVVVV` | `4.4e-10 … 7.4e-10` | `1e-9` | same |
| 6 `Wovvo`-family | `1.9e-10 … 5.7e-10` | `1e-9` | same |
| `add_vvvv_` standalone | `2.1e-11 … 4.7e-11` | `1e-9` | same |
| stage `:65-202` (Fock) | `1.4e-12 … 1.1e-10` | `1e-9` | `kuccsd_fock_block_matches_upstream` |
| stage `:205-226` (ovov+Woooo) | `8.7e-12 … 2.2e-10` | `1e-9` | `kuccsd_woooo_block_matches_upstream` |
| stage `:230-386` (Wovvo) | `6.4e-11 … 8.9e-11` | `1e-9` | `kuccsd_wovvo_block_matches_upstream` |
| `update_amps`, 15 arrays × 3 amplitude configurations | `1.0e-11 … 1.6e-10` | `1e-9` | `kuccsd_update_amps_matches_upstream` |
| converged `e_corr` | **`6.05e-10`** (14 cycles) | `1e-7` | `kuccsd_e_corr_matches_upstream` |
| `update_amps(0,0) == init_amps` | `6.9e-18` | `1e-15` | `kuccsd.rs`, ORACLE-FREE |

## Why the stage-wise gates exist and are not redundant

`update_amps` is five additive stages over five amplitude arrays and three spin
channels. When the assembled `t2new` was wrong, "the doubles are wrong" was not
a diagnosis. Each stage is now compared against upstream's OWN lines — copied
verbatim into the emitter, never re-transcribed, so a mismatch is attributable
to the port and not to a second reading of upstream.

That is what localised D1: every stage passed and the assembly still failed,
which is only possible if the fault is outside the equations. Without the
stage-wise gates the search space was 400 lines of einsum.

They also outlive this plan: 16-11 (EOM-KUCCSD) reuses these intermediates, and
an error that cancels in the assembled ground-state `t2new` would survive an
end-to-end gate and break EOM.

## DEFERRED, explicitly

* **`_make_df_eris`** (`kccsd_uhf.py:1017`) — the GDF-direct route that builds
  `Lpv`/`LPV` so `add_vvvv_` can form `Wvvvv` on the fly. The shipped
  incore/outcore route produces the SAME `Wvvvv` through `cc_wvvvv_half`, so
  nothing is missing from the energy — only the memory saving. Upstream's own
  refusal on that route (`cell.dimension == 2`, `:1022`) is reproduced in
  `KuEris::check_dimension_for_direct_df` and tested, so the condition cannot be
  lost.
* **`OOoo`** (`:830`) and **`VVvv`** (`:1013`) — `None` upstream, read by
  nothing. Not built; `the_block_table_is_upstreams` pins the omission so it
  stays a decision.
* **`kintermediates_uhf.py:590-1225`** — `Foo`, `Fvv`, `Fov`, `Wvvov`, `Wvvvo`,
  `Woooo`, `Woovo`, `Wooov`, `Wovvo`, `W1oovv`, `W2oovv`, `Woovv`,
  `_eri_spin2spatial`, `_eri_spatial2spin`. These are EOM-KUCCSD's (plan 16-11),
  not the ground state's. NOT stubbed.
