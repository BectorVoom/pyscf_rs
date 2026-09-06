# 16-07 — `KGCCSD`. MOSTLY COMPLETE 2026-09-06. **`e_corr` matches upstream to 2.07e-9.**

`crates/pyscf-ccsd/src/gccsd.rs`, `crates/pyscf-pbc-cc/src/{kccsd,kintermediates}.rs`,
gated by `tests/oracle_phase16.rs::kgccsd_matches_upstream` and
`crates/pyscf-ccsd/tests/gccsd.rs`.

## Measured

diamond `gth-szv` `[1,1,2]`, mesh `[15,15,15]`, on upstream's own **KGHF** mean
field (`measurements/README.md §10` explains why every CC gate is run that way):

| quantity | measured |
|---|---|
| **`KGCCSD e_corr`** vs upstream | **`2.066e-9`** (G3 = `1e-8`), converged in 19 cycles |
| `energy()` on synthetic amplitudes | `1.554e-9` |
| `update_amps` `t1new` / `t2new` | `1.10e-8` / `7.28e-8` |
| the seven `<pq\|\|rs>` blocks | `2.42e-8 … 4.68e-7` |

The block residuals are roughly **twice** the RHF ones, which is what the
antisymmetrisation does: `<pq||rs>` is a difference of two chemist integrals,
so it carries two copies of the FFT transform floor.

**G3 is `1e-8`, not the plan's `1e-10`.** 16-01 measured upstream's own
`KGCCSD` and `KRCCSD` differing by `4.95e-9` on this fixture, so the plan-time
`1e-10` would fail a correct implementation.

## Task 1 — the narrow molecular `gccsd` surface

`crates/pyscf-ccsd/src/gccsd.rs`. Of the four entry points `kccsd.py` consumes
(`:332`, `:339`, `:352`, `:395`, `:477`) only `_PhysicistsERIs` carries
arithmetic; the rest is object plumbing this port expresses as ordinary Rust
construction and as `pyscf_pbc_cc::kccsd::kernel`. The module doc lists what is
deliberately absent — the molecular `update_amps`, `energy`, the RDMs, Lambda,
and the molecular AO→MO transforms — and says a full molecular GCCSD belongs to
whatever phase actually needs molecular GHF correlation.

**The notation is in the type name and asserted, not assumed**
(`crates/pyscf-ccsd/tests/gccsd.rs`, 3 tests): `<pq||rs> = (pr|qs) - (ps|qr)`
element by element, the four antisymmetries, the vanishing `<pp||rr>` diagonal,
and that the result is NOT the chemist block — the last so a port that forgot
the transposition cannot pass by accident on a symmetric fixture.

## Task 2 — `spatial2spin` / `spin2spatial`

`spatial2spin_t1` and `spin2spatial_t1` ship. **The `t2` pair does NOT**
(`kccsd.py:262-287`, `:317-329`): its packing folds the `aa`, `ab`, `bb` and
`abba` blocks into one `(nocc², nvir²)` array through four `takebak_2d` calls
with transposed index products, and porting it correctly needs a test that this
plan did not reach. Recorded as a carry-over rather than written blind — the
`t2` packing is precisely where an off-by-one is silent.

## Task 3 — `kintermediates` and `KGCCSD`

`make_tau`, `cc_Fvv`, `cc_Foo`, `cc_Fov`, `cc_Woooo`, `cc_Wvvvv`, `cc_Wovvo`,
`eris_vovv`, `eris_ovvo_oovo`; then `KgEris::from_parts`, `energy`,
`init_amps`, `update_amps` and the DIIS-accelerated kernel.

`kccsd.py:414`'s refusal ships as `refuse_414`, with the upstream line in the
payload. `:486` is COMMENTED OUT upstream and is deliberately not ported as a
refusal — porting it would invent a restriction upstream does not impose.

### The two defects this plan's own gates caught

1. **`cc_Wovvo` gathered the wrong k-axis.** `kintermediates.py:183` is
   `eris.oovv[km,:,ke]` — the free index is the MIDDLE one — and the first port
   used a free-FIRST accessor, `oovv[:,km,ke]`. **Both produce the same SHAPE**,
   so nothing but a numerical comparison catches it: `t1new` still matched to
   `1.5e-8` while `t2new` was `7.7e-4` out. The two accessors are now named
   apart (`blk_free_mid` vs `blk_free1`) with the trap in both doc comments.
2. **The kernel had no DIIS.** `kccsd.py:395` reaches `ccsd.CCSD.ccsd`'s
   DIIS-accelerated driver; the first port wrote a plain iteration and did not
   converge in 50 cycles (`e_corr` was already `3.7e-9` from upstream, so the
   energy was right and only the convergence FLAG was wrong — which is exactly
   why the test asserts `converged`, not just the number). With the same
   amplitude-DIIS stack `kccsd_rhf` uses: **19 cycles**.

## Not shipped

* the `t2` half of `spatial2spin`/`spin2spatial` (above);
* a `Kgccsd` DRIVER that builds its own Fock from this port's `Kghf` — the GHF
  Fock needs the block-diagonal `hcore`/`veff` expansion, and every gate here
  runs on upstream's mean field anyway (`README §10`), so the driver adds no
  measured coverage. Carry-over;
* `scf.kghf.KGHF.CCSD` registration (`kccsd.py:805`), the surface **Phase 19**
  reads — it needs the driver above;
* `check_antisymm_3412`/`_12`/`_34` (`kccsd.py:641-704`) as tests on the built
  ERIs. The `gccsd.rs` tests assert the same four antisymmetries on a synthetic
  block, and the element-wise oracle comparison of all seven blocks is strictly
  stronger on the real one, but upstream's own three helpers are not ported;
* 16-07 test 2, `KGCCSD.e_corr == KRCCSD.e_corr` on a closed shell, ORACLE-FREE
  — needs the `t2` spatial2spin. Both sides ARE separately gated against
  upstream (`2.07e-9` and `6.56e-9`), and 16-01 measured upstream's own gap at
  `4.95e-9`, so the cross-check would add independence, not a new number.
