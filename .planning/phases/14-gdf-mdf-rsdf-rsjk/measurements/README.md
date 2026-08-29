# Phase 14 pre-implementation measurements (upstream PySCF 2.12.1)

Run every one of these from THIS directory as:

```bash
PYTHONPATH=<workspace root> ../../../../.venv/bin/python -u <script>.py
```

`PYTHONPATH` pointing at the workspace root is mandatory — it pins `import pyscf`
to the **vendored** 2.12.1 tree at `<root>/pyscf`, not site-packages 2.14. `-u`
is mandatory too: `builders.py` takes tens of minutes and a buffered pipe
swallows every row until exit.

| script | measures | feeds |
|---|---|---|
| `_cells.py` | the two reference cells (diamond/`gth-szv`, He-fcc/`sto-3g`) | all |
| `smoke.py` | that upstream `GDF`/`MDF`/`RSDF` build at all on these cells | — |
| `params.py` | `eta`, `mesh`, `naux`, `fused_cell.nao`, `j2c`/`cderi`/`auxbar` fingerprints, every `estimate_*` | 14-01, 14-02, 14-03 |
| `rscell.py` | the `_RangeSeparatedCell` split, `supmol.nbas`, the `j2c` decomposition tag | **D-PBC-23** |
| `ddblock.py` | what `exclude_dd_block` is WORTH in Ha | **D-PBC-23**, Gate 1 |
| `memory.py` | `_cderi` bytes vs the FFTDF AO table at mesh `[40,40,40]` | Gate 4 |
| `builders.py` | the KRHF energy of all five builders and every pairwise difference | Gate 2, Gate 3 |

## Results recorded 2026-08-29

Diamond, `gth-szv`/`gth-pade`, fcc `a0 = 6.74064` Bohr, `nao = 8`, default mesh
`[47,47,47]`, `cell.rcut = 21.319`.
He-fcc, `sto-3g`, ALL-ELECTRON, `nao = 1`, `cell.rcut = 12.300`.

### `params.py` — what the port must reproduce

| quantity | diamond 2×2×2 | diamond gamma | He-fcc 2×2×2 |
|---|---|---|---|
| `auxcell.nao` / `.nbas` | 108 / 36 | 108 / 36 | 23 / 9 |
| `eta` | 0.46488312492994555 | 0.6839707371739572 | 0.37482108075015924 |
| `mesh` (GDF's own) | `[11,11,11]` | `[13,13,13]` | `[9,9,9]` |
| `ke_cutoff` | 21.721883440437864 | 31.27951215423053 | 19.65348325887675 |
| `fused_cell.nao` / `.nbas` | 126 / 42 | 126 / 42 | 32 / 12 |
| `direct_scf_tol` | 1.1732759515929528e-13 | same | 9.597462925489878e-13 |
| `j2c(k=0)` ‖·‖ | 9.774955865744985 | 9.77495586574475 | 10.064640251330108 |
| `j2c` eig min / max | 3.171e-11 / 3.26898 | 3.171e-11 / 3.26898 | 6.638e-3 / 7.68841 |
| `j2ctag` | `CD` | `CD` | `CD` |
| `auxbar` nnz / ‖·‖ | 12 / 0.23012787965177506 | 12 / 0.2027689464353206 | 4 / 0.3187837520926407 |
| `cderi[0,0]` ‖R‖ / ‖I‖ | 1.5900257696377815 / 0 | 1.5900256974912226 / 0 | 0.6068683433161949 / 0 |
| `incore.estimate_rcut` | 17.266040957536866 | same | 9.53235156147295 |
| `gdf_builder.estimate_rcut` | 16.729034885581783 | same | 10.750308556151602 |
| `estimate_eta_min` | 0.1 (= `ETA_MIN`) | 0.1 | 0.17166722884078006 |

`auxcell` for diamond is even-tempered, six exponents per `l ∈ {0,1,2}` per atom,
`e = 7.6024170048 · 2^-n`, `n = 0..5`; He-fcc's is 4 s + 3 p + 2 d.
GDF's own `mesh` is TINY (`[11,11,11]`) — it only has to resolve the
compensating charge, not the density. That is the whole point of the scheme and
it is why GDF beats FFTDF on memory.

### `rscell.py` — the `_RangeSeparatedCell` split IS active on diamond

| | diamond | He-fcc |
|---|---|---|
| `cell.nbas` → `rs_cell.nbas` | 4 → **8** | 1 → 1 |
| `rs_cell.bas_type` | `[1 2 1 2 1 2 1 2]` | `[1]` |
| any `SMOOTH_BASIS` (= 2) | **True** | **False** |
| `supmol.nbas` (after `strip_basis`) | 1486 | 135 |
| `supmol_ft.nbas` | 2472 | 177 |
| `exclude_dd_block` (default) | True | True (inert) |

`_CCGDFBuilder.exclude_dd_block` defaults to **True**, and on diamond/`gth-szv`
it is LIVE: every contracted shell decontracts into a compact + a smooth block,
so `_int_dd_block` really does divert the smooth–smooth block of `(ij|L)` from
the real-space lattice sum into an FFT. On He-fcc/`sto-3g` no shell is smooth,
so the flag is inert and both routes are the same code path.

### `ddblock.py` — what `exclude_dd_block` is WORTH (the phase's defining number)

`KRHF`, `conv_tol = 1e-11`, `exxdiv='ewald'`, aux basis default:

| system | `exclude_dd_block=True` | `=False` | \|dE\| | Δ‖cderi_R‖ |
|---|---|---|---|---|
| diamond 2×2×2 | −10.93209469510983 | −10.93209471346292 | **1.835e-8** | 1.808e-7 |
| diamond gamma | −10.14369242019067 | −10.14369244919373 | **2.900e-8** | 1.838e-7 |
| He-fcc 2×2×2 | −2.80842508664874 | −2.80842508664874 | **0** | **0** |

**This is the measurement that sets the phase's architecture.** See D-PBC-23 in
`14-CONTEXT.md`.

### `memory.py` — Gate 4 is k-mesh dependent and the roadmap does not say so

FFTDF's resident cost is the AO table `nkpts·ngrids·nao·16 B`; GDF's is the
`_cderi` store. At mesh `[40,40,40]`:

| system | FFTDF AO table | GDF `_cderi` file | ratio |
|---|---|---|---|
| diamond 2×2×2 | 62.50 MiB | 3.86 MiB | **6.17 %** |
| diamond **3×3×3** | 210.94 MiB | 44.20 MiB | **20.95 %** |
| He-fcc 2×2×2 | 7.81 MiB | 0.12 MiB | 1.48 % |

GDF is `O(nkpts²·naux·nao_pair)`; FFTDF's table is `O(nkpts·ngrids·nao)`. The
ratio therefore GROWS linearly in `nkpts` and crosses the roadmap's 20 % between
2×2×2 and 3×3×3 on this system. **The gate is only meaningful with the k-mesh
pinned**, and 2×2×2 is the mesh at which it passes.

### `builders.py` — see `builders.out`

### `builders.py` — the five builders' KRHF energies

diamond/`gth-szv` 2×2×2, `conv_tol = 1e-11`, `exxdiv='ewald'`, FFTDF/AFTDF at
mesh 31, GDF/MDF/RSDF at their own defaults:

| builder | `E_KRHF` | wall |
|---|---|---|
| FFTDF | −10.93087316795858 | 30.0 s |
| AFTDF | −10.93087316798466 | 450.6 s |
| GDF | −10.93209529106369 | 6.4 s |
| MDF | −10.93087429163887 | 16.9 s |
| RSDF | −10.93209530458920 | 13.5 s |

| pair | \|dE\| | what it is |
|---|---|---|
| FFTDF − AFTDF | **2.607e-11** | two EXACT builders — Phase 13's floor, reproduced |
| GDF − RSDF | **1.353e-08** | two builders of the SAME fitted quantity → **Gate 3** |
| FFTDF − MDF | **1.124e-06** | MDF = GDF + AFT residual → **Gate 2** |
| FFTDF − GDF | **1.222e-03** | the DF **fitting error**. Not an error in either builder. |

diamond gamma: FFTDF −10.13717266004632, AFTDF −10.13717266289455,
GDF −10.14369692267130.

**The roadmap's "every DF builder gives the same KRHF energy to 1e-15" is off by
twelve orders of magnitude and is comparing incomparable quantities.** See
`14-CONTEXT.md` § "The gates the roadmap gets wrong".

### `ccdf.py` — `df.GDF`'s DEFAULT builder is `_RSGDFBuilder`, not `_CCGDFBuilder`

`GDF._prefer_ccdf = False` (`df.py:132`), so `df.GDF()` builds through
`rsdf_builder._RSGDFBuilder`. `_CCGDFBuilder` — which plan 14-02 ports, because
it is the self-contained one — is the fallback route.

| system | `_prefer_ccdf=False` (RS) | `=True` (CC) | \|dE\| |
|---|---|---|---|
| diamond 2×2×2 | −10.93209529106394 | −10.93209469510988 | **5.960e-07** |
| diamond gamma | −10.14369692267143 | −10.14369242019162 | **4.502e-06** |
| He-fcc 2×2×2 | −2.80842508717097 | −2.80842508664874 | **5.222e-10** |

Two consequences the plans must honour:

1. **Plans 14-02/14-03's oracle target is upstream with `_prefer_ccdf = True`.**
   Comparing the port's CC-route GDF against upstream's default would show a
   6e-7 "error" that is not the port's.
2. Upstream's own two GDF routes disagree by up to **4.5e-6**, which is another
   independent reason no "all builders agree to 1e-15" gate can stand. The
   port's default flips to the RS route in 14-07, matching upstream.

#### The full `builders.py` cross-builder table

| pair | diamond 2×2×2 | diamond gamma | He-fcc 2×2×2 |
|---|---|---|---|
| FFTDF − AFTDF | 2.607e-11 | 2.848e-09 | 3.006e-13 |
| **GDF − RSDF** | **1.353e-08** | **4.566e-09** | **1.113e-10** |
| **FFTDF − MDF** | **1.124e-06** | **1.079e-06** | **1.591e-06** |
| FFTDF − GDF (fitting error) | 1.222e-03 | 6.524e-03 | 6.006e-05 |

He-fcc energies: FFTDF −2.80848514315536, AFTDF −2.80848514315506,
GDF −2.80842508717097, MDF −2.80848673371326, RSDF −2.80842508705964.
diamond gamma: FFTDF −10.13717266004632, GDF −10.14369692267130,
MDF −10.13717373919459, RSDF −10.14369691810517.

Read the columns, not the rows: **GDF − RSDF** is the only pair with a floor
tight enough to gate a port on (Gate 3), **FFTDF − MDF** is the pair that
converges (Gate 2), and **FFTDF − GDF** is the DF fitting error, which is a
property of the auxiliary basis and belongs in no gate at all.

### `mdfladder.py` — Gate 2's ladder, and the trap in it

`E_KRHF(MDF, mesh)` against `E_KRHF(FFTDF, mesh 31)`, `conv_tol = 1e-11`:

| MDF mesh | diamond 2×2×2 | He-fcc 2×2×2 |
|---|---|---|
| 7 (MDF's own default) | 1.124e-06 | 1.624e-06 |
| 11 | 9.938e-09 | **3.730e-09** ← best |
| 15 | 1.296e-09 | 3.243e-08 |
| 21 | **3.241e-10** ← best | 3.345e-08 |
| 27 | 2.575e-09 | 2.965e-08 |

GDF alone, for scale: 1.222e-03 (diamond) and 6.002e-05 (He-fcc). MDF buys
**three to six orders of magnitude**, which is what makes this a real gate.

**But the ladder is NOT monotone, and a "monotone in the mesh" gate would fail a
correct implementation.** Both systems fall steeply, hit a floor, and then
*bounce*: diamond bottoms at mesh 21 and is 8× worse at 27; He-fcc bottoms at
mesh 11 and is ~9× worse from 15 on. Two independent floors are in play — MDF's
own aux-basis fit, and the **FFTDF reference's** mesh-31 truncation — and past
the crossover the comparison is measuring the reference, not MDF. This is the
same two-floor structure Phase 13 hit in its own Gate 2 (`13-CONTEXT.md`), for
the same reason.

**Gate 2 is therefore: falls by ≥3 orders from the default mesh to the plateau,
then STAYS within one order of the plateau.** Not "monotone", and not a single
number.

## Recorded during 14-01 — `aux_e2` against a CHARGED auxiliary cell is only conditionally convergent

Found while gating the port's `aux_e2` on upstream `incore.aux_e2`. Scripts:
`iso.py`, `conv.py`, `probe2.py` (in the session scratchpad; the numbers below
are the reproducible part and the conclusion is asserted in
`crates/pyscf-pbc-df/tests/incore.rs`).

**The port's per-triple algebra is EXACT.** On an isolated cell (He/`sto-3g`,
`a = 15 I` Bohr, one lattice image), the port and upstream `incore.aux_e2` agree
to **~1e-15** on all 23 auxiliary components:

```text
port     1.627575857940794  1.493095234125918  1.2102460352747024  0.8302924735218621
upstream 1.6275758579407937 1.493095234125919  1.2102460352747022  0.8302924735218626
```

So the modrho normalisation, the `modrho_scale` application, the cintx
`int3c2e` evaluation and both index orders are right.

**On a real periodic cell they differ, and the reason is that the raw quantity
is not well defined.** Brute-forcing the double lattice sum in upstream itself —
a `Mole` with one He per lattice image plus the auxiliary at the origin,
calibrated against upstream's own one-image value — gives, for He-fcc:

| | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| single sum, bra pinned at cell 0 | 1.782435 | 1.646241 | 1.358141 | 0.964684 |
| **double** sum, `R = 9.532` | 16.110074 | 15.972144 | 15.678667 | 15.270229 |
| **double** sum, `R = 14.0` | 34.052358 | 33.914427 | 33.620951 | 33.212513 |
| upstream `incore.aux_e2` | 10.928734 | 10.790804 | 10.497327 | 10.088889 |

The double sum **diverges with `R`** — each auxiliary function carries net
charge, so its Coulomb lattice sum decays as `1/R` and is only conditionally
convergent. And upstream is `double(R)` minus a **P-INDEPENDENT** offset:

```text
double(9.532) - upstream = 5.1814, 5.1814, 5.1814, 5.1814   (four identical digits)
double(14.0)  - upstream = 23.12,  23.12,  23.12,  23.12
```

P-independence is the signature of a term proportional to `S_mu_nu * q_P`, and
every modrho-normalised auxiliary function has the SAME monopole `q_P` by
construction — so what upstream removes is the divergent `G = 0` / background
-charge piece. Its own result is `R`-insensitive (identical at `estimate_rcut`
x1.0, x1.5 and x2.0), which is only possible for a regularised sum.

**The compensated tensor, which is the one GDF actually consumes, IS well
defined.** Running upstream's `fuse(aux_e2(cell, fused_cell))` at the same three
radii:

| `rcut` scale | `fuse(j3c)[:4]` |
|---|---|
| x1.0 | 0.942475  0.804544  0.511067  0.102629 |
| x1.5 | 0.942475  0.804544  0.511067  0.102629 |
| x2.0 | 0.942475  0.804544  0.511067  0.102629 |

Bit-identical across a 2x radius. The compensating charge neutralises the
auxiliary functions, the `1/R` tail cancels, and the lattice sum converges
absolutely.

### What this changes

1. **Plan 14-01's oracle gate on the raw periodic `aux_e2` is retired**, the way
   Phase 12 §1d and Phase 13 retired theirs: it gates a quantity that has no
   screening-independent value. It is replaced by (a) the **isolated-cell
   identity at 1e-15**, which pins the algebra with nothing else in the way, and
   (b) the structural identities (`s2` packs `s1`, the bra/ket conjugation
   identity, reality at gamma, Hermiticity of `j2c`).
2. **The 1e-11 oracle gate moves to plan 14-02, on `fuse(j3c)`** — the
   compensated tensor, which the table above shows is screening-independent.
3. `aux_e2`'s intended argument is the **fused** auxiliary cell. Its doc comment
   says so, and says why calling it with a charged one is only conditionally
   convergent.
4. The Gaussian-product prescreen inherited from `pseudo::vloc_part2` is
   **valid for the fused cell** (whose integrand is charge-neutralised and
   therefore exponentially decaying) and only approximate for a charged one.
   That is now documented at the call site rather than assumed.

## Recorded during 14-02 — the compensated tensor, and what the residual actually is

### The screens must be aggregated PER AUXILIARY ATOM

An auxiliary function and its model charge sit on the SAME centre, have very
different exponents, and are subtracted from each other immediately
(`FusedCell::fuse_rows`). Screening them independently keeps a triple for one
and drops it for the other, and the cancellation that makes the compensated
tensor converge is destroyed. Both screens hit this:

| screen | symptom when applied per SHELL | fix |
|---|---|---|
| shell-pair neighbour list (`rcut_by_shells`) | `fuse(j3c)[0]` = −3.408 against upstream's 0.942475 — the COMPACT auxiliaries wrong, the diffuse ones exact | radius aggregated per auxiliary ATOM (max) |
| Gaussian-product prescreen | `fuse(j3c)` = [−4.90, −4.88, −5.17, −0.72] against [0.942, 0.805, 0.511, 0.103] | exponent/coefficient aggregated per auxiliary ATOM (min exponent, max coefficient) |

Both aggregations are strictly more conservative than the per-shell bound, and
upstream's own `strip_basis(rcut)` does the same thing implicitly: its
`estimate_rcut(rs_cell, auxcell)` returns one radius per ORBITAL shell and takes
`aux_exps.argmin()` — a single, global auxiliary exponent.

### `libcint deduplicates identical basis blocks across atoms`

`make_modrho_basis` rewrites `_env` contraction coefficients in place. Two atoms
of the same element **share one `PTR_COEFF` slot**, so a naive per-shell loop
scales the same entries once per atom: the second atom reads already-normalised
coefficients, computes `scale = 1`, and leaves `_env` SQUARED. Measured on
diamond: the auxiliary metric came out `‖j2c‖ = 4495` where upstream says
`251.96191223`, with atom 0 correct and atom 1 untouched. Keying the
normalisation on the coefficient pointer makes it idempotent. He-fcc, having one
atom, could never have caught this.

### The port/upstream `fuse(j3c)` gap IS upstream's own `direct_scf_tol`

He-fcc, gamma, `fuse(aux_e2(cell, fused_cell))`:

| | `j3c[0]` | `j3c[1]` |
|---|---|---|
| upstream, `direct_scf_tol = None` (its default) | 0.94247478444056 | 0.80454392614668 |
| upstream, `direct_scf_tol = 1e-14` | 0.94247478635665 | 0.80454392805940 |
| upstream, `1e-18` / `1e-22` | 0.94247478635804 / …811 | 0.80454392806076 / …083 |
| **the port** (prescreen 1e-14) | **0.94247478635764** | **0.80454392806081** |

`Int3cBuilder.direct_scf_tol = None` derives
`cell.precision / lattice_sum_factor² · 0.1` = **1.46e-11** for this system —
four orders LOOSER than the port's 1e-14 Gaussian-product bound. Upstream
therefore discards a term the port retains, and that term is **P-independent**
(1.981e-9, 1.978e-9, 1.9e-9 across auxiliary functions whose exponents span
66.2 → 0.82), which is the `q_P · S_μν` signature — the same one 14-01 found in
the raw tensor.

Equalise the two screens and the port matches upstream to **1.41e-12**; the
metric matches to **7.11e-14**. Sweeping the port's own prescreen from 1e-14 to
1e-20 moves nothing (the bound is saturated), and the port's `fuse(j3c)` is
rcut-converged from ×1.3 to ×2.5. So the 1.98e-9 is upstream's screening choice,
not the port's algebra — and that is how `14-02`'s gate is stated.

## Recorded during the 14-02 performance pass

Two changes, and only one of them was the win.

### Threading the bra-image loop — 1.6x, and it must be SIZED

`aux_e2`'s outer sum over bra lattice images is embarrassingly parallel (each
`mi` is an independent additive term), so it is split with
`std::thread::scope`, each worker accumulating into its own output and the
partials reduced **in chunk order** — the result does not depend on the thread
count (FOUND-06 by construction).

He-fcc 2×2×2, `outcore_auxe2` over 8 k-pairs, debug build:

| threads | time |
|---|---|
| 1 | 14.85 s |
| **4** | **9.20 s** |
| 8 | 10.38 s |
| 16 | 11.14 s |

It gets WORSE past 4. `EvaluationContext::new()` costs ~0.3 s (a cubecl executor
plus a host scratch arena), so a small workload must not be split 16 ways — the
first attempt did exactly that and took He from 2.56 s to 7.30 s. The pool is
therefore sized by the estimated triple count
(`MIN_TRIPLES_PER_THREAD = 20_000`), which picks 4 for He-fcc and 16 for
diamond.

### Hoisting `get_2c2e` and `outcore_auxe2` out of the group loop — **8.4x**

This was the real one, and it is a **faithfulness fix as much as a performance
fix**. Upstream's `make_j3c` calls `outcore_auxe2` ONCE
(`rsdf_builder.py:930`) and `gen_uniq_kpts_groups` calls
`get_2c2e(j2c_uniq_kpts)` ONCE (`:853`), both before the group loop. The port
called each inside it.

Both `aux_e2` and `pbc_intor` fold every lattice image into EVERY requested
k-point in a single sweep, so `nkpts` separate calls cost `nkpts` times one call
and return identical numbers. Measured on He-fcc 2×2×2 (8 groups, debug):

| stage | per group | hoisted |
|---|---|---|
| `get_2c2e` | 5.47 s | one pass |
| `outcore_auxe2` | 10.98 s | one pass |
| **full `make_j3c`, all 64 k-pairs** | **135 s** | **16.0 s** |

Every gate re-run after the change: `fuse(j3c)` 1.448e-12, `j2c` 7.105e-14,
`cderi` 0.6068683433161949, 15 + 8 tests green.

### What is still slow

Diamond at GAMMA is one group, so the hoisting buys nothing there: its cost is a
single `aux_e2` over the fused cell — `429 images² × 10 s2 shell pairs × 42
fused auxiliary shells ≈ 77 million` cintx `SessionRequest`s at ~28 us each.
The threading DOES engage — a release run showed `user 280m59s` against
`real 22m20s`, i.e. **12.6 of 16 cores busy** — but the run was terminated at
22 min before it finished, so **diamond's wall time is unmeasured**, not
"about four minutes". It stays `#[ignore]`d as an opt-in acceptance run and the
number is owed.

The per-request cost is the floor: `cintx::ShellTuple` is a fixed-arity tuple, so
there is no way to ask for a shell RANGE in one request, and ~15 us of the 28 is
fixed overhead. Cutting it further needs either a batched cintx entry point or
upstream's `direct_scf_tol` Schwarz screen (2.05e-12 for diamond) applied to a
per-atom-aggregated bound — see the 14-02 note on why per-shell screening is not
available here.

## Recorded during 14-04 — the phase gate, and the mesh that nearly hid

### GATE 1 — a converged KRHF on GDF, against upstream

He-fcc/`sto-3g` 2×2×2, the all-electron control:

| | |
|---|---|
| port, `KRHF` on GDF | **−2.80842508692377** |
| upstream `df.GDF`, `_prefer_ccdf = True` | **−2.80842508664874** |
| \|dE\| | **2.750e-10** |
| \|E_GDF − E_FFTDF\| (the DF fitting error) | **6.0056e-05** vs upstream's **6.006e-05** |

### `get_nuc` on the compensated mesh is WRONG, and it cost 0.0743 Ha

Plan 14-03 delegated `get_nuc`/`get_pp` to AFTDF at `_CCNucBuilder`'s mesh,
reasoning that `eta` is chosen so the model charge is resolved by it. **The
mesh resolves the MODEL CHARGE, not the nuclear density** — `_CCNucBuilder`
splits `get_pp_loc_part1` into a real-space `_int_nuc_vloc` plus a
reciprocal-space remainder precisely so the coarse mesh suffices for what is
left. He-fcc 2×2×2:

| mesh | `v_nuc[0,0]` |
|---|---|
| `[9,9,9]` (`_CCNucBuilder`'s) | −1.835938176640 |
| `[15,15,15]` | −1.871405034120 |
| `[21,21,21]` | −1.872891481488 |
| `[31,31,31]` | −1.872934360277 |
| `[43,43,43]` (the cell's) | −1.872934388301 |

3.7e-2 per element → **0.0743 Ha** on the converged `KRHF`. Using the cell's
mesh — where AFTDF's `get_nuc` is oracle-gated to 2.755e-12 — took the SCF from
−2.73413100753339 to −2.80842508692377, i.e. from 7.4e-2 out to **2.75e-10**.

Only the phase GATE caught this. Every unit test upstream of it (auxiliary cell,
`j2c`, `fuse(j3c)`, `cderi`, `vj`, `vk`, Hermiticity) passed with the wrong mesh,
because none of them touches the nuclear attraction. That is the argument for
carrying an end-to-end SCF gate and not only component gates.
