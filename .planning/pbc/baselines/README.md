# W-00 baselines — `KRKS-OPTIMISATION-PLAN.md`

Captured **2026-08-31** on an idle machine (AMD Ryzen AI 7 350, 8 cores /
16 threads, load average < 2 for the whole run), CPU cubecl backend,
`--release`, after W-05 / W-01 / W-02 / W-02b had landed (commit `81b6c72`).

Re-run and diff with:

```bash
cargo run -p pyscf-bench --release --bin krks_profile -- \
    jk --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe0 \
    --compare .planning/pbc/baselines/jk-si222-mesh31-pbe0.json

cargo run -p pyscf-bench --release --bin krks_profile -- \
    transform --json /tmp/t.json   # diff against transform-baseline.json
```

## Files

| file | cell | k-mesh | FFT mesh | xc |
|---|---|---|---|---|
| `jk-si222-mesh21-pbe0.json` | Si `gth-szv`/`gth-pade` | 2×2×2 | 21³ | PBE0 (hybrid) |
| `jk-si222-mesh31-pbe0.json` | ″ | 2×2×2 | 31³ (**the gate mesh**) | PBE0 (hybrid) |
| `jk-si222-mesh21-pbe.json` | ″ | 2×2×2 | 21³ | PBE (pure) |
| `jk-si222-mesh31-pbe.json` | ″ | 2×2×2 | 31³ | PBE (pure) |
| `transform-baseline.json` | — | — | 16/21/25/27/31/32 | the isolated `2·Nk²·nao²` transform batch |

## The numbers, and what they say

### The isolated transform batch (§2.1's factorisation-cliff table)

| mesh | plan §2.1 (pre-W-02) | **here (post-W-02/W-02b)** | ratio |
|---|---|---|---|
| 16 | 22.0 ns/pt | **7.68** | 2.9× |
| 21 | 73.7 ns/pt | **13.39** | 5.5× |
| 25 | 86.7 ns/pt | **12.03** | 7.2× |
| 27 | 89.1 ns/pt | **16.52** | 5.4× |
| 31 | 103.5 ns/pt | **20.20** | 5.1× |
| 32 | 44.4 ns/pt | **7.63** | 5.8× |

**The factorisation cliff is gone.** §2.1's headline anomaly — `mesh 31`
costing 2.3× `mesh 32` per point despite having 10 % fewer points — is down
to 2.6× *per point* only because 31 is prime and goes through Rader; in
absolute terms `mesh 31` (38.5 ms) is now cheaper than `mesh 32` was in the
old measurement and within 2.4× of the power-of-two path rather than the
`O(n²)` factor the `Direct` plan cost.

### `get_k_kpts` — the dominant cost of a hybrid

| | plan §2.1 (mesh 21) | **here (mesh 21)** | here (mesh 31, the gate mesh) |
|---|---|---|---|
| `get_k_kpts` | 6 600 ms | **1 428 ms** | 7 707 ms |
| `get_j_kpts` | 13.9 ms | 13.6 ms | 52.9 ms |
| `nr_rks` warm | 22.7 ms | 26.9 ms | 87.9 ms |

`get_k_kpts` is **4.6× faster** than the plan's pre-W-02 measurement, and it
is still 99 % of a hybrid iteration.

**Re-attribution — this supersedes §2.1's "transform = 93 %".** The isolated
batch says `8192` transforms at mesh 21 now cost `128 × 7.94 ms ≈ 1 016 ms`
against a `get_k_kpts` of `1 428 ms` — **71 %**, not 93 %. The contractions
and the element-wise passes are now **~29 %** of `get_k_kpts`, up from 7 %,
purely because the transform got 5× cheaper and they did not. That is a
*fourfold* increase in W-03's share of the remaining cost.

### The pure functional — the one-off AO collocation dominates, not the iteration

| stage | mesh 21 | mesh 31 |
|---|---|---|
| `nr_rks` **warm** (per iteration) | 13.3 ms | 45.9 ms |
| `get_j_kpts` (per iteration) | 13.1 ms | 47.9 ms |
| `nr_rks` **cold** (one-off AO collocation) | **4 861 ms** | **16 772 ms** |
| `get_hcore` (one-off) | 1 693 ms | 4 888 ms |
| full `kernel()` to convergence | 7 430 ms | 22 669 ms |

A converged pure-PBE SCF iteration is ~27 ms (mesh 21) / ~94 ms (mesh 31).
Over the ~13 iterations it takes, that is **under 1.5 s of a 22.7 s run**:
**74 % of a pure-functional KRKS is the cold `eval_ao_kpts` pass, and another
21 % is `get_hcore`.** §2.1 measured the same 7.26 s cold AO figure but drew
the conclusion "a pure functional is already fast" from the *warm* number
alone. It is not — it is fast *per iteration* and slow *once*.

Consequence for the plan: **W-06 (zgemm in `numint`) targets ~46 ms per
iteration, i.e. ~3 % of a pure-functional run.** The item that would actually
move a pure functional is a faster `eval_ao_kpts` — which is W-09's territory
(AO screening) or a separate AO-collocation item the plan does not yet have.
W-06 remains worth doing as the ALG-06/D-PBC-03 compliance item it is; it
should not be sold as the pure-functional fix.
