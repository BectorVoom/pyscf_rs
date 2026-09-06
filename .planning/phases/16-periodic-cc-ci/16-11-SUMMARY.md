# 16-11 — EOM-KUCCSD (`eom_kccsd_uhf.py`)

**Status: SHIPPED. IP and EA are the whole surface.**

`eom_kccsd_uhf` declares **no `EOMEE` class at all** and its `_IMDS.make_ee`
(`:1120`) is a bare `raise NotImplementedError` — `16-CONTEXT §1.5` recorded
this before any code. Nothing is deferred here that upstream implements.

## What shipped

| Piece | Upstream | Notes |
|---|---|---|
| Thirteen EOM intermediates | `kintermediates_uhf.py:311-390`, `:590-1117` | `Foo`, `Fvv`, `Fov`, `Wooov`, `Wovvo`, `W1oovv`, `W2oovv`, `Woovv`, `Woooo`, `Wvvvv`, `get_Wvvvv`, `Wvvov`, `Woovo`, `Wvvvo` |
| `UhfEomImds` | `:1040-1130` | `make_shared` / `make_ip` / `make_ea` |
| IP | `:43-446` | four-block packing, matvec, diagonal, `mask_frozen` |
| EA | `:547-957` | the same four |
| The Davidson driver | `eom_kccsd_ghf.py:40-159` | this module's packings and matvecs |

## Gates (all measured)

| Gate | Measured | Set |
|---|---|---|
| **51 intermediate spin blocks** | `5.14e-12 … 8.17e-10` | `1e-9` |
| IP matvec / diagonal, every `kshift` | `9.66e-10 … 1.25e-9` | `1e-8` |
| EA matvec / diagonal, every `kshift` | `8.28e-10 … 1.51e-9` | `1e-8` |
| IP / EA vector round trip | **`0e0`** | exact |
| `ip_vector_size` / `ea_vector_size` | `104` / `350` == upstream, per `kshift` | exact |
| **IP roots** (2 shifts × 2) | `2.48e-10 … 2.61e-10` | `1e-5` |
| **EA roots** (2 shifts × 2) | `3.92e-10 … 4.13e-10` | `1e-5` |

All roots converged on both sides; quasiparticle weights `0.79 … 0.92`.

## The matvec gate is its own number, and that is measured

An EOM matvec is a linear combination of roughly forty intermediate blocks, each
measured here at up to `8.17e-10`, so its residual is an ACCUMULATION of theirs
and sits a small multiple above. `1e-8` is an order above the largest measured
and still seven orders below anything a transcription error produces. The
spin-orbital module shows the same accumulation the other way round — there the
blocks sit at `4.7e-7` and the matvecs at `2.7e-7`, BELOW the block floor,
because that fixture's mesh leaves far more room.

## The gating caught a real O(1) defect, again

`_IMDS.get_Wvvvv(ka, kb, kc)` returns `self.Wvvvv[ka, kc, kb]` (`:1129`) — the
last two k-indices are **swapped on lookup**. This port indexed `[ka, kb, kc]`
directly, and the EA matvec came out `4.9e-2` wrong and the EA diagonal
`9.7e-2` wrong while every one of the fifty-one intermediates was exact to
`8e-10`. After the fix: `1.5e-9` and `8.3e-10`.

That is the third time in this session that a stage-wise gate has caught
something an end-to-end comparison would have buried: 16-06's arena bug (every
stage exact, the assembly wrong), 16-09/10's packing assertions, and this.

## Index subtleties recorded at their sites

Each of these is silent — same shape, wrong answer:

* `ipccsd_matvec:207` reads `WooOO[kn, kj, km]` with the index pairs swapped
  relative to the three lines above it.
* `ipccsd_diag:368` reads `WooOO[kj, kj, ki]` as `jjII->Ij`, occupied labels
  swapped relative to its three neighbours.
* `ipccsd_diag:372` reads `WovOV` at `kb`, not at `kshift` like its neighbours.
* `eaccsd_matvec:681` writes `Hr2aba` at `[kj, kb]`, not `[kj, ka]`, and reads
  `r2aba` at `kd`.
* `eaccsd_matvec:722-726` indexes its OUTPUT at `[kl, ka]` — the loop variable
  `kl` plays the role of `kj`.
* `eaccsd_matvec:735-737` contracts `W[kshift,kx,ky]` against `r2[ky,kx]` — the
  `r2` index pair is TRANSPOSED relative to the `W` one.
* `eaccsd_diag:871` writes at `[kj, kb]` and reads `bbAA->Ab`, the two virtual
  labels swapped.
* `Wvvvo:772`/`:774` read `OVov`/`ovOV` at `kf`, not `ke`; `:812` writes
  `WVVvo` at `[kb, kf, ka]`; `Woovo:989` reads `WmiNJ[kb, kj, km]`.

## NOT ported, and recorded

* **`EOMEE`** — absent upstream. Not a deferral.
* **Both `partition='mp'` diagonal branches** (`ipccsd_diag:325`,
  `eaccsd_diag:835`). Upstream opens each with
  `raise Exception("MP diag is not tested")` over code it has itself commented
  out; reproducing that is not porting it.
* **`eaccsd_diag:869`'s `# FIXME: Do Wvvvv and WVVVV have a factor 0.5?`** —
  upstream's code has NO factor, and this port reproduces the code, not the
  question. Recorded here so the question is not lost.
