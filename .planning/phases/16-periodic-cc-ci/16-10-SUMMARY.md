# 16-10 — EOM-KRCCSD (`eom_kccsd_rhf.py`)

**Status: IP and EA SHIP and are oracle-gated to their roots.** EE is not
ported and the reason is recorded below.

## What shipped

| Piece | Upstream | Notes |
|---|---|---|
| Twelve EOM intermediates | `kintermediates_rhf.py:229-455` | `Wooov`, `Wvovv`, `W1ovvo`, `W2ovvo`, `Wovvo`, `W1ovov`, `W2ovov`, `Wovov`, `Woooo`, `Wvvvv`, `Wvvvo`, `Wovoo` |
| `RhfEomImds` | `:1497-1716` | `make_shared` / `make_ip` / `make_ea` / `get_Wvvvv` |
| IP | `:39-212` | packing, matvec, left matvec, diagonal |
| EA | `:430-615` | the same four |
| `mask_frozen_ip` / `_ea` with the RHF packing | `eom_kccsd_ghf.py:663`, `:1180` | see below |
| The Davidson driver | `eom_kccsd_ghf.py:40-159` | this module's matvecs substituted |

## Gates (all measured)

| Gate | Measured | Set |
|---|---|---|
| twelve EOM intermediates | `5.70e-9 … 2.39e-7` | `1e-6` |
| IP matvec / lmatvec / diag, every `kshift` | `2.19e-8 … 2.69e-7` | `1e-6` |
| EA matvec / lmatvec / diag, every `kshift` | `1.68e-8 … 5.43e-7` | `1e-6` |
| IP / EA vector round trip | **`0e0`** | exact |
| `ip_vector_size` / `ea_vector_size` | `260` == upstream | exact |
| **IP roots** (2 shifts × 2) | `6.86e-11 … 4.59e-10` | `1e-5` |
| **EA roots** (2 shifts × 2) | `2.51e-10 … 1.40e-9` | `1e-5` |

All roots converged on both sides, quasiparticle weights `0.97 … 0.99`.

## This is not 16-09 with `nocc` halved

`eom_kccsd_rhf.py:25` imports the GHF module and `EOMIP`/`EOMEA` inherit from
it, but only the DRIVER is shared. The matvecs, diagonals, intermediates and
vector packing are all different: the spin-adapted equations carry **thirteen
explicit `2·X − Xᵀ` combinations** (`St2`, `SWooov`, `SWovvo`, `SWvovv`,
`SWoovv`) that antisymmetry supplies for free in the spin-orbital treatment.

The packing is FLAT — `[(nocc,), (nkpts, nkpts, nocc, nocc, nvir)]`
concatenated, with no triangle and no `kshift` dependence. The spin-orbital
module's `tril` packing exists because ITS `r2` is antisymmetric; this one's is
not, and using either packing for the other would be silently wrong in
`vector_size` alone.

`W1ovvo`/`W2ovvo` and `W1ovov`/`W2ovov` are gated SEPARATELY from their sums
because upstream reuses the `W1` halves alone inside `Wvvvo` and `Wovoo`
(`kintermediates_rhf.py:382-383`, `:424-426`). An error confined to a `W2` half
would move `Wovvo` and leave `Wovoo` right.

## Three transcription points worth naming

* **`ipccsd_matvec:97` carries upstream's own `# typo in Ref` comment** — the
  published equation (Nooijen and Snijders 1995) has a different index there and
  upstream's code is the correct one. Transcribed from the CODE.
* **`ipccsd_diag:199-201` and `eaccsd_diag:603-604` each subtract a term on a
  DIAGONAL, and only when two k-indices coincide** (`ki == kj`, `ka == kb`).
  Both are written as index loops here; an index-free port drops them silently,
  and on a fixture where the k-points happen to coincide the omission is
  invisible.
* **`mask_frozen` could be inherited in Python but not here.** The masking is
  identical to the spin-orbital version — same `r2` shape, same `kb`, same rule
  — and only the vector layout differs, so Python's `EOMIP.mask_frozen =
  eom_kgccsd.mask_frozen_ip` works through method dispatch. This port's packing
  functions are free functions, so the two masks are written out.

## NOT ported, and recorded

* **`EOMEESinglet`** (`:1425`) — a different vector shape
  (`vector_to_amplitudes_singlet`, `:849`) and a different matvec
  (`eeccsd_matvec_singlet`, `:969`, 250 lines). [`eom_kernel`] refuses
  `Excitation::Ee` naming the line.
* **`EOMEETriplet`** (`:1483`) and **`EOMEESpinFlip`** (`:1489`) are SHELLS
  upstream: their only body is `vector_size -> None`, and `EOMEE.vector_size`
  (`:1417`) raises. `16-CONTEXT §1.5` recorded this; nothing is deferred by this
  port that upstream implements.
* **`ipccsd_star_contract` / `eaccsd_star_contract`** (`:214`, `:617`) and the
  `*_Ta` variants (`:419`, `:819`) — as in 16-09.
* **The `partition` branches** (`'mp'`, `'full'`). Nothing sets `partition`, and
  both left matvecs `assert eom.partition is None` outright (`:110`, `:510`), so
  the `None` branch is the only one they admit.
