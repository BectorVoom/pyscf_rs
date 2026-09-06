# 16-09 — EOM-KCCSD over spin orbitals (`eom_kccsd_ghf.py`)

**Status: the three excitation types SHIP and are oracle-gated end to end.**
IP, EA and EE each have their vector packing, matvec, diagonal and Davidson
driver ported and compared against upstream PySCF 2.12.1 — at the equation
level on fixed synthetic amplitudes, and at the root level on upstream's own
converged amplitudes.

## Why GHF is first

`eom_kccsd_rhf.py:25` and `eom_kccsd_uhf.py:29` both
`import eom_kccsd_ghf as eom_kgccsd` and inherit its `EOMIP`/`EOMEA`/`EOMEE`.
`PBC-MASTER-PLAN §8.8` ordered the base class LAST; `16-CONTEXT §1` (point 4)
corrected that before any code was written, and this is where the correction
lands.

## What shipped

| Piece | Upstream | Notes |
|---|---|---|
| Ten EOM intermediates | `kintermediates.py:206-353` | `Foo`, `Fvv`, `Fov`, `Woooo`, `Wovvo`, `Wooov`, `Wvovv`, `Wvvvv`, `Wovoo`, `Wvvvo` |
| `EomImds` | `:1841-1966` | `make_shared` / `make_ip` / `make_ea`, with `Woooo` shared |
| IP | `:318-476`, `:663-682` | packing, matvec, left matvec, diagonal, `mask_frozen` |
| EA | `:865-1036`, `:1180-1199` | the same five |
| EE | `:1397-1689`, `:1788-1834` | packing, matvec, diagonal, both `kconserv_ee_*` |
| The Davidson driver | `:40-159`, `:1288-1385` | guess, `LARGE_DENOM` mask, preconditioner, quasiparticle weight |

## Gates (all measured)

| Gate | Measured | Set |
|---|---|---|
| ten EOM intermediates | `2.6e-9 … 4.7e-7` | `1e-6` |
| IP matvec / lmatvec / diag, every `kshift` | `1.6e-8 … 2.7e-7` | `1e-6` |
| EA matvec / lmatvec / diag, every `kshift` | `1.5e-8 … 4.0e-7` | `1e-6` |
| EE matvec / diag, every `kshift` | `2.3e-8 … 4.4e-7` | `1e-6` |
| IP / EA / EE vector round trip | **`0e0`** | exact |
| `ip_vector_size`, `ea_vector_size`, `ee_vector_size` | equal, every `kshift` | exact |
| `kconserv_ee_r1` / `kconserv_ee_r2` | equal, every `kshift` | exact |
| **IP roots** (2 shifts × 2) | `1.54e-10 … 4.59e-10` | `1e-5` |
| **EA roots** (2 shifts × 2) | `5.52e-10 … 6.77e-10` | `1e-5` |
| **EE roots** (2 shifts × 2) | `2.98e-9 … 4.94e-9` | `1e-5` |

The `1e-6` block gate is the FFT integral-transform floor the spin-orbital ERI
blocks already sit at (`2.4e-8 … 4.7e-7`), so the EOM layer adds nothing
measurable of its own. The `1e-5` root gate is `measurements/README.md §1`'s:
upstream's own spread over `conv_tol` and `nroots` on these roots reaches
`5.1e-7`, and its own suite asserts EOM roots at 3 decimals — a tighter gate
would fail a correct solver. Every measured residual sits four orders inside it,
and every root converged on both sides.

## Three things the gates exist to catch

**1. Each piece is gated separately, not only through an eigenvalue.** A root is
an eigenvalue of a matrix these ten intermediates assemble; an error above the
ERI floor but below `5.1e-7` would hide inside an end-to-end comparison. 16-06
made that argument and then needed it.

**2. The packings are asserted exactly.** The IP `r2` keeps only the strict
lower triangle of a `(nkpts·nocc)²` array, mirrored with a minus sign, so a port
storing the full square would agree on every matvec and be wrong only in
`vector_size` — which is what the Davidson allocates. The EA pair list is chosen
per `(kj, ka)` from whether `ka < kb`, so it depends on `kshift` while upstream's
`vector_size` is a closed form that does not.

**3. EE's vector LENGTH is `kshift`-dependent, and this fixture proves it:**
7360 elements at `kshift = 0`, 7296 at `kshift = 1`. Upstream's docstring warns
of this for even `nkpts` (`:1716`). The port computes the length by WALKING the
packing, which is right in both the odd and even cases by construction, and the
test asserts it equals upstream's number for every shift.

## Two transcription decisions worth naming

**`kconserv_ee_r2` is COMPOSED, not rebuilt.** Upstream constructs it
geometrically from the k-point coordinates; this port composes it from the
ordinary `kconserv` (`t = kconserv[k,l,m]`, then `kconserv[t, kshift, 0]`),
which is the same array only when `k_0 = 0`. Upstream makes the same assumption
in `get_kconserv_ee_r1` (literally `kconserv[:, kshift, 0]`) — but an assumption
two modules share is still an assumption, so the whole array is compared against
upstream's before anything uses it.

**`eeccsd_diag:1591` indexes `t2[kk, ka, ka]`** where the pattern of the other
three terms would give `t2[kk, ki, ka]`. It is transcribed AS WRITTEN. A port
that "fixed" it would disagree with the reference it is gated against; this is
upstream's number to change, and the divergence is recorded here rather than
silently corrected.

## Upstream's refusals, reproduced

* **`EOMEE.get_init_guess` raises for `koopmans=True`** (`:1749`, with a
  `# TODO do Koopmans later`), while IP and EA both implement it. The asymmetry
  is reproduced with the line named, per RULE 2.
* **`kernel_ee` does no frozen-orbital masking** (`:1294-1296`, `:1313`), unlike
  `kernel`. Masking anyway would silently change both the guess and the
  diagonal, so the port does what upstream does.
* **`EOMEE` declares no `l_matvec`** (`:1701-1704`), so there is no left EE.

## NOT ported, and recorded

* **`ipccsd_star_contract` / `eaccsd_star_contract`** (`:478-610`, `:1038-1168`)
  — the CCSD* perturbative corrections, ~250 lines, needing both left and right
  eigenvectors.
* **`EOMIP_Ta` / `EOMEA_Ta`** (`:760`, `:1277`) and `make_t3p2_ip`/`_ea`
  (`:1895`, `:1934`), which need `kintermediates.get_t3p2_imds_slow`
  (`:416-...`).
* **The `partition='mp'` branch** of both diagonals (`ipccsd_diag:449-457`,
  `eaccsd_diag:1010-1018`). Nothing in this phase sets `partition`, and porting
  a branch with no caller means shipping untested arithmetic. Upstream
  implements it, so it is out of scope rather than a refusal.
