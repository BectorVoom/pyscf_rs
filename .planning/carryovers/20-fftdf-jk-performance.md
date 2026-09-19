# Carryover — native FFTDF `get_jk` is 5.95× slower than upstream (quiet box)

**Source:** `.planning/phases/20-pbc-python-bindings/20-VERIFICATION.md` §3.5 (20-18 Task 5,
2026-09-14). No fix attempted in Phase 20.

## Measured

One `FFTDF.get_jk(dm, hermi=1, kpts, exxdiv='ewald')`, example-22 diamond cell (`gth-szv` /
`gth-pade`, Bohr), 2×2×2, default mesh **[47,47,47]** both sides, the same complex128 density
(upstream `KRHF.get_init_guess()`), two warm worker processes driven alternately, default threads
(16 cores), `.so` 2026-09-14 19:52:34, vendored 2.12.1 asserted. Load average 1.91 and falling at
start; only the two workers running during the measurement.

| rep | native (s) | upstream (s) |
|---|---:|---:|
| 0 warm-up | 116.783 | 19.360 |
| 1 | 114.668 | 19.060 |
| 2 | 114.441 | 19.088 |
| 3 | 113.660 | 19.423 |
| mean 1–3 | **114.256** | **19.190** |

**5.95×** (5.85–6.02×). `|vj|`, `|vk|` agree to 13 printed digits. Scripts and log:
`target/p20-18-final/jk_{controller,worker}.py`, `jk_perf.log`.

Consistent with earlier loaded-box probes (`measurements/example-gate-gap-scope.md` §3): 115.5 s
cold / 120.6 s warm vs upstream 12–24 s on this cell, and 95–102 s vs 13.0 s (mo-tagged) on the
8-atom example-20 cell at 2 k / 65³ (7.3×). Upstream's mo-tagged density path (`dm.mo_coeff`
present) is a further ~1.5× faster than the plain-dm path timed here.

## Consequences

- `examples/pbc/22-k_points_mp2.py` through line 58 took 1637.7 s natively (loaded box) vs 554 s
  upstream for the whole script (upstream KRHF 2×2×2 126 s).
- Example 20 (64 k, K ∝ nk²) is ~27 h per native JK call vs ~3.7 h upstream.
- `examples/pbc/40-custom_gdf.py` (GDF, not FFTDF) was also much slower natively — see
  `20-VERIFICATION.md` §3.2 for its wall time.

## Unblock

1. Profile one native `get_k_kpts` on this fixture (host spans + CubeCL launch counts; memory
   `lazy-launches-blur-stage-spans`, `zgemm-dense-loses-to-host-rayon` — warm the backend).
2. Port upstream's mo-tagged K path (`fft_jk.get_k_kpts` with `mo_coeff`/`mo_occ`: contract
   occupied orbitals, not the full density) — the biggest algorithmic difference.
3. Gate speed as a same-binary A/B, bit-exact against the current path.

## GDF exchange is worse still (measured by 20-18 Task 2, item 4)

- `examples/pbc/40-custom_gdf.py` (diamond `sto3g` all-electron, 2×2×2; GDF `_cderi_to_save`,
  KRHF from `_cderi`, KRKS PBE from `_cderi`) runs in **53.5 s** upstream (RSS 0.33 GB). Natively
  the `_cderi` file (7.4 MB) was written after ~3 min, then line 35 `mf.run()` (KRHF over the loaded
  GDF) was still computing when killed at **4927 s** (82 min; `/proc/PID/io` flat, ~8 cores, RSS
  0.91 GB; Python stack captured with gdb at 61 min).
- Probe `target/p20-18-final/ex40_probe.py`: ONE native `GDF.get_jk(init_guess, kpts,
  exxdiv='ewald')` on the same cell and `_cderi` did not return within `timeout 580` s (run
  concurrently with the 82-min process).
- Not diagnosed. Candidates to measure first: the K contraction over the loaded `cderi` (per-(k,k')
  block reads vs upstream's `sr_loop` blocking), the `ewald` exxdiv probe on GDF, and whether a
  loaded `_cderi` takes a different (uncached) path from a freshly built one (20-10 measured GDF
  `get_jk` bitwise between saved/loaded builders, not their cost).
