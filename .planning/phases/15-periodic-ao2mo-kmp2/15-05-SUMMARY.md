# 15-05 — KMP2 and density matrices

**Status:** implementation complete; headline FFTDF gate met, GDF upstream
gate records an inherited numerical baseline difference.

The deterministic two-pass kernel implements both Lov and four-index routes,
both `1/nkpts` factors, `LARGE_DENOM = 1e14`, SS/OS tagging, optional T2, a
rayon outer loop, cache reuse, and a thread-aware memory preflight. RDM1/RDM2
and gamma1 intermediates are exposed in `krdm.rs`. The oracle-free He/6-31g
primitive `[1,1,2]`/gamma-supercell identity passes within `2e-8 Ha`.

The diamond FFTDF result meets the measured gate. On He/6-31g, the FFTDF route
matches upstream within `2e-6 Ha`; local GDF Lov and AO2MO agree within
`2e-15 Ha`, while the local GDF result is `-0.015572369890603862 Ha` versus
upstream `-0.016989369077568279 Ha`. This is reported as an inherited GDF
integral/SCF baseline gap, not absorbed into the Phase 15 tolerance.
