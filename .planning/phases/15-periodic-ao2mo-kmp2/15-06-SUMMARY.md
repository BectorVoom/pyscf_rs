# 15-06 — KUMP2 surface and staggered KMP2

**Status:** KUMP2 complete; staggered implementation complete with structural
tests, but its expensive live-PySCF energy oracle was not run.

KUMP2 provides the two-spin frozen/padding/count/mask surface and deliberately
refuses its energy kernel with the three upstream `NotImplementedError` line
references. `Kmp2Stagger` validates even meshes, constructs occupied/virtual
submeshes, supports an explicit full-mesh path with `vcut_sph`, and reuses the
KMP2 engine. Tests cover the refusal and staggered map invariants.
