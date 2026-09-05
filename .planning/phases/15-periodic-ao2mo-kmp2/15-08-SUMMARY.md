# 15-08 — MO-first FFTDF/AFTDF AO2MO

**Status:** correctness complete; dedicated before/after wall-clock benchmark
not measured.

FFTDF now transforms AO values to MOs on the real-space grid before pair
formation, and AFTDF transforms analytical AO-pair slices before the four-index
contraction. The old AO-first functions remain available as references.
Gamma and non-gamma tests compare every conserving tuple, and cache/no-cache
results are bit-identical. The work also exposed and fixed the old non-gamma
FFT AO-ERI reciprocal-bin permutation bug; the corrected He value agrees with
live PySCF 2.12.1.
