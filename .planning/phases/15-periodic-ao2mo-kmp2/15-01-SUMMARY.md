# 15-01 — measured references

**Status:** core measurements complete; the oversized optional matrix was not run.

Vendored PySCF 2.12.1 reproduced the diamond FFTDF anchor at
`-0.20472143304034024 Ha`, only `2.11344e-10 Ha` from the constant embedded in
upstream's source. He/6-31g `[1,1,2]` measured FFTDF/MO-first at
`-0.033241446759957924 Ha` and GDF/Lov at `-0.016989369077568279 Ha`; forced
GDF AO2MO differs by `3.469e-18 Ha`. The small-fixture kernel-time ratio was
5.6313x in GDF/Lov's favour. The synthetic ragged padding oracle is committed
in `measurements/padding.out`.

The measured headline tolerance is `2e-6 Ha`: it rejects normalization and
`exxdiv` mistakes while accommodating the independent SCF paths. The original
`1e-14` and `1e-8` guesses are superseded.
