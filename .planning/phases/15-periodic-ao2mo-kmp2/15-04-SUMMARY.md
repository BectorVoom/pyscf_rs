# 15-04 — PeriodicDf AO2MO dispatch and Lov

**Status:** functionality complete; the planned 1-vs-8-thread Lov timing was
not measured.

Every `PeriodicDf` implementor exposes AO2MO through the trait, including RSDF
delegation. `oracle_zdotu`/`oracle_zdotu_re` provide the required unconjugated
ordered reduction. `LovTable` stores `(ia,L)` with `L` contiguous and reuses
the existing `r_e2`. The legacy `pyscf-pbc-ao2mo::eris` surface delegates to
the same implementations. During route testing, `sr_loop` was corrected to
reconstruct upper k-pairs from the canonical lower pair by conjugate AO-index
transpose, matching PySCF's `_load3c` contract.
