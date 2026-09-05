# 15-02 — KptsHelper

**Status:** complete.

`pyscf-pbc-lib` now exports `KptsHelper`, its insertion-ordered `symm_map`, the
four ERI operations, and `transform_symm`. Existing `Kconserv` and
`Kconserv3` remain the single conservation implementation; `ktensor.py`
remains Phase 17 work. Three external tests cover construction, operation
closure, and transformed values.
