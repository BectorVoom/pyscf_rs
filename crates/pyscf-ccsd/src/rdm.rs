//! Closed-shell CCSD reduced density matrices — `make_rdm1`/`make_rdm2` +
//! `_gamma1`/`_gamma2_intermediates` + the `ao_repr=True` nmo⁴ AO
//! back-transform (port of `pyscf/cc/ccsd_rdm.py`).
//!
//! **STUB (Wave 3).** Reserves the module. The `ao_repr=True` branch routes the
//! nmo⁴ contraction through `pyscf-ao2mo` (the heaviest arena tenant). Same
//! materialize-then-`oracle_*` reduction discipline.
#![allow(dead_code)]

// make_rdm1 / make_rdm2 / gamma intermediates land in Wave 3.
