//! AO-direct CCSD — the on-the-fly `_contract_vvvv_t2` branch that replaces the
//! in-memory `vvvv` with an AO-space contraction (port of
//! `pyscf/cc/ccsd.py:473-570`; CCSD-07, `direct=True`).
//!
//! **STUB (Wave 4).** Reserves the module. No full in-tree analog: there is no
//! shell-sliced streaming `int2e` primitive in `pyscf-gto` yet (only the full
//! arity-4 tensor); v1 may compute the full tensor once and tile in MO space
//! (RESEARCH Open Q4 — confirm against the CCSD-07 `direct=True` contract).
#![allow(dead_code)]

// AO-direct _contract_vvvv_t2 lands in Wave 4.
