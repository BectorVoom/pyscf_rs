//! `AmplitudeSubspace: DiisStorable` — the CCSD amplitude DIIS storable
//! (`amplitudes_to_vector`/`vector_to_amplitudes`, `ccsd.py:670`/`:679`).
//!
//! **STUB (Wave 2).** Reserves the module. The implementation reuses the
//! generic `pyscf_diis::Diis<S>` machinery (NO new DIIS body) with
//! `Diis::<AmplitudeSubspace>::new(6)` (`diis_space=6`, NOT SCF's 8). The
//! `DiisStorable::dot` MUST route through `pyscf_algebra::oracle_dot`
//! (Pitfall 9). Packing reproduces the upstream lower-triangular `t2`
//! symmetric pack (`t2[iajb] == t2[jbia]`).
#![allow(dead_code)]

// AmplitudeSubspace + its DiisStorable impl land in Wave 2 (after the in-core
// RCCSD kernel — DIIS needs the kernel loop).
