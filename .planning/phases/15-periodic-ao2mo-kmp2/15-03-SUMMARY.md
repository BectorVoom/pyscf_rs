# 15-03 — frozen orbitals, padding, and the SCF/MO seam

**Status:** complete.

Restricted frozen-core bookkeeping, fractional-occupation refusal, split and
joint padding, padded energies/coefficients, and frozen masks now mirror
`kmp2.py`. Occupied columns are bottom-aligned and virtual columns top-aligned.
`moref.rs` owns the only SCF column-major to AO2MO row-major conversion and its
test uses a non-symmetric matrix so an accidental transpose cannot pass.
