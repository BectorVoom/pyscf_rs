//! Atomic radii tables — port of `pyscf/data/radii.py` (+ `radi.py:SG1RADII`).
//!
//! Source (Apache-2.0): `pyscf/data/radii.py` `BRAGG` / `COVALENT`
//! (JCP 41, 3199 (1964) Bragg-Slater; Cordero et al. covalent radii) and
//! `pyscf/dft/radi.py:SG1RADII` (Gill/Johnson/Pople, CPL 209 (1993) 506).
//! D-05 / D-06: small inline `const` slices, looked up by nuclear charge Z;
//! NO codegen, NO `build.rs`, NO runtime file reads.
//!
//! Byte-exactness note (Pitfall 10): upstream defines the Bragg/Covalent
//! tables in Angstrom and converts to Bohr with `1/BOHR * numpy.array(...)`,
//! i.e. element-wise `(1.0 / BOHR) * a_i`. We store the Angstrom literals
//! verbatim and reproduce the SAME `(1.0 / BOHR) * a` arithmetic per element
//! in [`bragg_radii`] / [`covalent_radii`], so the resulting f64 bit pattern
//! matches upstream under the FMA-free `release-oracle` profile. `SG1RADII` is
//! already in Bohr upstream (no conversion), so it is stored directly.

/// CODATA Bohr radius in Angstrom — `pyscf/data/nist.py:BOHR`.
/// Upstream `radii.py` uses `from pyscf.lib.parameters import BOHR`, which
/// re-exports this exact value.
pub const BOHR: f64 = 0.52917721092;

/// Sentinel for "unknown" element radius — upstream `radii.unknown`
/// (= 1.999999 Angstrom). Used for Z=0 (ghost) and rows with no tabulated
/// radius.
const UNKNOWN_ANG: f64 = 1.999999;

/// Bragg-Slater atomic radii in **Angstrom**, indexed by nuclear charge Z
/// (index 0 = ghost atom). Port of `pyscf/data/radii.py:BRAGG` (the array
/// argument BEFORE the `1/BOHR` scale). JCP 41, 3199 (1964).
#[rustfmt::skip]
const BRAGG_ANG: &[f64] = &[
    // 131 entries (ghost + Z=1..130), matching upstream `BRAGG` exactly.
    UNKNOWN_ANG, 0.35, 1.40, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50,
    1.50, 1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80, 2.20,
    1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35,
    1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.90, 2.35, 2.00, 1.80,
    1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55,
    1.45, 1.45, 1.40, 1.40, 2.10, 2.60, 2.15, 1.95, 1.85, 1.85,
    1.85, 1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35,
    1.50, 1.90, 1.80, 1.60, 1.90, 1.45, 2.10, 1.80, 2.15, 1.95,
    1.80, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75,
];

/// Covalent atomic radii in **Angstrom**, indexed by Z (0 = ghost).
/// Port of `pyscf/data/radii.py:COVALENT` (pre-`1/BOHR` array). Cordero et
/// al., Dalton Trans., 2008, 2832-2838.
#[rustfmt::skip]
const COVALENT_ANG: &[f64] = &[
    UNKNOWN_ANG,                                                       // Ghost
    0.31,                                                       0.28,  // 1s
    1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,                    // 2s2p
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,                    // 3s3p
    2.03, 1.76,                                                        // 4s
    1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22,        // 3d
                1.22, 1.20, 1.19, 1.20, 1.20, 1.16,                    // 4p
    2.20, 1.95,                                                        // 5s
    1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,        // 4d
                1.42, 1.39, 1.39, 1.38, 1.39, 1.40,                    // 5p
    2.44, 2.15,                                                        // 6s
    2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                          // La, Ce-Eu
    1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,                    // Gd, Tb-Lu
          1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,        // 5d
                1.45, 1.46, 1.48, 1.40, 1.50, 1.50,                    // 6p
    2.60, 2.21,                                                        // 7s
    2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
];

/// SG1 atomic radii in **Bohr**, indexed by Z (0 = ghost = 0.0).
/// Port of `pyscf/dft/radi.py:SG1RADII` (already in Bohr — no `1/BOHR`
/// scaling upstream). Gill/Johnson/Pople, CPL 209 (1993) 506-512.
/// Defined for Z = 0..18 (H..Ar) only — matches upstream length.
#[rustfmt::skip]
pub const SG1RADII: &[f64] = &[
    0.0,
    1.0000,                                                         0.5882,
    3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
    4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333,
];

/// Bragg-Slater radius (Bohr) for nuclear charge `z`. Reproduces upstream's
/// element-wise `(1.0 / BOHR) * angstrom` conversion exactly (byte-for-byte
/// under `release-oracle`). Out-of-range Z falls back to the `unknown`
/// radius, mirroring numpy fancy-indexing semantics for the tabulated range.
#[must_use]
pub fn bragg_radii(z: usize) -> f64 {
    let ang = BRAGG_ANG.get(z).copied().unwrap_or(UNKNOWN_ANG);
    (1.0 / BOHR) * ang
}

/// Covalent radius (Bohr) for nuclear charge `z`. Same `(1.0 / BOHR) * a`
/// element-wise conversion as upstream `COVALENT`.
#[must_use]
pub fn covalent_radii(z: usize) -> f64 {
    let ang = COVALENT_ANG.get(z).copied().unwrap_or(UNKNOWN_ANG);
    (1.0 / BOHR) * ang
}

/// SG1 radius (Bohr) for nuclear charge `z`. Already in Bohr upstream; Z
/// outside the tabulated H..Ar range returns 0.0 (matches the upstream
/// `SG1RADII[nuc]` index range — callers add the `1e-200` guard themselves).
#[must_use]
pub fn sg1_radii(z: usize) -> f64 {
    SG1RADII.get(z).copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bragg_ghost_and_known_elements_match_upstream() {
        // Z=0 ghost = unknown / BOHR.
        assert_eq!(bragg_radii(0), (1.0 / BOHR) * UNKNOWN_ANG);
        // Bragg Angstrom literals: H(1)=0.35, C(6)=0.70, N(7)=0.65, O(8)=0.60.
        assert_eq!(bragg_radii(1), (1.0 / BOHR) * 0.35);
        assert_eq!(bragg_radii(6), (1.0 / BOHR) * 0.70);
        assert_eq!(bragg_radii(7), (1.0 / BOHR) * 0.65);
        assert_eq!(bragg_radii(8), (1.0 / BOHR) * 0.60);
    }

    #[test]
    fn covalent_known_elements_match_upstream() {
        // Covalent Angstrom literals: H(1)=0.31, C(6)=0.73, O(8)=0.66.
        assert_eq!(covalent_radii(1), (1.0 / BOHR) * 0.31);
        assert_eq!(covalent_radii(6), (1.0 / BOHR) * 0.73);
        assert_eq!(covalent_radii(8), (1.0 / BOHR) * 0.66);
    }

    #[test]
    fn bragg_table_covers_full_periodic_table() {
        // Upstream BRAGG has 131 entries (ghost + Z=1..130).
        assert_eq!(BRAGG_ANG.len(), 131);
    }

    #[test]
    fn covalent_table_length_matches_upstream() {
        // Upstream COVALENT: 1 (ghost) + 96 entries = 97.
        assert_eq!(COVALENT_ANG.len(), 97);
    }

    #[test]
    fn sg1_table_covers_h_to_ar() {
        // 1 (ghost) + 18 (H..Ar) = 19 entries.
        assert_eq!(SG1RADII.len(), 19);
        assert_eq!(sg1_radii(1), 1.0000);
        assert_eq!(sg1_radii(18), 1.3333);
        // Out of range → 0.0.
        assert_eq!(sg1_radii(50), 0.0);
    }
}
