/// `df.addons.DEFAULT_AUXBASIS` — the Psi4 recommendations, verbatim from
/// vendored PySCF 2.12.1. `(canonical_orbital_basis, jkfit, rifit)`.
///
/// **Keys are `_format_basis_name`-CANONICAL** (lower-cased, `-`/`,`/`(`/`)`
/// and `*`/`+` mapped the way `canonicalise_basis_name` does), because that is
/// what `predefined_auxbasis` looks up. The dash-preserving table this crate
/// shipped in Phase 3 had both the wrong key convention AND a wrong `sto-3g`
/// row (`weigend`, where upstream says `def2-svp-jkfit`); plan 14-01 found it
/// because `make_auxbasis` on the He-fcc reference cell went through it.
pub const PSI4_AUXBASIS: [(&str, &str, &str); 27] = [
    ("321g", "def2-svp-jkfit", "def2-svp-ri"),
    ("631++g", "aug-cc-pvdz-jkfit", "aug-cc-pvdz-ri"),
    ("631+g", "heavy-aug-cc-pvdz-jkfit", "heavyaug-cc-pvdz-ri"),
    ("6311++g", "aug-cc-pvtz-jkfit", "aug-cc-pvtz-ri"),
    ("6311+g", "heavy-aug-cc-pvtz-jkfit", "heavyaug-cc-pvtz-ri"),
    ("6311g", "cc-pvtz-jkfit", "cc-pvtz-ri"),
    ("631g", "cc-pvdz-jkfit", "cc-pvdz-ri"),
    ("augccpv5z", "aug-cc-pv5z-jkfit", "aug-cc-pv5z-ri"),
    ("augccpvdz", "aug-cc-pvdz-jkfit", "aug-cc-pvdz-ri"),
    ("augccpvqz", "aug-cc-pvqz-jkfit", "aug-cc-pvqz-ri"),
    ("augccpvtz", "aug-cc-pvtz-jkfit", "aug-cc-pvtz-ri"),
    ("ccpv5z", "cc-pv5z-jkfit", "cc-pv5z-ri"),
    ("ccpvdz", "cc-pvdz-jkfit", "cc-pvdz-ri"),
    ("ccpvqz", "cc-pvqz-jkfit", "cc-pvqz-ri"),
    ("ccpvtz", "cc-pvtz-jkfit", "cc-pvtz-ri"),
    ("def2mtzvp", "def2-tzvp-jkfit", "def2-tzvp-ri"),
    ("def2mtzvpp", "def2-tzvpp-jkfit", "def2-tzvpp-ri"),
    ("def2qzvp", "def2-qzvp-jkfit", "def2-qzvp-ri"),
    ("def2qzvpp", "def2-qzvpp-jkfit", "def2-qzvpp-ri"),
    ("def2qzvppd", "def2-qzvpp-jkfit", "def2-qzvppd-ri"),
    ("def2svp", "def2-svp-jkfit", "def2-svp-ri"),
    ("def2svpd", "def2-svp-jkfit", "def2-svpd-ri"),
    ("def2tzvp", "def2-tzvp-jkfit", "def2-tzvp-ri"),
    ("def2tzvpd", "def2-tzvp-jkfit", "def2-tzvpd-ri"),
    ("def2tzvpp", "def2-tzvpp-jkfit", "def2-tzvpp-ri"),
    ("def2tzvppd", "def2-tzvpp-jkfit", "def2-tzvppd-ri"),
    ("sto3g", "def2-svp-jkfit", "def2-svp-ri"),
];
