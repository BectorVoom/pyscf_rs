//! Port of `pyscf/pbc/symm/tables.py:18-100` — `CrystalClass`, `LaueClass`,
//! `SchoenfliesNotation`. Three static lookup tables. Transcribed in the
//! exact order and content of the upstream Python dict literals: a
//! transcription error here is invisible until `PointGroup::group_name`
//! returns the wrong label, so `tests/group.rs` asserts a full table
//! comparison, not a spot check.
//!
//! [`SCHOENFLIES_NOTATION`]'s order is load-bearing: `PointGroup::group_index`
//! (`group.py:412-417`, `crate::group`) is
//! `list(SchoenfliesNotation.keys()).index(name)`, i.e. the index into this
//! exact insertion order — so the array order below must match
//! `tables.py:67-99` line for line, not merely contain the same 30 entries.

/// `tables.py:18-51` — the 30 crystallographic point (crystal) classes,
/// each keyed by its international (Hermann-Mauguin) symbol, with the
/// 10-element histogram `geom.py:149-216` builds and compares against:
/// `[-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]` counts of rotations by
/// `(trace, det)`.
pub const CRYSTAL_CLASS: &[(&str, [i32; 10])] = &[
    ("1", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ("-1", [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
    ("2", [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
    ("m", [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
    ("2/m", [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]),
    ("222", [0, 0, 0, 0, 0, 1, 3, 0, 0, 0]),
    ("mm2", [0, 0, 0, 2, 0, 1, 1, 0, 0, 0]),
    ("mmm", [0, 0, 0, 3, 1, 1, 3, 0, 0, 0]),
    ("4", [0, 0, 0, 0, 0, 1, 1, 0, 2, 0]),
    ("-4", [0, 2, 0, 0, 0, 1, 1, 0, 0, 0]),
    ("4/m", [0, 2, 0, 1, 1, 1, 1, 0, 2, 0]),
    ("422", [0, 0, 0, 0, 0, 1, 5, 0, 2, 0]),
    ("4mm", [0, 0, 0, 4, 0, 1, 1, 0, 2, 0]),
    ("-42m", [0, 2, 0, 2, 0, 1, 3, 0, 0, 0]),
    ("4/mmm", [0, 2, 0, 5, 1, 1, 5, 0, 2, 0]),
    ("3", [0, 0, 0, 0, 0, 1, 0, 2, 0, 0]),
    ("-3", [0, 0, 2, 0, 1, 1, 0, 2, 0, 0]),
    ("32", [0, 0, 0, 0, 0, 1, 3, 2, 0, 0]),
    ("3m", [0, 0, 0, 3, 0, 1, 0, 2, 0, 0]),
    ("-3m", [0, 0, 2, 3, 1, 1, 3, 2, 0, 0]),
    ("6", [0, 0, 0, 0, 0, 1, 1, 2, 0, 2]),
    ("-6", [2, 0, 0, 1, 0, 1, 0, 2, 0, 0]),
    ("6/m", [2, 0, 2, 1, 1, 1, 1, 2, 0, 2]),
    ("622", [0, 0, 0, 0, 0, 1, 7, 2, 0, 2]),
    ("6mm", [0, 0, 0, 6, 0, 1, 1, 2, 0, 2]),
    ("-6m2", [2, 0, 0, 4, 0, 1, 3, 2, 0, 0]),
    ("6/mmm", [2, 0, 2, 7, 1, 1, 7, 2, 0, 2]),
    ("23", [0, 0, 0, 0, 0, 1, 3, 8, 0, 0]),
    ("m-3", [0, 0, 8, 3, 1, 1, 3, 8, 0, 0]),
    ("432", [0, 0, 0, 0, 0, 1, 9, 8, 6, 0]),
    ("-43m", [0, 6, 0, 6, 0, 1, 3, 8, 0, 0]),
    ("m-3m", [0, 6, 8, 9, 1, 1, 9, 8, 6, 0]),
];

/// `tables.py:53-65` — the 11 Laue classes, each mapping to the ordered list
/// of crystal classes (from [`CRYSTAL_CLASS`]) that belong to it.
pub const LAUE_CLASS: &[(&str, &[&str])] = &[
    ("-1", &["1", "-1"]),
    ("2/m", &["2", "m", "2/m"]),
    ("mmm", &["222", "mm2", "mmm"]),
    ("4/m", &["4", "-4", "4/m"]),
    ("4/mmm", &["422", "4mm", "-42m", "4/mmm"]),
    ("-3", &["3", "-3"]),
    ("-3m", &["32", "3m", "-3m"]),
    ("6/m", &["6", "-6", "6/m"]),
    ("6/mmm", &["622", "6mm", "-6m2", "6/mmm"]),
    ("m-3", &["23", "m-3"]),
    ("m-3m", &["432", "-43m", "m-3m"]),
];

/// `tables.py:67-99` — international (Hermann-Mauguin) symbol -> Schoenflies
/// symbol, IN UPSTREAM'S EXACT INSERTION ORDER. See the module doc: this
/// order is read positionally by `PointGroup::group_index`.
pub const SCHOENFLIES_NOTATION: &[(&str, &str)] = &[
    ("1", "C1"),
    ("-1", "Ci"),
    ("2", "C2"),
    ("m", "Cs"),
    ("2/m", "C2h"),
    ("222", "D2"),
    ("mm2", "C2v"),
    ("mmm", "D2h"),
    ("4", "C4"),
    ("-4", "S4"),
    ("4/m", "C4h"),
    ("422", "D4"),
    ("4mm", "C4v"),
    ("-42m", "D2d"),
    ("4/mmm", "D4h"),
    ("3", "C3"),
    ("-3", "S6"),
    ("32", "D3"),
    ("3m", "C3v"),
    ("-3m", "D3d"),
    ("6", "C6"),
    ("-6", "C3h"),
    ("6/m", "C6h"),
    ("622", "D6"),
    ("6mm", "C6v"),
    ("-6m2", "D3h"),
    ("6/mmm", "D6h"),
    ("23", "T"),
    ("m-3", "Th"),
    ("432", "O"),
    ("-43m", "Td"),
    ("m-3m", "Oh"),
];

/// Look up the 10-entry rotation-count fingerprint for an international
/// crystal-class symbol.
pub fn crystal_class_table(name: &str) -> Option<&'static [i32; 10]> {
    CRYSTAL_CLASS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
}

/// Look up the Laue class a crystal class belongs to.
pub fn laue_class_for(crystal_class: &str) -> Option<&'static str> {
    LAUE_CLASS
        .iter()
        .find(|(_, members)| members.contains(&crystal_class))
        .map(|(k, _)| *k)
}

/// International (Hermann-Mauguin) symbol -> Schoenflies symbol.
pub fn schoenflies(name: &str) -> Option<&'static str> {
    SCHOENFLIES_NOTATION
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
}

/// `group.py:416` — `list(SchoenfliesNotation.keys()).index(name)`: the
/// position of an international symbol in [`SCHOENFLIES_NOTATION`]'s
/// insertion order.
pub fn group_index(name: &str) -> Option<usize> {
    SCHOENFLIES_NOTATION.iter().position(|(k, _)| *k == name)
}
