//! GTO-01: `format_atom` — convert `AtomInput` to the internal
//! `(symbol, [x, y, z])` representation in Bohr.
//!
//! Source: `pyscf/gto/mole.py:320-415` (Apache-2.0).
//!
//! Forms shipping in Phase 2: `String`, `Tuples`, `TupleVec`, `FilePath`.
//! Form 5 (`Callable`) returns `NotYetImplemented{phase:3}` per Deferred
//! Ideas (`02-CONTEXT.md`).

use crate::types::AtomInput;
use pyscf_core::{CoreError, ParsedAtom, PyscfRsError, Unit};
use std::path::Path;

/// Convert any `AtomInput` form to the internal
/// `Vec<(symbol, [x, y, z]_in_Bohr)>`.
///
/// Applies (in order):
///   1. Form-specific parsing (string / tuples / tuple-vec / file).
///   2. Origin shift (`coords - origin`).
///   3. Axes rotation (`(coords - origin) @ axes` — upstream convention).
///   4. Unit conversion to Bohr.
pub fn format_atom(
    input: &AtomInput,
    unit: Unit,
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Result<Vec<ParsedAtom>, PyscfRsError> {
    match input {
        AtomInput::String(s) => parse_atom_string(s, unit, origin, axes),
        AtomInput::Tuples(t) => Ok(apply_unit_origin_axes(
            t.iter().map(|(s, c)| (s.clone(), *c)).collect(),
            unit,
            origin,
            axes,
        )),
        AtomInput::TupleVec(t) => {
            let mut atoms = Vec::with_capacity(t.len());
            for (s, c) in t {
                if c.len() != 3 {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "atom '{s}' has {} coords, expected 3",
                        c.len()
                    ))));
                }
                atoms.push((s.clone(), [c[0], c[1], c[2]]));
            }
            // Validate atom symbols (so TupleVec gets the same normalisation +
            // unknown-symbol rejection that String form gets).
            let mut normalised = Vec::with_capacity(atoms.len());
            for (s, xyz) in atoms {
                let canonical = atom_symbol(&s)?;
                normalised.push((canonical, xyz));
            }
            Ok(apply_unit_origin_axes(normalised, unit, origin, axes))
        }
        AtomInput::FilePath(p) => parse_atom_file(p, unit, origin, axes),
        AtomInput::Callable => Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "atom callable form (GTO-01.5)",
        }),
    }
}

/// Parse the upstream string form: `"H 0 0 0; O 0 0 1; H 0 0 2"`.
///
/// Source: `pyscf/gto/mole.py:373-392`.
///
/// Separator handling:
///   - `;` is converted to `\n` (line separator)
///   - `,` and `\t` are converted to spaces (token separator)
///   - `#` introduces a line comment (anything after `#` on a line is dropped)
fn parse_atom_string(
    s: &str,
    unit: Unit,
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Result<Vec<ParsedAtom>, PyscfRsError> {
    let normalized = s.replace(';', "\n").replace(',', " ").replace('\t', " ");
    let mut atoms = Vec::new();
    for raw_line in normalized.lines() {
        // Strip "# comment" suffix.
        let line = match raw_line.find('#') {
            Some(i) => &raw_line[..i],
            None => raw_line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            // Z-matrix form (3 tokens per atom referencing previous atoms).
            // Phase 2 stub: NotYetImplemented.
            return Err(PyscfRsError::NotYetImplemented {
                phase: 2,
                what: "Z-matrix atom form (deferred to Phase 2.x)",
            });
        }
        let symb = atom_symbol(tokens[0])?;
        let xyz: [f64; 3] = [
            tokens[1].parse().map_err(|_| invalid_coord(line))?,
            tokens[2].parse().map_err(|_| invalid_coord(line))?,
            tokens[3].parse().map_err(|_| invalid_coord(line))?,
        ];
        atoms.push((symb, xyz));
    }
    Ok(apply_unit_origin_axes(atoms, unit, origin, axes))
}

/// Read an atom file (line-per-atom format like `.xyz`, but skipping the
/// `.xyz` header lines if detected).
///
/// Source: `pyscf/gto/mole.py:393-415`.
///
/// Note: `.xyz` files are conventionally Angstrom; the caller's `unit`
/// argument is honoured verbatim — if the user passes `Bohr` for an `.xyz`
/// file, the user is responsible for that mismatch.
fn parse_atom_file(
    path: &Path,
    unit: Unit,
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Result<Vec<ParsedAtom>, PyscfRsError> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "read atom file {}: {}",
            path.display(),
            e
        )))
    })?;
    // Detect .xyz format: extension is .xyz, OR the first line is purely
    // an integer (the atom count).
    let is_xyz_ext = path.extension().and_then(|s| s.to_str()) == Some("xyz");
    let first_line_is_count = contents
        .trim_start()
        .lines()
        .next()
        .and_then(|l| l.trim().parse::<usize>().ok())
        .is_some();
    let body = if is_xyz_ext || first_line_is_count {
        // Skip the count line + comment line.
        contents.lines().skip(2).collect::<Vec<_>>().join("\n")
    } else {
        contents
    };
    parse_atom_string(&body, unit, origin, axes)
}

/// Apply unit conversion + origin shift + axes rotation.
///
/// Upstream convention (`pyscf/gto/mole.py:362`):
/// `coords_new = (coords - origin) @ axes` (numpy `@`),
/// i.e. `coords_new[i][k] = sum_j (coords[i][j] - origin[j]) * axes[j][k]`.
fn apply_unit_origin_axes(
    atoms: Vec<ParsedAtom>,
    unit: Unit,
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Vec<ParsedAtom> {
    let f = unit.length_in_au();
    atoms
        .into_iter()
        .map(|(symb, xyz)| {
            let s = [
                xyz[0] - origin[0],
                xyz[1] - origin[1],
                xyz[2] - origin[2],
            ];
            // r[k] = sum_j s[j] * axes[j][k]
            let r = [
                f * (s[0] * axes[0][0] + s[1] * axes[1][0] + s[2] * axes[2][0]),
                f * (s[0] * axes[0][1] + s[1] * axes[1][1] + s[2] * axes[2][1]),
                f * (s[0] * axes[0][2] + s[1] * axes[1][2] + s[2] * axes[2][2]),
            ];
            (symb, r)
        })
        .collect()
}

/// Normalise an atom symbol token.
///
/// Source: `pyscf/gto/mole.py:299-318` `_atom_symbol`.
///
/// Examples:
///   - `"H"` → `"H"`
///   - `"H1"` → `"H1"` (suffix preserved per upstream "ghost atom" convention)
///   - `"h"` → `"H"` (canonicalised first-upper-rest-lower)
///   - `"Hh3.5%@"` → rejected (unknown element prefix `"Hh"`)
pub(crate) fn atom_symbol(token: &str) -> Result<String, PyscfRsError> {
    let s = token.trim();
    if s.is_empty() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "empty atom symbol".into(),
        )));
    }
    // Find the alphabetic prefix (allowing internal `-` / `_` for "GHOST-X" forms).
    let alpha_end = s
        .find(|c: char| !c.is_alphabetic() && c != '-' && c != '_')
        .unwrap_or(s.len());
    if alpha_end == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "atom symbol '{s}' must start with a letter"
        ))));
    }
    let leading = &s[..alpha_end];
    // Capitalise: first char upper, rest lower.
    let canonical_leading: String = leading
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 0 {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect();
    if charge_for_symbol(&canonical_leading).is_none() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "unknown element symbol '{canonical_leading}' in token '{s}'"
        ))));
    }
    // Preserve any trailing digit / dash suffix (ghost-atom convention).
    let suffix = &s[alpha_end..];
    Ok(format!("{}{}", canonical_leading, suffix))
}

/// Strip the suffix from an atom symbol and look up the leading-symbol's
/// nuclear charge.
///
/// Phase 2 minimal table covering Z=1..36 + ghost. Full table lives in
/// `pyscf/data/elements.py` `ELEMENTS_PROTON` (~118 entries); Phase 3 PyO3
/// can pull the full table at construction time.
pub fn charge_for_symbol(symb: &str) -> Option<i32> {
    let alpha_end = symb
        .find(|c: char| !c.is_alphabetic())
        .unwrap_or(symb.len());
    let leading = &symb[..alpha_end];
    match leading {
        "H" => Some(1),
        "He" => Some(2),
        "Li" => Some(3),
        "Be" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "Ne" => Some(10),
        "Na" => Some(11),
        "Mg" => Some(12),
        "Al" => Some(13),
        "Si" => Some(14),
        "P" => Some(15),
        "S" => Some(16),
        "Cl" => Some(17),
        "Ar" => Some(18),
        "K" => Some(19),
        "Ca" => Some(20),
        "Sc" => Some(21),
        "Ti" => Some(22),
        "V" => Some(23),
        "Cr" => Some(24),
        "Mn" => Some(25),
        "Fe" => Some(26),
        "Co" => Some(27),
        "Ni" => Some(28),
        "Cu" => Some(29),
        "Zn" => Some(30),
        "Ga" => Some(31),
        "Ge" => Some(32),
        "As" => Some(33),
        "Se" => Some(34),
        "Br" => Some(35),
        "Kr" => Some(36),
        // Ghost atom convention upstream: "GHOST" / "X" → 0 charge.
        "GHOST" | "X" | "Ghost" | "ghost" => Some(0),
        _ => None,
    }
}

fn invalid_coord(line: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "could not parse coordinate from line: {line}"
    )))
}
