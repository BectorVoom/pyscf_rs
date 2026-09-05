//! GTO-01: `format_atom` — convert `AtomInput` to the internal
//! `(symbol, [x, y, z])` representation in Bohr.
//!
//! Source: `pyscf/gto/mole.py:320-415` (Apache-2.0).
//!
//! Forms: `String`, `Tuples`, `TupleVec`, `FilePath`, `Callable`.
//! The `String`/`FilePath` forms accept BOTH the Cartesian (`Sym x y z`) and
//! the Z-matrix (`Sym ref bond [ref2 angle [ref3 dihedral]]`) layouts — the
//! block format is auto-detected exactly as upstream `format_atom` does.
//! Form 5 (`Callable`) invokes the user closure and resolves its produced spec
//! recursively (one level — see [`crate::types::AtomCallable`]), mirroring
//! upstream PySCF's callable `atom=` form (F-12).

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
        AtomInput::Callable(produce) => {
            // Invoke the user closure to obtain the concrete atom spec, then
            // resolve it through the same `format_atom` path so the produced
            // form gets identical unit/origin/axes treatment.
            let produced = produce()?;
            // One-level recursion guard: the callable must yield a concrete
            // spec, not another callable. This bounds the recursion and keeps
            // the "callable produces a concrete spec" contract (F-12 design
            // decision).
            if matches!(produced, AtomInput::Callable(_)) {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                    "atom callable returned another callable; nested callables are not supported"
                        .into(),
                )));
            }
            format_atom(&produced, unit, origin, axes)
        }
    }
}

/// Parse the upstream string form, supporting BOTH the Cartesian form
/// (`"H 0 0 0; O 0 0 1; H 0 0 2"`) and the Z-matrix form
/// (`"O; H 1 0.96; H 1 0.96 2 104.5"`).
///
/// Source: `pyscf/gto/mole.py:format_atom` (Apache-2.0).
///
/// Separator handling (applied before format detection):
///   - `;` is converted to `\n` (line separator)
///   - `,` and `\t` are converted to spaces (token separator)
///   - `#` introduces a line comment (anything after `#` on a line is dropped)
///
/// Format detection follows upstream exactly: after stripping blank/comment
/// lines, if the FIRST atom line has fewer than 4 whitespace tokens the entire
/// block is parsed as a Z-matrix; otherwise every line is Cartesian.
fn parse_atom_string(
    s: &str,
    unit: Unit,
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Result<Vec<ParsedAtom>, PyscfRsError> {
    let normalized = s.replace(';', "\n").replace([',', '\t'], " ");
    // Collect cleaned atom lines: strip the "# comment" suffix, trim, and drop
    // empty lines — mirroring upstream's `fmt_atoms` accumulation.
    let mut lines: Vec<&str> = Vec::new();
    for raw_line in normalized.lines() {
        let line = match raw_line.find('#') {
            Some(i) => &raw_line[..i],
            None => raw_line,
        };
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line);
        }
    }
    if lines.is_empty() {
        return Ok(Vec::new());
    }

    // Upstream detection: `len(fmt_atoms[0].split()) < 4` ⇒ Z-matrix block.
    let is_cartesian = lines[0].split_whitespace().count() >= 4;
    let atoms: Vec<ParsedAtom> = if is_cartesian {
        let mut atoms = Vec::with_capacity(lines.len());
        for line in lines {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 4 {
                // A Cartesian block requires `Sym x y z` on every line.
                return Err(invalid_coord(line));
            }
            let symb = atom_symbol(tokens[0])?;
            let xyz: [f64; 3] = [
                tokens[1].parse().map_err(|_| invalid_coord(line))?,
                tokens[2].parse().map_err(|_| invalid_coord(line))?,
                tokens[3].parse().map_err(|_| invalid_coord(line))?,
            ];
            atoms.push((symb, xyz));
        }
        atoms
    } else {
        from_zmatrix(&lines)?
    };
    Ok(apply_unit_origin_axes(atoms, unit, origin, axes))
}

/// Convert a Z-matrix atom block to Cartesian `(symbol, [x, y, z])` pairs in
/// the *input* length unit. The caller (`parse_atom_string`) applies
/// unit/origin/axes afterward, matching upstream `format_atom`, which
/// post-processes the Z-matrix output identically to the Cartesian path.
///
/// Faithful port of `pyscf/gto/mole.py::from_zmatrix` (+ `pyscf.symm.geom`
/// `rotation_mat`), Apache-2.0. Per-line shape (token count, not line index)
/// selects the construction, exactly as upstream:
///   - `Sym`                              → first atom at the origin
///   - `Sym ref bond`                     → second atom on +x at `bond`
///   - `Sym ref bond ref2 angle`          → third atom (bond + angle)
///   - `Sym ref bond ref2 angle ref3 dih` → general (bond + angle + dihedral)
///
/// `ref*` are 1-based indices into atoms defined earlier in the block; `bond`
/// is in the input unit; `angle`/`dihedral` are in degrees.
///
/// Intentional deviation: PySCF's `eval()` path (arbitrary numeric expressions
/// when `DISABLE_EVAL` is false) is NOT supported — coordinates must be plain
/// numbers, equivalent to upstream's `DISABLE_EVAL=True` behaviour.
fn from_zmatrix(lines: &[&str]) -> Result<Vec<ParsedAtom>, PyscfRsError> {
    use std::f64::consts::PI;

    let mut symbols: Vec<String> = Vec::with_capacity(lines.len());
    let mut coord: Vec<[f64; 3]> = Vec::with_capacity(lines.len());
    let mut min_items = 1usize;

    for line in lines {
        let rawd: Vec<&str> = line.split_whitespace().collect();
        if rawd.len() < min_items {
            return Err(zmat_err(line));
        }
        symbols.push(rawd[0].to_string());

        if rawd.len() < 3 {
            // First atom: at the origin.
            coord.push([0.0, 0.0, 0.0]);
            min_items = 3;
        } else if rawd.len() == 3 {
            // Second atom: on +x at distance `bond` (the reference is assumed
            // to be the first atom at the origin, as upstream does).
            let bond = parse_f64(rawd[2], line)?;
            coord.push([bond, 0.0, 0.0]);
            min_items = 5;
        } else if rawd.len() == 5 {
            // Third atom: bond length + bond angle, no dihedral.
            let bonda = parse_ref(rawd[1], coord.len(), line)?;
            let bond = parse_f64(rawd[2], line)?;
            let anga = parse_ref(rawd[3], coord.len(), line)?;
            let ang = parse_f64(rawd[4], line)? / 180.0 * PI;
            if ang < 0.0 {
                return Err(zmat_err(line));
            }
            let v1 = sub(coord[anga], coord[bonda]);
            let vecn = if near_zero_xy(v1) {
                [0.0, 0.0, 1.0]
            } else {
                cross(v1, [0.0, 0.0, 1.0])
            };
            let rmat = rotation_mat(vecn, ang);
            let scale = bond / norm(v1);
            let c = scale_vec(matvec(rmat, v1), scale);
            coord.push(add(coord[bonda], c));
            min_items = 7;
        } else {
            // Fourth atom onward: bond length + angle + dihedral.
            let bonda = parse_ref(rawd[1], coord.len(), line)?;
            let bond = parse_f64(rawd[2], line)?;
            let anga = parse_ref(rawd[3], coord.len(), line)?;
            let ang = parse_f64(rawd[4], line)? / 180.0 * PI;
            if !(0.0..=PI).contains(&ang) {
                return Err(zmat_err(line));
            }
            let v1 = normalize(sub(coord[anga], coord[bonda]));
            let c = if ang < 1e-7 {
                scale_vec(v1, bond)
            } else if PI - ang < 1e-7 {
                scale_vec(v1, -bond)
            } else {
                let diha = parse_ref(rawd[5], coord.len(), line)?;
                let dih = parse_f64(rawd[6], line)? / 180.0 * PI;
                let v2 = sub(coord[diha], coord[anga]);
                let vecn = cross(v2, scale_vec(v1, -1.0));
                let vecn_norm = norm(vecn);
                if vecn_norm < 1e-7 {
                    // Dihedral reference colinear with the bond axis ⇒ no usable
                    // plane; fall back to the bond+angle construction.
                    let vecn = if near_zero_xy(v1) {
                        [0.0, 0.0, 1.0]
                    } else {
                        cross(v1, [0.0, 0.0, 1.0])
                    };
                    let rmat = rotation_mat(vecn, ang);
                    scale_vec(matvec(rmat, v1), bond)
                } else {
                    let rmat = rotation_mat(v1, -dih);
                    let vecn = scale_vec(matvec(rmat, vecn), 1.0 / vecn_norm);
                    let rmat = rotation_mat(vecn, ang);
                    scale_vec(matvec(rmat, v1), bond)
                }
            };
            coord.push(add(coord[bonda], c));
        }
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (sym, xyz) in symbols.into_iter().zip(coord) {
        atoms.push((atom_symbol(&sym)?, xyz));
    }
    Ok(atoms)
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
            let s = [xyz[0] - origin[0], xyz[1] - origin[1], xyz[2] - origin[2]];
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

/// Nuclear charge for an atom symbol, ghost/label suffix stripped
/// (`"Cu1"` → `"Cu"`).
///
/// Delegates to [`pyscf_core::elements::charge_for_symbol`], the workspace's
/// single periodic table. This used to be a private Z≤36 stub, which made
/// [`atom_symbol`] reject every element above krypton.
pub fn charge_for_symbol(symb: &str) -> Option<i32> {
    pyscf_core::elements::charge_for_symbol(symb)
}

fn invalid_coord(line: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "could not parse coordinate from line: {line}"
    )))
}

// ── Z-matrix helpers ───────────────────────────────────────────────────────

fn zmat_err(line: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "Z-matrix format error at line: {line}"
    )))
}

/// Parse a plain `f64` token (no `eval` support — see `from_zmatrix` note).
fn parse_f64(token: &str, line: &str) -> Result<f64, PyscfRsError> {
    token.parse::<f64>().map_err(|_| zmat_err(line))
}

/// Parse a 1-based atom reference, validating it points at an atom already
/// placed (`1 ..= already_placed`); returns the 0-based index.
fn parse_ref(token: &str, already_placed: usize, line: &str) -> Result<usize, PyscfRsError> {
    let n: usize = token.parse().map_err(|_| zmat_err(line))?;
    if n < 1 || n > already_placed {
        return Err(zmat_err(line));
    }
    Ok(n - 1)
}

/// `numpy.allclose(v[:2], 0)` with the numpy default `atol = 1e-8`: true when
/// both the x and y components are within `1e-8` of zero (vector ∥ z-axis).
fn near_zero_xy(v: [f64; 3]) -> bool {
    v[0].abs() <= 1e-8 && v[1].abs() <= 1e-8
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale_vec(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n == 0.0 { v } else { scale_vec(v, 1.0 / n) }
}

fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Rodrigues axis-angle rotation matrix `R = cosθ·I + sinθ·[u]ₓ + (1-cosθ)·u⊗u`
/// for the normalized axis `u = vec/|vec|`. Port of `pyscf.symm.geom.rotation_mat`.
fn rotation_mat(vec: [f64; 3], theta: f64) -> [[f64; 3]; 3] {
    let u = normalize(vec);
    let (c, s) = (theta.cos(), theta.sin());
    let uu = [
        [u[0] * u[0], u[0] * u[1], u[0] * u[2]],
        [u[1] * u[0], u[1] * u[1], u[1] * u[2]],
        [u[2] * u[0], u[2] * u[1], u[2] * u[2]],
    ];
    let ux = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let eye = if i == j { 1.0 } else { 0.0 };
            r[i][j] = c * eye + s * ux[i][j] + (1.0 - c) * uu[i][j];
        }
    }
    r
}

#[cfg(test)]
mod zmatrix_tests {
    use super::*;
    use crate::types::AtomInput;
    use pyscf_core::Unit;

    const IDENT: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const ORIG: [f64; 3] = [0.0, 0.0, 0.0];

    /// Parse a Z-matrix string in Bohr (factor 1, no shift/rotation) so the
    /// result equals raw `from_zmatrix` output and can be pinned numerically.
    fn parse_zmat(s: &str) -> Vec<ParsedAtom> {
        format_atom(&AtomInput::String(s.into()), Unit::Bohr, ORIG, IDENT).unwrap()
    }

    fn assert_close(got: [f64; 3], want: [f64; 3], tol: f64) {
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() < tol,
                "coord[{k}]: got {}, want {} (Δ={:.3e})",
                got[k],
                want[k],
                (got[k] - want[k]).abs()
            );
        }
    }

    /// Upstream oracle: the exact example from `from_zmatrix`'s docstring
    /// (`pyscf/gto/mole.py`). Pins our port to PySCF's published output.
    #[test]
    fn from_zmatrix_docstring_oracle() {
        let s = "H\n\
                 H 1 2.67247631453057\n\
                 H 1 4.22555607338457 2 50.7684795164077\n\
                 H 1 2.90305235726773 2 79.3904651036893 3 6.20854462618583";
        let atoms = parse_zmat(s);
        assert_eq!(atoms.len(), 4);
        assert_close(atoms[0].1, [0.0, 0.0, 0.0], 1e-8);
        assert_close(atoms[1].1, [2.67247631, 0.0, 0.0], 1e-6);
        assert_close(atoms[2].1, [2.67247631, 0.0, 3.27310166], 1e-6);
        assert_close(atoms[3].1, [0.53449526, 0.30859098, 2.83668811], 1e-6);
        for a in &atoms {
            assert_eq!(a.0, "H");
        }
    }

    /// Water by Z-matrix: verify the two O–H bond lengths and the H–O–H angle
    /// match the requested internals (geometry invariants, not a fixed frame).
    #[test]
    fn from_zmatrix_water_internals() {
        let r = 1.8;
        let theta_deg = 104.5_f64;
        let s = format!("O\nH 1 {r}\nH 1 {r} 2 {theta_deg}");
        let atoms = parse_zmat(&s);
        assert_eq!(atoms.len(), 3);
        assert_eq!(atoms[0].0, "O");
        let o = atoms[0].1;
        let h1 = atoms[1].1;
        let h2 = atoms[2].1;
        let d1 = norm(sub(h1, o));
        let d2 = norm(sub(h2, o));
        assert!((d1 - r).abs() < 1e-9, "O-H1 = {d1}");
        assert!((d2 - r).abs() < 1e-9, "O-H2 = {d2}");
        // angle H1-O-H2
        let u = normalize(sub(h1, o));
        let v = normalize(sub(h2, o));
        let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let ang = dot.clamp(-1.0, 1.0).acos().to_degrees();
        assert!((ang - theta_deg).abs() < 1e-6, "H-O-H = {ang}");
    }

    /// A first line with < 4 tokens selects the Z-matrix path; one with ≥ 4
    /// keeps the Cartesian path. Both must round-trip the same molecule shape.
    #[test]
    fn cartesian_path_unaffected() {
        let atoms = parse_zmat("O 0 0 0; H 0 0 1.8; H 1.74 0 -0.45");
        assert_eq!(atoms.len(), 3);
        assert_eq!(atoms[0].0, "O");
        assert_close(atoms[1].1, [0.0, 0.0, 1.8], 1e-12);
        assert_close(atoms[2].1, [1.74, 0.0, -0.45], 1e-12);
    }

    /// Semicolons, the canonical Z-matrix separator, parse identically to
    /// newlines (upstream normalizes `;` → `\n` before detection).
    #[test]
    fn from_zmatrix_semicolon_separator() {
        let nl = parse_zmat("H\nH 1 1.4");
        let sc = parse_zmat("H; H 1 1.4");
        assert_eq!(nl.len(), 2);
        assert_eq!(sc.len(), 2);
        assert_close(nl[1].1, sc[1].1, 1e-12);
        assert_close(sc[1].1, [1.4, 0.0, 0.0], 1e-12);
    }

    /// An out-of-range reference is rejected (not silently misplaced). Note:
    /// the 2nd atom's reference is ignored by upstream (placed on +x), so the
    /// first validated reference is the 3rd atom's angle reference.
    #[test]
    fn from_zmatrix_bad_reference_rejected() {
        // Atom 3's angle reference (9) points past the 2 atoms defined so far.
        let r = format_atom(
            &AtomInput::String("H\nH 1 1.4\nH 1 1.4 9 90.0".into()),
            Unit::Bohr,
            ORIG,
            IDENT,
        );
        assert!(r.is_err(), "out-of-range reference must error, got {r:?}");
    }

    /// Unit conversion still applies to Z-matrix output (Angstrom → Bohr).
    #[test]
    fn from_zmatrix_unit_conversion_applies() {
        let bohr = parse_zmat("H\nH 1 1.0");
        let ang = format_atom(
            &AtomInput::String("H\nH 1 1.0".into()),
            Unit::Ang,
            ORIG,
            IDENT,
        )
        .unwrap();
        assert_close(bohr[1].1, [1.0, 0.0, 0.0], 1e-12);
        assert_close(ang[1].1, [1.8897261339213, 0.0, 0.0], 1e-9);
    }
}
