//! Redundant internal coordinates — bond / angle / dihedral primitives + the
//! connectivity graph from interatomic distances.
//!
//! **Algorithm ported from geomeTRIC** (`internal.py`, the `Distance` /
//! `Angle` / `Dihedral` primitive-coordinate classes + the bonding-graph
//! construction). geomeTRIC is "BSD 3-clause (aka BSD 2.0) Non-AI License"
//! (clauses 1-3 are the standard permissive BSD-3-Clause; clause 4 is an
//! additional anti-AI-training restriction) — the mathematical algorithm is
//! re-derived here in Rust (not a verbatim source copy). See the crate-level
//! license record in `lib.rs`.
//!
//! A *primitive internal coordinate* is a scalar function `q(x)` of the
//! Cartesian geometry `x ∈ ℝ^{3·natm}`:
//!   - `Distance(a,b)`  = |r_a − r_b|
//!   - `Angle(a,b,c)`   = ∠(r_a − r_b, r_c − r_b)  (b is the apex)
//!   - `Dihedral(a,b,c,d)` = the torsion about the b–c axis
//!
//! The redundant-internal set is generated from the *bonding graph*: a bond
//! between every atom pair whose distance is below a covalent-radius-scaled
//! threshold, then every angle a–b–c (b bonded to both a and c), then every
//! dihedral across a connected b–c bond. "Redundant" = more internals than the
//! `3·natm − 6` degrees of freedom; the Wilson `G = B Bᵀ` pseudo-inverse
//! (`bmatrix.rs`) projects onto the non-redundant subspace.
//!
//! Per Pitfall 1/2 every reduction materialises then routes through
//! `pyscf_algebra::oracle_sum` — no bare `+=` in any numeric accumulation.

use pyscf_algebra::oracle_sum;

/// A primitive redundant-internal coordinate (geomeTRIC `internal.py` analog).
///
/// Atom indices reference rows of the `(natm, 3)` Cartesian geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    /// Bond stretch `|r_a − r_b|` (Bohr).
    Distance(usize, usize),
    /// Bond angle a–b–c with `b` the apex (radians).
    Angle(usize, usize, usize),
    /// Dihedral / torsion a–b–c–d about the b–c axis (radians).
    Dihedral(usize, usize, usize, usize),
}

/// Length of a vector `[x,y,z]` via the oracle-ordered reduction.
fn norm3(v: [f64; 3]) -> f64 {
    oracle_sum(&[v[0] * v[0], v[1] * v[1], v[2] * v[2]]).sqrt()
}

/// `a − b` componentwise.
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product via the oracle-ordered reduction.
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    oracle_sum(&[a[0] * b[0], a[1] * b[1], a[2] * b[2]])
}

/// Cross product `a × b`.
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Primitive {
    /// Evaluate the scalar internal-coordinate value at the given Cartesian
    /// geometry (`coords[i] = [x,y,z]_i` in Bohr; angles/dihedrals in radians).
    pub fn value(&self, coords: &[[f64; 3]]) -> f64 {
        match *self {
            Primitive::Distance(a, b) => norm3(sub3(coords[a], coords[b])),
            Primitive::Angle(a, b, c) => {
                let u = sub3(coords[a], coords[b]);
                let v = sub3(coords[c], coords[b]);
                let nu = norm3(u);
                let nv = norm3(v);
                // cos θ = (u·v)/(|u||v|), clamped for numerical safety.
                let cos = (dot3(u, v) / (nu * nv)).clamp(-1.0, 1.0);
                cos.acos()
            }
            Primitive::Dihedral(a, b, c, d) => {
                // Standard torsion (geomeTRIC `internal.py` Dihedral.value):
                //   b1 = r_b − r_a, b2 = r_c − r_b, b3 = r_d − r_c
                //   φ = atan2( (|b2| b1)·(b2×b3), (b1×b2)·(b2×b3) )
                let b1 = sub3(coords[b], coords[a]);
                let b2 = sub3(coords[c], coords[b]);
                let b3 = sub3(coords[d], coords[c]);
                let c12 = cross3(b1, b2);
                let c23 = cross3(b2, b3);
                let nb2 = norm3(b2);
                let y_terms = [
                    nb2 * b1[0] * c23[0],
                    nb2 * b1[1] * c23[1],
                    nb2 * b1[2] * c23[2],
                ];
                let y = oracle_sum(&y_terms);
                let x = dot3(c12, c23);
                y.atan2(x)
            }
        }
    }
}

/// Covalent radii (Bohr) for the light elements we drive in-tree (H, C, N, O).
/// Source: Cordero et al. 2008 covalent radii (Å) × ang→Bohr, the table
/// geomeTRIC uses for bonding-graph construction. Unknown Z → a generous
/// 1.5 Å fallback so heavier atoms still bond.
fn covalent_radius_bohr(z: i32) -> f64 {
    const ANG2BOHR: f64 = 1.8897261339213;
    let r_ang = match z {
        1 => 0.31,  // H
        6 => 0.76,  // C
        7 => 0.71,  // N
        8 => 0.66,  // O
        9 => 0.57,  // F
        15 => 1.07, // P
        16 => 1.05, // S
        17 => 1.02, // Cl
        _ => 1.5,
    };
    r_ang * ANG2BOHR
}

/// Build the bonding graph: `bonds[i]` is the sorted list of atoms bonded to
/// `i`. An a–b pair bonds when `|r_a − r_b| < scale·(R_a + R_b)` (geomeTRIC's
/// default `scale = 1.2`).
fn build_bonds(coords: &[[f64; 3]], charges: &[i32]) -> Vec<Vec<usize>> {
    const SCALE: f64 = 1.2;
    let natm = coords.len();
    let mut bonds = vec![Vec::new(); natm];
    for i in 0..natm {
        for j in (i + 1)..natm {
            let r = norm3(sub3(coords[i], coords[j]));
            let cut = SCALE * (covalent_radius_bohr(charges[i]) + covalent_radius_bohr(charges[j]));
            if r < cut {
                bonds[i].push(j);
                bonds[j].push(i);
            }
        }
    }
    for b in &mut bonds {
        b.sort_unstable();
    }
    bonds
}

/// Generate the redundant-internal-coordinate set from the geometry + nuclear
/// charges (geomeTRIC `internal.py` `PrimitiveInternalCoordinates` build).
///
/// Order: all bonds, then all angles, then all dihedrals — the canonical
/// geomeTRIC ordering (so the B-matrix rows are reproducible across steps).
///
/// For a molecule too small/disconnected to yield `3·natm − 6` internals
/// (e.g. a single bond), the redundant set is still well-defined; the Wilson
/// pseudo-inverse (`bmatrix.rs`) handles the rank deficiency.
pub fn generate(coords: &[[f64; 3]], charges: &[i32]) -> Vec<Primitive> {
    let natm = coords.len();
    let bonds = build_bonds(coords, charges);
    let mut prims = Vec::new();

    // Bonds (a < b to avoid duplicates).
    for (a, neighbours) in bonds.iter().enumerate() {
        for &b in neighbours {
            if a < b {
                prims.push(Primitive::Distance(a, b));
            }
        }
    }

    // Angles a–b–c: b is the apex, a and c both bonded to b, a < c.
    for (b, nb) in bonds.iter().enumerate() {
        for ia in 0..nb.len() {
            for ic in (ia + 1)..nb.len() {
                let a = nb[ia];
                let c = nb[ic];
                prims.push(Primitive::Angle(a, b, c));
            }
        }
    }

    // Dihedrals a–b–c–d about each connected b–c bond.
    for b in 0..natm {
        for &c in &bonds[b] {
            if b >= c {
                continue; // each b–c bond once
            }
            for &a in &bonds[b] {
                if a == c {
                    continue;
                }
                for &d in &bonds[c] {
                    if d == b || d == a {
                        continue;
                    }
                    prims.push(Primitive::Dihedral(a, b, c, d));
                }
            }
        }
    }

    prims
}

/// Evaluate the full internal-coordinate vector `q(x)` at a geometry.
pub fn values(prims: &[Primitive], coords: &[[f64; 3]]) -> Vec<f64> {
    prims.iter().map(|p| p.value(coords)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn distance_value_matches_hand_calc() {
        // Two atoms 2 Bohr apart on the x-axis.
        let coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let d = Primitive::Distance(0, 1).value(&coords);
        assert_relative_eq!(d, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn angle_value_matches_hand_calc() {
        // Right angle: a at +x, b at origin, c at +y → 90° = π/2.
        let coords = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let theta = Primitive::Angle(0, 1, 2).value(&coords);
        assert_relative_eq!(theta, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn dihedral_value_matches_hand_calc() {
        // Planar cis arrangement → dihedral 0; trans → ±π.
        // a=(0,1,0) b=(0,0,0) c=(1,0,0) d=(1,1,0): all in z=0 plane,
        // d on the same side as a → cis, φ ≈ 0.
        let coords = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let phi = Primitive::Dihedral(0, 1, 2, 3).value(&coords);
        assert!(phi.abs() < 1e-9, "expected cis ~0, got {phi}");
    }

    #[test]
    fn h2o_generates_two_bonds_and_one_angle() {
        // H2O near equilibrium (Bohr): O at origin, two O–H ~1.81 Bohr.
        let coords = [
            [0.0, 0.0, 0.0],    // O
            [1.43, 1.11, 0.0],  // H
            [-1.43, 1.11, 0.0], // H
        ];
        let charges = [8, 1, 1];
        let prims = generate(&coords, &charges);
        let n_bonds = prims
            .iter()
            .filter(|p| matches!(p, Primitive::Distance(..)))
            .count();
        let n_angles = prims
            .iter()
            .filter(|p| matches!(p, Primitive::Angle(..)))
            .count();
        assert_eq!(n_bonds, 2, "H2O has two O–H bonds, got {n_bonds}");
        assert_eq!(n_angles, 1, "H2O has one H–O–H angle, got {n_angles}");
    }
}
