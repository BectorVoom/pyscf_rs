//! Band-structure k-paths — high-symmetry points and the segments between
//! them, ready to hand to `get_bands`.
//!
//! # There is no upstream to port
//!
//! PySCF does not implement this. `pyscf/pbc/tools/pyscf_ase.py:87-91`
//! forwards to `ase.cell.Cell.bandpath`, and `pyscf/pbc/tools/lattice.py:68-81`
//! hard-codes a single FCC path around `ase.dft.kpoints.get_bandpath`. So this
//! module follows ASE's conventions rather than porting PySCF line by line, and
//! its return shape mirrors `get_bandpath_fcc`'s `(kpts_reduced, kpts_cartes,
//! kpath, sp_points)`.
//!
//! # Special-point coordinates
//!
//! The tables below are the Setyawan-Curtarolo standard paths — W. Setyawan and
//! S. Curtarolo, *Comput. Mater. Sci.* **49**, 299 (2010) — which is also what
//! ASE implements. Coordinates are FRACTIONAL, in units of the primitive
//! reciprocal lattice vectors. They are transcribed from that convention, not
//! machine-checked against ASE (which is not a dependency here); what IS
//! machine-checked is that every non-Gamma point lies on a Bragg plane of the
//! cell's own reciprocal lattice, which is the defining property of a
//! Brillouin-zone-boundary point and catches a mistyped coordinate. See
//! `tests/kpath.rs`.
//!
//! # Lattice recognition declines rather than guesses
//!
//! [`detect_lattice`] returns `None` unless the cell's primitive vectors match
//! a standard form outright. Misclassifying a lattice yields a band path that
//! is wrong in a way no later stage can detect, so there is no "closest match"
//! fallback: name the lattice explicitly, or supply your own points.

use crate::cell::Cell;
use pyscf_core::{CoreError, PyscfRsError};

/// Tolerance for recognising a lattice from its primitive vectors, relative to
/// the mean vector length.
const LATTICE_TOL: f64 = 1e-6;

/// The Bravais lattices this module carries special-point tables for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BravaisLattice {
    /// Simple cubic.
    Cubic,
    /// Face-centred cubic.
    Fcc,
    /// Body-centred cubic — the lattice of europium, among others.
    Bcc,
    /// Hexagonal.
    Hexagonal,
    /// Simple tetragonal.
    Tetragonal,
}

impl BravaisLattice {
    /// The labelled high-symmetry points, in fractional reciprocal
    /// coordinates.
    #[must_use]
    pub fn special_points(self) -> &'static [(&'static str, [f64; 3])] {
        const THIRD: f64 = 1.0 / 3.0;
        match self {
            Self::Cubic => &[
                ("G", [0.0, 0.0, 0.0]),
                ("M", [0.5, 0.5, 0.0]),
                ("R", [0.5, 0.5, 0.5]),
                ("X", [0.0, 0.5, 0.0]),
            ],
            Self::Fcc => &[
                ("G", [0.0, 0.0, 0.0]),
                ("K", [0.375, 0.375, 0.75]),
                ("L", [0.5, 0.5, 0.5]),
                ("U", [0.625, 0.25, 0.625]),
                ("W", [0.5, 0.25, 0.75]),
                ("X", [0.5, 0.0, 0.5]),
            ],
            Self::Bcc => &[
                ("G", [0.0, 0.0, 0.0]),
                ("H", [0.5, -0.5, 0.5]),
                ("P", [0.25, 0.25, 0.25]),
                ("N", [0.0, 0.0, 0.5]),
            ],
            Self::Hexagonal => &[
                ("G", [0.0, 0.0, 0.0]),
                ("A", [0.0, 0.0, 0.5]),
                ("H", [THIRD, THIRD, 0.5]),
                ("K", [THIRD, THIRD, 0.0]),
                ("L", [0.5, 0.0, 0.5]),
                ("M", [0.5, 0.0, 0.0]),
            ],
            Self::Tetragonal => &[
                ("G", [0.0, 0.0, 0.0]),
                ("A", [0.5, 0.5, 0.5]),
                ("M", [0.5, 0.5, 0.0]),
                ("R", [0.0, 0.5, 0.5]),
                ("X", [0.0, 0.5, 0.0]),
                ("Z", [0.0, 0.0, 0.5]),
            ],
        }
    }

    /// The conventional path, as connected segments. A new inner slice starts a
    /// DISCONTINUOUS jump — ASE writes those with a comma, as in the BCC
    /// `GHNGPH,PN`.
    #[must_use]
    pub fn default_path(self) -> &'static [&'static [&'static str]] {
        match self {
            Self::Cubic => &[&["G", "X", "M", "G", "R", "X"], &["M", "R"]],
            Self::Fcc => &[
                &["G", "X", "W", "K", "G", "L", "U", "W", "L", "K"],
                &["U", "X"],
            ],
            Self::Bcc => &[&["G", "H", "N", "G", "P", "H"], &["P", "N"]],
            Self::Hexagonal => &[
                &["G", "M", "K", "G", "A", "L", "H", "A"],
                &["L", "M"],
                &["K", "H"],
            ],
            Self::Tetragonal => &[
                &["G", "X", "M", "G", "Z", "R", "A", "Z"],
                &["X", "R"],
                &["M", "A"],
            ],
        }
    }

    /// Look one labelled point up.
    #[must_use]
    pub fn point(self, label: &str) -> Option<[f64; 3]> {
        self.special_points()
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, k)| *k)
    }
}

/// A sampled band path.
///
/// Mirrors the tuple `pyscf/pbc/tools/lattice.py:68-81` returns, with the
/// plotting labels added.
#[derive(Debug, Clone)]
pub struct KPath {
    /// Fractional coordinates, in units of the reciprocal lattice.
    pub scaled: Vec<[f64; 3]>,
    /// Absolute k-points in 1/Bohr — what `get_bands` takes.
    pub abs: Vec<[f64; 3]>,
    /// Cumulative distance along the path, one per k-point. This is the
    /// band-plot x-axis, and it is measured in ABSOLUTE reciprocal space, so
    /// segment lengths stay physical for a non-cubic cell.
    pub x: Vec<f64>,
    /// x-coordinate of each labelled point, for axis ticks.
    pub tick_x: Vec<f64>,
    /// The label at each tick. A jump between two disconnected segments is
    /// rendered as a single `"A|B"` tick, ASE's convention.
    pub tick_labels: Vec<String>,
}

impl KPath {
    /// Number of k-points on the path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.abs.len()
    }

    /// Whether the path carries no k-points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.abs.is_empty()
    }
}

/// Sample the conventional path for `lattice` with roughly `npoints` k-points.
///
/// # Errors
/// Propagates the reciprocal-lattice construction, and rejects a label absent
/// from the lattice's table.
pub fn band_path(
    cell: &Cell,
    lattice: BravaisLattice,
    npoints: usize,
) -> Result<KPath, PyscfRsError> {
    let segments: Vec<Vec<(String, [f64; 3])>> = lattice
        .default_path()
        .iter()
        .map(|seg| {
            seg.iter()
                .map(|label| {
                    lattice
                        .point(label)
                        .map(|k| ((*label).to_string(), k))
                        .ok_or_else(|| {
                            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                                "band_path: {lattice:?} has no special point '{label}'"
                            )))
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    band_path_from_segments(cell, &segments, npoints)
}

/// Sample a caller-supplied path.
///
/// `segments` is a list of connected runs of `(label, fractional k)`; a new run
/// is a discontinuous jump. `npoints` is the TOTAL budget, distributed over the
/// segments in proportion to their length in absolute reciprocal space, so the
/// sampling density is uniform along the path rather than per segment.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when a run has fewer than two points, and
/// whatever the reciprocal-lattice construction raises.
pub fn band_path_from_segments(
    cell: &Cell,
    segments: &[Vec<(String, [f64; 3])>],
    npoints: usize,
) -> Result<KPath, PyscfRsError> {
    if segments.is_empty() {
        return Err(invalid("band_path: no segments given"));
    }
    for seg in segments {
        if seg.len() < 2 {
            return Err(invalid(
                "band_path: every segment needs at least two special points",
            ));
        }
    }

    // Leg lengths in ABSOLUTE reciprocal space. Doing this in fractional
    // coordinates would stretch the plot axis by the reciprocal metric.
    let mut leg_len: Vec<Vec<f64>> = Vec::with_capacity(segments.len());
    let mut total = 0.0;
    for seg in segments {
        let abs = cell.get_abs_kpts(&seg.iter().map(|(_, k)| *k).collect::<Vec<_>>())?;
        let lens: Vec<f64> = abs.windows(2).map(|w| dist(&w[0], &w[1])).collect();
        total += lens.iter().sum::<f64>();
        leg_len.push(lens);
    }
    if total <= 0.0 {
        return Err(invalid(
            "band_path: the path has zero total length; the special points coincide",
        ));
    }

    let n_legs: usize = leg_len.iter().map(Vec::len).sum();
    // Every leg gets at least one interval, so a short leg cannot vanish and
    // drop its endpoint label off the plot.
    let budget = npoints.max(n_legs + 1);

    let mut scaled: Vec<[f64; 3]> = Vec::with_capacity(budget);
    let mut x: Vec<f64> = Vec::with_capacity(budget);
    let mut tick_x: Vec<f64> = Vec::new();
    let mut tick_labels: Vec<String> = Vec::new();
    let mut travelled = 0.0;

    for (s, seg) in segments.iter().enumerate() {
        for (leg, pair) in seg.windows(2).enumerate() {
            let (label_a, ka) = &pair[0];
            let len = leg_len[s][leg];
            // Proportional share of the budget, at least one interval.
            let n_int = (((len / total) * budget as f64).round() as usize).max(1);

            // The starting point of the leg. It is emitted once — as the start
            // of the first leg, or as the shared endpoint of the previous leg,
            // which the previous iteration deliberately did not emit.
            if leg == 0 {
                let label = if s == 0 {
                    label_a.clone()
                } else {
                    // A jump: the previous segment ended at some point and this
                    // one starts elsewhere. ASE renders that as one "A|B" tick.
                    let prev = tick_labels.pop().unwrap_or_default();
                    format!("{prev}|{label_a}")
                };
                if s > 0 {
                    tick_x.pop();
                }
                tick_x.push(travelled);
                tick_labels.push(label);
                scaled.push(*ka);
                x.push(travelled);
            }

            let (_, kb) = &pair[1];
            for i in 1..=n_int {
                let t = i as f64 / n_int as f64;
                scaled.push([
                    ka[0] + t * (kb[0] - ka[0]),
                    ka[1] + t * (kb[1] - ka[1]),
                    ka[2] + t * (kb[2] - ka[2]),
                ]);
                x.push(travelled + t * len);
            }
            travelled += len;
            tick_x.push(travelled);
            tick_labels.push(pair[1].0.clone());
        }
    }

    let abs = cell.get_abs_kpts(&scaled)?;
    Ok(KPath {
        scaled,
        abs,
        x,
        tick_x,
        tick_labels,
    })
}

/// Recognise the Bravais lattice from the cell's primitive vectors, or return
/// `None`.
///
/// Recognition is deliberately strict: the vectors must match a standard
/// primitive form to within [`LATTICE_TOL`] relative to their mean length. A
/// near-miss returns `None` rather than the closest candidate, because a band
/// path drawn for the wrong lattice is silently, not visibly, wrong.
#[must_use]
pub fn detect_lattice(cell: &Cell) -> Option<BravaisLattice> {
    let a = cell.lattice_vectors();
    let norms = [norm(&a[0]), norm(&a[1]), norm(&a[2])];
    let scale = (norms[0] + norms[1] + norms[2]) / 3.0;
    if scale <= 0.0 {
        return None;
    }
    let tol = LATTICE_TOL * scale;

    // Work with the vectors scaled to unit mean length, so the reference forms
    // below are pure shape.
    let same = |x: f64, y: f64| (x - y).abs() < tol;
    let equal_lengths = same(norms[0], norms[1]) && same(norms[1], norms[2]);

    let cosines = [
        cos_angle(&a[1], &a[2]),
        cos_angle(&a[0], &a[2]),
        cos_angle(&a[0], &a[1]),
    ];
    let ctol = LATTICE_TOL;
    let all_cos = |v: f64| cosines.iter().all(|c| (c - v).abs() < ctol);

    if equal_lengths {
        // Simple cubic: three orthogonal equal vectors.
        if all_cos(0.0) {
            return Some(BravaisLattice::Cubic);
        }
        // FCC primitive: pairwise 60 degrees.
        if all_cos(0.5) {
            return Some(BravaisLattice::Fcc);
        }
        // BCC primitive: pairwise arccos(-1/3).
        if all_cos(-1.0 / 3.0) {
            return Some(BravaisLattice::Bcc);
        }
    }

    // Hexagonal: a == b perpendicular to c, 120 degrees between a and b.
    if same(norms[0], norms[1])
        && cosines[0].abs() < ctol
        && cosines[1].abs() < ctol
        && (cosines[2] + 0.5).abs() < ctol
    {
        return Some(BravaisLattice::Hexagonal);
    }

    // Simple tetragonal: three orthogonal vectors, exactly two equal.
    if all_cos(0.0) && same(norms[0], norms[1]) && !same(norms[1], norms[2]) {
        return Some(BravaisLattice::Tetragonal);
    }

    None
}

fn invalid(msg: &str) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(msg.to_string()))
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    norm(&d)
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cos_angle(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    let d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let n = norm(u) * norm(v);
    if n <= 0.0 { 0.0 } else { d / n }
}
