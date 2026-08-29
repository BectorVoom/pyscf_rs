//! Tier-2 hard-coded Ewald reference values for plan 09-08 (D-PBC-19 / §9.1).
//!
//! Every literal below was generated ONCE from live PySCF 2.12.1 and committed.
//! The generating snippet is [`UPSTREAM_SNIPPET`]; running it again must
//! reproduce this file byte for byte.
//!
//! # Why the lattices are specified in BOHR
//!
//! `pyscf_core::Unit::Ang.length_in_au()` is CODATA-2014
//! (`1.8897261339213`) while upstream `pyscf/data/nist.py` is CODATA-2010
//! (`1/0.52917721092 = 1.8897261245650618`) — a **4.951e-9 relative** gap
//! recorded in plan 09-03. Ewald energies scale as `1/length`, so an
//! Angstrom-specified diamond would differ from upstream by
//! `28.77 * 4.951e-9 ~= 1.4e-7 Ha`, two orders above this plan's 1e-9 Ha gate.
//!
//! Each system is therefore pinned by its BOHR lattice and BOHR coordinates —
//! upstream's own converted values — so both sides start from bit-identical
//! geometry and the gate measures the ALGORITHM, exactly as
//! `tests/kpts_mesh.rs::diamond_bohr` already does for `make_kpts`.
//! `tests/ewald.rs` additionally sweeps the §9.2 Angstrom systems at the
//! loosened tolerance the unit gap implies, so the conversion path stays
//! covered.
//!
//! # Two charge conventions, both gated
//!
//! `cell.atom_charges()` returns the PSEUDOPOTENTIAL valence charge when
//! `pseudo=` is given, and since plan 10-01 (D-PBC-11) this port does too. The
//! [`EWALD_REFERENCES`] literals are generated from cells built WITHOUT
//! `pseudo=`, so they pin the all-electron `Z` path; [`PSEUDISED_EWALD`] holds
//! the `gth-pade` numbers for the same five systems and pins the valence path.
//! Both are asserted by `tests/ewald.rs`.

#![allow(dead_code)]

/// The exact generating snippet for every literal in this file (PySCF 2.12.1,
/// `PYTHONPATH=. .venv/bin/python`).
///
/// ```python
/// import numpy as np
/// from pyscf.pbc import gto
/// from pyscf.pbc.gto.cell import _estimate_rcut
///
/// # 1. build from Angstrom to obtain upstream's own Bohr geometry ...
/// ang = gto.Cell()
/// ang.atom = [('C', (0, 0, 0)), ('C', (3.5668 / 4,) * 3)]
/// ang.a = [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]]
/// ang.basis = 'gth-szv'; ang.unit = 'Angstrom'; ang.dimension = 3
/// ang.build()
///
/// # 2. ... then REBUILD in Bohr from those exact numbers.
/// c = gto.Cell()
/// c.a = ang.lattice_vectors().tolist()
/// c.atom = [(ang.atom_symbol(i), tuple(r)) for i, r in enumerate(ang.atom_coords())]
/// c.basis = 'gth-szv'; c.unit = 'Bohr'; c.dimension = 3
/// c.build()
///
/// eta, cut = c.get_ewald_params()
/// ke = -2 * eta**2 * np.log(c.precision / (c.atom_charges().sum() * 16 * np.pi**2))
/// print(repr(c.vol), repr(eta), repr(cut),
///       len(c.get_lattice_Ls(rcut=cut)), list(c.cutoff_to_mesh(ke)), repr(c.ewald()))
/// for s in (0.5, 0.75, 1.25, 2.0):          # the eta-invariance scan
///     cut_s = _estimate_rcut((eta * s)**2, 0, 1., c.precision)
///     print(s, repr(cut_s), repr(c.ewald(eta * s, cut_s)))
/// ```
///
/// Repeat for `si` (fcc 5.4306 A), `lif` (fcc 4.03 A, F at a0/2), `he_fcc`
/// (fcc 3.0 A) and `graphene` (hexagonal 2.46 A, 20 A vacuum, `dimension = 2`).
pub const UPSTREAM_SNIPPET: &str = "see the doc comment above";

/// One §9.2 reference system, pinned in Bohr.
pub struct EwaldReference {
    /// System name, matching `pyscf_pbc_gto::test_systems`.
    pub name: &'static str,
    /// `cell.dimension`.
    pub dimension: u8,
    /// Element symbols, in input order.
    pub symbols: &'static [&'static str],
    /// `cell.lattice_vectors()` in Bohr, one row per lattice vector.
    pub a_bohr: [[f64; 3]; 3],
    /// `cell.atom_coords()` in Bohr.
    pub coords_bohr: &'static [[f64; 3]],
    /// `cell.atom_charges()` WITHOUT a pseudopotential — the all-electron `Z`.
    pub charges: &'static [i32],
    /// `cell.vol` in Bohr^3.
    pub vol: f64,
    /// `cell.get_ewald_params()[0]`.
    pub ew_eta: f64,
    /// `cell.get_ewald_params()[1]`.
    pub ew_cut: f64,
    /// `len(cell.get_lattice_Ls(rcut=ew_cut))`.
    pub n_ls: usize,
    /// `cell.cutoff_to_mesh(-2*eta^2*log(precision/(sum(q)*16*pi^2)))` — the
    /// mesh `ewald()` builds its G-space sum on.
    pub mesh: [usize; 3],
    /// `cell.ewald()` in Hartree. `None` where this port defers the branch —
    /// graphene is `dimension = 2`, whose truncated-Coulomb G-space sum is
    /// PBC-MASTER-PLAN plan 12-08 (D-PBC-20). The upstream value is recorded in
    /// the comment beside it so plan 12-08 has its target.
    pub ewald: Option<f64>,
}

/// See [`EwaldReference`] and [`UPSTREAM_SNIPPET`].
pub const EWALD_REFERENCES: [EwaldReference; 5] = [
    EwaldReference {
        name: "diamond",
        dimension: 3,
        symbols: &["C", "C"],
        a_bohr: [
            [0.0, 3.3701375705493315, 3.3701375705493315],
            [3.3701375705493315, 0.0, 3.3701375705493315],
            [3.3701375705493315, 3.3701375705493315, 0.0],
        ],
        coords_bohr: &[
            [0.0, 0.0, 0.0],
            [1.6850687852746657, 1.6850687852746657, 1.6850687852746657],
        ],
        charges: &[6, 6],
        vol: 76.55488063251218,
        ew_eta: 0.4852935502366724,
        ew_cut: 14.69856051295752,
        n_ls: 321,
        mesh: [9, 9, 9],
        ewald: Some(-28.771040577654524),
    },
    EwaldReference {
        name: "si",
        dimension: 3,
        symbols: &["Si", "Si"],
        a_bohr: [
            [0.0, 5.131173346031512, 5.131173346031512],
            [5.131173346031512, 0.0, 5.131173346031512],
            [5.131173346031512, 5.131173346031512, 0.0],
        ],
        coords_bohr: &[
            [0.0, 0.0, 0.0],
            [2.565586673015756, 2.565586673015756, 2.565586673015756],
        ],
        charges: &[14, 14],
        vol: 270.1967093603764,
        ew_eta: 0.39329641773158136,
        ew_cut: 18.204925606997,
        n_ls: 177,
        mesh: [11, 11, 11],
        ewald: Some(-102.88216217333321),
    },
    EwaldReference {
        name: "lif",
        dimension: 3,
        symbols: &["Li", "F"],
        a_bohr: [
            [0.0, 3.8077981409986, 3.8077981409986],
            [3.8077981409986, 0.0, 3.8077981409986],
            [3.8077981409986, 3.8077981409986, 0.0],
        ],
        coords_bohr: &[
            [0.0, 0.0, 0.0],
            [3.8077981409986, 3.8077981409986, 3.8077981409986],
        ],
        charges: &[3, 9],
        vol: 110.42101837541341,
        ew_eta: 0.4565531833791103,
        ew_cut: 15.640918598989309,
        n_ls: 429,
        mesh: [9, 9, 9],
        ewald: Some(-30.95510482656236),
    },
    EwaldReference {
        name: "he_fcc",
        dimension: 3,
        symbols: &["He"],
        a_bohr: [
            [0.0, 2.8345891868475928, 2.8345891868475928],
            [2.8345891868475928, 0.0, 2.8345891868475928],
            [2.8345891868475928, 2.8345891868475928, 0.0],
        ],
        coords_bohr: &[[0.0, 0.0, 0.0]],
        charges: &[2],
        vol: 45.551257834162435,
        ew_eta: 0.5291554470071487,
        ew_cut: 13.459295088515482,
        n_ls: 225,
        mesh: [9, 9, 9],
        ewald: Some(-1.6174696832216189),
    },
    EwaldReference {
        name: "graphene",
        dimension: 2,
        symbols: &["C", "C"],
        a_bohr: [
            [4.648726266430052, 0.0, 0.0],
            [-2.324363133215026, 4.025915041968412, 0.0],
            [0.0, 0.0, 37.79452249130124],
        ],
        coords_bohr: &[[0.0, 0.0, 0.0], [0.0, 2.6839433613122745, 0.0]],
        charges: &[6, 6],
        vol: 707.3387370358154,
        ew_eta: 0.26966050248398293,
        ew_cut: 18.89726124565062,
        n_ls: 85,
        mesh: [7, 7, 35],
        // Upstream `c.ewald()` via the dimension == 2 truncated-Coulomb branch
        // (cell.py:773-800). RECORDED IN PHASE 9 as plan 12-08's target, before
        // the branch existed — so it is a pre-committed reference, not a number
        // fitted after the fact.
        ewald: Some(-44.57202102404764),
    },
];

/// One point of the `ew_eta` invariance scan.
pub struct EtaScanPoint {
    /// Multiplier applied to `ew_eta` from `get_ewald_params`.
    pub scale: f64,
    /// `_estimate_rcut((eta*scale)^2, 0, 1., precision)` — the matching
    /// real-space cutoff. It MUST move with `eta`: holding `ew_cut` fixed makes
    /// upstream itself drift by 8.1e-7 Ha at `0.5*eta0`, because the real-space
    /// sum converges more slowly as the screening weakens.
    pub ew_cut: f64,
    /// `cell.ewald(eta*scale, ew_cut)`.
    pub ewald: f64,
}

/// Diamond (Bohr-specified, no pseudopotential), `eta0 = 0.4852935502366724`.
/// Upstream's own spread over this window is 3.4e-13 Ha.
pub const DIAMOND_ETA_SCAN: [EtaScanPoint; 4] = [
    EtaScanPoint {
        scale: 0.5,
        ew_cut: 29.760860832192574,
        ewald: -28.771040577654862,
    },
    EtaScanPoint {
        scale: 0.75,
        ew_cut: 19.69887957045394,
        ewald: -28.77104057765446,
    },
    EtaScanPoint {
        scale: 1.25,
        ew_cut: 11.711800495983008,
        ewald: -28.77104057765455,
    },
    EtaScanPoint {
        scale: 2.0,
        ew_cut: 7.257622705598762,
        ewald: -28.771040577654702,
    },
];

/// `cell.ewald()` for the same five systems built WITH `pseudo='gth-pade'`,
/// i.e. with upstream's valence charges. ASSERTED since plan 10-01 landed the
/// GTH parser (D-PBC-11) — see
/// `tests/ewald.rs::pseudised_ewald_matches_the_recorded_upstream_targets`.
/// In the order of [`EWALD_REFERENCES`]. `he_fcc` is unchanged because
/// `gth-pade` leaves He's two electrons in the valence.
pub const PSEUDISED_EWALD: [(&str, &[i32], f64); 5] = [
    ("diamond", &[4, 4], -12.78712914562424),
    ("si", &[4, 4], -8.398543850884348),
    ("lif", &[3, 7], -20.463977469434052),
    ("he_fcc", &[2], -1.6174696832216189),
    // graphene also shifts ew_eta (the dimension == 2 branch reads sum(q)):
    // 0.2675469466398444 instead of 0.26966050248398293.
    ("graphene", &[4, 4], -19.80978712179894),
];
