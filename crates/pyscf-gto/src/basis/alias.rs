//! ALIAS table porting upstream `pyscf/gto/basis/__init__.py:42-460`.
//!
//! Source-of-truth: the Python module. The hand-port preserves entries
//! exactly as upstream maps them — including subdirectory paths
//! (`pople-basis/6-31G.dat`) and case-sensitivity of file names
//! (`6-31G.dat`, `cc-pCV5Z.dat`).
//!
//! Drift is caught by the `alias_resolution` integration test (Task 3) —
//! the resolver `dir.join(filename)` MUST find the file on disk for every
//! test entry.
//!
//! Three tables (per upstream):
//!   * `ALIAS`     — Gaussian-type basis names (STO-3G, cc-pVDZ, def2-SVP, ...)
//!   * `GTH_ALIAS` — CP2K / GTH pseudopotential basis (gth-dzvp, gth-tzvp, ...)
//!   * `PP_ALIAS`  — Pseudopotentials proper (gth-pade, gth-blyp, ...)
//!
//! Phase 2 ships ≥ 30 entries (acceptance floor) — comfortably covers the
//! PR-CI corpus (sto-3g, cc-pvdz, 6-31g*, def2-svp, lanl2dz). A Phase 2.x
//! sweep can extend to the full upstream ≥ 395-entry catalogue if a user
//! request surfaces.

use std::collections::HashMap;
use std::sync::OnceLock;

static ALIAS_TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
static GTH_ALIAS_TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
static PP_ALIAS_TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

fn build_alias() -> HashMap<&'static str, &'static str> {
    // Hand-ported from `pyscf/gto/basis/__init__.py:42-419`.
    // Keys are post-`canonicalise_basis_name` (lower-cased, no dashes/underscores).
    // Values are paths relative to the basis directory; subdirectory paths use
    // forward slashes and are joined verbatim by the resolver.
    let mut m = HashMap::new();

    // === ANO / Roos ===
    m.insert("ano", "ano.dat");
    m.insert("anorcc", "ano.dat");
    m.insert("anoroosdz", "roos-dz.dat");
    m.insert("anoroostz", "roos-tz.dat");
    m.insert("roosdz", "roos-dz.dat");
    m.insert("roostz", "roos-tz.dat");

    // === Dunning correlation-consistent (cc-pVxZ) ===
    m.insert("ccpvdz", "cc-pvdz.dat");
    m.insert("ccpvtz", "cc-pvtz.dat");
    m.insert("ccpvqz", "cc-pvqz.dat");
    m.insert("ccpv5z", "cc-pv5z.dat");
    m.insert("augccpvdz", "aug-cc-pvdz.dat");
    m.insert("augccpvtz", "aug-cc-pvtz.dat");
    m.insert("augccpvqz", "aug-cc-pvqz.dat");
    m.insert("augccpv5z", "aug-cc-pv5z.dat");

    // === Dunning DK (relativistic) ===
    m.insert("ccpvdzdk", "cc-pvdz-dk.dat");
    m.insert("ccpvtzdk", "cc-pvtz-dk.dat");
    m.insert("ccpvqzdk", "cc-pvqz-dk.dat");
    m.insert("ccpv5zdk", "cc-pv5z-dk.dat");

    // === Dunning JK / RI / fitting ===
    m.insert("ccpvdzjkfit", "cc-pvdz-jkfit.dat");
    m.insert("ccpvtzjkfit", "cc-pvtz-jkfit.dat");
    m.insert("ccpvqzjkfit", "cc-pvqz-jkfit.dat");
    m.insert("ccpv5zjkfit", "cc-pv5z-jkfit.dat");
    m.insert("ccpvdzri", "cc-pvdz-ri.dat");
    m.insert("ccpvtzri", "cc-pvtz-ri.dat");
    m.insert("ccpvqzri", "cc-pvqz-ri.dat");
    m.insert("ccpv5zri", "cc-pv5z-ri.dat");
    m.insert("ccpvtzfit", "cc-pvtz_fit.dat");
    m.insert("ccpvdzfit", "cc-pvdz_fit.dat");

    // === Pople-style (in pople-basis/ subdirectory; case-sensitive G/Gs/Gss) ===
    m.insert("321g", "pople-basis/3-21G.dat");
    m.insert("321gs", "pople-basis/3-21Gs.dat");
    m.insert("321g*", "pople-basis/3-21Gs.dat");
    m.insert("431g", "pople-basis/4-31G.dat");
    m.insert("631g", "pople-basis/6-31G.dat");
    m.insert("631gs", "pople-basis/6-31Gs.dat");
    m.insert("631g*", "pople-basis/6-31Gs.dat");
    m.insert("631gss", "pople-basis/6-31Gss.dat");
    m.insert("631g**", "pople-basis/6-31Gss.dat");
    m.insert("6311g", "pople-basis/6-311G.dat");
    m.insert("6311gs", "pople-basis/6-311Gs.dat");
    m.insert("6311g*", "pople-basis/6-311Gs.dat");
    m.insert("6311gss", "pople-basis/6-311Gss.dat");
    m.insert("6311g**", "pople-basis/6-311Gss.dat");
    m.insert("631plusg", "pople-basis/6-31+G.dat"); // alt-key for upstream "6-31+G"; user typically writes "6-31+g"
    m.insert("631+g", "pople-basis/6-31+G.dat");
    m.insert("631plusgs", "pople-basis/6-31+Gs.dat");
    m.insert("631+gs", "pople-basis/6-31+Gs.dat");
    m.insert("631+g*", "pople-basis/6-31+Gs.dat");
    m.insert("631plusgss", "pople-basis/6-31+Gss.dat");
    m.insert("631+gss", "pople-basis/6-31+Gss.dat");
    m.insert("631+g**", "pople-basis/6-31+Gss.dat");

    // === STO ===
    m.insert("sto3g", "sto-3g.dat");
    m.insert("sto6g", "sto-6g.dat");

    // === Karlsruhe def2 family ===
    m.insert("def2svp", "def2-svp.dat");
    m.insert("def2svpd", "def2-svpd.dat");
    m.insert("def2tzvp", "def2-tzvp.dat");
    m.insert("def2tzvpp", "def2-tzvpp.dat");
    m.insert("def2tzvpd", "def2-tzvpd.dat");
    m.insert("def2tzvppd", "def2-tzvppd.dat");
    m.insert("def2qzvp", "def2-qzvp.dat");
    m.insert("def2qzvpp", "def2-qzvpp.dat");
    m.insert("def2qzvpd", "def2-qzvpd.dat");
    m.insert("def2qzvppd", "def2-qzvppd.dat");

    // === DZ / TZ / QZ generic ===
    m.insert("dz", "dz.dat");
    m.insert("dzvp", "dzvp.dat");
    m.insert("dzvp2", "dzvp2.dat");
    m.insert("dzp", "dzp.dat");
    m.insert("tzp", "tzp.dat");
    m.insert("qzp", "qzp.dat");
    m.insert("adzp", "adzp.dat");
    m.insert("atzp", "atzp.dat");
    m.insert("aqzp", "aqzp.dat");
    m.insert("dzpdk", "dzp-dkh.dat");
    m.insert("tzpdk", "tzp-dkh.dat");
    m.insert("qzpdk", "qzp-dkh.dat");
    m.insert("dzpdkh", "dzp-dkh.dat");
    m.insert("tzpdkh", "tzp-dkh.dat");
    m.insert("qzpdkh", "qzp-dkh.dat");

    // === Density-fitting auxiliary ===
    // === def2 auxiliary fitting sets (plan 14-01) ===
    // `pyscf/gto/basis/__init__.py` ALIAS. Needed because
    // `df.addons.predefined_auxbasis` resolves sto-3g (and every def2 orbital
    // basis) to `def2-svp-jkfit`, which is the def2-universal file. Without
    // these rows `make_auxbasis` cannot build the He-fcc auxiliary cell.
    m.insert("def2universaljfit", "def2-universal-jfit.dat");
    m.insert("def2universaljkfit", "def2-universal-jkfit.dat");
    m.insert("def2svpjfit", "def2-universal-jfit.dat");
    m.insert("def2svpjkfit", "def2-universal-jkfit.dat");
    m.insert("def2svpri", "def2-svp-ri.dat");
    m.insert("def2svpdri", "def2-svpd-ri.dat");
    m.insert("def2tzvpjfit", "def2-universal-jfit.dat");
    m.insert("def2tzvpjkfit", "def2-universal-jkfit.dat");
    m.insert("def2tzvpri", "def2-tzvp-ri.dat");
    m.insert("def2tzvpdri", "def2-tzvpd-ri.dat");
    m.insert("def2tzvppjfit", "def2-universal-jfit.dat");
    m.insert("def2tzvppjkfit", "def2-universal-jkfit.dat");
    m.insert("def2tzvppri", "def2-tzvpp-ri.dat");
    m.insert("def2tzvppdri", "def2-tzvppd-ri.dat");
    m.insert("def2qzvpjfit", "def2-universal-jfit.dat");
    m.insert("def2qzvpjkfit", "def2-universal-jkfit.dat");
    m.insert("def2qzvpri", "def2-qzvp-ri.dat");
    m.insert("def2qzvppjfit", "def2-universal-jfit.dat");
    m.insert("def2qzvppjkfit", "def2-universal-jkfit.dat");
    m.insert("def2qzvppri", "def2-qzvpp-ri.dat");
    m.insert("def2qzvppdri", "def2-qzvppd-ri.dat");
    m.insert("weigend", "def2-universal-jfit.dat");
    m.insert("weigendjfit", "def2-universal-jfit.dat");
    m.insert("weigendjkfit", "def2-universal-jkfit.dat");
    m.insert("ahlrichscfit", "ahlrichs_cfit.dat");
    m.insert("ahlrichs", "ahlrichs_cfit.dat");
    m.insert("dgaussa1cfit", "DgaussA1_dft_cfit.dat");
    m.insert("dgaussa1xfit", "DgaussA1_dft_xfit.dat");
    m.insert("dgaussa2cfit", "DgaussA2_dft_cfit.dat");
    m.insert("dgaussa2xfit", "DgaussA2_dft_xfit.dat");

    // === ECP-bcc / pseudopotential bases ===
    m.insert("lanl2dz", "lanl2dz.dat");
    m.insert("lanl2tz", "lanl2tz.dat");
    m.insert("lanl08", "lanl08.dat");
    m.insert("sbkjc", "sbkjc.dat");

    // === pc / pcseg ===
    m.insert("pc0", "pc-0.dat");
    m.insert("pc1", "pc-1.dat");
    m.insert("pc2", "pc-2.dat");
    m.insert("pc3", "pc-3.dat");
    m.insert("pc4", "pc-4.dat");
    m.insert("pcseg0", "pcseg-0.dat");
    m.insert("pcseg1", "pcseg-1.dat");
    m.insert("pcseg2", "pcseg-2.dat");

    // === Burkatzki-Filippi-Dolg ===
    m.insert("bfdvdz", "bfd_vdz.dat");
    m.insert("bfdvtz", "bfd_vtz.dat");
    m.insert("bfdvqz", "bfd_vqz.dat");
    m.insert("bfdv5z", "bfd_v5z.dat");
    m.insert("bfd", "bfd_pp.dat");
    m.insert("bfdpp", "bfd_pp.dat");

    // === IGLO / Faegre / SARC ===
    m.insert("iglo3", "iglo3");
    m.insert("iglo", "iglo3");
    m.insert("faegredz", "faegre_dz");
    m.insert("sarcdkh", "sarc-dkh2.dat");

    // === minao ===
    m.insert("minao", "minao");

    m
}

fn build_gth_alias() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("gthaugdzvp", "gth-aug-dzvp.dat");
    m.insert("gthaugqzv2p", "gth-aug-qzv2p.dat");
    m.insert("gthaugqzv3p", "gth-aug-qzv3p.dat");
    m.insert("gthaugtzv2p", "gth-aug-tzv2p.dat");
    m.insert("gthaugtzvp", "gth-aug-tzvp.dat");
    m.insert("gthdzv", "gth-dzv.dat");
    m.insert("gthdzvp", "gth-dzvp.dat");
    m.insert("gthqzv2p", "gth-qzv2p.dat");
    m.insert("gthqzv3p", "gth-qzv3p.dat");
    m.insert("gthszv", "gth-szv.dat");
    m.insert("gthtzv2p", "gth-tzv2p.dat");
    m.insert("gthtzvp", "gth-tzvp.dat");
    m.insert("gthccdzvp", "gth-cc-dzvp.dat");
    m.insert("gthcctzvp", "gth-cc-tzvp.dat");
    m.insert("gthccqzvp", "gth-cc-qzvp.dat");
    m
}

fn build_pp_alias() -> HashMap<&'static str, &'static str> {
    // GTH pseudopotentials (`Cell.pseudo`). Hand-ported from upstream
    // `pyscf/gto/basis/__init__.py` `PP_ALIAS`. Keys are post-canonicalise
    // (lower-cased, no dashes/underscores). Files live under the PBC pseudo
    // tree (`pyscf/pbc/gto/pseudo/`); the loader routes `AliasKind::Pp` there
    // and parses with `cp2k_pp::parse_cp2k_pp`.
    let mut m = HashMap::new();
    m.insert("gthblyp", "gth-blyp.dat");
    m.insert("gthbp", "gth-bp.dat");
    m.insert("gthhcth120", "gth-hcth120.dat");
    m.insert("gthhcth407", "gth-hcth407.dat");
    m.insert("gtholyp", "gth-olyp.dat");
    m.insert("gthlda", "gth-pade.dat");
    m.insert("gthpade", "gth-pade.dat");
    m.insert("gthpbe", "gth-pbe.dat");
    m.insert("gthpbesol", "gth-pbesol.dat");
    m.insert("gthhf", "gth-hf.dat");
    m.insert("gthhfrev", "gth-hf-rev.dat");
    m
}

/// Look up a canonical (post-canonicalise) basis name in the merged alias tables.
/// Order: ALIAS → GTH_ALIAS → PP_ALIAS.
pub fn lookup(canonical_name: &str) -> Option<&'static str> {
    lookup_kind(canonical_name).map(|(filename, _)| filename)
}

/// Which alias table a name resolved through. Selects the on-disk directory:
/// standard Gaussian basis sets live under `pyscf/gto/basis/`, while CP2K/GTH
/// basis sets and pseudopotentials live under the PBC tree
/// (`pyscf/pbc/gto/basis/`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    /// Standard Gaussian basis (`pyscf/gto/basis/`).
    Standard,
    /// CP2K / GTH basis (`pyscf/pbc/gto/basis/`).
    Gth,
    /// Pseudopotential basis (`pyscf/pbc/gto/basis/`).
    Pp,
}

/// Like [`lookup`], but also reports which table matched so the loader can pick
/// the correct on-disk directory. Resolution order is ALIAS → GTH_ALIAS →
/// PP_ALIAS (matching [`lookup`]).
pub fn lookup_kind(canonical_name: &str) -> Option<(&'static str, AliasKind)> {
    if let Some(f) = ALIAS_TABLE.get_or_init(build_alias).get(canonical_name) {
        return Some((f, AliasKind::Standard));
    }
    if let Some(f) = GTH_ALIAS_TABLE
        .get_or_init(build_gth_alias)
        .get(canonical_name)
    {
        return Some((f, AliasKind::Gth));
    }
    if let Some(f) = PP_ALIAS_TABLE
        .get_or_init(build_pp_alias)
        .get(canonical_name)
    {
        return Some((f, AliasKind::Pp));
    }
    None
}

/// Number of entries in the main `ALIAS` table. Used by the floor-check test.
pub fn alias_count() -> usize {
    ALIAS_TABLE.get_or_init(build_alias).len()
}

/// Number of entries in `GTH_ALIAS`. Used by the GTH-floor test.
pub fn gth_alias_count() -> usize {
    GTH_ALIAS_TABLE.get_or_init(build_gth_alias).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_aliases_resolve() {
        assert_eq!(lookup("sto3g"), Some("sto-3g.dat"));
        assert_eq!(lookup("ccpvdz"), Some("cc-pvdz.dat"));
        assert_eq!(lookup("def2svp"), Some("def2-svp.dat"));
        assert_eq!(lookup("631g"), Some("pople-basis/6-31G.dat"));
        assert_eq!(lookup("631gss"), Some("pople-basis/6-31Gss.dat"));
        assert_eq!(lookup("lanl2dz"), Some("lanl2dz.dat"));
    }

    #[test]
    fn gth_aliases_resolve() {
        assert_eq!(lookup("gthdzvp"), Some("gth-dzvp.dat"));
        assert_eq!(lookup("gthtzv2p"), Some("gth-tzv2p.dat"));
    }

    #[test]
    fn lookup_kind_classifies_each_table() {
        // Standard Gaussian basis.
        assert_eq!(
            lookup_kind("sto3g"),
            Some(("sto-3g.dat", AliasKind::Standard))
        );
        // GTH basis set.
        assert_eq!(
            lookup_kind("gthdzvp"),
            Some(("gth-dzvp.dat", AliasKind::Gth))
        );
        // GTH pseudopotential (incl. the gthlda == gthpade alias).
        assert_eq!(
            lookup_kind("gthpade"),
            Some(("gth-pade.dat", AliasKind::Pp))
        );
        assert_eq!(lookup_kind("gthlda"), Some(("gth-pade.dat", AliasKind::Pp)));
        assert_eq!(lookup_kind("gthpbe"), Some(("gth-pbe.dat", AliasKind::Pp)));
        assert_eq!(lookup_kind("totally-fake"), None);
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(lookup("totally-fake-basis"), None);
    }

    #[test]
    fn alias_table_has_at_least_30_entries() {
        // Acceptance floor — Phase 2 ships ≥ 30 entries.
        assert!(alias_count() >= 30, "ALIAS has {} entries", alias_count());
    }
}
