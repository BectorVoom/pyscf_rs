//! GTO-02 + GTO-03 basis-set loading.
//!
//! D-01: Runtime resolution + lazy parse + OnceLock cache. No build.rs codegen.
//! D-02: PYSCF_BASIS_PATH env override → CARGO_MANIFEST_DIR walk-up → error.
//!
//! Layout mirrors upstream `pyscf/gto/basis/` (4 parser modules per the
//! RESEARCH.md "Discretion: mirror upstream" mandate).
//!
//! Public surface:
//!   * [`load_basis`] — name + element-symbol → [`pyscf_core::ParsedBasis`]
//!   * [`load_pseudo`] — name + element-symbol → [`pyscf_core::GthPseudo`]
//!   * [`parse`]      — inline NWChem / Gaussian-94 text → [`pyscf_core::ParsedBasis`]
//!   * [`parse_pseudo`] — inline CP2K/GTH pseudopotential text → [`pyscf_core::GthPseudo`]
//!   * [`canonicalise_basis_name`] — case-insensitive ALIAS-key normaliser

pub mod alias;
pub mod cp2k;
pub mod cp2k_pp;
pub mod nwchem;
pub mod nwchem_ecp;
pub mod path;

use pyscf_core::{BasisLoadError, EcpLoadError, GthPseudo, ParsedBasis};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Process-local parsed-basis cache. Key: `(canonical_basis_name, element_symbol_upper)`.
///
/// Lookups are protected by a [`Mutex`] (per RESEARCH Pitfall 6 — simpler than
/// the per-name `OnceLock<RwLock<...>>` pattern; basis-load latency is dwarfed
/// by integral evaluation).
static BASIS_CACHE: OnceLock<Mutex<HashMap<(String, String), ParsedBasis>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<(String, String), ParsedBasis>> {
    BASIS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a basis for one element symbol. First call parses + caches; subsequent
/// calls hit the cache.
///
/// Source: `pyscf/gto/basis/__init__.py:621-728` `load`.
pub fn load_basis(name: &str, symbol: &str) -> Result<ParsedBasis, BasisLoadError> {
    let canonical = canonicalise_basis_name(name);
    let key = (canonical.clone(), symbol.to_ascii_uppercase());

    // Fast path: cache hit. The lock-poison case unwraps to a panic, which is
    // fine because if the cache is poisoned the wider basis-load path is
    // already in undefined territory and the user should see a clear stack.
    if let Some(parsed) = cache()
        .lock()
        .expect("BASIS_CACHE mutex poisoned")
        .get(&key)
    {
        return Ok(parsed.clone());
    }

    // Slow path: resolve filename + table via ALIAS, read file, dispatch
    // parser. The table tells us which on-disk tree holds the file: standard
    // Gaussian sets live under `pyscf/gto/basis/`, while CP2K/GTH basis sets
    // and pseudopotentials live under `pyscf/pbc/gto/basis/`.
    let (filename, kind) =
        alias::lookup_kind(&canonical).ok_or_else(|| BasisLoadError::UnknownName {
            name: canonical.clone(),
        })?;
    let dir = match kind {
        alias::AliasKind::Standard => path::basis_dir()?,
        alias::AliasKind::Gth => path::pbc_basis_dir()?,
        // PP entries are pseudopotentials, not basis sets: they parse into
        // `GthPseudo` (via `load_pseudo`), not `ParsedBasis`, and live in the
        // pseudo tree — so a PP name reaching `load_basis` is a usage error.
        alias::AliasKind::Pp => {
            return Err(BasisLoadError::UnknownName {
                name: format!(
                    "'{canonical}' is a GTH pseudopotential, not a basis set; use load_pseudo"
                ),
            });
        }
    };
    let full = dir.join(filename);
    let text = std::fs::read_to_string(&full).map_err(|e| BasisLoadError::Io {
        path: full.display().to_string(),
        source: e,
    })?;

    // Format detection: GTH-prefixed text routes to CP2K; otherwise NWChem /
    // Gaussian-94. The GTH detection mirrors upstream
    // `pyscf/gto/basis/__init__.py:_format_basis` line 656.
    let parsed = if text.contains("GTH") {
        cp2k::parse_cp2k(&text, &key.1, &full.display().to_string())?
    } else {
        nwchem::parse_nwchem(&text, &key.1, &full.display().to_string())?
    };

    cache()
        .lock()
        .expect("BASIS_CACHE mutex poisoned")
        .insert(key.clone(), parsed.clone());
    tracing::debug!(name = %canonical, symbol = %key.1, file = %full.display(), "basis loaded");
    Ok(parsed)
}

/// Lower-case + strip `-` / `_` so `"cc-pVDZ"`, `"CC-PVDZ"`, `"ccpvdz"` all map
/// to the same ALIAS key. Mirrors upstream `_format_basis_name` at
/// `pyscf/gto/basis/__init__.py:625`.
pub fn canonicalise_basis_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', '_'], "")
}

/// User entry point — parse Gaussian-94 / NWChem text directly (no file read).
///
/// Source: `pyscf/gto/basis/__init__.py:730-779` `parse`.
pub fn parse(text: &str, symbol: &str) -> Result<ParsedBasis, BasisLoadError> {
    if text.contains("GTH") {
        cp2k::parse_cp2k(text, symbol, "<inline-text>")
    } else {
        nwchem::parse_nwchem(text, symbol, "<inline-text>")
    }
}

/// Process-local parsed-pseudopotential cache. Key:
/// `(canonical_pseudo_name, element_symbol_upper)`.
static PSEUDO_CACHE: OnceLock<Mutex<HashMap<(String, String), GthPseudo>>> = OnceLock::new();

fn pseudo_cache() -> &'static Mutex<HashMap<(String, String), GthPseudo>> {
    PSEUDO_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a GTH pseudopotential for one element symbol. First call parses +
/// caches; subsequent calls hit the cache.
///
/// Resolves the name via `PP_ALIAS` (`AliasKind::Pp`), reads from the PBC
/// pseudo tree (`pyscf/pbc/gto/pseudo/`), and parses with
/// [`cp2k_pp::parse_cp2k_pp`] into [`pyscf_core::GthPseudo`]. Mirrors upstream
/// `pyscf/gto/basis/__init__.py` `load_pseudo`.
pub fn load_pseudo(name: &str, symbol: &str) -> Result<GthPseudo, EcpLoadError> {
    // Upstream `_format_pseudo_name` (`gto/basis/__init__.py:834-842`) splits a
    // trailing `q<digits>` off the canonicalised name: `"gth-pade-q4"` resolves
    // through the `gthpade` alias but selects the `-q4` block. Without the
    // split the name would simply miss `PP_ALIAS`.
    let (canonical, suffix) = split_pseudo_suffix(&canonicalise_basis_name(name));
    let key = (
        format!("{canonical}|{}", suffix.as_deref().unwrap_or("")),
        symbol.to_ascii_uppercase(),
    );

    if let Some(p) = pseudo_cache()
        .lock()
        .expect("PSEUDO_CACHE mutex poisoned")
        .get(&key)
    {
        return Ok(p.clone());
    }

    let (filename, kind) = alias::lookup_kind(&canonical)
        .ok_or_else(|| EcpLoadError::UnknownName(canonical.clone()))?;
    if kind != alias::AliasKind::Pp {
        return Err(EcpLoadError::Parse {
            file: canonical.clone(),
            line: 0,
            reason: format!(
                "'{canonical}' is a basis set, not a GTH pseudopotential; use load_basis"
            ),
        });
    }

    let dir = path::pbc_pseudo_dir().map_err(|e| EcpLoadError::Parse {
        file: "<pseudo-dir>".into(),
        line: 0,
        reason: e.to_string(),
    })?;
    let full = dir.join(filename);
    let text = std::fs::read_to_string(&full).map_err(|e| EcpLoadError::Parse {
        file: full.display().to_string(),
        line: 0,
        reason: format!("io error reading pseudopotential file: {e}"),
    })?;

    let parsed = cp2k_pp::parse_cp2k_pp_with_suffix(
        &text,
        &key.1,
        suffix.as_deref(),
        &full.display().to_string(),
    )?;
    pseudo_cache()
        .lock()
        .expect("PSEUDO_CACHE mutex poisoned")
        .insert(key.clone(), parsed.clone());
    tracing::debug!(name = %canonical, symbol = %key.1, file = %full.display(), "pseudopotential loaded");
    Ok(parsed)
}

/// Split a canonicalised pseudopotential name into `(alias_key, q_suffix)`.
/// Ports `_format_pseudo_name` (`pyscf/gto/basis/__init__.py:834-842`) with
/// upstream's `SUFFIX_PATTERN` = a trailing `q<digits>`.
fn split_pseudo_suffix(canonical: &str) -> (String, Option<String>) {
    let bytes = canonical.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i < bytes.len() && i > 0 && bytes[i - 1] == b'q' {
        return (canonical[..i - 1].to_string(), Some(canonical[i - 1..].to_string()));
    }
    (canonical.to_string(), None)
}

/// User entry point — parse CP2K/GTH pseudopotential text directly (no file
/// read). Mirrors [`parse`] for the pseudopotential surface.
pub fn parse_pseudo(text: &str, symbol: &str) -> Result<GthPseudo, EcpLoadError> {
    cp2k_pp::parse_cp2k_pp(text, symbol, "<inline-pseudo-text>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalise_strips_dashes_and_lowercases() {
        assert_eq!(canonicalise_basis_name("cc-pVDZ"), "ccpvdz");
        assert_eq!(canonicalise_basis_name("CC_PVDZ"), "ccpvdz");
        assert_eq!(canonicalise_basis_name("STO-3G"), "sto3g");
        assert_eq!(canonicalise_basis_name("def2-SVP"), "def2svp");
    }

    /// End-to-end GTH basis load: the `gthszv` alias resolves to `gth-szv.dat`,
    /// which lives under the PBC tree (`pyscf/pbc/gto/basis/`), NOT the standard
    /// `pyscf/gto/basis/`. Exercises the GTH-aware directory routing
    /// (`lookup_kind` → `pbc_basis_dir`). Mirrors `path::dev_walk_up_*` in
    /// relying on the in-repo `pyscf/` tree.
    #[test]
    fn load_basis_gth_szv_resolves_from_pbc_tree() {
        // H SZV-GTH: a single l=0 shell with 4 primitives, one contraction.
        let parsed = load_basis("gth-szv", "H").expect("gth-szv must resolve from the PBC tree");
        assert_eq!(parsed.shells.len(), 1);
        assert_eq!(parsed.shells[0].l, 0);
        assert_eq!(parsed.shells[0].exponents.len(), 4);
        approx::assert_abs_diff_eq!(parsed.shells[0].exponents[0], 8.3744350009, epsilon = 1e-9);
    }

    /// End-to-end GTH pseudopotential load: `gth-pade` resolves via PP_ALIAS to
    /// `gth-pade.dat` under the PBC pseudo tree (`pyscf/pbc/gto/pseudo/`) and
    /// parses into a `GthPseudo`.
    ///
    /// Na is the regression case for the DEFAULT-BLOCK rule
    /// (`parse_cp2k_pp.py:145-158`, ported in `cp2k_pp::find_element_block`):
    /// the file holds `Na GTH-PADE-q1 GTH-LDA-q1` FIRST and
    /// `Na GTH-PADE-q9 … GTH-PADE GTH-LDA` second, and the default potential is
    /// the **q9** one — the block whose last alias is not `-q<n>`. Taking the
    /// first matching block instead gave `Zion = 1` for sodium and a cell with
    /// eight electrons too few. Values below are upstream
    /// `pyscf.gto.basis.load_pseudo('gth-pade', 'Na')`.
    #[test]
    fn load_pseudo_gth_pade_resolves_from_pseudo_tree() {
        let pp = load_pseudo("gth-pade", "Na").expect("gth-pade must resolve from the pseudo tree");
        assert_eq!(pp.nelec, vec![3, 6], "Na's DEFAULT gth-pade potential is q9");
        approx::assert_abs_diff_eq!(pp.rloc, 0.24631780, epsilon = 1e-12);
        assert_eq!(pp.local_coeffs, vec![-7.54559253, 1.12599671]);
        assert_eq!(pp.projectors.len(), 2);
        assert_eq!(pp.projectors[0].nproj, 1);
        approx::assert_abs_diff_eq!(pp.projectors[0].r, 0.14125125, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(pp.projectors[0].h[0], 36.55698653, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(pp.projectors[1].h[0], -10.39208332, epsilon = 1e-12);

        // An explicit `-q<n>` suffix selects that block instead — here the q1
        // one, whose l=0 channel does carry a 2x2 symmetric h-matrix.
        let q1 = load_pseudo("gth-pade-q1", "Na").expect("explicit q1 suffix must resolve");
        assert_eq!(q1.nelec, vec![1]);
        approx::assert_abs_diff_eq!(q1.rloc, 0.88550938, epsilon = 1e-12);
        assert_eq!(q1.projectors[0].nproj, 2);
        approx::assert_abs_diff_eq!(q1.projectors[0].h[1], q1.projectors[0].h[2], epsilon = 1e-15);

        // `gthlda` is an alias for the same PADE file (LDA == PADE).
        let lda = load_pseudo("gth-lda", "H").expect("gthlda alias must resolve");
        assert_eq!(lda.nelec, vec![1]);
        assert!(lda.projectors.is_empty()); // H GTH-PADE-q1 is local-only.
    }

    /// A pseudopotential name passed to `load_basis` is a usage error, not a
    /// silent wrong-directory read.
    #[test]
    fn load_basis_rejects_pseudopotential_name() {
        let err = load_basis("gth-pade", "Na").unwrap_err();
        match err {
            BasisLoadError::UnknownName { name } => {
                assert!(name.contains("pseudopotential"), "{name}");
            }
            other => panic!("expected UnknownName, got {other:?}"),
        }
    }

    /// A basis name passed to `load_pseudo` is the mirror-image usage error.
    #[test]
    fn load_pseudo_rejects_basis_name() {
        let err = load_pseudo("gth-szv", "H").unwrap_err();
        match err {
            EcpLoadError::Parse { reason, .. } => {
                assert!(reason.contains("not a GTH pseudopotential"), "{reason}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
