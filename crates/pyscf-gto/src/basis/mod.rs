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
//!   * [`parse`]      — inline NWChem / Gaussian-94 text → [`pyscf_core::ParsedBasis`]
//!   * [`canonicalise_basis_name`] — case-insensitive ALIAS-key normaliser

pub mod alias;
pub mod cp2k;
pub mod cp2k_pp;
pub mod nwchem;
pub mod nwchem_ecp;
pub mod path;

use pyscf_core::{BasisLoadError, ParsedBasis};
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
        alias::AliasKind::Gth | alias::AliasKind::Pp => path::pbc_basis_dir()?,
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
}
