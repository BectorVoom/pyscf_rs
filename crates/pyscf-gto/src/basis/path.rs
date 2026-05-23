//! D-02: PYSCF_BASIS_PATH resolver with priority chain.
//!
//! Priority:
//! 1. PYSCF_BASIS_PATH env var (if set and `is_dir`)
//! 2. CARGO_MANIFEST_DIR walk-up looking for `../../pyscf/gto/basis`
//! 3. CARGO_MANIFEST_DIR walk-up looking for `../pyscf/gto/basis`
//! 4. CARGO_MANIFEST_DIR / `pyscf/gto/basis` (when running from crate dir variant)
//! 5. Error: `BasisLoadError::PathNotFound`
//!
//! The resolver caches its result in a `OnceLock<Option<PathBuf>>` so subsequent
//! calls are free; the env var is read once at first call.
//!
//! Source: pattern aligns with Phase 1 D-07 PYSCF_BACKEND env-var resolver.

use pyscf_core::BasisLoadError;
use std::path::PathBuf;
use std::sync::OnceLock;

static BASIS_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Resolve the basis directory once and return a borrow of the cached path.
///
/// Returns `Err(BasisLoadError::PathNotFound)` if no candidate location exists.
pub fn basis_dir() -> Result<&'static PathBuf, BasisLoadError> {
    let resolved = BASIS_DIR.get_or_init(|| {
        // (1) Env-var override.
        if let Ok(p) = std::env::var("PYSCF_BASIS_PATH") {
            let path = PathBuf::from(&p);
            if path.is_dir() {
                tracing::info!(path = %path.display(), "PYSCF_BASIS_PATH resolved basis dir");
                return Some(path);
            }
            tracing::warn!(path = %p, "PYSCF_BASIS_PATH set but not a directory; falling through");
        }
        // (2)/(3)/(4) Walk-up candidates relative to this crate's manifest dir.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest.join("../../pyscf/gto/basis"),
            manifest.join("../pyscf/gto/basis"),
            manifest.join("pyscf/gto/basis"),
        ];
        for c in &candidates {
            if c.is_dir() {
                let canonical = c.canonicalize().unwrap_or_else(|_| c.clone());
                tracing::info!(path = %canonical.display(), "walk-up resolved basis dir");
                return Some(canonical);
            }
        }
        None
    });

    resolved
        .as_ref()
        .ok_or_else(|| BasisLoadError::PathNotFound {
            tried: "PYSCF_BASIS_PATH env, CARGO_MANIFEST_DIR/../../pyscf/gto/basis, \
                CARGO_MANIFEST_DIR/../pyscf/gto/basis, CARGO_MANIFEST_DIR/pyscf/gto/basis"
                .into(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_walk_up_finds_pyscf_basis() {
        // The repo root has `pyscf/gto/basis/` at top level; walk-up MUST find it.
        let dir = basis_dir().expect("walk-up should find pyscf/gto/basis");
        assert!(dir.is_dir(), "{:?} is not a dir", dir);
        // Sanity: the directory contains sto-3g.dat (a known builtin).
        assert!(
            dir.join("sto-3g.dat").is_file(),
            "expected {}/sto-3g.dat to exist",
            dir.display()
        );
    }
}
