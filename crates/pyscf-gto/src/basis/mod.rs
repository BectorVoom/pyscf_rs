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
pub mod bse;
pub mod cp2k;
pub mod cp2k_pp;
pub mod nwchem;
pub mod nwchem_ecp;
pub mod path;
pub mod pydict;

use pyscf_core::{BasisLoadError, EcpLoadError, GthPseudo, ParsedBasis, ParsedEcp};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Where a cached basis came from. Part of the cache key, so a downloaded
/// basis can never be handed back to [`load_basis_local`]: that function
/// promises "the vendored files say X", and a cache shared with the network
/// path would make its answer depend on whether something else happened to
/// download the same basis earlier in the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    /// Parsed from a file under the vendored basis tree.
    Local,
    /// Fetched from the Basis Set Exchange.
    Bse,
}

/// Process-local parsed-basis cache. Key:
/// `(canonical_basis_name, element_symbol_upper, source)`.
///
/// Lookups are protected by a [`Mutex`] (per RESEARCH Pitfall 6 — simpler than
/// the per-name `OnceLock<RwLock<...>>` pattern; basis-load latency is dwarfed
/// by integral evaluation).
static BASIS_CACHE: OnceLock<Mutex<HashMap<(String, String, Source), ParsedBasis>>> =
    OnceLock::new();

fn cache() -> &'static Mutex<HashMap<(String, String, Source), ParsedBasis>> {
    BASIS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a basis for one element symbol, falling back to the Basis Set Exchange
/// when the vendored files cannot supply it.
///
/// Two misses route to BSE: a name absent from every ALIAS table, and a known
/// name whose file carries no block for this element (`def2-svp` + `Eu`, say).
/// Upstream only covers the first — its second case raises
/// `BasisNotFoundError` — but the second is the one that puts a lanthanide in
/// reach of a def2 basis, and the alternative is a hard error either way.
///
/// Use [`load_basis_local`] instead when you only want to know whether the
/// local files cover an element; this function may go to the network.
///
/// Source: `pyscf/gto/basis/__init__.py:621-728` `load`.
pub fn load_basis(name: &str, symbol: &str) -> Result<ParsedBasis, BasisLoadError> {
    // Local data always wins; only a local miss looks outward.
    let miss = match load_basis_local(name, symbol) {
        Ok(parsed) => return Ok(parsed),
        // Only a genuine "the local files do not have this" routes onward. An
        // IO or parse failure means the local data IS meant to answer and is
        // broken, which the network must not paper over.
        Err(e @ (BasisLoadError::UnknownName { .. } | BasisLoadError::ElementAbsent { .. })) => e,
        Err(other) => return Err(other),
    };

    let key = (
        canonicalise_basis_name(name),
        symbol.to_ascii_uppercase(),
        Source::Bse,
    );
    if let Some(parsed) = cache()
        .lock()
        .expect("BASIS_CACHE mutex poisoned")
        .get(&key)
    {
        return Ok(parsed.clone());
    }

    if !bse::is_available() {
        tracing::warn!(
            basis = %name,
            symbol = %symbol,
            "not available from the local basis files; it may exist at the Basis Set \
             Exchange — rebuild pyscf-gto with --features bse to fetch it"
        );
        return Err(miss);
    }

    let parsed = bse::fetch_basis(name, symbol)?;
    cache()
        .lock()
        .expect("BASIS_CACHE mutex poisoned")
        .insert(key, parsed.clone());
    Ok(parsed)
}

/// Resolve a basis from the vendored files only — never the network.
///
/// This is the right entry point for a *probe*: `make_auxbasis` asks whether a
/// predefined auxiliary basis happens to cover an element and quietly generates
/// even-tempered functions when it does not (`pyscf/df/addons.py:196-214`).
/// Routing that question through [`load_basis`] would turn each unanswered
/// probe into an HTTP round-trip and let a downloaded basis silently displace
/// the even-tempered fallback.
pub fn load_basis_local(name: &str, symbol: &str) -> Result<ParsedBasis, BasisLoadError> {
    let canonical = canonicalise_basis_name(name);
    let key = (
        canonical.clone(),
        symbol.to_ascii_uppercase(),
        Source::Local,
    );

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
    // A few upstream sets — MINAO among them — are stored as a Python MODULE
    // (`<name>.py`, one nested-list literal per element) rather than as NWChem
    // text; `basis/__init__.py:665-676` imports those with `importlib`. The
    // ALIAS table names them without the extension, so try the bare name first
    // and fall back to `<name>.py`. Nothing here executes Python — `pydict`
    // reads the literal directly.
    let bare = dir.join(filename);
    let (full, text) = match std::fs::read_to_string(&bare) {
        Ok(t) => (bare, t),
        Err(bare_err) => {
            let py = dir.join(format!("{filename}.py"));
            match std::fs::read_to_string(&py) {
                Ok(t) => (py, t),
                // Report the ORIGINAL miss: the `.py` form is the fallback, and
                // naming it in the error would send a reader looking for the
                // wrong file.
                Err(_) => {
                    return Err(BasisLoadError::Io {
                        path: bare.display().to_string(),
                        source: bare_err,
                    });
                }
            }
        }
    };

    // Format detection: GTH-prefixed text routes to CP2K; a `.py` module routes
    // to the literal parser; otherwise NWChem / Gaussian-94. The GTH detection
    // mirrors upstream `pyscf/gto/basis/__init__.py:_format_basis` line 656.
    let parsed = if full.extension().is_some_and(|e| e == "py") {
        pydict::parse_pydict(&text, &key.1, &full.display().to_string())?
    } else if text.contains("GTH") {
        cp2k::parse_cp2k(&text, &key.1, &full.display().to_string())?
    } else {
        nwchem::parse_nwchem(&text, &key.1, &full.display().to_string())?
    };

    // An element with no block in the file parses to zero shells. Upstream
    // raises here (`pyscf/gto/basis/__init__.py`, via `parse_nwchem.load`);
    // returning `Ok` with an empty basis would let the element contribute no
    // AOs and corrupt the calculation silently. Never cached, so a later
    // `load_basis` can still try the network for it.
    if parsed.shells.is_empty() {
        return Err(BasisLoadError::ElementAbsent {
            name: canonical.clone(),
            symbol: key.1.clone(),
            file: full.display().to_string(),
        });
    }

    cache()
        .lock()
        .expect("BASIS_CACHE mutex poisoned")
        .insert(key.clone(), parsed.clone());
    tracing::debug!(name = %canonical, symbol = %key.1, file = %full.display(), "basis loaded");
    Ok(parsed)
}

/// Resolve the ECP that accompanies a basis set, falling back to the Basis Set
/// Exchange the way [`load_basis`] does.
///
/// `Ok(None)` means "this basis is all-electron for this element" — a normal
/// answer, not a failure (`pyscf/gto/basis/__init__.py:773-777`).
///
/// The fallback is gated on `bse_meta.json`, which records exactly which
/// elements each basis gives an ECP to. Without that gate every all-electron
/// element of every calculation would trigger a network round-trip just to be
/// told "no ECP here".
///
/// Source: `pyscf/gto/basis/__init__.py:730-779` `load_ecp`.
pub fn load_ecp(name: &str, symbol: &str) -> Result<Option<ParsedEcp>, EcpLoadError> {
    match load_ecp_local(name, symbol) {
        Ok(Some(ecp)) => return Ok(Some(ecp)),
        // The local files answered "all-electron" AND the metadata agrees, so
        // that is the real answer — no reason to ask the network.
        Ok(None) if !bse_defines_ecp(name, symbol) => return Ok(None),
        // Either the metadata says this element DOES take an ECP under this
        // basis (so the local files are simply incomplete — def2-svp + Eu), or
        // the name is not one we carry at all. Both look outward.
        Ok(None) => {}
        Err(EcpLoadError::UnknownName(_)) => {}
        Err(other) => return Err(other),
    }

    if !bse::is_available() {
        // Returning `None` here would be silently catastrophic: a valence-only
        // basis paired with no ECP describes an atom with the wrong number of
        // electrons, and nothing downstream can detect it.
        return Err(EcpLoadError::Bse {
            name: name.to_string(),
            symbol: symbol.to_string(),
            reason: "this basis defines an ECP for this element but the local files do not \
                     carry it; rebuild pyscf-gto with --features bse to fetch it"
                .into(),
        });
    }
    bse::fetch_ecp(name, symbol)
}

/// ECP resolution from the vendored files only — never the network. The
/// counterpart of [`load_basis_local`].
pub fn load_ecp_local(name: &str, symbol: &str) -> Result<Option<ParsedEcp>, EcpLoadError> {
    let canonical = canonicalise_basis_name(name);
    let filename =
        alias::lookup(&canonical).ok_or_else(|| EcpLoadError::UnknownName(canonical.clone()))?;

    let dir = path::basis_dir().map_err(|e| EcpLoadError::Parse {
        file: "<basis-dir>".into(),
        line: 0,
        reason: e.to_string(),
    })?;
    let full = dir.join(filename);
    let text = std::fs::read_to_string(&full).map_err(|e| EcpLoadError::Parse {
        file: full.display().to_string(),
        line: 0,
        reason: format!("io error reading ECP file: {e}"),
    })?;

    // No ECP section at all — normal for sto-3g and friends.
    if !text.contains("ECP") {
        return Ok(None);
    }
    match nwchem_ecp::parse_nwchem_ecp(&text, symbol, &full.display().to_string()) {
        Ok(p) if p.channels.is_empty() => Ok(None),
        Ok(p) => Ok(Some(p)),
        // The file has ECP blocks, just not for this element.
        Err(EcpLoadError::UnknownName(_)) => Ok(None),
        Err(other) => Err(other),
    }
}

/// Does `bse_meta.json` record an ECP for this element under this basis?
///
/// This is upstream's `bse_predefined_ecp` test (`pyscf/gto/mole.py:4317-4334`)
/// and it is what keeps the ECP fallback from stampeding the network.
fn bse_defines_ecp(name: &str, symbol: &str) -> bool {
    let Some(zs) = bse::ecp_elements(name) else {
        return false;
    };
    let Some(z) = pyscf_core::elements::charge_for_symbol(symbol) else {
        return false;
    };
    u32::try_from(z).is_ok_and(|z| zs.contains(&z))
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
        return (
            canonical[..i - 1].to_string(),
            Some(canonical[i - 1..].to_string()),
        );
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
        assert_eq!(
            pp.nelec,
            vec![3, 6],
            "Na's DEFAULT gth-pade potential is q9"
        );
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
        approx::assert_abs_diff_eq!(
            q1.projectors[0].h[1],
            q1.projectors[0].h[2],
            epsilon = 1e-15
        );

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
