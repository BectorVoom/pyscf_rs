//! Basis Set Exchange (BSE) online download — the last-resort basis/ECP source.
//!
//! Upstream reaches BSE through the installed `basis_set_exchange` Python
//! package (`pyscf/gto/basis/__init__.py:699-714` for orbital basis,
//! `765-779` for ECP), converting that package's JSON via
//! `pyscf/gto/basis/bse.py`. There is no such package in Rust, so this module
//! talks to the same database over its public REST API and asks for the
//! **NWChem** serialisation, which the already-tested [`nwchem`] and
//! [`nwchem_ecp`] parsers consume unchanged. BSE emits the orbital basis and
//! the ECP in a single document, so one request serves both surfaces.
//!
//! Two caches sit in front of the network:
//!   * on disk, so a fetched basis survives the process and later runs may be
//!     fully offline;
//!   * in memory, via the existing `BASIS_CACHE` in the parent module.
//!
//! Environment:
//!   * `PYSCF_BSE_CACHE_DIR` — override the on-disk cache location.
//!   * `PYSCF_BSE_OFFLINE=1` — serve from the disk cache only; never hit the
//!     network. A cache miss then fails instead of downloading.

use super::{nwchem, nwchem_ecp, path};
use pyscf_core::{BasisLoadError, EcpLoadError, ParsedBasis, ParsedEcp};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

#[cfg(feature = "bse")]
const BSE_API_ROOT: &str = "https://www.basissetexchange.org/api";
#[cfg(feature = "bse")]
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Fetch an orbital basis for one element from BSE.
///
/// `name` is the user-supplied basis name; it is resolved through
/// `bse_meta.json` to the database's official spelling when possible.
pub fn fetch_basis(name: &str, symbol: &str) -> Result<ParsedBasis, BasisLoadError> {
    let bse_name = official_name(name);
    let text = nwchem_document(&bse_name, symbol).map_err(|reason| BasisLoadError::Bse {
        name: bse_name.clone(),
        symbol: symbol.to_string(),
        reason,
    })?;

    let source = format!("<bse:{bse_name}/{symbol}>");
    let parsed = nwchem::parse_nwchem(&text, symbol, &source)?;
    if parsed.shells.is_empty() {
        return Err(BasisLoadError::Bse {
            name: bse_name,
            symbol: symbol.to_string(),
            reason: "the database returned no orbital shells for this element".into(),
        });
    }
    tracing::info!(basis = %bse_name, symbol = %symbol, shells = parsed.shells.len(),
        "basis obtained from the Basis Set Exchange");
    Ok(parsed)
}

/// Fetch the ECP that accompanies `name` for one element, if the basis defines
/// one. `Ok(None)` means the basis exists at BSE but is all-electron for this
/// element — upstream's `load_ecp` likewise returns an empty record rather than
/// raising (`pyscf/gto/basis/__init__.py:773-777`).
pub fn fetch_ecp(name: &str, symbol: &str) -> Result<Option<ParsedEcp>, EcpLoadError> {
    let bse_name = official_name(name);
    let text = nwchem_document(&bse_name, symbol).map_err(|reason| EcpLoadError::Bse {
        name: bse_name.clone(),
        symbol: symbol.to_string(),
        reason,
    })?;

    if !text.contains("ECP") {
        return Ok(None);
    }
    let source = format!("<bse:{bse_name}/{symbol}>");
    match nwchem_ecp::parse_nwchem_ecp(&text, symbol, &source) {
        Ok(p) if p.channels.is_empty() => Ok(None),
        Ok(p) => {
            tracing::info!(ecp = %bse_name, symbol = %symbol, n_core = p.n_core,
                "ECP obtained from the Basis Set Exchange");
            Ok(Some(p))
        }
        Err(EcpLoadError::UnknownName(_)) => Ok(None),
        Err(other) => Err(other),
    }
}

/// Disk cache → network. Returns the raw NWChem document.
fn nwchem_document(bse_name: &str, symbol: &str) -> Result<String, String> {
    let cached = cache_path(bse_name, symbol);
    if let Some(p) = &cached
        && let Ok(text) = std::fs::read_to_string(p)
    {
        tracing::debug!(path = %p.display(), "BSE disk-cache hit");
        return Ok(text);
    }

    if offline() {
        return Err(format!(
            "PYSCF_BSE_OFFLINE is set and no cached copy exists at {}",
            cached
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<no cache dir>".into())
        ));
    }

    let text = download(bse_name, symbol)?;

    if let Some(p) = &cached {
        // A cache write is an optimisation: report and carry on if it fails.
        match store(p, &text) {
            Ok(()) => tracing::debug!(path = %p.display(), "BSE response cached"),
            Err(e) => {
                tracing::warn!(path = %p.display(), error = %e, "could not cache BSE response")
            }
        }
    }
    Ok(text)
}

/// Write the cache entry through a process-private temporary file and rename it
/// into place. An MPI job starts every rank at once and they all miss the cache
/// together; a plain write would let them interleave into one torn file that
/// every later run then reads back as a corrupt basis. `rename` within a
/// directory is atomic, so a reader sees either no entry or a complete one.
fn store(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    let Some(dir) = path.parent() else {
        return std::fs::write(path, text);
    };
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("entry"),
        std::process::id()
    ));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

#[cfg(feature = "bse")]
fn download(bse_name: &str, symbol: &str) -> Result<String, String> {
    let url = format!(
        "{BSE_API_ROOT}/basis/{}/format/nwchem/?elements={}",
        percent_encode(bse_name),
        percent_encode(symbol)
    );
    tracing::info!(%url, "requesting basis from the Basis Set Exchange");

    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .user_agent(concat!("pyscf-rs/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("could not build the HTTP client: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .map_err(|e| format!("request to {url} failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .map_err(|e| format!("could not read the response body from {url}: {e}"))?;

    if !status.is_success() {
        // BSE answers an unknown basis or element with a 4xx and a short
        // plain-text explanation; surfacing it verbatim is the most useful
        // thing we can tell the caller.
        let detail = body.trim();
        let detail = if detail.len() > 300 {
            &detail[..300]
        } else {
            detail
        };
        return Err(format!("the database answered {status}: {detail}"));
    }
    Ok(body)
}

#[cfg(not(feature = "bse"))]
fn download(_bse_name: &str, _symbol: &str) -> Result<String, String> {
    Err(
        "this build has no Basis Set Exchange support; rebuild pyscf-gto with \
         --features bse to enable downloads"
            .into(),
    )
}

/// Whether this build can reach the network at all. The `format_*` layers use
/// it to choose between a "not in ALIAS" and a "not at BSE either" message.
pub fn is_available() -> bool {
    cfg!(feature = "bse")
}

fn offline() -> bool {
    std::env::var("PYSCF_BSE_OFFLINE").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// `<cache dir>/<basis>/<SYMBOL>.nwchem`, or `None` when no cache directory can
/// be determined (the fetch then simply goes uncached).
///
/// `bse_name` must already be the official spelling from [`official_name`].
/// Public so a cache can be pre-seeded on a machine with egress and shipped to
/// one without, which with `PYSCF_BSE_OFFLINE=1` is the air-gapped workflow.
pub fn cache_path(bse_name: &str, symbol: &str) -> Option<PathBuf> {
    Some(
        cache_dir()?
            .join(sanitise(bse_name))
            .join(format!("{}.nwchem", sanitise(&symbol.to_ascii_uppercase()))),
    )
}

/// `PYSCF_BSE_CACHE_DIR` → `$XDG_CACHE_HOME/pyscf-rs/bse` → `$HOME/.cache/pyscf-rs/bse`.
pub fn cache_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PYSCF_BSE_CACHE_DIR")
        && !p.is_empty()
    {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME")
        && !p.is_empty()
    {
        return Some(PathBuf::from(p).join("pyscf-rs/bse"));
    }
    std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".cache/pyscf-rs/bse"))
}

/// Keep cache paths to one flat, filesystem-safe component. Basis names carry
/// `*`, `+` and `/` (`6-31+G*`, `Stuttgart RSC 1997`), none of which belong in
/// a path segment.
fn sanitise(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Percent-encode everything outside the RFC 3986 unreserved set, so basis
/// names like `6-31+G*` survive both the path segment and the query string.
#[cfg(feature = "bse")]
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(*b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Resolve a pyscf basis alias to the database's official spelling using the
/// vendored `bse_meta.json`, the same table upstream loads at
/// `pyscf/gto/mole.py:102-104` and consults in `bse_predefined_ecp`
/// (`mole.py:4317-4334`). Unknown names pass through untouched — BSE does its
/// own name normalisation and will answer 4xx if it really has no such set.
pub fn official_name(name: &str) -> String {
    let canonical = super::canonicalise_basis_name(name);
    meta()
        .get(&canonical)
        .and_then(|e| e.as_ref())
        .map(|e| e.official.clone())
        .unwrap_or_else(|| name.to_string())
}

/// Atomic numbers for which `name` carries an ECP, per `bse_meta.json`. Empty
/// when the set is all-electron; `None` when the name is not in the table.
pub fn ecp_elements(name: &str) -> Option<Vec<u32>> {
    let canonical = super::canonicalise_basis_name(name);
    meta()
        .get(&canonical)
        .and_then(|e| e.as_ref())
        .map(|e| e.ecp_elements.clone())
}

/// One `bse_meta.json` row: `[official name, [ECP element Z...], {variants}]`.
/// Rows for basis sets with no BSE counterpart are `null`.
struct MetaEntry {
    official: String,
    ecp_elements: Vec<u32>,
}

static META: OnceLock<HashMap<String, Option<MetaEntry>>> = OnceLock::new();

fn meta() -> &'static HashMap<String, Option<MetaEntry>> {
    META.get_or_init(|| {
        let Ok(dir) = path::basis_dir() else {
            return HashMap::new();
        };
        let file = dir.join("bse_meta.json");
        let Ok(text) = std::fs::read_to_string(&file) else {
            tracing::warn!(path = %file.display(), "bse_meta.json unreadable; \
                BSE names will be passed through verbatim");
            return HashMap::new();
        };
        let Ok(raw) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&text) else {
            tracing::warn!(path = %file.display(), "bse_meta.json is not a JSON object; ignoring");
            return HashMap::new();
        };

        raw.into_iter()
            .map(|(k, v)| {
                let entry = v.as_array().and_then(|row| {
                    let official = row.first()?.as_str()?.to_string();
                    let ecp_elements = row
                        .get(1)
                        .and_then(|e| e.as_array())
                        .map(|zs| {
                            zs.iter()
                                .filter_map(|z| z.as_u64().map(|z| z as u32))
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(MetaEntry {
                        official,
                        ecp_elements,
                    })
                });
                (k, entry)
            })
            .collect()
    })
}
