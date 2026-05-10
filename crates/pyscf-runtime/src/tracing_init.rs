//! Library-level tracing helper. Per RESEARCH §12: this crate emits
//! `tracing::info!` events; binaries (Phase 3 pyscf-py, future pyscf-cli)
//! install the subscriber. `init_tracing` is a convenience wrapper a
//! binary can call to map `mol.verbose` (0..=9) to a LevelFilter.

use tracing_subscriber::{filter::LevelFilter, fmt, prelude::*};

/// Map upstream PySCF's verbose 0..=9 to LevelFilter.
/// 0 = Off, 1..=2 = Error, 3..=4 = Warn, 5..=6 = Info, 7 = Debug,
/// 8..=9 = Trace.
pub fn verbose_to_filter(verbose: u8) -> LevelFilter {
    match verbose {
        0 => LevelFilter::OFF,
        1..=2 => LevelFilter::ERROR,
        3..=4 => LevelFilter::WARN,
        5..=6 => LevelFilter::INFO,
        7 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    }
}

/// Install a tracing subscriber at the given verbose level. **Idempotent**:
/// safe to call multiple times (subsequent calls are no-ops). Library
/// crates should NOT call this; binaries should.
///
/// Returns `true` if this call installed the subscriber, `false` if one
/// was already installed (allowing tests to install a per-test
/// subscriber once via `try_init`).
pub fn init_tracing(verbose: u8) -> bool {
    let filter = verbose_to_filter(verbose);
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .try_init()
        .is_ok()
}
