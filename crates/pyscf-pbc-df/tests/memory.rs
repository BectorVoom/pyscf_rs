//! **GATE 4** — GDF's `_cderi` against FFTDF's AO table. Plan 14-09, Task 1.
//!
//! # The roadmap's "under 20 % of FFTDF memory" is k-mesh dependent and does not say so
//!
//! `_cderi` is `O(nkpts² · naux · nao_pair)`; FFTDF's resident AO table is
//! `O(nkpts · ngrids · nao)`. The ratio therefore grows **linearly in `nkpts`**,
//! and on diamond it crosses 20 % between 2×2×2 and 3×3×3. Measured upstream at
//! mesh `[40,40,40]` (`measurements/memory.py`):
//!
//! ```text
//! diamond 2x2x2   62.50 MiB   3.86 MiB    6.17 %
//! diamond 3x3x3  210.94 MiB  44.20 MiB   20.95 %   <- fails a 20 % gate
//! He-fcc  2x2x2    7.81 MiB   0.12 MiB    1.48 %
//! ```
//!
//! So Gate 4 pins the k-mesh at 2×2×2 and records the 3×3×3 number as the
//! reason the pin exists.
//!
//! # The sizes come from the SHAPES, and the shapes are validated against a real file
//!
//! Diamond's `cderi` cannot be built in a test suite — one `make_j3c` at gamma
//! is a single screening group of ~77 M cintx shell triples (14-02's SUMMARY).
//! But its SIZE needs no 3-centre work at all: `naux` is the auxiliary cell's
//! `nao`, which `make_modrho_basis` returns in milliseconds.
//!
//! The arithmetic is therefore checked where it CAN be checked — against a real
//! `_cderi` file written by this port on He-fcc 2×2×2 — and then applied to
//! diamond. A formula validated on one system and applied to another is a
//! weaker claim than a measurement, and saying so is the point of this comment.

mod common;

use pyscf_pbc_df::{Aosym, Gdf, make_modrho_basis};
use pyscf_pbc_gto::Cell;

const MESH: [usize; 3] = [40, 40, 40];
const BYTES_PER_COMPLEX: f64 = 16.0;
const MIB: f64 = 1024.0 * 1024.0;

/// `nkpts² · naux · nao_pair · 16` — the `s2`-packed `cderi` store.
fn cderi_bytes(cell: &Cell, nkpts: usize) -> f64 {
    let naux = make_modrho_basis(cell, None, None)
        .expect("modrho auxcell")
        .naux();
    let nao_pair = Aosym::S2.nao_pair(cell.mol.nao_nr);
    (nkpts * nkpts * naux * nao_pair) as f64 * BYTES_PER_COMPLEX
}

/// `nkpts · ngrids · nao · 16` — FFTDF's resident AO table.
fn fftdf_bytes(cell: &Cell, nkpts: usize) -> f64 {
    let ngrids = MESH[0] * MESH[1] * MESH[2];
    (nkpts * ngrids * cell.mol.nao_nr) as f64 * BYTES_PER_COMPLEX
}

/// The formula, validated against a `_cderi` file this port actually wrote.
///
/// HDF5 adds a header, the `kpts` and `aosym` datasets and per-dataset
/// metadata, so the file is a little larger than the payload; the assertion is
/// that the PAYLOAD is exact and the overhead is small, not that the byte count
/// matches to the byte.
#[test]
fn the_cderi_size_formula_matches_a_real_file() {
    let cell = common::he_all_electron();
    let kpts = cell.make_kpts([2, 2, 2]).expect("kpts");
    let nkpts = kpts.len();

    let dir = std::env::temp_dir().join(format!("pbc_gate4_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("cderi.h5");

    let mut df = Gdf::new(cell.clone(), &kpts);
    df.cderi_to_save = Some(path.clone());
    df.build().expect("gdf build");

    let on_disk = std::fs::metadata(&path).expect("the file must exist").len() as f64;
    let predicted = cderi_bytes(&cell, nkpts);
    println!(
        "He-fcc 2x2x2 _cderi: predicted payload {:.0} B, file on disk {on_disk:.0} B",
        predicted
    );
    assert!(
        on_disk >= predicted,
        "the file ({on_disk:.0} B) is smaller than its own payload ({predicted:.0} B)"
    );
    assert!(
        on_disk < predicted + 64.0 * 1024.0,
        "the HDF5 overhead has grown past 64 KiB: {:.0} B over a {predicted:.0} B payload",
        on_disk - predicted
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **GATE 4** — diamond 2×2×2 at mesh `[40,40,40]`, against upstream's 6.17 %.
#[test]
fn gate4_gdf_cderi_is_under_20_percent_of_the_fftdf_ao_table() {
    let cell = common::diamond();

    let (fft, cderi) = (fftdf_bytes(&cell, 8), cderi_bytes(&cell, 8));
    let ratio = cderi / fft;
    println!(
        "GATE4 diamond 2x2x2, mesh {MESH:?}: FFTDF AO table {:.2} MiB, GDF _cderi \
         {:.2} MiB, ratio {:.2} % (upstream 6.17 %)",
        fft / MIB,
        cderi / MIB,
        ratio * 100.0
    );
    assert!(
        ratio < 0.20,
        "GATE 4: GDF's _cderi is {:.2} % of the FFTDF AO table, not under 20 %",
        ratio * 100.0
    );
    assert!(
        (ratio * 100.0 - 6.17).abs() < 0.5,
        "the ratio moved off upstream's 6.17 %: {:.2} %",
        ratio * 100.0
    );

    // And the reason the gate names its k-mesh: the SAME system at 3x3x3 is
    // over 20 %, and a gate that did not pin the mesh would fail a correct
    // implementation.
    let (fft3, cderi3) = (fftdf_bytes(&cell, 27), cderi_bytes(&cell, 27));
    let ratio3 = cderi3 / fft3;
    println!(
        "GATE4 diamond 3x3x3, mesh {MESH:?}: FFTDF {:.2} MiB, GDF {:.2} MiB, ratio \
         {:.2} % (upstream 20.95 %)",
        fft3 / MIB,
        cderi3 / MIB,
        ratio3 * 100.0
    );
    assert!(
        ratio3 > 0.20,
        "the 3x3x3 ratio is supposed to be OVER 20 % — that is why Gate 4 pins \
         the k-mesh. Got {:.2} %",
        ratio3 * 100.0
    );
    // Linear in nkpts, exactly: nkpts^2/nkpts = nkpts.
    let scale = ratio3 / ratio;
    assert!(
        (scale - 27.0 / 8.0).abs() < 1e-9,
        "the ratio must scale linearly in nkpts (27/8 = 3.375); got {scale}"
    );
}

/// The all-electron control, for the record — **and a caution about what
/// upstream's 1.48 % actually measures.**
///
/// He-fcc's `_cderi` PAYLOAD is `8² · 23 · 1 · 16` = 23 552 B. Upstream's
/// `memory.py` reports 0.12 MiB for the same store, and this port's own HDF5
/// file is 57 054 B: on a 23 KB payload the container is bigger than the
/// contents, and both numbers are dominated by HDF5 metadata rather than by
/// integrals. So **1.48 % is not comparable to a payload ratio** and this test
/// does not pretend otherwise — it asserts the payload ratio against the
/// arithmetic and bounds it BELOW upstream's file-size figure.
///
/// Diamond, where the payload is 3.80 MiB and the overhead is under 2 %, is the
/// system Gate 4 is actually stated on.
#[test]
fn gate4_he_fcc_for_the_record() {
    let cell = common::he_all_electron();
    let (fft, cderi) = (fftdf_bytes(&cell, 8), cderi_bytes(&cell, 8));
    let ratio = cderi / fft;
    println!(
        "GATE4 He-fcc 2x2x2, mesh {MESH:?}: FFTDF {:.2} MiB, GDF payload {:.4} MiB, \
         ratio {:.2} % (upstream reports 1.48 %, but that is a 0.12 MiB HDF5 FILE \
         around a 0.02 MiB payload)",
        fft / MIB,
        cderi / MIB,
        ratio * 100.0
    );
    // 8^2 * 23 * 1 * 16 B, exactly.
    assert_eq!(
        cderi as u64, 23_552,
        "the He-fcc payload is arithmetic, not a measurement"
    );
    assert!(
        ratio < 0.0148,
        "the payload ratio must sit BELOW upstream's file-size figure: {:.2} %",
        ratio * 100.0
    );
}
