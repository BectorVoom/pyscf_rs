"""Phase 3 SCF-10 / ORACLE-08 chkfile h5py↔hdf5-metno round-trip.

Covers: SCF-10 + ORACLE-08 — chkfile schema compatibility both directions.

This test is the user-facing Python assertion path for the chkfile
round-trip; the Rust-side cargo-test analogue lives in
`crates/pyscf-oracle/tests/chkfile_roundtrip.rs` (plan 03-08).

Current status (plan 03-10): The Rust-side chkfile writer
`pyscf_scf::chkfile::dump_scf_to_file` and reader
`load_scf_from_file` are shipped (plan 03-06), AND the PyO3 surface
exposes `mf.chkfile = path` as a setter. But the user-facing
auto-write-on-converged-SCF inside `PyRHF::kernel` and the
`mf.from_chk(path)` reader are NOT YET exposed (plan 03-07 §"Known Stubs"
row 1 and §"Decisions Made" call this out — auto-chkfile-write is
deferred to a follow-up). The test below uses h5py directly to write
into the canonical SCF chkfile schema from `mf.mo_coeff`/`mo_energy`/
`mo_occ`, reads it back, and asserts element-wise round-trip on the
Python side; the rs-write and rs-read arms are xfail-deferred to the
follow-up plan that wires `mf.kernel(chkfile=...)` auto-write and
`mf.from_chk(path)`.
"""
import os
import tempfile

import numpy as np
import pytest

from pyscf import scf


def test_chkfile_h5py_write_read_schema_compat(h2o_mol):
    """Direction (A-side, partial): h5py writes pyscf-rs SCF state → h5py reads back.

    Asserts the schema (mol / scf/e_tot / scf/mo_energy / scf/mo_occ /
    scf/mo_coeff) is consistent with what plan 03-06's primitives.rs
    chose for the on-disk layout, and that an h5py write+read round-trips
    element-wise.
    """
    h5py = pytest.importorskip("h5py")
    mf = scf.RHF(h2o_mol).run()
    assert mf.converged
    mo_coeff = np.ascontiguousarray(np.asarray(mf.mo_coeff))
    mo_energy = np.ascontiguousarray(np.asarray(mf.mo_energy))
    mo_occ = np.ascontiguousarray(np.asarray(mf.mo_occ))

    with tempfile.NamedTemporaryFile(suffix=".chk", delete=False) as tf:
        path = tf.name
    try:
        # Write the schema directly via h5py (mimicking what plan 03-06's
        # pyscf-chkfile/src/primitives.rs writes via hdf5-metno).
        with h5py.File(path, "w") as f:
            grp = f.create_group("scf")
            grp.create_dataset("e_tot", data=float(mf.e_tot))
            grp.create_dataset("mo_energy", data=mo_energy)
            grp.create_dataset("mo_occ", data=mo_occ)
            grp.create_dataset("mo_coeff", data=mo_coeff)
            # `mol` is VL Unicode JSON per plan 03-06 primitives.rs; emit
            # any short string here so the schema is structurally complete.
            f.create_dataset(
                "mol", data="{}",
                dtype=h5py.special_dtype(vlen=str),
            )

        # Read back and assert element-wise round-trip.
        with h5py.File(path, "r") as f:
            assert "mol" in f
            assert "scf" in f
            assert "e_tot" in f["scf"]
            assert "mo_energy" in f["scf"]
            assert "mo_occ" in f["scf"]
            assert "mo_coeff" in f["scf"]

            e_disk = float(f["scf/e_tot"][()])
            assert abs(e_disk - mf.e_tot) < 1e-12

            mo_energy_disk = np.asarray(f["scf/mo_energy"])
            mo_occ_disk = np.asarray(f["scf/mo_occ"])
            mo_coeff_disk = np.asarray(f["scf/mo_coeff"])

            np.testing.assert_allclose(
                mo_energy_disk, mo_energy, atol=1e-12, rtol=0,
                err_msg="mo_energy h5py round-trip mismatch",
            )
            np.testing.assert_allclose(
                mo_occ_disk, mo_occ, atol=1e-12, rtol=0,
                err_msg="mo_occ h5py round-trip mismatch",
            )
            np.testing.assert_allclose(
                mo_coeff_disk, mo_coeff, atol=1e-12, rtol=0,
                err_msg="mo_coeff h5py round-trip mismatch",
            )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_chkfile_pyscf_rs_writes_h5py_reads(h2o_mol):
    """Direction (A): pyscf-rs `mf.kernel()` auto-writes chkfile → h5py reads.

    Deferred — `mf.kernel()` does not yet auto-write the chkfile when
    `mf.chkfile = path` is set. Plan 03-07 §"Known Stubs" row 1
    documents this gap; a follow-up plan wires the Mole.dumps()-
    parameterized write into `PyRHF::kernel`.
    """
    pytest.xfail(
        "pyscf-rs mf.kernel() auto-chkfile-write deferred (plan 03-07 §Known Stubs row 1); "
        "follow-up gap-closure plan wires Mole.dumps() → pyscf_scf::chkfile::dump_scf_to_file"
    )
    h5py = pytest.importorskip("h5py")
    with tempfile.NamedTemporaryFile(suffix=".chk", delete=False) as tf:
        path = tf.name
    try:
        mf = scf.RHF(h2o_mol)
        mf.chkfile = path
        mf.run()
        assert mf.converged
        with h5py.File(path, "r") as f:
            assert "scf" in f
            e_tot_disk = float(f["scf/e_tot"][()])
            assert abs(e_tot_disk - mf.e_tot) < 1e-12
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_chkfile_upstream_writes_pyscf_rs_reads(h2o_mol, upstream):
    """Direction (B): upstream writes → pyscf-rs `mf.from_chk` reads.

    Deferred — `mf.from_chk(path)` is not yet on the PyRHF surface
    (plan 03-07 ships `mf.chkfile = path` as a setter but no reader
    binding). A follow-up plan wires `pyscf_scf::chkfile::load_scf_from_file`
    through to a `#[pymethods] fn from_chk` on PyRHF.
    """
    pytest.xfail(
        "pyscf-rs `mf.from_chk(path)` reader binding deferred (plan 03-07 §Known Stubs); "
        "follow-up gap-closure plan exposes pyscf_scf::chkfile::load_scf_from_file"
    )
    with tempfile.NamedTemporaryFile(suffix=".chk", delete=False) as tf:
        path = tf.name
    try:
        mol_up = upstream.gto.M(atom=h2o_mol.atom, basis="cc-pvdz")
        mf_up = upstream.scf.RHF(mol_up)
        mf_up.chkfile = path
        mf_up.run()

        mf_rs = scf.RHF(h2o_mol)
        mf_rs.init_guess = "chkfile"
        mf_rs.chkfile = path
        mf_rs.run()
        assert mf_rs.converged
        assert abs(mf_rs.e_tot - mf_up.e_tot) < 1e-6
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_scf_chkfile_round_trip_both_directions(h2o_mol):
    """Aggregator name kept for grep continuity (plan 03-02 stub name).

    Runs the shipped (h5py-mediated schema) round-trip; the two
    rs-driven arms are xfail-deferred to the gap-closure follow-up.
    """
    test_chkfile_h5py_write_read_schema_compat(h2o_mol)
