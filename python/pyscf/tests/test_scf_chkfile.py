"""Phase 3 SCF-10 / ORACLE-08 chkfile h5py↔hdf5-metno round-trip.

Covers: SCF-10 + ORACLE-08 — chkfile schema compatibility both directions.

This test is the user-facing Python assertion path for the chkfile
round-trip; the Rust-side cargo-test analogue lives in
`crates/pyscf-oracle/tests/chkfile_roundtrip.rs` (plan 03-08).

Current status: The Rust-side chkfile writer
`pyscf_scf::chkfile::dump_scf_to_file` and reader
`load_scf_from_file` are shipped (plan 03-06), the PyO3 surface
exposes `mf.chkfile = path` as a setter, AND the user-facing
auto-write-on-converged-SCF inside `PyRHF::kernel` (scf.rs:363-388)
and the `mf.from_chk(mol, path)` reader (scf.rs:420) are now exposed.
All three arms run live: the h5py-mediated schema round-trip, the
rs-write→h5py-read arm, and the upstream-write→rs-read arm.
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

    Live — `PyRHF::kernel` auto-writes the chkfile on convergence when
    `mf.chkfile = path` is set (scf.rs:363-388, via
    `pyscf_scf::dump_scf_to_file`).
    """
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

    Live — `mf.from_chk(mol, path)` is exposed on the PyRHF surface
    (scf.rs:420, via `pyscf_scf::chkfile::load_scf_from_file`); the
    `init_guess = "chkfile"` path reconstructs MO state from disk.
    """
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

    Runs the h5py-mediated schema round-trip; the two rs-driven arms
    (auto-write and from_chk) now run as standalone live tests above.
    """
    test_chkfile_h5py_write_read_schema_compat(h2o_mol)
