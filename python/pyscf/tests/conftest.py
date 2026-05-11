"""Shared pytest fixtures for pyscf-rs Phase 3 SCF tests.

Phase 3 plan 03-02 ships these as skip-stubs; plan 03-10 fills in real
Mole-construction bodies using pyscf-rs's Mole API + cintx integrals.
"""
import pytest


@pytest.fixture
def h2o_mol():
    """H2O / cc-pVDZ — primary SCF test fixture (SCF-01 corpus)."""
    pytest.skip("conftest fixture body pending — plan 03-10")


@pytest.fixture
def benzene_mol():
    """Benzene / 6-31G* — SCF-01 secondary test corpus entry."""
    pytest.skip("conftest fixture body pending — plan 03-10")


@pytest.fixture
def water_trimer_mol():
    """Water trimer / cc-pVDZ — chkfile round-trip fixture (ORACLE-08)."""
    pytest.skip("conftest fixture body pending — plan 03-10")
