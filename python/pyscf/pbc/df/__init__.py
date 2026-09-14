"""pyscf.pbc.df overlay — re-exports the native periodic DF builders (plan 20-10).

``FFTDF``, ``AFTDF``, ``GDF``, ``MDF``, ``RSDF`` are the SAME objects as
``pyscf._native.pbc.df.*``; ``PWDF is AFTDF``, ``DF is GDF`` and
``RSGDF is RSDF`` hold, as upstream's aliases do (``pbc/df/__init__.py:21,31``).
All five share the native base class ``PeriodicDf``, which carries ``build``,
``get_nuc``, ``get_pp``, ``get_hcore``, ``get_jk``, ``get_naoaux``,
``sr_loop``, ``get_eri``, ``ao2mo``, ``ao2mo_7d`` and the ``_cderi`` /
``_cderi_to_save`` persistence attributes.

Upstream names not ported here (``incore``, ``outcore``, ``fft``, ``aft``,
``df``, ``mdf``, ``aug_etb``, ``make_auxcell``, ...) fall through lazily to the
upstream submodules; whether that is announced is plan 20-17's decision.
"""

import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
del _pkgutil

from pyscf._native.pbc.df import (  # type: ignore[attr-defined]  # noqa: E402
    AFTDF,
    DF,
    FFTDF,
    GDF,
    MDF,
    PWDF,
    RSDF,
    RSGDF,
    PeriodicDf,
    density_fit,
)

__all__ = ["FFTDF", "AFTDF", "PWDF", "GDF", "DF", "MDF", "RSDF", "RSGDF", "PeriodicDf",
           "density_fit"]

_UPSTREAM_SUBMODULES = ("incore", "outcore", "fft", "aft", "df", "mdf", "rsdf", "fft_jk",
                        "aft_jk", "df_jk", "mdf_jk", "rsdf_jk", "ft_ao", "gdf_builder",
                        "rsdf_builder", "df_ao2mo", "aft_ao2mo", "fft_ao2mo", "mdf_ao2mo")


def __getattr__(name):
    import importlib

    if name in _UPSTREAM_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "pwdf":
        return importlib.import_module(f"{__name__}.aft")
    if name == "aug_etb":
        return importlib.import_module("pyscf.df.addons").aug_etb
    if name == "make_auxcell":
        return importlib.import_module(f"{__name__}.incore").make_auxcell
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
