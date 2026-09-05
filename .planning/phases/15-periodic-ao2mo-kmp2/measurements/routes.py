import statistics
import time

from pyscf import __version__, pbc

assert __version__ == "2.12.1", __version__


def helium():
    return pbc.gto.Cell(
        atom="He 0 0 0",
        basis="6-31g",
        a=[[0, 2.834589, 2.834589], [2.834589, 0, 2.834589], [2.834589, 2.834589, 0]],
        unit="B",
        mesh=[9, 9, 9],
        verbose=0,
    ).build()


cell = helium()
kpts = cell.make_kpts([1, 1, 2])
for name, builder in (
    ("FFTDF", pbc.df.FFTDF(cell, kpts)),
    ("GDF", pbc.df.GDF(cell, kpts)),
):
    mf = pbc.scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    mf.with_df = builder
    mf.conv_tol = 1e-11
    mf.kernel()
    samples = []
    values = []
    for _ in range(3):
        mp = pbc.mp.KMP2(mf)
        start = time.perf_counter()
        e, _ = mp.kernel(with_t2=False)
        samples.append(time.perf_counter() - start)
        values.append(float(e))
    print(
        name,
        f"e_corr={values[-1]:.17g}",
        f"ss={mp.e_corr_ss:.17g}",
        f"os={mp.e_corr_os:.17g}",
        f"median_s={statistics.median(samples):.9g}",
        f"spread_s={max(samples)-min(samples):.9g}",
        f"run_spread={max(values)-min(values):.17g}",
    )
    if name == "GDF":
        direct = pbc.mp.KMP2(mf)
        direct.with_df_ints = False
        direct_e, _ = direct.kernel(with_t2=False)
        print(f"GDF_forced_ao2mo e_corr={direct_e:.17g}")
