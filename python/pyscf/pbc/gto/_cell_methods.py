"""Method-style constructors on the native ``Cell`` (plan 20-18).

Port of upstream ``Cell.__getattr__`` (``pyscf/pbc/gto/cell.py:1407-1511``,
PySCF 2.12.1): ``cell.KRKS(xc='pbe', kpts=...)``, ``cell.KRHF()``,
``cell.RKS(...)``, ``cell.KMP2()``, ... The native ``Cell.__getattr__``
(``crates/pyscf-py/src/pbc/gto.rs``) calls :func:`cell_method` for every public
name the class does not bind; ``NotImplemented`` means "not a method" and the
native ``AttributeError`` is raised.

The SCF objects are built by the overlay ``pyscf.pbc.dft`` / ``pyscf.pbc.scf``
names, so ``type(cell.KRKS(...)) is pyscf.pbc.dft.KRKS`` (the native class).

Deviations from upstream, each forced by the overlay:

* upstream first runs ``from pyscf.pbc import __all__`` to register post-SCF
  methods on the SCF classes; under the overlay most of the families that
  file imports fail to import (20-17 measurement), and the native classes do
  not take registered methods, so the import is skipped. Post-SCF names
  (``cell.KMP2()``) therefore resolve as upstream does, and fail at call time
  with the ``AttributeError`` of ``getattr(mf, 'MP2')`` if the native SCF
  class has no such method;
* ``pyscf.dft.XC`` is imported only on the TD branch, which is where upstream
  uses it; if the molecular overlay lacks it the lookup raises ``AttributeError``;
* an ``ImportError`` raised by an overlay module's lazy upstream fallthrough
  while probing ``getattr(mod, key, None)`` counts as "no such callable"
  (upstream's module ``__getattr__`` never imports).
"""

SCF_KW = {'kpt', 'kpts', 'xc', 'exxdiv',
          'U_idx', 'U_val', 'C_ao_lo', 'minao_ref'}  # cell.py:1472-1473


class _MoleLazyCallAdapter:
    '''Port of ``pyscf/gto/mole.py:4368-4383`` (adapter for API updates).'''

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getattr__(self, key):
        import warnings
        warnings.warn(
            f'The API mol.{self.name}.{key} is deprecated and will be '
            f'removed in a future release. Please use mol.{self.name}().{key} instead.',
            _deprecation_warning(), stacklevel=1)
        out = self.fn()
        return getattr(out, key)


def _deprecation_warning():
    '''``pyscf.lib.exceptions.DeprecationWarning`` (a ``UserWarning``, exceptions.py:30).'''
    try:
        from pyscf.lib.exceptions import DeprecationWarning as warning_cls
    except ImportError:
        warning_cls = UserWarning
    return warning_cls


def _probe(mod, key):
    try:
        return getattr(mod, key, None)
    except ImportError:
        return None


def _set(mf, **kwargs):
    '''``lib.StreamObject.set`` (``pyscf/lib/misc.py:642-661``) for native classes.'''
    setter = getattr(type(mf), 'set', None)
    if setter is not None:
        return mf.set(**kwargs)
    for k, v in kwargs.items():
        setattr(mf, k, v)
    return mf


def _xc_table():
    try:
        from pyscf.dft import XC
    except ImportError as exc:
        raise AttributeError(f'TD method lookup needs pyscf.dft.XC: {exc}') from exc
    return XC


def cell_method(cell, key):
    '''The value of upstream ``Cell.__getattr__(key)``, or ``NotImplemented``
    where upstream falls back to ``object.__getattribute__`` (cell.py:1449,1469).'''
    if not key or key[0] == '_':  # cell.py:1411-1416
        return NotImplemented

    from pyscf.pbc import dft, scf  # cell.py:1421

    attr_name = key
    mf_xc = None
    for mod in (dft, scf):  # cell.py:1426-1430
        mf_method = _probe(mod, key)
        if callable(mf_method):
            key = None
            break
    else:
        if key[0] == 'K':  # with k-point sampling, cell.py:1432-1451
            if 'TD' in key[:4]:
                if 'KTDA' in key:
                    mf_method = 'KSCF_TO_BE_DETERMINED'
                elif 'KTDHF' in key:
                    mf_method = scf.KHF
                else:
                    mf_method = dft.KKS
                    xc = key.split('TD', 1)[1]
                    if xc in _xc_table():
                        mf_xc = xc
                        key = 'KTDDFT'
                    elif 'TDDFT' not in key:
                        raise AttributeError(f'method {key} not supported')
            elif 'CI' in key or 'CC' in key or 'MP' in key:
                mf_method = scf.KHF
            else:
                return NotImplemented
            # Remove prefix 'K' because methods are registered without the leading 'K'
            key = key[1:]
        else:  # cell.py:1452-1469
            if 'TD' in key[:3]:
                if 'TDA' in key:
                    mf_method = 'SCF_TO_BE_DETERMINED'
                elif 'TDHF' in key:
                    mf_method = scf.HF
                else:
                    mf_method = dft.KS
                    xc = key.split('TD', 1)[1]
                    if xc in _xc_table():
                        mf_xc = xc
                        key = 'TDDFT'
                    elif 'TDDFT' not in key:
                        raise AttributeError(f'method {key} not supported')
            elif 'CI' in key or 'CC' in key or 'MP' in key:
                mf_method = scf.HF
            else:
                return NotImplemented

    post_mf_key = key

    def fn(*args, **kwargs):  # cell.py:1475-1510
        if mf_xc is not None:
            assert 'xc' not in kwargs
            kwargs['xc'] = mf_xc

        mf_kw = {}
        remaining_kw = {}
        for k, v in kwargs.items():
            if k in SCF_KW:
                mf_kw[k] = v
            else:
                remaining_kw[k] = v

        if mf_method == 'SCF_TO_BE_DETERMINED':
            if 'xc' in mf_kw:
                mf = dft.KS(cell, **mf_kw)
            else:
                mf = scf.HF(cell, **mf_kw)
        elif mf_method == 'KSCF_TO_BE_DETERMINED':
            if 'xc' in mf_kw:
                mf = dft.KKS(cell, **mf_kw)
            else:
                mf = scf.KHF(cell, **mf_kw)
        else:
            mf = mf_method(cell, **mf_kw)

        if post_mf_key is None:
            if args:
                raise AttributeError(
                    f'cell.{attr_name} function does not support positional arguments')
            return _set(mf, **remaining_kw)

        post_mf = getattr(mf, post_mf_key)
        if cell.nelectron != 0:
            mf.run()
        return post_mf(*args, **remaining_kw)
    return _MoleLazyCallAdapter(fn, attr_name)
