"""Silent fallthrough from the MOLECULAR overlay packages to upstream PySCF (plan 20-19 A).

The molecular overlay packages (``pyscf.gto``, ``pyscf.scf``, ``pyscf.dft``,
``pyscf.mp``, ``pyscf.cc``, ``pyscf.grad``, ``pyscf.geomopt``) re-export a few
``pyscf._native`` classes. Upstream PySCF Python — including every upstream
``pyscf.pbc`` module the PBC overlay passes through to (D-PBC-35) — imports far
more from them (``pyscf.gto.basis``, ``pyscf.gto.moleintor``, ``gto.ATOM_OF``,
``pyscf.dft.radi``, ``pyscf.scf.addons``, ...). This module supplies the two
halves of the fallthrough:

* each package sets ``__path__ = pkgutil.extend_path(__path__, __name__)``, so
  upstream SUBMODULES (``pyscf.gto.basis``) import from the next ``pyscf`` on
  ``sys.path``;
* ``package_getattr(globals())`` returns a PEP 562 ``__getattr__`` resolving a
  missing TOP-LEVEL name the way upstream's ``__init__.py`` would bind it, WITHOUT
  executing that ``__init__`` when it can avoid it: the upstream ``__init__`` is
  parsed (``ast``, never executed) into "name -> (module, attribute)" for its
  ``from X import a as b`` lines, star-import sources for ``from X import *``,
  and the set of names its body defines (``def``/``class``/assignment). Lookup
  order for a missing ``name``:

  1. the package's native module(s) (``pyscf._native.<pkg>``), for native names
     the overlay's ``__all__`` does not list (``pyscf.dft.NumInt``);
  2. an explicit ``from X import name`` of the upstream ``__init__``;
  3. an upstream submodule file ``<name>.py`` / ``<name>/__init__.py``;
  4. the upstream ``__init__``'s star-import sources, in order;
  5. only for names the upstream ``__init__`` body itself defines (``scf.HF``,
     ``scf.rhf``, ``dft.XC``, ``grad.grad_nuc``): the upstream ``__init__`` is
     executed ONCE under the private non-package name ``<pkg>._upstream_init``
     and the name read from it.

  Names the overlay already binds are module globals, so ``__getattr__`` never
  sees them: ``pyscf.scf.RHF is pyscf._native.scf.RHF`` is untouched.

* ``module_getattr(__name__)`` is the same fallthrough for an overlay SHIM
  module that shadows an upstream module file (``pyscf/scf/hf.py``): missing
  names come from the upstream file executed as ``<pkg>._upstream_<leaf>``.

Unlike ``pyscf.pbc._unported`` this fallthrough is SILENT: the molecular
packages are not PBC families, and D-PBC-35's one-time warning stays scoped to
``pyscf.pbc.*``.

KNOWN SEMANTIC MISMATCH (recorded, not fixed): upstream code that reaches a name
the overlay binds natively gets the native object — ``class ROHF(hf.RHF)`` in
upstream ``scf/rohf.py`` subclasses ``pyscf._native.scf.RHF``;
``isinstance(x, gto.Mole)`` tests against ``pyscf._native.gto.Mole``;
``gto.M(...)`` inside upstream code builds a native ``Mole``. See
``.planning/phases/20-pbc-python-bindings/20-19-AC-SUMMARY.md``.
"""

import ast
import importlib
import importlib.util
import os
import sys

__all__ = ["package_getattr", "module_getattr", "load_upstream_file"]

OVERLAY_ROOT = os.path.dirname(os.path.abspath(__file__))

_RESOLVING: set = set()


def _is_overlay(path):
    path = os.path.abspath(path)
    return path == OVERLAY_ROOT or path.startswith(OVERLAY_ROOT + os.sep)


def _upstream_dirs(package_path):
    return [p for p in package_path if not _is_overlay(p)]


def _find_upstream_file(package_path, leaf):
    for entry in _upstream_dirs(package_path):
        for cand in (os.path.join(entry, leaf + ".py"), os.path.join(entry, leaf, "__init__.py")):
            if os.path.isfile(cand):
                return cand
    return None


def load_upstream_file(private_name, path):
    """Execute upstream ``path`` as the module ``private_name`` (cached in ``sys.modules``)."""
    mod = sys.modules.get(private_name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(private_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(private_name, None)
        raise
    return mod


class _InitIndex:
    """What an upstream package ``__init__.py`` binds, read with ``ast`` (not executed)."""

    def __init__(self, pkg, path):
        self.path = path
        self.named = {}      # alias -> (module fullname, attribute or None for a submodule)
        self.stars = []      # module fullnames, in source order
        self.defined = set()  # names bound by def / class / assignment in the body
        if path is None:
            return
        with open(path, "rb") as fh:
            tree = ast.parse(fh.read(), filename=path)
        self._walk(pkg, tree.body)

    def _walk(self, pkg, body):
        for node in body:
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                    module = f"{base}.{node.module}" if node.module else base
                else:
                    module = node.module
                for alias in node.names:
                    if alias.name == "*":
                        self.stars.append(module)
                    elif module == pkg:
                        self.named[alias.asname or alias.name] = (f"{pkg}.{alias.name}", None)
                    else:
                        self.named[alias.asname or alias.name] = (module, alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            self.defined.add(sub.id)
            elif isinstance(node, ast.Try):
                for part in (node.body, node.orelse, node.finalbody,
                             *[h.body for h in node.handlers]):
                    self._walk(pkg, part)
            elif isinstance(node, (ast.If, ast.With)):
                self._walk(pkg, node.body)
                self._walk(pkg, getattr(node, "orelse", []))


def _star_has(module, name):
    public = getattr(module, "__all__", None)
    if public is not None:
        return name in public
    return not name.startswith("_") and hasattr(module, name)


def package_getattr(namespace, native_modules=()):
    """A PEP 562 ``__getattr__`` for the overlay package whose globals are ``namespace``.

    ``native_modules`` are dotted names of ``pyscf._native`` modules consulted first.
    """
    pkg = namespace["__name__"]
    state = {}

    def index():
        idx = state.get("index")
        if idx is None:
            init = _find_upstream_file(namespace.get("__path__", ()), "__init__")
            idx = state["index"] = _InitIndex(pkg, init)
        return idx

    def resolve(name):
        for native in native_modules:
            mod = importlib.import_module(native)
            if hasattr(mod, name):
                return getattr(mod, name)
        idx = index()
        target = idx.named.get(name)
        if target is not None:
            module, attr = target
            if attr is None:
                return importlib.import_module(module)
            value = getattr(importlib.import_module(module), attr)
            # Bind it, as upstream's `from X import name` does: importing X may itself
            # have bound a same-named SUBMODULE here (`gto.eval_gto` is upstream's
            # function, not the `pyscf.gto.eval_gto` module).
            namespace[name] = value
            return value
        if _find_upstream_file(namespace.get("__path__", ()), name) is not None:
            return importlib.import_module(f"{pkg}.{name}")
        for module in idx.stars:
            mod = importlib.import_module(module)
            if _star_has(mod, name):
                return getattr(mod, name)
        if name in idx.defined and idx.path is not None:
            mod = load_upstream_file(f"{pkg}._upstream_init", idx.path)
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"module {pkg!r} has no attribute {name!r}")

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {pkg!r} has no attribute {name!r}")
        key = (pkg, name)
        if key in _RESOLVING:  # re-entered while importing the source of `name`
            raise AttributeError(f"module {pkg!r} has no attribute {name!r} (import cycle)")
        _RESOLVING.add(key)
        try:
            value = resolve(name)
        except AttributeError:
            raise
        except Exception as exc:  # noqa: BLE001 — keep hasattr() safe (20-17 convention)
            raise AttributeError(
                f"module {pkg!r} has no native attribute {name!r}, and the upstream "
                f"fallthrough failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            _RESOLVING.discard(key)
        return value

    return __getattr__


def module_getattr(module_name):
    """A PEP 562 ``__getattr__`` for an overlay shim module shadowing an upstream file."""
    parent_name, _, leaf = module_name.rpartition(".")

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        key = (module_name, name)
        if key in _RESOLVING:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r} (import cycle)")
        _RESOLVING.add(key)
        try:
            parent = importlib.import_module(parent_name)
            path = _find_upstream_file(getattr(parent, "__path__", ()), leaf)
            if path is None:
                raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
            upstream = load_upstream_file(f"{parent_name}._upstream_{leaf}", path)
        except AttributeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AttributeError(
                f"module {module_name!r} has no native attribute {name!r}, and the upstream "
                f"fallthrough failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            _RESOLVING.discard(key)
        try:
            return getattr(upstream, name)
        except AttributeError:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from None

    return __getattr__
