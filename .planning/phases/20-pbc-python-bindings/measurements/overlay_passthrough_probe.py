#!/usr/bin/env python
"""20-17 measurement: which upstream ``pyscf.pbc.*`` modules import under the overlay.

For every module of the VENDORED 2.12.1 tree (``pyscf/pbc/**/*.py``, ``test/``
directories excluded) this runs ``importlib.import_module(name)`` in a FRESH
interpreter (so one failure cannot poison the next) and records ok/error, the
exception, and which tree the module's ``__file__`` resolved to.

Two sys.path configurations, both with the overlay ``python/`` FIRST:

* ``site``     — ``PYTHONPATH=<repo>/python``: what pytest sees (rootdir
  insertion). The passthrough target is ``.venv`` site-packages pyscf 2.14.0.
* ``vendored`` — ``PYTHONPATH=<repo>/python:<repo>``: the passthrough target
  is the vendored 2.12.1 tree (then 2.14.0).

The interpreter cwd is a scratch directory, never the repo root (from the repo
root, cwd ``''`` puts the vendored ``pyscf/__init__.py`` ahead of the overlay).

Usage: overlay_passthrough_probe.py <out.json> [site|vendored ...]
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")

CHILD = r"""
import importlib, json, sys, warnings
name = sys.argv[1]
out = {"name": name}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        m = importlib.import_module(name)
        out["status"] = "ok"
        out["file"] = getattr(m, "__file__", None)
    except BaseException as e:  # noqa: BLE001
        out["status"] = "error"
        msg = str(e).splitlines()[0] if str(e) else ""
        out["error"] = f"{type(e).__name__}: {msg}"[:200]
    out["pbc_warnings"] = sorted({str(x.message).split(":")[0] for x in w
                                  if "pyscf-rs" in str(x.message)})
import pyscf
out["pyscf_file"] = pyscf.__file__
print("__PROBE__" + json.dumps(out))
"""


def module_names():
    root = os.path.join(REPO, "pyscf")
    names = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "pbc")):
        dirnames[:] = sorted(d for d in dirnames if d not in ("test", "__pycache__"))
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), os.path.dirname(root))
            mod = rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            names.append(mod)
    return sorted(names)


def classify(path):
    if path is None:
        return "no-file (stub or native)"
    if path.startswith(os.path.join(REPO, "python") + os.sep):
        return "overlay"
    if "site-packages" in path:
        return "site-packages 2.14.0"
    if path.startswith(os.path.join(REPO, "pyscf") + os.sep):
        return "vendored 2.12.1"
    return path


def probe(name, config, cwd):
    pp = os.path.join(REPO, "python")
    if config == "vendored":
        pp = pp + os.pathsep + REPO
    env = dict(os.environ, PYTHONPATH=pp)
    env.pop("PYSCF_EXT_PATH", None)
    proc = subprocess.run([PY, "-c", CHILD, name], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("__PROBE__")), None)
    if line is None:
        tail = (proc.stderr.strip().splitlines() or ["<no output>"])[-1]
        return {"name": name, "status": "crash", "error": f"rc={proc.returncode}: {tail[:200]}"}
    out = json.loads(line[len("__PROBE__"):])
    out["resolved"] = classify(out.get("file")) if out["status"] == "ok" else None
    return out


def main():
    out_path = sys.argv[1]
    configs = sys.argv[2:] or ["site", "vendored"]
    names = module_names()
    result = {"modules": names, "configs": {}}
    with tempfile.TemporaryDirectory() as cwd:
        for config in configs:
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                rows = list(ex.map(lambda n: probe(n, config, cwd), names))
            result["configs"][config] = rows
            ok = sum(r["status"] == "ok" for r in rows)
            print(f"{config}: {ok}/{len(rows)} import ok", flush=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
