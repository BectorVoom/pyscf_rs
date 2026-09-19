"""20-18 Task 2 wrapper: run an UNMODIFIED examples/pbc script natively and report
what served it.

Usage (from any cwd; the overlay must be the imported pyscf):
    PYTHONPATH=$REPO/python $REPO/.venv/bin/python example_gate_wrapper.py \
        $REPO/examples/pbc/<script>.py out.json NAME [NAME ...]

The script is compiled from its own file and exec'd in a fresh ``__main__``-like
namespace, so its source, line numbers and tracebacks are the upstream file's
own; nothing in it is patched. A line tracer confined to the script's
module-level frame (it returns ``None`` for every other frame) snapshots the
type of every pyscf-typed global after each line, so objects that the script
later rebinds (``kmf``, ``mypt``) are still recorded. After the run (normal exit
or exception) it records:
  * the terminal state: ``ok`` or the exception type + the script line it left at;
  * every (global name, type) pair seen, with ``type.__module__``;
  * ``pyscf.pbc._unported.which_impl(name)`` for each NAME given on the command line;
  * every ``pyscf.pbc.*`` module loaded from outside the overlay (upstream Python);
  * every ``PbcUpstreamFallthroughWarning`` raised;
  * ``e_tot``/``e_corr``/``e_free``/``entropy``/``sigma``/``converged`` of every pyscf object the
    script bound, read after the run (added after the example-22 run, which therefore lacks it).
"""
import json
import os
import sys
import time
import traceback
import warnings

script, out_path, names = sys.argv[1], sys.argv[2], sys.argv[3:]
REPO = os.path.abspath(os.path.join(os.path.dirname(script), "..", ".."))
OVERLAY = os.path.join(REPO, "python", "pyscf")

import pyscf  # noqa: E402

assert os.path.abspath(pyscf.__file__) == os.path.join(OVERLAY, "__init__.py"), pyscf.__file__

seen = {}  # name -> set of "module.qualname"
objs = {}  # id -> (first global name, object): every pyscf object the script bound, in order


def _snapshot(g):
    for k, v in list(g.items()):
        t = type(v)
        mod = getattr(t, "__module__", "") or ""
        if mod.startswith("pyscf"):
            seen.setdefault(k, set()).add(f"{mod}.{t.__qualname__}")
            objs.setdefault(id(v), (k, v))


code = compile(open(script).read(), script, "exec")
g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}


def _tracer(frame, event, arg):
    if frame.f_code is not code:
        return None

    def _local(fr, ev, a):
        if ev in ("line", "return"):
            _snapshot(fr.f_globals)
        return _local

    return _local


sys.argv = [script]
t0 = time.time()
state = {"status": "ok"}
with warnings.catch_warnings(record=True) as wlog:
    warnings.simplefilter("always")
    sys.settrace(_tracer)
    try:
        exec(code, g)
    except BaseException as e:  # noqa: BLE001 — the terminal state is the measurement
        sys.settrace(None)
        tb = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == script]
        state = {
            "status": "exception",
            "type": type(e).__name__,
            "message": str(e)[:500],
            "script_line": tb[-1].lineno if tb else None,
            "script_source": tb[-1].line if tb else None,
        }
    finally:
        sys.settrace(None)
_snapshot(g)
wall = time.time() - t0

from pyscf.pbc import _unported  # noqa: E402

impl = {}
for n in names:
    try:
        impl[n] = _unported.which_impl(n)
    except Exception as e:  # noqa: BLE001
        impl[n] = f"<{type(e).__name__}: {e}>"

upstream_pbc = sorted(
    m
    for m, mod in list(sys.modules.items())
    if m.startswith("pyscf.pbc")
    and getattr(mod, "__file__", None)
    and not os.path.abspath(mod.__file__).startswith(OVERLAY)
)
fallthrough_warnings = [
    str(w.message)[:200] for w in wlog if w.category.__name__ == "PbcUpstreamFallthroughWarning"
]

final_values = {}  # "<index>:<name>:<type>.<attr>" for every bound object, read after the run
for i, (k, v) in enumerate(objs.values()):
    k = f"{i}:{k}:{type(v).__qualname__}"
    for attr in ("e_tot", "e_corr", "e_free", "entropy", "sigma", "converged"):
        try:
            x = getattr(v, attr)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(x, (bool, int, float)) or hasattr(x, "__float__"):
            final_values[f"{k}.{attr}"] = x if isinstance(x, bool) else float(x)

report = {
    "final_values": final_values,
    "script": os.path.relpath(script, REPO),
    "pyscf_file": pyscf.__file__,
    "wall_s": round(wall, 1),
    "terminal_state": state,
    "types_seen": {k: sorted(v) for k, v in sorted(seen.items())},
    "which_impl": impl,
    "upstream_pbc_modules_loaded": upstream_pbc,
    "pbc_fallthrough_warnings": fallthrough_warnings,
}
json.dump(report, open(out_path, "w"), indent=1)
print("__REPORT__ " + json.dumps(report), flush=True)
