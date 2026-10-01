"""Generate the Kaggle notebook for one session of a resumable pyscf-rs run.

The notebook is self-contained: the run's environment, the expected runner
SHA-256 and the session number are baked in, so a stale dataset mount or a
checkpoint from the wrong session fails the first cell instead of wasting a
GPU session.
"""
import json

GPU_GUARDS = {
    # accelerator: (substring of the nvidia-smi name, minimum MiB, Kaggle machine_shape)
    "t4": ("T4", 14000, "NvidiaTeslaT4"),
    "p100": ("P100", 15000, "NvidiaTeslaP100"),
    "rtxpro6000": ("RTX PRO 6000", 90000, "NvidiaRtxPro6000"),
}

SETUP = r'''import glob, hashlib, json, os, shutil, subprocess, time
T_START = time.time()
WORK = "/kaggle/working"
SESSION = @@SESSION@@
EXPECT_SHA = "@@SHA@@"
ACCEL = "@@ACCEL@@"
GUARD = @@GUARD@@
RUN_ENV = @@ENV@@
SESSION_HOURS = @@HOURS@@

def sh(cmd, env=None):
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, errors="replace")
    print(p.stdout + p.stderr, flush=True)

sh("nvidia-smi; nproc; free -g; df -h /kaggle/working | tail -1")
env = dict(os.environ)
if ACCEL != "cpu":
    # Kaggle can silently hand out a different GPU than requested: refuse.
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    name, mem = [x.strip() for x in gpu.split(",")]
    print("GPU:", name, mem, "MiB", flush=True)
    assert GUARD[0] in name and int(mem) >= GUARD[1], f"wrong machine for {ACCEL}: {gpu}"
    libdirs = sorted(set(glob.glob("/usr/local/cuda*/lib64") + glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib")
                         + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvrtc/lib")))
    libdirs = [d for d in libdirs if glob.glob(d + "/libnvrtc.so*")]
    if not libdirs:
        sh("pip install -q nvidia-cuda-nvrtc-cu12")
        libdirs = glob.glob("/usr/local/lib/python3*/*-packages/nvidia/cuda_nvrtc/lib")
    assert libdirs, "no libnvrtc on this image"
    for d in libdirs:
        so = sorted(glob.glob(d + "/libnvrtc.so.*"))
        if so and not os.path.exists(d + "/libnvrtc.so"):
            try: os.symlink(so[0], d + "/libnvrtc.so")
            except OSError: pass
    env["LD_LIBRARY_PATH"] = ":".join(libdirs + [env.get("LD_LIBRARY_PATH", "")])

# The runner: Kaggle unpacks archives in datasets, so accept both forms.
R = "/tmp/runner"
shutil.rmtree(R, ignore_errors=True); os.makedirs(R)
tars = glob.glob("/kaggle/input/**/runner.tar.gz", recursive=True)
if tars:
    sh(f"tar -xzf {tars[0]} -C {R}")
else:
    man = glob.glob("/kaggle/input/**/RUNNER_MANIFEST.txt", recursive=True)
    assert man, "runner dataset not mounted"
    shutil.copytree(os.path.dirname(man[0]), R, dirs_exist_ok=True)
BIN = f"{R}/yta7o19_bands"; os.chmod(BIN, 0o755)
sha = hashlib.sha256(open(BIN, "rb").read()).hexdigest()
print("runner sha256", sha, flush=True)
assert sha == EXPECT_SHA, f"stale runner mounted ({sha}); expected {EXPECT_SHA}"

# The checkpoint: what the previous session left, verified file by file.
CK = f"{WORK}/ckpt"; os.makedirs(CK, exist_ok=True)
mans = glob.glob("/kaggle/input/**/ckpt_manifest.json", recursive=True)
if SESSION > 1 or mans:
    assert mans, f"session {SESSION} needs the previous checkpoint dataset"
    src = os.path.dirname(mans[0]); man = json.load(open(mans[0]))
    assert man["session"] == SESSION - 1, f"checkpoint is from session {man['session']}, this is session {SESSION}"
    for f, h in man["sha256"].items():
        shutil.copy(f"{src}/{f}", f"{CK}/{f}")
        assert hashlib.sha256(open(f"{CK}/{f}", "rb").read()).hexdigest() == h, f"corrupt checkpoint file {f}"
    print("checkpoint from session", man["session"], "->", sorted(man["sha256"]), flush=True)
else:
    print("session 1: starting from scratch", flush=True)

env.update(RUN_ENV)
env.update(YTA_CKPT_DIR=CK, PYSCF_BASIS_PATH=f"{R}/pyscf/gto/basis", RUST_BACKTRACE="1")
CC = f"/tmp/cubecl_cache_{sha[:16]}"; os.makedirs(CC, exist_ok=True)
open(f"{CC}/cubecl.toml", "w").write('[compilation]\ncache = "local"\n')
'''

RUN = r'''def progress():
    try:
        st = json.load(open(f"{CK}/status.json"))
        return f"{st['stage']} {json.dumps(st['detail'])[:160]}"
    except Exception:
        return "(no status yet)"

cap = SESSION_HOURS * 3600 - (time.time() - T_START) - 15 * 60   # 15 min left to save the output
t0 = time.time(); last = None; rc = None
with open(f"{WORK}/run.log", "a") as log:
    p = subprocess.Popen([BIN], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=CC)
    while time.time() - t0 < cap:
        pid, status, _ = os.wait4(p.pid, os.WNOHANG)
        if pid:
            rc = os.waitstatus_to_exitcode(status); break
        time.sleep(60)
        cur = progress()
        if cur != last:
            g = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1",
                               shell=True, capture_output=True, text=True).stdout.strip()
            print(f"[{(time.time()-t0)/3600:5.2f} h] gpu {g} | {cur}", flush=True); last = cur
    if rc is None:
        p.terminate()
        try: p.wait(120)
        except Exception: p.kill(); p.wait()
        rc = "time-capped (resumes next session)"
print(f"== exit: {rc}   wall {(time.time()-t0)/3600:.2f} h", flush=True)
sh(f"grep -E 'panicked|ERROR' {WORK}/run.log | tail -5 | cut -c1-300")
'''

FINISH = r'''# Hand the checkpoint to the next session.
for f in os.listdir(CK):
    if f.endswith(".tmp"): os.remove(f"{CK}/{f}")
files = {f: hashlib.sha256(open(f"{CK}/{f}", "rb").read()).hexdigest()
         for f in sorted(os.listdir(CK)) if f != "ckpt_manifest.json"}
json.dump({"session": SESSION, "exit": str(rc), "runner_sha256": sha, "sha256": files},
          open(f"{CK}/ckpt_manifest.json", "w"), indent=1)

# Human-readable progress.
print(open(f"{CK}/status.json").read() if os.path.exists(f"{CK}/status.json") else "no status.json", flush=True)
if os.path.exists(f"{CK}/history.jsonl"):
    rows = []
    for l in open(f"{CK}/history.jsonl", errors="replace"):
        try: rows.append(json.loads(l))
        except ValueError: pass   # torn last line from a kill mid-append
    print(f"{'stage':>6} {'cycle':>5} {'e_tot (Ha)':>20}")
    for r in rows[-25:]:
        print(f"{r['stage']:>6} {r['cycle']:>5} {r['e_tot']:>20.10f}")
try:
    res = json.load(open(f"{CK}/result.json"))
    if res.get("summary"):
        print("SUMMARY", json.dumps(res["summary"]))
        import numpy as np
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        b = res["bands"]; HA = 27.211386245988
        e = np.array(b["energies_ha"]) * HA; x = np.array(b["x"][:len(e)])
        nocc = res["method"]["nocc"]; e = e - e[:, nocc - 1].max()
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(e.shape[1]):
            ax.plot(x, e[:, i], color="C0" if i < nocc else "C1", lw=0.8)
        for t in b["tick_x"]: ax.axvline(t, color="0.7", lw=0.5)
        ax.set_xticks(b["tick_x"]); ax.set_xticklabels(b["tick_labels"]); ax.set_ylim(-6, 6)
        ax.set_ylabel("E - VBM (eV)"); ax.set_title(f"gap {res['summary']['gap_ev']:.3f} eV")
        fig.savefig(f"{WORK}/bands.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print("no result summary yet:", e)
'''


def build(session, sha, accelerator, env, session_hours, title):
    guard = GPU_GUARDS.get(accelerator, ("", 0, None))
    subst = {
        "@@SESSION@@": str(session), "@@SHA@@": sha, "@@ACCEL@@": accelerator,
        "@@GUARD@@": repr(guard[:2]), "@@ENV@@": repr({k: str(v) for k, v in env.items()}),
        "@@HOURS@@": repr(float(session_hours)),
    }
    def fill(s):
        for k, v in subst.items():
            s = s.replace(k, v)
        return s
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source":
            f"# {title} — session {session}\n\nResumable pyscf-rs run: every stage and every SCF cycle is "
            "checkpointed in `/kaggle/working/ckpt`, which the next session reads back. Generated by "
            "`tools/kaggle-t4/ktrun.py`; do not edit here."},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": fill(SETUP)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": fill(RUN)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": fill(FINISH)},
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
            "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def machine_shape(accelerator):
    return GPU_GUARDS.get(accelerator, (None, None, None))[2]
