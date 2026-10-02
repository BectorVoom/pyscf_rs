#!/usr/bin/env python3
"""Drive a resumable pyscf-rs band-structure run through Kaggle GPU sessions.

    ktrun.py publish-runner  CONFIG            upload the runner binary (once per build)
    ktrun.py seed-ckpt       CONFIG DIR        (optional) start from files you already have
    ktrun.py run             CONFIG [-n N]     push sessions until the run is done (or N sessions)
    ktrun.py status          CONFIG            print where the run is

A session runs for at most `session_hours`; the binary checkpoints every stage
and every SCF cycle, and `run` hands each session's checkpoint to the next one
as a new private dataset (`<user>/<run_name>-ckpt-s<N>`). A fresh dataset slug
per session and per runner build sidesteps Kaggle mounting a stale dataset
version; the notebook still refuses a runner or checkpoint that is not the
one it was generated for.

`run` can be interrupted at any time (Ctrl-C, closed laptop): rerun it and it
picks up the session that is still running, or pulls the one that finished.
Only Python 3 and the `kaggle` CLI are needed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import notebook as nbgen

POLL_SECONDS = int(os.environ.get("KTRUN_POLL_SECONDS", "300"))


# ----------------------------------------------------------------- config / state

def load_config(path):
    cfg = json.load(open(path))
    for key in ("run_name", "kaggle", "runner", "env"):
        if key not in cfg:
            sys.exit(f"config {path}: missing '{key}'")
    name = cfg["run_name"]
    if not (6 <= len(name) <= 32) or not all(c.isalnum() or c == "-" for c in name):
        sys.exit("run_name: 6-32 characters, letters, digits and '-' only (it becomes dataset slugs)")
    cfg.setdefault("session_hours", 11.5)
    cfg["kaggle"].setdefault("accelerator", "t4")
    cfg["local_dir"] = str(Path(os.path.expanduser(cfg.get("local_dir", "~/ktrun"))) / name)
    return cfg


def state_path(cfg):
    return Path(cfg["local_dir"]) / "state.json"


def load_state(cfg):
    p = state_path(cfg)
    return json.load(open(p)) if p.exists() else {"session": 0}


def save_state(cfg, st):
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    json.dump(st, open(tmp, "w"), indent=1)
    tmp.replace(p)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------- kaggle CLI

def kaggle(cfg, *args, check=True, timeout=1800):
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = open(os.path.expanduser(cfg["kaggle"]["token_file"])).read().strip()
    p = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def check_account(cfg):
    out = kaggle(cfg, "config", "view")
    user = next((l.split()[-1] for l in out.splitlines() if "username" in l), None)
    if user != cfg["kaggle"]["user"]:
        sys.exit(f"token file belongs to {user!r}, config says {cfg['kaggle']['user']!r}")


def dataset_status(cfg, slug):
    return kaggle(cfg, "datasets", "status", slug, check=False, timeout=120).splitlines()[-1].strip()


def remote_manifest(cfg, slug):
    """The `ckpt_manifest.json` of an existing dataset, or None."""
    tmp = Path(cfg["local_dir"]) / "remote_check"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    kaggle(cfg, "datasets", "download", slug, "-f", "ckpt_manifest.json", "-p", str(tmp), check=False, timeout=600)
    for z in tmp.glob("*.zip"):
        import zipfile
        zipfile.ZipFile(z).extractall(tmp)
    p = tmp / "ckpt_manifest.json"
    try:
        return json.load(open(p))
    except Exception:
        return None


def create_dataset(cfg, folder, slug, title):
    """Create `slug` from `folder`, idempotently: slugs are unique per run and
    session/build, so a dataset that already exists is normally this same
    upload whose response was lost (a killed driver, a dropped connection).
    Whatever path ends in a ready dataset — fresh upload, adopted, or
    reconciled after a CLI error — a checkpoint dataset is accepted only if
    its manifest matches what we meant to upload. A runner dataset's slug
    carries the binary's SHA-256 prefix and the notebook checks the full hash."""
    def verify():
        local = Path(folder) / "ckpt_manifest.json"
        if local.exists():
            want = json.load(open(local)).get("sha256")
            got = (remote_manifest(cfg, slug) or {}).get("sha256")
            if got != want:
                raise RuntimeError(f"dataset {slug} exists with DIFFERENT content; delete it on kaggle.com "
                                   "(or use a new run_name) before continuing")

    if dataset_status(cfg, slug) == "ready":
        verify()
        log(f"dataset {slug} already exists with the same content; using it")
        return
    json.dump({"title": title, "id": slug, "licenses": [{"name": "other"}]},
              open(Path(folder) / "dataset-metadata.json", "w"))
    log(f"uploading dataset {slug} ...")
    try:
        kaggle(cfg, "datasets", "create", "-p", str(folder), timeout=7200)
    except RuntimeError:
        # The upload may have landed even though the CLI reported a failure.
        if dataset_status(cfg, slug) not in ("ready", "pending"):
            raise
    for _ in range(90):
        s = dataset_status(cfg, slug)
        if s == "ready":
            verify()
            log(f"dataset {slug} ready")
            return
        time.sleep(20)
    raise RuntimeError(f"dataset {slug} not ready after 30 min (last status: {s})")


def kernel_status(cfg):
    out = kaggle(cfg, "kernels", "status", f"{cfg['kaggle']['user']}/{cfg['run_name']}", check=False, timeout=120)
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out:
            return word
    low = out.lower()
    if "404" in low or "not found" in low or "permission" in low and "denied" in low:
        return "MISSING"      # no such notebook (yet): Kaggle never received it
    return "UNKNOWN"          # transient: network, rate limit, ...


def read_history(path):
    """history.jsonl rows; a torn last line (killed mid-append) is skipped."""
    rows = []
    if path.exists():
        for line in open(path, errors="replace"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- commands

def cmd_publish_runner(cfg, args):
    check_account(cfg)
    binary = Path(os.path.expanduser(cfg["runner"]["binary"]))
    data = Path(os.path.expanduser(cfg["runner"]["pyscf_data_dir"]))
    if not binary.exists() or not (data / "gto" / "basis").is_dir():
        sys.exit(f"runner.binary ({binary}) or runner.pyscf_data_dir/gto/basis ({data}) missing")
    sha = sha256(binary)
    slug = f"{cfg['kaggle']['user']}/{cfg['run_name']}-runner-{sha[:8]}"
    st = load_state(cfg)
    if st.get("runner_sha256") == sha:
        log(f"runner {sha[:12]} already published as {st['runner_dataset']}")
        return
    stage = Path(cfg["local_dir"]) / "runner_upload"
    shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
    # Only the data the binary reads (basis sets, GTH pseudopotentials) —
    # not the whole PySCF package: Kaggle unpacks dataset archives and
    # silently drops datasets whose archive it cannot process (thousands of
    # files, `name:Zone.Identifier` entries).
    def keep(info):
        name = info.name.rsplit("/", 1)[-1]
        if ":" in name or name == "__pycache__" or name.endswith((".pyc", ".py")):
            return None
        return info
    with tarfile.open(stage / "runner.tar.gz", "w:gz") as tar:
        tar.add(binary, arcname="yta7o19_bands")
        for sub in ("gto/basis", "pbc/gto/basis", "pbc/gto/pseudo"):
            if (data / sub).is_dir():
                tar.add(data / sub, arcname=f"pyscf/{sub}", filter=keep)
        manifest = f"binary sha256 {sha}\nbuilt {time.ctime(binary.stat().st_mtime)}\n"
        info = tarfile.TarInfo("RUNNER_MANIFEST.txt"); info.size = len(manifest)
        import io
        tar.addfile(info, io.BytesIO(manifest.encode()))
    create_dataset(cfg, stage, slug, f"{cfg['run_name']} runner {sha[:8]}")
    st.update(runner_sha256=sha, runner_dataset=slug)
    save_state(cfg, st)


def publish_ckpt(cfg, folder, session):
    slug = f"{cfg['kaggle']['user']}/{cfg['run_name']}-ckpt-s{session}"
    create_dataset(cfg, folder, slug, f"{cfg['run_name']} checkpoint after session {session}")
    return slug


def cmd_seed_ckpt(cfg, args):
    """Publish existing checkpoint files (e.g. s1e.bin/h1e.bin) as 'session 0'."""
    check_account(cfg)
    st = load_state(cfg)
    if st["session"] != 0 or st.get("pending"):
        sys.exit("seed-ckpt only before the first session")
    src = Path(args.dir)
    stage = Path(cfg["local_dir"]) / "seed"
    shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
    files = {}
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name not in ("ckpt_manifest.json", "dataset-metadata.json"):
            shutil.copy(f, stage / f.name)
            files[f.name] = sha256(stage / f.name)
    json.dump({"session": 0, "exit": "seeded", "sha256": files}, open(stage / "ckpt_manifest.json", "w"), indent=1)
    st["ckpt_dataset"] = publish_ckpt(cfg, stage, 0)
    # Seeded files carry no fingerprint: session 1 adopts them explicitly.
    st["adopt_seed"] = "fingerprint.json" not in files
    save_state(cfg, st)
    log(f"seeded from {src}: {sorted(files)}")


def push_session(cfg, st, session):
    d = Path(cfg["local_dir"]) / f"session_{session}" / "nb"
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    acc = cfg["kaggle"]["accelerator"]
    env = dict(cfg["env"])
    if session == 1 and st.get("adopt_seed"):
        env["YTA_ADOPT_CHECKPOINT"] = "1"
    nb = nbgen.build(session, st["runner_sha256"], acc, env, cfg["session_hours"], cfg["run_name"])
    json.dump(nb, open(d / "run.ipynb", "w"), indent=1)
    sources = [st["runner_dataset"]] + ([st["ckpt_dataset"]] if st.get("ckpt_dataset") else [])
    meta = {"id": f"{cfg['kaggle']['user']}/{cfg['run_name']}", "title": cfg["run_name"], "code_file": "run.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": acc != "cpu", "enable_tpu": False,
            # Some GPUs are only offered through a competition (the RTX PRO
            # 6000 on Kaggle: attach it, and the notebook must run offline).
            "enable_internet": bool(cfg["kaggle"].get("enable_internet", True)), "keywords": [],
            "dataset_sources": sources, "kernel_sources": [],
            "competition_sources": list(cfg["kaggle"].get("competition_sources", [])), "model_sources": []}
    shape = nbgen.machine_shape(acc)
    if shape:
        meta["machine_shape"] = shape
    json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=1)
    # Record the intent BEFORE pushing: if this process dies after Kaggle
    # accepted the push, the next `run` waits for that session instead of
    # pushing a second one (and detects a push that never landed, below).
    st["pending"] = session
    st["push_confirmed"] = False
    save_state(cfg, st)
    log(f"pushing session {session} ({acc}, datasets {sources})")
    out = kaggle(cfg, "kernels", "push", "-p", str(d))
    log(out.splitlines()[-1] if out else "(no output from kaggle)")
    if "successfully pushed" not in out:
        # The CLI exits 0 even when Kaggle refuses the push (e.g. "Maximum
        # weekly GPU quota of 30.00 hours reached"): nothing was submitted, so
        # nothing is pending. The checkpoint handoff is untouched.
        st.pop("pending", None)
        st.pop("push_confirmed", None)
        save_state(cfg, st)
        sys.exit(f"Kaggle refused session {session}: {out.splitlines()[-1] if out else 'no output'}\n"
                 "Nothing was submitted; `run` again once the cause is gone (a GPU quota resets weekly).")
    st["push_confirmed"] = True
    st["pushed_unix"] = time.time()
    save_state(cfg, st)


def wait_session(cfg, session, push_confirmed=True):
    """Poll until the session ends. Returns "MISSING" when the notebook does
    not exist on Kaggle three polls in a row and the push was never
    confirmed — the push did not land and should be repeated."""
    last, missing = None, 0
    while True:
        s = kernel_status(cfg)
        if s != last:
            log(f"session {session}: {s}")
            last = s
        if s in ("COMPLETE", "ERROR", "CANCEL"):
            return s
        missing = missing + 1 if s == "MISSING" else 0
        if missing >= 3 and not push_confirmed:
            return "MISSING"
        time.sleep(POLL_SECONDS)


def pull_session(cfg, session):
    out = Path(cfg["local_dir"]) / f"session_{session}" / "out"
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    kaggle(cfg, "kernels", "output", f"{cfg['kaggle']['user']}/{cfg['run_name']}", "-p", str(out), timeout=7200)
    return out


def verify_ckpt(ck, session):
    man_p = ck / "ckpt_manifest.json"
    if not man_p.exists():
        return None, "no ckpt_manifest.json (the session died before saving its checkpoint)"
    man = json.load(open(man_p))
    if man.get("session") != session:
        return None, f"manifest is for session {man.get('session')}, expected {session}"
    for f, h in man["sha256"].items():
        if not (ck / f).exists() or sha256(ck / f) != h:
            return None, f"checkpoint file {f} is missing or corrupt"
    return man, None


def print_progress(ck):
    st_p, hist_p, res_p = ck / "status.json", ck / "history.jsonl", ck / "result.json"
    if st_p.exists():
        st = json.load(open(st_p))
        print(f"  stage: {st['stage']}   {json.dumps(st['detail'])}")
    if hist_p.exists():
        rows = read_history(hist_p)
        by_stage = {}
        for r in rows:
            by_stage.setdefault(r["stage"], []).append(r)
        for stage, rs in by_stage.items():
            tail = rs[-6:]
            es = "  ".join(f"{r['e_tot']:.6f}" for r in tail)
            print(f"  {stage:>4}: {len(rs):3d} cycles, last e_tot: {es}")
            if len(rs) >= 2:
                dt = (rs[-1]["t_unix"] - rs[0]["t_unix"]) / max(1, len(rs) - 1)
                if 0 < dt < 86400:
                    print(f"        ~{dt / 60:.0f} min per cycle")
    if res_p.exists():
        res = json.load(open(res_p))
        if res.get("summary"):
            s = res["summary"]
            print(f"  DONE: gap {s['gap_ev']:.4f} eV (direct {s['direct_gap_ev']:.4f} eV)")
        elif res.get("bands"):
            b = res["bands"]
            print(f"  bands: {len(b['energies_ha'])}/{len(b['kpts_scaled'])} k-points")


def progress_mark(ck):
    """(stage, SCF cycles recorded, band k-points done) — must grow every session."""
    stage = json.load(open(ck / "status.json"))["stage"] if (ck / "status.json").exists() else "?"
    cycles = len(read_history(ck / "history.jsonl"))
    bands = 0
    if (ck / "result.json").exists():
        bands = len((json.load(open(ck / "result.json")).get("bands") or {}).get("energies_ha") or [])
    files = sorted(p.name for p in ck.iterdir())
    return [stage, cycles, bands, files]


def cmd_run(cfg, args):
    check_account(cfg)
    st = load_state(cfg)
    if st.get("done"):
        print(f"run {cfg['run_name']} is already complete (session {st['session']}):")
        print_progress(Path(cfg["local_dir"]) / f"session_{st['session']}" / "out" / "ckpt")
        return
    if not st.get("runner_dataset"):
        sys.exit("publish the runner first: ktrun.py publish-runner CONFIG")
    sessions_left = args.sessions
    while sessions_left is None or sessions_left > 0:
        session = st.get("pending") or st["session"] + 1
        if not st.get("pending"):
            push_session(cfg, st, session)
        else:
            log(f"session {session} was pushed earlier; waiting for it")
        status = wait_session(cfg, session, st.get("push_confirmed", True))
        if status == "MISSING":
            log(f"the push of session {session} never reached Kaggle (no such notebook); pushing again")
            st.pop("pending", None)
            save_state(cfg, st)
            continue
        out = pull_session(cfg, session)
        ck = out / "ckpt"
        man, err = verify_ckpt(ck, session)
        if err and not st.get("push_confirmed", True):
            prev = None
            try:
                prev = json.load(open(ck / "ckpt_manifest.json")).get("session")
            except Exception:
                pass
            if prev != session:
                log(f"the push of session {session} never reached Kaggle (its output is session {prev}'s); pushing again")
                st.pop("pending", None)
                save_state(cfg, st)
                continue
        print(f"session {session} finished ({status}):")
        if err:
            log_tail = out / "run.log"
            print(f"  {err}")
            if log_tail.exists():
                print("  last lines of run.log:")
                for l in open(log_tail, errors="replace").read().splitlines()[-15:]:
                    print("   ", l[:200])
            # Nothing usable came back: session N+1 is pushed again from the
            # last good checkpoint, which `ckpt_dataset` still names.
            st.pop("pending", None)
            save_state(cfg, st)
            sys.exit("stopping: fix the cause, then `run` again (it re-pushes this session from the last good checkpoint)")
        print_progress(ck)
        stage = json.load(open(ck / "status.json"))["stage"] if (ck / "status.json").exists() else "?"
        st["session"] = session
        st.pop("pending", None)
        st.pop("push_confirmed", None)
        if stage == "done":
            st["done"] = True
            save_state(cfg, st)
            log(f"run complete; results in {ck}/result.json")
            return
        # Hand this session's checkpoint on BEFORE any stop, so `session` and
        # `ckpt_dataset` always agree and the next `run` resumes from here.
        st["ckpt_dataset"] = publish_ckpt(cfg, ck, session)
        mark = progress_mark(ck)
        stalled = mark == st.get("progress_mark")
        st["progress_mark"] = mark
        save_state(cfg, st)
        if stage == "failed":
            sys.exit("the run reported 'failed' (see status above; its checkpoint is kept). Adjust the config "
                     "(SCF settings, YTA_MAXCYC) and `run` again.")
        if stalled:
            sys.exit("stopping: this session made no progress (same stage, cycles, band points and files as "
                     "the previous one) — check run.log before spending another session")
        if sessions_left is not None:
            sessions_left -= 1
    log("session budget used up; `run` again to continue")


def cmd_status(cfg, args):
    st = load_state(cfg)
    print(f"run {cfg['run_name']}: {st['session']} session(s) finished"
          + (f", session {st['pending']} pushed and not yet pulled" if st.get("pending") else ""))
    if st.get("pending"):
        check_account(cfg)
        print(f"  Kaggle status of session {st['pending']}: {kernel_status(cfg)}")
    if st["session"]:
        print_progress(Path(cfg["local_dir"]) / f"session_{st['session']}" / "out" / "ckpt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("publish-runner", "run", "status", "seed-ckpt"):
        p = sub.add_parser(name)
        p.add_argument("config")
        if name == "run":
            p.add_argument("-n", "--sessions", type=int, default=None, help="stop after this many sessions")
        if name == "seed-ckpt":
            p.add_argument("dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"publish-runner": cmd_publish_runner, "run": cmd_run, "status": cmd_status,
     "seed-ckpt": cmd_seed_ckpt}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
