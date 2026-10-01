#!/usr/bin/env python3
"""A stand-in for the `kaggle` CLI, driven by $FAKE_KAGGLE_DIR/server.json.

server.json:
  user            username `config view` reports
  plan            {session: {"stage": ..., "cycles": N, "status": [...], "manifest_session": S?}}
  push_mode       "ok" | "crash_after_accept" | "drop"   (applies to the next push, then resets to "ok")
  kernel          the currently submitted session (set by push)
  statuses        remaining statuses for the current kernel
  datasets        slug -> "ready"
  pushes          list of pushed session numbers
"""
import hashlib, json, os, re, sys
from pathlib import Path

D = Path(os.environ["FAKE_KAGGLE_DIR"])
S = json.load(open(D / "server.json"))


def save():
    json.dump(S, open(D / "server.json", "w"), indent=1)


def arg(flag):
    a = sys.argv
    return a[a.index(flag) + 1]


cmd = sys.argv[1:3]
if cmd == ["config", "view"]:
    print(f"- username: {S['user']}")
elif cmd == ["datasets", "create"]:
    meta = json.load(open(Path(arg("-p")) / "dataset-metadata.json"))
    if meta["id"] in S["datasets"]:
        print(f"409 Conflict: dataset {meta['id']} already exists", file=sys.stderr); sys.exit(1)
    S["datasets"][meta["id"]] = "ready"
    S.setdefault("dataset_files", {})[meta["id"]] = sorted(os.listdir(arg("-p")))
    man = Path(arg("-p")) / "ckpt_manifest.json"
    if man.exists():
        S.setdefault("manifests", {})[meta["id"]] = json.load(open(man))
    mode = S.get("dataset_mode", {}).pop(meta["id"], "ok")
    save()
    if mode == "crash_after_create":
        print("connection reset", file=sys.stderr); sys.exit(1)
    print("created")
elif cmd == ["datasets", "download"]:
    m = S.get("manifests", {}).get(sys.argv[3])
    if m is None:
        print("404 Not Found", file=sys.stderr); sys.exit(1)
    json.dump(m, open(Path(arg("-p")) / "ckpt_manifest.json", "w"))
elif cmd == ["datasets", "status"]:
    slug = sys.argv[3]
    pend = S.get("dataset_pending", {})
    if pend.get(slug, 0) > 0:
        pend[slug] -= 1; save(); print("pending")
    else:
        print(S["datasets"].get(slug, "not found"))
elif cmd == ["kernels", "push"]:
    src = open(Path(arg("-p")) / "run.ipynb").read()
    session = int(re.search(r"SESSION = (\d+)", src).group(1))
    mode = S.get("push_mode", "ok"); S["push_mode"] = "ok"
    if mode == "drop":
        save(); print("network error", file=sys.stderr); sys.exit(1)
    S["kernel"] = session
    S["pushes"].append(session)
    meta = json.load(open(Path(arg("-p")) / "kernel-metadata.json"))
    want = f"{S['user']}/test-run-ckpt-s{session - 1}"
    if session > 1 and want not in meta["dataset_sources"]:
        S.setdefault("bad_handoffs", []).append([session, meta["dataset_sources"]])
    for ds in meta["dataset_sources"]:
        if S["datasets"].get(ds) != "ready":
            S.setdefault("bad_handoffs", []).append([session, f"unpublished {ds}"])
    S["statuses"] = list(S["plan"][str(session)].get("status", ["RUNNING", "COMPLETE"]))
    S["meta"] = json.load(open(Path(arg("-p")) / "kernel-metadata.json"))
    save()
    if mode == "crash_after_accept":
        print("connection reset", file=sys.stderr); sys.exit(1)
    print(f"Kernel version {len(S['pushes'])} successfully pushed.")
elif cmd == ["kernels", "status"]:
    if S["kernel"] is None:
        print("404 Client Error: Not Found for url: https://www.kaggle.com/api/v1/kernels/status", file=sys.stderr)
        sys.exit(1)
    st = S["statuses"].pop(0) if len(S["statuses"]) > 1 else S["statuses"][0]
    save(); print(f'has status "KernelWorkerStatus.{st}"')
elif cmd == ["kernels", "output"]:
    session = S["kernel"]
    plan = S["plan"][str(session)]
    ck = Path(arg("-p")) / "ckpt"; ck.mkdir(parents=True, exist_ok=True)
    (Path(arg("-p")) / "run.log").write_text("fake log\n")
    json.dump({"stage": plan["stage"], "detail": {}}, open(ck / "status.json", "w"))
    with open(ck / "history.jsonl", "w") as f:
        for c in range(plan["cycles"]):
            f.write(json.dumps({"stage": "s1", "cycle": c, "e_tot": -1.0 - c, "t_unix": 1000.0 + 60 * c}) + "\n")
        if plan.get("torn"):
            f.write('{"stage": "s1", "cyc')
    (ck / "scf_s1.bin").write_bytes(b"x" * (10 + plan["cycles"]))
    if plan["stage"] == "done":
        json.dump({"summary": {"gap_ev": 1.0, "direct_gap_ev": 1.1}}, open(ck / "result.json", "w"))
    files = {f: hashlib.sha256((ck / f).read_bytes()).hexdigest() for f in sorted(os.listdir(ck))}
    json.dump({"session": plan.get("manifest_session", session), "sha256": files}, open(ck / "ckpt_manifest.json", "w"))
else:
    print("fake kaggle: unhandled", sys.argv[1:], file=sys.stderr); sys.exit(2)
