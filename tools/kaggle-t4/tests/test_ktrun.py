"""ktrun.py against a fake `kaggle` CLI: the session loop, its recovery paths
and its stop conditions. Run: python3 -m unittest discover tools/kaggle-t4/tests"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
KTRUN = HERE.parent / "ktrun.py"


class KtrunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ktrun-test-"))
        self.srv = self.tmp / "server"; self.srv.mkdir()
        bindir = self.tmp / "bin"; bindir.mkdir()
        (bindir / "kaggle").write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_kaggle.py'} \"$@\"\n")
        (bindir / "kaggle").chmod(0o755)
        (self.tmp / "token").write_text("KGAT_fake")
        runner = self.tmp / "runner"; (runner / "pyscf" / "gto" / "basis").mkdir(parents=True)
        (runner / "yta7o19_bands").write_bytes(b"\x7fELF fake runner")
        self.cfg = self.tmp / "cfg.json"
        json.dump({"run_name": "test-run", "local_dir": str(self.tmp / "local"), "session_hours": 1,
                   "kaggle": {"user": "tester", "token_file": str(self.tmp / "token"), "accelerator": "t4"},
                   "runner": {"binary": str(runner / "yta7o19_bands"), "pyscf_data_dir": str(runner / "pyscf")},
                   "env": {"YTA_KE": "20"}}, open(self.cfg, "w"))
        self.env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FAKE_KAGGLE_DIR=str(self.srv),
                        KTRUN_POLL_SECONDS="0")

    def tearDown(self):
        bad = self.srv_state().get("bad_handoffs") if (self.srv / "server.json").exists() else None
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.assertFalse(bad, f"a session was pushed without the previous checkpoint: {bad}")

    def server(self, **kw):
        s = {"user": "tester", "plan": {}, "push_mode": "ok", "kernel": None, "statuses": ["COMPLETE"],
             "datasets": {}, "pushes": []}
        s.update(kw)
        json.dump(s, open(self.srv / "server.json", "w"))

    def srv_state(self):
        return json.load(open(self.srv / "server.json"))

    def set_srv(self, **kw):
        s = self.srv_state(); s.update(kw); json.dump(s, open(self.srv / "server.json", "w"))

    def ktrun(self, *args):
        return subprocess.run([sys.executable, str(KTRUN), *args, str(self.cfg)] if args[0] != "run-n" else
                              [sys.executable, str(KTRUN), "run", str(self.cfg), "-n", args[1]],
                              env=self.env, capture_output=True, text=True, timeout=120)

    def state(self):
        return json.load(open(self.tmp / "local" / "test-run" / "state.json"))

    def test_runs_to_completion_and_then_does_nothing(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 3}, "2": {"stage": "s1", "cycles": 7},
                          "3": {"stage": "done", "cycles": 9}})
        self.assertEqual(self.ktrun("publish-runner").returncode, 0)
        r = self.ktrun("run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        st = self.state()
        self.assertTrue(st["done"]); self.assertEqual(st["session"], 3)
        self.assertEqual(self.srv_state()["pushes"], [1, 2, 3])
        meta = self.srv_state()["meta"]
        self.assertIn("tester/test-run-ckpt-s2", meta["dataset_sources"])      # session 3 read session 2's ckpt
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        again = self.ktrun("run")
        self.assertEqual(again.returncode, 0)
        self.assertIn("already complete", again.stdout)
        self.assertEqual(self.srv_state()["pushes"], [1, 2, 3], "a finished run must not push again")

    def test_driver_death_after_an_accepted_push_waits_instead_of_repushing(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 2, "status": ["RUNNING", "RUNNING", "COMPLETE"]}},
                    push_mode="crash_after_accept")
        self.ktrun("publish-runner")
        r = self.ktrun("run-n", "1")
        self.assertNotEqual(r.returncode, 0)                  # the push "failed" from the driver's view
        self.assertEqual(self.state()["pending"], 1)
        r = self.ktrun("run-n", "1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.srv_state()["pushes"], [1], "the accepted session must not be pushed twice")
        self.assertEqual(self.state()["session"], 1)
        self.assertEqual(self.state()["ckpt_dataset"], "tester/test-run-ckpt-s1")

    def test_push_that_never_landed_is_pushed_again(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 2}, "2": {"stage": "s1", "cycles": 5}})
        self.ktrun("publish-runner")
        self.assertEqual(self.ktrun("run-n", "1").returncode, 0)
        self.set_srv(push_mode="drop")
        r = self.ktrun("run-n", "1")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.state()["pending"], 2)
        # The kernel on Kaggle is still session 1 (COMPLETE): its output says so.
        r = self.ktrun("run-n", "1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("never reached Kaggle", r.stdout)
        self.assertEqual(self.srv_state()["pushes"], [1, 2])
        self.assertEqual(self.state()["session"], 2)

    def test_first_push_that_never_landed_is_pushed_again(self):
        self.server(plan={"1": {"stage": "done", "cycles": 3}}, push_mode="drop")
        self.ktrun("publish-runner")
        r = self.ktrun("run")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.state()["pending"], 1)
        r = self.ktrun("run")                                 # notebook missing on Kaggle -> push again
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("never reached Kaggle", r.stdout)
        self.assertEqual(self.srv_state()["pushes"], [1])
        self.assertTrue(self.state()["done"])

    def test_interrupted_checkpoint_upload_is_adopted(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 2}, "2": {"stage": "done", "cycles": 4}},
                    dataset_mode={"tester/test-run-ckpt-s1": "crash_after_create"})
        self.ktrun("publish-runner")
        r = self.ktrun("run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)     # the lost response is reconciled in place
        self.assertTrue(self.state()["done"])
        self.assertEqual(self.srv_state()["pushes"], [1, 2])

    def test_rerun_after_an_upload_crash_reuses_the_dataset(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 2}, "2": {"stage": "done", "cycles": 4}})
        self.ktrun("publish-runner")
        # Simulate a driver killed right after the dataset landed: pre-create
        # it with the manifest session 1 will produce.
        import hashlib
        files = {"history.jsonl": "".join(json.dumps({"stage": "s1", "cycle": c, "e_tot": -1.0 - c,
                                                      "t_unix": 1000.0 + 60 * c}) + "\n" for c in range(2)),
                 "scf_s1.bin": "x" * 12, "status.json": json.dumps({"stage": "s1", "detail": {}})}
        sha = {f: hashlib.sha256(v.encode()).hexdigest() for f, v in files.items()}
        s = self.srv_state(); s["datasets"]["tester/test-run-ckpt-s1"] = "ready"
        s.setdefault("manifests", {})["tester/test-run-ckpt-s1"] = {"session": 1, "sha256": sha}
        json.dump(s, open(self.srv / "server.json", "w"))
        r = self.ktrun("run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("already exists", r.stdout)
        self.assertTrue(self.state()["done"])

    def test_existing_dataset_with_different_content_is_refused(self):
        self.server(plan={})
        self.ktrun("publish-runner")
        seed = self.tmp / "seed"; seed.mkdir(); (seed / "s1e.bin").write_bytes(b"old")
        run = lambda: subprocess.run([sys.executable, str(KTRUN), "seed-ckpt", str(self.cfg), str(seed)],
                                     env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(run().returncode, 0)
        (seed / "s1e.bin").write_bytes(b"new")                 # different files, same slug ...-ckpt-s0
        st = self.state(); st.pop("ckpt_dataset", None)
        json.dump(st, open(self.tmp / "local" / "test-run" / "state.json", "w"))
        r = run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DIFFERENT content", r.stdout + r.stderr)

    def test_reconciled_pending_dataset_with_different_content_is_refused(self):
        # Another upload of the same slug is still processing ("pending") when
        # we look; our create then conflicts; once ready it must be verified.
        self.server(plan={}, datasets={"tester/test-run-ckpt-s0": "ready"},
                    dataset_pending={"tester/test-run-ckpt-s0": 1},
                    manifests={"tester/test-run-ckpt-s0": {"session": 0, "sha256": {"s1e.bin": "0" * 64}}})
        self.ktrun("publish-runner")
        seed = self.tmp / "seed"; seed.mkdir(); (seed / "s1e.bin").write_bytes(b"mine")
        r = subprocess.run([sys.executable, str(KTRUN), "seed-ckpt", str(self.cfg), str(seed)],
                           env=self.env, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DIFFERENT content", r.stdout + r.stderr)

    def test_seed_marks_adoption_for_session_one_only(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 2}, "2": {"stage": "done", "cycles": 4}})
        seed = self.tmp / "seed"; seed.mkdir(); (seed / "s1e.bin").write_bytes(b"s"); (seed / "h1e.bin").write_bytes(b"h")
        self.ktrun("publish-runner")
        r = subprocess.run([sys.executable, str(KTRUN), "seed-ckpt", str(self.cfg), str(seed)], env=self.env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.ktrun("run-n", "1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        nb1 = (self.tmp / "local" / "test-run" / "session_1" / "nb" / "run.ipynb").read_text()
        self.assertIn("YTA_ADOPT_CHECKPOINT", nb1)
        self.assertIn("tester/test-run-ckpt-s0", self.srv_state()["meta"]["dataset_sources"])
        r = self.ktrun("run")
        nb2 = (self.tmp / "local" / "test-run" / "session_2" / "nb" / "run.ipynb").read_text()
        self.assertNotIn("YTA_ADOPT_CHECKPOINT", nb2)

    def test_competition_gpu_settings_reach_the_notebook_metadata(self):
        cfg = json.load(open(self.cfg))
        cfg["kaggle"].update(accelerator="rtxpro6000", competition_sources=["some-competition"], enable_internet=False)
        json.dump(cfg, open(self.cfg, "w"))
        self.server(plan={"1": {"stage": "done", "cycles": 1}})
        self.ktrun("publish-runner")
        self.assertEqual(self.ktrun("run").returncode, 0)
        meta = self.srv_state()["meta"]
        self.assertEqual(meta["machine_shape"], "NvidiaRtxPro6000")
        self.assertEqual(meta["competition_sources"], ["some-competition"])
        self.assertFalse(meta["enable_internet"])
        nb = (self.tmp / "local" / "test-run" / "session_1" / "nb" / "run.ipynb").read_text()
        self.assertIn("RTX PRO 6000", nb)                      # the GPU guard matches the accelerator

    def test_refused_push_stops_without_a_phantom_session(self):
        # Kaggle prints a quota error and exits 0; the old kernel still says COMPLETE.
        self.server(plan={"1": {"stage": "s1", "cycles": 2}, "2": {"stage": "done", "cycles": 4}})
        self.ktrun("publish-runner")
        self.assertEqual(self.ktrun("run-n", "1").returncode, 0)
        self.set_srv(push_mode="refuse")
        r = self.ktrun("run")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("quota", r.stdout + r.stderr)
        st = self.state()
        self.assertEqual(st["session"], 1)
        self.assertNotIn("pending", st)
        self.assertEqual(st["ckpt_dataset"], "tester/test-run-ckpt-s1")
        r = self.ktrun("run")                                  # quota back: session 2 goes through
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.srv_state()["pushes"], [1, 2])
        self.assertTrue(self.state()["done"])

    def test_no_progress_stops_with_a_consistent_handoff(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 4}, "2": {"stage": "s1", "cycles": 4}})
        self.ktrun("publish-runner")
        r = self.ktrun("run")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no progress", r.stdout + r.stderr)
        st = self.state()
        self.assertEqual(st["session"], 2)
        self.assertEqual(st["ckpt_dataset"], "tester/test-run-ckpt-s2", "session and checkpoint must agree")
        self.assertNotIn("pending", st)

    def test_failed_stage_keeps_its_checkpoint(self):
        self.server(plan={"1": {"stage": "failed", "cycles": 80}})
        self.ktrun("publish-runner")
        r = self.ktrun("run")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("failed", r.stdout + r.stderr)
        self.assertEqual(self.state()["ckpt_dataset"], "tester/test-run-ckpt-s1")

    def test_torn_history_line_is_tolerated(self):
        self.server(plan={"1": {"stage": "s1", "cycles": 3, "torn": True}, "2": {"stage": "done", "cycles": 5}})
        self.ktrun("publish-runner")
        r = self.ktrun("run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 cycles", r.stdout)

    def test_wrong_account_is_refused(self):
        self.server(user="someone-else")
        r = self.ktrun("publish-runner")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("belongs to", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
