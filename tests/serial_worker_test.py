"""Failure and replay fixtures for the background receipt contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import serial_worker as worker


class SerialWorkerTest(unittest.TestCase):
    def test_research_budget_carries_forward_and_cannot_be_reused(self):
        contract=dict(question="claim",competing_explanation="rival",single_change="one intervention",
                      refuter="counterexample",study="phase-test",routes={k:k for k in worker.RESEARCH_OUTCOMES})
        plan=dict(ticket="fixture",steps=[dict(seconds=30)],inputs={worker.RESEARCH_BUDGET:"bound"},research=contract)
        old={"phase-test":dict(hard_seconds=70,allocated_seconds={"prior":30})}
        current={"phase-test":dict(hard_seconds=70,allocated_seconds={"prior":30,"fixture":30})}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"ops").mkdir()
            (root/worker.RESEARCH_BUDGET).write_text(json.dumps(current),encoding="utf8")
            with patch.object(worker,"ROOT",root), patch.object(worker.subprocess,"run") as git:
                git.return_value.returncode=0
                git.return_value.stdout=json.dumps(old).encode()
                worker.validate_research(plan,"sha")
                current["phase-test"]["allocated_seconds"]["prior"]=29
                (root/worker.RESEARCH_BUDGET).write_text(json.dumps(current),encoding="utf8")
                with self.assertRaisesRegex(ValueError,"carry forward"):
                    worker.validate_research(plan,"sha")
            worker.bind_research_reservation(plan,"first",root)
            worker.bind_research_reservation(plan,"first",root)
            with self.assertRaisesRegex(ValueError,"already bound"):
                worker.bind_research_reservation(plan,"second",root)

    def test_research_result_can_be_inconclusive_after_successful_execution(self):
        plan=dict(ticket="fixture",baseline="sha",steps=[dict(name="compare")],
                  research=dict(routes={k:k for k in worker.RESEARCH_OUTCOMES}))
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            def run(step, log):
                log.write_text("JUNQI_RESEARCH inconclusive\n",encoding="utf8")
            self.assertEqual(worker.execute_stage(plan,0,folder,run)[0],"passed")
            self.assertEqual(worker.read(folder/"0.json")["research_outcome"],"inconclusive")

    def test_invalid_later_stage_is_rejected_at_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ops").mkdir()
            (root / "tools").mkdir()
            (root / "tools" / "check.ps1").touch()
            (root / "tools" / "experiment.py").touch()
            plan = dict(ticket="fixture", baseline="sha", steps=[
                dict(kind="check", name="preflight", seconds=90),
                dict(kind="python", name="confirmation", seconds=60,
                     script="tools/experiment.py", args=["audit-current"]),
            ])
            task = dict(id="fixture", baseline="sha", allowed=[])
            (root / "ops" / "task.json").write_text(json.dumps(task), encoding="utf8")
            (root / "ops" / "serial.json").write_text(json.dumps(plan), encoding="utf8")
            with patch.object(worker, "ROOT", root), patch.object(worker.subprocess, "check_output", return_value="sha"):
                with patch.object(worker, "plan_digest") as digest:
                    with self.assertRaisesRegex(ValueError, "confirmation: invalid script arguments"):
                        worker.load_plan()
                    digest.assert_not_called()
                plan["steps"][1]["args"] = ["audit"]
                (root / "ops" / "serial.json").write_text(json.dumps(plan), encoding="utf8")
                self.assertEqual(worker.load_plan()[0], plan)

    def test_failed_step_cannot_advance_or_repeat(self):
        calls = []

        def fail(step, log):
            calls.append(step["name"])
            log.write_text("fixture failure", encoding="utf8")
            raise RuntimeError("fixture failure")

        plan = dict(ticket="fixture", baseline="sha", steps=[dict(name="check"), dict(name="run")])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertEqual(worker.execute_stage(plan, 1, folder, fail)[0], "blocked")
            self.assertEqual(worker.execute_stage(plan, 0, folder, fail)[0], "blocked")
            self.assertEqual(worker.execute_stage(plan, 0, folder, fail)[0], "blocked")
            self.assertEqual(worker.execute_stage(plan, 1, folder, fail)[0], "blocked")
            self.assertEqual(calls, ["check"])
            self.assertEqual(worker.read(folder / "0.json")["status"], "blocked")
            self.assertFalse((folder / "1.json").exists())

    def test_stage_identity_and_replay(self):
        event = dict(type=worker.EVENT, id=worker.event_id("plan", 1))
        self.assertEqual(worker.requested_stage(dict(events=[event]), "plan", 2), 1)
        self.assertIsNone(worker.requested_stage(dict(events=[event]), "new-plan", 2))
        calls = []
        plan = dict(ticket="fixture", baseline="sha", steps=[dict(name="check"), dict(name="run")])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run = lambda step, log: calls.append(step["name"])
            for index in (0, 0, 1, 1):
                self.assertEqual(worker.execute_stage(plan, index, folder, run)[0], "passed")
        self.assertEqual(calls, ["check", "run"])

    def test_execution_changes_invalidate_events_but_reports_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}", encoding="utf8")
            (root / "tools").mkdir()
            script = root / "tools/experiment.py"
            script.write_text("print(1)", encoding="utf8")
            paths = ["tools/experiment.py", "report.md"]
            before = worker.plan_digest(plan, paths, root)
            (root / "report.md").write_text("result", encoding="utf8")
            self.assertEqual(worker.plan_digest(plan, paths, root), before)
            script.write_text("print(2)", encoding="utf8")
            self.assertNotEqual(worker.plan_digest(plan, paths, root), before)

    def test_rejected_business_result_cannot_advance(self):
        plan = dict(ticket="fixture", baseline="sha", steps=[dict(name="review"), dict(name="successor")])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)

            def reject(step, log):
                raise worker.CandidateRejected("wrong encoding")

            self.assertEqual(worker.execute_stage(plan, 0, folder, reject)[0], "rejected")
            self.assertEqual(worker.execute_stage(plan, 1, folder, reject)[0], "blocked")
            self.assertFalse((folder / "1.json").exists())

    def test_explicit_resume_preserves_previous_and_interrupted_receipts(self):
        plan = dict(ticket="fixture", baseline="sha",
                    steps=[dict(name="first"), dict(name="second", resumable=True)])
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run = lambda step, log: calls.append(step["name"])
            worker.execute_stage(plan, 0, folder, run)
            first = (folder / "0.json").read_bytes()
            worker.write_receipt(folder / "1.json", dict(status="running", message="interrupted"))
            (folder / "1.log").write_text("before interruption", encoding="utf8")
            self.assertEqual(worker.execute_stage(plan, 1, folder, run)[0], "running")
            self.assertEqual(worker.execute_stage(plan, 1, folder, run, resume=True)[0], "passed")
            self.assertEqual(worker.execute_stage(plan, 1, folder, run, resume=True)[0], "passed")
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual((folder / "0.json").read_bytes(), first)
            self.assertEqual(worker.read(folder / "1.json")["attempt"], 2)
            self.assertEqual(worker.read(folder / "1.interrupted.json")["status"], "running")

    def test_input_change_during_execution_blocks_before_success(self):
        plan = dict(ticket="fixture", baseline="sha", steps=[dict(name="measure")])

        def changed():
            raise ValueError("input fingerprint mismatch")

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            status, _ = worker.execute_stage(plan, 0, folder, lambda step, log: None, recheck=changed)
            self.assertEqual(status, "blocked")
            self.assertEqual(worker.read(folder / "0.json")["status"], "blocked")

    def test_recovery_marker_and_worker_exclusion(self):
        manual = dict(type="run.manual", detail="resume:current:1")
        self.assertEqual(worker.resume_index(dict(events=[manual]), "current", 2), (True, 1))
        self.assertEqual(worker.resume_index(dict(events=[manual]), "old", 2), (True, None))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with worker.worker_lock(folder), self.assertRaises(OSError), worker.worker_lock(folder):
                self.fail("second worker acquired the same execution lock")


if __name__ == "__main__":
    unittest.main()
