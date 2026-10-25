"""Unit tests for KOBRA's verified task executor. Run with: python -m unittest discover -s tests"""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from models import Task, TaskResult
from verified_executor import VerificationResult, VerifiedTaskExecutor


class FakeAgent:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def execute(self, task, abort_flag):
        self.calls += 1
        item = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(item, Exception):
            raise item
        return TaskResult(task.id, task.agent_name, item[0], item[1])


class VerifiedExecutorTests(unittest.TestCase):
    def test_downstream_runs_only_after_dependency_is_verified(self):
        calls = []
        agent_a = FakeAgent([(True, "created file")])
        agent_b = FakeAgent([(True, "ran checks")])

        def verifier(task, result):
            calls.append(task.id)
            return VerificationResult(task.id == "a", "file exists" if task.id == "a" else "checks not verified")

        report = VerifiedTaskExecutor(verifier).execute(
            "create and check file",
            [
                Task("a", "dev", "create file"),
                Task("b", "dev", "run checks", depends_on=["a"]),
            ],
            {"dev": agent_a},
        )

        self.assertEqual(report.status, "PARTIAL")
        self.assertEqual(agent_a.calls, 1)  # same fake agent represents both steps below
        self.assertEqual([step.status for step in report.steps], ["VERIFIED", "UNVERIFIED"])
        self.assertEqual(calls, ["a", "b"])

    def test_retry_requires_verifier_to_mark_failure_retryable(self):
        agent = FakeAgent([(True, "attempt one"), (True, "attempt two")])
        attempts = {"count": 0}

        def verifier(task, result):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return VerificationResult(False, "output not found yet", retryable=True)
            return VerificationResult(True, "output confirmed")

        report = VerifiedTaskExecutor(verifier, max_attempts=2).execute(
            "run a step", [Task("a", "dev", "do it")], {"dev": agent}
        )
        self.assertTrue(report.success)
        self.assertEqual(report.steps[0].attempts, 2)
        self.assertEqual(agent.calls, 2)

    def test_failed_verification_does_not_retry_by_default(self):
        agent = FakeAgent([(True, "action attempted")])
        report = VerifiedTaskExecutor(
            lambda task, result: VerificationResult(False, "postcondition missing"),
            max_attempts=3,
        ).execute("run a step", [Task("a", "dev", "do it")], {"dev": agent})
        self.assertEqual(report.steps[0].status, "UNVERIFIED")
        self.assertEqual(agent.calls, 1)

    def test_unknown_agent_is_reported_without_claiming_success(self):
        report = VerifiedTaskExecutor(
            lambda task, result: VerificationResult(True, "verified")
        ).execute("missing agent", [Task("a", "ghost", "do it")], {})
        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.steps[0].status, "FAILED")

    def test_cycle_is_rejected_before_execution(self):
        agent = FakeAgent([(True, "should not run")])
        executor = VerifiedTaskExecutor(
            lambda task, result: VerificationResult(True, "verified")
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            executor.execute(
                "cycle",
                [Task("a", "dev", "a", depends_on=["b"]), Task("b", "dev", "b", depends_on=["a"])],
                {"dev": agent},
            )
        self.assertEqual(agent.calls, 0)

    def test_checkpoint_is_atomic_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.json"
            report = VerifiedTaskExecutor(
                lambda task, result: VerificationResult(True, "checked"),
                checkpoint_path=path,
            ).execute("checkpoint test", [Task("a", "dev", "do it")],
                      {"dev": FakeAgent([(True, "done")])})
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["execution_id"], report.execution_id)
            self.assertEqual(saved["status"], "VERIFIED")
            self.assertEqual(saved["steps"][0]["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
