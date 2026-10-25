"""Tests for independent file-creation postcondition verification."""
import tempfile
import unittest
from pathlib import Path

from models import Task, TaskResult
from file_verifier import verify_file_creation


class FileVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "notes.txt"
        self.task = Task("t1", "dev", "Create notes.txt")

    def result(self, output, success=True):
        return TaskResult("t1", "dev", success, output)

    def test_verifies_real_readable_file(self):
        self.path.write_text("three things to remember", encoding="utf-8")
        result = verify_file_creation(self.task, self.result(f"File created: {self.path}"))
        self.assertTrue(result.verified)
        self.assertIn("exists and is readable", result.reason)

    def test_success_message_without_file_is_unverified(self):
        result = verify_file_creation(self.task, self.result(f"File created: {self.path}"))
        self.assertFalse(result.verified)
        self.assertIn("does not exist", result.reason)

    def test_empty_file_is_still_a_valid_created_file(self):
        self.path.write_text("", encoding="utf-8")
        result = verify_file_creation(self.task, self.result(f"File created: {self.path}"))
        self.assertTrue(result.verified)

    def test_missing_path_in_output_is_unverified(self):
        result = verify_file_creation(self.task, self.result("Done, sir."))
        self.assertFalse(result.verified)
        self.assertIn("did not identify", result.reason)

    def test_agent_failure_is_not_verified(self):
        self.path.write_text("present", encoding="utf-8")
        result = verify_file_creation(self.task, self.result(f"File created: {self.path}", success=False))
        self.assertFalse(result.verified)
        self.assertIn("reported failure", result.reason)

    def test_directory_is_not_a_file(self):
        result = verify_file_creation(self.task, self.result(f"File created: {self.temp_dir.name}"))
        self.assertFalse(result.verified)
        self.assertIn("does not exist", result.reason)


if __name__ == "__main__":
    unittest.main()
