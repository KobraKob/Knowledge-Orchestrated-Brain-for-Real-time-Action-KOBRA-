"""Unit tests for the optional Cognee memory adapter; no Cognee install required."""
import unittest

from cognee_memory import CogneeMemory


class FakeCognee:
    def __init__(self):
        self.remember_calls = []
        self.recall_calls = []

    async def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        return {"ok": True}

    async def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return [{"source": "graph", "content": "KOBRA uses SQLite as its source of truth."}]


class CogneeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = CogneeMemory()
        self.fake = FakeCognee()
        self.memory._cognee = self.fake

    def test_remember_sends_dataset_and_payload(self):
        ok = self.memory.remember(
            "KOBRA keeps SQLite authoritative",
            kind="fact",
            source_id="fact:memory_backend",
        )
        self.assertTrue(ok)
        self.assertEqual(len(self.fake.remember_calls), 1)
        call = self.fake.remember_calls[0]
        self.assertEqual(call["dataset_name"], self.memory._dataset)
        self.assertFalse(call["self_improvement"])
        self.assertIn("KOBRA keeps SQLite authoritative", call["data"])
        self.assertIn("fact:memory_backend", call["data"])

    def test_recall_passes_query_and_normalizes_results(self):
        results = self.memory.recall("What is KOBRA's memory backend?")
        self.assertEqual(len(self.fake.recall_calls), 1)
        call = self.fake.recall_calls[0]
        self.assertEqual(call["query_text"], "What is KOBRA's memory backend?")
        self.assertEqual(call["datasets"], [self.memory._dataset])
        self.assertEqual(call["top_k"], 5)
        self.assertEqual(results[0]["source"], "graph")
        self.assertIn("SQLite", results[0]["content"])

    def test_missing_cognee_falls_back_without_crashing(self):
        self.memory._cognee = None
        self.memory._disabled = True
        self.assertFalse(self.memory.remember("test", kind="fact", source_id="test"))
        self.assertEqual(self.memory.recall("test"), [])


if __name__ == "__main__":
    unittest.main()
