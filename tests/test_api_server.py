from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("fastapi is not installed")

from fastapi.testclient import TestClient

from app.api.server import create_app


class APIServerTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.client = TestClient(create_app())
            # Trigger app build early to catch incompatible fastapi/starlette/httpx sets.
            self.client.get("/v1/health")
        except ValueError as exc:
            if "too many values to unpack" in str(exc):
                raise unittest.SkipTest("Incompatible fastapi/starlette/httpx in current python env") from exc
            raise

    def test_health(self) -> None:
        res = self.client.get("/v1/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_chat_endpoint(self) -> None:
        res = self.client.post("/v1/agent/chat", json={"prompt": "app以下のpythonファイルを教えて"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("message", body)
        self.assertIn("data", body)

    def test_search_endpoint(self) -> None:
        res = self.client.post("/v1/tools/search", json={"root_path": "app", "pattern": "*.py", "max_results": 5})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("Found", body["message"])


if __name__ == "__main__":
    unittest.main()
