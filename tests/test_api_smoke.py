import importlib.util
import os
from pathlib import Path
import unittest
import warnings

from fastapi.testclient import TestClient

warnings.filterwarnings(
    "ignore",
    message=r"unclosed event loop.*",
    category=ResourceWarning,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "web-app.py"


class APISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SKIP_MODEL_LOAD"] = "1"
        spec = importlib.util.spec_from_file_location("web_app", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.client = TestClient(module.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("status", payload)
        self.assertIn("model_ready", payload)
        self.assertIn("app_version", payload)

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("model_mode", payload)
        self.assertIn("model_ready", payload)

    def test_detect_requires_file(self):
        response = self.client.post("/detect")
        self.assertEqual(response.status_code, 422)

    def test_batch_requires_files(self):
        response = self.client.post("/detect/batch")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
