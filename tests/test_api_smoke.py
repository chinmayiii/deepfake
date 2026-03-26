import importlib.util
import asyncio
import os
from pathlib import Path
import unittest
import warnings

from fastapi.testclient import TestClient

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*unclosed event loop.*",
    category=ResourceWarning,
    module=r"asyncio\.base_events",
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "web-app.py"


class APISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SKIP_MODEL_LOAD"] = "1"
        os.environ["METRICS_ADMIN_TOKEN"] = "test-token"
        spec = importlib.util.spec_from_file_location("web_app", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.client = TestClient(module.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and not loop.is_closed():
            loop.close()
            asyncio.set_event_loop(None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("status", payload)
        self.assertIn("model_ready", payload)
        self.assertIn("app_version", payload)

    def test_live_endpoint(self):
        response = self.client.get("/live")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "alive")
        self.assertIn("app_version", payload)
        self.assertIn("uptime_sec", payload)

    def test_ready_endpoint(self):
        response = self.client.get("/ready")
        self.assertIn(response.status_code, (200, 503))
        payload = response.json()
        self.assertIn("status", payload)
        self.assertIn("model_ready", payload)
        self.assertIn("app_version", payload)
        if payload["model_ready"]:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["status"], "ready")
        else:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(payload["status"], "not_ready")

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("model_mode", payload)
        self.assertIn("model_ready", payload)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("totals", payload)
        self.assertIn("endpoints", payload)
        self.assertIn("uptime_sec", payload)
        self.assertIn("/metrics", payload["endpoints"])
        self.assertIn("avg_latency_ms", payload["totals"])

    def test_config_endpoint(self):
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("max_upload_mb", payload)
        self.assertIn("max_batch_files", payload)
        self.assertIn("rate_limit_per_min", payload)
        self.assertIn("max_request_seconds", payload)
        self.assertIn("supported_extensions", payload)
        self.assertIn("metrics_reset_enabled", payload)
        self.assertIn("eval_summary_path", payload)
        self.assertTrue(payload["metrics_reset_enabled"])

    def test_eval_summary_endpoint(self):
        response = self.client.get("/eval/summary")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("num_samples", payload)
        self.assertIn("accuracy", payload)
        self.assertIn("f1", payload)
        self.assertIn("confusion_matrix", payload)

    def test_capabilities_endpoint(self):
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("app_version", payload)
        self.assertIn("probes", payload)
        self.assertIn("inference", payload)
        self.assertIn("observability", payload)
        self.assertIn("controls", payload)
        self.assertIn("evaluation_headline", payload)
        self.assertEqual(payload["inference"]["single"], "/detect")

    def test_metrics_reset_requires_token(self):
        response = self.client.post("/metrics/reset")
        self.assertEqual(response.status_code, 403)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "INVALID_ADMIN_TOKEN")

    def test_metrics_reset_success(self):
        self.client.get("/health")
        response = self.client.post(
            "/metrics/reset", headers={"x-admin-token": "test-token"}
        )
        self.assertEqual(response.status_code, 200)

        metrics_response = self.client.get("/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_payload = metrics_response.json()
        self.assertEqual(metrics_payload["totals"]["requests"], 1)
        self.assertEqual(metrics_payload["totals"]["successful"], 1)
        self.assertIn("/metrics", metrics_payload["endpoints"])

    def test_detect_requires_file(self):
        response = self.client.post("/detect")
        self.assertEqual(response.status_code, 422)

    def test_batch_requires_files(self):
        response = self.client.post("/detect/batch")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
