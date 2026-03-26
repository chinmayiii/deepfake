import os
import io
import json
import base64
import tempfile
import time
import uuid
import hashlib
import threading
from pathlib import Path
import gradio as gr
import torch
import mimetypes
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Header, Request
from fastapi.responses import JSONResponse
from torchcam.methods import GradCAM
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from lightning_modules.detector import HybridEfficientNet
from utils.diffusion_heuristics import diffusion_heuristic_score, classify_generation
from utils.fft_utils import fft_from_pil

ROOT_DIR = Path(__file__).resolve().parent
HYBRID_MODEL_PATH = ROOT_DIR / "models" / "best_model-hybrid.pt"
LEGACY_MODEL_PATH = ROOT_DIR / "models" / "best_model-v3.pt"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
METRICS_ADMIN_TOKEN = os.environ.get("METRICS_ADMIN_TOKEN", "")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
MAX_REQUEST_SECONDS = int(os.environ.get("MAX_REQUEST_SECONDS", "30"))
EVAL_SUMMARY_PATH = Path(
    os.environ.get("EVAL_SUMMARY_PATH", str(ROOT_DIR / "docs" / "eval_summary.json"))
)
APP_START_TIME = time.time()

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".mp4", ".mov"]
MAX_BATCH_FILES = 10

_METRICS_LOCK = threading.Lock()
_API_METRICS = {
    "totals": {
        "requests": 0,
        "successful": 0,
        "failed": 0,
        "items_processed": 0,
        "manipulated": 0,
        "authentic": 0,
        "uncertain": 0,
        "latency_ms_sum": 0.0,
    },
    "endpoints": {},
}

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STATE = {}
_RATE_LIMIT_PRUNE_EVERY = 200
_RATE_LIMIT_HITS = 0


def _reset_api_metrics():
    with _METRICS_LOCK:
        totals = _API_METRICS["totals"]
        totals["requests"] = 0
        totals["successful"] = 0
        totals["failed"] = 0
        totals["items_processed"] = 0
        totals["manipulated"] = 0
        totals["authentic"] = 0
        totals["uncertain"] = 0
        totals["latency_ms_sum"] = 0.0
        _API_METRICS["endpoints"] = {}


def _enforce_rate_limit(endpoint, client_key):
    global _RATE_LIMIT_HITS
    now = time.time()
    window_start = now - 60
    key = (endpoint, str(client_key or "unknown"))

    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_HITS += 1
        recent_calls = _RATE_LIMIT_STATE.get(key, [])
        recent_calls = [ts for ts in recent_calls if ts >= window_start]

        if RATE_LIMIT_PER_MIN > 0 and len(recent_calls) >= RATE_LIMIT_PER_MIN:
            _RATE_LIMIT_STATE[key] = recent_calls
            raise PermissionError("Rate limit exceeded. Please retry later.")

        recent_calls.append(now)
        _RATE_LIMIT_STATE[key] = recent_calls

        if _RATE_LIMIT_HITS % _RATE_LIMIT_PRUNE_EVERY == 0:
            for state_key, timestamps in list(_RATE_LIMIT_STATE.items()):
                fresh = [ts for ts in timestamps if ts >= window_start]
                if fresh:
                    _RATE_LIMIT_STATE[state_key] = fresh
                else:
                    _RATE_LIMIT_STATE.pop(state_key, None)


def _ensure_not_timed_out(start_time):
    if (
        MAX_REQUEST_SECONDS > 0
        and (time.perf_counter() - start_time) > MAX_REQUEST_SECONDS
    ):
        raise TimeoutError(f"Request exceeded timeout of {MAX_REQUEST_SECONDS} seconds")


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _update_api_metrics(
    endpoint,
    duration_ms,
    success,
    items_processed=1,
    manipulated=0,
    authentic=0,
    uncertain=0,
):
    with _METRICS_LOCK:
        totals = _API_METRICS["totals"]
        totals["requests"] += 1
        totals["successful"] += 1 if success else 0
        totals["failed"] += 0 if success else 1
        totals["items_processed"] += max(int(items_processed), 0)
        totals["manipulated"] += max(int(manipulated), 0)
        totals["authentic"] += max(int(authentic), 0)
        totals["uncertain"] += max(int(uncertain), 0)
        totals["latency_ms_sum"] += max(float(duration_ms), 0.0)

        endpoint_metrics = _API_METRICS["endpoints"].setdefault(
            endpoint,
            {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "items_processed": 0,
                "manipulated": 0,
                "authentic": 0,
                "uncertain": 0,
                "latency_ms_sum": 0.0,
            },
        )
        endpoint_metrics["requests"] += 1
        endpoint_metrics["successful"] += 1 if success else 0
        endpoint_metrics["failed"] += 0 if success else 1
        endpoint_metrics["items_processed"] += max(int(items_processed), 0)
        endpoint_metrics["manipulated"] += max(int(manipulated), 0)
        endpoint_metrics["authentic"] += max(int(authentic), 0)
        endpoint_metrics["uncertain"] += max(int(uncertain), 0)
        endpoint_metrics["latency_ms_sum"] += max(float(duration_ms), 0.0)


def _verdict_counts_from_prediction(prediction_text):
    text = str(prediction_text or "").lower()
    if "deepfake" in text or "manipulated" in text:
        return 1, 0, 0
    if "real" in text or "authentic" in text:
        return 0, 1, 0
    return 0, 0, 1


def _build_metrics_response():
    uptime_sec = round(time.time() - APP_START_TIME, 2)
    with _METRICS_LOCK:
        totals = dict(_API_METRICS["totals"])
        endpoints_raw = {
            key: dict(value) for key, value in _API_METRICS["endpoints"].items()
        }

    total_requests = totals["requests"]
    totals["avg_latency_ms"] = (
        round(totals["latency_ms_sum"] / total_requests, 2) if total_requests else 0.0
    )
    totals["success_rate"] = _safe_ratio(totals["successful"], total_requests)

    endpoints = {}
    for endpoint, data in endpoints_raw.items():
        count = data["requests"]
        endpoints[endpoint] = {
            "requests": count,
            "successful": data["successful"],
            "failed": data["failed"],
            "items_processed": data["items_processed"],
            "manipulated": data["manipulated"],
            "authentic": data["authentic"],
            "uncertain": data["uncertain"],
            "avg_latency_ms": round(data["latency_ms_sum"] / count, 2)
            if count
            else 0.0,
            "success_rate": _safe_ratio(data["successful"], count),
        }

    return {
        "app_version": APP_VERSION,
        "model_mode": model_mode,
        "model_ready": not bool(model_load_error),
        "uptime_sec": uptime_sec,
        "totals": totals,
        "endpoints": endpoints,
    }


def _api_error(status_code, message, error_code, request_id=None):
    payload = {
        "error": message,
        "error_code": error_code,
    }
    if request_id:
        payload["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=payload)


def _load_eval_summary():
    eval_path = EVAL_SUMMARY_PATH
    if not eval_path.is_absolute():
        eval_path = (ROOT_DIR / eval_path).resolve()

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found at {eval_path}. Run realeval.py with --out to generate it."
        )

    with open(eval_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Evaluation summary must be a JSON object")

    payload.setdefault("source_path", str(eval_path))
    return payload


def _build_capabilities_summary():
    summary = {
        "app_version": APP_VERSION,
        "model_mode": model_mode,
        "probes": {
            "liveness": "/live",
            "readiness": "/ready",
            "health": "/health",
        },
        "inference": {
            "single": "/detect",
            "batch": "/detect/batch",
            "max_batch_files": MAX_BATCH_FILES,
            "supported_extensions": SUPPORTED_EXTENSIONS,
        },
        "observability": {
            "metrics": "/metrics",
            "config": "/config",
            "model_info": "/model-info",
            "eval_summary": "/eval/summary",
        },
        "controls": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
            "max_request_seconds": MAX_REQUEST_SECONDS,
            "metrics_reset_requires_token": bool(METRICS_ADMIN_TOKEN),
        },
    }

    try:
        eval_summary = _load_eval_summary()
        summary["evaluation_headline"] = {
            "num_samples": eval_summary.get("num_samples"),
            "accuracy": eval_summary.get("accuracy"),
            "f1": eval_summary.get("f1"),
            "roc_auc": eval_summary.get("roc_auc"),
            "source_path": eval_summary.get("source_path"),
        }
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        summary["evaluation_headline"] = {
            "num_samples": None,
            "accuracy": None,
            "f1": None,
            "roc_auc": None,
            "source_path": str(EVAL_SUMMARY_PATH),
        }

    return summary


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state dict")

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ("model.", "module.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)


# === Load Model ===
def load_model():
    # Keep startup offline-safe: no external downloads needed when loading local checkpoints.
    if os.environ.get("SKIP_MODEL_LOAD", "0") == "1":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.eval()
        return model, "single", "Model loading skipped (SKIP_MODEL_LOAD=1)"

    weights = None
    errors = []

    if HYBRID_MODEL_PATH.exists():
        try:
            model = HybridEfficientNet(weights=weights)
            _load_checkpoint(model, HYBRID_MODEL_PATH)
            model.eval()
            return model, "hybrid", ""
        except Exception as exc:
            errors.append(f"hybrid checkpoint load failed: {exc}")

    if LEGACY_MODEL_PATH.exists():
        model = efficientnet_b0(weights=weights)
        try:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            _load_checkpoint(model, LEGACY_MODEL_PATH)
            model.eval()
            return model, "single", ""
        except Exception as exc:
            errors.append(f"legacy checkpoint load failed: {exc}")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.eval()
    if errors:
        return model, "single", "; ".join(errors)
    return (
        model,
        "single",
        f"No model checkpoint found at {HYBRID_MODEL_PATH} or {LEGACY_MODEL_PATH}",
    )


model, model_mode, model_load_error = load_model()
if model_mode == "hybrid":
    cam_extractor = GradCAM(model, target_layer=model.rgb_backbone.features[-1])
else:
    cam_extractor = GradCAM(model, target_layer=model.features[-1])


def get_model_error_message():
    if not model_load_error:
        return ""
    return f"Model is not ready: {model_load_error}"


def hash_bytes(content):
    return hashlib.sha256(content).hexdigest()


def validate_upload(content, filename):
    if not content:
        raise ValueError("Uploaded file is empty")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"File too large ({size_mb:.2f} MB). Limit is {MAX_UPLOAD_MB} MB"
        )
    if filename and Path(filename).suffix.lower() not in set(SUPPORTED_EXTENSIONS):
        raise ValueError("Unsupported extension. Use .jpg, .jpeg, .png, .mp4, .mov")


def build_meta(start_time, filename, content):
    return {
        "request_id": str(uuid.uuid4()),
        "processing_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "app_version": APP_VERSION,
        "model_mode": model_mode,
        "model_checkpoint": str(
            HYBRID_MODEL_PATH if model_mode == "hybrid" else LEGACY_MODEL_PATH
        ),
        "filename": filename,
        "sha256": hash_bytes(content),
    }


# === Preprocessing ===
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# === Inference Logic ===
def forward_model(input_img):
    tensor = preprocess(input_img).unsqueeze(0)
    if model_mode == "hybrid":
        fft_img = fft_from_pil(input_img)
        fft_tensor = preprocess(fft_img).unsqueeze(0)
        return model(tensor, fft_tensor)
    return model(tensor)


def predict_with_cam(img):
    if model_load_error:
        raise RuntimeError(get_model_error_message())
    input_img = img.resize((224, 224))
    model.zero_grad(set_to_none=True)
    out = forward_model(input_img)
    probs = torch.softmax(out, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)
    cam = cam_extractor(pred.item(), out)[0]
    if cam.dim() == 3:
        cam = cam[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = (
        torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    cam = cam.clamp(0, 1).pow(0.5)
    red_mask = torch.zeros((3, cam.shape[0], cam.shape[1]), dtype=cam.dtype)
    red_mask[0] = cam
    overlay = Image.blend(input_img, to_pil_image(red_mask), alpha=0.6)
    label = "🟢 Real" if pred.item() == 0 else "🔴 Deepfake"
    return label, f"{conf.item() * 100:.2f}%", overlay, probs[1].item()


def predict_fake_prob(img):
    if model_load_error:
        raise RuntimeError(get_model_error_message())
    input_img = img.resize((224, 224))
    out = forward_model(input_img)
    probs = torch.softmax(out, dim=1)[0]
    return probs[1].item()


def plot_frame_probs(frame_probs):
    fig = plt.figure(figsize=(5, 3), dpi=150)
    plt.plot(frame_probs, color="red", linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel("Frame Number")
    plt.ylabel("Fake Probability")
    plt.title("Frame-Level Fake Confidence")
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_reasons(fake_prob, diffusion_score, frame_probs=None):
    reasons = []
    if fake_prob >= 0.7:
        reasons.append("High fake probability")
    elif fake_prob >= 0.5:
        reasons.append("Moderate fake probability")

    if diffusion_score >= 0.6:
        reasons.append("Diffusion texture/noise signature detected")

    if frame_probs is not None and len(frame_probs) > 1:
        if float(np.std(frame_probs)) >= 0.15:
            reasons.append("Frame-level spikes detected")

    if not reasons:
        reasons.append("No strong forensic signals detected")
    return reasons


def build_explanation_text(
    label, confidence, generation, diffusion_score, reasons, has_video
):
    sentences = []
    sentences.append(f"The model predicts {label} with confidence {confidence}.")
    if generation:
        sentences.append(f"Generation type is estimated as {generation}.")
    sentences.append(
        f"Diffusion score is {diffusion_score}, where higher values suggest diffusion artifacts."
    )
    if reasons:
        sentences.append("Key signals: " + ", ".join(reasons) + ".")
    if has_video:
        sentences.append(
            "Frame-level graph highlights how confidence changes over time."
        )
    return " ".join(sentences)


def sample_video_frames(path, max_frames=24):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = max_frames
    count = min(max_frames, total)
    indices = np.linspace(0, max(total - 1, 0), num=count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def build_verdict(fake_prob):
    if fake_prob >= 0.6:
        return "Manipulated", "status-manipulated"
    if fake_prob <= 0.4:
        return "Authentic", "status-authentic"
    return "Uncertain", "status-uncertain"


def build_status_panel(label, confidence, fake_prob):
    verdict, verdict_class = build_verdict(fake_prob)
    if "deepfake" in str(label).lower() and verdict == "Uncertain":
        verdict, verdict_class = "Manipulated", "status-manipulated"
    confidence_value = confidence or f"{fake_prob * 100:.2f}%"
    return f"""
    <div class="status-panel">
        <div class="status-item">
            <div class="status-label">AI Confidence</div>
            <div class="status-value">{confidence_value}</div>
        </div>
        <div class="status-item">
            <div class="status-label">Forensic Verdict</div>
            <div class="status-badge {verdict_class}">{verdict}</div>
        </div>
    </div>
    """


def build_map_badge(confidence, fake_prob):
    verdict, verdict_class = build_verdict(fake_prob)
    confidence_value = confidence or f"{fake_prob * 100:.2f}%"
    return f'<div class="map-score {verdict_class}">{confidence_value} {verdict}</div>'


def build_upload_feedback(file_obj):
    if file_obj is None:
        return '<div class="upload-meta idle">Upload an image or video to start forensic analysis.</div>'
    filename = Path(file_obj.name).name
    return f"""
    <div class="upload-meta ready">
        <div class="upload-file">{filename}</div>
        <div class="upload-progress"><span></span></div>
    </div>
    """


def predict_file(file_obj):
    if model_load_error:
        return (
            "❌ Model load failed",
            "",
            "",
            "",
            build_upload_feedback(file_obj),
            None,
            None,
            None,
            "",
            "",
            get_model_error_message(),
        )

    if file_obj is None:
        return (
            "⚠️ No file selected",
            "",
            "",
            "",
            build_upload_feedback(None),
            None,
            None,
            None,
            "",
            "",
            "",
        )

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
        label, confidence, overlay, fake_prob = predict_with_cam(img)
        diffusion_score = diffusion_heuristic_score(img)
        generation = classify_generation(fake_prob, diffusion_score)
        reasons = build_reasons(fake_prob, diffusion_score)
        explanation = build_explanation_text(
            label, confidence, generation, f"{diffusion_score:.2f}", reasons, False
        )
        status_panel = build_status_panel(label, confidence, fake_prob)
        map_badge = build_map_badge(confidence, fake_prob)
        upload_feedback = build_upload_feedback(file_obj)
        upload_thumb = img.resize((300, 300))
        return (
            label,
            confidence,
            status_panel,
            map_badge,
            upload_feedback,
            upload_thumb,
            overlay,
            None,
            generation,
            f"{diffusion_score:.2f}",
            explanation,
        )

    elif mime and mime.startswith("video"):
        frames = sample_video_frames(path)
        if not frames:
            return (
                "❌ Error reading video",
                "",
                "",
                "",
                build_upload_feedback(file_obj),
                None,
                None,
                None,
                "",
                "",
                "",
            )
        first_frame = frames[0]
        label, _, overlay, _ = predict_with_cam(first_frame)
        frame_probs = [predict_fake_prob(frame) for frame in frames]
        diffusion_scores = [diffusion_heuristic_score(frame) for frame in frames]
        avg_prob = float(np.mean(frame_probs))
        avg_diffusion = float(np.mean(diffusion_scores))
        if avg_prob >= 0.5:
            label = "🔴 Deepfake (avg)"
        else:
            label = "🟢 Real (avg)"
        confidence = f"{avg_prob * 100:.2f}%"
        graph = plot_frame_probs(frame_probs)
        generation = classify_generation(avg_prob, avg_diffusion)
        reasons = build_reasons(avg_prob, avg_diffusion, frame_probs=frame_probs)
        explanation = build_explanation_text(
            label, confidence, generation, f"{avg_diffusion:.2f}", reasons, True
        )
        status_panel = build_status_panel(label, confidence, avg_prob)
        map_badge = build_map_badge(confidence, avg_prob)
        upload_feedback = build_upload_feedback(file_obj)
        return (
            label,
            confidence,
            status_panel,
            map_badge,
            upload_feedback,
            first_frame.resize((300, 300)),
            overlay,
            graph,
            generation,
            f"{avg_diffusion:.2f}",
            explanation,
        )

    else:
        return (
            "Unsupported file type",
            "",
            "",
            "",
            build_upload_feedback(file_obj),
            None,
            None,
            None,
            "",
            "",
            "",
        )


LAB_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Geist:wght@400;500;600;700&display=swap');

:root {
    --bg-sand: #F7F5F2;
    --bg-sage: #EAF2EE;
    --bg-cool: #E8EEF4;
    --ink: #1D2430;
    --muted: #5C6677;
    --panel: rgba(255, 255, 255, 0.94);
    --panel-strong: rgba(255, 255, 255, 0.99);
    --line: rgba(37, 52, 80, 0.14);
    --accent: #2F8C84;
    --accent-2: #78BDAA;
    --coral: #C97A72;
    --success: #25885D;
    --danger: #C74A5D;
    --warn: #B98020;
    --shadow-soft: 0 14px 30px rgba(19, 31, 48, 0.08), 0 3px 10px rgba(19, 31, 48, 0.05);
    --shadow-hover: 0 20px 40px rgba(19, 31, 48, 0.12), 0 8px 16px rgba(19, 31, 48, 0.08);
}

html, body, .gradio-container {
    font-family: 'Inter', 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
    background:
        radial-gradient(980px 680px at 5% -8%, rgba(247, 245, 242, 0.95) 0%, transparent 58%),
        radial-gradient(920px 620px at 96% 8%, rgba(234, 242, 238, 0.92) 0%, transparent 60%),
        radial-gradient(760px 520px at 56% 92%, rgba(232, 238, 244, 0.86) 0%, transparent 64%),
        linear-gradient(132deg, #F7F5F2 0%, #EAF2EE 58%, #E8EEF4 100%);
    color: var(--ink);
}

.gradio-container {
    padding-bottom: 28px !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'%3E%3Cg fill='%23333842' fill-opacity='0.032'%3E%3Ccircle cx='8' cy='12' r='1'/%3E%3Ccircle cx='44' cy='22' r='1'/%3E%3Ccircle cx='70' cy='54' r='1'/%3E%3Ccircle cx='110' cy='30' r='1'/%3E%3Ccircle cx='20' cy='96' r='1'/%3E%3Ccircle cx='90' cy='102' r='1'/%3E%3C/g%3E%3C/svg%3E");
    opacity: 0.42;
    z-index: 0;
}

.gradio-container > * {
    position: relative;
    z-index: 1;
}

.lab-shell {
    max-width: 1240px;
    margin: 0 auto;
    padding: 16px 18px 36px;
}

.lab-hero {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(246, 250, 248, 0.86));
    border: 1px solid var(--line);
    border-radius: 22px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-soft);
    padding: 30px 32px;
    margin-bottom: 36px;
}

.lab-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 10px;
    color: var(--ink);
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
}

.lab-subtitle {
    font-size: 15px;
    line-height: 1.65;
    color: var(--muted);
    max-width: 760px;
}

.lab-top-meta {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.lab-meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: #2A3448;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(75, 98, 131, 0.2);
}

.lab-meta-chip.ok {
    color: #1c6f4d;
    border-color: rgba(36, 160, 107, 0.35);
    background: rgba(206, 244, 225, 0.58);
}

.lab-meta-chip.warn {
    color: #8c3d4a;
    border-color: rgba(216, 75, 95, 0.35);
    background: rgba(252, 219, 225, 0.62);
}

.lab-main-grid {
    display: grid !important;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 24px !important;
    align-items: start !important;
    margin-bottom: 36px;
    position: relative;
}

.lab-main-grid::after {
    content: "";
    position: absolute;
    top: 6px;
    bottom: 6px;
    left: calc(50% - 1px);
    width: 1px;
    background: linear-gradient(180deg, transparent, rgba(51, 56, 66, 0.18), transparent);
}

.lab-col-evidence {
    grid-column: span 6;
    display: grid;
    gap: 18px;
}

.lab-col-map {
    grid-column: span 6;
    display: grid;
    gap: 18px;
}

.lab-bottom-grid {
    display: grid !important;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 24px !important;
}

.lab-col-summary {
    grid-column: span 7;
}

.lab-col-temporal {
    grid-column: span 5;
}

.lab-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-soft);
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}

.lab-panel:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
    border-color: rgba(58, 80, 110, 0.18);
}

.lab-panel + .lab-panel {
    position: relative;
}

.lab-panel + .lab-panel::before {
    content: "";
    position: absolute;
    top: -10px;
    left: 6px;
    right: 6px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(70, 90, 120, 0.24), transparent);
}

.lab-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 6px 12px;
    margin-right: 10px;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6C4644;
    background: linear-gradient(145deg, rgba(201, 122, 114, 0.2), rgba(201, 122, 114, 0.13));
    border: 1px solid rgba(201, 122, 114, 0.32);
}

.section-card {
    margin-top: 14px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px;
    box-shadow: var(--shadow-soft);
}

.section-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 4px;
}

.hint-text {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 10px;
}

.lab-panel-title {
    font-size: 19px;
    font-weight: 600;
    color: var(--ink);
}

.lab-helper {
    margin-top: 14px;
    font-size: 13px;
    line-height: 1.55;
    color: var(--muted);
}

.gr-file {
    border-radius: 16px !important;
    border: 1.8px dashed rgba(47, 140, 132, 0.62) !important;
    background: rgba(255, 255, 255, 0.72) !important;
    position: relative;
    overflow: hidden;
    transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
    box-shadow: inset 0 1px 6px rgba(19, 31, 48, 0.12);
}

.gr-file::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    border-radius: 16px;
    background: radial-gradient(circle at 50% 120%, rgba(232, 138, 133, 0.28), transparent 55%);
    opacity: 0;
    transition: opacity 220ms ease;
}

.gr-file:hover,
.gr-file:focus-within,
.gr-file.dragging {
    border-color: rgba(47, 140, 132, 0.98) !important;
    box-shadow: 0 0 0 1px rgba(47, 140, 132, 0.35), 0 0 18px rgba(201, 122, 114, 0.2), inset 0 1px 8px rgba(19, 31, 48, 0.08);
    transform: translateY(-1px);
}

.gr-file.dragging {
    animation: dragPulse 900ms ease-in-out infinite;
}

.gr-file:hover::after,
.gr-file:focus-within::after,
.gr-file.dragging::after {
    opacity: 1;
}

.lab-actions button,
.gr-button-primary {
    border: 0 !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #2F8C84 0%, #78BDAA 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 22px rgba(47, 140, 132, 0.28) !important;
    transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease !important;
}

.lab-actions {
    margin-top: 6px;
    gap: 10px !important;
}

.lab-actions .secondary {
    border: 1px solid rgba(72, 90, 126, 0.24) !important;
    background: rgba(255, 255, 255, 0.84) !important;
    color: #25324a !important;
    box-shadow: none !important;
}

.lab-actions .secondary:hover {
    border-color: rgba(72, 90, 126, 0.38) !important;
    box-shadow: 0 10px 22px rgba(31, 45, 68, 0.12) !important;
    transform: translateY(-1px);
}

.lab-actions button:hover,
.gr-button-primary:hover {
    filter: brightness(1.07);
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 0 0 1px rgba(47, 140, 132, 0.52), 0 14px 30px rgba(47, 140, 132, 0.3) !important;
}

.upload-meta {
    margin-top: 14px;
    border-radius: 14px;
    padding: 10px 12px;
    border: 1px solid rgba(58, 76, 108, 0.15);
    background: rgba(255, 255, 255, 0.5);
    color: #42506a;
    font-size: 12px;
}

.upload-file {
    font-weight: 600;
    color: #2d3b58;
    margin-bottom: 8px;
}

.upload-progress {
    height: 7px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(69, 88, 126, 0.14);
}

.upload-progress span {
    display: block;
    width: 100%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #2F8C84 0%, #78BDAA 65%, #C97A72 100%);
    transform-origin: left center;
    animation: uploadFill 1.15s ease-out;
}

.upload-thumb .gr-image {
    max-height: 180px;
}

.gr-textbox input,
.gr-textbox textarea {
    background: rgba(248, 251, 255, 0.85) !important;
    border: 1px solid rgba(65, 84, 118, 0.16) !important;
    border-radius: 14px !important;
    color: #1c2432 !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    padding-top: 8px;
    padding-bottom: 8px;
}

.gr-textbox textarea::placeholder,
.gr-textbox input::placeholder {
    color: #8a95aa !important;
}

label,
.gr-label,
.gradio-container .block-title,
.gradio-container .block-info {
    color: var(--ink) !important;
    font-weight: 600 !important;
}

.lab-preview-panel {
    position: relative;
    overflow: hidden;
}

.lab-preview-panel::after {
    content: "";
    position: absolute;
    inset: 14px;
    border-radius: 14px;
    pointer-events: none;
    border: 1px solid rgba(119, 138, 183, 0.28);
    background-image:
        linear-gradient(rgba(109, 132, 188, 0.09) 1px, transparent 1px),
        linear-gradient(90deg, rgba(109, 132, 188, 0.09) 1px, transparent 1px);
    background-size: 22px 22px;
    mix-blend-mode: multiply;
}

.lab-preview-panel::before {
    content: "";
    position: absolute;
    top: 16px;
    left: 16px;
    right: 16px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent 0%, rgba(76, 181, 174, 0.9) 48%, transparent 100%);
    animation: scanMove 2.8s linear infinite;
    z-index: 2;
    pointer-events: none;
}

.lab-preview-image {
    position: relative;
    z-index: 3;
}

.map-score {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 8px 14px;
    margin-bottom: 10px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #ffffff;
}

.map-score.status-authentic {
    background: linear-gradient(135deg, #24A06B 0%, #41BF87 100%);
}

.map-score.status-manipulated {
    background: linear-gradient(135deg, #D84B5F 0%, #E88A85 100%);
}

.map-score.status-uncertain {
    background: linear-gradient(135deg, #c7851d 0%, #e6a43a 100%);
}

.gr-image {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(106, 126, 166, 0.2) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: inset 0 1px 8px rgba(19, 31, 48, 0.12);
}

.status-panel {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}

.status-item {
    border-radius: 14px;
    padding: 12px 14px;
    border: 1px solid rgba(59, 80, 119, 0.16);
    background: rgba(255, 255, 255, 0.72);
}

.status-label {
    color: #6f7c93;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.status-value {
    color: #1a2538;
    font-size: 20px;
    font-weight: 700;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    color: #ffffff;
}

.status-authentic {
    background: linear-gradient(135deg, rgba(69, 212, 131, 0.95), rgba(64, 181, 125, 0.95));
}

.status-manipulated {
    background: linear-gradient(135deg, rgba(255, 106, 122, 0.95), rgba(233, 87, 118, 0.95));
}

.status-uncertain {
    background: linear-gradient(135deg, rgba(209, 138, 31, 0.95), rgba(230, 164, 58, 0.95));
}

.result-fade {
    animation: fadeResult 300ms ease;
}

@keyframes dashPulse {
    0%, 100% {
        border-color: rgba(141, 163, 255, 0.58);
    }
    50% {
        border-color: rgba(141, 163, 255, 0.92);
    }
}

@keyframes dragPulse {
    0%, 100% {
        transform: translateY(-1px) scale(1);
    }
    50% {
        transform: translateY(-1px) scale(1.02);
    }
}

@keyframes uploadFill {
    from {
        transform: scaleX(0);
    }
    to {
        transform: scaleX(1);
    }
}

@keyframes scanMove {
    0% {
        top: 18px;
        opacity: 0.2;
    }
    50% {
        opacity: 0.9;
    }
    100% {
        top: calc(100% - 18px);
        opacity: 0.2;
    }
}

@keyframes fadeResult {
    from {
        opacity: 0;
        transform: translateY(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 1040px) {
        .lab-main-grid::after {
            display: none;
        }
    .lab-main-grid {
        grid-template-columns: 1fr;
    }

    .lab-bottom-grid {
        grid-template-columns: 1fr;
    }

    .lab-col-evidence,
    .lab-col-map,
    .lab-col-summary,
    .lab-col-temporal {
        grid-column: auto;
    }
}

@media (max-width: 720px) {
    .lab-hero {
        padding: 22px 18px;
        border-radius: 20px;
    }

    .lab-title {
        font-size: 28px;
    }

    .lab-subtitle {
        font-size: 14px;
    }

    .lab-panel {
        padding: 22px;
        border-radius: 16px;
    }

    .status-panel {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks(title="Deepfake Detector", css=LAB_CSS) as demo:
    gr.Markdown(
        """
            <div class="lab-hero">
                <div class="lab-hero-main">
                    <div class="lab-title">Deepfake Detector Lab</div>
                    <div class="lab-subtitle">Upload an image or video and get a clean forensic verdict, confidence, visual evidence, and a plain-language summary.</div>
                </div>
            </div>
        """,
        elem_id="lab-hero",
    )
    gr.HTML(
        f"""
        <div class="lab-top-meta">
            <div class="lab-meta-chip {"ok" if not model_load_error else "warn"}">Model: {"Ready" if not model_load_error else "Degraded"}</div>
            <div class="lab-meta-chip">Mode: {model_mode.title()}</div>
            <div class="lab-meta-chip">Max Upload: {MAX_UPLOAD_MB} MB</div>
            <div class="lab-meta-chip">Version: {APP_VERSION}</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.HTML(
                '<div class="section-card"><div class="section-title">1) Upload Media</div><div class="hint-text">Supported: JPG, PNG, MP4, MOV. For videos, multiple frames are sampled automatically.</div></div>'
            )
            file_input = gr.File(
                label="Drag & drop image or video",
                file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
            )
            upload_feedback = gr.HTML(build_upload_feedback(None))
            with gr.Row(elem_classes=["lab-actions"]):
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear", elem_classes=["secondary"])
            upload_thumb = gr.Image(
                label="Upload Preview",
                interactive=False,
                elem_classes=["upload-thumb", "result-fade"],
            )

        with gr.Column(scale=7):
            gr.HTML(
                '<div class="section-card"><div class="section-title">2) Forensic Result</div><div class="hint-text">Confidence and verdict are shown together for quick interpretation.</div></div>'
            )
            prediction = gr.Textbox(label="Prediction", interactive=False)
            confidence = gr.Textbox(label="Confidence (%)", interactive=False)
            status_panel = gr.HTML(value="")
            map_score = gr.HTML(value="")
            preview = gr.Image(
                label="Heatmap Preview (Grad-CAM)",
                interactive=False,
                elem_classes=["lab-preview-image", "result-fade"],
            )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            generation = gr.Textbox(label="Generation Type", interactive=False)
        with gr.Column(scale=4):
            diffusion_score = gr.Textbox(label="Diffusion Score", interactive=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            explanation = gr.Textbox(
                label="Readable Explanation", lines=4, interactive=False
            )
        with gr.Column(scale=5):
            graph = gr.Image(label="Frame Confidence Graph", interactive=False)

    # Gradio event bindings
    def handle_input(file_obj):
        return predict_file(file_obj)

    def clear_outputs():
        return (
            None,
            "",
            "",
            "",
            "",
            build_upload_feedback(None),
            None,
            None,
            None,
            "",
            "",
            "",
        )

    def preview_file(file_obj):
        if file_obj is None:
            return build_upload_feedback(None), None
        path = file_obj.name
        mime, _ = mimetypes.guess_type(path)
        if mime and mime.startswith("image"):
            try:
                img = Image.open(path).convert("RGB")
                return build_upload_feedback(file_obj), img.resize((300, 300))
            except Exception:
                return build_upload_feedback(file_obj), None
        return build_upload_feedback(file_obj), None

    file_input.change(
        fn=preview_file,
        inputs=file_input,
        outputs=[upload_feedback, upload_thumb],
    )

    analyze_btn.click(
        fn=handle_input,
        inputs=file_input,
        outputs=[
            prediction,
            confidence,
            status_panel,
            map_score,
            upload_feedback,
            upload_thumb,
            preview,
            graph,
            generation,
            diffusion_score,
            explanation,
        ],
    )

    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            file_input,
            prediction,
            confidence,
            status_panel,
            map_score,
            upload_feedback,
            upload_thumb,
            preview,
            graph,
            generation,
            diffusion_score,
            explanation,
        ],
    )

app = FastAPI()


@app.get("/health")
async def health():
    start_time = time.perf_counter()
    payload = {
        "status": "ok" if not model_load_error else "degraded",
        "model_ready": not bool(model_load_error),
        "model_mode": model_mode,
        "app_version": APP_VERSION,
        "max_upload_mb": MAX_UPLOAD_MB,
        "error": model_load_error or "",
    }
    _update_api_metrics(
        endpoint="/health",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return payload


@app.get("/live")
async def live():
    start_time = time.perf_counter()
    payload = {
        "status": "alive",
        "app_version": APP_VERSION,
        "uptime_sec": round(time.time() - APP_START_TIME, 2),
    }
    _update_api_metrics(
        endpoint="/live",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return payload


@app.get("/ready")
async def ready():
    start_time = time.perf_counter()
    model_ready = not bool(model_load_error)
    payload = {
        "status": "ready" if model_ready else "not_ready",
        "model_ready": model_ready,
        "model_mode": model_mode,
        "app_version": APP_VERSION,
        "error": model_load_error or "",
    }
    _update_api_metrics(
        endpoint="/ready",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=model_ready,
        items_processed=0,
    )
    if not model_ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/model-info")
async def model_info():
    start_time = time.perf_counter()
    payload = {
        "model_mode": model_mode,
        "hybrid_checkpoint": str(HYBRID_MODEL_PATH),
        "legacy_checkpoint": str(LEGACY_MODEL_PATH),
        "loaded_checkpoint": str(
            HYBRID_MODEL_PATH if model_mode == "hybrid" else LEGACY_MODEL_PATH
        ),
        "model_ready": not bool(model_load_error),
        "load_error": model_load_error or "",
        "app_version": APP_VERSION,
    }
    _update_api_metrics(
        endpoint="/model-info",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return payload


@app.get("/metrics")
async def metrics():
    start_time = time.perf_counter()
    _update_api_metrics(
        endpoint="/metrics",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return _build_metrics_response()


@app.get("/config")
async def config_info():
    start_time = time.perf_counter()
    payload = {
        "app_version": APP_VERSION,
        "model_mode": model_mode,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_batch_files": MAX_BATCH_FILES,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "max_request_seconds": MAX_REQUEST_SECONDS,
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "metrics_reset_enabled": bool(METRICS_ADMIN_TOKEN),
        "eval_summary_path": str(EVAL_SUMMARY_PATH),
    }
    _update_api_metrics(
        endpoint="/config",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return payload


@app.get("/capabilities")
async def capabilities():
    start_time = time.perf_counter()
    payload = _build_capabilities_summary()
    _update_api_metrics(
        endpoint="/capabilities",
        duration_ms=(time.perf_counter() - start_time) * 1000,
        success=True,
        items_processed=0,
    )
    return payload


@app.post("/metrics/reset")
async def reset_metrics(x_admin_token: str | None = Header(default=None)):
    if not METRICS_ADMIN_TOKEN:
        return _api_error(
            status_code=403,
            message="Metrics reset is disabled on this deployment",
            error_code="METRICS_RESET_DISABLED",
        )
    if x_admin_token != METRICS_ADMIN_TOKEN:
        return _api_error(
            status_code=403,
            message="Invalid admin token",
            error_code="INVALID_ADMIN_TOKEN",
        )

    _reset_api_metrics()
    return {
        "status": "ok",
        "message": "Metrics reset successfully",
    }


@app.get("/eval/summary")
async def eval_summary():
    start_time = time.perf_counter()
    try:
        payload = _load_eval_summary()
        _update_api_metrics(
            endpoint="/eval/summary",
            duration_ms=(time.perf_counter() - start_time) * 1000,
            success=True,
            items_processed=0,
        )
        return payload
    except FileNotFoundError as exc:
        _update_api_metrics(
            endpoint="/eval/summary",
            duration_ms=(time.perf_counter() - start_time) * 1000,
            success=False,
            items_processed=0,
        )
        return _api_error(
            status_code=404,
            message=str(exc),
            error_code="EVAL_SUMMARY_NOT_FOUND",
        )
    except (ValueError, json.JSONDecodeError) as exc:
        _update_api_metrics(
            endpoint="/eval/summary",
            duration_ms=(time.perf_counter() - start_time) * 1000,
            success=False,
            items_processed=0,
        )
        return _api_error(
            status_code=500,
            message=f"Invalid evaluation summary: {exc}",
            error_code="EVAL_SUMMARY_INVALID",
        )


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    start_time = time.perf_counter()
    success = False
    manipulated = 0
    authentic = 0
    uncertain = 0
    result = {"error": "Unknown error"}
    tmp_path = ""
    content = b""

    try:
        _enforce_rate_limit(
            endpoint="/detect",
            client_key=request.client.host if request.client else "unknown",
        )
        if model_load_error:
            result = _api_error(
                status_code=503,
                message=get_model_error_message(),
                error_code="MODEL_NOT_READY",
            )
            return result

        content = await file.read()
        _ensure_not_timed_out(start_time)
        validate_upload(content, file.filename)
        suffix = Path(file.filename or "").suffix.lower()
        if not suffix:
            suffix = ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        mime, _ = mimetypes.guess_type(tmp_path)
        if mime and mime.startswith("image"):
            img = Image.open(tmp_path).convert("RGB")
            _ensure_not_timed_out(start_time)
            label, confidence, overlay, fake_prob = predict_with_cam(img)
            diffusion_score = diffusion_heuristic_score(img)
            generation = classify_generation(fake_prob, diffusion_score)
            reasons = build_reasons(fake_prob, diffusion_score)
            result = {
                "type": "image",
                "prediction": label,
                "confidence": round(fake_prob, 4),
                "generation": generation,
                "diffusion_score": round(diffusion_score, 4),
                "reasons": reasons,
                "heatmap": image_to_base64(overlay),
                "meta": build_meta(start_time, file.filename, content),
            }
            manipulated, authentic, uncertain = _verdict_counts_from_prediction(label)
            success = True
            return result

        if mime and mime.startswith("video"):
            frames = sample_video_frames(tmp_path)
            _ensure_not_timed_out(start_time)
            if not frames:
                result = _api_error(
                    status_code=422,
                    message="Could not read video",
                    error_code="VIDEO_DECODE_FAILED",
                )
                return result
            frame_probs = [predict_fake_prob(frame) for frame in frames]
            _ensure_not_timed_out(start_time)
            diffusion_scores = [diffusion_heuristic_score(frame) for frame in frames]
            avg_prob = float(np.mean(frame_probs))
            avg_diffusion = float(np.mean(diffusion_scores))
            generation = classify_generation(avg_prob, avg_diffusion)
            reasons = build_reasons(avg_prob, avg_diffusion, frame_probs=frame_probs)
            overlay = predict_with_cam(frames[0])[2]
            graph = plot_frame_probs(frame_probs)
            prediction = "🔴 Deepfake (avg)" if avg_prob >= 0.5 else "🟢 Real (avg)"
            result = {
                "type": "video",
                "prediction": prediction,
                "confidence": round(avg_prob, 4),
                "generation": generation,
                "diffusion_score": round(avg_diffusion, 4),
                "reasons": reasons,
                "frame_probs": [round(p, 4) for p in frame_probs],
                "heatmap": image_to_base64(overlay),
                "frame_graph": image_to_base64(graph),
                "meta": build_meta(start_time, file.filename, content),
            }
            manipulated, authentic, uncertain = _verdict_counts_from_prediction(
                prediction
            )
            success = True
            return result

        result = _api_error(
            status_code=415,
            message="Unsupported file type",
            error_code="UNSUPPORTED_FILE_TYPE",
        )
        return result
    except ValueError as exc:
        result = _api_error(
            status_code=400,
            message=str(exc),
            error_code="VALIDATION_ERROR",
        )
        return result
    except PermissionError as exc:
        result = _api_error(
            status_code=429,
            message=str(exc),
            error_code="RATE_LIMIT_EXCEEDED",
        )
        return result
    except TimeoutError as exc:
        result = _api_error(
            status_code=408,
            message=str(exc),
            error_code="REQUEST_TIMEOUT",
        )
        return result
    except Exception as exc:
        result = _api_error(
            status_code=500,
            message=f"Inference failed: {exc}",
            error_code="INFERENCE_ERROR",
        )
        return result
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        _update_api_metrics(
            endpoint="/detect",
            duration_ms=(time.perf_counter() - start_time) * 1000,
            success=success,
            items_processed=1,
            manipulated=manipulated,
            authentic=authentic,
            uncertain=uncertain,
        )


@app.post("/detect/batch")
async def detect_batch(request: Request, files: list[UploadFile] = File(...)):
    request_start_time = time.perf_counter()
    try:
        _enforce_rate_limit(
            endpoint="/detect/batch",
            client_key=request.client.host if request.client else "unknown",
        )
    except PermissionError as exc:
        payload = _api_error(
            status_code=429,
            message=str(exc),
            error_code="RATE_LIMIT_EXCEEDED",
        )
        _update_api_metrics(
            endpoint="/detect/batch",
            duration_ms=(time.perf_counter() - request_start_time) * 1000,
            success=False,
            items_processed=0,
        )
        return payload

    if model_load_error:
        payload = _api_error(
            status_code=503,
            message=get_model_error_message(),
            error_code="MODEL_NOT_READY",
        )
        _update_api_metrics(
            endpoint="/detect/batch",
            duration_ms=(time.perf_counter() - request_start_time) * 1000,
            success=False,
            items_processed=0,
        )
        return payload

    if len(files) > MAX_BATCH_FILES:
        payload = _api_error(
            status_code=400,
            message=f"Batch limit exceeded. Maximum {MAX_BATCH_FILES} files per request.",
            error_code="BATCH_LIMIT_EXCEEDED",
        )
        _update_api_metrics(
            endpoint="/detect/batch",
            duration_ms=(time.perf_counter() - request_start_time) * 1000,
            success=False,
            items_processed=len(files),
        )
        return payload

    results = []
    try:
        for file in files:
            item_start_time = time.perf_counter()
            _ensure_not_timed_out(request_start_time)
            content = await file.read()
            try:
                validate_upload(content, file.filename)
            except ValueError as exc:
                results.append(
                    {
                        "filename": file.filename,
                        "error": str(exc),
                        "meta": build_meta(item_start_time, file.filename, content),
                    }
                )
                continue

            suffix = Path(file.filename or "").suffix.lower() or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                mime, _ = mimetypes.guess_type(tmp_path)
                if mime and mime.startswith("image"):
                    _ensure_not_timed_out(request_start_time)
                    img = Image.open(tmp_path).convert("RGB")
                    label, _, _, fake_prob = predict_with_cam(img)
                    diffusion_score = diffusion_heuristic_score(img)
                    generation = classify_generation(fake_prob, diffusion_score)
                    reasons = build_reasons(fake_prob, diffusion_score)
                    results.append(
                        {
                            "filename": file.filename,
                            "type": "image",
                            "prediction": label,
                            "confidence": round(fake_prob, 4),
                            "generation": generation,
                            "diffusion_score": round(diffusion_score, 4),
                            "reasons": reasons,
                            "meta": build_meta(item_start_time, file.filename, content),
                        }
                    )
                elif mime and mime.startswith("video"):
                    _ensure_not_timed_out(request_start_time)
                    frames = sample_video_frames(tmp_path)
                    if not frames:
                        results.append(
                            {
                                "filename": file.filename,
                                "error": "Could not read video",
                                "meta": build_meta(
                                    item_start_time, file.filename, content
                                ),
                            }
                        )
                        continue

                    frame_probs = [predict_fake_prob(frame) for frame in frames]
                    _ensure_not_timed_out(request_start_time)
                    diffusion_scores = [
                        diffusion_heuristic_score(frame) for frame in frames
                    ]
                    avg_prob = float(np.mean(frame_probs))
                    avg_diffusion = float(np.mean(diffusion_scores))
                    generation = classify_generation(avg_prob, avg_diffusion)
                    reasons = build_reasons(
                        avg_prob, avg_diffusion, frame_probs=frame_probs
                    )
                    results.append(
                        {
                            "filename": file.filename,
                            "type": "video",
                            "prediction": "🔴 Deepfake (avg)"
                            if avg_prob >= 0.5
                            else "🟢 Real (avg)",
                            "confidence": round(avg_prob, 4),
                            "generation": generation,
                            "diffusion_score": round(avg_diffusion, 4),
                            "reasons": reasons,
                            "meta": build_meta(item_start_time, file.filename, content),
                        }
                    )
                else:
                    results.append(
                        {
                            "filename": file.filename,
                            "error": "Unsupported file type",
                            "meta": build_meta(item_start_time, file.filename, content),
                        }
                    )
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except TimeoutError as exc:
        payload = _api_error(
            status_code=408,
            message=str(exc),
            error_code="REQUEST_TIMEOUT",
        )
        _update_api_metrics(
            endpoint="/detect/batch",
            duration_ms=(time.perf_counter() - request_start_time) * 1000,
            success=False,
            items_processed=len(results),
        )
        return payload

    manipulated = sum(
        1
        for item in results
        if "prediction" in item and "Deepfake" in item["prediction"]
    )
    authentic = sum(
        1 for item in results if "prediction" in item and "Real" in item["prediction"]
    )
    failed = sum(1 for item in results if "error" in item)

    payload = {
        "count": len(results),
        "summary": {
            "manipulated": manipulated,
            "authentic": authentic,
            "failed": failed,
        },
        "results": results,
    }
    _update_api_metrics(
        endpoint="/detect/batch",
        duration_ms=(time.perf_counter() - request_start_time) * 1000,
        success=(failed == 0),
        items_processed=len(results),
        manipulated=manipulated,
        authentic=authentic,
        uncertain=max(len(results) - manipulated - authentic - failed, 0),
    )
    return payload


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
