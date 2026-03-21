import os
import io
import base64
import tempfile
import time
import uuid
import hashlib
from pathlib import Path
import gradio as gr
import torch
import mimetypes
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
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
    if filename and Path(filename).suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
        ".mov",
    }:
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
    --bg-sand: #FFF9F4;
    --bg-sage: #E6F3F1;
    --bg-cool: #EEF2F3;
    --ink: #333842;
    --muted: #6B7280;
    --panel: rgba(255, 255, 255, 0.94);
    --panel-strong: rgba(255, 255, 255, 0.98);
    --line: rgba(51, 56, 66, 0.12);
    --accent: #4CB5AE;
    --accent-2: #7DD1CA;
    --coral: #E88A85;
    --success: #24a06b;
    --danger: #d84b5f;
    --warn: #d18a1f;
    --shadow-soft: 0 14px 30px rgba(0, 0, 0, 0.05), 0 3px 10px rgba(0, 0, 0, 0.04);
    --shadow-hover: 0 18px 36px rgba(0, 0, 0, 0.08), 0 8px 16px rgba(0, 0, 0, 0.06);
}

html, body, .gradio-container {
    font-family: 'Inter', 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
    background:
        radial-gradient(980px 680px at 5% -8%, rgba(255, 249, 244, 0.95) 0%, transparent 58%),
        radial-gradient(920px 620px at 96% 8%, rgba(230, 243, 241, 0.92) 0%, transparent 60%),
        radial-gradient(760px 520px at 56% 92%, rgba(238, 242, 243, 0.82) 0%, transparent 64%),
        linear-gradient(132deg, #FFF9F4 0%, #E6F3F1 60%, #EEF2F3 100%);
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
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(245, 252, 250, 0.82));
    border: 1px solid var(--line);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-soft);
    padding: 30px 32px;
    margin-bottom: 36px;
}

.lab-title {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 10px;
    color: var(--ink);
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
}

.lab-subtitle {
    font-size: 15px;
    line-height: 1.5;
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
    color: #2e384d;
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(75, 98, 131, 0.18);
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
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-soft);
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}

.lab-panel:hover {
    transform: scale(1.01) translateY(-2px);
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
    color: #5a3b3a;
    background: linear-gradient(145deg, rgba(232, 138, 133, 0.22), rgba(232, 138, 133, 0.14));
    border: 1px solid rgba(232, 138, 133, 0.35);
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
    border: 1.8px dashed rgba(76, 181, 174, 0.64) !important;
    background: #22252A !important;
    position: relative;
    overflow: hidden;
    transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
    box-shadow: inset 0 1px 8px rgba(0, 0, 0, 0.3);
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
    border-color: rgba(76, 181, 174, 0.98) !important;
    box-shadow: 0 0 0 1px rgba(76, 181, 174, 0.42), 0 0 22px rgba(232, 138, 133, 0.28), inset 0 1px 8px rgba(0, 0, 0, 0.3);
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
    background: linear-gradient(135deg, #4CB5AE 0%, #7DD1CA 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 22px rgba(76, 181, 174, 0.28) !important;
    transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease !important;
}

.lab-actions {
    margin-top: 6px;
    gap: 10px !important;
}

.lab-actions .secondary {
    border: 1px solid rgba(72, 90, 126, 0.22) !important;
    background: rgba(255, 255, 255, 0.72) !important;
    color: #2a3347 !important;
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
    transform: translateY(-1px) scale(1.05);
    box-shadow: 0 0 0 1px rgba(76, 181, 174, 0.55), 0 14px 30px rgba(76, 181, 174, 0.34) !important;
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
    background: linear-gradient(90deg, #4CB5AE 0%, #7DD1CA 65%, #E88A85 100%);
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
    border: 1px solid rgba(106, 126, 166, 0.22) !important;
    background: #22252A !important;
    box-shadow: inset 0 1px 8px rgba(0, 0, 0, 0.28);
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
    border: 1px solid rgba(59, 80, 119, 0.15);
    background: rgba(255, 255, 255, 0.58);
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
    with gr.Column(elem_classes=["lab-shell"]):
        gr.Markdown(
            """
                        <div class="lab-hero">
                            <div class="lab-hero-main">
                                <div class="lab-title">Deepfake Detector Lab</div>
                                <div class="lab-subtitle">Drop an image or video to see authenticity, heatmap evidence, and a plain-language summary.</div>
                            </div>
                        </div>
                        """,
            elem_id="lab-hero",
        )

        gr.HTML(
            f"""
            <div class="lab-top-meta">
                <div class="lab-meta-chip {'ok' if not model_load_error else 'warn'}">Model: {'Ready' if not model_load_error else 'Degraded'}</div>
                <div class="lab-meta-chip">Mode: {model_mode.title()}</div>
                <div class="lab-meta-chip">Max Upload: {MAX_UPLOAD_MB} MB</div>
                <div class="lab-meta-chip">Version: {APP_VERSION}</div>
            </div>
            """
        )

        with gr.Row(equal_height=True, elem_classes=["lab-main-grid"]):
            with gr.Column(scale=5, min_width=320, elem_classes=["lab-col-evidence"]):
                with gr.Column(elem_classes=["lab-panel"]):
                    gr.Markdown(
                        '<span class="lab-badge">Upload</span><span class="lab-panel-title">Evidence Input</span>'
                    )
                    file_input = gr.File(
                        label="Drag & drop image or video",
                        file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
                    )
                    gr.Markdown(
                        '<div class="lab-helper">Accepted: JPG, PNG, MP4, MOV. Best results with faces >= 256px.</div>'
                    )
                    upload_feedback = gr.HTML(build_upload_feedback(None))
                    upload_thumb = gr.Image(
                        label="Upload Preview",
                        interactive=False,
                        elem_classes=["upload-thumb", "result-fade"],
                    )
                    with gr.Row(elem_classes=["lab-actions"]):
                        analyze_btn = gr.Button("Analyze Evidence", variant="primary")
                        clear_btn = gr.ClearButton("Clear", variant="secondary")

            with gr.Column(scale=7, min_width=360, elem_classes=["lab-col-map"]):
                with gr.Column(elem_classes=["lab-panel", "lab-preview-panel"]):
                    gr.Markdown(
                        '<span class="lab-badge">Visuals</span><span class="lab-panel-title">Evidence Map</span>'
                    )
                    map_score = gr.HTML(value="")
                    preview = gr.Image(
                        label="Preview (Red Grad-CAM)",
                        interactive=False,
                        elem_classes=["lab-preview-image", "result-fade"],
                    )

        with gr.Row(equal_height=True, elem_classes=["lab-bottom-grid"]):
            with gr.Column(scale=7, min_width=340, elem_classes=["lab-col-summary"]):
                with gr.Column(
                    elem_classes=["lab-panel", "lab-metrics", "result-fade"]
                ):
                    gr.Markdown(
                        '<span class="lab-badge">Summary</span><span class="lab-panel-title">Decision</span>'
                    )
                    status_panel = gr.HTML(value="")
                    with gr.Row():
                        prediction = gr.Textbox(label="Prediction", interactive=False)
                        confidence = gr.Textbox(
                            label="Confidence (%)", interactive=False
                        )

                    with gr.Row():
                        generation = gr.Textbox(
                            label="Generation Type", interactive=False
                        )
                        diffusion_score = gr.Textbox(
                            label="Diffusion Score", interactive=False
                        )

                    explanation = gr.Textbox(
                        label="Explanation", lines=4, interactive=False
                    )

            with gr.Column(scale=5, min_width=340, elem_classes=["lab-col-temporal"]):
                with gr.Column(elem_classes=["lab-panel", "result-fade"]):
                    gr.Markdown(
                        '<span class="lab-badge">Temporal</span><span class="lab-panel-title">Frame Graph</span>'
                    )
                    graph = gr.Image(
                        label="Frame-Level Fake Confidence", interactive=False
                    )

        def handle_input(file_obj):
            return predict_file(file_obj)

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

        clear_btn.add(
            [
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
            ]
        )

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "status": "ok" if not model_load_error else "degraded",
        "model_ready": not bool(model_load_error),
        "model_mode": model_mode,
        "app_version": APP_VERSION,
        "max_upload_mb": MAX_UPLOAD_MB,
        "error": model_load_error or "",
    }


@app.get("/model-info")
async def model_info():
    return {
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


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    if model_load_error:
        return {"error": get_model_error_message()}

    content = await file.read()
    validate_upload(content, file.filename)
    suffix = Path(file.filename or "").suffix.lower()
    if not suffix:
        suffix = ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        mime, _ = mimetypes.guess_type(tmp_path)
        if mime and mime.startswith("image"):
            img = Image.open(tmp_path).convert("RGB")
            label, confidence, overlay, fake_prob = predict_with_cam(img)
            diffusion_score = diffusion_heuristic_score(img)
            generation = classify_generation(fake_prob, diffusion_score)
            reasons = build_reasons(fake_prob, diffusion_score)
            return {
                "type": "image",
                "prediction": label,
                "confidence": round(fake_prob, 4),
                "generation": generation,
                "diffusion_score": round(diffusion_score, 4),
                "reasons": reasons,
                "heatmap": image_to_base64(overlay),
                "meta": build_meta(start_time, file.filename, content),
            }

        if mime and mime.startswith("video"):
            frames = sample_video_frames(tmp_path)
            if not frames:
                return {"error": "Could not read video"}
            frame_probs = [predict_fake_prob(frame) for frame in frames]
            diffusion_scores = [diffusion_heuristic_score(frame) for frame in frames]
            avg_prob = float(np.mean(frame_probs))
            avg_diffusion = float(np.mean(diffusion_scores))
            generation = classify_generation(avg_prob, avg_diffusion)
            reasons = build_reasons(avg_prob, avg_diffusion, frame_probs=frame_probs)
            overlay = predict_with_cam(frames[0])[2]
            graph = plot_frame_probs(frame_probs)
            return {
                "type": "video",
                "prediction": "🔴 Deepfake (avg)"
                if avg_prob >= 0.5
                else "🟢 Real (avg)",
                "confidence": round(avg_prob, 4),
                "generation": generation,
                "diffusion_score": round(avg_diffusion, 4),
                "reasons": reasons,
                "frame_probs": [round(p, 4) for p in frame_probs],
                "heatmap": image_to_base64(overlay),
                "frame_graph": image_to_base64(graph),
                "meta": build_meta(start_time, file.filename, content),
            }

        return {"error": "Unsupported file type"}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    if model_load_error:
        return {"error": get_model_error_message()}

    if len(files) > 10:
        return {"error": "Batch limit exceeded. Maximum 10 files per request."}

    results = []
    for file in files:
        start_time = time.perf_counter()
        content = await file.read()
        try:
            validate_upload(content, file.filename)
        except ValueError as exc:
            results.append(
                {
                    "filename": file.filename,
                    "error": str(exc),
                    "meta": build_meta(start_time, file.filename, content),
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
                        "meta": build_meta(start_time, file.filename, content),
                    }
                )
            elif mime and mime.startswith("video"):
                frames = sample_video_frames(tmp_path)
                if not frames:
                    results.append(
                        {
                            "filename": file.filename,
                            "error": "Could not read video",
                            "meta": build_meta(start_time, file.filename, content),
                        }
                    )
                    continue

                frame_probs = [predict_fake_prob(frame) for frame in frames]
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
                        "meta": build_meta(start_time, file.filename, content),
                    }
                )
            else:
                results.append(
                    {
                        "filename": file.filename,
                        "error": "Unsupported file type",
                        "meta": build_meta(start_time, file.filename, content),
                    }
                )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    manipulated = sum(
        1
        for item in results
        if "prediction" in item and "Deepfake" in item["prediction"]
    )
    authentic = sum(
        1 for item in results if "prediction" in item and "Real" in item["prediction"]
    )
    failed = sum(1 for item in results if "error" in item)

    return {
        "count": len(results),
        "summary": {
            "manipulated": manipulated,
            "authentic": authentic,
            "failed": failed,
        },
        "results": results,
    }


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
