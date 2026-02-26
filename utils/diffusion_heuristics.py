import numpy as np
import cv2
from PIL import Image


def _normalize_score(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def diffusion_heuristic_score(image):
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    low_texture = 1.0 - _normalize_score(lap_var, 50.0, 500.0)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    noise = gray - blurred
    noise_energy = noise * noise
    local_energy = cv2.blur(noise_energy, (7, 7))
    uniformity = 1.0 - _normalize_score(local_energy.std(), 0.0, 12.0)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = radius.max() + 1e-8
    high_mask = radius > (0.55 * max_r)
    total_energy = magnitude.sum() + 1e-8
    high_energy = magnitude[high_mask].sum() / total_energy
    low_high_freq = 1.0 - _normalize_score(high_energy, 0.15, 0.55)

    score = 0.45 * low_texture + 0.35 * uniformity + 0.20 * low_high_freq
    return float(np.clip(score, 0.0, 1.0))


def classify_generation(fake_prob, diffusion_score, diffusion_threshold=0.6):
    if diffusion_score >= diffusion_threshold:
        return "DIFFUSION FAKE"
    if fake_prob >= 0.5:
        return "GAN FAKE"
    return "REAL"
