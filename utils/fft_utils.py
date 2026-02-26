import numpy as np
import cv2
from PIL import Image


def fft_from_pil(image):
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude = magnitude - magnitude.min()
    magnitude = magnitude / (magnitude.max() + 1e-8)
    magnitude = (magnitude * 255).astype(np.uint8)
    return Image.fromarray(magnitude, mode="L").convert("RGB")
