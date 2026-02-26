import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

from lightning_modules.detector import HybridEfficientNet
from utils.diffusion_heuristics import diffusion_heuristic_score, classify_generation
from utils.fft_utils import fft_from_pil

# 🔄 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = HybridEfficientNet(weights=weights)
model.load_state_dict(torch.load("models/best_model-hybrid.pt", map_location=device))
model = model.to(device)
model.eval()

# 📦 Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 🎥 Extract N frames
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes = np.linspace(0, total - 1, num=num_frames, dtype=int)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indexes:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


# 🔍 Predict
def predict_video(video_path):
    frames = extract_frames(video_path)
    all_probs = []
    diffusion_scores = []
    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            fft_tensor = transform(fft_from_pil(frame)).unsqueeze(0).to(device)
            out = model(input_tensor, fft_tensor)
            prob = torch.softmax(out, dim=1)
            all_probs.append(prob.cpu())
            diffusion_scores.append(diffusion_heuristic_score(frame))
    avg_prob = torch.mean(torch.stack(all_probs), dim=0)
    predicted = torch.argmax(avg_prob).item()
    avg_diffusion = float(np.mean(diffusion_scores)) if diffusion_scores else 0.0
    return predicted, avg_prob.numpy(), avg_diffusion


# 🚀 Run on folder
video_folder = "videos_to_predict"
for vid in os.listdir(video_folder):
    if vid.endswith(".mp4"):
        path = os.path.join(video_folder, vid)
        label, prob, diffusion_score = predict_video(path)
        generation = classify_generation(float(prob[0][1]), diffusion_score)
        print(
            f"{vid}: {'FAKE' if label == 1 else 'REAL'} | Real: {prob[0][0]:.3f}, Fake: {prob[0][1]:.3f} | Generation: {generation} | Diffusion: {diffusion_score:.2f}"
        )
