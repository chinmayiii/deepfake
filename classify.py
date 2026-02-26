import torch
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import argparse

from lightning_modules.detector import HybridEfficientNet
from utils.diffusion_heuristics import diffusion_heuristic_score, classify_generation
from utils.fft_utils import fft_from_pil


# Load your trained model
def load_model(model_path="models/best_model-hybrid.pt"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = HybridEfficientNet(weights=weights)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# Preprocess and classify image
def predict_image(image_path, model):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    fft_image = fft_from_pil(image)
    input_tensor = transform(image).unsqueeze(0)
    fft_tensor = transform(fft_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, fft_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    diffusion_score = diffusion_heuristic_score(image)
    generation = classify_generation(probs[1].item(), diffusion_score)
    print(f"\n🧠 Prediction: {label}")
    print(f"Real: {probs[0]:.3f} | Fake: {probs[1]:.3f}")
    print(f"Generation: {generation} | Diffusion Score: {diffusion_score:.2f}")


# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file (.jpg/.png)")
    args = parser.parse_args()

    model = load_model()
    predict_image(args.image_path, model)
