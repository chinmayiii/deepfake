import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from lightning_modules.detector import HybridEfficientNet
from utils.fft_utils import fft_from_pil

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_model_path(model_path: str | None):
    root = Path(__file__).resolve().parent
    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        return candidate

    hybrid_default = root / "models" / "best_model-hybrid.pt"
    legacy_default = root / "models" / "best_model-v3.pt"
    if hybrid_default.exists():
        return hybrid_default
    return legacy_default


def load_model(model_path: Path):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    if "hybrid" in model_path.name.lower():
        model = HybridEfficientNet(weights=weights)
        mode = "hybrid"
    else:
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        mode = "single"

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, mode


def load_samples(dataset_root: Path):
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Expected subfolders 'real' and 'fake' inside {dataset_root}"
        )

    samples = []
    for label_name, label_idx in (("real", 0), ("fake", 1)):
        class_dir = dataset_root / label_name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
                samples.append((file_path, label_idx))

    if not samples:
        raise ValueError(
            f"No images found in {dataset_root}/real or {dataset_root}/fake"
        )
    return samples


def predict_prob(model, mode: str, image: Image.Image, preprocess):
    rgb_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        if mode == "hybrid":
            fft_image = fft_from_pil(image)
            fft_tensor = preprocess(fft_image).unsqueeze(0)
            logits = model(rgb_tensor, fft_tensor)
        else:
            logits = model(rgb_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        fake_prob = probs[1].item()
    return fake_prob


def evaluate(dataset_root: Path, model_path: Path):
    preprocess = build_preprocess()
    model, mode = load_model(model_path)
    samples = load_samples(dataset_root)

    y_true, y_pred, y_score = [], [], []
    for image_path, label in samples:
        image = Image.open(image_path).convert("RGB")
        fake_prob = predict_prob(model, mode, image, preprocess)
        pred = 1 if fake_prob >= 0.5 else 0

        y_true.append(label)
        y_pred.append(pred)
        y_score.append(fake_prob)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "num_samples": len(y_true),
        "model_path": str(model_path),
        "model_mode": mode,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepfakeDetector model on real/fake folders"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset folder containing subfolders: real/ and fake/",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model path. Defaults to models/best_model-hybrid.pt if present, otherwise models/best_model-v3.pt",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path for evaluation summary (for /eval/summary API endpoint)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset_root = Path(args.data)
    if not dataset_root.is_absolute():
        dataset_root = (root / dataset_root).resolve()
    model_path = resolve_model_path(args.model)

    metrics = evaluate(dataset_root, model_path)
    cm = metrics["confusion_matrix"]
    print(f"Model: {metrics['model_path']} ({metrics['model_mode']})")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion Matrix [rows=true (real,fake), cols=pred (real,fake)]")
    print(np.array2string(cm))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "num_samples": int(metrics["num_samples"]),
            "model_path": metrics["model_path"],
            "model_mode": metrics["model_mode"],
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]),
            "confusion_matrix": cm.tolist(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation summary JSON to: {out_path}")


if __name__ == "__main__":
    main()
