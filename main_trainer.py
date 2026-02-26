import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector, HybridEfficientNet
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# === Load YAML config ===
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# === Transforms ===
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# === Dataset Paths ===
train_sources = [(p, None) for p in cfg["train_paths"]]
val_sources = [(p, None) for p in cfg["val_paths"]]

# === Datasets & Loaders ===
train_dataset = HybridDeepfakeDataset(train_sources, transform=transform)
val_dataset = HybridDeepfakeDataset(val_sources, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
)

# === Model Architecture ===
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
hybrid_model = HybridEfficientNet(weights=weights)

model = DeepfakeDetector(hybrid_model, lr=cfg["lr"])

# === Callbacks ===
checkpoint = ModelCheckpoint(
    monitor=cfg.get("monitor_metric", "val_loss"),
    dirpath="models",
    filename="best_model",
    save_top_k=1,
    mode="min",
)

early_stop = EarlyStopping(
    monitor=cfg.get("monitor_metric", "val_loss"), patience=3, mode="min"
)

# === Trainer ===
trainer = pl.Trainer(
    max_epochs=cfg["num_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint, early_stop],
    enable_progress_bar=True,
    log_every_n_steps=cfg.get("log_every_n_steps", 1),
)

# === Start Training ===
trainer.fit(model, train_loader, val_loader)
