import pytorch_lightning as pl
import torch
from torchvision.models import efficientnet_b0


class HybridEfficientNet(torch.nn.Module):
    def __init__(self, weights=None, dropout=0.4):
        super().__init__()
        self.rgb_backbone = efficientnet_b0(weights=weights)
        self.fft_backbone = efficientnet_b0(weights=weights)
        in_features = self.rgb_backbone.classifier[1].in_features
        self.rgb_backbone.classifier = torch.nn.Identity()
        self.fft_backbone.classifier = torch.nn.Identity()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features * 2, 2),
        )

    def forward(self, rgb, fft):
        rgb_feat = self.rgb_backbone(rgb)
        fft_feat = self.fft_backbone(fft)
        merged = torch.cat([rgb_feat, fft_feat], dim=1)
        return self.classifier(merged)


class DeepfakeDetector(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, rgb, fft=None):
        if fft is None:
            return self.model(rgb)
        return self.model(rgb, fft)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, (tuple, list)):
            logits = self(*x)
        else:
            logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, (tuple, list)):
            logits = self(*x)
        else:
            logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
