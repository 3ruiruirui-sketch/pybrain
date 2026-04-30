"""
Dual-Head SegResNet + ResNet50 Classification
============================================
SegResNet encoder → segmentation head (original)
                └→ classification head (ResNet50-style on bottleneck features)

Architecture:
  Encoder:  down_layers (32→64→128→256 channels, 4 levels)
  Decoder:  up_layers + up_samples → segmentation (3 classes: TC, WT, ET)
  Classifier: GlobalAvgPool3d → FC(256, 128) → Dropout → FC(128, num_classes)

Supports:
  - Joint inference (both heads active)
  - Segmentation-only (freeze classification head)
  - Classification-only (discard segmentation output)
  - Fine-tuning from pretrained SegResNet weights

Classes:
  LGG (Grade II) — Low-grade glioma
  HGG (Grade III-IV) — High-grade glioma

Author: Integration into PY-BRAIN pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet
from typing import Optional


class DualHeadSegResNet(nn.Module):
    """
    Dual-head SegResNet:
      - Head A (Segmentation): 3D output (TC, WT, ET tumor regions)
      - Head B (Classification): Grade prediction (LGG vs HGG, or 4-class WHO grade)

    The classification head uses the encoder bottleneck features (init_filters*8 channels,
    after all down_layers), pooled via 3D Global Average Pooling → 2 FC layers → softmax.

    To extract raw encoder features for training:
      model.extract_bottleneck(x) → (B, 128, D', H', W')
    """

    def extract_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run encoder only, return the raw bottleneck tensor.
        Use this to extract features for training the classifier head.
        """
        x_perm = x[:, (2, 1, 3, 0), :, :, :].clone()
        bottleneck, _ = self.seg_backbone.encode(x_perm)
        return bottleneck

    @staticmethod
    def get_encoder_bottleneck_hook():
        """Returns a hook function to capture encoder bottleneck features."""
        bottleneck_out = {}

        def hook_fn(module, input, output):
            bottleneck_out["feats"] = output.detach()

        return hook_fn, bottleneck_out

    def forward(
        self,
        x: torch.Tensor,
        return_class_logits: bool = True,
        return_seg: bool = True,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 4, D, H, W) — 4 modalities
            return_class_logits: If True, compute and return classification logits
            return_seg: If True, compute and return segmentation

        Returns:
            dict with keys:
              - seg: segmentation probabilities (B, 3, D, H, W) [TC, WT, ET]
              - class_logits: classification logits (B, num_classes)
              - class_probs: softmax probabilities (B, num_classes)
              - bottleneck: raw encoder bottleneck (B, 128, D', H', W') — always returned
        """
        # Channel permutation: [FLAIR, T1, T1c, T2] → [T1c, T1, T2, FLAIR]
        x_perm = x[:, (2, 1, 3, 0), :, :, :].clone()

        # Encode — capture bottleneck features
        bottleneck, down_x = self.seg_backbone.encode(x_perm)
        # down_x: list of encoder outputs for skip connections
        # bottleneck: (B, 128, D', H', W') with init_filters=16

        # ── Classification head ──────────────────────────────────────────────
        if return_class_logits:
            class_logits = self.classifier(bottleneck)
            class_probs = F.softmax(class_logits, dim=1)
        else:
            class_logits = None
            class_probs = None

        # ── Segmentation head ────────────────────────────────────────────────
        if return_seg:
            # Decode (same as MONAI SegResNet)
            down_x.reverse()  # [layer3, layer2, layer1, layer0]
            seg = self.seg_backbone.decode(bottleneck, down_x)
            seg_probs = torch.sigmoid(seg)
        else:
            seg_probs = None

        return {
            "seg": seg_probs,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "bottleneck": bottleneck,
        }

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        init_filters: int = 16,
        blocks_down: list = [1, 2, 2, 4],
        blocks_up: list = [1, 1, 1],
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        # ── Segmentation backbone (SegResNet) ───────────────────────────────────
        self.seg_backbone = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
        )

        # ── Classification head (ResNet50-style on bottleneck) ────────────────
        # Bottleneck features: init_filters*8 channels after last down layer
        # e.g. init_filters=16 → 128-ch bottleneck; init_filters=32 → 256-ch
        bottleneck_ch = init_filters * 8
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # → (B, bottleneck_ch, 1, 1, 1)
            nn.Flatten(start_dim=1),  # → (B, bottleneck_ch)
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Load pretrained segmentation weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        """Load pretrained SegResNet weights, preserving segmentation performance."""
        import logging

        logger = logging.getLogger("models.dual_head")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
            result = self.seg_backbone.load_state_dict(sd, strict=False)
            logger.info(f"Loaded pretrained SegResNet from {path}")
            if result.missing_keys:
                logger.warning(f"Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys: {result.unexpected_keys}")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Training wrapper with combined loss
# ─────────────────────────────────────────────────────────────────────────────


class CombinedLoss(nn.Module):
    """
    Combined loss for dual-head training:
      - Segmentation: Dice + CrossEntropy
      - Classification: CrossEntropy
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ce_seg = nn.BCEWithLogitsLoss()
        self.ce_cls = nn.CrossEntropyLoss()

    def forward(self, seg_pred, cls_pred, seg_target, cls_target):
        seg_loss = self.ce_seg(seg_pred, seg_target)
        cls_loss = self.ce_cls(cls_pred, cls_target)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────


def load_dual_head(
    bundle_dir: str,
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> DualHeadSegResNet:
    """
    Load a DualHeadSegResNet initialized from pretrained SegResNet weights.

    Args:
        bundle_dir: Path to models/brats_bundle/
        num_classes: 2 (LGG/HGG) or 4 (Grade II/III/IV/Normal)
        dropout_rate: Dropout in classification head
        device: torch device

    Returns:
        DualHeadSegResNet with pretrained segmentation weights loaded
    """
    from pathlib import Path

    bundle_path = Path(bundle_dir) / "brats_mri_segmentation" / "models" / "model.pt"

    model = DualHeadSegResNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained_path=str(bundle_path) if bundle_path.exists() else None,
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test
    model = DualHeadSegResNet(num_classes=2, dropout_rate=0.3)
    total = sum(p.numel() for p in model.parameters())
    print(f"DualHeadSegResNet: {total:,} parameters")

    # Test forward
    x = torch.randn(1, 4, 128, 128, 128)
    out = model(x)
    print(f"Seg shape:   {out['seg'].shape}")
    print(f"Cls logits:  {out['class_logits'].shape}")
    print(f"Cls probs:   {out['class_probs'].shape}")

    # Test classification only
    out_cls = model(x, return_seg=False)
    print(f"Cls only:    {out_cls['class_probs'].shape}")
