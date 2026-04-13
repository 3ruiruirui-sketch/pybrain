import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from radiomics import featureextractor

# -----------------------------
# 1) CRITICAL WEIGHT CHECK
# -----------------------------
WEIGHTS_PATH = Path("./models/BrainIAC/weights/idh_weights.pth")

def resolve_idh_weights_path(download_fn=None):
    """
    Enforces:
    - check whether ./models/BrainIAC/weights/idh_weights.pth exists
    - if not, either trigger a caller-supplied download flow or raise a clear error
    """
    if WEIGHTS_PATH.exists():
        return WEIGHTS_PATH

    if download_fn is not None:
        download_fn(WEIGHTS_PATH)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Missing BrainIAC downstream weights: "
            "./models/BrainIAC/weights/idh_weights.pth\n"
            "Please obtain the IDH weights and place them at that exact path "
            "before training or inference."
        )

    return WEIGHTS_PATH

# Optional example hook; replace with your own authenticated logic if desired.
def optional_download_flow(target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Put your authenticated/manual handoff here if you actually want to automate it.
    # Example:
    # raise RuntimeError("Authenticated download not implemented in codebase.")
    pass


# -----------------------------
# 2) PYRADIOMICS CONFIG
# -----------------------------
RADIOMICS_PARAMS = {
    "imageType": {
        "Original": {},
        "LoG": {"sigma": [1.0, 2.0, 3.0]},
        "Wavelet": {}
    },
    "featureClass": {
        "firstorder": [],
        "shape": [],
        "glcm": [],
        "glrlm": [],
        "glszm": [],
        "gldm": [],
        "ngtdm": []
    },
    "setting": {
        "label": 1,
        "normalize": True,
        "normalizeScale": 100,
        "removeOutliers": 3,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "correctMask": True,
        "minimumROISize": 50,
        "binWidth": 25,
        "preCrop": True,
        "padDistance": 5
    }
}

radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_PARAMS)


def extract_radiomics_vector(image_path, mask_path):
    result = radiomics_extractor.execute(str(image_path), str(mask_path))
    features = {
        k: float(v) for k, v in result.items()
        if not k.startswith("diagnostics_")
    }
    return features


# -----------------------------
# 3) MRI LOADING
# -----------------------------
def load_nifti_as_tensor(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
    arr = np.nan_to_num(arr)

    if arr.std() > 0:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    arr = np.expand_dims(arr, axis=0)  # c, z, y, x
    return torch.tensor(arr, dtype=torch.float32)


# -----------------------------
# 4) BRAINIAC WRAPPER
# -----------------------------
class BrainIACBackbone(nn.Module):
    """
    Replace this wrapper with the actual BrainIAC model import/init from your local repo.
    This wrapper expects:
      - load checkpoint from idh_weights.pth
      - expose an embedding vector before the final classifier
    """
    def __init__(self, embedding_dim=512, freeze_backbone=True):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Placeholder 3D encoder; swap for real BrainIAC architecture
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.proj = nn.Linear(64, embedding_dim)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_downstream_weights(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        print("Loaded checkpoint")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        x = self.proj(x)
        return x


# -----------------------------
# 5) DATASET
# -----------------------------
class GBMFusionDataset(Dataset):
    """
    dataframe columns:
      patient_id, image_path, mask_path, target
    radiomics_cache_csv:
      optional path to save/load precomputed radiomics
    """
    def __init__(self, dataframe, radiomics_cache_csv=None):
        self.df = dataframe.reset_index(drop=True)
        self.radiomics_cache_csv = radiomics_cache_csv
        self.radiomics_df = self._prepare_radiomics()

        self.feature_cols = [
            c for c in self.radiomics_df.columns if c != "patient_id"
        ]

    def _prepare_radiomics(self):
        if self.radiomics_cache_csv and Path(self.radiomics_cache_csv).exists():
            return pd.read_csv(self.radiomics_cache_csv)

        rows = []
        for _, row in self.df.iterrows():
            feats = extract_radiomics_vector(row["image_path"], row["mask_path"])
            feats["patient_id"] = row["patient_id"]
            rows.append(feats)

        rad_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        rad_df = rad_df.fillna(rad_df.median(numeric_only=True))

        if self.radiomics_cache_csv:
            Path(self.radiomics_cache_csv).parent.mkdir(parents=True, exist_ok=True)
            rad_df.to_csv(self.radiomics_cache_csv, index=False)

        return rad_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_nifti_as_tensor(row["image_path"])

        rad_row = self.radiomics_df[self.radiomics_df["patient_id"] == row["patient_id"]]
        rad = rad_row[self.feature_cols].values.squeeze().astype(np.float32)

        target = np.float32(row["target"])

        return {
            "image": image,
            "radiomics": torch.tensor(rad, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_id": row["patient_id"]
        }


# -----------------------------
# 6) FUSION MODEL
# -----------------------------
class BrainIACRadiomicsFusion(nn.Module):
    def __init__(self, radiomics_dim, embedding_dim=512, hidden_dim=256, freeze_backbone=True):
        super().__init__()
        self.brainiac = BrainIACBackbone(
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone
        )

        self.rad_norm = nn.BatchNorm1d(radiomics_dim)

        self.head = nn.Sequential(
            nn.Linear(embedding_dim + radiomics_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def load_weights(self, checkpoint_path):
        self.brainiac.load_downstream_weights(checkpoint_path)

    def forward(self, image, radiomics):
        emb = self.brainiac(image)
        rad = self.rad_norm(radiomics)
        x = torch.cat([emb, rad], dim=1)
        logits = self.head(x).squeeze(1)
        return logits


# -----------------------------
# 7) TRAIN LOOP
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        radiomics = batch["radiomics"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(image, radiomics)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probs_all, y_all = [], []

    for batch in loader:
        image = batch["image"].to(device)
        radiomics = batch["radiomics"].to(device)
        target = batch["target"].to(device)

        logits = model(image, radiomics)
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * image.size(0)
        probs_all.extend(probs.cpu().numpy().tolist())
        y_all.extend(target.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), probs_all, y_all


# -----------------------------
# 8) ENTRYPOINT
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = resolve_idh_weights_path(download_fn=None)
    # Or:
    # checkpoint_path = resolve_idh_weights_path(download_fn=optional_download_flow)

    df = pd.read_csv("train_manifest.csv")
    # expected columns: patient_id,image_path,mask_path,target,split

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    train_ds = GBMFusionDataset(train_df, radiomics_cache_csv="cache/train_radiomics.csv")
    val_ds = GBMFusionDataset(val_df, radiomics_cache_csv="cache/val_radiomics.csv")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    radiomics_dim = len(train_ds.feature_cols)
    model = BrainIACRadiomicsFusion(
        radiomics_dim=radiomics_dim,
        embedding_dim=512,
        hidden_dim=256,
        freeze_backbone=True
    ).to(device)

    model.load_weights(checkpoint_path)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    for epoch in range(1, 11):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, probs, y_true = validate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), "fusion_model.pt")


if __name__ == "__main__":
    main()
