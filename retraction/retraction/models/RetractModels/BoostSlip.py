import os
import glob
import pickle

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm  # <--- NEW

# Import your backbone + ROI util from the existing file
from UnsupervisedObserver import ImageBackboneEncoder, apply_circular_roi


DATA_DIR = "data/twin_data"          # folder with demo_*.pkl
BACKBONE = 'eff'                # 'resnet50' or 'vit' or 'efficientnet_b0'
BATCH_SIZE = 32
NUM_WORKERS = 4
ROI_CENTER = (112, 112)              # for 224x224 images
ROI_RADIUS = 112                     # full circle; reduce if you want tighter tumor ROI
TEST_SIZE = 0.2
RANDOM_STATE = 42
SLIP_LABEL_VALUE = 1                 # assumes 1 = slip, 0 = no slip


class SlipDataset(Dataset):
    """
    Expects each pickle file to have:
      - 'image':      pre image, RGB, HxWx3 numpy
      - 'end_image':  post image, RGB, HxWx3 numpy
      - 'label':      'no slip' / 'slip' (string) or 0/1
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "demo_*.pkl")))
        if not self.files:
            raise RuntimeError(f"No demo_*.pkl files found in {data_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with open(p, "rb") as f:
            data = pickle.load(f)

        img_pre_np = data["image"]       # RGB HxWx3
        img_post_np = data["end_image"]  # RGB HxWx3
        label_raw = data["label"]        # could be string or int

        # Map label to 0/1
        if isinstance(label_raw, str):
            if "no" in label_raw.lower():
                label = 0
            else:
                label = 1
        else:
            label = int(label_raw)

        # Ensure uint8 RGB
        img_pre_np = np.asarray(img_pre_np, dtype=np.uint8)
        img_post_np = np.asarray(img_post_np, dtype=np.uint8)

        img_pre = Image.fromarray(img_pre_np)
        img_post = Image.fromarray(img_post_np)

        if self.transform is not None:
            img_pre = self.transform(img_pre)     # [C, H, W]
            img_post = self.transform(img_post)   # [C, H, W]

        return img_pre, img_post, label



def build_feature_matrix(
    backbone_name=BACKBONE,
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    roi_center=ROI_CENTER,
    roi_radius=ROI_RADIUS,
    device=None,
):
    """
    Uses ImageBackboneEncoder to extract latent features for pre and post images,
    then builds XGBoost-friendly feature vectors:

    x = [ z_pre,
          z_post,
          |z_pre - z_post|,
          (z_pre - z_post)^2,
          cos_dist,
          l2_dist ]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = SlipDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    encoder = ImageBackboneEncoder(backbone=backbone_name, train_backbone=False)
    encoder = encoder.to(device)
    encoder.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        # Wrap dataloader with tqdm for a nice progress bar
        for img_pre, img_post, labels in tqdm(dataloader, desc="Extracting features"):
            img_pre = img_pre.to(device)   # [B, C, H, W]
            img_post = img_post.to(device)

            # Apply same circular ROI
            img_pre_roi = apply_circular_roi(img_pre, roi_center, roi_radius)
            img_post_roi = apply_circular_roi(img_post, roi_center, roi_radius)

            # Encode
            feats_pre = encoder(img_pre_roi)    # [B, D]
            feats_post = encoder(img_post_roi)  # [B, D]

            # Build feature components
            diff = feats_pre - feats_post
            diff_abs = torch.abs(diff)
            diff_sq = diff ** 2

            # Distances
            cos_sim = F.cosine_similarity(feats_pre, feats_post, dim=-1, eps=1e-8)  # [B]
            cos_dist = (1.0 - cos_sim).unsqueeze(-1)  # [B, 1]
            l2_dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, 1]

            feat = torch.cat(
                [
                    feats_pre,
                    feats_post,
                    diff_abs,
                    diff_sq,
                    cos_dist,
                    l2_dist,
                ],
                dim=-1,
            )  # [B, D_total]

            all_features.append(feat.cpu().numpy())
            all_labels.append(np.array(labels))

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    return X, y



def train_xgboost_classifier(X, y, slip_label=SLIP_LABEL_VALUE):
    """
    Train an XGBoost classifier on (X, y) with class-imbalance handling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pos = np.sum(y_train == slip_label)
    neg = np.sum(y_train != slip_label)

    if pos == 0:
        print("Warning: no positive (slip) samples in training set!")
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = neg / max(pos, 1)
    print(f"Training samples: pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f}")

    model = XGBClassifier(
        max_depth=5,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=8,
        tree_method="hist",
        verbosity=1,   # show some training info
    )

    # ---- FIX for `_estimator_type` bug ----
    # Some xgboost versions don't set this automatically.
    model._estimator_type = "classifier"

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]  # probability of slip
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Classification report (threshold 0.5) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC-AUC: {e}")

    return model


def main():
    X, y = build_feature_matrix()
    model = train_xgboost_classifier(X, y)

    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "xgb_slip_classifier.json")
    model.save_model(model_path)
    print(f"\nSaved XGBoost model to {model_path}")

    stats_path = os.path.join(out_dir, "xgb_features_stats.npz")
    np.savez(
        stats_path,
        mean=X.mean(axis=0),
        std=X.std(axis=0) + 1e-8,
    )
    print(f"Saved feature stats to {stats_path}")


if __name__ == "__main__":
    main()
