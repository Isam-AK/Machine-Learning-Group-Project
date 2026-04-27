"""
Generate and save prediction arrays for both models on the shared test set.
Run this once from the Machine-Learning-Group-Project directory:
    python generate_predictions.py

Outputs (saved to predictions/):
    y_true_binary.npy          — ground truth: 0=benign 1=malignant (1503,)
    mn_pred_binary.npy         — MobileNetV2 Stage 1 predictions (1503,)
    mn_probs_binary.npy        — MobileNetV2 Stage 1 probabilities (1503, 2)
    mn_true_s2.npy             — ground truth for Stage 1 flagged images (n_flagged,)
    mn_pred_s2.npy             — MobileNetV2 Stage 2 predictions (n_flagged,)
    mn_probs_s2.npy            — MobileNetV2 Stage 2 probabilities (n_flagged, 3)
    vit_pred_binary.npy        — ViT Stage 1 predictions (1503,)
    vit_probs_binary.npy       — ViT Stage 1 probabilities (1503, 2)
    vit_true_s2.npy            — ground truth for Stage 1 flagged images (n_flagged,)
    vit_pred_s2.npy            — ViT Stage 2 predictions (n_flagged,)
    vit_probs_s2.npy           — ViT Stage 2 probabilities (n_flagged, 3)
"""
import os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, vit_b_16, ViT_B_16_Weights
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

# ── Config ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

_dir = Path(__file__).parent
ARCHIVE_DIR = next((p for p in [_dir.parent / 'archive', _dir / 'archive'] if p.is_dir()))
METADATA_CSV = ARCHIVE_DIR / 'HAM10000_metadata.csv'
IMG_DIRS = [ARCHIVE_DIR / 'HAM10000_images_part_1',
            ARCHIVE_DIR / 'HAM10000_images_part_2']

BENIGN_CLASSES    = {'nv', 'bkl', 'df', 'vasc'}
MAL_SUBCLASS_MAP  = {'akiec': 0, 'bcc': 1, 'mel': 2}

OUT_DIR = _dir / 'predictions'
OUT_DIR.mkdir(exist_ok=True)

# ── Load metadata & test split ────────────────────────────────────────────────
meta = pd.read_csv(METADATA_CSV)
def find_image(img_id):
    for d in IMG_DIRS:
        p = d / f'{img_id}.jpg'
        if p.exists():
            return str(p)
    return None

meta['filepath']     = meta['image_id'].apply(find_image)
meta['stage1_label'] = meta['dx'].apply(lambda x: 0 if x in BENIGN_CLASSES else 1)
meta['stage2_label'] = meta['dx'].apply(lambda x: MAL_SUBCLASS_MAP.get(x, -1))
meta = meta.dropna(subset=['filepath']).reset_index(drop=True)

# Load shared test indices (saved by MobileNetV2 notebook)
test_idx = pd.read_csv(_dir / 'test_indices.csv')['0'].tolist()
test_df  = meta.iloc[test_idx].reset_index(drop=True)
print(f'Test set: {len(test_df)} images | Malignant: {(test_df["stage1_label"]==1).sum()}')

# ── Dataset ───────────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        return self.transform(img), int(row['stage1_label']), int(row['stage2_label'])

test_loader = DataLoader(SkinDataset(test_df, val_transform), batch_size=32,
                         shuffle=False, num_workers=0, pin_memory=True)

y_true_binary = test_df['stage1_label'].values.astype(np.int64)
np.save(OUT_DIR / 'y_true_binary.npy', y_true_binary)
print('Ground truth saved.')

# ── MobileNetV2 ───────────────────────────────────────────────────────────────
print('\nRunning MobileNetV2...')

mn_s1 = mobilenet_v2(weights=None)
mn_s1.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 2))
mn_s1.load_state_dict(torch.load('best_model_stage1.pt', map_location=device))
mn_s1 = mn_s1.to(device).eval()

mn_s2 = mobilenet_v2(weights=None)
mn_s2.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 3))
mn_s2.load_state_dict(torch.load('best_model_stage2.pt', map_location=device))
mn_s2 = mn_s2.to(device).eval()

mn_pred_binary, mn_probs_binary = [], []
all_s1_labels_list, all_s2_labels_list, all_imgs_list = [], [], []

with torch.no_grad():
    for imgs, s1_labels, s2_labels in test_loader:
        imgs_dev = imgs.to(device)
        with autocast():
            logits = mn_s1(imgs_dev)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        mn_pred_binary.append(preds)
        mn_probs_binary.append(probs)
        all_imgs_list.append(imgs)
        all_s1_labels_list.append(s1_labels.numpy())
        all_s2_labels_list.append(s2_labels.numpy())

mn_pred_binary  = np.concatenate(mn_pred_binary)
mn_probs_binary = np.concatenate(mn_probs_binary)
all_s1_labels   = np.concatenate(all_s1_labels_list)
all_s2_labels   = np.concatenate(all_s2_labels_list)

# Stage 2: only run on TRUE POSITIVES (Stage 1 flagged malignant AND ground truth is malignant)
# This gives a clean pipeline_recall = correctly subtyped TPs / total malignant
tp_mask = (mn_pred_binary == 1) & (all_s1_labels == 1)
tp_indices = np.where(tp_mask)[0]

mn_true_s2_list, mn_pred_s2_list, mn_probs_s2_list = [], [], []
BATCH = 32
with torch.no_grad():
    for start in range(0, len(tp_indices), BATCH):
        batch_idx = tp_indices[start:start+BATCH]
        # rebuild image batch from stored tensors
        batch_imgs = []
        for all_i, bi in enumerate(batch_idx):
            # find which loader batch and position
            b_num = bi // 32; b_pos = bi % 32
            batch_imgs.append(all_imgs_list[b_num][b_pos])
        batch_imgs = torch.stack(batch_imgs).to(device)
        with autocast():
            logits2 = mn_s2(batch_imgs)
        probs2 = torch.softmax(logits2, dim=1).cpu().numpy()
        pred2  = probs2.argmax(axis=1)
        mn_pred_s2_list.append(pred2)
        mn_probs_s2_list.append(probs2)
        mn_true_s2_list.append(all_s2_labels[batch_idx])

mn_true_s2  = np.concatenate(mn_true_s2_list)  if mn_true_s2_list  else np.array([])
mn_pred_s2  = np.concatenate(mn_pred_s2_list)  if mn_pred_s2_list  else np.array([])
mn_probs_s2 = np.concatenate(mn_probs_s2_list) if mn_probs_s2_list else np.zeros((0, 3))

np.save(OUT_DIR / 'mn_pred_binary.npy',  mn_pred_binary)
np.save(OUT_DIR / 'mn_probs_binary.npy', mn_probs_binary)
np.save(OUT_DIR / 'mn_true_s2.npy',      mn_true_s2)
np.save(OUT_DIR / 'mn_pred_s2.npy',      mn_pred_s2)
np.save(OUT_DIR / 'mn_probs_s2.npy',     mn_probs_s2)
print(f'MobileNetV2 done. Flagged malignant: {mn_pred_binary.sum()}/{len(mn_pred_binary)}')

# ── ViT-B/16 ─────────────────────────────────────────────────────────────────
print('\nRunning ViT-B/16...')

class PretrainedTwoStageViT(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Identity()
        embed_dim = 768
        self.stage1_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, 2)
        )
        self.stage2_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, 3)
        )
    def forward(self, x):
        feat = self.backbone(x)
        return self.stage1_head(feat), self.stage2_head(feat)

vit = PretrainedTwoStageViT().to(device)

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)

# Stage 1: use best_stage1_pretrained.pt (backbone optimised for binary classification)
load_ckpt(vit, 'checkpoints_pretrained/best_stage1_pretrained.pt')
vit.eval()

vit_pred_binary, vit_probs_binary = [], []
vit_s1_labels_list, vit_s2_labels_list, vit_imgs_list = [], [], []

with torch.no_grad():
    for imgs, s1_labels, s2_labels in test_loader:
        imgs_dev = imgs.to(device)
        with autocast():
            logits1, _ = vit(imgs_dev)
        probs1 = torch.softmax(logits1, dim=1).cpu().numpy()
        pred1  = probs1.argmax(axis=1)
        vit_pred_binary.append(pred1)
        vit_probs_binary.append(probs1)
        vit_imgs_list.append(imgs)
        vit_s1_labels_list.append(s1_labels.numpy())
        vit_s2_labels_list.append(s2_labels.numpy())

vit_pred_binary  = np.concatenate(vit_pred_binary)
vit_probs_binary = np.concatenate(vit_probs_binary)
vit_s1_all = np.concatenate(vit_s1_labels_list)
vit_s2_all = np.concatenate(vit_s2_labels_list)

# Stage 2: load best_stage2_pretrained.pt (stage2_head properly trained)
load_ckpt(vit, 'checkpoints_pretrained/best_stage2_pretrained.pt')
vit.eval()

# Only run Stage 2 on TRUE POSITIVES (flagged malignant AND actually malignant)
vit_tp_mask   = (vit_pred_binary == 1) & (vit_s1_all == 1)
vit_tp_indices = np.where(vit_tp_mask)[0]

vit_true_s2_list, vit_pred_s2_list, vit_probs_s2_list = [], [], []
with torch.no_grad():
    for start in range(0, len(vit_tp_indices), BATCH):
        batch_idx = vit_tp_indices[start:start+BATCH]
        batch_imgs = []
        for bi in batch_idx:
            b_num = bi // 32; b_pos = bi % 32
            batch_imgs.append(vit_imgs_list[b_num][b_pos])
        batch_imgs = torch.stack(batch_imgs).to(device)
        with autocast():
            _, logits2 = vit(batch_imgs)
        probs2 = torch.softmax(logits2, dim=1).cpu().numpy()
        pred2  = probs2.argmax(axis=1)
        vit_pred_s2_list.append(pred2)
        vit_probs_s2_list.append(probs2)
        vit_true_s2_list.append(vit_s2_all[batch_idx])

vit_true_s2  = np.concatenate(vit_true_s2_list)  if vit_true_s2_list  else np.array([], dtype=np.int64)
vit_pred_s2  = np.concatenate(vit_pred_s2_list)  if vit_pred_s2_list  else np.array([], dtype=np.int64)
vit_probs_s2 = np.concatenate(vit_probs_s2_list) if vit_probs_s2_list else np.zeros((0, 3))

np.save(OUT_DIR / 'vit_pred_binary.npy',  vit_pred_binary)
np.save(OUT_DIR / 'vit_probs_binary.npy', vit_probs_binary)
np.save(OUT_DIR / 'vit_true_s2.npy',      vit_true_s2)
np.save(OUT_DIR / 'vit_pred_s2.npy',      vit_pred_s2)
np.save(OUT_DIR / 'vit_probs_s2.npy',     vit_probs_s2)
print(f'ViT done. Flagged malignant: {vit_pred_binary.sum()}/{len(vit_pred_binary)}')

print('\nAll predictions saved to predictions/')
