#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroNet — Colab one‑click training (improved)

Upgrades over the previous version:
- Deterministic seeding for reproducibility
- Dynamic feature sizing (no magic 16576)
- BCEWithLogitsLoss (numerically stable) — model outputs logits
- Class imbalance handling via pos_weight estimated from train set
- AdamW optimizer + CosineAnnealingLR schedule (weight decay)
- Mixed precision training (autocast + GradScaler)
- Early stopping on Validation AP with best checkpoint restore
- Validation-driven threshold (F1) reused on Test
- Cleaner metrics/plots that operate on probabilities derived from logits
- Fewer DataLoader workers by default (8)
"""

import os
import glob
import random
import shutil
import zipfile
import tarfile
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================
# Global settings
# ============================
DATA_DIR = "/content/astronet_data"
DRIVE_DATA_DIR = "/content/drive/MyDrive/astronet_data/"
NUM_WORKERS = 8  # avoid Colab warning / oversubscription
BATCH_SIZE = 64
MAX_EPOCHS = 50
LR = 2e-5  # slightly higher than before; AdamW + WD
WEIGHT_DECAY = 1e-4
PATIENCE = 8  # early stopping patience by Val AP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_seed(seed: int = 42, deterministic: bool = True):
    import numpy as _np
    import random as _rnd
    torch.manual_seed(seed)
    _np.random.seed(seed) 
    _rnd.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------
# Data prep utilities
# ----------------------------
def check_existing_data() -> bool:
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        return False
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            return False
        if len(glob.glob(os.path.join(split_dir, '*.npy'))) == 0:
            return False
    total_samples = 0
    for split in ['train', 'val', 'test']:
        total_samples += len(glob.glob(os.path.join(DATA_DIR, split, '*.npy'))) // 3
    print(f"Found existing data: {total_samples} samples")
    print(f"Data directory: {data_dir}")
    return total_samples > 0


def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')


def find_and_extract_data() -> bool:
    print("Finding data files...")
    patterns = ['*.tar.gz', '*.zip']
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(os.path.join(DRIVE_DATA_DIR, pattern)))
    if not found_files:
        print("No data files found! Please upload your AstroNet data to Google Drive.")
        return False
    print(f"Found {len(found_files)} files")
    os.makedirs(DATA_DIR, exist_ok=True)
    for file_path in tqdm(found_files, desc="Extracting"):
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        elif file_path.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(DATA_DIR)
    return True


def organize_data() -> bool:
    source_dir = DATA_DIR
    target_dir = DATA_DIR
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    groups = {}
    for root, _, files in os.walk(source_dir):
        rel = os.path.relpath(root, source_dir)
        parts = rel.split(os.sep) if rel != '.' else []
        split = 'train' if 'train' in parts else 'val' if 'val' in parts else 'test' if 'test' in parts else None
        if split is None:
            continue
        for fname in files:
            if not fname.endswith('.npy') or 'cen' in fname:
                continue
            name_parts = fname.split('_')
            if len(name_parts) < 3:
                continue
            sample_id = '_'.join(name_parts[:3])
            key = (split, sample_id)
            fpath = os.path.join(root, fname)
            if fname.endswith('_global.npy'):
                groups.setdefault(key, {})['global'] = fpath
            elif fname.endswith('_local.npy'):
                groups.setdefault(key, {})['local'] = fpath
            elif fname.endswith('_info.npy'):
                groups.setdefault(key, {})['info'] = fpath
    total = 0
    for (split, _sid), files_dict in groups.items():
        if all(k in files_dict for k in ('global', 'local', 'info')):
            for _kind, fpath in files_dict.items():
                dst = os.path.join(target_dir, split, os.path.basename(fpath))
                if fpath != dst:
                    shutil.copy2(fpath, dst)
            total += 1
    print(f"✅ Total: {total} samples organized")
    return total > 0


def prepare_data() -> bool:
    if not check_existing_data():
        print("📦 Preparing data...")
        if not find_and_extract_data():
            return False
        if not organize_data():
            return False
    return True

# ----------------------------
# Dataset
# ----------------------------
class KeplerDataLoader(Dataset):
    """
    PURPOSE: DATA LOADER FOR KEPLER LIGHT CURVES
    INPUT: PATH TO DIRECTORY WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    """
    def __init__(self, filepath):
        self.flist_global = np.sort(glob.glob(os.path.join(filepath, '*global.npy')))
        self.flist_local  = np.sort(glob.glob(os.path.join(filepath, '*local.npy')))
        self.flist_info   = np.sort(glob.glob(os.path.join(filepath, '*info.npy')))
        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2]
                            for x in self.flist_global])
        # Safety: ensure alignment by basename prefixes
        assert len(self.flist_global) == len(self.flist_local) == len(self.flist_info), \
            "Mismatch in file counts for global/local/info"

    def __len__(self):
        return len(self.flist_global)

    def __getitem__(self, idx):
        data_global = np.load(self.flist_global[idx])
        data_local  = np.load(self.flist_local[idx])
        data_info   = np.load(self.flist_info[idx])  # [.., label] at index 5
        return (data_local, data_global), data_info[5]

# ----------------------------
# Model (dynamic in_features via probing)
# ----------------------------
class AstronetModel(nn.Module):
    def __init__(self, local_len: int, global_len: int):
        super().__init__()
        # Convolutional stacks
        self.fc_global = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        self.fc_local = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )
        # Probe to compute in_features dynamically
        with torch.no_grad():
            g = torch.zeros(1, 1, global_len)
            l = torch.zeros(1, 1, local_len)
            g_out = torch.flatten(self.fc_global(g), 1)
            l_out = torch.flatten(self.fc_local(l), 1)
            in_features = g_out.size(1) + l_out.size(1)
        # Classification head (no sigmoid here; we output logits)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x_local, x_global):
        out_g = torch.flatten(self.fc_global(x_global), 1)
        out_l = torch.flatten(self.fc_local(x_local), 1)
        fused = torch.cat([out_g, out_l], dim=1)
        logits = self.head(fused)
        return logits  # logits (no sigmoid)

# ----------------------------
# Helpers
# ----------------------------

def estimate_pos_weight(train_dir: str) -> float:
    """Estimate pos_weight = N_neg / N_pos from info files in a split."""
    info_files = np.sort(glob.glob(os.path.join(train_dir, '*info.npy')))
    if len(info_files) == 0:
        return 1.0
    labels = []
    for fp in info_files:
        try:
            arr = np.load(fp)
            labels.append(int(arr[5]))
        except Exception:
            continue
    labels = np.array(labels)
    pos = int(labels.sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / max(pos, 1), 1.0)


def get_example_lengths(dataset: Dataset):
    """Peek one sample to infer local/global lengths."""
    (loc, glob), _ = dataset[0]
    return int(len(loc)), int(len(glob))

# ----------------------------
# Training loop
# ----------------------------

def train_astronet():
    print("Training AstroNet (improved)...")
    print(f"Device: {DEVICE}")

    set_seed(42, deterministic=True)

    # Data
    train_ds = KeplerDataLoader(os.path.join(DATA_DIR, 'train'))
    val_ds   = KeplerDataLoader(os.path.join(DATA_DIR, 'val'))
    test_ds  = KeplerDataLoader(os.path.join(DATA_DIR, 'test'))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    # Dynamic lengths → build model
    local_len, global_len = get_example_lengths(train_ds)
    model = AstronetModel(local_len=local_len, global_len=global_len).to(DEVICE)

    # Loss (logits) + class imbalance handling
    pw = estimate_pos_weight(os.path.join(DATA_DIR, 'train'))
    print(f"pos_weight (neg/pos) ≈ {pw:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))

    # Optimizer / Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Bookkeeping
    best_ap = -1.0
    best_state = None
    best_threshold = 0.5
    bad_epochs = 0

    train_losses, val_losses, accuracies, avg_precisions = [], [], [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ------------------ TRAIN ------------------
        model.train()
        running = 0.0
        for (x_local, x_global), target in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            x_local = x_local.unsqueeze(1).float().to(DEVICE)
            x_global = x_global.unsqueeze(1).float().to(DEVICE)
            target = target.unsqueeze(1).float().to(DEVICE)

            # Random time flip (vectorized)
            flip_mask = torch.rand(x_local.size(0), device=DEVICE) < 0.5
            if flip_mask.any():
                idx = flip_mask.nonzero(as_tuple=False).squeeze(1)
                x_local[idx]  = torch.flip(x_local[idx],  dims=[-1])
                x_global[idx] = torch.flip(x_global[idx], dims=[-1])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x_local, x_global)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        # ------------------ VALIDATE ------------------
        model.eval()
        val_running = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for (x_local, x_global), target in val_loader:
                x_local = x_local.unsqueeze(1).float().to(DEVICE)
                x_global = x_global.unsqueeze(1).float().to(DEVICE)
                target = target.unsqueeze(1).float().to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(x_local, x_global)
                    loss = criterion(logits, target)
                probs = torch.sigmoid(logits)
                val_running += loss.item()
                all_probs.extend(probs.view(-1).cpu().tolist())
                all_targets.extend(target.view(-1).cpu().tolist())
        val_loss = val_running / max(len(val_loader), 1)

        # Metrics (threshold-free + 0.5 for reference)
        ap = average_precision_score(all_targets, all_probs)
        acc_05 = accuracy_score(all_targets, (np.array(all_probs) >= 0.5).astype(int))

        # Early stopping selection: choose threshold on VAL by F1
        ts = np.linspace(0.0, 1.0, 201)
        f1s = [f1_score(all_targets, (np.array(all_probs) >= t).astype(int), zero_division=0) for t in ts]
        t_best = float(ts[int(np.argmax(f1s))])

        # Bookkeeping
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(acc_05)
        avg_precisions.append(ap)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val={val_loss:.4f}, Acc@0.5={acc_05:.4f}, AP={ap:.4f}, t*={t_best:.3f}")

        # Early stopping logic on AP
        if ap > best_ap + 1e-4:
            best_ap = ap
            best_state = deepcopy(model.state_dict())
            best_threshold = t_best
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best AP={best_ap:.4f})")
                break
        scheduler.step()

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Val AP={best_ap:.4f}, Val-derived threshold={best_threshold:.3f}")

    # --------------- TEST EVALUATION ---------------
    model.eval()
    test_running = 0.0
    test_probs, test_targets = [], []
    with torch.no_grad():
        for (x_local, x_global), target in test_loader:
            x_local = x_local.unsqueeze(1).float().to(DEVICE)
            x_global = x_global.unsqueeze(1).float().to(DEVICE)
            target = target.unsqueeze(1).float().to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x_local, x_global)
                loss = criterion(logits, target)
            probs = torch.sigmoid(logits)
            test_running += loss.item()
            test_probs.extend(probs.view(-1).cpu().tolist())
            test_targets.extend(target.view(-1).cpu().tolist())
    test_loss = test_running / max(len(test_loader), 1)

    # Threshold-free and thresholded metrics on TEST
    test_ap = average_precision_score(test_targets, test_probs)
    test_auc = roc_auc_score(test_targets, test_probs)
    test_pred_05 = (np.array(test_probs) >= 0.5).astype(int)
    test_acc_05 = accuracy_score(test_targets, test_pred_05)
    test_pred_best = (np.array(test_probs) >= best_threshold).astype(int)
    test_acc_best = accuracy_score(test_targets, test_pred_best)

    print(f"Test Results → Loss={test_loss:.4f}, Acc@0.5={test_acc_05:.4f}, Acc@t*={test_acc_best:.4f}, AP={test_ap:.4f}, AUC={test_auc:.4f}")

    # --------------- PLOTS ---------------
    # Recompute full VAL arrays (already have all_probs/targets for last epoch but we want best checkpoint effects)
    val_probs_full, val_targets_full = [], []
    with torch.no_grad():
        for (x_local, x_global), target in val_loader:
            x_local = x_local.unsqueeze(1).float().to(DEVICE)
            x_global = x_global.unsqueeze(1).float().to(DEVICE)
            target = target.unsqueeze(1).float().to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x_local, x_global)
            probs = torch.sigmoid(logits)
            val_probs_full.extend(probs.view(-1).cpu().tolist())
            val_targets_full.extend(target.view(-1).cpu().tolist())

    # ROC (Val & Test)
    val_p = np.array(val_probs_full); val_y = np.array(val_targets_full).astype(int)
    test_p = np.array(test_probs);    test_y = np.array(test_targets).astype(int)

    fpr_v, tpr_v, _ = roc_curve(val_y, val_p); auc_v = roc_auc_score(val_y, val_p)
    fpr_t, tpr_t, _ = roc_curve(test_y, test_p); auc_t = roc_auc_score(test_y, test_p)
    plt.figure(); plt.plot(fpr_v, tpr_v, label=f'VAL ROC (AUC={auc_v:.3f})')
    plt.plot(fpr_t, tpr_t, label=f'TEST ROC (AUC={auc_t:.3f})'); plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves (Val & Test)')
    plt.legend(); plt.grid(True); plt.show()

    # PR (Val & Test)
    prec_v, rec_v, _ = precision_recall_curve(val_y, val_p); ap_v = average_precision_score(val_y, val_p)
    prec_t, rec_t, _ = precision_recall_curve(test_y, test_p); ap_t = average_precision_score(test_y, test_p)
    plt.figure(); plt.plot(rec_v, prec_v, label=f'VAL PR (AP={ap_v:.3f})')
    plt.plot(rec_t, prec_t, label=f'TEST PR (AP={ap_t:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curves'); plt.legend(); plt.grid(True); plt.show()

    # Threshold sweep (Val)
    ts = np.linspace(0.0, 1.0, 201)
    accs, precs, recs, f1s = [], [], [], []
    for t in ts:
        pred = (val_p >= t).astype(int)
        accs.append(accuracy_score(val_y, pred))
        precs.append(precision_score(val_y, pred, zero_division=0))
        recs.append(recall_score(val_y, pred, zero_division=0))
        f1s.append(f1_score(val_y, pred, zero_division=0))
    plt.figure(); plt.plot(ts, accs, label='Accuracy'); plt.plot(ts, precs, label='Precision')
    plt.plot(ts, recs, label='Recall'); plt.plot(ts, f1s, label='F1')
    plt.axvline(best_threshold, linestyle='--', color='k', label=f't*={best_threshold:.2f}')
    plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Validation Metrics vs Threshold')
    plt.legend(); plt.grid(True); plt.show()

    # Confusion matrix (Test @ t*)
    cm = confusion_matrix(test_y, (test_p >= best_threshold).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(); disp.plot(values_format='d', cmap='Blues'); plt.title(f'Confusion Matrix (Test) @ t={best_threshold:.2f}')
    plt.grid(False); plt.show()

    # Calibration curve (Test)
    prob_true, prob_pred = calibration_curve(test_y, test_p, n_bins=10, strategy='uniform')
    brier = brier_score_loss(test_y, test_p)
    plt.figure(); plt.plot([0,1],[0,1], '--', alpha=0.5, label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label=f'Test (Brier={brier:.3f})')
    plt.xlabel('Predicted probability'); plt.ylabel('Empirical frequency'); plt.title('Calibration / Reliability Curve (Test)')
    plt.legend(); plt.grid(True); plt.show()

    # Training curves
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(); plt.plot(epochs, train_losses, label='Train Loss'); plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training & Validation Loss'); plt.legend(); plt.grid(True); plt.show()

    plt.figure(); plt.plot(epochs, accuracies, label='Val Acc@0.5'); plt.plot(epochs, avg_precisions, label='Val AP')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Accuracy@0.5 and AP'); plt.legend(); plt.grid(True); plt.show()

    return model, best_threshold

# ============================
# Main
# ============================
if __name__ == "__main__":
    print("ASTRONET COLAB (improved)")
    print("=" * 50)

    if check_existing_data():
        print("Using existing data...")
    else:
        print("No existing data found, preparing data...")
        from google.colab import drive  # only when needed
        mount_google_drive()
        if not prepare_data():
            raise SystemExit("Data preparation failed")
        print("Data ready!")

    model, best_t = train_astronet()
    print("Training complete! Best validation threshold:", best_t)