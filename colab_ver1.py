########################################
########### IMPORT PACKAGES ############
########################################

### standard packages
import os
import numpy as np
import glob 
from tqdm import tqdm 
from random import random
import zipfile
import tarfile

import csv

### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
import random as py_random

### sklearn packages
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### plotting packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================
# Paths & constants
# ============================
DATA_DIR = '/content/astronet_data'
DRIVE_DIR = '/content/drive/MyDrive/astronet_data'
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 40
LR = 1e-5

# ============================
# Data presence / preparation
# ============================

# Check if data already exists in Colab environment. 
# DATA_DIR should contain 'train', 'val', 'test' subdirs with .npy files.
def check_existing_data() -> bool:
    if not os.path.exists(DATA_DIR):
        return False
    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            return False
        if len(glob.glob(os.path.join(split_dir, '*.npy'))) == 0:
            return False
    total = 0
    for split in ('train', 'val', 'test'):
        total += len(glob.glob(os.path.join(DATA_DIR, split, '*.npy'))) // 3
    print(f"Found existing data: {total} samples")
    print(f"Data directory: {DATA_DIR}")
    return total > 0


def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')

# Find and extract .tar.gz or .zip files from DRIVE_DIR to DATA_DIR
def find_and_extract_data() -> bool:
    files = []
    for pattern in ('*.tar.gz', '*.zip'):
        files.extend(glob.glob(os.path.join(DRIVE_DIR, pattern)))
    if not files:
        print('No data files found! Please upload AstroNet data to Google Drive.')
        return False
    print(f'Found {len(files)} files')
    os.makedirs(DATA_DIR, exist_ok=True)
    for fp in tqdm(files, desc='Extracting'):
        if fp.endswith('.zip'):
            with zipfile.ZipFile(fp, 'r') as zf:
                zf.extractall(DATA_DIR)
        else:
            with tarfile.open(fp, 'r:gz') as tf:
                tf.extractall(DATA_DIR)
    return True


def organize_data() -> bool: 
    for split in ('train', 'val', 'test'):  # create split dirs
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)
    groups = {}

    for root, _, files in os.walk(DATA_DIR):
        rel = os.path.relpath(root, DATA_DIR)  # e.g., train/batch1
        parts = rel.split(os.sep) if rel != '.' else [] # e.g., ['train', 'batch1']
        split = 'train' if 'train' in parts else 'val' if 'val' in parts else 'test' if 'test' in parts else None  
        if split is None:
            continue

        for fname in files:
            if not fname.endswith('.npy') or 'cen' in fname:
                continue  # skip non-npy and centroid files
            toks = fname.split('_')  # e.g., ['kplr', '001433399', '02', 'local.npy']
            if len(toks) < 3:
                continue
            sid = '_'.join(toks[:3])  # e.g., 'kplr_001433399_02'
            key = (split, sid)  # e.g., ('train', 'kplr_001433399_02')
            fpath = os.path.join(root, fname)  # full path
            d = groups.setdefault(key, {})
            if fname.endswith('_global.npy'): d['global'] = fpath
            elif fname.endswith('_local.npy'): d['local'] = fpath
            elif fname.endswith('_info.npy'): d['info'] = fpath
    kept = 0
    for (split, _sid), d in groups.items():
        if all(k in d for k in ('global', 'local', 'info')):
            for f in d.values():
                dst = os.path.join(DATA_DIR, split, os.path.basename(f))
                if f != dst:
                    shutil.copy2(f, dst)
            kept += 1
    print(f'✅ Total: {kept} samples organized')
    return kept > 0


def prepare_data() -> bool:
    if check_existing_data():
        return True
    print('Preparing data...')
    if not find_and_extract_data():
        return False
    return organize_data()

# ============================
# Dataset
# ============================
class KeplerDataLoader(Dataset):
    """Returns ((local, global), label) from a split directory."""
    def __init__(self, path: str):
        fg = sorted(glob.glob(os.path.join(path, '*global.npy')))
        fl = sorted(glob.glob(os.path.join(path, '*local.npy')))
        fi = sorted(glob.glob(os.path.join(path, '*info.npy')))

        def base_id(fp: str) -> str:
            # e.g., 'kplr_001433399_02_global.npy' → 'kplr_001433399_02'
            toks = os.path.basename(fp).split('_')
            return '_'.join(toks[:3]) if len(toks) >= 3 else ''

        fg_ids = {base_id(f): f for f in fg}
        fl_ids = {base_id(f): f for f in fl}
        fi_ids = {base_id(f): f for f in fi}

        common = sorted(set(fg_ids) & set(fl_ids) & set(fi_ids))
        assert len(common) > 0, 'No matching triplets found in split'

        # aligned lists (same index → same sample)
        self.fg = [fg_ids[sid] for sid in common]
        self.fl = [fl_ids[sid] for sid in common]
        self.fi = [fi_ids[sid] for sid in common]

    def __len__(self):
        return len(self.fg)

    def __getitem__(self, idx):
        g = np.load(self.fg[idx])
        l = np.load(self.fl[idx])
        info = np.load(self.fi[idx])  # label at index 5
        return (l, g), info[5]

# ============================
# Model
# ============================
class AstronetModel(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.final_layer = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),  # logits (no Sigmoid here)
        )

    def forward(self, x_local, x_global):
        g = torch.flatten(self.fc_global(x_global), 1)
        l = torch.flatten(self.fc_local(x_local), 1)
        fused = torch.cat((g, l), dim=1)
        return self.final_layer(fused)

# ============================
# Training
# ============================

def train_astronet():
    print('Training AstroNet...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    py_random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    train_loader = DataLoader(KeplerDataLoader(os.path.join(DATA_DIR, 'train')), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(KeplerDataLoader(os.path.join(DATA_DIR, 'val')),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Estimate class imbalance from training split for pos_weight
    train_dataset = train_loader.dataset
    pos = 0; neg = 0
    for info_fp in train_dataset.fi:
        label = float(np.load(info_fp)[5])
        if label >= 0.5:
            pos += 1
        else:
            neg += 1
    pos_weight = torch.tensor([ (neg / max(1, pos)) if pos > 0 else 1.0 ], device=device, dtype=torch.float32)

    print(f'Train: {len(train_loader.dataset)} samples')
    print(f'Val: {len(val_loader.dataset)} samples')

    model = AstronetModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, accuracies, avg_precisions = [], [], [], []

    best_ap = -1.0
    best_path = os.path.join(DATA_DIR, 'best_astronet.pt')
    patience, since_best = 6, 0

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for (x_local, x_global), target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            x_local  = x_local.unsqueeze(1).float().to(device)
            x_global = x_global.unsqueeze(1).float().to(device)
            target   = target.unsqueeze(1).float().to(device)
        

            # 50% random time flip (vectorized)
            flip_mask = torch.rand(x_local.size(0), device=device) < 0.5
            if flip_mask.any():
                idx = flip_mask.nonzero(as_tuple=False).squeeze(1)
                x_local[idx]  = torch.flip(x_local[idx],  dims=[-1])
                x_global[idx] = torch.flip(x_global[idx], dims=[-1])

            optimizer.zero_grad(set_to_none=True)
            out  = model(x_local, x_global)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        val_running, all_preds, all_targs = 0.0, [], []
        with torch.no_grad():
            for (x_local, x_global), target in val_loader:
                x_local  = x_local.unsqueeze(1).float().to(device)
                x_global = x_global.unsqueeze(1).float().to(device)
                target   = target.unsqueeze(1).float().to(device)
                out  = model(x_local, x_global)       # logits
                loss = criterion(out, target)
                val_running += loss.item()
                probs = torch.sigmoid(out)
                all_preds.extend(probs.view(-1).cpu().tolist())
                all_targs.extend(target.view(-1).cpu().tolist())

        train_losses.append(running / max(1, len(train_loader)))
        val_losses.append(val_running / max(1, len(val_loader)))
        ap = average_precision_score(all_targs, all_preds)
        acc = accuracy_score(all_targs, [1 if p > 0.5 else 0 for p in all_preds])
        accuracies.append(acc)
        avg_precisions.append(ap)
        print(f'Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}, Acc={acc:.4f}, AP={ap:.4f}')


    # =============================
    # Post-training: load best & evaluate on VAL (and optional TEST)
    # =============================
    # Reload best checkpoint (by validation AP)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best checkpoint from {best_path} (best AP={best_ap:.4f})")

    # Helper to run a full loader and collect probs/labels
    def collect_preds(loader):
        probs_all, labels_all = [], []
        model.eval()
        with torch.no_grad():
            for (x_local, x_global), target in loader:
                x_local  = x_local.unsqueeze(1).float().to(device)
                x_global = x_global.unsqueeze(1).float().to(device)
                target   = target.unsqueeze(1).float().to(device)
                logits = model(x_local, x_global)
                probs  = torch.sigmoid(logits)
                probs_all.extend(probs.view(-1).cpu().tolist())
                labels_all.extend(target.view(-1).cpu().tolist())
        return np.array(probs_all, dtype=float), np.array(labels_all, dtype=float)

    # Collect on validation set
    val_probs, val_labels = collect_preds(val_loader)

    # Metrics: PR, AP, Accuracy@0.5, confusion at multiple thresholds
    P, R, _ = precision_recall_curve(val_labels, val_probs)
    AP_val  = average_precision_score(val_labels, val_probs)
    acc_val = accuracy_score(val_labels, (val_probs >= 0.5).astype(int))
    print(f"Final (VAL): AP={AP_val:.4f}, Acc@0.5={acc_val:.4f}")

    # Confusion matrices across thresholds (like original)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    conf_rows = [("threshold","tn","fp","fn","tp","precision","recall")]
    for t in thresholds:
        pred_bin = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_labels, pred_bin).ravel()
        prec = precision_score(val_labels, pred_bin, zero_division=0)
        rec  = recall_score(val_labels, pred_bin, zero_division=0)
        print(f"thresh={t:.2f} → precision={prec:.4f}, recall={rec:.4f} | TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        conf_rows.append((t, tn, fp, fn, tp, prec, rec))

    # =============================
    # Save CSVs (predictions and epoch curves)
    # =============================
    run_dir = os.path.join(DATA_DIR, 'runs')
    os.makedirs(run_dir, exist_ok=True)

    # 1) Validation predictions CSV
    val_csv = os.path.join(run_dir, 'val_predictions.csv')
    with open(val_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prob', 'label'])
        for p, y in zip(val_probs, val_labels):
            writer.writerow([f"{p:.6f}", int(y)])
    print(f"Saved: {val_csv}")

    # 2) Confusion-by-threshold CSV
    conf_csv = os.path.join(run_dir, 'val_confusions.csv')
    with open(conf_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(conf_rows)
    print(f"Saved: {conf_csv}")

    # 3) Epoch metrics CSV (train/val curves)
    epoch_csv = os.path.join(run_dir, 'epoch_metrics.csv')
    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss','val_acc','val_ap'])
        for i, (tr, vl, ac, ap_) in enumerate(zip(train_losses, val_losses, accuracies, avg_precisions), start=1):
            writer.writerow([i, f"{tr:.6f}", f"{vl:.6f}", f"{ac:.6f}", f"{ap_:.6f}"])
    print(f"Saved: {epoch_csv}")

    # =============================
    # 2×2 Figure (like original): PR, Loss, AP/epoch, Acc/epoch
    # =============================
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # (1) PR curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(R, P, linewidth=2.5, color='black')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'PR (Val)  AP={AP_val:.3f}')
    ax1.grid(True, alpha=0.3)

    # (2) Loss curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(train_losses)+1), [x*BATCH_SIZE for x in train_losses], label='Train (batch-loss)', linewidth=2.0)
    ax2.plot(range(1, len(val_losses)+1),   [x*BATCH_SIZE for x in val_losses],   label='Val (batch-loss)',   linewidth=2.0)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss per batch')
    ax2.set_title('Loss (Train vs Val)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # (3) AP per epoch
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(range(1, len(avg_precisions)+1), avg_precisions, color='orangered', linewidth=2.0)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('AP')
    ax3.set_title('Validation AP per Epoch')
    ax3.grid(True, alpha=0.3)

    # (4) Accuracy per epoch
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(1, len(accuracies)+1), accuracies, color='cadetblue', linewidth=2.0)
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy per Epoch')
    ax4.grid(True, alpha=0.3)

    plot_pdf = os.path.join(run_dir, 'summary_plots.pdf')
    plt.tight_layout()
    plt.savefig(plot_pdf, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {plot_pdf}")

    # =============================
    # Optional: Evaluate TEST set if present
    # =============================
    test_dir = os.path.join(DATA_DIR, 'test')
    if os.path.isdir(test_dir) and len(glob.glob(os.path.join(test_dir, '*.npy'))) > 0:
        test_loader = DataLoader(KeplerDataLoader(test_dir), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_probs, test_labels = collect_preds(test_loader)
        AP_test  = average_precision_score(test_labels, test_probs)
        acc_test = accuracy_score(test_labels, (test_probs >= 0.5).astype(int))
        print(f"Final (TEST): AP={AP_test:.4f}, Acc@0.5={acc_test:.4f}")
        # Save test predictions
        test_csv = os.path.join(run_dir, 'test_predictions.csv')
        with open(test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prob', 'label'])
            for p, y in zip(test_probs, test_labels):
                writer.writerow([f"{p:.6f}", int(y)])
        print(f"Saved: {test_csv}")

    return model

# ============================
# Main
# ============================
if __name__ == '__main__':
    print('ASTRONET COLAB')
    print('=' * 50)

    if check_existing_data():
        print('Existing data found')
        model = train_astronet()
        print('Training complete!')
    else:
        print('No existing data found')
        mount_google_drive()
        if prepare_data():
            print('Data ready!')
            model = train_astronet()
            print('Training complete!')
        else:
            print('Data preparation failed')
