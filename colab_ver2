#!/usr/bin/env python3
# 
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import tarfile
import shutil

# -----------------------------------------------------------------------------
# - Looks for zipped data in Google Drive (/content/drive/MyDrive/astronet_data)
# - Extracts them into /content/astronet_data
# - Organizes files into train/val/test folders (only complete triplets kept)
# - Trains the original AstroNet model (same architecture and behavior)
# The goal: one-click run in Colab, with clear, readable steps.
# -----------------------------------------------------------------------------

# =============================================================================
# CHECK EXISTING DATA
# =============================================================================
def check_existing_data():
    """
    Quick check: do we already have prepared data in /content/astronet_data?
    - We require train/val/test folders
    - Each must contain .npy files
    - We report the total number of samples (triplets)
    """
    data_dir = '/content/astronet_data'
    
    if not os.path.exists(data_dir):
        return False
    
    # Check that all split folders exist and have files
    required_splits = ['train', 'val', 'test']
    for split in required_splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            return False
        
        # Look for any .npy files (we only count completeness later)
        npy_files = glob.glob(os.path.join(split_dir, '*.npy'))
        if len(npy_files) == 0:
            return False
    
    # Count how many complete samples we likely have (3 files per sample)
    total_samples = 0
    for split in required_splits:
        split_dir = os.path.join(data_dir, split)
        npy_files = glob.glob(os.path.join(split_dir, '*.npy'))
        total_samples += len(npy_files) // 3
    
    print(f"Found existing data: {total_samples} samples")
    print(f"Data directory: {data_dir}")
    
    return total_samples > 0

# =============================================================================
# GOOGLE DRIVE SETUP
# =============================================================================
def mount_google_drive():
    """
    Colab step: mount your Google Drive at /content/drive so we can read zips.
    """
    from google.colab import drive
    drive.mount('/content/drive')


def find_and_extract_data():
    """
    Find archives in Drive and extract them into /content/astronet_data.
    We search only the astronet_data folder (no recursion) for:
    - *.tar.gz and *.zip
    """
    print("Finding data files...")
    
    # Patterns to match zip/tar files in Drive (single folder)
    patterns = ['*.tar.gz', '*.zip']
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(os.path.join('/content/drive/MyDrive/astronet_data/', pattern)))
    
    if not found_files:
        print("No data files found! Please upload your AstroNet data to Google Drive.")
        return False
    
    print(f"Found {len(found_files)} files")
    
    # Extract into Colab's fast local disk
    extract_dir = '/content/astronet_data'
    os.makedirs(extract_dir, exist_ok=True)
    
    for file_path in tqdm(found_files, desc="Extracting"):
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
    
    return True


def organize_data():
    """
    Walk the extracted files and copy only complete samples into:
    /content/astronet_data/{train,val,test}
    A complete sample means three files with the same ID prefix:
    *_global.npy, *_local.npy, *_info.npy
    We infer the split from the folder path (train/val/test).
    """
    source_dir = '/content/astronet_data'
    target_dir = '/content/astronet_data'  # organize in-place for simplicity
    
    # Make sure split folders exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # Group per (split, sample_id)
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
    
    # Copy only complete triplets into target_dir/split
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

# Simple orchestrator

def prepare_data():
    """
    Prepare data if needed:
    - If already present, do nothing
    - Else: mount Drive → extract → organize
    """
    if not check_existing_data():
        print("📦 Preparing data...")
        if not find_and_extract_data():
            return False
        if not organize_data():
            return False
    return True

# =============================================================================
# DATA LOADER
# =============================================================================
class KeplerDataLoader(Dataset):
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTORY WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, filepath):
        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global = np.sort(glob.glob(os.path.join(filepath, '*global.npy')))
        self.flist_local = np.sort(glob.glob(os.path.join(filepath, '*local.npy')))
        self.flist_info = np.sort(glob.glob(os.path.join(filepath, '*info.npy')))

        ### ids = {KIC}_{TCE}
        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2] for x in self.flist_global])

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ### grab local and global views
        data_global = np.load(self.flist_global[idx])
        data_local = np.load(self.flist_local[idx])

        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = np.load(self.flist_info[idx])

        return (data_local, data_global), data_info[5]

# =============================================================================
# ASTRONET MODEL
# =============================================================================
class AstronetModel(nn.Module):

    '''
    
    PURPOSE: DEFINE ASTRONET MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES
    OUTPUT: BINARY CLASSIFIER
    
    '''
    
    def __init__(self):
        super(AstronetModel, self).__init__()
        
        # Global branch: processes the long light curve (overall context)
        self.fc_global = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        
        # Local branch: processes the short, detailed transit window
        self.fc_local = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )
        
        # After both branches, we flatten and concatenate features, then classify
        self.final_layer = nn.Sequential(
            nn.Linear(16576, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    
    def forward(self, x_local, x_global):
  
        # Global path → convs → flatten to (batch, features)
        out_global = self.fc_global(x_global)
        out_global = torch.flatten(out_global, 1)
        
        # Local path → convs → flatten to (batch, features)
        out_local = self.fc_local(x_local)
        out_local  = torch.flatten(out_local, 1)
        
        # Concatenate features from both paths, then classify
        fused = torch.cat((out_global, out_local), dim=1)
        return self.final_layer(fused)
        

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
def invert_tensor(tensor):
    """Time-reverse a 1D tensor safely on any device."""
    return torch.flip(tensor, dims=[0])

# =============================================================================
# TRAINING
# =============================================================================
def train_astronet():
    """
    Train the AstroNet model.
    Loop over epochs:
    - get batches from DataLoader
    - optional augmentation (flip signals left↔right)
    - forward → loss → backward → optimizer step
    - compute validation metrics (loss, accuracy, average precision)
    """
    print("Training AstroNet...")
    
    # Use GPU if available; else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data loaders read from prepared folders
    train_loader = DataLoader(KeplerDataLoader('/content/astronet_data/train'), batch_size=64, shuffle=True, num_workers=16)
    val_loader = DataLoader(KeplerDataLoader('/content/astronet_data/val'), batch_size=64, shuffle=False, num_workers=16)
    test_loader = DataLoader(KeplerDataLoader('/content/astronet_data/test'), batch_size=64, shuffle=False, num_workers=16)
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    
    # Model + loss + optimizer
    model = AstronetModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Bookkeeping for plotting
    n_epochs = 50
    train_losses, val_losses, accuracies, avg_precisions = [], [], [], []
    
    for epoch in range(n_epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # Unpack and add a channel dimension (1) for Conv1d
            x_local, x_global = data
            x_local = x_local.unsqueeze(1).float().to(device)
            x_global = x_global.unsqueeze(1).float().to(device)
            target = target.unsqueeze(1).float().to(device)
            
            # Data augmentation: randomly flip both views (50% chance)
            flip_mask = torch.rand(x_local.size(0), device=x_local.device) < 0.5  # get 64 true or false mark
            if flip_mask.any():
                idx = flip_mask.nonzero().squeeze(1)  # 1D indices to flip 
                x_local[idx]  = torch.flip(x_local[idx],  dims=[-1])
                x_global[idx] = torch.flip(x_global[idx], dims=[-1])

            # Zero grad → forward → loss → backward → step
            optimizer.zero_grad() 
            output = model(x_local, x_global)  # forward pass
            loss = criterion(output, target)  
            loss.backward()  # backpropagation, store gradients
            optimizer.step()  # update weights
            train_loss += loss.item()  # accumulate loss 
        
        # ---- Validation phase ----
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                x_local, x_global = data
                x_local = x_local.unsqueeze(1).float().to(device)
                x_global = x_global.unsqueeze(1).float().to(device)
                target = target.unsqueeze(1).float().to(device)

                output = model(x_local, x_global)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                all_preds.extend(output.view(-1).cpu().tolist())
                all_targets.extend(target.view(-1).cpu().tolist())
        
        # Convert to epoch metrics
        avg_precision = average_precision_score(all_targets, all_preds)
        predictions = [1 if p > 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_targets, predictions)
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        accuracies.append(accuracy)
        avg_precisions.append(avg_precision)
        print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}, Acc={accuracy:.4f}, AP={avg_precision:.4f}")
    
    # ---- After training: evaluate on test set ----
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            x_local, x_global = data
            x_local = x_local.unsqueeze(1).float().to(device)
            x_global = x_global.unsqueeze(1).float().to(device)
            target = target.unsqueeze(1).float().to(device)

            output = model(x_local, x_global)
            loss = criterion(output, target)
            test_loss += loss.item()
            test_preds.extend(output.view(-1).cpu().tolist())
            test_targets.extend(target.view(-1).cpu().tolist())

    # Compute test metrics
    test_avg_precision = average_precision_score(test_targets, test_preds)
    test_predictions = [1 if p > 0.5 else 0 for p in test_preds]
    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f"Test Results → Loss={test_loss/len(test_loader):.4f}, Acc={test_accuracy:.4f}, AP={test_avg_precision:.4f}")

    # ============================
    # Plot training curves (Loss)
    # ============================
    epochs = list(range(1, n_epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ================================
    # Plot accuracy over epochs
    # ================================
    plt.figure()
    plt.plot(epochs, accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # =====================================
    # Plot Average Precision (Val) per epoch
    # =====================================
    plt.figure()
    plt.plot(epochs, avg_precisions, label='Val Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.title('Validation Average Precision per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # =====================================
    # Plot Precision-Recall curve on Test
    # =====================================
    precision, recall, _ = precision_recall_curve(test_targets, test_preds)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve (Test set)')
    plt.grid(True)
    plt.show()

    return model

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("ASTRONET COLAB")
    print("=" * 50)
    
    # Check if data already exists
    if check_existing_data():
        print("Using existing data...")
        # Train model with existing data
        model = train_astronet()
        print("Training complete!")
    else:
        print("No existing data found, preparing data...")
        
        # Mount Google Drive
        mount_google_drive()
        
        # Prepare data
        if prepare_data():
            print("Data ready!")
            # Train model
            model = train_astronet()
            print("Training complete!")
        else:
            print("Data preparation failed")
