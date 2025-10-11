#!/usr/bin/env python3
"""
AstroNet: Exoplanet Detection with Deep Learning
===============================================

"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import glob
import shutil
import zipfile
import tarfile
from random import random

# Data science
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Machine learning metrics
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix,
    precision_score, 
    recall_score, 
    accuracy_score
)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths for Google Colab
DATA_DIR = '/content/astronet_data'
DRIVE_DIR = '/content/drive/MyDrive/astronet_data'

# Training hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 12
EPOCHS = 50
LEARNING_RATE = 1e-5

# =============================================================================
# DATA PREPARATION (Google Colab specific)
# =============================================================================

def check_existing_data() -> bool:
    """
    Check if data is already prepared in Colab environment.
    
    Returns:
        bool: True if data exists and is properly organized
    """
    if not os.path.exists(DATA_DIR):
        return False
    
    # Check that all required folders exist and contain .npy files
    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            return False
        if len(glob.glob(os.path.join(split_dir, '*.npy'))) == 0:
            return False
    
    # Count total samples (3 files per sample: global, local, info)
    total_samples = 0
    for split in ('train', 'val', 'test'):
        total_samples += len(glob.glob(os.path.join(DATA_DIR, split, '*.npy'))) // 3
    
    print(f"Found existing data: {total_samples} samples")
    print(f"Data directory: {DATA_DIR}")
    return total_samples > 0

def mount_google_drive():
    """Mount Google Drive to access uploaded data files."""
    from google.colab import drive
    drive.mount('/content/drive')

def find_and_extract_data() -> bool:
    """
    Find and extract compressed data files from Google Drive.
    
    Returns:
        bool: True if extraction successful
    """
    # Look for compressed files in the Drive directory
    files = []
    for pattern in ('*.tar.gz', '*.zip'):
        files.extend(glob.glob(os.path.join(DRIVE_DIR, pattern)))
    
    if not files:
        print('No data files found! Please upload AstroNet data to Google Drive.')
        return False
    
    print(f'Found {len(files)} files')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Extract each file
    for file_path in tqdm(files, desc='Extracting'):
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(DATA_DIR)
        else:  # .tar.gz
            with tarfile.open(file_path, 'r:gz') as tf:
                tf.extractall(DATA_DIR)
    
    return True

def organize_data() -> bool:
    """
    Organize extracted files into proper train/val/test structure.
    
    Each sample needs 3 files: *_global.npy, *_local.npy, *_info.npy
    Only complete samples (all 3 files present) are kept.
    
    Returns:
        bool: True if organization successful
    """
    # Create output directories
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)
    
    # Group files by sample ID and split
    groups = {}
    
    # Walk through all extracted files
    for root, _, files in os.walk(DATA_DIR):
        # Determine which split this folder belongs to
        rel_path = os.path.relpath(root, DATA_DIR)
        parts = rel_path.split(os.sep) if rel_path != '.' else []
        
        # Check if this is a train/val/test folder
        split = None
        if 'train' in parts:
            split = 'train'
        elif 'val' in parts:
            split = 'val'
        elif 'test' in parts:
            split = 'test'
        
        if split is None:
            continue
        
        # Process each .npy file in this folder
        for filename in files:
            # Skip non-npy files and centroid files
            if not filename.endswith('.npy') or 'cen' in filename:
                continue
            
            # Extract sample ID from filename (e.g., 'kplr_001433399_02')
            name_parts = filename.split('_')
            if len(name_parts) < 3:
                continue
            
            sample_id = '_'.join(name_parts[:3])
            key = (split, sample_id)
            file_path = os.path.join(root, filename)
            
            # Group by file type
            if filename.endswith('_global.npy'):
                groups.setdefault(key, {})['global'] = file_path
            elif filename.endswith('_local.npy'):
                groups.setdefault(key, {})['local'] = file_path
            elif filename.endswith('_info.npy'):
                groups.setdefault(key, {})['info'] = file_path
    
    # Copy only complete samples (all 3 files present)
    kept_samples = 0
    for (split, _sample_id), file_dict in groups.items():
        if all(file_type in file_dict for file_type in ('global', 'local', 'info')):
            # Copy all 3 files to the organized directory
            for file_type, source_path in file_dict.items():
                dest_path = os.path.join(DATA_DIR, split, os.path.basename(source_path))
                if source_path != dest_path:
                    shutil.copy2(source_path, dest_path)
            kept_samples += 1
    
    print(f'✅ Total: {kept_samples} samples organized')
    return kept_samples > 0

def prepare_data() -> bool:
    """
    Prepare data for training. Check if data exists, otherwise extract and organize.
    
    Returns:
        bool: True if data preparation successful
    """
    if check_existing_data():
        return True
    
    print('Preparing data...')
    if not find_and_extract_data():
        return False
    
    return organize_data()

# =============================================================================
# DATASET CLASS
# =============================================================================

class KeplerDataset(Dataset):
    """
    PyTorch Dataset for loading Kepler light curve data.
    
    Each sample contains:
    - Global view: Full light curve (2001 time points)
    - Local view: Zoomed-in view around transit (201 time points)
    - Label: 1 for exoplanet, 0 for false positive
    """
    
    def __init__(self, data_path: str):
        """
        Initialize dataset from directory containing .npy files.
        
        Args:
            data_path: Path to directory with train/val/test data
        """
        # Find all data files (sorted for consistency)
        self.global_files = sorted(glob.glob(os.path.join(data_path, '*global.npy')))
        self.local_files = sorted(glob.glob(os.path.join(data_path, '*local.npy')))
        self.info_files = sorted(glob.glob(os.path.join(data_path, '*info.npy')))
        
        # Extract sample IDs for verification
        self.sample_ids = []
        for global_file in self.global_files:
            # Extract ID from filename: kplr_001433399_02_global.npy -> 001433399_02
            filename = os.path.basename(global_file)
            parts = filename.split('_')
            if len(parts) >= 3:
                sample_id = f"{parts[1]}_{parts[2]}"  # KIC_TCE
                self.sample_ids.append(sample_id)
        
        self.sample_ids = sorted(self.sample_ids)
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            tuple: ((local_view, global_view), label)
        """
        # Load the three data files for this sample
        global_data = np.load(self.global_files[idx])
        local_data = np.load(self.local_files[idx])
        info_data = np.load(self.info_files[idx])
        
        # Info file contains: [KIC, TCE, period, epoch, duration, label]
        # We only need the label (index 5)
        label = info_data[5]
        
        return (local_data, global_data), label

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class AstroNet(nn.Module):
    """
    AstroNet: Two-branch CNN for exoplanet detection.
    
    Architecture:
    1. Global branch: Processes full light curve to capture long-term patterns
    2. Local branch: Processes zoomed-in view to detect transit shapes
    3. Fusion: Combines features from both branches for final classification
    """
    
    def __init__(self):
        super(AstroNet, self).__init__()
        
        # Global branch: analyzes the full light curve (2001 time points)
        # This captures long-term patterns and overall star behavior
        self.global_branch = nn.Sequential(
            # First conv block: 1 → 16 channels, kernel=5 (looks at 5 nearby points)
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # padding=2 keeps same length
            nn.ReLU(),  # activation: allows learning non-linear patterns
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),  # reduce length by factor of 2.5
            
            # Second conv block: 16 → 32 channels
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            # Third conv block: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            # Fourth conv block: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            # Fifth conv block: 128 → 256 channels
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )
        
        # Local branch: analyzes the zoomed-in transit view (201 time points)
        # This captures the detailed shape of the transit event
        self.local_branch = nn.Sequential(
            # First conv block: 1 → 16 channels
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2),  # larger kernel for local view
            
            # Second conv block: 16 → 32 channels
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2),
        )
        
        # Fusion layer: combines features from both branches
        # Input size: 16576 (calculated from architecture)
        self.classifier = nn.Sequential(
            nn.Linear(16576, 512),  # first fully connected layer
            nn.ReLU(),
            nn.Linear(512, 512),    # second fully connected layer
            nn.ReLU(),
            nn.Linear(512, 512),    # third fully connected layer
            nn.ReLU(),
            nn.Linear(512, 1),      # output layer: 1 neuron for binary classification
            nn.Sigmoid()            # sigmoid: outputs probability between 0 and 1
        )
    
    def forward(self, local_view, global_view):
        """
        Forward pass through the network.
        
        Args:
            local_view: Local light curve view (batch_size, 1, 201)
            global_view: Global light curve view (batch_size, 1, 2001)
            
        Returns:
            torch.Tensor: Probability of exoplanet (batch_size, 1)
        """
        # Process global view through global branch
        global_features = self.global_branch(global_view)
        global_features = global_features.view(global_features.size(0), -1)  # flatten
        
        # Process local view through local branch
        local_features = self.local_branch(local_view)
        local_features = local_features.view(local_features.size(0), -1)  # flatten
        
        # Combine features from both branches
        combined_features = torch.cat([global_features, local_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output

class AstroNetGlobalOnly(nn.Module):
    """
    AstroNet with only global view (no local view).
    This model uses only the global light curve for exoplanet detection.
    """
    def __init__(self):
        super(AstroNetGlobalOnly, self).__init__()
        
        # Global branch: analyzes the full light curve (2001 time points)
        self.global_branch = nn.Sequential(
            # First conv block: 1 → 16 channels, kernel=5
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            # Second conv block: 16 → 32 channels
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            # Third conv block: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            # Fourth conv block: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=5, stride=2),
            # Fifth conv block: 128 → 256 channels
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )
        
        # Classifier: only uses global features
        # Input size: 15104 (calculated from global branch only)
        self.classifier = nn.Sequential(
            nn.Linear(15104, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, local_view, global_view):
        """
        Forward pass through the network (ignores local_view).
        Args:
            local_view: Local light curve view (ignored)
            global_view: Global light curve view (batch_size, 1, 2001)
        Returns:
            torch.Tensor: Probability of exoplanet (batch_size, 1)
        """
        # Process only global view
        global_features = self.global_branch(global_view)
        global_features = global_features.view(global_features.size(0), -1)  # flatten
        
        # Final classification using only global features
        output = self.classifier(global_features)
        return output

class AstroNetLocalOnly(nn.Module):
    """
    AstroNet with only local view (no global view).
    This model uses only the local light curve for exoplanet detection.
    """
    def __init__(self):
        super(AstroNetLocalOnly, self).__init__()
        
        # Local branch: analyzes the zoomed-in transit view (201 time points)
        self.local_branch = nn.Sequential(
            # First conv block: 1 → 16 channels
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2),  # larger kernel for local view
            # Second conv block: 16 → 32 channels
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2),
        )
        
        # Classifier: only uses local features
        # Input size: 1472 (calculated from local branch only)
        self.classifier = nn.Sequential(
            nn.Linear(1472, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, local_view, global_view):
        """
        Forward pass through the network (ignores global_view).
        Args:
            local_view: Local light curve view (batch_size, 1, 201)
            global_view: Global light curve view (ignored)
        Returns:
            torch.Tensor: Probability of exoplanet (batch_size, 1)
        """
        # Process only local view
        local_features = self.local_branch(local_view)
        local_features = local_features.view(local_features.size(0), -1)  # flatten
        
        # Final classification using only local features
        output = self.classifier(local_features)
        return output

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run on (CPU/GPU)
        
    Returns:
        float: Average training loss for this epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Unpack the data
        local_view, global_view = data
        
        # Move data to device and add channel dimension
        local_view = local_view.unsqueeze(1).float().to(device)
        global_view = global_view.unsqueeze(1).float().to(device)
        target = target.unsqueeze(1).float().to(device)
        
        # Data augmentation: randomly flip 50% of light curves in time
        # This makes the model more robust to different orientations
        flip_mask = torch.rand(local_view.size(0), device=device) < 0.5
        if flip_mask.any():
            # Flip the time dimension for selected samples
            local_view[flip_mask] = torch.flip(local_view[flip_mask], dims=[-1])
            global_view[flip_mask] = torch.flip(global_view[flip_mask], dims=[-1])
        
        # Forward pass: compute predictions
        optimizer.zero_grad()  # Clear gradients from previous batch
        predictions = model(local_view, global_view)
        loss = criterion(predictions, target)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: The neural network
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy, average_precision, all_predictions, all_targets)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, target in val_loader:
            # Unpack the data
            local_view, global_view = data
            
            # Move data to device and add channel dimension
            local_view = local_view.unsqueeze(1).float().to(device)
            global_view = global_view.unsqueeze(1).float().to(device)
            target = target.unsqueeze(1).float().to(device)
            
            # Forward pass
            predictions = model(local_view, global_view)
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
            
            # Convert predictions to binary (threshold = 0.5)
            binary_predictions = (predictions >= 0.5).float()
            correct_predictions += (binary_predictions == target).sum().item()
            total_samples += target.size(0)
            
            # Store for metrics calculation
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    avg_precision = average_precision_score(all_targets, all_predictions)
    
    return avg_loss, accuracy, avg_precision, all_predictions, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run on (CPU/GPU)
        num_epochs: Number of training epochs
        
    Returns:
        tuple: Training history (losses, accuracies, precisions)
    """
    # Lists to store training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    
    print(f"Training for {num_epochs} epochs...")
    
    # Add progress bar for epochs (like original AstroNet)
    for epoch in tqdm(range(num_epochs)):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate for one epoch
        val_loss, val_acc, val_prec, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"Val AP={val_prec:.4f}")
    
    return train_losses, val_losses, val_accuracies, val_precisions, val_preds, val_targets

# =============================================================================
# EVALUATION AND VISUALIZATION
# =============================================================================

def calculate_metrics(predictions, targets):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth labels
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(targets, predictions)
    avg_precision = average_precision_score(targets, predictions)
    
    # Calculate metrics at different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_metrics = {}
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        binary_preds = (np.array(predictions) >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
        
        # Calculate precision and recall
        prec = precision_score(targets, binary_preds, zero_division=0)
        rec = recall_score(targets, binary_preds, zero_division=0)
        
        threshold_metrics[threshold] = {
            'precision': prec,
            'recall': rec,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
        print(f"Threshold {threshold:.1f}: "
              f"Precision={prec:.3f}, Recall={rec:.3f} "
              f"(TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    
    return {
        'avg_precision': avg_precision,
        'precision_curve': precision,
        'recall_curve': recall,
        'threshold_metrics': threshold_metrics
    }

def create_plots(train_losses, val_losses, val_accuracies, val_precisions, metrics, output_dir, test_metrics=None):
    """
    Create comprehensive visualization plots.
    
    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        val_accuracies: Validation accuracy history
        val_precisions: Validation average precision history
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save plots
        test_metrics: Optional test set metrics for comparison
    """
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)
    
    # 1. Precision-Recall Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['recall_curve'], metrics['precision_curve'], 
             linewidth=2, color='blue', label='Validation')
    if test_metrics:
        ax1.plot(test_metrics['recall_curve'], test_metrics['precision_curve'], 
                 linewidth=2, color='red', label='Test')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall Curve\nVal AP = {metrics["avg_precision"]:.3f}' + 
                  (f', Test AP = {test_metrics["avg_precision"]:.3f}' if test_metrics else ''))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. Training and Validation Loss
    ax2 = fig.add_subplot(gs[0, 1])
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, [loss * BATCH_SIZE for loss in train_losses], 
             label='Training Loss', linewidth=2, color='blue')
    ax2.plot(epochs, [loss * BATCH_SIZE for loss in val_losses], 
             label='Validation Loss', linewidth=2, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (per batch)')
    ax2.set_title('Training Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Average Precision over time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_precisions, linewidth=2, color='green', marker='o', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Precision')
    ax3.set_title('Validation Average Precision')
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Accuracy over time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, val_accuracies, linewidth=2, color='orange', marker='o', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'training_results.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.show()
    
    print(f"Plots saved to: {plot_path}")

def create_three_way_comparison_plot(global_results, local_results, combined_results, output_dir):
    """
    Create comparison plots between global-only, local-only, and global+local models.
    
    Args:
        global_results: Dictionary with global-only model results
        local_results: Dictionary with local-only model results
        combined_results: Dictionary with global+local model results
        output_dir: Directory to save plots
    """
    # Create 2x2 subplot layout for comparison
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)
    
    # 1. Precision-Recall Curves Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(global_results['val_recall'], global_results['val_precision'], 
             linewidth=3, color='blue', label='Global Only (Val)', linestyle='-')
    ax1.plot(global_results['test_recall'], global_results['test_precision'], 
             linewidth=3, color='blue', label='Global Only (Test)', linestyle='--')
    ax1.plot(local_results['val_recall'], local_results['val_precision'], 
             linewidth=3, color='green', label='Local Only (Val)', linestyle='-')
    ax1.plot(local_results['test_recall'], local_results['test_precision'], 
             linewidth=3, color='green', label='Local Only (Test)', linestyle='--')
    ax1.plot(combined_results['val_recall'], combined_results['val_precision'], 
             linewidth=3, color='red', label='Global+Local (Val)', linestyle='-')
    ax1.plot(combined_results['test_recall'], combined_results['test_precision'], 
             linewidth=3, color='red', label='Global+Local (Test)', linestyle='--')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. Performance Metrics Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Val AP', 'Test AP', 'Val Acc', 'Test Acc']
    global_values = [global_results['val_ap'], global_results['test_ap'], 
                     global_results['val_acc'], global_results['test_acc']]
    local_values = [local_results['val_ap'], local_results['test_ap'], 
                    local_results['val_acc'], local_results['test_acc']]
    combined_values = [combined_results['val_ap'], combined_results['test_ap'], 
                       combined_results['val_acc'], combined_results['test_acc']]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax2.bar(x - width, global_values, width, label='Global Only', color='blue', alpha=0.7)
    bars2 = ax2.bar(x, local_values, width, label='Local Only', color='green', alpha=0.7)
    bars3 = ax2.bar(x + width, combined_values, width, label='Global+Local', color='red', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # 3. Training Loss Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(global_results['train_losses']) + 1)
    ax3.plot(epochs, global_results['train_losses'], linewidth=2, color='blue', 
             label='Global Only (Train)', linestyle='-')
    ax3.plot(epochs, global_results['val_losses'], linewidth=2, color='blue', 
             label='Global Only (Val)', linestyle='--')
    ax3.plot(epochs, local_results['train_losses'], linewidth=2, color='green', 
             label='Local Only (Train)', linestyle='-')
    ax3.plot(epochs, local_results['val_losses'], linewidth=2, color='green', 
             label='Local Only (Val)', linestyle='--')
    ax3.plot(epochs, combined_results['train_losses'], linewidth=2, color='red', 
             label='Global+Local (Train)', linestyle='-')
    ax3.plot(epochs, combined_results['val_losses'], linewidth=2, color='red', 
             label='Global+Local (Val)', linestyle='--')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Precision Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, global_results['val_precisions'], linewidth=2, color='blue', 
             label='Global Only', marker='o', markersize=4)
    ax4.plot(epochs, local_results['val_precisions'], linewidth=2, color='green', 
             label='Local Only', marker='^', markersize=4)
    ax4.plot(epochs, combined_results['val_precisions'], linewidth=2, color='red', 
             label='Global+Local', marker='s', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Average Precision', fontsize=12)
    ax4.set_title('Validation AP Over Time', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add improvement text
    global_to_combined_val = combined_results['val_ap'] - global_results['val_ap']
    global_to_combined_test = combined_results['test_ap'] - global_results['test_ap']
    local_to_combined_val = combined_results['val_ap'] - local_results['val_ap']
    local_to_combined_test = combined_results['test_ap'] - local_results['test_ap']
    
    fig.suptitle(f'Three-Way Model Comparison: Global vs Local vs Global+Local\n'
                 f'Combined vs Global: Val AP +{global_to_combined_val:.4f} | Test AP +{global_to_combined_test:.4f}\n'
                 f'Combined vs Local: Val AP +{local_to_combined_val:.4f} | Test AP +{local_to_combined_test:.4f}', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'three_way_model_comparison.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"Three-way model comparison plots saved to: {plot_path}")

def run_three_way_comparison_experiment(train_loader, val_loader, test_loader, device, epochs=10):
    """
    Run comparison experiment between global-only, local-only, and global+local models.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        device: Device to run on
        epochs: Number of training epochs
        
    Returns:
        tuple: (global_results, local_results, combined_results)
    """
    print("=" * 60)
    print("RUNNING THREE-WAY COMPARISON EXPERIMENT")
    print("=" * 60)
    
    # Train Global-Only Model
    print("\n1. Training Global-Only Model...")
    global_model = AstroNetGlobalOnly().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)
    
    global_train_losses, global_val_losses, global_val_accuracies, global_val_precisions, global_val_preds, global_val_targets = train_model(
        global_model, train_loader, val_loader, criterion, optimizer, device, epochs
    )
    
    # Evaluate global model on test set
    global_test_loss, global_test_acc, global_test_prec, global_test_preds, global_test_targets = validate_epoch(
        global_model, test_loader, criterion, device
    )
    global_test_metrics = calculate_metrics(global_test_preds, global_test_targets)
    global_val_metrics = calculate_metrics(global_val_preds, global_val_targets)
    
    # Train Local-Only Model
    print("\n2. Training Local-Only Model...")
    local_model = AstroNetLocalOnly().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
    
    local_train_losses, local_val_losses, local_val_accuracies, local_val_precisions, local_val_preds, local_val_targets = train_model(
        local_model, train_loader, val_loader, criterion, optimizer, device, epochs
    )
    
    # Evaluate local model on test set
    local_test_loss, local_test_acc, local_test_prec, local_test_preds, local_test_targets = validate_epoch(
        local_model, test_loader, criterion, device
    )
    local_test_metrics = calculate_metrics(local_test_preds, local_test_targets)
    local_val_metrics = calculate_metrics(local_val_preds, local_val_targets)
    
    # Train Combined Model
    print("\n3. Training Global+Local Model...")
    combined_model = AstroNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=LEARNING_RATE)
    
    combined_train_losses, combined_val_losses, combined_val_accuracies, combined_val_precisions, combined_val_preds, combined_val_targets = train_model(
        combined_model, train_loader, val_loader, criterion, optimizer, device, epochs
    )
    
    # Evaluate combined model on test set
    combined_test_loss, combined_test_acc, combined_test_prec, combined_test_preds, combined_test_targets = validate_epoch(
        combined_model, test_loader, criterion, device
    )
    combined_test_metrics = calculate_metrics(combined_test_preds, combined_test_targets)
    combined_val_metrics = calculate_metrics(combined_val_preds, combined_val_targets)
    
    # Prepare results for comparison
    global_results = {
        'train_losses': global_train_losses,
        'val_losses': global_val_losses,
        'val_precisions': global_val_precisions,
        'val_ap': global_val_metrics['avg_precision'],
        'test_ap': global_test_metrics['avg_precision'],
        'val_acc': global_val_accuracies[-1],
        'test_acc': global_test_acc,
        'val_precision': global_val_metrics['precision_curve'],
        'val_recall': global_val_metrics['recall_curve'],
        'test_precision': global_test_metrics['precision_curve'],
        'test_recall': global_test_metrics['recall_curve']
    }
    
    local_results = {
        'train_losses': local_train_losses,
        'val_losses': local_val_losses,
        'val_precisions': local_val_precisions,
        'val_ap': local_val_metrics['avg_precision'],
        'test_ap': local_test_metrics['avg_precision'],
        'val_acc': local_val_accuracies[-1],
        'test_acc': local_test_acc,
        'val_precision': local_val_metrics['precision_curve'],
        'val_recall': local_val_metrics['recall_curve'],
        'test_precision': local_test_metrics['precision_curve'],
        'test_recall': local_test_metrics['recall_curve']
    }
    
    combined_results = {
        'train_losses': combined_train_losses,
        'val_losses': combined_val_losses,
        'val_precisions': combined_val_precisions,
        'val_ap': combined_val_metrics['avg_precision'],
        'test_ap': combined_test_metrics['avg_precision'],
        'val_acc': combined_val_accuracies[-1],
        'test_acc': combined_test_acc,
        'val_precision': combined_val_metrics['precision_curve'],
        'val_recall': combined_val_metrics['recall_curve'],
        'test_precision': combined_test_metrics['precision_curve'],
        'test_recall': combined_test_metrics['recall_curve']
    }
    
    return global_results, local_results, combined_results

def save_results(predictions, targets, train_losses, val_losses, val_accuracies, val_precisions, model, output_dir):
    """
    Save all results to files.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        train_losses: Training loss history
        val_losses: Validation loss history
        val_accuracies: Validation accuracy history
        val_precisions: Validation average precision history
        model: Trained model
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'ground_truth': targets,
        'predictions': predictions
    })
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': [loss * BATCH_SIZE for loss in train_losses],
        'val_loss': [loss * BATCH_SIZE for loss in val_losses],
        'val_accuracy': val_accuracies,
        'val_avg_precision': val_precisions
    })
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    # Save model
    model_path = os.path.join(output_dir, 'astronet_model.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"Results saved to: {output_dir}")
    print(f"  - Predictions: {predictions_path}")
    print(f"  - Training history: {history_path}")
    print(f"  - Model weights: {model_path}")

# ==================================
#===========================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """
    Main function to run the complete training pipeline.
    """
    print("=" * 60)
    print("ASTRONET: Exoplanet Detection with Deep Learning")
    print("=" * 60)
    
    # Set up device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
        
    # Prepare data
    if not prepare_data():
        print("Failed to prepare data. Exiting.")
        return
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = KeplerDataset(os.path.join(DATA_DIR, 'train'))
    val_dataset = KeplerDataset(os.path.join(DATA_DIR, 'val'))
    test_dataset = KeplerDataset(os.path.join(DATA_DIR, 'test'))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Create model
    print("\nCreating model...")
    model = AstroNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up training
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses, val_losses, val_accuracies, val_precisions, final_preds, final_targets = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
    )
    
    # Calculate final metrics on validation set
    print("\nCalculating final metrics on validation set...")
    metrics = calculate_metrics(final_preds, final_targets)
    print(f"\nValidation Average Precision: {metrics['avg_precision']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_prec, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device
    )
    test_metrics = calculate_metrics(test_preds, test_targets)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Average Precision: {test_prec:.4f}")
    
    # Create visualizations
    output_dir = os.path.join(DATA_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)  # Create results directory
    create_plots(train_losses, val_losses, val_accuracies, val_precisions, metrics, output_dir)
    
    # Save all results
    save_results(final_preds, final_targets, train_losses, val_losses, val_accuracies, val_precisions, model, output_dir)
    
    print("\nTraining complete!")
    return model

def main_comparison():
    """
    Main function to run the comparison experiment between global-only vs global+local models.
    """
    print("=" * 60)
    print("ASTRONET: Global vs Global+Local Comparison")
    print("=" * 60)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    if not prepare_data():
        print("Failed to prepare data. Exiting.")
        return
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = KeplerDataset(os.path.join(DATA_DIR, 'train'))
    val_dataset = KeplerDataset(os.path.join(DATA_DIR, 'val'))
    test_dataset = KeplerDataset(os.path.join(DATA_DIR, 'test'))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Run three-way comparison experiment
    global_results, local_results, combined_results = run_three_way_comparison_experiment(
        train_loader, val_loader, test_loader, device, epochs=EPOCHS
    )
    
    # Create comparison plots
    output_dir = os.path.join(DATA_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)
    create_three_way_comparison_plot(global_results, local_results, combined_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("THREE-WAY COMPARISON RESULTS SUMMARY")
    print("=" * 60)
    print(f"Global Only  - Val AP: {global_results['val_ap']:.4f}, Test AP: {global_results['test_ap']:.4f}")
    print(f"Local Only   - Val AP: {local_results['val_ap']:.4f}, Test AP: {local_results['test_ap']:.4f}")
    print(f"Global+Local - Val AP: {combined_results['val_ap']:.4f}, Test AP: {combined_results['test_ap']:.4f}")
    print(f"\nImprovements over Global Only:")
    print(f"  - Val AP: +{combined_results['val_ap'] - global_results['val_ap']:.4f}")
    print(f"  - Test AP: +{combined_results['test_ap'] - global_results['test_ap']:.4f}")
    print(f"\nImprovements over Local Only:")
    print(f"  - Val AP: +{combined_results['val_ap'] - local_results['val_ap']:.4f}")
    print(f"  - Test AP: +{combined_results['test_ap'] - local_results['test_ap']:.4f}")
    print("=" * 60)
    
    return global_results, local_results, combined_results

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Run three-way comparison experiment
    global_results, local_results, combined_results = main_comparison()