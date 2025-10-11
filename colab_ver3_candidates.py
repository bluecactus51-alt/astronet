#!/usr/bin/env python3
"""
AstroNet: Exoplanet Candidate Detection and Ranking
==================================================

This version focuses on identifying and ranking high-probability exoplanet candidates.
It provides a ranked list of the most promising exoplanet candidates based on model predictions.

Features:
- Trains AstroNet model for exoplanet detection
- Evaluates on test set and ranks candidates by probability
- Provides detailed candidate analysis with confidence scores
- Exports ranked candidate list for further investigation
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

# Paths for data storage
DATA_DIR = './astronet_data'  # Local directory
DRIVE_DIR = '/content/drive/MyDrive/astronet_data'  # Google Colab Drive

# Auto-detect environment and set optimal parameters
import sys
IS_COLAB = 'google.colab' in sys.modules

# Training hyperparameters - optimized for performance
BATCH_SIZE = 64  # Match standard version
NUM_WORKERS = 12 if IS_COLAB else 4  # Colab-optimized / 4 for local
EPOCHS = 50  # Match standard version (minimum for good performance)
LEARNING_RATE = 1e-5  # Keep same

# Learning rate scheduling
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5  # Reduce LR if no improvement for 5 epochs
LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by 0.5 when reducing

# Configuration profiles for easy mode switching
PROFILES = {
    'quick_test': {'epochs': 10, 'batch_size': 32, 'workers': 4},
    'standard': {'epochs': 50, 'batch_size': 64, 'workers': 12},
    'high_performance': {'epochs': 100, 'batch_size': 64, 'workers': 12},
    'production': {'epochs': 225, 'batch_size': 64, 'workers': 12}
}

# Select profile (change this to switch modes)
PROFILE = 'standard'  # Options: 'quick_test', 'standard', 'high_performance', 'production'
config = PROFILES[PROFILE]

# Override settings based on profile
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['workers'] if IS_COLAB else min(config['workers'], 4)
EPOCHS = config['epochs']

# Candidate detection parameters
HIGH_CONFIDENCE_THRESHOLD = 0.8  # High confidence exoplanet candidates
MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # Medium confidence candidates
CANDIDATE_TOP_N = 50  # Top N candidates to report

# =============================================================================
# DATA PREPARATION
# =============================================================================

def check_existing_data() -> bool:
    """
    Check if data is already prepared in the environment.
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
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except ImportError:
        print("Not running in Google Colab - skipping drive mount")
        return False

def create_sample_data():
    """
    Create sample data for local testing when no real data is available.
    This generates synthetic light curves for demonstration purposes.
    """
    print("Creating sample data for local testing...")
    
    # Create directories
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)
    
    # Generate sample data for each split (more realistic counts)
    splits = {'train': 2000, 'val': 400, 'test': 400}  # More realistic sample counts
    
    for split, num_samples in splits.items():
        split_dir = os.path.join(DATA_DIR, split)
        
        for i in range(num_samples):
            sample_id = f'kplr_{i:06d}_01'
            
            # Generate synthetic global view (2001 time points)
            # Simulate more realistic stellar variability
            t_global = np.linspace(0, 10, 2001)  # 10-day observation
            global_flux = 1.0 + 0.02 * np.sin(2 * np.pi * t_global / 2.5)  # Stronger variability
            global_flux += 0.01 * np.random.randn(2001)  # More realistic noise
            
            # Add transit if this is an exoplanet sample (50% chance)
            is_exoplanet = np.random.rand() < 0.5
            if is_exoplanet:
                # Add realistic transit with variable parameters
                transit_center = np.random.uniform(2, 8)  # Random transit time
                transit_duration = np.random.uniform(0.05, 0.2)  # Variable durations
                transit_depth = np.random.uniform(0.005, 0.02)  # Variable depths (0.5-2%)
                
                transit_mask = np.abs(t_global - transit_center) < transit_duration
                global_flux[transit_mask] -= transit_depth
            
            # Generate synthetic local view (201 time points)
            # Focused around potential transit
            t_local = np.linspace(-0.5, 0.5, 201)  # 1-day window
            local_flux = 1.0 + 0.02 * np.sin(2 * np.pi * t_local / 0.5)  # Stronger variability
            local_flux += 0.01 * np.random.randn(201)  # More realistic noise
            
            if is_exoplanet:
                # Add transit to local view with realistic shape
                transit_mask = np.abs(t_local) < 0.05
                local_flux[transit_mask] -= 0.01
            else:
                # Add false positive scenarios (30% chance)
                if np.random.rand() < 0.3:
                    # Add stellar activity that mimics transits
                    activity_center = np.random.uniform(-0.2, 0.2)
                    activity_mask = np.abs(t_local - activity_center) < 0.03
                    local_flux[activity_mask] -= 0.005  # Shallow dip
            
            # Create info array (label is at index 5)
            info = np.zeros(10)
            info[5] = 1.0 if is_exoplanet else 0.0  # Label
            
            # Save files
            np.save(os.path.join(split_dir, f'{sample_id}_global.npy'), global_flux.astype(np.float32))
            np.save(os.path.join(split_dir, f'{sample_id}_local.npy'), local_flux.astype(np.float32))
            np.save(os.path.join(split_dir, f'{sample_id}_info.npy'), info.astype(np.float32))
    
    print(f"Created sample data: {sum(splits.values())} samples total")
    return True

def find_and_extract_data() -> bool:
    """
    Find and extract compressed data files from Google Drive.
    Returns:
        bool: True if extraction successful
    """
    # Check if we're in Google Colab environment
    try:
        import google.colab
        # We're in Colab, look for Drive files
        compressed_files = []
        for pattern in ('*.tar.gz', '*.zip'):
            compressed_files.extend(glob.glob(os.path.join(DRIVE_DIR, pattern)))
        
        if not compressed_files:
            print('No data files found! Please upload AstroNet data to Google Drive.')
            return False
        
        print(f'Found {len(compressed_files)} files')
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Extract each file
        for file_path in tqdm(compressed_files, desc='Extracting'):
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(DATA_DIR)
            else:  # .tar.gz
                with tarfile.open(file_path, 'r:gz') as tf:
                    tf.extractall(DATA_DIR)
        
        return True
        
    except ImportError:
        # Not in Colab, no real data available
        print('Not running in Google Colab - no real data available')
        return False

def organize_data() -> bool:
    """
    Organize extracted files into proper train/val/test structure.
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
    
    # Copy samples (all 3 files required for complete samples)
    kept_samples = 0
    for (split, _sample_id), file_dict in groups.items():
        # Must have all 3 files: global, local, info
        if len(file_dict) == 3:
            # Copy all 3 files to the organized directory
            for file_type, source_path in file_dict.items():
                dest_path = os.path.join(DATA_DIR, split, os.path.basename(source_path))
                if source_path != dest_path:
                    shutil.copy2(source_path, dest_path)
            kept_samples += 1
    
    print(f'✅ Total: {kept_samples} complete samples organized')
    return kept_samples > 0

def prepare_data() -> bool:
    """
    Prepare data for training. Check if data exists, otherwise extract and organize.
    Falls back to sample data for local testing if no real data is available.
    Returns:
        bool: True if data preparation successful
    """
    if check_existing_data():
        return True
    
    print('Preparing data...')
    
    # Try to find and extract real data first
    if find_and_extract_data():
        return organize_data()
    
    # If no real data found, create sample data for local testing
    print("No real data found. Creating sample data for local testing...")
    return create_sample_data()

# =============================================================================
# DATASET CLASS
# =============================================================================

class KeplerDataset(Dataset):
    """
    PyTorch Dataset for loading Kepler light curve data.
    Each sample contains:
    - Global view: Full light curve (2001 time points)
    - Local view: Zoomed-in view (201 time points)
    - Label: 1 for exoplanet, 0 for false positive
    """
    def __init__(self, data_path: str):
        """
        Initialize dataset from directory containing .npy files.
        Args:
            data_path: Path to directory with train/val/test data
        """
        # Find all global files (they all exist for complete samples)
        self.global_files = sorted(glob.glob(os.path.join(data_path, '*_global.npy')))
        
        # Create mapping of sample IDs to file paths
        self.sample_ids = []
        for global_file in self.global_files:
            # Extract ID from filename: kplr_001433399_02_global.npy -> 001433399_02
            filename = os.path.basename(global_file)
            parts = filename.split('_')
            if len(parts) >= 3:
                sample_id = '_'.join(parts[:3])  # kplr_KIC_TCE
                self.sample_ids.append(sample_id)
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Args:
            idx: Index of sample to retrieve
        Returns:
            tuple: ((local_view, global_view), label, sample_id)
        """
        sample_id = self.sample_ids[idx]
        
        # Load global view
        global_file = os.path.join(os.path.dirname(self.global_files[idx]), f'{sample_id}_global.npy')
        global_data = np.load(global_file)
        
        # Load local view
        local_file = os.path.join(os.path.dirname(self.global_files[idx]), f'{sample_id}_local.npy')
        local_data = np.load(local_file)
        
        # Load label
        info_file = os.path.join(os.path.dirname(self.global_files[idx]), f'{sample_id}_info.npy')
        info_data = np.load(info_file)
        label = info_data[5]
        
        return (local_data, global_data), label, sample_id

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

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0
        return self.early_stop

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
    
    for batch_idx, (data, target, _) in enumerate(train_loader):
        # Unpack the data
        local_view, global_view = data
        
        # Move data to device and add channel dimension
        local_view = local_view.unsqueeze(1).float().to(device)
        global_view = global_view.unsqueeze(1).float().to(device)
        target = target.unsqueeze(1).float().to(device)
        
        # Enhanced data augmentation
        # 1. Time flipping (50% of samples)
        flip_mask = torch.rand(local_view.size(0), device=device) < 0.5
        if flip_mask.any():
            local_view[flip_mask] = torch.flip(local_view[flip_mask], dims=[-1])
            global_view[flip_mask] = torch.flip(global_view[flip_mask], dims=[-1])
        
        # 2. Add Gaussian noise (10% of samples)
        noise_mask = torch.rand(local_view.size(0), device=device) < 0.1
        if noise_mask.any():
            noise_std = 0.01
            local_view[noise_mask] += torch.randn_like(local_view[noise_mask]) * noise_std
            global_view[noise_mask] += torch.randn_like(global_view[noise_mask]) * noise_std
        
        # 3. Add amplitude scaling (10% of samples)
        scale_mask = torch.rand(local_view.size(0), device=device) < 0.1
        if scale_mask.any():
            scale_factor = torch.FloatTensor(scale_mask.sum().item()).uniform_(0.9, 1.1).to(device)
            local_view[scale_mask] *= scale_factor.view(-1, 1, 1)
            global_view[scale_mask] *= scale_factor.view(-1, 1, 1)
        
        # Forward pass: compute predictions
        optimizer.zero_grad()  # Clear gradients from previous batch
        predictions = model(local_view, global_view)
        loss = criterion(predictions, target)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
        for data, target, _ in val_loader:
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
    Train the model for multiple epochs with enhanced features.
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
    
    # Initialize learning rate scheduler
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE
        )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Model checkpointing
    best_val_ap = 0.0
    best_model_state = None
    
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
        
        # Learning rate scheduling
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_prec)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Model checkpointing - save best model
        if val_prec > best_val_ap:
            best_val_ap = val_prec
            best_model_state = model.state_dict().copy()
            print(f"New best model! AP: {val_prec:.4f}")
        
        # Early stopping check
        if early_stopping(val_prec):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress (like original AstroNet)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"Val AP={val_prec:.4f}, "
              f"LR={current_lr:.2e}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with AP: {best_val_ap:.4f}")
    
    return train_losses, val_losses, val_accuracies, val_precisions, val_preds, val_targets

# =============================================================================
# CANDIDATE DETECTION AND RANKING
# =============================================================================

def detect_candidates(model, test_loader, device, threshold=0.5):
    """
    Detect exoplanet candidates from test set and rank by probability.
    Args:
        model: Trained AstroNet model
        test_loader: DataLoader for test data
        device: Device to run on (CPU/GPU)
        threshold: Probability threshold for candidate detection
    Returns:
        pandas.DataFrame: Ranked list of candidates with probabilities and metadata
    """
    model.eval()
    candidates = []
    
    print(f"Detecting exoplanet candidates with threshold >= {threshold}...")
    
    with torch.no_grad():
        for data, target, sample_ids in test_loader:
            # Unpack the data
            local_view, global_view = data
            
            # Move data to device and add channel dimension
            local_view = local_view.unsqueeze(1).float().to(device)
            global_view = global_view.unsqueeze(1).float().to(device)
            
            # Forward pass
            predictions = model(local_view, global_view)
            probabilities = predictions.cpu().numpy().flatten()
            targets = target.numpy()
            
            # Process each sample in the batch
            for i, (prob, true_label, sample_id) in enumerate(zip(probabilities, targets, sample_ids)):
                # Determine confidence level
                if prob >= HIGH_CONFIDENCE_THRESHOLD:
                    confidence = "High"
                elif prob >= MEDIUM_CONFIDENCE_THRESHOLD:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                # Determine if this is a candidate (above threshold)
                is_candidate = prob >= threshold
                
                candidates.append({
                    'sample_id': sample_id,
                    'probability': prob,
                    'confidence_level': confidence,
                    'is_candidate': is_candidate,
                    'true_label': int(true_label),
                    'correct_prediction': (prob >= 0.5) == (true_label >= 0.5)
                })
    
    # Convert to DataFrame and sort by probability (highest first)
    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values('probability', ascending=False).reset_index(drop=True)
    
    # Add ranking
    candidates_df['rank'] = range(1, len(candidates_df) + 1)
    
    return candidates_df

def analyze_candidates(candidates_df):
    """
    Analyze the detected candidates and provide statistics.
    Args:
        candidates_df: DataFrame with candidate information
    Returns:
        dict: Analysis results
    """
    total_samples = len(candidates_df)
    total_candidates = candidates_df['is_candidate'].sum()
    high_conf_candidates = (candidates_df['confidence_level'] == 'High').sum()
    medium_conf_candidates = (candidates_df['confidence_level'] == 'Medium').sum()
    
    # True positives among candidates
    candidate_mask = candidates_df['is_candidate']
    true_positives = candidates_df[candidate_mask]['true_label'].sum()
    false_positives = candidate_mask.sum() - true_positives
    
    # Top candidates analysis
    top_candidates = candidates_df.head(CANDIDATE_TOP_N)
    top_true_positives = top_candidates['true_label'].sum()
    
    analysis = {
        'total_samples': total_samples,
        'total_candidates': total_candidates,
        'high_confidence_candidates': high_conf_candidates,
        'medium_confidence_candidates': medium_conf_candidates,
        'true_positives_among_candidates': true_positives,
        'false_positives_among_candidates': false_positives,
        'top_n_candidates': CANDIDATE_TOP_N,
        'top_n_true_positives': top_true_positives,
        'candidate_precision': true_positives / total_candidates if total_candidates > 0 else 0,
        'top_n_precision': top_true_positives / CANDIDATE_TOP_N
    }
    
    return analysis

def create_candidate_plots(candidates_df, analysis, output_dir):
    """
    Create visualization plots for candidate analysis.
    Args:
        candidates_df: DataFrame with candidate information
        analysis: Analysis results dictionary
        output_dir: Directory to save plots
    """
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)
    
    # 1. Probability distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(candidates_df['probability'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(HIGH_CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'High Confidence ({HIGH_CONFIDENCE_THRESHOLD})')
    ax1.axvline(MEDIUM_CONFIDENCE_THRESHOLD, color='orange', linestyle='--', label=f'Medium Confidence ({MEDIUM_CONFIDENCE_THRESHOLD})')
    ax1.set_xlabel('Exoplanet Probability')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Probability Distribution of Test Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top candidates by probability
    ax2 = fig.add_subplot(gs[0, 1])
    top_n = min(20, len(candidates_df))
    top_candidates = candidates_df.head(top_n)
    colors = ['red' if label == 1 else 'blue' for label in top_candidates['true_label']]
    bars = ax2.barh(range(top_n), top_candidates['probability'], color=colors, alpha=0.7)
    ax2.set_xlabel('Exoplanet Probability')
    ax2.set_ylabel('Sample Rank')
    ax2.set_title(f'Top {top_n} Candidates (Red=True Exoplanet, Blue=False Positive)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence level distribution
    ax3 = fig.add_subplot(gs[1, 0])
    confidence_counts = candidates_df['confidence_level'].value_counts()
    colors = ['red', 'orange', 'green']
    wedges, texts, autotexts = ax3.pie(confidence_counts.values, labels=confidence_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Confidence Level Distribution')
    
    # 4. Precision at different thresholds
    ax4 = fig.add_subplot(gs[1, 1])
    thresholds = np.arange(0.1, 1.0, 0.1)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        candidates_at_threshold = candidates_df['probability'] >= threshold
        if candidates_at_threshold.sum() > 0:
            true_positives = candidates_df[candidates_at_threshold]['true_label'].sum()
            precision = true_positives / candidates_at_threshold.sum()
            recall = true_positives / candidates_df['true_label'].sum()
        else:
            precision = 0
            recall = 0
        precisions.append(precision)
        recalls.append(recall)
    
    ax4.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=6)
    ax4.plot(thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=6)
    ax4.set_xlabel('Probability Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision and Recall vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add summary text
    fig.suptitle(f'Exoplanet Candidate Analysis\n'
                 f'Total Candidates: {analysis["total_candidates"]} | '
                 f'High Confidence: {analysis["high_confidence_candidates"]} | '
                 f'Precision: {analysis["candidate_precision"]:.3f}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'candidate_analysis.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"Candidate analysis plots saved to: {plot_path}")

def export_candidate_list(candidates_df, analysis, output_dir):
    """
    Export ranked candidate list to CSV files.
    Args:
        candidates_df: DataFrame with candidate information
        analysis: Analysis results dictionary
        output_dir: Directory to save files
    """
    # Export all candidates
    all_candidates_path = os.path.join(output_dir, 'all_candidates.csv')
    candidates_df.to_csv(all_candidates_path, index=False)
    
    # Export high-confidence candidates only
    high_conf_candidates = candidates_df[candidates_df['confidence_level'] == 'High']
    high_conf_path = os.path.join(output_dir, 'high_confidence_candidates.csv')
    high_conf_candidates.to_csv(high_conf_path, index=False)
    
    # Export top N candidates
    top_candidates = candidates_df.head(CANDIDATE_TOP_N)
    top_candidates_path = os.path.join(output_dir, f'top_{CANDIDATE_TOP_N}_candidates.csv')
    top_candidates.to_csv(top_candidates_path, index=False)
    
    # Export analysis summary
    summary_data = {
        'Metric': ['Total Samples', 'Total Candidates', 'High Confidence Candidates', 
                   'Medium Confidence Candidates', 'True Positives Among Candidates',
                   'False Positives Among Candidates', 'Candidate Precision',
                   'Top N Precision'],
        'Value': [analysis['total_samples'], analysis['total_candidates'],
                 analysis['high_confidence_candidates'], analysis['medium_confidence_candidates'],
                 analysis['true_positives_among_candidates'], analysis['false_positives_among_candidates'],
                 f"{analysis['candidate_precision']:.4f}", f"{analysis['top_n_precision']:.4f}"]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'candidate_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Candidate lists exported to: {output_dir}")
    print(f"  - All candidates: {all_candidates_path}")
    print(f"  - High confidence: {high_conf_path}")
    print(f"  - Top {CANDIDATE_TOP_N}: {top_candidates_path}")
    print(f"  - Summary: {summary_path}")

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """
    Main function to run the complete training and candidate detection pipeline.
    """
    print("=" * 60)
    print("ASTRONET: Exoplanet Candidate Detection and Ranking")
    print("=" * 60)
    
    # Display configuration
    print(f"Configuration Profile: {PROFILE}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"Learning Rate: {LEARNING_RATE}, LR Scheduler: {USE_LR_SCHEDULER}")
    print(f"Environment: {'Google Colab' if IS_COLAB else 'Local'}")
    print("-" * 60)
    
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
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
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(final_targets, final_preds)
    val_ap = average_precision_score(final_targets, final_preds)
    print(f"Validation Average Precision: {val_ap:.4f}")
    
    # Detect exoplanet candidates
    print("\nDetecting exoplanet candidates...")
    candidates_df = detect_candidates(model, test_loader, device, threshold=0.5)
    
    # Analyze candidates
    analysis = analyze_candidates(candidates_df)
    
    # Print candidate analysis
    print("\n" + "=" * 60)
    print("CANDIDATE DETECTION RESULTS")
    print("=" * 60)
    print(f"Total samples analyzed: {analysis['total_samples']}")
    print(f"Exoplanet candidates found: {analysis['total_candidates']}")
    print(f"High confidence candidates: {analysis['high_confidence_candidates']}")
    print(f"Medium confidence candidates: {analysis['medium_confidence_candidates']}")
    print(f"True positives among candidates: {analysis['true_positives_among_candidates']}")
    print(f"False positives among candidates: {analysis['false_positives_among_candidates']}")
    print(f"Candidate precision: {analysis['candidate_precision']:.4f}")
    print(f"Top {CANDIDATE_TOP_N} precision: {analysis['top_n_precision']:.4f}")
    
    # Show top candidates
    print(f"\nTop {min(10, CANDIDATE_TOP_N)} Exoplanet Candidates:")
    print("-" * 80)
    top_candidates = candidates_df.head(10)
    for _, candidate in top_candidates.iterrows():
        status = "✓ TRUE EXOPLANET" if candidate['true_label'] == 1 else "✗ False Positive"
        print(f"Rank {candidate['rank']:2d}: {candidate['sample_id']} | "
              f"Probability: {candidate['probability']:.4f} | "
              f"Confidence: {candidate['confidence_level']:6s} | {status}")
    
    # Create output directory and save results
    output_dir = os.path.join(DATA_DIR, 'candidate_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    create_candidate_plots(candidates_df, analysis, output_dir)
    
    # Export candidate lists
    export_candidate_list(candidates_df, analysis, output_dir)
    
    print("\nCandidate detection complete!")
    print(f"Results saved to: {output_dir}")
    
    return model, candidates_df, analysis

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    model, candidates_df, analysis = main()
