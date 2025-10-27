# **AstroNet colab_ver3.py - Clean Structure Chart**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ASTRONET: Exoplanet Detection Pipeline                   │
│                              (907 lines)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
sfsdfsdfwefwfsfsdfsf
┌─ IMPORTS & CONFIGURATION ───────────────────────────────────────────────────┐
│ • Standard library (os, glob, shutil, zipfile, tarfile)                    │
│ • Data science (numpy, pandas, tqdm)                                       │
│ • PyTorch (torch, torch.nn, torch.optim, torch.utils.data)                 │
│ • ML metrics (sklearn.metrics)                                             │
│ • Visualization (matplotlib)                                                │
│ • Configuration (paths, hyperparameters)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ DATA PREPARATION ──────────────────────────────────────────────────────────┐
│ check_existing_data()     → bool    │ Check if data already exists         │
│ mount_google_drive()      → bool    │ Mount Google Drive (Colab)          │
│ find_and_extract_data()   → bool    │ Extract compressed files             │
│ organize_data()              → bool    │ Organize into train/val/test       │
│ prepare_data()              → bool    │ Main data preparation orchestrator  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ DATASET & MODEL ───────────────────────────────────────────────────────────┐
│ KeplerDataset(Dataset)    │ PyTorch dataset for light curves              │
│ ├─ __init__(data_path)    │ Initialize with data directory                 │
│ ├─ __len__()             │ Return number of samples                       │
│ └─ __getitem__(idx)      │ Return (data, label, sample_id)               │
│                                                                           │
│ AstroNet(nn.Module)      │ Two-branch CNN architecture                   │
│ ├─ global_branch         │ Process full light curve (2001 points)        │
│ ├─ local_branch          │ Process transit view (201 points)              │
│ ├─ classifier            │ Fusion + classification layers               │
│ └─ forward()             │ Return exoplanet probability                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ TRAINING PIPELINE ─────────────────────────────────────────────────────────┐
│ train_epoch()            │ Single epoch training with data augmentation   │
│ validate_epoch()         │ Single epoch validation with metrics          │
│ train_model()            │ Multi-epoch training orchestrator              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ EVALUATION & VISUALIZATION ────────────────────────────────────────────────┐
│ calculate_metrics()      │ Compute precision, recall, AP, confusion matrix │
│ create_plots()           │ Training curves + validation plots             │
│ create_test_evaluation_plot() │ Test set evaluation plots               │
│ save_results()           │ Export CSV files + model weights               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ MAIN EXECUTION ────────────────────────────────────────────────────────────┐
│ main()                   │ Complete training pipeline                      │
│ ├─ Setup device          │ GPU/CPU detection                              │
│ ├─ Prepare data          │ Data loading + organization                     │
│ ├─ Create datasets       │ Train/val/test DataLoaders                     │
│ ├─ Initialize model      │ AstroNet + optimizer + criterion              │
│ ├─ Train model          │ Multi-epoch training                          │
│ ├─ Evaluate model       │ Validation + test metrics                     │
│ ├─ Create visualizations │ Training curves + evaluation plots            │
│ └─ Save results         │ Model weights + CSV exports                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## **Function Call Flow**

```
main()
│
├─ prepare_data()
│  ├─ check_existing_data()
│  └─ find_and_extract_data()
│     └─ mount_google_drive()
│     └─ organize_data()
│
├─ KeplerDataset() × 3 (train/val/test)
├─ DataLoader() × 3 (train/val/test)
├─ AstroNet() → model
│
├─ train_model()
│  ├─ train_epoch() × epochs
│  └─ validate_epoch() × epochs
│
├─ validate_epoch() (test set)
├─ calculate_metrics() (validation)
├─ calculate_metrics() (test)
│
├─ create_plots()
├─ create_test_evaluation_plot()
└─ save_results()
```

## **Data Flow Architecture**

```
INPUT DATA
├─ Google Drive files (*.tar.gz, *.zip)
├─ Local .npy files (global, local, info)
└─ Sample data (synthetic fallback)

DATA PROCESSING
├─ Mount Google Drive (Colab)
├─ Extract & organize → train/val/test
├─ KeplerDataset → PyTorch Dataset
└─ DataLoader → Batched training

MODEL TRAINING
├─ AstroNet Architecture
│  ├─ Global Branch (2001 → features)
│  ├─ Local Branch (201 → features)
│  └─ Classifier (combined → probability)
├─ train_epoch() → weight updates
└─ validate_epoch() → metrics

EVALUATION
├─ Validation metrics (AP, accuracy, loss)
├─ Test metrics (AP, accuracy, loss)
├─ Precision-Recall curves
└─ Confusion matrices

OUTPUTS
├─ Model weights (.pth)
├─ Training plots (.pdf)
├─ Results CSV files
└─ Performance metrics
```

## **Key Features**

| Feature | Description | Lines |
|---------|-------------|-------|
| **Data Preparation** | Automated Google Drive + local data handling | ~160 |
| **Model Architecture** | Two-branch CNN (global + local views) | ~110 |
| **Training Pipeline** | Separated epoch functions for modularity | ~150 |
| **Evaluation** | Comprehensive metrics + visualizations | ~200 |
| **Documentation** | Extensive docstrings + comments | ~150 |
| **Error Handling** | Robust fallbacks for different environments | ~50 |

## **File Statistics**

- **Total Lines:** 907
- **Functions:** 13
- **Classes:** 2
- **Sections:** 8
- **Documentation:** ~150 lines (16%)
- **Core Logic:** ~600 lines (66%)
- **Configuration:** ~50 lines (6%)

## **Function List**

1. `check_existing_data()` → bool
2. `mount_google_drive()` → bool
3. `find_and_extract_data()` → bool
4. `organize_data()` → bool
5. `prepare_data()` → bool
6. `train_epoch()` → float
7. `validate_epoch()` → tuple
8. `train_model()` → tuple
9. `calculate_metrics()` → dict
10. `create_plots()` → None
11. `create_test_evaluation_plot()` → None
12. `save_results()` → None
13. `main()` → model

## **Class List**

1. `KeplerDataset(Dataset)` - PyTorch dataset for light curves
2. `AstroNet(nn.Module)` - Two-branch CNN architecture

## **Configuration Variables**

- `DATA_DIR` - Main data directory path
- `DRIVE_DIR` - Google Drive data path
- `BATCH_SIZE` - Training batch size
- `NUM_WORKERS` - DataLoader workers
- `EPOCHS` - Number of training epochs
- `LEARNING_RATE` - Adam optimizer learning rate
