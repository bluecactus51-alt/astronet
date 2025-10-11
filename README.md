# AstroNet colab_ver3: Exoplanet Detection with Deep Learning

A modern PyTorch implementation of AstroNet for exoplanet detection in Kepler light curves, optimized for Google Colab environments.

## Abstract

This repository provides a PyTorch implementation of AstroNet, a deep convolutional neural network originally developed by Google Research for detecting exoplanet transits in Kepler light curves. The colab_ver3 implementation offers enhanced features for research and practical exoplanet discovery applications, building upon the original TensorFlow work by Shallue & Vanderburg (2018) and subsequent PyTorch translation by the NASA Frontier Development Lab team.

## Background

### Original Research

AstroNet was originally developed in TensorFlow by [Shallue & Vanderburg (2018)](https://arxiv.org/abs/1712.05044) for classifying exoplanet transits in Kepler light curves. The model uses a two-branch CNN architecture:

The model uses a two-branch CNN architecture:
- **Global Branch**: Processes full light curves (2001 time points) to capture long-term stellar behavior
- **Local Branch**: Analyzes zoomed-in transit views (201 time points) to detect transit shapes  
- **Fusion**: Combines features from both branches for final classification

**Original TensorFlow Implementation**: [Google Research exoplanet-ml repository](https://github.com/google-research/exoplanet-ml)

### NASA FDL Contribution

In 2018, [NASA's Frontier Development Lab](https://frontierdevelopmentlab.org/) (FDL) formed a team of scientists and machine learning experts to enhance AstroNet with scientific domain knowledge. The team translated the model from TensorFlow to PyTorch and added improvements for practical exoplanet discovery.

**2018 NASA FDL Exoplanet Team:**
- [Megan Ansdell](https://www.meganansdell.com)
- [Yani Ioannou](https://yani.io/annou/)
- [Hugh Osborn](https://www.hughosborn.co.uk/)
- [Michele Sasdelli](https://uk.linkedin.com/in/michelesasdelli)

**Reference**: [2018 NASA FDL Exoplanet Team (2018), ApJ Letters, 869, L7](http://adsabs.harvard.edu/abs/2018ApJ...869L...7A)

## Architecture

### Model Structure

```
Input: Light Curve Data
├── Global View (2001 points) → Global Branch → Features
├── Local View (201 points)   → Local Branch  → Features
└── Fusion Layer → Classification → Exoplanet Probability
```

### Implementation Structure

See [colab_ver3_structure_chart.md](colab_ver3_structure_chart.md) for detailed architecture documentation.

## Features

### Core Implementation
- **Google Colab Integration**: Automated data extraction from Google Drive
- **Data Organization**: Automatic file organization into train/val/test splits
- **Model Training**: 50 epochs, batch size 64, 12 workers for Colab
- **Evaluation**: Validation and test set metrics with visualizations

### Scientific Applications
- **Model Training**: Train AstroNet on Kepler light curves
- **Performance Evaluation**: Calculate accuracy, precision, recall, and average precision
- **Visualization**: Generate training curves and evaluation plots
- **Results Export**: Save model weights and CSV files with predictions

## Installation

### Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Upload Kepler data files to Google Drive in the `astronet_data` folder
3. Run the training pipeline

### Local Installation

```bash
git clone https://github.com/your-repo/astronet.git
cd astronet
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm
```

## Usage

```python
python colab_ver3.py
```

## Data Requirements

Each sample requires three files:
- `*_global.npy`: Full light curve (2001 time points)
- `*_local.npy`: Zoomed-in transit view (201 time points)  
- `*_info.npy`: Metadata including labels

Download Kepler data from [Google Drive](https://drive.google.com/file/d/1N6bA2rahvV5kcOGmJnTA5gl_Thno7hTh/view?usp=sharing) and organize into `train/`, `val/`, `test/` folders.

## Performance

### Version Comparison

| Version | Epochs | Batch Size | Workers | Expected AP | Use Case |
|---------|--------|------------|---------|-------------|----------|
| **Original** | 225 | 64 | 4 | ~0.95 | Research baseline |
| **colab_ver3** | 50 | 64 | 12 | ~0.90 | Google Colab optimized |

### Key Improvements over Original

- **Modular Architecture**: 13 functions vs 3 monolithic functions
- **Google Colab Integration**: Automated data handling vs manual command-line
- **Enhanced Visualization**: 8+ plots vs 4 basic plots
- **Comprehensive Metrics**: 10+ metrics vs 3 basic metrics
- **Error Handling**: Robust fallbacks vs minimal validation
- **Documentation**: 150+ lines of docstrings vs basic comments

## Configuration

### Default Settings

```python
# Training hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 12
EPOCHS = 50
LEARNING_RATE = 1e-5
```

### Data Paths

```python
# Paths for Google Colab
DATA_DIR = '/content/astronet_data'
DRIVE_DIR = '/content/drive/MyDrive/astronet_data'
```

## Output

### Training Output
```
Training for 50 epochs...
Epoch 25/50: Train Loss=0.1234, Val Loss=0.1456, Val Acc=0.9234, Val AP=0.9456
```

### Generated Files
- `astronet_model.pth`: Trained model weights
- `predictions.csv`: Model predictions and ground truth
- `training_history.csv`: Training metrics per epoch
- `training_results.pdf`: Visualization plots

## File Structure

```
astronet/
├── README.md                           # This file
├── colab_ver3.py                      # Main implementation (907 lines)
├── colab_ver3_structure_chart.md      # Detailed architecture documentation
├── astronet_original.py               # Original PyTorch implementation (460 lines)
├── exonet.py                          # Enhanced version with domain knowledge (563 lines)
├── colab_ver1.py                      # First Colab version (468 lines)
├── colab_ver2.py                      # Second Colab version (482 lines)
├── README_original.md                 # Original AstroNet documentation
└── README for exonet.md               # ExoNet documentation
```

## Scientific Applications

### Research
- Exoplanet detection algorithm development
- Machine learning in astronomy
- Transit detection methodology
- Stellar variability analysis

### Practical Discovery
- **Kepler data analysis**: Process thousands of light curves
- **TESS data adaptation**: Extend to TESS mission data
- **Follow-up planning**: Prioritize telescope observations
- **Discovery validation**: Confirm exoplanet candidates

## Documentation

- **[Architecture Details](colab_ver3_structure_chart.md)**: Complete system architecture
- **[Original AstroNet](README_original.md)**: Original TensorFlow implementation
- **[ExoNet Documentation](README%20for%20exonet.md)**: Enhanced version with domain knowledge

## Citation

If you use this work, please cite the original research and NASA FDL contributions:

### Original AstroNet Research
```bibtex
@article{shallue2018identifying,
  title={Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90},
  author={Shallue, Christopher J and Vanderburg, Andrew},
  journal={The Astronomical Journal},
  volume={155},
  number={2},
  pages={94},
  year={2018},
  month={February},
  doi={10.3847/1538-3881/aa9e09},
  publisher={IOP Publishing}
}
```

### NASA FDL PyTorch Translation
```bibtex
@article{ansdell2018exoplanet,
  title={Exoplanet Detection with Machine Learning},
  author={Ansdell, Megan and Ioannou, Yani and Osborn, Hugh and Sasdelli, Michele},
  journal={Astrophysical Journal Letters},
  volume={869},
  pages={L7},
  year={2018},
  publisher={IOP Publishing}
}
```

### Original TensorFlow Implementation
- **Repository**: [Google Research exoplanet-ml](https://github.com/google-research/exoplanet-ml)
- **Author**: Chris Shallue ([@cshallue](https://github.com/cshallue))
- **Paper**: [Shallue & Vanderburg (2018)](https://arxiv.org/abs/1712.05044)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google Research**: Original AstroNet development and TensorFlow implementation
- **Chris Shallue**: Original AstroNet author ([@cshallue](https://github.com/cshallue))
- **NASA FDL**: PyTorch translation and scientific enhancements
- **Kepler Mission**: Data and scientific foundation
- **PyTorch Team**: Deep learning framework
- **Open Source Community**: Tools and libraries

## Disclaimer

This is not an official Google product. This implementation builds upon the original Google Research work and NASA FDL contributions.