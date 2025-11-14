# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of shift and flip invariant convolutional neural networks for flow channel images. The project focuses on predicting laminar flow channel parameters (Cf and St) from flow channel geometry images using custom CNN architectures with specialized invariance properties.

## Development Commands

### Environment Setup
```bash
# Create virtual environment using uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install package in development mode
uv pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run tests with verbose output
pytest -v
```

### Training Models
```bash
# Run the base CNN experiment (uses MockCNN by default)
python flow_channel_cnn/experiments/train_cnn.py

# Run InvariantCNN experiment
python flow_channel_cnn/experiments/train_invariant_cnn.py
# Or using the new modular approach:
python flow_channel_cnn/experiments/train_cnn__invariant.py

# Run StandardCNN experiment
python flow_channel_cnn/experiments/train_cnn__standard.py

# Run ChannelVAE experiment
python flow_channel_cnn/experiments/train_channel_vae.py
```

## Architecture

### Core Models (`flow_channel_cnn/models.py`)

All CNN models inherit from `AbstractCNN(pl.LightningModule)`, which provides:
- Model checkpointing (save/load methods)
- StandardScaler integration for output normalization
- `forward_array()` convenience method for numpy array inputs with batching

**Model Hierarchy:**
- **AbstractCNN**: Base class with common functionality
  - **InvariantCNN**: Shift and flip invariant CNN using custom architecture
  - **StandardCNN**: Baseline CNN without invariance guarantees
  - **MockCNN**: Simple test model for development
  - **ChannelVAE**: Variational autoencoder (separate from CNN hierarchy)

### Key Architectural Components

**InvariantCNN Design:**
- Implements horizontal shift invariance via circular padding + `AdaptivePolyphaseSampling` (APS)
- Implements vertical flip invariance by embedding both original and flipped images, then summing embeddings
- Uses width-wise aggregation (sum over width dimension) in encoder to maintain shift invariance
- Concatenates intermediate convolutional layer embeddings for richer representation

**AdaptivePolyphaseSampling Layer (`flow_channel_cnn/layers.py`):**
- Eliminates shift variance introduced by strided operations
- Computes all possible polyphase components for a given stride
- Selects component with maximum norm (L2 by default) deterministically
- Critical for maintaining shift invariance through downsampling

### Experiment Framework

The project uses **pycomex** for experiment management with a hook-based system:

**Base Experiment (`train_cnn.py`):**
- Defines the complete training pipeline
- Provides default hooks that can be overridden in sub-experiments
- Main hooks:
  - `load_dataset`: Load training/validation/test data
  - `construct_model`: Create model instance (must be overridden)
  - `evaluate_shift_invariance`: Test horizontal shift invariance
  - `evaluate_flip_invariance`: Test vertical flip invariance

**Sub-Experiments (e.g., `train_cnn__invariant.py`):**
- Extend base experiment using `Experiment.extend('train_cnn.py', ...)`
- Override specific hooks (typically just `construct_model`)
- Inherit all other functionality from base experiment
- Configure via global parameters (CONV_UNITS, DENSE_UNITS, EPOCHS, etc.)

### Dataset Structure

Expected dataset files in `flow_channel_cnn/experiments/assets/dataset/`:
- `X_removed_island.npy`: Training images
- `y_removed_island.npy`: Training labels
- `X_test_removed_island.npy`: Test images
- `y_test_removed_island.npy`: Test labels
- `X_flat.npy`: Flat channel images
- `y_flat.npy`: Flat channel labels

Image format: numpy arrays with shape (height, width, channels)
Labels: (Cf, St) parameter pairs

### Custom Callbacks

**GracefulTermination** (`utils.py`):
- Allows Ctrl+C interruption during training without losing progress
- Properly stops trainer when interrupted

**BestModelRestorer** (`utils.py`):
- Tracks best model during training based on specified metric
- Restores best model weights at end of training
- Default: monitors `val_metric` in max mode

**KL Schedulers** (for VAE training):
- `WarmupKLScheduler`: Linear warmup of KL divergence weight
- `CyclicKLScheduler`: Cyclic annealing of KL weight

## Important Implementation Notes

- **Kernel size must be even** for InvariantCNN to maintain shift invariance with circular padding
- Models use **circular padding** in horizontal dimension (for shift invariance)
- StandardScaler is attached to models and saved/loaded with checkpoints
- Training uses **R2 score** as validation metric (higher is better)
- Results automatically saved to `flow_channel_cnn/experiments/results/<experiment_name>/`
- PyTorch Lightning handles all training loops and device management

