"""
Training experiment for ChannelNormalizingFlow model on flow channel dataset.

This experiment trains a normalizing flow model (Glow architecture) on the flow channel
dataset and evaluates it by:
- Plotting training and validation negative log-likelihood over epochs
- Generating random samples from the learned distribution
- Interpolating between samples in the latent space
"""
import os
import random
from typing import Tuple, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import Callback
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from flow_channel_cnn.models import ChannelNormalizingFlow
from flow_channel_cnn.utils import EXPERIMENTS_PATH

# == SOURCE PARAMETERS ==
# The path to the source dataset folder

# :param SOURCE_PATH:
#       The path to the source dataset folder. This folder should contain specific .NPY files which
#       contain the images and the labels of the dataset.
SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'dataset')

# == MODEL PARAMETERS ==
# Parameters related to the model architecture

# :param NUM_FLOW_STEPS:
#       Number of Glow blocks in the normalizing flow. More steps = more expressive model.
NUM_FLOW_STEPS: int = 12
# :param HIDDEN_CHANNELS:
#       Number of hidden channels in the coupling layer networks.
HIDDEN_CHANNELS: int = 128
# :param SPLIT_MODE:
#       Masking pattern for coupling layers. Options: 'checkerboard', 'checkerboard_inv', 'channel', 'channel_inv'
SPLIT_MODE: str = 'checkerboard'
# :param SCALE_MAP:
#       Scaling function for affine coupling. Options: 'sigmoid' (Glow), 'exp' (RealNVP)
SCALE_MAP: str = 'sigmoid'

# == TRAINING PARAMETERS ==

# :param EPOCHS:
#       Number of training epochs.
EPOCHS: int = 100
# :param BATCH_SIZE:
#       Batch size for training.
BATCH_SIZE: int = 20
# :param LEARNING_RATE:
#       Learning rate for the Adam optimizer.
LEARNING_RATE: float = 1e-4

# == EVALUATION PARAMETERS ==

# :param NUM_SAMPLES:
#       Number of random samples to generate from the model.
NUM_SAMPLES: int = 15
# :param NUM_INTERPOLATIONS:
#       Number of interpolation steps between two latent vectors.
NUM_INTERPOLATIONS: int = 10
# :param NUM_INTERPOLATION_PAIRS:
#       Number of interpolation pairs to generate.
NUM_INTERPOLATION_PAIRS: int = 5


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


class MetricsCallback(Callback):
    """
    Callback to track training and validation metrics across epochs.
    """

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_bpds = []
        self.val_bpds = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Record training metrics at the end of each epoch."""
        metrics = trainer.callback_metrics
        if 'train_loss_epoch' in metrics:
            self.train_losses.append(metrics['train_loss_epoch'].item())
        if 'train_bpd_epoch' in metrics:
            self.train_bpds.append(metrics['train_bpd_epoch'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation metrics at the end of each epoch."""
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'].item())
        if 'val_bpd' in metrics:
            self.val_bpds.append(metrics['val_bpd'].item())


def load_dataset(e: Experiment) -> Tuple[list, list]:
    """
    Load the flow channel dataset from numpy files.

    :param e: The experiment object.

    :returns: Tuple of (train_data, test_data) where each is a list of (image, label) tuples.
    """

    # ~ train set
    x_train_path = os.path.join(e.SOURCE_PATH, 'X_removed_island.npy')
    x_train = np.load(x_train_path)

    y_train_path = os.path.join(e.SOURCE_PATH, 'y_removed_island.npy')
    y_train = np.load(y_train_path)

    # ~ test set
    x_test_path = os.path.join(e.SOURCE_PATH, 'X_test_removed_island.npy')
    x_test = np.load(x_test_path)

    y_test_path = os.path.join(e.SOURCE_PATH, 'y_test_removed_island.npy')
    y_test = np.load(y_test_path)

    # ~ flat channels
    x_flat_path = os.path.join(e.SOURCE_PATH, 'X_flat.npy')
    x_flat = np.load(x_flat_path)

    y_flat_path = os.path.join(e.SOURCE_PATH, 'y_flat.npy')
    y_flat = np.load(y_flat_path)

    return (
        [(x, y) for x, y in zip(x_train, y_train)] + [(x, y) for x, y in zip(x_flat[5:], y_flat[5:])],
        [(x, y) for x, y in zip(x_test, y_test)] + [(x, y) for x, y in zip(x_flat[:5], y_flat[:5])],
    )


@experiment
def experiment(e: Experiment):

    e.log('starting normalizing flow experiment...')

    # ~ data loading
    e.log('loading flow channel dataset...')
    train, test = load_dataset(e)

    # Transpose to (channels, height, width) format and crop to 128 height
    x_train = np.array([x.transpose(2, 0, 1) for x, _ in train])[:, :, :128, :]
    y_train = np.array([y for _, y in train])

    x_test = np.array([x.transpose(2, 0, 1) for x, _ in test])[:, :, :128, :]
    y_test = np.array([y for _, y in test])

    e.log('example shape:')
    e.log(x_train[0].shape)
    input_shape = (x_train[0].shape[1], x_train[0].shape[2])
    height, width = input_shape
    e.log(f'Input shape: {input_shape}')

    # ~ model training
    e.log('constructing normalizing flow model...')
    model = ChannelNormalizingFlow(
        input_channels=1,
        input_shape=input_shape,
        num_flow_steps=e.NUM_FLOW_STEPS,
        hidden_channels=e.HIDDEN_CHANNELS,
        split_mode=e.SPLIT_MODE,
        scale_map=e.SCALE_MAP,
        learning_rate=e.LEARNING_RATE,
    )
    e.log('model summary:')
    e.log(str(model))

    # Convert data to PyTorch tensors
    e.log('converting data to tensors...')
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    e.log('creating tensor datasets...')
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=e.BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=e.BATCH_SIZE,
        shuffle=False
    )

    # Set up metrics callback
    metrics_callback = MetricsCallback()

    # Train the model
    e.log('training model...')
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        callbacks=[metrics_callback],
    )
    trainer.fit(model, train_loader, test_loader)
    model.eval()

    # ~ Plot training curves
    e.log('plotting training curves...')

    fig, ((ax_loss, ax_bpd)) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # Plot negative log-likelihood
    epochs = range(1, len(metrics_callback.train_losses) + 1)
    ax_loss.plot(epochs, metrics_callback.train_losses, label='Train NLL', marker='o')
    ax_loss.plot(epochs, metrics_callback.val_losses, label='Val NLL', marker='s')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Negative Log-Likelihood')
    ax_loss.set_title('Training and Validation NLL')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot bits per dimension
    ax_bpd.plot(epochs, metrics_callback.train_bpds, label='Train BPD', marker='o')
    ax_bpd.plot(epochs, metrics_callback.val_bpds, label='Val BPD', marker='s')
    ax_bpd.set_xlabel('Epoch')
    ax_bpd.set_ylabel('Bits per Dimension')
    ax_bpd.set_title('Training and Validation Bits/Dim')
    ax_bpd.legend()
    ax_bpd.grid(True, alpha=0.3)

    plt.tight_layout()
    e.commit_fig('training_curves.png', fig)

    # ~ Generate random samples
    e.log(f'generating {e.NUM_SAMPLES} random samples from the model...')
    with torch.no_grad():
        samples = model.sample(e.NUM_SAMPLES)
        samples_np = samples.cpu().numpy()

    # Plot random samples in a grid
    n_cols = 5
    n_rows = (e.NUM_SAMPLES + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if e.NUM_SAMPLES > 1 else [axes]

    for i in range(e.NUM_SAMPLES):
        axes[i].imshow(samples_np[i, 0, :, :], cmap='Greys')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')

    # Hide remaining subplots
    for i in range(e.NUM_SAMPLES, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    e.commit_fig('random_samples.png', fig)

    # ~ Latent space interpolations
    e.log(f'generating latent space interpolations...')

    for pair_idx in range(e.NUM_INTERPOLATION_PAIRS):
        e.log(f' * interpolation pair {pair_idx + 1}/{e.NUM_INTERPOLATION_PAIRS}')

        # Choose two random test images
        idx1, idx2 = random.sample(range(len(x_test)), 2)
        x1 = torch.tensor(x_test[idx1:idx1+1], dtype=torch.float32)
        x2 = torch.tensor(x_test[idx2:idx2+1], dtype=torch.float32)

        # Transform to latent space using inverse flow
        with torch.no_grad():
            z1 = model.flow.inverse(x1)
            z2 = model.flow.inverse(x2)

        # Interpolate in latent space
        alphas = np.linspace(0, 1, e.NUM_INTERPOLATIONS)

        fig, axes = plt.subplots(2, e.NUM_INTERPOLATIONS, figsize=(e.NUM_INTERPOLATIONS * 2, 4))

        for i, alpha in enumerate(alphas):
            # Linear interpolation in latent space
            z_interp = (1 - alpha) * z1 + alpha * z2

            # Transform back to image space using forward flow
            with torch.no_grad():
                x_interp = model.flow.forward(z_interp)
                x_interp_np = x_interp.cpu().numpy()

            axes[1, i].imshow(x_interp_np[0, 0, :, :], cmap='Greys')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'α={alpha:.2f}')

        # Plot original images in top row
        axes[0, 0].imshow(x_test[idx1, 0, :, :], cmap='Greys')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Image 1')

        axes[0, -1].imshow(x_test[idx2, 0, :, :], cmap='Greys')
        axes[0, -1].axis('off')
        axes[0, -1].set_title('Image 2')

        # Hide middle axes in top row
        for ax in axes[0, 1:-1]:
            ax.axis('off')

        plt.tight_layout()
        e.commit_fig(f'interpolation_{pair_idx}.png', fig)

    # ~ Compute and log final test set metrics
    e.log('computing final test set metrics...')
    test_nlls = []
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(test_loader):
            log_prob = model.forward(x_batch)
            nll = -log_prob.mean().item()
            test_nlls.append(nll)

    mean_test_nll = np.mean(test_nlls)
    n_dims = 1 * height * width
    mean_test_bpd = mean_test_nll / (n_dims * np.log(2))

    e.log(f'Final Test NLL: {mean_test_nll:.4f}')
    e.log(f'Final Test BPD: {mean_test_bpd:.4f}')

    # Save metrics to experiment
    e['final_test_nll'] = mean_test_nll
    e['final_test_bpd'] = mean_test_bpd
    e['train_nlls'] = metrics_callback.train_losses
    e['val_nlls'] = metrics_callback.val_losses
    e['train_bpds'] = metrics_callback.train_bpds
    e['val_bpds'] = metrics_callback.val_bpds

    e.log('experiment completed!')


experiment.run_if_main()
