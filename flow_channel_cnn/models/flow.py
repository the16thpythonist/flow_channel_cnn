"""
Normalizing Flow models for flow channel image generation and density estimation.

This module implements normalizing flow architectures using the normflows library,
providing invertible transformations for generative modeling of flow channel images.
"""
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import normflows as nf


class ChannelNormalizingFlow(pl.LightningModule):
    """
    Normalizing Flow model for flow channel image generation and density estimation.

    This model uses Glow-style architecture with coupling layers and checkerboard masking
    to create an invertible transformation between image space and a simple base distribution
    (typically Gaussian). Unlike VAEs, normalizing flows are fully invertible and provide
    exact likelihood computation.

    **Architecture:**
    The model uses the Glow architecture from normflows, which consists of multiple flow steps.
    Each GlowBlock contains:
    1. **ActNorm**: Activation normalization layer (data-dependent initialization)
    2. **Invertible1x1Conv**: Learnable invertible convolution for channel mixing
    3. **AffineCouplingBlock**: Affine transformation with spatial masking
       - Split: Divides features using checkerboard or channel pattern
       - AffineCoupling: Transforms one part conditioned on the other
       - Merge: Recombines the transformed features

    The model works directly on image-shaped tensors (B, C, H, W), preserving spatial structure
    throughout the transformation. This is crucial for modeling spatial relationships in images.

    **Checkerboard vs Channel Masking:**
    - Checkerboard: Splits pixels in a checkerboard pattern across spatial dimensions.
      Good for preserving fine-grained spatial structure.
    - Channel: Splits along the channel dimension. More efficient but requires multiple channels.

    **Training:**
    The model is trained by maximizing the likelihood of the training data under the
    learned distribution, which corresponds to minimizing negative log-likelihood.

    :param input_channels: Number of channels in input images (e.g., 1 for grayscale).
    :param input_shape: Tuple (height, width) specifying input image dimensions.
    :param num_flow_steps: Number of Glow blocks. More steps = more expressive model.
    :param hidden_channels: Number of hidden channels in the coupling layer networks.
    :param split_mode: Masking pattern for coupling layers. Options:
        - 'checkerboard': Spatial checkerboard pattern
        - 'checkerboard_inv': Inverted checkerboard
        - 'channel': Split along channel dimension
        - 'channel_inv': Inverted channel split
    :param scale_map: Scaling function for affine coupling. Options:
        - 'sigmoid': Used in Glow (recommended)
        - 'exp': Used in RealNVP
    :param learning_rate: Learning rate for the Adam optimizer.
    """

    def __init__(self,
                 input_channels: int,
                 input_shape: Tuple[int, int],
                 num_flow_steps: int = 8,
                 hidden_channels: int = 64,
                 split_mode: str = 'checkerboard',
                 scale_map: str = 'sigmoid',
                 learning_rate: float = 1e-3,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.num_flow_steps = num_flow_steps
        self.hidden_channels = hidden_channels
        self.split_mode = split_mode
        self.scale_map = scale_map
        self.learning_rate = learning_rate

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        height, width = input_shape

        # For checkerboard mode with 1 channel, we need at least 2D spatial dimensions
        if input_channels == 1 and 'checkerboard' in split_mode:
            # This is valid - checkerboard works on spatial dimensions
            pass
        elif input_channels == 1 and 'channel' in split_mode:
            # Channel split requires at least 2 channels
            raise ValueError(
                f"Channel split mode '{split_mode}' requires at least 2 input channels, "
                f"but got {input_channels}. Use 'checkerboard' mode instead."
            )

        # ~ Base Distribution ~
        # The base distribution is a simple Gaussian in the image space
        # Shape: (channels, height, width)
        latent_shape = (input_channels, height, width)
        self.base = nf.distributions.DiagGaussian(latent_shape, trainable=False)

        # ~ Build Glow Flow Transformations ~
        flows = []

        for i in range(num_flow_steps):
            # Each Glow block contains:
            # - ActNorm
            # - Invertible 1x1 Convolution (if channels > 1)
            # - Affine Coupling Layer with split/merge

            # Alternate split mode for better mixing
            # This ensures all pixels/channels get transformed
            if 'checkerboard' in split_mode:
                # Alternate between checkerboard and checkerboard_inv
                current_split_mode = 'checkerboard' if i % 2 == 0 else 'checkerboard_inv'
            else:
                # For channel mode, use as specified
                current_split_mode = split_mode

            glow_block = nf.flows.GlowBlock(
                channels=input_channels,
                hidden_channels=hidden_channels,
                split_mode=current_split_mode,
                scale=True,  # Use affine transformation (scale + shift)
                scale_map=scale_map,
            )
            flows.append(glow_block)

        # Create the normalizing flow model
        self.flow = nf.NormalizingFlow(q0=self.base, flows=flows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of input images under the learned distribution.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Log probability for each sample, shape (batch_size,).
        """
        # normflows expects images in (B, C, H, W) format - no flattening needed!
        log_prob = self.flow.log_prob(x)
        return log_prob

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate new samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Generated images of shape (num_samples, channels, height, width).
        """
        # Sample from the flow (transformation from base to data)
        with torch.no_grad():
            samples, _ = self.flow.sample(num_samples)

        return samples

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Compute the training loss for a single batch.

        The loss is the negative log-likelihood of the data under the learned distribution.
        Minimizing this is equivalent to maximum likelihood estimation.

        Args:
            batch: Tuple of (images, labels). Labels are ignored for density estimation.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss value.
        """
        x, _ = batch  # Ignore labels for unsupervised density estimation

        # Compute negative log-likelihood
        log_prob = self.forward(x)
        loss = -torch.mean(log_prob)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_nll', loss, prog_bar=False, on_epoch=True)

        # Also log bits per dimension (common metric for generative models)
        # bits/dim = nll / (dims * log(2))
        n_dims = self.input_channels * self.input_shape[0] * self.input_shape[1]
        bits_per_dim = loss / (n_dims * math.log(2))
        self.log('train_bpd', bits_per_dim, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Compute the validation loss for a single batch.

        Args:
            batch: Tuple of (images, labels). Labels are ignored.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss value.
        """
        x, _ = batch

        log_prob = self.forward(x)
        loss = -torch.mean(log_prob)

        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_nll', loss, prog_bar=False, on_epoch=True)

        n_dims = self.input_channels * self.input_shape[0] * self.input_shape[1]
        bits_per_dim = loss / (n_dims * math.log(2))
        self.log('val_bpd', bits_per_dim, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            Adam optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
