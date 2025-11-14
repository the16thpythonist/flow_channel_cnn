"""
Flow Channel CNN Models Package

This package contains CNN, VAE, and Normalizing Flow models for flow channel analysis.
The package is organized into submodules for better code organization:
- cnn: Convolutional neural network models
- vae: Variational autoencoder models
- flow: Normalizing flow models

All models are re-exported at the package level for backwards compatibility.
"""

# Import CNN models
from flow_channel_cnn.models.cnn import (
    AbstractCNN,
    InvariantCNN,
    StandardCNN,
    MockCNN,
)

# Import VAE models and schedulers
from flow_channel_cnn.models.vae import (
    ChannelVAE,
    WarmupKLScheduler,
    CyclicKLScheduler,
)

# Import Normalizing Flow models
from flow_channel_cnn.models.flow import (
    ChannelNormalizingFlow,
)

# Define what gets exported when using "from flow_channel_cnn.models import *"
__all__ = [
    # CNN models
    'AbstractCNN',
    'InvariantCNN',
    'StandardCNN',
    'MockCNN',
    # VAE models
    'ChannelVAE',
    'WarmupKLScheduler',
    'CyclicKLScheduler',
    # Flow models
    'ChannelNormalizingFlow',
]
