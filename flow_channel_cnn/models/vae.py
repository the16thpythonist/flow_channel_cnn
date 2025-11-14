import math
from typing import List, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.distributions import Normal


# == VARIATIONAL AUTOENCODER ==


class WarmupKLScheduler(pl.Callback):
    """
    Scheduler that linearly increases the KL divergence weight during VAE training.

    This callback implements KL annealing to help prevent posterior collapse in VAEs.
    By starting with a very small KL weight and gradually increasing it, the model can
    first learn to reconstruct inputs before enforcing the latent space regularization.

    :param value_start: Initial KL divergence weight (very small, e.g., 1e-10).
    :param value_end: Final KL divergence weight after warmup.
    :param warmup_steps: Number of training steps over which to linearly increase the KL weight.
    """

    def __init__(self,
                    value_start: float = 1e-10,
                    value_end: float = 5.0,
                    warmup_steps: int = 10_000,
                    ) -> None:
        super().__init__()
        self.value_start = value_start
        self.value_end = value_end
        self.warmup_steps = warmup_steps
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        """
        Update the KL factor after each training batch.

        :param trainer: The PyTorch Lightning trainer.
        :param pl_module: The VAE model being trained.
        """
        if self.step < self.warmup_steps:
            # Linear interpolation from value_start to value_end
            kl_factor = self.value_start + (self.value_end - self.value_start) * (self.step / self.warmup_steps)
        else:
            # After warmup, keep the factor constant at value_end
            kl_factor = self.value_end

        pl_module.kl_factor = kl_factor
        self.step += 1


class CyclicKLScheduler(pl.Callback):
    """
    Scheduler that cyclically varies the KL divergence weight during VAE training.

    This callback implements cyclic KL annealing, which periodically reduces and increases
    the KL weight. This can help the model escape local optima and improve the balance between
    reconstruction quality and latent space regularity throughout training.

    :param value_start: Minimum KL divergence weight in each cycle.
    :param value_end: Maximum KL divergence weight in each cycle.
    :param frequency: Number of training steps per cycle.
    :param warmup_steps: Initial steps before cyclic annealing begins (KL weight stays at value_start).
    """

    def __init__(self,
                 value_start: float = 1e-10,
                 value_end: float = 5.0,
                 frequency: int = 100,
                 warmup_steps: int = 250,
                 ) -> None:
        super().__init__()
        self.value_start = value_start
        self.value_end = value_end
        self.frequency = frequency
        self.warmup_steps = warmup_steps
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        """
        Update the KL factor after each training batch using a cosine schedule.

        :param trainer: The PyTorch Lightning trainer.
        :param pl_module: The VAE model being trained.
        """
        if self.step < self.warmup_steps:
            # Keep KL factor at minimum during warmup
            kl_factor = self.value_start
        else:
            # Cosine annealing within each cycle
            cycle = (self.step - self.warmup_steps) // self.frequency
            x = ((self.step - self.warmup_steps) % self.frequency) / self.frequency
            kl_factor = self.value_start + (self.value_end - self.value_start) * (0.5 * (1 - math.cos(math.pi * x)))

        pl_module.kl_factor = kl_factor
        self.step += 1



class ChannelVAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) with adversarial training for flow channel image generation.

    This model implements a VAE with an additional discriminator network, combining the benefits
    of variational inference with adversarial training. The architecture consists of three main
    components:

    1. **Encoder**: Maps input images to a latent distribution (mean and log-variance).
    2. **Decoder**: Reconstructs images from latent space samples.
    3. **Discriminator**: Distinguishes between real and generated images to improve reconstruction quality.

    The model uses convolutional layers for spatial processing and employs the reparameterization
    trick to enable backpropagation through the stochastic sampling operation.

    **Loss Function:**
    The total VAE loss combines multiple objectives:
    - Reconstruction loss (L1): Measures pixel-wise reconstruction accuracy
    - KL divergence: Regularizes the latent space to follow a unit Gaussian distribution
        - Individual KL loss: Encourages each latent distribution to match N(0,1)
        - Global KL loss: Encourages the batch-level latent statistics to match N(0,1)
    - Adversarial loss: Encourages generated images to fool the discriminator

    :param input_channels: Number of channels in input images (e.g., 1 for grayscale, 3 for RGB).
    :param input_shape: Tuple (height, width) specifying input image dimensions.
    :param units: List of integers specifying the number of channels in each encoder/decoder layer.
        The decoder mirrors this architecture in reverse.
    :param latent_dim: Dimensionality of the latent space representation.
    :param stride: Stride value for downsampling (encoder) and upsampling (decoder).
    :param kernel_size: Convolutional kernel size for encoder and decoder.
    :param learning_rate: Learning rate for both VAE and discriminator optimizers.
    :param kl_factor: Weight coefficient for the KL divergence term in the loss function.
        This is typically adjusted during training using KL annealing schedulers.
    :param use_discriminator: Flag to enable/disable discriminator (currently not fully implemented).
    :param discriminator_units: List of integers specifying the number of channels in each
        discriminator layer.
    """

    def __init__(self,
                 input_channels: int,
                 input_shape: Tuple[int, int],
                 units: List[int],
                 latent_dim: int,
                 stride: int = 2,
                 kernel_size: int = 3,
                 learning_rate: float = 1e-3,
                 kl_factor: float = 0.1,
                 # discriminator related
                 use_discriminator: bool = False,
                 discriminator_units: List[int] = [128, 128, 128, 128, 128],
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.units = units
        self.latent_dim = latent_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kl_factor = kl_factor
        self.use_discriminator = use_discriminator
        self.discriminator_units = discriminator_units

        # ~ Encoder Network ~
        # Progressively downsamples the input image and extracts hierarchical features
        _height, _width = self.input_shape
        self.encoder_layers = nn.ModuleList()
        prev_units = input_channels
        for units in units:
            # Each encoder block: Conv2d -> BatchNorm -> ReLU -> MaxPool
            lay = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_units,
                    out_channels=units,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(units),
                nn.ReLU(),
                # MaxPool reduces spatial dimensions by stride factor
                nn.MaxPool2d(kernel_size=self.stride, stride=self.stride),
            )
            self.encoder_layers.append(lay)
            prev_units = units

            # Track spatial dimensions after downsampling
            _height = math.ceil(_height / self.stride)
            _width = math.ceil(_width / self.stride)

        # Store the shape and dimensionality before the latent space projection
        self.pre_latent_shape = (prev_units, _height, _width)
        self.pre_latent_dim = _height * _width * prev_units

        # ~ Latent Space Projection ~
        # The encoder output is projected to two vectors: mean (mu) and log-variance
        # These parameterize a Gaussian distribution in the latent space
        self.lay_mu = nn.Sequential(
            nn.Linear(self.pre_latent_dim, latent_dim),
        )
        self.lay_log_var = nn.Sequential(
            nn.Linear(self.pre_latent_dim, latent_dim),
        )

        # ~ Decoder Network ~
        # Mirrors the encoder architecture, progressively upsampling from latent space to image space
        self.lay_from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.pre_latent_dim),
        )
        self.decoder_layers = nn.ModuleList()
        prev_units = self.units[-1]
        # Reverse the encoder architecture (excluding the last layer which is handled separately)
        for units in list(reversed(self.units[:-1])):
            # Each decoder block: Conv2d -> BatchNorm -> ReLU -> Upsample
            lay = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_units,
                    out_channels=units,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(units),
                nn.ReLU(),
                # Upsample increases spatial dimensions by stride factor
                nn.Upsample(scale_factor=self.stride),
            )
            self.decoder_layers.append(lay)
            prev_units = units

        # Final decoder layer outputs the reconstructed image with Sigmoid activation (values in [0, 1])
        lay = nn.Sequential(
            nn.Conv2d(
                in_channels=prev_units,
                out_channels=input_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1
            ),
            nn.Upsample(scale_factor=self.stride),
            nn.Sigmoid(),  # Ensures output is in valid pixel range [0, 1]
        )
        self.decoder_layers.append(lay)

        # ~ Discriminator Network ~
        # Used for adversarial training to improve reconstruction quality
        # Disable automatic optimization since we manually handle VAE and discriminator updates
        self.automatic_optimization = False

        self.discriminator_layers = nn.ModuleList()
        prev_units = input_channels
        _height, _width = self.input_shape
        for units in self.discriminator_units:
            # Each discriminator block: Conv2d -> ReLU -> MaxPool (no BatchNorm for stability)
            lay = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_units,
                    out_channels=units,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.stride, stride=self.stride),
            )
            self.discriminator_layers.append(lay)
            prev_units = units

            _height = math.ceil(_height / self.stride)
            _width = math.ceil(_width / self.stride)

        # Final discriminator layer outputs a single probability value (real vs fake)
        lay = nn.Sequential(
            nn.Flatten(),
            nn.Linear(units * _height * _width, 1),
            nn.Sigmoid(),  # Outputs probability in [0, 1]
        )
        self.discriminator_layers.append(lay)
        self.discriminator_shape = (1, _height, _width)

    def _init_weights(self, m):
        """
        Initialize weights using Kaiming (He) normal initialization for ReLU activations.

        This method can be applied to model modules to initialize convolutional and linear
        layer weights. It's currently defined but not automatically applied.

        :param m: A PyTorch module to initialize.
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through the discriminator network.

        The discriminator attempts to classify whether an image is real (from the dataset)
        or fake (reconstructed by the VAE decoder).

        :param x: Input image tensor of shape (batch_size, channels, height, width).

        :returns: Probability score in [0, 1] indicating realness, shape (batch_size, 1).
        """
        for lay in self.discriminator_layers:
            x = lay(x)

        return x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images into latent space distribution parameters.

        The encoder processes the input through convolutional layers and outputs two
        vectors that parameterize a Gaussian distribution: mean (mu) and log-variance.

        :param x: Input image tensor of shape (batch_size, channels, height, width).

        :returns: Tuple of (mu, log_var), each of shape (batch_size, latent_dim).
            - mu: Mean of the latent distribution.
            - log_var: Log-variance of the latent distribution (log(σ²)).
        """
        # Pass through convolutional encoder layers
        for lay in self.encoder_layers:
            x = lay(x)

        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)

        # Project to latent distribution parameters
        mu = self.lay_mu(x)
        log_var = self.lay_log_var(x)
        # Apply softplus to ensure log_var is positive
        log_var = F.softplus(log_var)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to sample from the latent distribution.

        The reparameterization trick allows gradients to flow through the stochastic
        sampling operation by expressing the sample as: z = μ + σ * ε, where ε ~ N(0, 1).

        :param mu: Mean of the latent distribution, shape (batch_size, latent_dim).
        :param log_var: Log-variance of the latent distribution, shape (batch_size, latent_dim).

        :returns: Sampled latent vector z of shape (batch_size, latent_dim).
        """
        # Convert log-variance to standard deviation: σ = exp(0.5 * log(σ²))
        std = torch.exp(0.5 * log_var)
        # Sample random noise from standard normal distribution
        eps = torch.randn_like(std)
        # Apply reparameterization: z = μ + σ * ε
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors back into image space.

        The decoder progressively upsamples the latent representation through
        convolutional layers to reconstruct the input image.

        :param z: Latent vector of shape (batch_size, latent_dim).

        :returns: Reconstructed image tensor of shape (batch_size, channels, height, width).
        """
        # Project latent vector to pre-latent spatial dimensions
        z = self.lay_from_latent(z)

        # Reshape to spatial dimensions
        z = z.view(z.size(0), *self.pre_latent_shape)

        # Pass through decoder layers (upsampling)
        for lay in self.decoder_layers:
            z = lay(z)

        return z

    def sample_latent(self) -> torch.Tensor:
        """
        Sample a random vector from the prior latent distribution N(0, I).

        This can be used to generate new images by sampling from the learned latent space.

        :returns: Random latent vector sampled from N(0, 1), shape (latent_dim,).
        """
        # Sample from standard normal distribution (prior)
        mu = torch.zeros(self.latent_dim)
        sigma = torch.ones(self.latent_dim)

        std_normal = Normal(mu, sigma)
        return std_normal.sample()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a complete forward pass through the VAE.

        This method encodes the input, samples from the latent distribution, and decodes
        back to image space.

        :param x: Input image tensor of shape (batch_size, channels, height, width).

        :returns: Tuple of (x_recon, mu, log_var):
            - x_recon: Reconstructed image, shape (batch_size, channels, height, width).
            - mu: Latent distribution mean, shape (batch_size, latent_dim).
            - log_var: Latent distribution log-variance, shape (batch_size, latent_dim).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with manual optimization for VAE and discriminator.

        This method implements a two-stage training procedure:
        1. Update VAE (encoder + decoder) to minimize reconstruction loss, KL divergence,
           and adversarial loss (fool the discriminator).
        2. Update discriminator to distinguish real from generated images.

        The loss functions are:
        - **Reconstruction Loss (L1)**: Encourages pixel-wise reconstruction accuracy.
        - **KL Divergence**: Regularizes latent space to match a unit Gaussian prior:
            - Individual KL: KL(q(z|x) || p(z)) for each sample
            - Global KL: Encourages batch-level latent statistics to match N(0,1)
        - **Adversarial Loss**: Encourages reconstructions to appear realistic.

        :param batch: Tuple of (images, labels), where labels are typically ignored.
        :param batch_idx: Index of the current batch.

        :returns: None (logs metrics to tensorboard/logger).
        """
        # Retrieve the two optimizers (VAE and discriminator)
        opt_vae, opt_disc = self.optimizers()

        # ==== STEP 1: Train VAE (Encoder + Decoder) ====

        x, _ = batch  # Extract images, ignore labels
        x_recon, mu, log_var = self.forward(x)

        # Reconstruction loss: L1 distance between input and reconstruction
        recon_loss = F.l1_loss(x_recon, x, reduction="mean")

        # KL divergence loss (Individual): Measures how much each q(z|x) deviates from N(0,1)
        # Formula: KL(q||p) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        # Note: The -0.1 coefficient is a hyperparameter that scales this term
        kl_loss_ind = -0.1 * (1 + log_var - log_var.exp())
        kl_loss_ind = torch.mean(torch.sum(kl_loss_ind, axis=1))

        # KL divergence loss (Global): Encourages the batch-level statistics to match N(0,1)
        # This computes KL between the empirical distribution of z and the prior
        Z_mean = torch.mean(mu, axis=0)  # Mean across batch
        Z_log_var = torch.log(torch.var(mu, axis=0))  # Log-variance across batch

        kl_loss_g = -0.5 * (1 + Z_log_var - Z_mean.pow(2) - Z_log_var.exp())
        kl_loss_g = torch.sum(kl_loss_g, axis=0)

        # Total KL loss combines individual and global components
        kl_loss = kl_loss_ind + kl_loss_g

        # Adversarial loss (Generator): Encourages reconstructions to fool the discriminator
        # We want the discriminator to classify reconstructions as "real" (label=1)
        u_fake = self.discriminate(x_recon)
        loss_gen_fake = F.binary_cross_entropy(u_fake, torch.ones_like(u_fake))

        # Also apply adversarial loss to randomly sampled latent vectors
        x_rand = self.decode(torch.randn_like(mu).to(self.device))
        u_rand = self.discriminate(x_rand)
        loss_gen_rand = F.binary_cross_entropy(u_rand, torch.ones_like(u_rand))

        # Total VAE loss: weighted combination of reconstruction, KL, and adversarial losses
        # Coefficients: 1000 for reconstruction, self.kl_factor for KL, 1 for adversarial
        loss_vae = 1000 * recon_loss + self.kl_factor * kl_loss + 1 * (loss_gen_fake + loss_gen_rand)

        # Manually update VAE parameters
        opt_vae.zero_grad()
        self.manual_backward(loss_vae)
        opt_vae.step()

        # ==== STEP 2: Train Discriminator ====

        # Generate new reconstructions (detached from VAE gradients)
        x_recon, mu, log_var = self.forward(x)

        # Discriminator should classify real images as "real" (label=1)
        u_real = self.discriminate(x)
        loss_disc_real = F.binary_cross_entropy(u_real, torch.ones_like(u_real))

        # Discriminator should classify reconstructions as "fake" (label=0)
        u_fake = self.discriminate(x_recon)
        loss_disc_fake = F.binary_cross_entropy(u_fake, torch.zeros_like(u_fake))

        # Total discriminator loss
        loss_disc = (loss_disc_real + loss_disc_fake)

        # Manually update discriminator parameters
        opt_disc.zero_grad()
        self.manual_backward(loss_disc)
        opt_disc.step()

        # ==== Logging ====
        # Log all loss components to tensorboard/logger
        self.log("l_recon", recon_loss, prog_bar=True,)
        self.log("l_kl", kl_loss, prog_bar=True)
        self.log('l_gen_fk', loss_gen_fake, prog_bar=True)
        # Optionally log the KL factor (useful for monitoring annealing schedules)
        # self.log("kl_factor", self.kl_factor, prog_bar=True)
        self.log("l_disc", loss_disc, prog_bar=True)
        self.log("loss", loss_vae, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure separate optimizers for the VAE and discriminator.

        Since we use manual optimization (automatic_optimization=False), we need to return
        two separate optimizers that will be retrieved in training_step via self.optimizers().

        :returns: Tuple of (opt_vae, opt_disc):
            - opt_vae: Adam optimizer for encoder and decoder parameters.
            - opt_disc: Adam optimizer for discriminator parameters.
        """
        # Optimizer for VAE components (encoder + latent projections + decoder)
        opt_vae = torch.optim.Adam(
            list(self.encoder_layers.parameters()) +
            list(self.lay_mu.parameters()) +
            list(self.lay_log_var.parameters()) +
            list(self.lay_from_latent.parameters()) +
            list(self.decoder_layers.parameters()),
            lr=self.learning_rate
        )
        # Optimizer for discriminator
        opt_disc = torch.optim.Adam(self.discriminator_layers.parameters(), lr=self.learning_rate)
        return opt_vae, opt_disc

    def configure_callbacks(self):
        """
        Configure training callbacks for KL annealing.

        Returns a list of callbacks that will be applied during training. By default,
        uses WarmupKLScheduler to gradually increase the KL divergence weight.

        :returns: List of PyTorch Lightning callbacks.
        """
        return [
            # CyclicKLScheduler(),  # Alternative: cyclic annealing
            WarmupKLScheduler(),    # Default: linear warmup
        ]
