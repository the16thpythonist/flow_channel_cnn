import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePolyphaseSampling(nn.Module):
    """
    Adaptive Polyphase Sampling (APS) Layer.
    
    :param stride: The stride for downsampling.
    :param p: The norm to use for selecting the polyphase component.
    """
    
    def __init__(self, 
                 stride: int = 2, 
                 p: int = 2
                 ):
        
        super(AdaptivePolyphaseSampling, self).__init__()
        self.stride = stride
        self.p = p

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the APS layer. Given an input tensor ``x``, this method essentially applies a strided 
        reduction of the input shape by selecting the polyphase component with the maximum norm.
        
        :param x: Input tensor of shape (batch_size, channels, height, width)
        
        :returns: torch.Tensor of the shape (batch_size, channels, height // stride, width // stride)
        """
        b, c, h, w = x.shape
        
        pad_h = (0, 1) if h % 2 != 0 else (0, 0)  # Add 1 row if height is odd
        pad_w = (0, 1) if w % 2 != 0 else (0, 0)  # Add 1 column if width is odd

        # Apply circular padding
        x = F.pad(x, pad_w + pad_h, mode='circular')
        
        s = self.stride
        # Generate polyphase components
        # polyphase: (stride^2, batch_size, channels, height, width)
        polyphase = [
            x[:, :, i::s, j::s] for i in range(s) for j in range(s)
        ]
        
        # Calculate the norm of each component
        # norms: (stride^2, batch_size)
        norms = [torch.norm(component, p=self.p, dim=(1, 2, 3)) for component in polyphase]
        norms = torch.stack(norms, dim=0) 
        
        # Select the component with the maximum norm
        max_indices = torch.argmax(norms, dim=0)  # Shape: (batch_size,)
        # Apply stride to the selected components
        selected = [polyphase[i][batch_idx] for batch_idx, i in enumerate(max_indices)]
        selected = torch.stack(selected, dim=0)
        
        return selected


class ResBlock2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        A simple 2D Residual Block.
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int): The stride of the convolutional kernel.
            padding (int): The padding of the convolutional kernel.
        """
        super(ResBlock2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        
        # Linear layer for identity mapping
        self.identity_mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        identity = self.identity_mapping(x)
        out = self.conv(x)
        out = out + identity
        return out