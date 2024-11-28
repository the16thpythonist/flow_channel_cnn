import torch
import torch.nn as nn


class AdaptivePolyphaseSampling(nn.Module):
    
    def __init__(self, stride=2, p=2):
        """
        Adaptive Polyphase Sampling (APS) Layer.
        Args:
            stride (int): The stride for downsampling.
            p (int): The norm to use for selecting the polyphase component.
        """
        super(AdaptivePolyphaseSampling, self).__init__()
        self.stride = stride
        self.p = p

    def forward(self, x):
        b, c, h, w = x.shape
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
        selected = torch.stack([polyphase[i][batch_idx] for batch_idx, i in enumerate(max_indices)], dim=0)
        
        return selected

