import torch
import torch.nn as nn
from torch import nn

class MaskedBatchNorm1d(nn.Module):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))  # Gamma
            self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))  # Beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.eps = eps
        self.momentum = momentum

        # Running stats
        self.register_buffer("running_mean", torch.zeros(num_features, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(num_features, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, mask=None):
        # x: (B, C, L)
        # mask: (B, 1, L)
        if mask is None:
            mask = torch.ones_like(x[:, 0, :], device=x.device)
        B, C, L = x.size()

        # Ensure mask has the correct shape and type
        # mask: (B, 1, L), dtype=torch.float32
        mask = mask.float()

        # Compute the total number of valid elements (scalar)
        valid_elements = mask.sum()  # Scalar

        # Avoid division by zero
        valid_elements = valid_elements.clamp(min=1)

        if self.training:
            # Compute the mean over valid elements
            # Sum over batch and length dimensions
            sum_x = (x * mask).sum(dim=(0, 2))  # Shape: (C,)
            mean = sum_x / valid_elements  # Shape: (C,)

            # Center the inputs
            x_centered = x - mean.view(1, C, 1)

            # Compute the variance over valid elements
            var = ((x_centered * mask) ** 2).sum(dim=(0, 2)) / valid_elements  # Shape: (C,)

            # Update running statistics
            with torch.no_grad():
                momentum = self.momentum
                self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
                self.running_var = (1 - momentum) * self.running_var + momentum * var
        else:
            # Use running stats during evaluation
            mean = self.running_mean
            var = self.running_var
            
            # Center using running mean
            x_centered = x - mean.view(1, C, 1)

        # Normalize
        x = (
            x_centered / torch.sqrt(var + self.eps).view(1, C, 1)
        ) * mask  # Multiply by mask to zero out padded positions

        # Apply affine transformation if enabled
        if self.affine:
            x = x * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)

        return x