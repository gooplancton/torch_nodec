import torch
import torch.nn as nn


class ControlledSystem(nn.Module):
    def __init__(self, controller: nn.Module, x0_ranges: torch.Tensor):
        super().__init__()
        self.controller = controller
        self.x0_ranges = x0_ranges
        self.x_dim = x0_ranges.shape[0]

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def _dynamics(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        x_prime = self.dynamics(t, x, control)
        return torch.cat([x_prime, control], axis=1)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        x = x[:, :self.x_dim]
        control = self.controller.forward(x)

        return self._dynamics(t, x, control)
