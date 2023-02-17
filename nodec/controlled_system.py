import torch
import torch.nn as nn


class ControlledSystem(nn.Module):
    def __init__(self, controller: nn.Module, x0_ranges: torch.Tensor):
        super().__init__()
        self.controller = controller
        self.x0_ranges = x0_ranges
        self.control_buffer = []

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        control = self.controller.forward(x)
        self.control_buffer.append(control)

        return self.dynamics(t, x, control)
