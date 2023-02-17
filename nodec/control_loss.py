import torch
import torch.nn as nn


class ControlLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def running_loss(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        raise NotImplementedError()
    
    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        raise NotImplementedError()
    
    def forward(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        T, xT, uT = times[-1], trajectory[-1], controls[-1]
        return torch.mean(self.running_loss(times, trajectory, controls) + self.terminal_loss(T, xT, uT))
