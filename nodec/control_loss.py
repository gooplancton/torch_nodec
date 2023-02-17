import torch
import torch.nn as nn


class ControlLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def running_loss(self, times: torch.Tensor, trajectory: torch.Tensor):
        raise NotImplementedError()
    
    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor):
        raise NotImplementedError()
    
    def forward(self, times: torch.Tensor, trajectory: torch.Tensor):
        T, xT = times[-1], trajectory[-1]
        return torch.mean(self.running_loss(times, trajectory) + self.terminal_loss(T, xT))
