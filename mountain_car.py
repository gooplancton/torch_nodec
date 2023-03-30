import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from nodec import ControlLearner, ControlledSystem, ControlLoss
from visualization import train_and_plot_state_trajectory


class MountainCarLoss(ControlLoss):
    def running_loss(
        self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor
    ):
        u = controls[:, :, 0]
        return 2*torch.mean(u**2, axis=0)

    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        posT = xT[:, 0]
        return -20*posT


class MountainCar(ControlledSystem):
    def __init__(self, controller: nn.Module, power=0.0015, alpha=0.0025):
        self.power = power
        self.alpha = alpha
        super().__init__(controller, torch.tensor([[-0.6, -0.4], [0.0, 0.0]]).float())

    def dynamics(
        self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        control = torch.clamp(control, -1, 1)
        position, velocity = x[:, :1], x[:, 1:]
        acceleration = self.power*control - self.alpha*torch.cos(3*position)
        x_prime = torch.cat([velocity, acceleration], axis=1)

        return x_prime


if __name__ == "__main__":
    controller = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))
    system = MountainCar(controller)
    loss = MountainCarLoss()
    learner = ControlLearner(
        system,
        loss,
        n_trajectories=2000,
        batch_size=10,
        time_span=torch.linspace(0, 200, 200, dtype=torch.float32),
        ode_params={},
    )
    train_and_plot_state_trajectory(learner, 1, 4, 0, True, lambda x: x, "t", "position", "./mountaincar.png")
