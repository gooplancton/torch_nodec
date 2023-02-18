import torch
import torch.nn as nn
import pytorch_lightning as pl

from nodec import ControlLearner, ControlledSystem, ControlLoss


class QuadrotorLoss(ControlLoss):
    def __init__(self, tracked_trajectory: torch.Tensor):
        self.tracked_trajectory = tracked_trajectory
        super().__init__()

    def running_loss(
        self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor
    ):
        y, y_star, z, z_star = (
            trajectory[:, :, 0],
            self.tracked_trajectory[:, 0],
            trajectory[:, :, 2],
            self.tracked_trajectory[:, 1],
        )

        return torch.mean((y - y_star)**2 + (z-z_star)**2, axis=0)

    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        return 0


class Quadrotor(ControlledSystem):
    def __init__(
        self, controller: nn.Module, g: float = 9.8, m: float = 1.0, I: float = 1
    ):
        x0_ranges = torch.tensor([[-0.5, 0.5] for _ in range(6)]).float()
        control_dim = 2
        self.g = g
        self.m = m
        self.I = I
        super().__init__(controller, x0_ranges, control_dim)

    def dynamics(
        self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        y_dot, z_dot, theta, theta_dot = x[:, 1:2], x[:, 3:4], x[:, 4:5], x[:, 5:6]
        u1, u2 = control[:, :1], control[:, 1:]
        y_ddot = -(u1 / self.m) * torch.sin(theta)
        z_ddot = -self.g + (u1 / self.m) * torch.cos(theta)
        theta_ddot = u2 / self.I
        x_prime = torch.cat(
            [y_dot, y_ddot, z_dot, z_ddot, theta_dot, theta_ddot], axis=1
        )

        return x_prime


if __name__ == "__main__":
    controller = nn.Sequential(nn.Linear(6, 10), nn.Tanh(), nn.Linear(10, 2))
    system = Quadrotor(controller)
    tracked_trajectory = torch.tensor([])
    loss = QuadrotorLoss(tracked_trajectory)
    learner = ControlLearner(
        system, loss, 1000, torch.linspace(0, 10, 100, dtype=torch.float32), {}
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=5)
    trainer.fit(learner)
