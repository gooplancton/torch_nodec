import torch
import torch.nn as nn
import pytorch_lightning as pl

from nodec import ControlLearner, ControlledSystem, ControlLoss


class QuadrotorLoss(ControlLoss):
    def __init__(self, tracked_trajectory: torch.Tensor):
        self.y_star = tracked_trajectory[:, 0].unsqueeze(1)
        self.z_star = tracked_trajectory[:, 1].unsqueeze(1)
        super().__init__()

    def running_loss(
        self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor
    ):
        y, z = trajectory[:, :, 0], trajectory[:, :, 2]
        return torch.sum((y - self.y_star) ** 2 + (z - self.z_star) ** 2, axis=0)

    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        ydotT = xT[:, 1]
        zdotT = xT[:, 3]
        return ydotT**2 + zdotT**2


class Quadrotor(ControlledSystem):
    def __init__(
        self, controller: nn.Module, g: float = 9.8, m: float = 1.0, I: float = 1
    ):
        x0_ranges = torch.tensor([[-0.5, 0.5]] + [[0, 0] for _ in range(5)]).float()
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
            [y_dot, y_ddot, z_dot, z_ddot, theta_dot, theta_ddot],
            axis=1,
        )

        return x_prime


if __name__ == "__main__":
    ys = torch.linspace(0, 4, 50).unsqueeze(1)
    zs = torch.sqrt(ys)
    tracked_trajectory = torch.cat(
        [ys, zs], axis=1
    )  # we follow sqrt(y) for y in [0, 4]
    controller = nn.Sequential(nn.Linear(6, 20), nn.Tanh(), nn.Linear(20, 2))
    system = Quadrotor(controller)
    loss = QuadrotorLoss(tracked_trajectory)
    learner = ControlLearner(
        system, loss, 1000, 5, torch.linspace(0, 5, 50, dtype=torch.float32), {}
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=2)
    trainer.fit(learner)
    print("finished")
