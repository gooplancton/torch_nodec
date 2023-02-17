import torch
import torch.nn as nn
import pytorch_lightning as pl

from nodec import ControlLearner, ControlledSystem, ControlLoss


class CartPoleLoss(ControlLoss):
    def running_loss(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        theta = trajectory[:, :, 2]
        return torch.mean(theta**2, axis=0)

    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        vT = xT[:, 1]
        thetaT = xT[:, 2]
        return 4 * thetaT**2 + vT**2


class CartPoleLossAlt(ControlLoss):
    def running_loss(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        pos = trajectory[:, :, 0]
        theta = trajectory[:, :, 2]
        return torch.mean(theta**2 + 0.1*(pos**2), axis=0)
    
    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        vT = xT[:, 1]
        thetaT = xT[:, 2]
        return 4 * thetaT**2 + vT**2
    
    def extra_loss(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        u = controls[:, :, 0]
        # return 0.001*torch.max(controls**2, axis=0).values.squeeze(1)
        return 0.001*torch.max(u**2, axis=0).values


class CartPole(ControlledSystem):
    def __init__(self, controller, m_c=1.0, m_p=0.1, l=0.5, g=9.81):
        super().__init__(
            controller,
            torch.tensor(
                [
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                ]
            ).float(),
        )
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g

    def dynamics(
        self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        x_dot_c, theta, theta_dot = x[:, 1:2], x[:, 2:3], x[:, 3:]

        acceleration = (
            control + self.m_p * self.l * (theta_dot**2) * torch.sin(theta)
        ) / (self.m_c + self.m_p)

        pole_acceleration = (
            (self.m_c + self.m_p) * self.g * torch.sin(theta)
            - control * torch.cos(theta)
            - self.m_p * self.l * (theta_dot**2) * torch.sin(theta) * torch.cos(theta)
        ) / (self.l * (self.m_c + self.m_p * (torch.sin(theta) ** 2)))

        x_dot = torch.cat([x_dot_c, acceleration, theta_dot, pole_acceleration], axis=1)

        return x_dot


controller = nn.Sequential(nn.Linear(4, 10), nn.Tanh(), nn.Linear(10, 1))
system = CartPole(controller)
loss = CartPoleLossAlt()
learner = ControlLearner(
    system, loss, 100, torch.linspace(0, 10, 100, dtype=torch.float32), {}
)
trainer = pl.Trainer(min_epochs=1, max_epochs=50)
trainer.fit(learner)
