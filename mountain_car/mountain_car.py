import torch
import torch.nn as nn
import pytorch_lightning as pl

from nodec import ControlLearner, ControlledSystem, ControlLoss


class MountainCarLoss(ControlLoss):
    def running_loss(self, times: torch.Tensor, trajectory: torch.Tensor, controls: torch.Tensor):
        return 0.1*torch.sum(trajectory**2)

    def terminal_loss(self, T: torch.Tensor, xT: torch.Tensor, uT: torch.Tensor):
        return 0


class MountainCar(ControlledSystem):
    def __init__(self, controller: nn.Module, m: float=1e3, g: float=2.5e-3):
        self.m = m
        self.g = g
        super().__init__(controller, torch.tensor([[-0.6, 0.4], [0.0, 0.0]]).float())

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        
        position, velocity = x[:, :0], x[:, 1:]
        acceleration = 1/self.m * control - self.g * torch.sin(torch.pi + 3*position)
        x_prime = torch.cat([velocity, acceleration], axis=1)

        return x_prime


controller = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)
system = MountainCar(controller)
loss = MountainCarLoss()
learner = ControlLearner(system, loss, 10, torch.linspace(0, 10, 10, dtype=torch.float32), {})
trainer = pl.Trainer(min_epochs=1, max_epochs=10)
trainer.fit(learner)
