from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torchdyn.core import NeuralODE
from torch.utils.data import DataLoader

from nodec.control_loss import ControlLoss
from nodec.controlled_system import ControlledSystem
from nodec.dataset import IntialPointsDataset


class ControlLearner(pl.LightningModule):
    def __init__(
        self,
        system: ControlledSystem,
        loss: ControlLoss,
        n_trajectories: int = 10,
        time_span: Optional[torch.Tensor] = None,
        ode_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.loss = loss
        self.n_trajectories = n_trajectories
        self.ode = NeuralODE(self.system, **ode_params)
        self.time_span = (
            time_span
            if time_span is not None
            else torch.linspace(0, 1, 2, dtype=torch.float32)
        )
    
    def differentiate_impulse(self, impulse: torch.Tensor) -> torch.Tensor:
        dt = (self.time_span[-1] - self.time_span[0])/len(self.time_span)
        tmp = (impulse - torch.cat((impulse[-1:], impulse[:-1]))) / dt
        controls = torch.cat((tmp[1:2], tmp[1:]), dim=0)
        
        return controls

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        times, trajectory_with_impulse = self.ode.forward(x, self.time_span)
        x_dim = self.system.x_dim
        trajectory = trajectory_with_impulse[:, :, :x_dim]
        impulse = trajectory_with_impulse[:, :, x_dim:]
        controls = self.differentiate_impulse(impulse)

        return times, trajectory, controls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        ds = IntialPointsDataset(self.system.x0_ranges, self.system.control_dim, self.n_trajectories, None)
        trainloader = DataLoader(ds, num_workers=4, batch_size=2)
        return trainloader

    def training_step(self, x0, batch_idx):
        times, trajectory, controls = self.forward(x0)
        loss = self.loss(times, trajectory, controls)

        return {"loss": loss}
