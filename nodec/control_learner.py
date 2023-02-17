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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        times, trajectory = self.ode.forward(x, self.time_span)
        controls = torch.cat(self.system.control_buffer, axis=1)
        self.system.control_buffer = []

        return times, trajectory, controls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        ds = IntialPointsDataset(self.system.x0_ranges, self.n_trajectories, None)
        trainloader = DataLoader(ds, num_workers=4, batch_size=2)
        return trainloader

    def training_step(self, x0, batch_idx):
        times, trajectory, controls = self.forward(x0)
        loss = self.loss(times, trajectory, controls)

        return {"loss": loss}
