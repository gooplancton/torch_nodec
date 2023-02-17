from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class IntialPointsDataset(Dataset):
    def __init__(self, ranges: torch.Tensor, N: int, seed: Optional[int] = None) -> None:
        if seed:
            torch.random.seed(seed)

        self.ranges = ranges
        self.x_dim = ranges.shape[0]
        self.N = N
        self.points = [self._generate_point() for _ in range(N)]
    
    def _generate_point(self) -> torch.Tensor:
        r1, r2 = self.ranges[:, 0], self.ranges[:, 1]
        point = (r1 - r2) * torch.rand(self.x_dim) + r2
        return point
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.points[index]
