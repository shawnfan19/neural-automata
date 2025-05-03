from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CAConfig:
    seed: int = 42
    channel_n: int = 16
    color_channel_n: int = 3
    cell_fire_rate: float = 0.5
    initialize: str = "center-point"


class CAModel(torch.nn.Module):

    def __init__(self, cfg: CAConfig):
        super().__init__()
        self.cfg = cfg

        self.rng = np.random.default_rng(cfg.seed)

        self.update_rule = torch.nn.Sequential(
            torch.nn.Conv2d(self.cfg.channel_n * 3, 128, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.cfg.channel_n, kernel_size=1),
        )

        torch.nn.init.xavier_uniform_(self.update_rule[0].weight)  # type: ignore
        torch.nn.init.constant_(self.update_rule[0].bias, 0)  # type: ignore
        torch.nn.init.constant_(self.update_rule[2].weight, 0)  # type: ignore
        torch.nn.init.constant_(self.update_rule[2].bias, 0)  # type: ignore

        angle = 0.0

        identify = np.array([0, 1, 0]).astype(np.float32)
        identify = np.outer(identify, identify)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c, s = np.cos(angle), np.sin(angle)

        kernel = np.stack([identify, c * dx - s * dy, s * dx + c * dy], axis=0)[
            :, None, :, :
        ]
        kernel = np.tile(kernel, (self.cfg.channel_n, 1, 1, 1))

        self.register_buffer("kernel", torch.tensor(kernel, dtype=torch.float32))

        print(f"built model with parameter count: {count_parameters(self)}")

    def build_seed(self, x: np.ndarray):

        _, h, w = x.shape
        cent_y, cent_x = h // 2, w // 2
        x0 = np.zeros([self.cfg.channel_n, h, w], np.float32)
        if self.cfg.initialize == "center-point":
            x0[self.cfg.color_channel_n :, cent_y, cent_x] = 1.0
        elif self.cfg.initialize == "inside-point":
            y_candidates, x_candidates = np.where(x[self.cfg.color_channel_n, :, :] > 0)
            idx = self.rng.choice(len(x_candidates))
            x0[self.cfg.color_channel_n :, y_candidates[idx], x_candidates[idx]] = 1.0
        elif self.cfg.initialize == "circle":
            y_indices, x_indices = np.ogrid[:h, :w]
            radius = min(h, w) // 2
            mask = ((x_indices - cent_x) ** 2 + (y_indices - cent_y) ** 2) <= radius**2
            x0[self.cfg.color_channel_n :, mask] = 1.0
        else:
            raise ValueError(f"Unknown initialize method: {self.cfg.initialize}")

        return x0

    def get_living_mask(self, x):

        alpha = x[:, self.cfg.color_channel_n : self.cfg.color_channel_n + 1, :, :]

        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def loss_f(self, x: torch.Tensor, pad_target: torch.Tensor) -> torch.Tensor:

        x = x[:, : pad_target.shape[1], :, :]
        batch_pad_target = pad_target.expand(x.shape[0], *pad_target.shape[1:])

        return F.mse_loss(x, batch_pad_target, reduction="mean")

    def perceive(self, x: torch.Tensor, angle: float = 0.0):

        y = F.conv2d(
            input=x, weight=self.kernel, stride=1, padding="same", groups=self.cfg.channel_n  # type: ignore
        )

        return y

    def forward(
        self,
        x: torch.Tensor,
        fire_rate: Optional[float] = None,
        angle: float = 0.0,
        step_size: float = 1.0,
    ):

        pre_life_mask = self.get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.update_rule(y) * step_size
        if fire_rate is None:
            fire_rate = self.cfg.cell_fire_rate

        update_mask = torch.rand(x[:, :1, :, :].shape, device=x.device) <= fire_rate
        x = x + dx * update_mask.to(torch.float32)

        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        return x * life_mask.to(torch.float32)


def count_parameters(model: torch.nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
