from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CAConfig:
    seed: int = 42
    autoencode_seed: bool = False
    phenotype_projector: bool = False
    channel_n: int = 16
    color_channel_n: int = 3
    cell_fire_rate: float = 0.5
    initialize: str = "center-point"


class CAModel(torch.nn.Module):

    def __init__(self, cfg: CAConfig):
        super().__init__()
        self.cfg = cfg

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

        self.alpha_slice = slice(self.cfg.color_channel_n, self.cfg.color_channel_n + 1)
        self.initialize_slice = slice(self.cfg.color_channel_n, None)

        print(f"built CA model with parameter count: {count_parameters(self)}")

    def build_seed(self, x: torch.Tensor) -> torch.Tensor:

        b, _, h, w = x.shape
        cent_y, cent_x = h // 2, w // 2
        x0 = torch.zeros(b, self.cfg.channel_n, h, w, device=x.device)
        if self.cfg.initialize == "center-point":
            x0[:, self.initialize_slice, cent_y, cent_x] = 1.0
        elif self.cfg.initialize == "inside-point":
            y_coords, x_coords = torch.where(x[0, self.cfg.color_channel_n, :, :] > 0)
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.choice(len(x_coords))
            x0[:, self.initialize_slice, y_coords[idx], x_coords[idx]] = 1.0
        elif self.cfg.initialize == "circle":
            y_indices, x_indices = np.ogrid[:h, :w]
            radius = min(h, w) // 2
            mask = ((x_indices - cent_x) ** 2 + (y_indices - cent_y) ** 2) <= radius**2
            x0[:, self.initialize_slice, mask] = 1.0
        else:
            raise ValueError(f"Unknown initialize method: {self.cfg.initialize}")

        return x0

    def get_living_mask(self, x):

        alpha = self.life(x)

        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def loss_f(self, x: torch.Tensor, pad_target: torch.Tensor) -> torch.Tensor:

        pheno = self.express(x)
        alpha = self.life(x)
        x = torch.cat([pheno, alpha], dim=1)

        batch_pad_target = pad_target.expand(x.shape[0], *pad_target.shape[1:])

        return F.mse_loss(x, batch_pad_target, reduction="mean")

    def perceive(self, x: torch.Tensor, angle: float = 0.0):

        y = F.conv2d(
            input=x, weight=self.kernel, stride=1, padding="same", groups=self.cfg.channel_n  # type: ignore
        )

        return y

    def express(self, x: torch.Tensor) -> torch.Tensor:

        pheno = x[:, : self.cfg.color_channel_n, :, :]

        return pheno

    def life(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        alpha = x[:, self.alpha_slice, :, :]

        return alpha

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


class AutoencodeCA(CAModel):

    def __init__(self, cfg: CAConfig):
        super().__init__(cfg=cfg)

        self.seed_pos_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.cfg.color_channel_n + 1, 8, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        print(
            f"built CA model + autoencoder with parameter count: {count_parameters(self)}"
        )

    def build_seed(self, x: torch.Tensor) -> torch.Tensor:

        b, _, h, w = x.shape
        x0 = torch.zeros(b, self.cfg.channel_n, h, w, device=x.device)

        pos_heatmap = self.seed_pos_encoder(x)
        seed_coords = soft_argmax_2d(pos_heatmap)

        x0 = plant_seed_differentiably(
            feature_map=x0,
            coordinates=seed_coords,
            sigma=1.0,
        )

        return x0


class PhenoProjectorCA(CAModel):

    def __init__(self, cfg: CAConfig):
        super().__init__(cfg=cfg)

        self.phenotype_projector = torch.nn.Sequential(
            torch.nn.Conv2d(self.cfg.channel_n, 8, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=1),
            torch.nn.ReLU(),
        )

        self.alpha_slice = slice(0, 1)
        self.initialize_slice = slice(None, None)

        print(
            f"built CA model + phenotype projector with parameter count: {count_parameters(self)}"
        )

    def express(self, x: torch.Tensor) -> torch.Tensor:

        pheno = self.phenotype_projector(x)

        return pheno


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D soft-argmax of a batch of heatmaps.

    Args:
        heatmaps: Tensor of shape (B, C, H, W)

    Returns:
        coords: Tensor of shape (B, C, 2) with (x, y) coordinates for each keypoint.
    """
    B, C, H, W = heatmaps.shape

    # Flatten spatial dimensions and apply softmax
    heatmaps_flat = heatmaps.view(B, C, -1)
    softmax = F.softmax(heatmaps_flat, dim=-1)

    # Create coordinate grids
    coords_x = torch.linspace(0, W - 1, W, device=heatmaps.device)
    coords_y = torch.linspace(0, H - 1, H, device=heatmaps.device)
    yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")  # (H, W)

    # Flatten coordinate grids
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    # Compute expected x and y positions
    exp_x = torch.sum(softmax * xx, dim=-1)  # (B, C)
    exp_y = torch.sum(softmax * yy, dim=-1)  # (B, C)

    coords = torch.stack([exp_x, exp_y], dim=-1)  # (B, C, 2)
    return coords


def plant_seed_differentiably(
    feature_map: torch.Tensor, coordinates: torch.Tensor, sigma: float = 1.0
):
    """
    Write values to feature map using differentiable Gaussian attention.

    Args:
        feature_map: [B, C, H, W] tensor to write to
        coordinates: [B, N, 2] tensor of (x, y) coordinates
        values: [B, N, C] tensor of values to write
        sigma: Width of Gaussian kernel
    """
    _, channels, H, W = feature_map.shape
    B, N, _ = coordinates.shape

    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=coordinates.device),
        torch.arange(W, device=coordinates.device),
    )
    grid = torch.stack([x_grid, y_grid], dim=-1).float()  # [H, W, 2]

    # Expand grid and coordinates
    grid = grid.unsqueeze(0).unsqueeze(1).expand(B, N, H, W, 2)  # [B, N, H, W, 2]
    coords = (
        coordinates.unsqueeze(2).unsqueeze(3).expand(B, N, H, W, 2)
    )  # [B, N, H, W, 2]

    # gaussian weight for each position
    dist_sq = ((grid - coords) ** 2).sum(dim=-1)  # [B, N, H, W]
    weights = torch.exp(-dist_sq / (2 * sigma**2))  # [B, N, H, W]

    # Apply values with attention weights
    # values_expanded = values.unsqueeze(2).unsqueeze(3).expand(B, N, height, width, channels)
    # values_expanded = values_expanded.permute(0, 1, 4, 2, 3)  # [B, N, C, H, W]
    weights_expanded = weights.unsqueeze(2)  # [B, N, 1, H, W]

    # update = (values_expanded * weights_expanded).sum(dim=1)  # [B, C, H, W]
    update = (1.0 * weights_expanded).sum(dim=1)  # [B, C, H, W]

    return feature_map + update


def count_parameters(model: torch.nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
