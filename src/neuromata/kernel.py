import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KernelConfig:
    kernel_size: int = 5
    angles: list = field(default_factory=lambda: [0, 90])


def sobel_kernel(cfg: KernelConfig) -> np.ndarray:
    """Sobel kernel for edge detection."""

    half_size = math.floor(cfg.kernel_size / 2)

    identify = np.zeros(cfg.kernel_size).astype(np.float32)
    identify[half_size] = 1
    identify = np.array(identify).astype(np.float32)
    identify = np.outer(identify, identify)

    xv, yv = np.meshgrid(
        np.linspace(-half_size, half_size, cfg.kernel_size),
        np.linspace(-half_size, half_size, cfg.kernel_size),
    )
    dx = xv / (xv**2 + yv**2)
    dx[half_size, half_size] = 0
    dy = yv / (xv**2 + yv**2)
    dy[half_size, half_size] = 0

    dlst = [identify]
    for angle in cfg.angles:
        angle_rad = degree_to_rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        d_angle = cos_angle * dx + sin_angle * dy
        d_angle /= np.sum(np.abs(d_angle))
        dlst.append(d_angle)

    kernel = np.stack(dlst, axis=0)[:, None, :, :]
    return kernel.astype(np.float32)


def degree_to_rad(degree: float) -> float:
    """Convert degree to radian."""
    return degree * np.pi / 180.0
