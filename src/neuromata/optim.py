from dataclasses import dataclass, field
from functools import partial

import torch


@dataclass
class OptimizerConfig:
    lr: float = 2e-3
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95

    scheduler: str = "piecewise-constant"
    scheduler_params: dict = field(default_factory=dict)


def piecewise_constant_lr(
    it: int,
    boundaries: list,
    values: list,
) -> float:

    piece = len(boundaries)
    for i, b in enumerate(boundaries):
        if it < b:
            piece = i
            break

    return values[piece]


def configure_optimizer(
    model: torch.nn.Module, cfg: OptimizerConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:

    if cfg.scheduler == "piecewise-constant":
        boundaries = cfg.scheduler_params["boundaries"]
        values = cfg.scheduler_params["values"]
        assert (
            len(boundaries) == len(values) - 1
        ), "len(boundaries) must be equal to len(learn_rates) - 1"
        lr_schedule_fn = partial(
            piecewise_constant_lr,
            boundaries=boundaries,
            values=values,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-7
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_schedule_fn,
    )

    return optimizer, scheduler
