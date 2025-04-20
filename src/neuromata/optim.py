from dataclasses import dataclass, field
from functools import partial

import keras
import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.optimizers.schedules import LearningRateSchedule


@dataclass
class OptimizerConfig:
    lr: float = 2e-3
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95

    scheduler: str = "piecewise-constant"
    scheduler_params: dict = field(default_factory=dict)


# class PiecewiseConstantLR(LearningRateSchedule):

#     def __init__(
#         self,
#         lr: float,
#         boundaries: list[int],
#         learn_rates: list[int]):

#         assert lr == learn_rates[0], \
#             "lr must be equal to the first learn rate"
#         assert len(boundaries) == len(learn_rates) - 1, \
#             "len(boundaries) must be equal to len(learn_rates) - 1"

#         self.lr = lr
#         self.boundaries = boundaries
#         self.learn_rates = learn_rates

#     def __call__(
#         self,
#         step: int) -> float:

#         for i, b in enumerate(self.boundaries):
#             if step < b:
#                 return self.learn_rates[i]

#         return self.learn_rates[-1]


def configure_optimizer(cfg: OptimizerConfig):

    if cfg.scheduler == "piecewise-constant":
        boundaries = cfg.scheduler_params["boundaries"]
        values = cfg.scheduler_params["values"]
        assert cfg.lr == values[0], "lr must be equal to the first learn rate"
        assert (
            len(boundaries) == len(values) - 1
        ), "len(boundaries) must be equal to len(learn_rates) - 1"
        scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    optimizer = Adam(
        learning_rate=scheduler, beta_1=cfg.beta1, beta_2=cfg.beta2, epsilon=cfg.epsilon
    )

    return scheduler, optimizer
