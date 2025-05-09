import math
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import PIL.Image as Image
import torch
import wandb

from neuromata.model import Automaton


@dataclass
class LogConfig:
    use_wandb: bool = True
    ckpt_dir: str = "checkpoints"
    run_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_project: str = "neuromata"
    loss_log_freq: int = 1
    vis_log_freq: int = 100
    eval_log_freq: int = 100


class Logger:

    def __init__(
        self,
        cfg: LogConfig,
        model: Automaton,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ):

        self.cfg = cfg
        self.run_dir = os.path.join(cfg.ckpt_dir, cfg.run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def init_wandb(self, exp_cfg: dict):
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config=exp_cfg,
            )

    def log_image(self, img: Image.Image, name: str = "img", caption: str = "caption"):

        if self.cfg.use_wandb:
            wandb.log(
                {
                    name: wandb.Image(img, caption=caption),
                },
                commit=False,
            )

    def log_tensor(self, tensor: np.ndarray, name: str = "tensor"):

        if self.cfg.use_wandb:
            wandb.log(
                {
                    f"{name}": wandb.Histogram(tensor.tolist()),
                    f"{name}/mean": np.mean(tensor, axis=-1),
                    f"{name}/std": np.std(tensor, axis=-1),
                    f"{name}/min": np.min(tensor, axis=-1),
                    f"{name}/max": np.max(tensor, axis=-1),
                },
                commit=False,
            )

    def log_grad(self):
        if self.cfg.use_wandb:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {
                            f"grad/{name}": param.grad.norm().item(),
                        },
                        commit=False,
                    )

    def log_metrics(self, step: int, loss: float, growth_iter: int) -> None:

        if step % self.cfg.loss_log_freq == 0:
            print(
                "\r step: %d, log10(loss): %.3f" % (step, np.log10(loss)),
                end="",
            )
            if self.cfg.use_wandb:
                wandb.log(
                    {
                        "step": step,
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[-1],
                        "growth_iter": growth_iter,
                    },
                )


def collage(img_lst: list[np.ndarray], ncol: int = 10) -> np.ndarray:

    l = len(img_lst)
    row_lst = []
    for i in range(math.ceil(l / ncol)):
        start = i * ncol
        end = (i + 1) * ncol
        if end > l:
            filler = [np.zeros_like(img_lst[0])] * (end - l)
            end = l
            row = np.hstack(img_lst[start:end] + filler)
        else:
            row = np.hstack(img_lst[start:end])
        row_lst.append(row)
    collage = np.vstack(row_lst)

    return collage
