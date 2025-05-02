import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from neuromata.data import DataConfig
from neuromata.data.emoji import load_emoji
from neuromata.data.mnist import load_mnist
from neuromata.model import CAConfig, CAModel
from neuromata.optim import OptimizerConfig, configure_optimizer
from neuromata.utils.image import (
    to_rgb,
    visualize_batch,
)


@dataclass
class LogConfig:
    use_wandb: bool = True
    wandb_project: str = "neuromata"
    wandb_run_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    loss_log_freq: int = 1
    vis_log_freq: int = 100


@dataclass
class TrainConfig:
    use_pattern_pool: bool = False
    pool_size: int = 1024
    batch_size: int = 8
    device: str = "mps"
    n_steps: int = 8000
    data: DataConfig = field(default_factory=DataConfig)
    model: CAConfig = field(default_factory=CAConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    log: LogConfig = field(default_factory=LogConfig)


def train(cfg: TrainConfig):

    if cfg.log.use_wandb:
        wandb.init(
            project=cfg.log.wandb_project,
            name=cfg.log.wandb_run_name,
            config=asdict(cfg),
        )

    target_img, target_idx = load_mnist(cfg.data)
    if cfg.log.use_wandb:
        wandb.log(
            {
                "target_idx": target_idx,
                "target_img": wandb.Image(
                    to_rgb(
                        np.permute_dims(target_img, (1, 2, 0)),
                        cdims=cfg.model.color_channel_n,
                    ),
                    caption="target.jpg",
                ),
            },
            commit=False,
        )

    ca = CAModel(cfg=cfg.model)
    ca.to(cfg.device)

    optimizer, scheduler = configure_optimizer(model=ca, cfg=cfg.optim)

    seed = ca.build_seed(target_img)
    x0 = np.repeat(seed[None, ...], cfg.batch_size, 0)
    x0 = torch.tensor(x0)

    loss_log = []
    for i in range(cfg.n_steps):

        x_target = torch.tensor(target_img[None, ...])
        x_target = x_target.to(cfg.device)

        x = x0.clone()
        x = x.to(cfg.device)

        iter_n = np.random.randint(low=64, high=96)
        for _ in np.arange(iter_n):
            x = ca(x)

        loss = ca.loss_f(x, x_target)

        loss.backward()
        for param in ca.parameters():
            if param.grad is not None:
                grad = param.grad
                norm = grad.norm() + 1e-8
                param.grad = grad / norm
        optimizer.step()
        scheduler.step()

        # grads = [g / (tf.norm(g) + 1e-8) for g in grads]

        step_i = len(loss_log)
        loss_log.append(loss.detach().cpu().numpy())

        if step_i % cfg.log.vis_log_freq == 0:
            img = visualize_batch(
                x0.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                cdims=ca.cfg.color_channel_n,
            )
            if cfg.log.use_wandb:
                wandb.log(
                    {
                        "img": wandb.Image(
                            img, caption="train_batches_%04d.jpg" % step_i
                        )
                    },
                    commit=False,
                )
        if step_i % cfg.log.loss_log_freq == 0:
            print(
                "\r step: %d, log10(loss): %.3f"
                % (len(loss_log), np.log10(loss_log[-1])),
                end="",
            )
            if cfg.log.use_wandb:
                wandb.log(
                    {
                        "loss": loss_log[-1],
                        "lr": scheduler.get_last_lr()[-1],
                    }
                )

            # export_model(ca, "train_log/%04d.weights.h5" % step_i)


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    # convert structured config back to underlying dataclass
    cfg = OmegaConf.to_object(cfg)

    train(cfg=cfg)  # type: ignore


if __name__ == "__main__":
    main()
