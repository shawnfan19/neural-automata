import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import keras
import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

import wandb
from omegaconf import OmegaConf

from neuromata.data import DataConfig
from neuromata.data.emoji import load_emoji
from neuromata.data.mnist import load_mnist
from neuromata.model import CAConfig, CAModel, export_model
from neuromata.optim import OptimizerConfig, configure_optimizer
from neuromata.pool import SamplePool
from neuromata.utils.image import (
    generate_pool_figures,
    to_rgb,
    visualize_batch,
)

os.environ["FFMPEG_BINARY"] = "ffmpeg"


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

    if cfg.data.dataset == "emoji":
        target_img = load_emoji(cfg.data)
    elif cfg.data.dataset == "mnist":
        target_img, target_idx = load_mnist(cfg.data)
        if cfg.log.use_wandb:
            wandb.log(
                {
                    "target_idx": target_idx,
                    "target_img": wandb.Image(
                        to_rgb(target_img, cdims=cfg.model.color_channel_n),
                        caption="target.jpg",
                    ),
                },
                commit=False,
            )

    p = cfg.data.pad
    pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])

    ca = CAModel(cfg=cfg.model)
    ca.dmodel.summary()

    # lr = 2e-3
    # lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
    # optim = keras.optimizers.Adam(lr_schedule)
    lr_schedule, optim = configure_optimizer(cfg.optim)

    seed = ca.build_seed(pad_target)
    # loss0 = loss_f(x=seed, pad_target=pad_target).numpy()
    if cfg.use_pattern_pool:
        pool = SamplePool(x=np.repeat(seed[None, ...], cfg.pool_size, 0))

    @tf.function
    def train_step(x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = ca(x)
            loss = tf.reduce_mean(ca.loss_f(x, pad_target))
        grads = g.gradient(loss, ca.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        optim.apply_gradients(zip(grads, ca.weights))
        return x, loss

    loss_log = []
    for i in range(cfg.n_steps):
        if cfg.use_pattern_pool:
            batch = pool.sample(cfg.batch_size)
            x0 = batch.x
            loss_rank = ca.loss_f(x0, pad_target).numpy().argsort()[::-1]
            x0 = x0[loss_rank]
            x0[:1] = seed
        else:
            x0 = np.repeat(seed[None, ...], cfg.batch_size, 0)

        x, loss = train_step(x0)

        if cfg.use_pattern_pool:
            batch.x[:] = x
            batch.commit()

        step_i = len(loss_log)
        loss_log.append(loss.numpy())

        if step_i % cfg.log.vis_log_freq == 0:
            if cfg.use_pattern_pool:
                generate_pool_figures(pool, step_i)
            img = visualize_batch(x0, x, cdims=ca.color_channel_n)
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
                "\r step: %d, log10(loss): %.3f" % (len(loss_log), np.log10(loss)),
                end="",
            )
            if cfg.log.use_wandb:
                wandb.log(
                    {
                        "loss": loss,
                        "lr": lr_schedule(step_i),
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

    train(cfg=cfg)


if __name__ == "__main__":
    main()
