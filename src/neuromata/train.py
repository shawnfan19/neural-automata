import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import keras
import matplotlib.pylab as pl
import numpy as np
import tensorflow as tf
import wandb
from omegaconf import OmegaConf

from neuromata.model import CAModel, export_model
from neuromata.pool import SamplePool
from neuromata.utils.image import (
    generate_pool_figures,
    load_emoji,
    to_rgba,
    visualize_batch,
)

os.environ["FFMPEG_BINARY"] = "ffmpeg"


@dataclass
class LogConfig:
    use_wandb: bool = True
    wandb_project: str = "neuromata"
    wandb_run_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class CAConfig:
    channel_n = 16
    cell_fire_rate = 0.5


@dataclass
class TrainConfig:
    use_pattern_pool: bool = False
    pool_size: int = 1024
    batch_size: int = 8
    target_emoji: str = "🦎"
    target_size: int = 40
    target_padding: int = 16
    model: CAConfig = field(default_factory=CAConfig)
    log: LogConfig = field(default_factory=LogConfig)


def plot_loss(loss_log):
    pl.figure(figsize=(10, 4))
    pl.title("Loss history (log10)")
    pl.plot(np.log10(loss_log), ".", alpha=0.1)
    pl.show()


def loss_f(x, pad_target):
    return tf.reduce_mean(tf.square(to_rgba(x) - pad_target), [-2, -3, -1])


def train(cfg: TrainConfig):

    target_img = load_emoji(cfg.target_emoji, max_size=cfg.target_size)

    p = cfg.target_padding
    pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w = pad_target.shape[:2]
    seed = np.zeros([h, w, cfg.model.channel_n], np.float32)
    seed[h // 2, w // 2, 3:] = 1.0

    ca = CAModel(
        channel_n=cfg.model.channel_n,
        fire_rate=cfg.model.cell_fire_rate,
    )
    ca.dmodel.summary()

    loss_log = []

    lr = 2e-3
    lr_sched = keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
    trainer = keras.optimizers.Adam(lr_sched)

    # loss0 = loss_f(x=seed, pad_target=pad_target).numpy()
    if cfg.use_pattern_pool:
        pool = SamplePool(x=np.repeat(seed[None, ...], cfg.pool_size, 0))

    @tf.function
    def train_step(x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = ca(x)
            loss = tf.reduce_mean(loss_f(x, pad_target))
        grads = g.gradient(loss, ca.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        trainer.apply_gradients(zip(grads, ca.weights))
        return x, loss

    if cfg.log.use_wandb:
        wandb.init(
            project=cfg.log.wandb_project,
            name=cfg.log.wandb_run_name,
            config=asdict(cfg),
        )

    for i in range(8000 + 1):
        if cfg.use_pattern_pool:
            batch = pool.sample(cfg.batch_size)
            x0 = batch.x
            loss_rank = loss_f(x0, pad_target).numpy().argsort()[::-1]
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

        if step_i % 100 == 0:
            if cfg.use_pattern_pool:
                generate_pool_figures(pool, step_i)
            img = visualize_batch(x0, x, step_i)
            if cfg.log.use_wandb:
                wandb.log(
                    {
                        "step": step_i,
                        "loss": loss.numpy(),
                        "img": wandb.Image(
                            img, caption="train_batches_%04d.jpg" % step_i
                        ),
                    }
                )
            # export_model(ca, "train_log/%04d.weights.h5" % step_i)

            print(
                "\r step: %d, log10(loss): %.3f" % (len(loss_log), np.log10(loss)),
                end="",
            )


def main():

    # cfg = OmegaConf.load("configs/train.yaml")
    # cfg = OmegaConf.merge(TrainConfig(), cfg)

    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.to_object(default_cfg)

    train(cfg=cfg)


if __name__ == "__main__":
    main()
