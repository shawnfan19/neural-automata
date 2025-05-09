from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf

from neuromata.data import DataConfig
from neuromata.data.mnist import load_mnist
from neuromata.log import LogConfig, Logger, collage
from neuromata.model import Automaton, CAConfig
from neuromata.optim import OptimizerConfig, configure_optimizer
from neuromata.utils.image import (
    to_grayscale,
    to_pil,
)


@dataclass
class TrainConfig:
    use_pattern_pool: bool = False
    pool_size: int = 1024
    batch_size: int = 8
    device: str = "mps"
    n_steps: int = 8000
    growth_iter: int = 64
    data: DataConfig = field(default_factory=DataConfig)
    model: CAConfig = field(default_factory=CAConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    log: LogConfig = field(default_factory=LogConfig)


def eval_step(model: Automaton, x: torch.Tensor, iter_n: int):

    x_lst = []
    x_pheno_lst = []
    x_alpha_lst = []

    for _ in np.arange(iter_n):

        x = model(x)
        pheno = model.express(x)
        alpha = model.life(x)

        x_lst.append(x.squeeze().detach().cpu().numpy())
        x_pheno_lst.append(pheno.squeeze().detach().cpu().numpy())
        x_alpha_lst.append(alpha.squeeze().detach().cpu().numpy())

    return x_lst, x_pheno_lst, x_alpha_lst


def train(cfg: TrainConfig):

    ca = Automaton(cfg=cfg.model)
    ca.to(cfg.device)

    optimizer, scheduler = configure_optimizer(model=ca, cfg=cfg.optim)

    logger = Logger(cfg.log, model=ca, optimizer=optimizer, scheduler=scheduler)
    logger.init_wandb(exp_cfg=asdict(cfg))

    target_img, target_np = load_mnist(cfg.data)
    logger.log_image(img=target_img, name="target_img", caption="target_img")

    loss_log = []
    for i in range(cfg.n_steps):

        x_target = torch.tensor(target_np[None, ...])
        x_target = x_target.repeat(cfg.batch_size, 1, 1, 1)
        x_target = x_target.to(cfg.device)

        x0, seed_values, seed_x, seed_y = ca.build_seed(x_target)
        x = x0

        growth_iter = cfg.growth_iter
        for _ in np.arange(growth_iter):
            x = ca(x)

        loss = ca.loss_f(x, x_target)

        optimizer.zero_grad()
        loss.backward()
        logger.log_grad()

        if cfg.optim.layerwise_norm:
            for param in ca.parameters():
                if param.grad is not None:
                    grad = param.grad
                    norm = grad.norm() + 1e-8
                    param.grad = grad / norm

        if cfg.optim.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                ca.parameters(), max_norm=cfg.optim.grad_clip
            )

        optimizer.step()
        scheduler.step()

        step_i = len(loss_log)
        loss_log.append(loss.detach().cpu().numpy())

        if step_i % cfg.log.vis_log_freq == 0:

            batch_size = x0.shape[0]
            before = ca.express(x0)
            after = ca.express(x)
            before = before.squeeze(1).detach().cpu().numpy()
            after = after.squeeze(1).detach().cpu().numpy()
            before = [to_grayscale(i) for i in before]
            after = [to_grayscale(i) for i in after]

            x0_row = collage(before, ncol=batch_size)
            x_row = collage(after, ncol=batch_size)
            batch_collage = collage([x0_row, x_row], ncol=1)
            logger.log_image(
                img=to_pil(batch_collage),
                name="train_batch",
                caption="train_batch_%04d" % step_i,
            )

        if step_i % cfg.log.eval_log_freq == 0:

            logger.log_tensor(tensor=seed_x.detach().cpu().numpy(), name="seed_x")
            logger.log_tensor(tensor=seed_y.detach().cpu().numpy(), name="seed_y")
            logger.log_tensor(
                tensor=seed_values.squeeze(1).detach().cpu().numpy(),
                name="seed_tensor",
            )
            x_lst, x_pheno_lst, x_alpha_lst = eval_step(
                model=ca,
                x=x0[[0], ...],
                iter_n=growth_iter,
            )
            x_pheno_lst = [to_grayscale(i) for i in x_pheno_lst]
            x_alpha_lst = [to_grayscale(i) for i in x_alpha_lst]
            evo_pheno = to_pil(collage(x_pheno_lst, ncol=batch_size))
            evo_alpha = to_pil(collage(x_alpha_lst, ncol=batch_size))
            logger.log_image(
                img=evo_pheno,
                name="evo_phenotype",
                caption="evo_phenotype_%04d" % step_i,
            )
            logger.log_image(
                img=evo_alpha,
                name="evo_alpha",
                caption="evo_alpha_%04d" % step_i,
            )
            for i in range(cfg.model.channel_n):
                x_i_lst = [to_grayscale(x[i, :, :]) for x in x_lst]
                evo_i = to_pil(collage(x_i_lst, ncol=batch_size))
                logger.log_image(
                    img=evo_i,
                    name="evo_%d" % i,
                    caption="evo_%d_%04d" % (i, step_i),
                )

        logger.log_metrics(
            step=step_i,
            loss=loss_log[-1],
            growth_iter=growth_iter,
        )


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
