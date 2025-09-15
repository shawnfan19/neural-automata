from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)
from torch.distributions import NegativeBinomial, Normal
from torch.distributions import kl_divergence as kl


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "none", "softmax"],
    ):
        """Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input.
        n_output
            The dimensionality of the output.
        link_var
            The final non-linearity.
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output


def percept(X: torch.Tensor):

    # batch_size, height, width, hidden = X.shape

    X180 = torch.roll(X, shifts=2, dims=2)
    X180[..., :, :2] = 0

    X000 = torch.roll(X, shifts=-2, dims=2)
    X000[..., :, -2:] = 0

    X060 = torch.roll(X, shifts=(-1, -1), dims=(1, 2))
    X060[..., -1:, -1:] = 0

    X120 = torch.roll(X, shifts=(-1, 1), dims=(1, 2))
    X120[..., -1:, :1] = 0

    X240 = torch.roll(X, shifts=(1, 1), dims=(1, 2))
    X240[..., :1, :1] = 0

    X300 = torch.roll(X, shifts=(1, -1), dims=(1, 2))
    X300[..., :1, -1:] = 0

    X_percept = torch.cat([X, X000, X060, X120, X180, X240, X300], dim=-1)

    return X_percept


class NicheAutomaton(BaseModuleClass):

    def __init__(
        self,
        n_input: int,
        n_latent: int = 10,
    ):
        super().__init__()

        self.n_latent = n_latent
        n_hidden = n_latent * 7
        self.nca = torch.nn.Sequential(
            torch.nn.Conv2d(n_hidden, n_latent, kernel_size=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_latent, n_latent, kernel_size=1, bias=False),
        )

        self.embed = torch.nn.Linear(n_input, n_latent)
        self.mean_encoder = torch.nn.Linear(n_latent, n_latent)
        self.var_encoder = torch.nn.Linear(n_latent, n_latent)
        self.decoder = MLP(n_latent, n_input, "softmax")
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        return {"x": tensors[REGISTRY_KEYS.X_KEY]}

    @auto_move_data
    def inference(self, x: torch.Tensor, iter: int = 2) -> dict[str, torch.Tensor]:

        x_ = torch.log1p(x)
        x_ = self.embed(x_)

        # x_ = torch.permute(x_, (0, 3, 1, 2))
        for _ in range(iter):
            x_ = percept(x_)
            x_ = torch.permute(x_, (0, 3, 1, 2))
            x_ = self.nca(x_)
            x_ = torch.permute(x_, (0, 2, 3, 1))

        qz_m = self.mean_encoder(x_)
        qz_v = torch.exp(self.var_encoder(x_))

        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        return {"qz_m": qz_m, "qz_v": qz_v, "z": z}

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:  # type: ignore
        return {
            "z": inference_outputs["z"],
            "library": torch.sum(tensors[REGISTRY_KEYS.X_KEY], dim=-1, keepdim=True),
        }

    @auto_move_data
    def generative(
        self, z: torch.Tensor, library: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        px_scale = self.decoder(z)
        px_rate = library * px_scale
        theta = torch.exp(self.log_theta)

        return {
            "px_scale": px_scale,
            "theta": theta,
            "px_rate": px_rate,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> LossOutput:
        # here, we would like to form the ELBO. There are two terms:
        #   1. one that pertains to the likelihood of the data
        #   2. one that pertains to the variational distribution
        # so we extract all the required information
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the
        # pytorch code)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = (
            NegativeBinomial(total_count=theta, logits=nb_logits)
            .log_prob(x)
            .sum(dim=-1)
        )

        # term 2
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=-1)

        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)
        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_lik,
            kl_local=kl_divergence,
            kl_global=0.0,
        )


class SCAutomaton(UnsupervisedTrainingMixin, BaseModelClass):

    def __init__(
        self,
        adata,
        n_input: int,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.module = NicheAutomaton(
            n_input=n_input,
            n_latent=n_latent,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"SCVI Automaton Model with the following params: \nn_latent: {n_latent}"
        )

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        X = adata.X.toarray()
        X = torch.from_numpy(X)
        max_rows = adata.obs["array_row"].max() + 1
        max_cols = adata.obs["array_col"].max() + 1

        X_niche = torch.zeros((max_rows, max_cols, X.shape[1]))
        y_coords = adata.obs["array_row"].to_numpy()
        x_coords = adata.obs["array_col"].to_numpy()
        X_niche[y_coords, x_coords] = X
        tensors = {REGISTRY_KEYS.X_KEY: X_niche.unsqueeze(0)}
        inference_inputs = self.module._get_inference_input(tensors)
        outputs = self.module.inference(**inference_inputs)
        qz_m = outputs["qz_m"].squeeze(0).cpu().numpy()

        latent = qz_m[adata.obs["array_row"], adata.obs["array_col"], :]

        return latent
