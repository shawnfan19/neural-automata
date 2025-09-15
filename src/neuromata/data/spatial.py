import lightning as L
import squidpy as sq
import torch
from scvi import REGISTRY_KEYS
from torch.utils.data import DataLoader, Dataset


class SquidpyDataset(Dataset):

    def __init__(self, spatial: bool = False, dataset: str = "visium-hne"):

        if dataset == "visium-hne":
            self.img = sq.datasets.visium_hne_image()
            adata = sq.datasets.visium_hne_adata()
            adata.X = adata.raw.X
            self.adata = adata
        else:
            raise NotImplementedError
        self.spatial = spatial

    def __len__(self):
        if self.spatial:
            return 1
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if self.spatial:
            X = self.adata.X.toarray()
            X = torch.from_numpy(X)
            max_rows = self.adata.obs["array_row"].max() + 1
            max_cols = self.adata.obs["array_col"].max() + 1

            X_niche = torch.zeros((max_rows, max_cols, X.shape[1]))
            y_coords = self.adata.obs["array_row"].to_numpy()
            x_coords = self.adata.obs["array_col"].to_numpy()
            X_niche[y_coords, x_coords] = X
            return {REGISTRY_KEYS.X_KEY: X_niche}
        else:
            cell = self.adata.X[idx, :].toarray().ravel()
            return {REGISTRY_KEYS.X_KEY: torch.Tensor(cell)}


class SpatialDataModule(L.LightningDataModule):

    def __init__(
        self, dataset: str, batch_size: int | None = None, spatial: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.spatial = spatial

    @property
    def anndata(self):
        return self.squidpy_dataset.adata

    @property
    def n_var(self):
        return self.squidpy_dataset.adata.shape[1]

    def prepare_data(self):
        self.squidpy_dataset = SquidpyDataset(
            dataset=self.dataset, spatial=self.spatial
        )
        if self.batch_size is None:
            self.batch_size = self.squidpy_dataset.adata.shape[0]
            print(f"batch size (inferred): {self.batch_size}")

    def train_dataloader(self):
        return DataLoader(self.squidpy_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.squidpy_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.squidpy_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(
            self.squidpy_dataset, batch_size=self.batch_size, shuffle=False
        )
