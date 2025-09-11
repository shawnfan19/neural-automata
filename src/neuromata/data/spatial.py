import lightning as L
import squidpy as sq
import torch
from scvi import REGISTRY_KEYS
from torch.utils.data import DataLoader, Dataset


class SpatialDataset(Dataset):

    def __init__(self, spatial: bool = False):
        self.img = sq.datasets.visium_hne_image()
        self.adata = sq.datasets.visium_hne_adata()
        self.spatial = spatial

    def __len__(self):
        if self.spatial:
            return 1
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if self.spatial:
            X = self.adata.raw.X.toarray()
            X = torch.from_numpy(X)
            max_rows = self.adata.obs["array_row"].max() + 1
            max_cols = self.adata.obs["array_col"].max() + 1

            X_niche = torch.zeros((max_rows, max_cols, X.shape[1]))
            y_coords = self.adata.obs["array_row"].to_numpy()
            x_coords = self.adata.obs["array_col"].to_numpy()
            X_niche[y_coords, x_coords] = X
            return {REGISTRY_KEYS.X_KEY: X_niche}
        else:
            cell = self.adata.raw.X[idx, :].toarray().ravel()
            return {REGISTRY_KEYS.X_KEY: torch.tensor(cell)}


class SpatialDataModule(L.LightningDataModule):

    def __init__(self, batch_size: int, spatial: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.spatial = spatial

    def setup(self, stage):
        self.dataset = SpatialDataset(spatial=self.spatial)
        return None

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
