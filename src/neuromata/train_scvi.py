from neuromata.data.spatial import SpatialDataModule, SpatialDataset
from neuromata.scvi_model import SCVI

dataset = SpatialDataset()
datamodule = SpatialDataModule(batch_size=128)
model = SCVI(n_input=dataset.adata.raw.shape[1])  # type: ignore
model.train(max_epochs=200, datamodule=datamodule)
