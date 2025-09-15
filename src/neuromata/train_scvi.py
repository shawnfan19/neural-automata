import argparse
import os
from pathlib import Path

import torch

from neuromata.data.spatial import SpatialDataModule
from neuromata.env import AUTOMATON_CKPT_DIR
from neuromata.model import SCAutomaton
from neuromata.scvi_model import SCVI

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="visium-hne")
parser.add_argument("--model", type=str, default="scvi")
parser.add_argument("--max_epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()
dataset = args.dataset
print(f"dataset: {dataset}")
max_epochs = args.max_epochs
print(f"model: {args.model}")
if args.model == "scvi":
    model_cls = SCVI
    spatial = False
elif args.model == "automaton":
    model_cls = SCAutomaton
    spatial = True
else:
    raise NotImplementedError
print(f"max_epochs: {max_epochs}")
batch_size = args.batch_size
if batch_size is not None:
    print(f"batch_size: {batch_size}")
run_name = args.run_name
if run_name is None:
    run_name = f"{args.model}_{dataset}"
run_dir = Path(AUTOMATON_CKPT_DIR) / run_name
print(f"writing to {run_dir}")
os.makedirs(run_dir, exist_ok=True)

gpu = torch.cuda.is_available()
print(f"GPU available: {gpu}")
device = "cuda" if gpu else "cpu"

datamodule = SpatialDataModule(dataset=dataset, spatial=spatial)
datamodule.prepare_data()
adata = datamodule.anndata
model_cls.setup_anndata(adata)
print(f"adata UUID (assigned by setup_anndata): {adata.uns['_scvi_uuid']}")
print(
    f"AnnDataManager: {model_cls._setup_adata_manager_store[adata.uns['_scvi_uuid']]}"
)
model = model_cls(adata=adata, n_input=datamodule.n_var)  # type: ignore
model.train(max_epochs=max_epochs, datamodule=datamodule)
embd = model.get_latent_representation(adata)
adata.obsm["X_embd"] = embd
adata.write_h5ad(run_dir / f"{dataset}.h5ad")
