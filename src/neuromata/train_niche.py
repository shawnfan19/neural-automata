import squidpy as sq

from neuromata.data.spatial import SpatialDataModule, SpatialDataset
from neuromata.model import SCAutomaton

# n_steps = 500
# lr = 1e-3
# beta1 = 0.9
# beta2 = 0.95
# epsilon = 1e-8
#
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()
datamodule = SpatialDataModule(batch_size=1, spatial=True)
model = SCAutomaton(n_input=adata.shape[1], n_latent=10)
model.train(max_epochs=200, datamodule=datamodule)
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon
# )
#
# for i in range(n_steps):
#
#     loss = model(X_niche.unsqueeze(0))
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(loss)
