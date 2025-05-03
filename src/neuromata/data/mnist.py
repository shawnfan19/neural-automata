import numpy as np
from mnist import MNIST

from neuromata.data import DataConfig
from neuromata.utils.image import to_pil


class MNISTDataset:

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        mndata = MNIST(cfg.dataset_path)
        images, labels = mndata.load_training()
        self.images = np.array(images)
        self.labels = np.array(labels)

    def __len__(self):
        return self.images.shape[0]

    def load_batch(self, idx):

        batch_images = self.images[idx]
        batch_labels = self.labels[idx]
        batch_images = batch_images.reshape(-1, self.cfg.size, self.cfg.size, 1)
        batch_images = batch_images.astype("float32") / 255.0

        return batch_images, batch_labels


def load_mnist(cfg: DataConfig):

    rng = np.random.default_rng(seed=cfg.seed)

    ds = MNISTDataset(cfg)
    target_idx = np.arange(len(ds))[ds.labels == int(cfg.target)]
    selected_idx = rng.choice(target_idx)
    target = ds.images[selected_idx]

    target = target.reshape(28, 28).astype(np.uint8)

    target_img = to_pil(target)
    target_img = target_img.resize((cfg.size, cfg.size))

    target_np = np.array(target_img)
    phenotype_np = (target_np / 255.0).astype(np.float32)
    alpha_np = (target_np > 0).astype(np.float32)
    target_np = np.stack([phenotype_np, alpha_np], axis=0)

    p = cfg.pad
    target_np = np.pad(target_np, [(0, 0), (p, p), (p, p)])

    return target_img, target_np
