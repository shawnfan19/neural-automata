import numpy as np
from mnist import MNIST

from neuromata.data import DataConfig
from neuromata.utils.image import np2pil


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
    target = np.expand_dims(target, axis=-1)
    target = np.repeat(target, 2, axis=-1)
    target[..., 1] = (target[..., 0] > 0).astype(np.uint8) * 255

    target_img = np2pil(target)
    target_img = target_img.resize((cfg.size, cfg.size))
    target_img = target_img.convert("LA")

    target_img = np.array(target_img)
    target_img = target_img.astype("float32") / 255.0

    target_img = np.permute_dims(target_img, (2, 0, 1))

    p = cfg.pad
    target_img = np.pad(target_img, [(0, 0), (p, p), (p, p)])

    return target_img, selected_idx
