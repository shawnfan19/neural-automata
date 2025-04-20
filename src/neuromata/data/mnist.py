import numpy as np
import tensorflow as tf
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

    def load_batch(self, idx):

        batch_images = self.images[idx]
        batch_labels = self.labels[idx]
        batch_images = batch_images.reshape(-1, self.cfg.size, self.cfg.size, 1)
        batch_images = batch_images.astype("float32") / 255.0

        return batch_images, batch_labels


def load_mnist(cfg: DataConfig):

    ds = MNISTDataset(cfg)
    is_target = ds.labels == int(cfg.target)
    candidates = ds.images[is_target]
    target_idx = np.random.choice(candidates.shape[0])
    target = candidates[target_idx]

    target = target.reshape(28, 28).astype(np.uint8)
    target = np.expand_dims(target, axis=-1)
    target = np.repeat(target, 2, axis=-1)
    target[..., 1] = (target[..., 0] > 0).astype(np.uint8) * 255

    target_img = np2pil(target)
    target_img = target_img.resize((cfg.size, cfg.size))
    target_img = target_img.convert("LA")
    # target_img = target_img.convert("RGBA")

    target_img = np.array(target_img)
    target_img = target_img.astype("float32") / 255.0

    return target_img
