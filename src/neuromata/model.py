import json
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from keras.api.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.python.framework import convert_to_constants


@dataclass
class CAConfig:
    seed: int = 42
    channel_n: int = 16
    color_channel_n: int = 3
    cell_fire_rate: float = 0.5
    initialize: str = "center-point"


class CAModel(keras.Model):

    def __init__(self, cfg: CAConfig):
        super().__init__()
        self.cfg = cfg

        self.rng = np.random.default_rng(cfg.seed)

        self.dmodel = keras.Sequential(
            [
                Conv2D(128, 1, activation=tf.nn.relu),
                Conv2D(
                    cfg.channel_n,
                    1,
                    activation=None,
                    kernel_initializer=tf.zeros_initializer,
                ),
            ]
        )

        self(tf.zeros([1, 3, 3, cfg.channel_n]))  # dummy call to build the model

    def build_seed(self, x: np.ndarray):

        h, w, _ = x.shape
        cent_y, cent_x = h // 2, w // 2
        x0 = np.zeros([h, w, self.cfg.channel_n], np.float32)
        if self.cfg.initialize == "center-point":
            x0[cent_y, cent_x, self.cfg.color_channel_n :] = 1.0
        elif self.cfg.initialize == "inside-point":
            y_candidates, x_candidates = np.where(x[:, :, self.cfg.color_channel_n] > 0)
            idx = self.rng.choice(len(x_candidates))
            x0[y_candidates[idx], x_candidates[idx], self.cfg.color_channel_n :] = 1.0
        elif self.cfg.initialize == "circle":
            y_indices, x_indices = np.ogrid[:h, :w]
            radius = min(h, w) // 2
            mask = ((x_indices - cent_x) ** 2 + (y_indices - cent_y) ** 2) <= radius**2
            x0[mask, self.cfg.color_channel_n :] = 1.0
        else:
            raise ValueError(f"Unknown initialize method: {self.cfg.initialize}")

        return x0

    @tf.function
    def get_living_mask(self, x):

        alpha = x[:, :, :, self.cfg.color_channel_n : self.cfg.color_channel_n + 1]

        return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], "SAME") > 0.1

    @tf.function
    def loss_f(self, x, pad_target):

        x = x[..., : pad_target.shape[-1]]

        return tf.reduce_mean(tf.square(x - pad_target), [-2, -3, -1])

    @tf.function
    def perceive(self, x, angle=0.0):
        identify = np.float32([0, 1, 0])
        identify = np.outer(identify, identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c, s = tf.cos(angle), tf.sin(angle)
        kernel = tf.stack([identify, c * dx - s * dy, s * dx + c * dy], -1)[
            :, :, None, :
        ]
        kernel = tf.repeat(kernel, self.cfg.channel_n, 2)
        y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME")
        return y

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = self.get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.cfg.cell_fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)

        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)


class AutoencodeCAModel(keras.Model):

    def __init__(self, cfg: CAConfig):
        super().__init__()

        self.cfg = cfg
        self.emodel = keras.Sequential(
            [
                Conv2D(128, 1, activation=tf.nn.relu),
                Conv2D(
                    cfg.channel_n - cfg.color_channel_n - 1, 1, activation=tf.nn.relu
                ),
                GlobalAveragePooling2D(),
            ]
        )
        self.dmodel = CAModel(
            channel_n=cfg.channel_n,
            fire_rate=cfg.cell_fire_rate,
        )

    def call(self, x: tf.Tensor, iter_n: int):

        seed_cell = self.emodel(x)
        x0 = tf.zeros_like(x)
        b, h, w, c = x.shape
        x0[:, h // 2, w // 2, self.cfg.color_channel_n] = 1.0
        x0[:, h // 2, w // 2, self.cfg.color_channel_n :] = seed_cell

        x = x0
        for i in tf.range(iter_n):
            x = self.dmodel(x)

        return x


def export_model(ca: CAModel, base_fn):
    ca.save_weights(base_fn)

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, ca.channel_n]),
        fire_rate=tf.constant(0.5),
        angle=tf.constant(0.0),
        step_size=tf.constant(1.0),
    )
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json["versions"] = dict(producer="1.14", minConsumer="1.14")
    model_json = {
        "format": "graph-model",
        "modelTopology": graph_json,
        "weightsManifest": [],
    }
    with open(base_fn + ".json", "w") as f:
        json.dump(model_json, f)
