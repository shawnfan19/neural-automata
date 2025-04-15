import json

import keras
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from keras.api.layers import Conv2D
from tensorflow.python.framework import convert_to_constants


def get_living_mask(x):
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], "SAME") > 0.1


class CAModel(keras.Model):

    def __init__(self, channel_n, fire_rate):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.dmodel = keras.Sequential(
            [
                Conv2D(128, 1, activation=tf.nn.relu),
                Conv2D(
                    self.channel_n,
                    1,
                    activation=None,
                    kernel_initializer=tf.zeros_initializer,
                ),
            ]
        )

        self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model

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
        kernel = tf.repeat(kernel, self.channel_n, 2)
        y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME")
        return y

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)


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
