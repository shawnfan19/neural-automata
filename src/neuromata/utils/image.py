import base64
import io

import numpy as np
import PIL.Image
import PIL.ImageDraw
import requests
import tensorflow as tf


def np2pil(a: np.ndarray):
    if a.dtype in [np.float32, np.float64]:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None) -> None:
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        f = open(f, "wb")
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt="jpeg"):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt="jpeg"):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode("ascii")
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), "constant")
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def load_image(url, max_size):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def load_emoji(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    return load_image(url, max_size)


def to_rgba(x):
    return x[..., :4]


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x: np.ndarray):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def generate_pool_figures(pool, step_i):
    tiled_pool = tile2d(to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[
        None, :, None
    ]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[
        None, ::-1, None
    ]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[
        :, None, None
    ]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[
        ::-1, None, None
    ]
    imwrite("train_log/%04d_pool.jpg" % step_i, tiled_pool)


def visualize_batch(x0, x, step_i):
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])
    return vis
