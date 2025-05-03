import base64
import io

import numpy as np
import PIL.Image
import requests


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


def to_grayscale(x: np.ndarray) -> np.ndarray:

    x = np.clip(x, 0.0, 1.0) * 255.0
    x = x.astype(np.uint8)

    return x


def to_pil(x: np.ndarray, mode: str = "L") -> PIL.Image.Image:

    return PIL.Image.fromarray(x, mode=mode)


def to_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return np.transpose(x, (2, 0, 1))
    elif x.ndim == 4:
        return np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unknown image shape: {x.shape}")


def to_rgb(x: np.ndarray, cdims: int) -> np.ndarray:

    if cdims == 3:
        rgb = x[..., :3]
        a = np.clip(x[..., 3:4], 0.0, 1.0)
        # assume rgb premultiplied by alpha
        return 1.0 - a + rgb
    elif cdims == 1:
        rgb = np.repeat(x[..., :1], 3, -1)
        a = np.clip(x[..., 1:2], 0.0, 1.0)
        return rgb * a
    else:
        raise ValueError(f"Unknown color channel dimension: {cdims}")
