import numpy as np

from neuromata.data import DataConfig
from neuromata.utils.image import load_image


def load_emoji(cfg: DataConfig) -> np.ndarray:
    code = hex(ord(cfg.target))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    return load_image(url, cfg.size)
