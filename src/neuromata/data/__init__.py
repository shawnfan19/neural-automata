from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str = "mnist"
    dataset_path: str = "data/mnist"
    target: str = "🦎"
    size: int = 40
    pad: int = 16
