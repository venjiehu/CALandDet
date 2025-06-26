import yaml
from typing import Union
from pathlib import Path


def load_cigs(path: Union[str, Path]) -> dict:
    with open(path, 'r', encoding='utf-8') as fr:
        data = yaml.load(fr, yaml.FullLoader)

    return data
