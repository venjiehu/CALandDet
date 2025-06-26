import numpy as np
from typing import Union


def tuple_n(inputs: Union[float, list, np.array, tuple],
            n: int):
    if isinstance(inputs, float) or isinstance(inputs, np.float32) or isinstance(inputs, int):
        return [inputs, ] * n
    elif isinstance(inputs, tuple) or isinstance(inputs, list):
        if n == len(inputs):
            return inputs
        raise ValueError("tuple length err!!")
    else:
        raise ValueError("inputs parameter only accept tuple or list or int or float or np.float32!!")
