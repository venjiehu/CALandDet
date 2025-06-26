import torch
from torch import nn
import numpy as np
from typing import Union
from einops import rearrange
import torch.nn.functional as F  # noqa
from functools import partial
#
from nets.utils.tuple_n import tuple_n
from nets.modules.utils import NonLinearLayer


class SegHeadBlock(nn.Module):
    def __init__(self,
                 dims: int = 192,
                 drop_ratio: Union[float, list] = 0.,  # float or (nll1_drop, nll2_drop, nll3_drop)
                 up_sample: bool = True
                 ):
        super().__init__()

        self.nll = NonLinearLayer(dims, 16, drop_ratio)
        self.up_sample = partial(F.interpolate, size=(224, 224), mode='bilinear',  # noqa
                                 align_corners=False) if up_sample else nn.Identity()

    def forward(self, x):
        x = self.nll(x)
        x = rearrange(x, "b h w (p q) -> b (h p) (w q)", p=4, q=4).unsqueeze(1)
        x = self.up_sample(x)
        return x


class SegmentationHead(nn.Module):
    """
    head need inputs tuple:
                             shape: (b, h, w, c)
                            input3: (b, 56, 56, 48)
                            input2: (b, 28, 28, 96)
                            input1: (b, 14, 14, 192)
                    outputs:
                            shape: (b, 1, h, w)
                            out: (b, 1, 224, 224)
    """

    def __init__(self,
                 in_dims: Union[tuple, list] = (192, 96, 48),
                 drop_ratio: Union[float, np.array, list] = 0.,  # float or (head1_drop, head2_drop, head3_drop)
                 ):
        super().__init__()

        drop_ratio = tuple_n(drop_ratio, 3)
        self.head_block1 = SegHeadBlock(dims=in_dims[0], drop_ratio=drop_ratio[0])  # 14*14*192
        self.head_block2 = SegHeadBlock(dims=in_dims[1], drop_ratio=drop_ratio[1])  # 28*28*96
        self.head_block3 = SegHeadBlock(dims=in_dims[2], drop_ratio=drop_ratio[2], up_sample=False)  # 56*56*48

        self.final = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, inputs: tuple):
        head1 = self.head_block1(inputs[0])
        head2 = self.head_block2(inputs[1])
        head3 = self.head_block3(inputs[2])

        x = torch.cat([head1, head2, head3], dim=1)
        x = self.final(x)

        return x


if __name__ == '__main__':
    a = torch.randn(1, 14, 14, 192)
    b = torch.randn(1, 28, 28, 96)
    c = torch.randn(1, 56, 56, 48)

    cat = [a, b, c]

    model = SegmentationHead()

    out = model(cat)

    print(out.shape)
