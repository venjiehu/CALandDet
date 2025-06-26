import torch
from torch import nn
from typing import Union
from torchvision.ops.stochastic_depth import StochasticDepth
from torch.nn.functional import normalize
from functools import partial
#
from nets.utils.tuple_n import tuple_n
from nets.modules.utils import NonLinearLayer


class CrossAttn(nn.Module):
    """
    cross attention block;

    """

    def __init__(self,
                 patch_inputs_dims: int = 192,
                 drop_ratio: Union[float, list] = 0.,  # (token_proj drop, x_proj drop)
                 sd_prob: float = 0.1,
                 token_input_dims: int = 768,
                 attn_dims: int = 128,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)
                 ):
        super().__init__()

        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)  # att scale zoom parameter

        drop_ratio = tuple_n(drop_ratio, 2)
        self.norm1 = norm_layer(token_input_dims)
        self.token_proj = NonLinearLayer(in_dims=token_input_dims,
                                         out_dims=attn_dims,
                                         drop_ratio=drop_ratio[0],
                                         )

        self.norm2 = norm_layer(patch_inputs_dims)
        self.x_proj = NonLinearLayer(in_dims=patch_inputs_dims,
                                     out_dims=attn_dims,
                                     drop_ratio=drop_ratio[1],
                                     )

        self.norm3 = norm_layer(patch_inputs_dims)
        self.sd_prob = StochasticDepth(sd_prob, "row") if sd_prob > 0. else nn.Identity()

    def _att(self, x, token, p):
        # x: (b, n, d)
        # token: (b, d)
        # p: (b, 1)

        x = normalize(x, p=2, dim=2)  # x l2 norm
        token = normalize(token, p=2, dim=1).unsqueeze(-1)  # token l2 norm
        # p: (b, 1, 1); x: (b, n, d); token: (b, d, 1)
        # attn: (b, n, 1)
        attn = p.unsqueeze(-1) * (x @ token) * torch.exp(self.alpha)  # attn map
        return attn

    def forward(self, x: torch.Tensor, cls: tuple):
        b, h, w, d = x.shape
        # token: (b, d); p: (b, 1)
        token, p = cls

        token = self.token_proj(self.norm1(token))
        x_ = self.x_proj(self.norm2(x.reshape(b, -1, d)))
        attn = self._att(x_, token, p).reshape(b, h, w)

        x = 0.5 * x + self.sd_prob(self.norm3(x * attn.unsqueeze(-1)))

        return x


if __name__ == '__main__':
    p = torch.randn((4, 1))
    token = torch.randn((4, 768))
    x = torch.randn((4, 112, 112, 192))

    model = CrossAttn(patch_inputs_dims=192, token_input_dims=768)
    out = model(x, (token, p))
    print(out[0].shape)
    print(out[1].shape)
