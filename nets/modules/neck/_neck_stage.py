from torch import nn
from functools import partial
from typing import List
from torchvision.models.swin_transformer import SwinTransformerBlock  # noqa
#
from nets.modules.utils import SkipCatBlock
from nets.modules.neck._cross_attn import CrossAttn


class SwinBlocks(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 depth: int,
                 window_size: List[int],
                 dropout: float = 0.,
                 attention_dropout: float = 0.,
                 sd_prob: float = 0.1,
                 #
                 mlp_ratio: float = 4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5)

                 ):
        super().__init__()
        self.swin_blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim,
                num_heads,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth_prob=sd_prob,
                norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        x = self.swin_blocks(x)

        return x


class NeckBaseStage(nn.Module):

    def __init__(self,
                 in_dims: int,
                 out_dim: int,
                 num_heads: int,
                 depth: int,
                 window_size: List[int],
                 dropout: float = 0.,
                 attention_dropout: float = 0.,
                 sd_prob: float = 0.1,
                 ):
        super().__init__()

        self.skip_block = SkipCatBlock(in_dims=in_dims, out_dims=out_dim, drop_ratio=dropout)
        self.swin_blocks = SwinBlocks(dim=out_dim, num_heads=num_heads, depth=depth, window_size=window_size,
                                      dropout=dropout, attention_dropout=attention_dropout, sd_prob=sd_prob)

    def forward(self, x, x_skip):
        x = self.skip_block(x, x_skip)
        x = self.swin_blocks(x)

        return x


class NeckCroAttnStage(nn.Module):
    """
        input: x, x_skip, cls(token, p) ==> output;
        x: (B, h, w, c);
        x_skip: (B, n, d);
        token: (b, d);
        p: (b,1);
    """

    def __init__(self,
                 in_dims: int,
                 out_dim: int,
                 token_dims: int,
                 num_heads: int,
                 depth: int,
                 window_size: List[int],
                 dropout: float = 0.,
                 attention_dropout: float = 0.,
                 sd_prob: float = 0.1,
                 ):
        super().__init__()

        self.skip_block = SkipCatBlock(in_dims=in_dims, out_dims=out_dim, drop_ratio=dropout)
        self.cro_attn = CrossAttn(patch_inputs_dims=out_dim, drop_ratio=dropout, sd_prob=sd_prob,
                                  token_input_dims=token_dims)

        self.swin_blocks = SwinBlocks(dim=out_dim, num_heads=num_heads, depth=depth, window_size=window_size,
                                      dropout=dropout, attention_dropout=attention_dropout, sd_prob=sd_prob)

    def forward(self, x, x_skip, cls):
        x = self.skip_block(x, x_skip)
        x = self.cro_attn(x, cls)
        x = self.swin_blocks(x)

        return x


if __name__ == '__main__':
    import torch

    x = torch.randn((4, 7, 7, 384))
    x_skip = torch.randn((4, 7, 7, 768))
    p = torch.randn((4, 1))
    token = torch.randn((4, 768))

    model = NeckCroAttnStage(token_dims=768, in_dims=1152, out_dim=192, num_heads=24, depth=2, window_size=[7, 7])
    out = model(x, x_skip, (token, p))
    print(out[0].shape)
    print(out[1].shape)
