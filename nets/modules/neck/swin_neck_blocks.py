import numpy as np
from torch import nn
from typing import List

#
from nets.modules.neck._neck_stage import SwinBlocks, NeckBaseStage, NeckCroAttnStage
from nets.modules.utils import NonLinearLayer


class NeckBase(nn.Module):
    def __init__(self,
                 in_dims: List[int],
                 out_dims: List[int],
                 num_heads: List[int],
                 depths: List[int],
                 window_size: List[int],
                 #
                 dropout: float = 0.,
                 attention_dropout: float = 0.,
                 sd_prob: float = 0.1,
                 ):
        super().__init__()
        dropout_list = np.linspace(0, dropout, len(depths))
        attention_dropout_list = np.linspace(0, attention_dropout, len(depths))
        sd_prob_list = np.linspace(0, sd_prob, len(depths))
        self.layer1 = nn.Sequential(
            NonLinearLayer(in_dims=in_dims[0], out_dims=out_dims[0], drop_ratio=dropout),
            SwinBlocks(dim=out_dims[0], num_heads=num_heads[0], depth=depths[0], window_size=window_size,
                       dropout=float(dropout_list[0]), attention_dropout=float(attention_dropout_list[0]),
                       sd_prob=float(sd_prob_list[0]))
        )

        self.layers = nn.ModuleList([
            NeckBaseStage(in_dims=in_dims[i], out_dim=out_dims[i], num_heads=num_heads[i],
                          depth=depths[i], window_size=window_size,
                          dropout=float(dropout_list[i]), attention_dropout=float(attention_dropout_list[i]),
                          sd_prob=float(sd_prob_list[i])
                          )
            for i in range(1, len(depths))])

    def forward(self, skip: list):
        net_out = []
        x = self.layer1(skip.pop())
        for i in range(len(self.layers)):
            x = self.layers[i](x, skip.pop())
            net_out.append(x)

        return net_out


class ZiHaoNeckBase(nn.Module):
    def __init__(self,
                 in_dims: List[int],
                 out_dims: List[int],
                 num_heads: List[int],
                 depths: List[int],
                 window_size: List[int],
                 #
                 token_dims: int = 768,
                 dropout: float = 0.,
                 attention_dropout: float = 0.,
                 sd_prob: float = 0.1,
                 ):
        super().__init__()
        dropout_list = np.linspace(0, dropout, len(depths))
        attention_dropout_list = np.linspace(0, attention_dropout, len(depths))
        sd_prob_list = np.linspace(0, sd_prob, len(depths))
        self.layer1 = nn.Sequential(
            NonLinearLayer(in_dims=in_dims[0], out_dims=out_dims[0], drop_ratio=dropout),
            SwinBlocks(dim=out_dims[0], num_heads=num_heads[0], depth=depths[0], window_size=window_size,
                       dropout=float(dropout_list[0]), attention_dropout=float(attention_dropout_list[0]),
                       sd_prob=float(sd_prob_list[0]))
        )

        self.layers = nn.ModuleList([
            NeckCroAttnStage(in_dims=in_dims[i], out_dim=out_dims[i], token_dims=token_dims, num_heads=num_heads[i],
                             depth=depths[i], window_size=window_size,
                             dropout=float(dropout_list[i]), attention_dropout=float(attention_dropout_list[i]),
                             sd_prob=float(sd_prob_list[i])
                             )
            for i in range(1, len(depths))])

    def forward(self, skip: list, cls):
        #   a = (b, 28, 28, 192)
        #   b = (b, 14, 14, 384)
        #   c = (b, 7, 7, 768)
        #   d = (b, 7, 7, 768)
        #   skip = [a, b, c, d]
        #
        #   token = (b, 768)
        #   p = (, 1)
        #   cls = [token, p]

        net_out = []
        x = self.layer1(skip.pop())
        for i in range(len(self.layers)):
            x = self.layers[i](x, skip.pop(), cls)
            net_out.append(x)

        # net_out
        # 0__torch.Size([b, 14, 14, 192])
        # 1__torch.Size([b, 28, 28, 96])
        # 2__torch.Size([b, 56, 56, 48])

        return net_out


if __name__ == '__main__':
    import torch

    a = torch.randn(1, 28, 28, 192)
    b = torch.randn(1, 14, 14, 384)
    c = torch.randn(1, 7, 7, 768)
    d = torch.randn(1, 7, 7, 768)
    cat = [a, b, c, d]

    token = torch.randn(1, 768)
    p = torch.randn(1, 1)
    cls = (token, p)

    model = ZiHaoNeckBase(in_dims=[768, 1152, 576, 288],
                          out_dims=[384, 192, 96, 48],
                          num_heads=[24, 12, 6, 3],
                          depths=[2, 2, 6, 2],
                          window_size=[7, 7]
                          )

    out = model(cat, cls)

    for i in range(len(out[1])):
        print(f"{i}__{out[0][i].shape}")
