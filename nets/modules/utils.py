import torch
from torch import nn
from einops import rearrange


class NonLinearLayer(nn.Module):
    def __init__(self,
                 in_dims: int = 192,
                 out_dims: int = 192,
                 drop_ratio: float = 0.,
                 bias: bool = False,
                 ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dims, eps=1e-6)
        self.fc = nn.Linear(in_dims, out_dims, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_ratio) if drop_ratio > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc(self.norm(x))
        x = self.act(x)
        x = self.drop(x)

        return x


class SkipCatBlock(nn.Module):
    def __init__(self,
                 in_dims=1152,  # in_dims = x dims + x_skip_dims
                 out_dims=192,  # output dims
                 drop_ratio: float = 0.,  # float or (patch reshape drop, proj drop)
                 ):
        super().__init__()
        pro_dims = int(out_dims * 4)
        self.nll = NonLinearLayer(in_dims=in_dims,
                                  out_dims=pro_dims,
                                  drop_ratio=drop_ratio,
                                  bias=False)

    def forward(self, x, x_skip):
        #
        x = torch.cat([x, x_skip], dim=-1)
        x = self.nll(x)
        x = rearrange(x, "b h w (p q c) -> b (h p) (w q) c", p=2, q=2)

        return x


if __name__ == '__main__':
    a = torch.randn((4, 7, 7, 768))
    b = torch.randn((4, 7, 7, 384))
    model = SkipCatBlock()
    out = model(b, a)
    print(out.shape)
