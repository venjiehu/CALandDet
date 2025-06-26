from torch import nn
from einops import rearrange
#
from nets.modules.utils import NonLinearLayer


class SwinClsHead(nn.Module):
    def __init__(self,
                 input_dims: int = 768):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.nll = NonLinearLayer(input_dims, input_dims)
        self.head = nn.Linear(input_dims, 1)

    def forward(self, x):
        # x (b,768)
        # token (b,768)
        # p (b,1)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.avg_pool(x)
        x = self.flatten(x)
        token = self.nll(x)
        p = self.head(token)

        return token, p


if __name__ == '__main__':
    import torch

    a = torch.randn(1, 7, 7, 768)

    model = SwinClsHead()

    b = model(a)

    for i in range(len(b)):
        print(f"{i}___{b[i].shape}")
