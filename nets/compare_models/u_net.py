# --------------------------------------------------------
# References: https://arxiv.org/abs/1505.04597
# Following: https://nn.labml.ai/unet/index.html
# --------------------------------------------------------

import torch
import torchvision.transforms.functional as TF  # noqa
from torch import nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.c2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.c1(x)
        x = self.act1(x)
        x = self.c2(x)
        return self.act2(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.down(x)


class Concat(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor, x_concat: torch.Tensor):
        contracting_x = TF.center_crop(x_concat, x.shape[-2:])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.ModuleList(
            [DoubleConv2d(i, c1, c2) for i, c1, c2 in
             [(in_channels, 64, 64), (64, 128, 128), (128, 256, 256), (256, 512, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        self.mid_conv = DoubleConv2d(512, 1024, 1024)

        self.concat = nn.ModuleList([Concat() for _ in range(4)])
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList(
            [DoubleConv2d(i, c1, c2) for i, c1, c2 in
             [(1024, 512, 512), (512, 256, 256), (256, 128, 128), (128, 64, 64)]])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        down_path = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            down_path.append(x)
            x = self.down_sample[i](x)

        x = self.mid_conv(x)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, down_path.pop())
            x = self.up_conv[i](x)

        return self.final_conv(x)


def unet(in_channels: int = 3, out_channels: int = 1):
    model = UNet(in_channels, out_channels)
    return model


if __name__ == '__main__':
    from nets.utils.compute_params import compute_params

    model = UNet(3, 1)

    compute_params(model)
