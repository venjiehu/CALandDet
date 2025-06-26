import torch
import thop
from thop import clever_format


def compute_params(model, input_size=(3, 224, 224)):
    model.eval()
    c, h, w = input_size
    image = torch.randn(1, c, h, w)
    flops, params = thop.profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], '%.3f')

    print('params：', params)
    print('flops：', flops)


if __name__ == '__main__':
    from nets.compare_models.u_net import unet

    model = unet()
    compute_params(model)
