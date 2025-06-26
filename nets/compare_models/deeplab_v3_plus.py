# --------------------------------------------------------
# References:
# Following: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/doc/deeplab_xception.py
# --------------------------------------------------------

from nets.compare_models.deeplab_v3_plus_utils.deeplab_v3_plus import DeepLabv3_plus


def deeplab_v3_plus():
    return DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=False, _print=False)


if __name__ == "__main__":
    from nets.utils.compute_params import compute_params

    model = deeplab_v3_plus()

    compute_params(model)
