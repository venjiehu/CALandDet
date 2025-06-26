import torch
from tif2mask.seg_work import inference_tif2mask
from nets.calanddet import calanddet_model


class CAlandDetWrapping(torch.nn.Module):
    def __init__(self, cig, pretrain_weight: str = None):
        super().__init__()
        self.model = calanddet_model(name="calanddet", model_cigs=cig)
        if pretrain_weight:
            print("load pretrain weight: ", pretrain_weight)
            self.model.load_state_dict(torch.load(pretrain_weight), strict=True)

    def forward(self, x):
        return self.model(x)[0]


if __name__ == '__main__':
    cig = r"./configs/net_cigs/swin_segmentation_model.yaml"
    # 预训练权重
    pretrain_weight = r"model weight"

    # 加载模型，以及预训练模型
    seg_net = CAlandDetWrapping(cig=cig, pretrain_weight=pretrain_weight)

    img_path = r"your tif file path"
    work_path = r"your work path"

    """
     sample_size: 输入模型的样本尺寸
     patch_size: 切成的patch块的尺寸= sample_size*(1-overlap_rate)*(n-1) + sample_size, n为一行的样本数量，最后一个
                  patch块包含 n*n个样本数

    overlap_rate: 滑动窗口重叠率
    """
    inference_tif2mask(seg_net, work_path, img_path, sample_size=224, patch_size=2688, overlap_rate=0.5)
