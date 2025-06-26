from torch import nn
from pathlib import Path
from typing import Union
#
from nets.modules.backbone.swin_backbone import SwinTransformerBackBone
from nets.modules.neck.swin_neck_blocks import ZiHaoNeckBase
from nets.modules.head.segmentation_head import SegmentationHead
from nets.modules.head.sw_cls_head import SwinClsHead
from nets.utils.load_cigs import load_cigs


class CAlandDetModel(nn.Module):
    def __init__(self,
                 backbone_cigs: dict,
                 neck_cigs: dict,
                 head_cigs: dict,
                 ):
        super().__init__()
        self.backbone = SwinTransformerBackBone(**backbone_cigs)
        self.cls_head = SwinClsHead(input_dims=768)
        self.neck = ZiHaoNeckBase(**neck_cigs)
        self.head = SegmentationHead(**head_cigs)

    def forward(self, x):
        backbone_out = self.backbone(x)
        cls = self.cls_head(backbone_out[-1])
        neck_out = self.neck(backbone_out, cls)
        head_out = self.head(neck_out)

        return head_out, cls[-1]


def calanddet_model(name: str, model_cigs: Union[str, Path]):
    model_name = name
    print(f"model name: {model_name}")
    cigs = load_cigs(model_cigs)

    model = CAlandDetModel(backbone_cigs=cigs["backbone"],
                           neck_cigs=cigs["neck"],
                           head_cigs=cigs["head"]
                           )
    return model


if __name__ == '__main__':
    from nets.utils.compute_params import compute_params

    cig = r"   your root path   /configs/net_cigs/swin_segmentation_model.yaml"
    model = calanddet_model(name="calanddet", model_cigs=cig)

    compute_params(model)
