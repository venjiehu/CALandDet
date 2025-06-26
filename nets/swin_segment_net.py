from pathlib import Path
from typing import Union
from torch import nn
#
from nets.modules.backbone.swin_backbone import SwinTransformerBackBone
from nets.modules.neck.swin_neck_blocks import NeckBase
from nets.modules.head.segmentation_head import SegmentationHead
from nets.utils.load_cigs import load_cigs
from nets.modules.head.sw_cls_head import SwinClsHead


class SwinSegmentationModel(nn.Module):
    def __init__(self,
                 backbone_cigs: dict,
                 neck_cigs: dict,
                 head_cigs: dict,
                 ):
        super().__init__()
        self.backbone = SwinTransformerBackBone(**backbone_cigs)
        self.neck = NeckBase(**neck_cigs)
        self.head = SegmentationHead(**head_cigs)

    def forward(self, x):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        head_out = self.head(neck_out)

        return head_out


class SwinSegClsHeadModel(SwinSegmentationModel):
    def __init__(self,
                 backbone_cigs: dict,
                 neck_cigs: dict,
                 head_cigs: dict,
                 ):
        super().__init__(backbone_cigs, neck_cigs, head_cigs)
        self.cls_head = SwinClsHead(input_dims=768)

    def forward(self, x):
        backbone_out = self.backbone(x)
        cls = self.cls_head(backbone_out[-1])
        neck_out = self.neck(backbone_out)
        head_out = self.head(neck_out)

        return head_out, cls[-1]


def swin_segmentation_model(name: str, model_cigs: Union[str, Path]):
    model_name = name
    print(f"model name: {model_name}")
    cigs = load_cigs(model_cigs)

    model = SwinSegmentationModel(backbone_cigs=cigs["backbone"],
                                  neck_cigs=cigs["neck"],
                                  head_cigs=cigs["head"]
                                  )
    return model


def swin_seg_cls_model(name: str, model_cigs: Union[str, Path]):
    model_name = name
    print(f"model name: {model_name}")
    cigs = load_cigs(model_cigs)

    model = SwinSegClsHeadModel(backbone_cigs=cigs["backbone"],
                                neck_cigs=cigs["neck"],
                                head_cigs=cigs["head"]
                                )
    return model
