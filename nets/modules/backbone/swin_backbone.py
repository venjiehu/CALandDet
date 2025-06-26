from functools import partial
from torch import nn
from typing import List
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging  # noqa
from torchvision.ops.misc import Permute


class SwinTransformerBackBone(nn.Module):
    """
        using torchvision SwinTransformer
    """

    def __init__(
            self,
            patch_size: List[int],
            embed_dim: int,
            depths: List[int],
            num_heads: List[int],
            window_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.1,
    ):
        super().__init__()

        block = SwinTransformerBlock
        down_sample_layer = PatchMerging
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                stage.append(down_sample_layer(dim, norm_layer))
            layers.append(nn.Sequential(*stage))
        self.features = nn.ModuleList(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skip = []
        x = self.patch_embed(x)
        for i in range(len(self.features)):
            x = self.features[i](x)
            skip.append(x)

        return skip


if __name__ == '__main__':
    import torch

    a = torch.randn(1, 3, 224, 224)
    model = SwinTransformerBackBone(patch_size=[4, 4], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                    window_size=[7, 7])

    b = model(a)

    for i in range(len(b)):
        print(f"{i}__{b[i].shape}")
