import torch
from torch import nn


class DiceBinaryLoss(nn.Module):
    """
        Binary DiceLoss
    """

    def __init__(self, delta=1e-5):  # Prevent the denominator from being 0
        super().__init__()
        self.delta = delta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # binary
        preds = inputs.flatten(1).sigmoid()
        targets = targets.flatten(1).float()
        loss = self.compute_loss(preds, targets)
        return loss

    def compute_loss(self, preds, targets):
        a = torch.sum(preds * targets, 1)  # |X⋂Y| (batch,1)
        b = torch.sum(preds * preds, 1)  # |X|
        c = torch.sum(targets * targets, 1)  # |Y|
        dice = (2 * a) / (b + c + self.delta)
        loss = 1 - dice
        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, bce_ratio: float = 0.5, reduction: bool = False):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceBinaryLoss()
        self.bce_ratio = bce_ratio
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        bce_loss = torch.mean(self.bce(inputs, targets.float()).flatten(1), 1)
        dice_loss = self.dice(inputs, targets)
        total_loss = bce_loss * self.bce_ratio + dice_loss * (1 - self.bce_ratio)
        total_loss = torch.mean(total_loss) if self.reduction else total_loss

        return total_loss


class AuxHeadDiceBCELoss(nn.Module):
    def __init__(self, bce_ratio: float = 0.5):
        super().__init__()
        self.head1 = DiceBCELoss(bce_ratio)
        self.head2 = DiceBCELoss(bce_ratio)
        self.head3 = DiceBCELoss(bce_ratio)

    def forward(self, inputs: list, targets: torch.Tensor):
        loss1 = self.head1(inputs[0], targets)
        loss2 = self.head1(inputs[1], targets)
        loss3 = self.head1(inputs[2], targets)

        loss = torch.mean(loss1 + loss2 + loss3)

        return loss


class OHEM(nn.Module):
    def __init__(self, keep_ratio: float = 0.7, loss_fn=None):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.loss_fn = loss_fn

    def forward(self, inputs, targets):
        loss = self.loss_fn(inputs, targets).flatten()

        # Compute OHEM loss
        keep_index = int(self.keep_ratio * len(loss))
        sorted_loss, _ = loss.sort(descending=True)
        keep_loss = sorted_loss[:keep_index]
        ohem_loss = keep_loss.mean()

        return ohem_loss


class DiceBCELossWithOHEM(nn.Module):
    def __init__(self,
                 keep_ratio: float = 0.7,
                 bce_ratio: float = 0.5
                 ):
        super().__init__()
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        dice_loss = DiceBinaryLoss()
        self.bce_ohem = OHEM(keep_ratio=keep_ratio, loss_fn=bce_loss)
        self.dice_ohem = OHEM(keep_ratio=keep_ratio, loss_fn=dice_loss)
        self.bce_ratio = bce_ratio

    def forward(self, inputs, targets):
        bce_ohem_loss = self.bce_ohem(inputs, targets)
        dice_ohem_loss = self.dice_ohem(inputs, targets)

        loss = self.bce_ratio * bce_ohem_loss + (1 - self.bce_ratio) * dice_ohem_loss

        return loss


class ClsSegmentationLossWithOHEM(nn.Module):
    def __init__(self,
                 keep_ratio: float = 0.7,
                 bce_ratio: float = 0.5):
        super().__init__()
        self.seg_loss = DiceBCELossWithOHEM(keep_ratio=keep_ratio, bce_ratio=bce_ratio)
        cls_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.cls_loss = OHEM(keep_ratio=keep_ratio, loss_fn=cls_fn)

    def forward(self, inputs, targets):
        head_out = inputs[0]
        cls_out = inputs[1]
        #
        head_targets = targets[0]
        cls_targets = targets[1].unsqueeze(1).float()
        #
        head_loss = self.seg_loss(head_out, head_targets)
        cls_loss = self.cls_loss(cls_out, cls_targets)

        total_loss = torch.sum(head_loss) + torch.sum(cls_loss)

        return total_loss
