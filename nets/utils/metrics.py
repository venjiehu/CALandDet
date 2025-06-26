import torch
from torch import nn


class BaseMetrics(nn.Module):
    def __init__(self, threshold: float = 0.5, input_mode: str = "normal"):
        super().__init__()
        self.threshold = threshold
        self.input_mode = input_mode

    def _threshold_fn(self, inputs):
        """
            将小于指定阈值的设置为0，大于阈值的设置为1
        """
        tensor = torch.clone(inputs)
        tensor[tensor > self.threshold] = 1
        tensor[tensor <= self.threshold] = 0
        return tensor

    def _prepare_inputs(self, preds, targets):
        if self.input_mode != "normal":
            preds = preds[0]
            targets = targets[0]
        preds = nn.functional.sigmoid(preds)
        preds = self._threshold_fn(preds)

        return preds, targets


class IOU(BaseMetrics):
    """
         IoU for foreground class
         """

    def __init__(self, threshold: float = 0.5, input_mode: str = "normal"):
        super().__init__(threshold=threshold, input_mode=input_mode)

    @staticmethod
    def _iou_binary(preds, labels, empy=1., ignore=None, per_image=True):
        """
        IoU for foreground class
        binary: 1 foreground, 0 background
        following: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py#L33
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            intersection = ((label == 1) & (pred == 1)).sum()
            union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
            if not union:
                iou = empy
            else:
                iou = float(intersection) / float(union)
            ious.append(iou)
        iou = sum(ious) / len(ious)
        return iou

    def forward(self, preds, targets):
        with torch.no_grad():
            preds, targets = self._prepare_inputs(preds, targets)
            iou = self._iou_binary(preds, targets)

        return iou


class F1Score(BaseMetrics):
    def __init__(self, threshold: float = 0.5, input_mode: str = "normal"):
        super().__init__(threshold=threshold, input_mode=input_mode)

    @staticmethod
    def _chaos_matrix(predicted_masks, true_masks):
        true_positives = torch.sum(predicted_masks * true_masks)
        false_positives = torch.sum(predicted_masks) - true_positives
        false_negatives = torch.sum(true_masks) - true_positives

        return true_positives, false_positives, false_negatives

    def _precision(self, predicted_masks, true_masks, esp=1e-8):
        true_positives, false_positives, false_negatives = self._chaos_matrix(predicted_masks, true_masks)
        precision = float(true_positives) / float(true_positives + false_positives + esp)

        return precision

    def _recall(self, predicted_masks, true_masks, esp=1e-8):
        true_positives, false_positives, false_negatives = self._chaos_matrix(predicted_masks, true_masks)
        recall = float(true_positives) / float(true_positives + false_negatives + esp)

        return recall

    def _f1_score_fn(self, predicted_masks, true_masks, esp=1e-8):
        # esp: 极小量，防止分母为0
        # 计算每个样本的TP、FP和FN

        # 计算精确率和召回率
        precision = self._precision(predicted_masks, true_masks, esp=esp)
        recall = self._recall(predicted_masks, true_masks, esp=esp)

        # 计算f1分数
        f1_score = 2 * (precision * recall) / (precision + recall + esp)

        return f1_score

    def forward(self, preds, targets):
        with torch.no_grad():
            preds, targets = self._prepare_inputs(preds, targets)
            f1_score = self._f1_score_fn(preds, targets)

        return f1_score


class Precision(F1Score):
    def forward(self, preds, targets):
        with torch.no_grad():
            preds, targets = self._prepare_inputs(preds, targets)
            precision = self._precision(preds, targets)

        return precision


class Recall(F1Score):
    def forward(self, preds, targets):
        with torch.no_grad():
            preds, targets = self._prepare_inputs(preds, targets)
            recall = self._recall(preds, targets)

        return recall


class OverallAccuracy(BaseMetrics):
    @staticmethod
    def _overall_accuracy(predicted_masks, true_masks, esp=1e-8):
        total = torch.numel(true_masks)
        correct_pixels = (predicted_masks == true_masks).sum()
        overall_accuracy = correct_pixels / (total + esp)

        return overall_accuracy

    def forward(self, preds, targets):
        with torch.no_grad():
            preds, targets = self._prepare_inputs(preds, targets)
            overall_accuracy = self._overall_accuracy(preds, targets)

        return overall_accuracy
