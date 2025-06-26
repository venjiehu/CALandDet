from pathlib import Path
from trainer.trainer import Trainer
from accelerate import Accelerator
from collections import OrderedDict
from torch.utils.data import DataLoader
from datasets.dataset import TrainValDataset
from nets.utils.loss import ClsSegmentationLossWithOHEM
from nets.utils.metrics import IOU, F1Score, Precision, Recall, OverallAccuracy
#
from nets.calanddet import calanddet_model


def main():
    accelerator = Accelerator()

    # 定义损失函数
    loss_func = ClsSegmentationLossWithOHEM(bce_ratio=0.5, keep_ratio=0.7)
    # 定义评估函数
    mode = "cls"  # normal or cls

    # 定义精度评价函数
    metrics_dict = OrderedDict({
        'iou': IOU(input_mode=mode),
        'f1_score': F1Score(input_mode=mode),
        'precision': Precision(input_mode=mode),
        'recall': Recall(input_mode=mode),
        'oa': OverallAccuracy(input_mode=mode),
    })

    start_weights = r"your model weights"

    # create model
    net = calanddet_model(name="calanddet_model",
                       model_cigs=r".\configs\net_cigs\swin_segmentation_model.yaml")

    trainer = Trainer(net,
                      loss_fn=loss_func,
                      accelerator=accelerator,
                      metrics_dicts=metrics_dict,
                      load_weights=start_weights
                      )

    val_datasets = r"your validation datasets path"

    ds_val = TrainValDataset(Path(val_datasets), img_size=(224, 224), mode=mode, random=False, img_aug=None)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=4)

    eval_dict = trainer.evaluate(dl_val)

    for key, value in eval_dict.items():
        print(f"{key:15} ---> {value}")


if __name__ == '__main__':
    main()
