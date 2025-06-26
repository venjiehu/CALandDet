import os
import time
import albumentations as A  # noqa
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from typing import Union
from collections import OrderedDict
from accelerate import Accelerator
#
from trainer.trainer import Trainer
from nets.utils.load_cigs import load_cigs
from nets.utils.lr_scheduler import get_param_groups, LearningRateScheduler


def freeze_layers(model, freeze_fraction=0.9):
    # 获取所有可训练参数的总数
    total_params = sum(1 for _ in model.parameters())
    # 计算需要冻结的参数数量
    num_freeze = int(total_params * freeze_fraction)

    # 冻结指定数量的参数
    frozen_count = 0
    for param in model.parameters():
        param.requires_grad = False
        frozen_count += 1
        if frozen_count >= num_freeze:
            break

    return model


def main(net: nn.Module,
         train_datasets: Union[str, Path], val_datasets: Union[str, Path],
         transforms: dict,
         net_name: str,
         metrics_func_dict: OrderedDict,
         loss_func: nn.Module,
         custom_datasets,
         monitor,
         train_times,
         #
         img_size,
         batch_size,
         num_workers,
         lr,
         epochs,
         #
         optimizer_method,
         layer_decay_dict,
         weight_decay_dict,
         warmup_epochs,
         start_weights,
         ):
    # 初始化 Accelerator()
    accelerator = Accelerator()

    # 获取工作路径并打印，获取开始时间用于文件统一命名
    root_path = os.getcwd()
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    accelerator.print(f"ROOT PATH: {root_path}\nstart time: {start_time}")

    # 生成logs、ckpt、tbs路径
    save_dir = os.path.join(net_name, net_name + "_" + train_times)
    save_name = net_name + "_" + train_times + "_" + start_time
    logs_path = Path(os.path.join(root_path, "save", "logs", save_dir, save_name + "_log.csv"))
    best_path = Path(os.path.join(root_path, "save", "best", save_dir, save_name + "_best.pt"))
    tbs_path = Path(os.path.join(root_path, "save", "tbs", save_dir, save_name))

    # 创建dataloader
    ds_train = custom_datasets(Path(train_datasets), img_size=img_size, random=True, img_aug=transforms["train"])
    ds_val = custom_datasets(Path(val_datasets), img_size=img_size, random=False, img_aug=transforms["val"])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True,
                          pin_memory=True, num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=num_workers)

    # 设置优化器参数
    lr = lr * batch_size / 256.
    param_groups = get_param_groups(net=net,
                                    optimizer_method=optimizer_method,
                                    layer_decay_dict=layer_decay_dict,
                                    weight_decay_dict=weight_decay_dict,
                                    )

    optimizer = AdamW(param_groups, lr=lr)
    # 设置学习率调度器
    lr_scheduler = LearningRateScheduler(optimizer=optimizer,
                                         lr=lr,
                                         epochs=epochs,
                                         warmup_epochs=warmup_epochs,
                                         )
    # 创建训练模型
    trainer = Trainer(net=net,
                      loss_fn=loss_func,
                      accelerator=accelerator,
                      metrics_dicts=metrics_func_dict,
                      load_weights=start_weights,
                      tbs_dir=tbs_path,
                      best_file=best_path,
                      log_file=logs_path
                      )

    trainer.fit(epochs=epochs,
                optimizer=optimizer,
                train_dataloader=dl_train,
                val_dataloader=dl_val,
                lr_scheduler=lr_scheduler,
                monitor=monitor,
                )

    accelerator.print("\n" + "==========" * 8)


if __name__ == '__main__':
    from functools import partial
    from collections import OrderedDict
    # 导入模型
    from nets.calanddet import calanddet_model
    # 导入loss fn
    from nets.utils.loss import ClsSegmentationLossWithOHEM
    # 导入 metrics fn
    from nets.utils.metrics import IOU, F1Score
    # 导入 Dataset
    from datasets.dataset import TrainValDataset

    # 数据增强
    transform = {
        "train": A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                ToTensorV2(p=1),
            ],
            p=1.0),

        "val": None
    }

    # 训练的配置文件，包括训练参数，数据集路径以及模型相关参数
    cigs_file = "./configs/train_cigs.yaml"
    load_cigs = load_cigs(cigs_file)

    # 加载模型
    load_model = calanddet_model(name=load_cigs["model"]["name"], model_cigs=load_cigs["model"]["cigs_file"])

    # 定义损失函数
    loss_fn = ClsSegmentationLossWithOHEM(bce_ratio=0.5, keep_ratio=0.7)

    # 选择数据集和评估函数的模式
    mode = "cls"

    # 定义精度评价函数
    metrics_dict = OrderedDict({
        'iou': IOU(input_mode=mode),
        'f1_score': F1Score(input_mode=mode),
    })

    # 定义数据集
    datasets = partial(TrainValDataset, mode=mode)

    # 定义监控指标
    monitor_value = "eval_iou"

    main(net=load_model,
         transforms=transform,
         net_name=load_cigs["model"]["name"],
         metrics_func_dict=metrics_dict,
         loss_func=loss_fn,
         custom_datasets=datasets,
         monitor=monitor_value,
         train_times=load_cigs["model"]["train_times"],
         start_weights=load_cigs["model"]["start_weights"],
         **load_cigs["train"]
         )
