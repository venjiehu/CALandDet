import torch
import pandas as pd
from torch import nn
from pathlib import Path
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from accelerate import Accelerator
from typing import Union, Optional
from colorama import Fore, Style
#
from trainer._train_epoch import train_epoch
from trainer._utils import check_dir, MetricsAccumulate, TbsLog, TotalTime


class Trainer:
    """
        用于单卡训练，基于Accelerator搭建
    """

    def __init__(self,
                 net, loss_fn,
                 accelerator: Accelerator,
                 metrics_dicts: Optional[OrderedDict[str, nn.Module]] = None,
                 load_weights: Union[str, Path] = None, strict: bool = True,
                 tbs_dir=None, ckp_file=None, best_file=None, log_file=None
                 ):
        self.tbs_dir = tbs_dir
        self.ckp_file = ckp_file
        self.best_file = best_file
        self.log_file = log_file
        self.accelerator = accelerator
        # 初始化精度评估函数。如果其存在
        metrics_names = list(metrics_dicts.keys()) if metrics_dicts else None
        metrics_fn = self.accelerator.prepare(*list(metrics_dicts.values())) if metrics_dicts else None
        metrics_fn = [metrics_fn] if not isinstance(metrics_fn, tuple) and not None else metrics_fn

        # 初始化累加器
        metrics_accumulate = MetricsAccumulate((len(metrics_fn))) if metrics_fn else None
        loss_accumulate = MetricsAccumulate(1)

        # 加载初始权重
        if load_weights:
            self.accelerator.print(Fore.RED, f"load weights, Strict: {strict}\n\t{load_weights}", Fore.RESET)
            msg = net.load_state_dict(torch.load(load_weights, map_location='cpu'), strict=strict)
            self.accelerator.print(Fore.RED, f"weights load msg:\n\t{msg}", Fore.RESET)

        # prepare fn
        self.net, loss_fn = self.accelerator.prepare(net, loss_fn)

        # 初始化训练函数以及评估函数
        self.train_epoch = partial(train_epoch, loss_fn=loss_fn, accelerator=self.accelerator,
                                   metrics_names=metrics_names, metrics_fn=metrics_fn,
                                   loss_accumulate=loss_accumulate, metrics_accumulate=metrics_accumulate)
        self.eval_epoch = partial(train_epoch, optimizer=None, lr_scheduler=None, epoch=1,
                                  loss_fn=loss_fn, accelerator=self.accelerator,
                                  metrics_names=metrics_names, metrics_fn=metrics_fn,
                                  loss_accumulate=loss_accumulate, metrics_accumulate=metrics_accumulate)

        # 初始化tensorboard，以及创建其他需要的文件夹
        if self.tbs_dir:
            self.tb_log = TbsLog(log_dir=str(tbs_dir), metrics_names=metrics_names)
            self.accelerator.print(Fore.RED, "TensorBoard has been used.", Fore.RESET)
        self._check_or_create_dir(self.ckp_file, "ckp")
        self._check_or_create_dir(self.best_file, "best")
        self._check_or_create_dir(self.log_file, "log")

    def _check_or_create_dir(self, file_path, dir_name):
        if file_path:
            if check_dir(Path(file_path)):
                self.accelerator.print(Fore.RED, f"create {dir_name} dir: {Path(file_path).parent}", Fore.RESET)
            else:
                self.accelerator.print(Fore.RED, f"{dir_name} dir already exists.", Fore.RESET)

    def _save_best_weight(self):
        net_dict = self.accelerator.get_state_dict(self.net)
        self.accelerator.save(net_dict, self.best_file)

    def _save_log2csv(self, metrics_log: OrderedDict):
        data = [{"epoch": key, **value} for key, value in metrics_log.items()]
        df = pd.DataFrame(data)
        df.to_csv(self.log_file, index=False)

    def _load_ckp(self, path):
        ...

    def _save_ckp(self, path):
        ...

    def fit(self, epochs: int, optimizer, train_dataloader, val_dataloader=None, lr_scheduler=None,
            monitor: str = None, monitor_mode: str = "max", early_stop: int = None,
            start_epoch: int = 1, load_ckp=None):
        """
        monitor: 需要监控的指标，用于early_stop
        monitor_mode: max or min, 选择max或者min模型：
                max是指用若当前数值大于历史best_value，则更新best_value，数值越大精度越高
                max是小于best_value，则更新best_value，数值越小精度越高
        """
        train_dataloader, optimizer = self.accelerator.prepare(train_dataloader, optimizer)
        val_dataloader = self.accelerator.prepare(val_dataloader) if val_dataloader else None
        lr_scheduler = self.accelerator.prepare(lr_scheduler) if lr_scheduler else None
        # 如果不是从头训练则加载检查点，并从检查点继续训练！加载网络权重以及优化器权重！
        if start_epoch != 1:
            # need fix!
            self._load_ckp(load_ckp)

        # 初始化存放需要监控的精度指标
        monitor_best = 0
        monitor_best_epoch = start_epoch
        metrics_epoch = OrderedDict()
        metrics_log = OrderedDict()

        # 获取开始训练时间
        time_recorder = TotalTime()

        # 保存初始的模型初始参数
        if self.tbs_dir:
            self.tb_log.tbs_log_start(self.accelerator, self.net)
        # 开始训练
        start_epoch = start_epoch
        for epoch in range(start_epoch, epochs + 1):

            # 训练============================
            # 模型训练模式
            self.net.train()
            #
            self.accelerator.print(Fore.CYAN + Style.BRIGHT, f"\nEpoch: {epoch}")
            train_metrics_epoch = self.train_epoch(self.net, train_dataloader, optimizer, lr_scheduler, epoch)
            # 将训练精度存入字典
            metrics_epoch.update(deepcopy(train_metrics_epoch))
            # 打印训练结果
            self.accelerator.print("\ttrain--->",
                                   *[f"{name:>10}={value:.5f} " if name != "lr" else f"{name:>10}={value:.2e} "
                                     for name, value in train_metrics_epoch.items()])

            # 测试============================
            if val_dataloader:
                with torch.no_grad():
                    # 模型评估模式
                    self.net.eval()
                    #
                    val_metrics_epoch = self.eval_epoch(self.net, val_dataloader)
                # 如果存在验证集，将验证精度存入字典
                metrics_epoch.update(deepcopy(val_metrics_epoch))
                # 打印评估结果
                self.accelerator.print(f"\tval", "-" * 25,
                                       *[f"{name:>10}={value:.5f} " for name, value in val_metrics_epoch.items()])

            # 检测是否达到最优monitor value
            monitor_value = deepcopy(metrics_epoch[monitor])
            if epoch == start_epoch:
                monitor_best = monitor_value
            if ((monitor_mode == "max" and monitor_value > monitor_best) or
                    (monitor_mode == "min" and monitor_value < monitor_best)):
                monitor_best = monitor_value
                monitor_best_epoch = epoch
                # 打印信息
                self.accelerator.print(Fore.MAGENTA + Style.BRIGHT, f"\treach best {monitor}: {monitor_best}",
                                       Fore.RESET + Style.RESET_ALL)
                # 保存最优模型模型
                if self.best_file:
                    self._save_best_weight()

            # 保存loss,精度曲线到tensorboard训练参数
            if self.tbs_dir:
                self.tb_log.tbs_log_epoch(self.accelerator, self.net, metrics_epoch, epoch)
            # 将结果存放在metrics_log中
            metrics_log.update({f"epoch_{epoch}": deepcopy(metrics_epoch)})

            # early stop
            if early_stop:
                if epoch - monitor_best_epoch > early_stop:
                    self.accelerator.print(
                        Fore.LIGHTRED_EX, f"\n{early_stop} epochs without improvement early stop! "
                                          f"In {monitor_best_epoch} epoch reach best {monitor}: {monitor_best}",
                        Fore.RESET)
                    break

        self.accelerator.print(Fore.LIGHTRED_EX,
                               f"\ntrain over! In {monitor_best_epoch} epoch reach best {monitor}: {monitor_best}",
                               Fore.RESET)
        # 关闭tensorboard
        if self.tbs_dir:
            self.tb_log.writer.close()

        # 保存log文件
        if self.log_file:
            self._save_log2csv(metrics_log)

        # 获取训练时间所花费的时间
        work_hours = time_recorder.get_hours()
        self.accelerator.print(f"Train hours: {work_hours}")

    def evaluate(self, val_dataloader):
        """
        测试模型精度.
            将精度列表里面的函数分别求值
        """
        val_dataloader = self.accelerator.prepare(val_dataloader)
        with torch.no_grad():
            # 模型评估模式
            self.net.eval()
            # 将验证精度以及loss值存入字典
            val_metrics_epoch = self.eval_epoch(self.net, val_dataloader)
            #
        self.accelerator.print(Fore.LIGHTGREEN_EX, f"val", "-" * 25,
                               *[f"{name:>10}={value:.5f} " for name, value in val_metrics_epoch.items()], Fore.RESET)

        return val_metrics_epoch
