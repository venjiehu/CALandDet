import os
import time
from pathlib import Path
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


def check_dir(path: Path):
    assert isinstance(path, Path), "路径只接受Path类型！！！"
    path = path.parent
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


class MetricsAccumulate:
    def __init__(self, nums):  # need accumulated metrics values nums
        self.nums = nums
        self.mat = [0., ] * nums
        self.n = 0

    def _init(self):
        self.mat = [0., ] * self.nums
        self.n = 0

    def add(self, x_list):
        assert len(x_list) == self.nums, "输入数据维度与初始化不同！！！"
        self.mat = [i + j for i, j in zip(self.mat, x_list)]
        self.n += 1

    def compute(self) -> list:
        """
        :return:  metrics average values
        """
        values = [value / self.n for value in self.mat]
        self._init()

        return values


class TbsLog:
    def __init__(self,
                 log_dir,
                 metrics_names=None,
                 ):
        self.writer = SummaryWriter(log_dir)
        self.metrics_names = metrics_names

    def tbs_log_start(self, accelerator: Accelerator, net):
        net = accelerator.unwrap_model(net)
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
        self.writer.flush()

    def tbs_log_epoch(self, accelerator: Accelerator, net, metrics_epoch: OrderedDict,
                      epoch: int):
        net = accelerator.unwrap_model(net)
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        self.writer.add_scalar("lr", metrics_epoch["lr"], epoch)
        self.writer.add_scalars("loss",
                                {"train_loss": metrics_epoch["train_loss"], "eval_loss": metrics_epoch["eval_loss"]},
                                epoch)
        if self.metrics_names:
            for name in self.metrics_names:
                self.writer.add_scalars(name, {
                    f"train_{name}": metrics_epoch[f"train_{name}"],
                    f"eval_{name}": metrics_epoch[f"eval_{name}"]
                }, epoch)

        self.writer.flush()


class TotalTime:
    def __init__(self):
        self.start_time = time.time()

    def get_hours(self) -> str:
        end_time = time.time()
        run_time = format_time(end_time - self.start_time)
        return run_time
