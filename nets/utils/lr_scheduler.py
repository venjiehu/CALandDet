import math
import re
from typing import Tuple, Union
from torch.optim.lr_scheduler import LRScheduler


def param_groups_layer_decay(model,
                             weight_decay: float = 0.05,
                             no_weight_decay_list: Tuple[str] = (),
                             layer_decay: float = 0.75
                             ):
    """
        逐层递减的学习率策略，同时包含权重衰减策略
        输出的模块名字： part.block.index
            part为backbone、neck、head这三部分
            block为每一层结构的名字
            index为每一层的编号
    """

    # 创建存放学习率的字典
    lr_group_names = []
    # 创建存放参数的字典
    param_group_names = {}

    # 创建学习率衰减以及权重衰减字典，筛选需要衰减的参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_list = name.split('.')
        part = name_list[0]
        block = name_list[1]
        index = name_list[2] if len(name_list) > 2 else 0

        # 所有的1D变量以及传入的不需要权重衰减的变量，不会执行权重衰减
        if param.ndim <= 1 or name.endswith(".bias") or (name in no_weight_decay_list):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        """
        获取当前模块的名字，用于后续判定。
            1、当为part.block.index形式式
            neck.layers.2.swin_blocks.swin_blocks.1.mlp.3.weight
            neck.layers.2.swin_blocks.swin_blocks.1.mlp.3.bias
            
            2、当block有层数后缀或者index不代表层数时，执行else
            head.head_block1.nll.norm.weight
            head.head_block1.nll.norm.bias
            head.head_block1.nll.fc.weight
        """
        if not re.search(r'\d', block) and re.search(r'\d', index):
            lr_group_name = f"{part}_{block}_{index}"
        else:
            lr_group_name = f"{part}_{block}"

        # 将需要不同学习率衰减的层名字添加进去
        if lr_group_name not in lr_group_names:
            lr_group_names.append(lr_group_name)

        # 将每层添加是否需要weight decay后缀，用'-'符号隔断方便后续分割查询
        group_name = f"{lr_group_name}-{g_decay}"
        # 如果param_group_names没有该grop name则创建相应的键值对，设置weight decay;设置lr scale为1，后续对其更新
        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": 1,
                "weight_decay": this_decay,
                "params": [],
            }

        # 以层的名字为key值，将每层的参数加入到参数列表
        param_group_names[group_name]["params"].append(param)

    # 反转存放学习率的列表，使其从尾到头
    lr_group_names_reversed = lr_group_names[::-1]
    # 创建存放学习率衰减的字典，key为衰减的层名字，value为衰减测scale
    lr_decay_dicts = {}
    for exp, name in enumerate(lr_group_names_reversed):
        lr_scale = layer_decay ** exp
        lr_decay_dicts[name] = lr_scale

    # 遍历参数列表，查询需要的衰减学习率并跟新衰减scale
    for key in param_group_names.keys():
        lr_decay_query = key.split("-")[0]
        param_group_names[key]["lr_scale"] = lr_decay_dicts.get(lr_decay_query, 1)

    return list(param_group_names.values())


def param_groups_weight_decay(
        model,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    """
    using timm code
    following: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py#L41
    """
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_param_groups(net,
                     optimizer_method: str = None,
                     layer_decay_dict: dict = None,
                     weight_decay_dict: dict = None
                     ):
    if "layer_decay" == optimizer_method:
        layer_decay_rate = layer_decay_dict.get("layer_decay_rate", 0.75)
        weight_decay = layer_decay_dict.get("weight_decay", 0.05)
        no_weight_decay_list = layer_decay_dict.get("no_weight_decay_list", None)
        no_weight_decay_list = no_weight_decay_list if no_weight_decay_list else ()

        print("use layer decay!")
        print(f"\tlayer decay rate: {layer_decay_rate}")
        print(f"\tweight decay: {weight_decay}")
        print(f"\tno weight decay list: {no_weight_decay_list}")

        param_groups = param_groups_layer_decay(model=net,
                                                weight_decay=weight_decay,
                                                no_weight_decay_list=no_weight_decay_list,
                                                layer_decay=layer_decay_rate
                                                )
        return param_groups

    elif "weight_decay" == optimizer_method:
        weight_decay = weight_decay_dict.get("weight_decay", 0.05)
        no_weight_decay_list = weight_decay_dict.get("no_weight_decay_list", None)
        no_weight_decay_list = no_weight_decay_list if no_weight_decay_list else ()

        print("use weight decay!")
        print(f"\tweight decay: {weight_decay}")
        print(f"\tno weight decay list: {no_weight_decay_list}")

        param_groups = param_groups_weight_decay(model=net,
                                                 weight_decay=weight_decay,
                                                 no_weight_decay_list=no_weight_decay_list,
                                                 )
        return param_groups


class LearningRateScheduler(LRScheduler):
    """
        学习率调度器:
            模型前半段进行预热，后半段学习率采用余弦退火学习率调度
    """

    def __init__(self,
                 optimizer,
                 lr: float = 1e-3,
                 min_lr: float = 0.,
                 epochs: int = 100,
                 warmup_epochs: int = None,
                 ):
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.epochs = epochs
        super().__init__(optimizer=optimizer)

    def _warmup(self, epoch):
        lr = self.lr * epoch / self.warmup_epochs

        return lr

    def _half_cycle_cosine(self, epoch):
        lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))

        return lr

    def step(self, epoch: Union[int, float] = 1, *args, **kwargs):
        if epoch < self.warmup_epochs:
            lr = self._warmup(epoch)
        else:
            lr = self._half_cycle_cosine(epoch)

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
