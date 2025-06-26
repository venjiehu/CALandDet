from tqdm import tqdm
from collections import OrderedDict
from accelerate import Accelerator
#
from trainer._train_step import train_step
from trainer._utils import MetricsAccumulate


def train_epoch(net, dataloader, optimizer, lr_scheduler, epoch,
                loss_fn,
                accelerator: Accelerator,
                metrics_names, metrics_fn,
                loss_accumulate: MetricsAccumulate, metrics_accumulate: MetricsAccumulate
                ) -> OrderedDict:
    #  判断模型是否处于训练状态
    mode = "train" if net.training else "eval"
    batch_nums = len(dataloader)
    loop = tqdm(enumerate(dataloader, start=1),
                total=batch_nums,
                ncols=0,
                desc=f"{mode}: ",
                unit="step",
                leave=False,
                )
    # 初始化精度字典，metrics_dict用于tqdm动态显示精度，metrics_epoch用于存放当前周期最后的精度
    metrics_dict = OrderedDict()
    metrics_epoch = OrderedDict()
    for step, batch in loop:
        # 梯度累加模块，net需要被accelerator.prepare包裹
        # accelerator = Accelerator(gradient_accumulation_steps=1)
        with accelerator.accumulate(net):
            step_loss, step_metrics = train_step(net=net, loss_fn=loss_fn, mode=mode, metrics_fn=metrics_fn,
                                                 accelerator=accelerator, batch=batch, epoch=epoch, step=step,
                                                 data_loader_length=batch_nums, lr_scheduler=lr_scheduler,
                                                 optimizer=optimizer)

        if optimizer is not None and mode == "train":
            max_lr = 0
            for group in optimizer.param_groups:
                max_lr = max(max_lr, group["lr"])
            metrics_dict.update({"lr": '{:.4e}'.format(max_lr)})
            metrics_epoch.update({"lr": max_lr})

        # 将loss存放如累加器以及精度字典
        loss_accumulate.add([step_loss])
        metrics_dict.update({f"{mode}_loss": '{:.4f}'.format(step_loss)})

        # 如果存在metrics_fn则计算精度
        if metrics_fn:
            metrics_accumulate.add(step_metrics)
            metrics_dict.update(
                {f"{mode}_{name}": '{:.3f}'.format(value) for name, value in zip(metrics_names, step_metrics)})

        # 传入tqdm动态显示
        loop.set_postfix(metrics_dict)

    metrics_epoch.update({f"{mode}_loss": loss_accumulate.compute().pop()})
    if metrics_fn:
        metrics_epoch.update(
            {f"{mode}_{name}": value for name, value in zip(metrics_names, metrics_accumulate.compute())})

    return metrics_epoch
