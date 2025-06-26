import torch
from accelerate import Accelerator


def train_step(net, loss_fn, mode: str, metrics_fn, accelerator: Accelerator,  # mode:str train or eval
               batch, epoch, step, data_loader_length, lr_scheduler, optimizer):
    # 输入的batch数据
    features, labels = batch['image'], batch['mask']

    # loss
    with accelerator.autocast():
        preds = net(features)
        loss = loss_fn(preds, labels)

    # backward()
    if optimizer is not None and mode == "train":
        optimizer.zero_grad()
        accelerator.backward(loss)
        #
        optimizer.step()
        # lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(step / data_loader_length + epoch)

    # init metrics list
    metrics = []
    if metrics_fn:
        with torch.no_grad():
            metrics = [float(fn(preds, labels)) for fn in metrics_fn]

    return loss.item(), metrics
