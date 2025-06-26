import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_img(img: np.array = None, title='image', axis='on', color_bar=False):
    plt.figure()
    plt.imshow(img)
    plt.axis(axis)  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    if color_bar:
        plt.colorbar()
    plt.show()


def plot_history(history: pd.DataFrame):
    epoch = history["epoch"]
    train_loss = history["train_loss"]
    train_iou = history["train_iou"]
    lr = history["lr"]
    val_iou = history["val_iou"]

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].plot(epoch, train_iou, label="train iou", color="#a92faf", linestyle="--")
    axs[0].plot(epoch, val_iou, label='val iou', color="#417928", linestyle="-.")
    axs[0].plot(epoch, train_loss, label="train loss", color="#4478ad")
    axs[0].grid(True)
    axs[0].set_xlabel("epoch")
    axs[0].legend()

    axs[1].plot(epoch, lr, label="lr")
    axs[1].set_xlabel("epoch")
    axs[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[1].legend()
    plt.show()


def concat_img_mask(img, mask, rgb):
    r, g, b = rgb
    mask[..., 0] = mask[..., 0] / 255 * r  # b
    mask[..., 1] = mask[..., 1] / 255 * b  # g
    mask[..., 2] = mask[..., 2] / 255 * r  # r

    overlapping = cv2.addWeighted(img, 1, mask, 0.6, 0)
    stack = np.concatenate((img, overlapping), axis=1)
    return stack


def show_img_mask(img, mask, rgb=(160, 26, 26), img_name=None, waite_key: int = 0):
    """
    cv2 read img: H,W,C
    img == maks == h,w,c
    """
    img_list, mask_list = np.array(img), np.array(mask)
    if img_list.ndim == mask_list.ndim == 3:
        stack = concat_img_mask(img_list, mask_list, rgb)
        windows_name = f"show sample: {img_name}"
        cv2.namedWindow(windows_name, 0)
        cv2.resizeWindow(windows_name, 600, 300)
        cv2.moveWindow(windows_name, 100, 100)
        cv2.imshow(windows_name, stack)
        cv2.waitKey(waite_key)
        cv2.destroyWindow(windows_name)

    else:
        raise ValueError("input value error!!!")
