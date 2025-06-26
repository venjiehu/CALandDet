import os
import albumentations as A  # noqa
import cv2
import torch
import torchvision
import numpy as np
from functools import partial
#
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from typing import Optional, Tuple
#
from datasets.plot import show_img_mask


def pad_img_func(img: Image.Image,
                 mask: Image.Image = None,
                 width: int = 224,
                 height: int = 224,
                 random: bool = True,
                 ):
    """
    random cut or center cut
    inputs:
        img: img.convert('RGB')
        mask: mask.convert('L')
    """

    if mask:
        assert img.size == mask.size, "输入图像与标签尺寸不匹配！！！"

    w, h = img.size

    img_in = np.array(img).astype(np.float32)
    img_out = np.zeros((height, width, 3), dtype=np.float32)

    # 计算裁剪或填充位置
    h_out_offset = max(height - h, 0)
    w_out_offset = max(width - w, 0)

    h_in_offset = max(h - height, 0)
    w_in_offset = max(w - width, 0)

    if random:
        h_out_offset = np.random.randint(0, h_out_offset + 1)
        w_out_offset = np.random.randint(0, w_out_offset + 1)
        h_in_offset = np.random.randint(0, h_in_offset + 1)
        w_in_offset = np.random.randint(0, w_in_offset + 1)
    else:
        # 中心裁剪位置
        h_out_offset = h_out_offset // 2
        w_out_offset = w_out_offset // 2
        h_in_offset = h_in_offset // 2
        w_in_offset = w_in_offset // 2

    # 计算裁剪或填充后的尺寸
    delta_height = min(h, height)
    delta_width = min(w, width)

    # 将输入张量矩阵裁剪或填充到新的张量矩阵中心位置
    img_out[h_out_offset:h_out_offset + delta_height, w_out_offset:w_out_offset + delta_width, :] \
        = img_in[h_in_offset:h_in_offset + delta_height, w_in_offset:w_in_offset + delta_width, :]

    if mask:
        mask_in = np.array(mask).astype(np.float32) / 255.
        mask_out = np.zeros((height, width), dtype=np.float32)

        mask_out[h_out_offset:h_out_offset + delta_height, w_out_offset:w_out_offset + delta_width] \
            = mask_in[h_in_offset:h_in_offset + delta_height, w_in_offset:w_in_offset + delta_width]
        return img_out, mask_out

    return img_out


def red_img_file(file_path: Path):
    parent = file_path.parent
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines, parent


class BaseDataset(Dataset):
    def __init__(self,
                 img_file: Path,
                 img_size: Tuple[int, int] = (224, 224),
                 img_aug: Optional[A.Compose] = None):
        self.img_list, self.img_parent = red_img_file(img_file)
        self.img_h, self.img_w = img_size
        self.img_aug = img_aug if img_aug else ToTensorV2(p=1.)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # use ImageNet mean,std
        )

    def __len__(self) -> int:
        return len(self.img_list)

    def normalize_img(self, img) -> torch.Tensor:
        return self.normalize(img.float() / 255.)

    def get_img_mask(self, index: int, show: bool = False):
        img_path = os.path.join(self.img_parent, self.img_list[index])
        mask_path = os.path.join(self.img_parent, self.img_list[index].replace("image", "mask"))
        if not show:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        else:
            """
              使用opencv展示，show_sample函数展示样本
            """
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

        return image, mask

    def show_sample(self, index_list: list):
        """
            example:
                datasets = TrainValDataset(...)
                datasets.show_sample(index_list)
        """

        for index in index_list:
            assert index < self.__len__(), "index超出范围！！！"
            img, mask = self.get_img_mask(index, show=True)
            show_img_mask(img, mask, img_name=index, waite_key=0)


class TrainValDataset(BaseDataset):
    """
        返回值指定大小的图像和标签,
        可选择随机裁剪和中心裁剪,
    """

    def __init__(self,
                 img_file: Path,
                 img_size: Tuple[int, int] = (224, 224),
                 mode: str = "normal",
                 random: bool = True,
                 img_aug: Optional[A.Compose] = None):
        super().__init__(img_file, img_size, img_aug=img_aug)
        self.mode = mode
        self.pad_img = partial(pad_img_func, width=self.img_w, height=self.img_h, random=random)

    def __getitem__(self, index: int):
        image, mask = self.get_img_mask(index, show=False)
        image, mask = self.pad_img(img=image, mask=mask)

        sample = self.img_aug(image=image, mask=mask)

        # 将mask维度为H x W修改为，C x H x W, 同时添加滑坡标签p
        if self.mode == "normal":
            sample['mask'] = sample['mask'].unsqueeze(0)
        else:
            p = 1 if mask.sum() > 0 else 0
            sample['mask'] = (sample['mask'].unsqueeze(0), p)

        sample['image'] = self.normalize_img(sample['image'])

        return sample
