"""YamahaCMU Dataloaders"""

import glob
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2


class YamahaCMUDataset(VisionDataset):
    #这里定义了一个类 YamahaCMUDataset，继承自 VisionDataset
    """这个类专门表示 Yamaha-CMU 越野道路数据集，负责回答数据集有多少样本第 i 个样本长什么样"""

    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        """初始化 a YamahaCMUDataset object

        Args:
            root: (str)
                数据集的根目录
            transforms: (Optional[Callable])
                要运用的torch transforms
        """
        super().__init__(root, transforms)  #调用父类
        self.image_paths = []   #创建空列表  存放图像路径
        self.mask_paths = []    #创建空列表 存放掩膜路径
        image_mask_pairs = glob.glob(root + "/*/")   #找root（数据集根目录）下所有一级子文件夹
        for image_mask in image_mask_pairs:
            self.image_paths.append(glob.glob(f"{image_mask}*.jpg")[0]) #找到第一个.jpg文件放入图像路径列表
            self.mask_paths.append(glob.glob(f"{image_mask}*.png")[0])  #找到第一个.png文件放入掩膜路径列表

    def __len__(self) -> int:
        """返回数据集的长度，即总共有多少个样本"""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """给定一个索引 index，返回对应的一个样本，返回的为元组[图像tensor,mask tensor]

        Args:
            index: (int)
                the index of the item to get

        Returns:
            the sample at the given index
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert("RGB") #打开图像 转换为RGB
        mask = Image.open(mask_path) #打开mask文件
        mask = np.array(mask)  #从PIL转化为Numpy
        if mask.ndim == 3:
            mask = mask[:, :, 0]        #如果mask是三维的取第一个通道,因为语义标签理论上是二维的
        mask = tv_tensors.Mask(mask, dtype=torch.long)  #mask转化成tv_tensor格式 损失函数一般要求long

        if self.transforms:
            image, mask = self.transforms(image, mask)   #如果有变换则变换俩个

        return image, mask


def get_dataloaders(data_dir: str, batch_size: int = 2) -> Dict[str, DataLoader]:
    """根据YamahaCMU dataset数据目录，创建 train 和 valid 两个 DataLoader

    Args:
        data_dir: (str)
            数据集的总目录
        batch_size: (int)
           每批多少张图，默认 2

    Returns:
        Dict[str, DataLoader]: 返回值是一个字典 containing the train and validation dataloaders
        {
            "train": train_loader,
            "valid": valid_loader
        }
    """
    #定义了一组变换
    train_transforms = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1),  #对图像做轻微（0.1）的亮度和对比度改变
            v2.ToImage(),                                  #把输入转换成 torchvision 的图像对象格式
            v2.ToDtype(torch.float32, scale=True),           # 数据类型变成float32，像素值从0-255缩放到0-1
            v2.RandomCrop(513),                              #随机裁剪出一个513*513区域，统一输入尺寸
            v2.RandomHorizontalFlip(p=0.5),                  #50%的概率做水平翻转
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),#归一化ImageNet 标准归一化参数
        ]
    )

    valid_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(513),
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ]
    )
    #创建 train / valid 数据集对象
    image_datasets = {
        "train": YamahaCMUDataset(data_dir + "train", transforms=train_transforms),
        "valid": YamahaCMUDataset(data_dir + "valid", transforms=valid_transforms)
    }
    #创建Dataloader 又是一个字典推导式
    #dataloaders={
    # "train": DataLoader(image_datasets["train"], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    # "valid": DataLoader(image_datasets["valid"], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    # }
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4),
        "valid": DataLoader(image_datasets["valid"], batch_size=batch_size, pin_memory=True,shuffle=False, num_workers=4)
    }
    return dataloaders
