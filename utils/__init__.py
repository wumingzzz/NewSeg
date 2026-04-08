"""Initializes the utils module"""

#初始化utils模块
#这样做之后外部可以直接from utils import get_dataloaders
#而不用写成from utils.dataset import get_dataloaders
from .dataset import get_dataloaders
from .trainer import Trainer
from .utils import (
    display_example_pair,
    overlay_mask_cv2,
    vis_segmentation,
)

#当使用from utils import ...时 允许被导出的名字有哪些
__all__ = [
    "get_dataloaders",
    "Trainer",
    "vis_segmentation",
    "display_example_pair",
    "overlay_mask_cv2",
]
