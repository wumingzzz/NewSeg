"""Initializes the utils module"""

#初始化utils模块
#这样做之后外部可以直接from utils import get_dataloaders
#而不用写成from utils.dataset import get_dataloaders
from .dataset import get_dataloaders
from .trainer import Trainer
from .drivable_area import (
    HIGH_VEGETATION,
    LOW_VEGETATION,
    NON_TRAVERSABLE,
    OBSTACLE,
    ROUGH_TRAIL,
    SKY,
    SMOOTH_TRAIL,
    TRAVERSABLE_GRASS,
    UNKNOWN,
    drivable_to_color,
    mask_to_drivable,
    overlay_drivable_mask,
)

from .utils import (
    display_example_pair,
    overlay_mask_cv2,
    vis_segmentation,
    LABEL_NAMES,
    RGB_COLORS
)

#当使用from utils import ...时 允许被导出的名字有哪些
__all__ = [
    "get_dataloaders",
    "Trainer",
    "vis_segmentation",
    "display_example_pair",
    "overlay_mask_cv2",
    "LABEL_NAMES",
    "RGB_COLORS",
    "UNKNOWN",
    "NON_TRAVERSABLE",
    "ROUGH_TRAIL",
    "SMOOTH_TRAIL",
    "TRAVERSABLE_GRASS",
    "LOW_VEGETATION",
    "OBSTACLE",
    "HIGH_VEGETATION",
    "SKY",
    "mask_to_drivable",
    "drivable_to_color",
    "overlay_drivable_mask"
]
