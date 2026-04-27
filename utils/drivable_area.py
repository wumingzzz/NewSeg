"""可行驶区域重映射工具。

本模块提供将 9 类语义分割结果重映射为二值可行驶区域的方法，
并提供 OpenCV 友好的可视化与叠加函数。
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

# ==============================
# 语义类别常量（与项目约定保持一致）
# ==============================
UNKNOWN = 0
NON_TRAVERSABLE = 1
ROUGH_TRAIL = 2
SMOOTH_TRAIL = 3
TRAVERSABLE_GRASS = 4
LOW_VEGETATION = 5
OBSTACLE = 6
HIGH_VEGETATION = 7
SKY = 8


def mask_to_drivable(mask: np.ndarray, drivable_classes: Iterable[int] | None = None) -> np.ndarray:
    """将语义分割类别 mask 转换为二值可行驶区域 mask。

    默认将以下类别视为可行驶区域：
    - ROUGH_TRAIL（粗糙道路）
    - SMOOTH_TRAIL（平整道路）
    - TRAVERSABLE_GRASS（可通行草地）

    Args:
        mask: H×W 的二维类别 mask（每个像素是类别编号）。
        drivable_classes: 可行驶类别集合；若为 None，则使用默认类别。

    Returns:
        np.ndarray: H×W 的 uint8 二值 mask。
            - 可行驶区域像素值为 255
            - 不可行驶区域像素值为 0

    Raises:
        ValueError: 输入 mask 不是二维数组时抛出。
    """
    mask_array = np.asarray(mask)

    # 只接受二维类别图，避免把彩色图或批量张量误传入后产生隐式错误。
    if mask_array.ndim != 2:
        raise ValueError(
            f"mask_to_drivable 期望输入二维 mask（H×W），但收到 shape={mask_array.shape}。"
        )

    if drivable_classes is None:
        drivable_classes = (ROUGH_TRAIL, SMOOTH_TRAIL, TRAVERSABLE_GRASS)

    # np.isin 会返回布尔图，表示每个像素是否属于可行驶类别。
    drivable_bool = np.isin(mask_array, list(drivable_classes))

    # 转成 uint8 二值图：True -> 255，False -> 0。
    drivable_mask = np.where(drivable_bool, 255, 0).astype(np.uint8)
    return drivable_mask


def drivable_to_color(mask: np.ndarray) -> np.ndarray:
    """将二值可行驶区域 mask 转换为 BGR 彩色图。

    该函数输出适配 OpenCV：
    - 可行驶区域（>0）显示为绿色（BGR: [0, 255, 0]）
    - 不可行驶区域显示为黑色（BGR: [0, 0, 0]）

    Args:
        mask: H×W 的二值可行驶区域 mask。

    Returns:
        np.ndarray: H×W×3 的 uint8 BGR 彩色图。

    Raises:
        ValueError: 输入 mask 不是二维数组时抛出。
    """
    mask_array = np.asarray(mask)

    if mask_array.ndim != 2:
        raise ValueError(
            f"drivable_to_color 期望输入二维二值 mask（H×W），但收到 shape={mask_array.shape}。"
        )

    color = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    color[mask_array > 0] = (0, 255, 0)  # BGR 绿色
    return color


def overlay_drivable_mask(image: np.ndarray, drivable_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """将可行驶区域掩膜半透明叠加到原图（BGR）上。

    Args:
        image: 原始图像，OpenCV BGR 格式，shape 应为 H×W×3。
        drivable_mask: 二值可行驶区域图，shape 应为 H×W。
        alpha: 掩膜叠加透明度，范围 [0, 1]，默认 0.45。

    Returns:
        np.ndarray: 叠加后的 BGR 图像（uint8）。

    Raises:
        ValueError: 当输入维度或尺寸不匹配时抛出。
    """
    image_array = np.asarray(image)
    mask_array = np.asarray(drivable_mask)

    # 图像必须是 H×W×3，且为彩色 BGR。
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            f"overlay_drivable_mask 期望 image 为三维 BGR 图（H×W×3），但收到 shape={image_array.shape}。"
        )

    # mask 必须是二维二值图。
    if mask_array.ndim != 2:
        raise ValueError(
            f"overlay_drivable_mask 期望 drivable_mask 为二维图（H×W），但收到 shape={mask_array.shape}。"
        )

    # 尺寸一致性校验，避免叠加时发生广播错误。
    if image_array.shape[:2] != mask_array.shape:
        raise ValueError(
            "overlay_drivable_mask 收到的 image 与 drivable_mask 尺寸不一致："
            f"image(H,W)={image_array.shape[:2]}，mask(H,W)={mask_array.shape}。"
        )

    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 先生成绿色二值彩色图，再按 alpha 与原图融合。
    color_mask = drivable_to_color(mask_array)
    overlay = cv2.addWeighted(image_array.astype(np.uint8), 1 - alpha, color_mask, alpha, 0)
    return overlay
