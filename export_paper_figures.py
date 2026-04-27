"""单张图像推理结果的论文出图模块。

本脚本用于：
1) 加载配置与训练好的模型；
2) 对单张图像进行语义分割推理；
3) 导出语义分割图、可行驶区域图与论文可用对比图。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from models import DeepLabWrapper
from utils import LABEL_NAMES, RGB_COLORS
from utils.drivable_area import mask_to_drivable, overlay_drivable_mask


DEFAULT_IMAGE_PATH = "data/yamaha_v0/valid/iid000898/rgb.jpg"
DEFAULT_MASK_PATH = "data/yamaha_v0/valid/iid000898/labels.png"
DEFAULT_OUTPUT_DIR = "outputs/paper_figures"


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """读取 YAML 配置文件。

    Args:
        config_path: 配置文件路径。

    Returns:
        解析后的配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predicted_mask_to_color(predicted_mask: np.ndarray) -> np.ndarray:
    """将类别编号语义图转换为 RGB 彩色语义分割图。

    Args:
        predicted_mask: H×W 的类别编号图。

    Returns:
        H×W×3 的 RGB 彩色图。
    """
    color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(RGB_COLORS[: len(LABEL_NAMES)]):
        color_mask[predicted_mask == class_id] = rgb
    return color_mask


def make_semantic_overlay(image_rgb: np.ndarray, semantic_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将语义分割彩色图叠加到原图上（RGB）。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = image_rgb.astype(np.float32) * (1.0 - alpha) + semantic_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def ensure_exists(path: Path, desc: str) -> None:
    """检查路径是否存在，不存在时抛出清晰错误。"""
    if not path.exists():
        raise FileNotFoundError(f"{desc}不存在：{path}")


def export_figures(image_path: str | None = None, mask_path: str | None = None, output_dir: str | None = None) -> Path:
    """执行单图推理并导出论文图片。

    Args:
        image_path: 输入图像路径；为空时使用默认值。
        mask_path: 标签图路径；为空时使用默认值（仅用于同步 resize/crop，不参与训练）。
        output_dir: 输出根目录；为空时使用默认值。

    Returns:
        本次样本的输出目录路径。
    """
    config = load_config("config/config.yaml")

    # 使用配置中的核心路径（按需求保留）。
    data_path = config.get("DATA_PATH")
    model_path = Path(str(config.get("LOAD_MODEL_PATH", "")))

    if not data_path:
        raise ValueError("config/config.yaml 中缺少 DATA_PATH。")
    if not model_path:
        raise ValueError("config/config.yaml 中缺少 LOAD_MODEL_PATH。")

    ensure_exists(model_path, "模型文件")

    img_path = Path(image_path or DEFAULT_IMAGE_PATH)
    gt_mask_path = Path(mask_path or DEFAULT_MASK_PATH)

    # 保底：如果默认路径不存在，尝试根据 DATA_PATH 拼接同名结构。
    if not img_path.exists() and image_path is None:
        fallback = Path(str(data_path)) / "valid/iid000898/rgb.jpg"
        img_path = fallback
    if not gt_mask_path.exists() and mask_path is None:
        fallback = Path(str(data_path)) / "valid/iid000898/labels.png"
        gt_mask_path = fallback

    ensure_exists(img_path, "输入图片")
    ensure_exists(gt_mask_path, "输入标签")

    model = DeepLabWrapper(model_path=str(model_path))

    image_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.open(gt_mask_path)

    # 与项目现有 test.py 一致：先对 image/mask 同步 resize_and_crop。
    processed_image_pil, _ = model.resize_and_crop_input(image_pil, mask_pil)
    predicted_mask_pil = model(processed_image_pil)

    image_rgb = np.array(processed_image_pil)
    predicted_mask = np.array(predicted_mask_pil)

    # 语义分割可视化
    semantic_mask_rgb = predicted_mask_to_color(predicted_mask)
    semantic_overlay_rgb = make_semantic_overlay(image_rgb, semantic_mask_rgb, alpha=0.5)

    # 可行驶区域可视化（使用新增工具模块）
    drivable_mask = mask_to_drivable(predicted_mask)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    drivable_overlay_bgr = overlay_drivable_mask(image_bgr, drivable_mask, alpha=0.45)

    # 输出目录：outputs/paper_figures/样本名/
    sample_name = img_path.parent.name if img_path.parent.name else img_path.stem
    out_root = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir = out_root / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分别保存论文图
    Image.fromarray(image_rgb).save(out_dir / "original.png")
    Image.fromarray(semantic_mask_rgb).save(out_dir / "semantic_mask.png")
    Image.fromarray(semantic_overlay_rgb).save(out_dir / "semantic_overlay.png")
    Image.fromarray(drivable_mask).save(out_dir / "drivable_mask.png")
    cv2.imwrite(str(out_dir / "drivable_overlay.png"), drivable_overlay_bgr)

    # comparison.png：2x2 论文组合图（matplotlib, dpi=300, 英文标题, 无坐标轴）
    drivable_overlay_rgb = cv2.cvtColor(drivable_overlay_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("(a) Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(semantic_overlay_rgb)
    axes[0, 1].set_title("(b) Semantic Segmentation")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(drivable_mask, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(c) Drivable Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(drivable_overlay_rgb)
    axes[1, 1].set_title("(d) Drivable Overlay")
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="导出单张图像语义分割与可行驶区域论文图")
    parser.add_argument("--image_path", type=str, default=None, help="输入图片路径")
    parser.add_argument("--mask_path", type=str, default=None, help="输入标签路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出根目录")
    return parser


def main() -> None:
    """程序入口。"""
    args = build_arg_parser().parse_args()
    out_dir = export_figures(
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
    )
    print(f"论文图片已导出到：{out_dir}")


if __name__ == "__main__":
    main()
