"""生成典型场景下三种骨干网络预测结果对比图。

该脚本放在项目根目录运行，只用于生成论文插图，不参与模型训练。
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from models import DeepLabWrapper
from utils import RGB_COLORS


DATA_PATH = PROJECT_ROOT / "data" / "yamaha_v0"
VALID_DIR = DATA_PATH / "valid"

MODEL_PATHS = {
    "MobileNetV3-Large": PROJECT_ROOT / "runs" / "mobilenetv3large_v1.50.pt",
    "ResNet50": PROJECT_ROOT / "runs" / "resnet50_v1.50.pt",
    "ResNet101": PROJECT_ROOT / "runs" / "resnet101_v1.50.pt",
}

CASE_NAMES = {
    "clear_road": "道路主体清晰场景",
    "fuzzy_boundary": "道路边界模糊场景",
    "scale_change": "道路尺度变化场景",
    "complex_scene": "复杂背景与障碍物场景",
}

OUTPUT_NAMES = {
    "clear_road": "case_clear_road_compare",
    "fuzzy_boundary": "case_fuzzy_boundary_compare",
    "scale_change": "case_scale_change_compare",
    "complex_scene": "case_complex_scene_compare",
}

# 如果自动筛选结果不理想，可以在这里手动指定验证集样本ID
MANUAL_CASES = {
    "clear_road": "iid001052",
    "fuzzy_boundary": "iid000881",
    "scale_change": "iid000987",
    "complex_scene": "iid000876",
}

# 不希望再次选中的样本ID
EXCLUDE_SAMPLE_IDS = {"iid001040", "iid000975", "iid000941", "iid000906"}


def read_mask(mask_path: Path) -> np.ndarray:
    """读取标签图"""
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """把语义标签 mask 转换成彩色图"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(RGB_COLORS):
        color_mask[mask == class_id] = rgb
    return color_mask


def get_sample_paths(sample_id: str) -> tuple[Path, Path]:
    """根据样本ID获取图像和标签路径"""
    sample_dir = VALID_DIR / sample_id
    image_path = sample_dir / "rgb.jpg"
    mask_path = sample_dir / "labels.png"
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"验证集样本文件不完整：{sample_dir}")
    return image_path, mask_path


def calculate_label_stats(mask: np.ndarray) -> dict:
    """根据真实标签统计场景特征"""
    total_pixels = mask.size
    road = np.isin(mask, [1, 3])
    vegetation = np.isin(mask, [2, 6, 7])
    risk = np.isin(mask, [0, 4, 5])

    height, width = mask.shape
    bottom_road = road[int(height * 0.65):].mean()
    upper_road = road[: int(height * 0.45)].mean()
    center_road = road[int(height * 0.45):, int(width * 0.25): int(width * 0.75)].mean()

    road_veg_boundary = 0
    road_veg_boundary += int((road[:-1, :] & vegetation[1:, :]).sum())
    road_veg_boundary += int((vegetation[:-1, :] & road[1:, :]).sum())
    road_veg_boundary += int((road[:, :-1] & vegetation[:, 1:]).sum())
    road_veg_boundary += int((vegetation[:, :-1] & road[:, 1:]).sum())

    return {
        "road_ratio": float(road.mean()),
        "vegetation_ratio": float(vegetation.mean()),
        "risk_ratio": float(risk.mean()),
        "unknown_ratio": float((mask == 0).sum() / total_pixels),
        "non_traversable_ratio": float((mask == 4).sum() / total_pixels),
        "obstacle_ratio": float((mask == 5).sum() / total_pixels),
        "bottom_road_ratio": float(bottom_road),
        "upper_road_ratio": float(upper_road),
        "center_road_ratio": float(center_road),
        "road_scale_score": float(bottom_road - upper_road),
        "road_veg_boundary_ratio": float(road_veg_boundary / total_pixels),
    }


def score_case(case_key: str, stats: dict) -> float:
    """计算不同场景的筛选分数"""
    if case_key == "clear_road":
        return (
            stats["road_ratio"] * 2
            + stats["center_road_ratio"]
            - stats["risk_ratio"]
            - stats["vegetation_ratio"] * 0.4
        )
    if case_key == "fuzzy_boundary":
        if stats["road_ratio"] < 0.08:
            return -1.0
        return (
            stats["road_veg_boundary_ratio"] * 10
            + stats["vegetation_ratio"] * 0.3
            + stats["road_ratio"] * 0.5
            - stats["risk_ratio"] * 0.4
        )
    if case_key == "scale_change":
        return (
            stats["road_scale_score"]
            + stats["bottom_road_ratio"] * 0.5
            + stats["upper_road_ratio"] * 0.1
        )
    if case_key == "complex_scene":
        return (
            stats["risk_ratio"]
            + stats["obstacle_ratio"] * 6
            + stats["non_traversable_ratio"] * 3
            + stats["unknown_ratio"]
            + stats["vegetation_ratio"] * 0.2
        )
    return 0.0


def collect_valid_samples() -> list[dict]:
    """统计验证集每张图的标签信息"""
    rows = []
    for sample_dir in sorted(VALID_DIR.glob("iid*")):
        if sample_dir.name in EXCLUDE_SAMPLE_IDS:
            continue
        mask_path = sample_dir / "labels.png"
        if not mask_path.exists():
            continue
        stats = calculate_label_stats(read_mask(mask_path))
        rows.append({"sample_id": sample_dir.name, **stats})
    return rows


def select_cases(rows: list[dict]) -> dict:
    """自动或手动选择四类典型样本"""
    selected_cases = {}
    for case_key in CASE_NAMES:
        if case_key in MANUAL_CASES:
            sample_id = MANUAL_CASES[case_key]
            row = next((item for item in rows if item["sample_id"] == sample_id), None)
            if row is None:
                raise ValueError(f"手动指定的样本不存在：{case_key}={sample_id}")
            selected_cases[case_key] = row
            continue

        selected_cases[case_key] = max(rows, key=lambda item: score_case(case_key, item))
    return selected_cases


def load_models() -> dict:
    """加载三个模型"""
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        models[model_name] = DeepLabWrapper(model_path=str(model_path))
    return models


def predict_one_sample(models: dict, image: Image.Image) -> dict:
    """对同一张图分别使用三个模型预测"""
    predictions = {}
    for model_name, model in models.items():
        predicted_mask = model(image)
        predictions[model_name] = np.array(predicted_mask)
    return predictions


def draw_case_figure(
    case_key: str,
    sample_id: str,
    image: Image.Image,
    mask: Image.Image,
    predictions: dict,
) -> None:
    """绘制单个典型场景的三模型预测对比图"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    image_array = np.array(image)
    mask_array = np.array(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    fig = plt.figure(figsize=(15, 8))
    grid_spec = fig.add_gridspec(2, 6)

    axes = [
        fig.add_subplot(grid_spec[0, 1:3]),
        fig.add_subplot(grid_spec[0, 3:5]),
        fig.add_subplot(grid_spec[1, 0:2]),
        fig.add_subplot(grid_spec[1, 2:4]),
        fig.add_subplot(grid_spec[1, 4:6]),
    ]

    titles = [
        "原始图像",
        "真实标签",
        "MobileNetV3-Large预测",
        "ResNet50预测",
        "ResNet101预测",
    ]
    images = [
        image_array,
        colorize_mask(mask_array),
        colorize_mask(predictions["MobileNetV3-Large"]),
        colorize_mask(predictions["ResNet50"]),
        colorize_mask(predictions["ResNet101"]),
    ]

    for ax, title, show_image in zip(axes, titles, images):
        ax.imshow(show_image)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(CASE_NAMES[case_key], fontsize=15)
    plt.tight_layout()

    output_name = OUTPUT_NAMES[case_key]
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.close()


def save_selected_cases(selected_cases: dict) -> None:
    """保存筛选出的样本信息"""
    output_path = OUTPUT_DIR / "selected_case_samples.csv"
    fieldnames = [
        "case_key",
        "case_name",
        "sample_id",
        "road_ratio",
        "vegetation_ratio",
        "risk_ratio",
        "unknown_ratio",
        "non_traversable_ratio",
        "obstacle_ratio",
        "bottom_road_ratio",
        "upper_road_ratio",
        "center_road_ratio",
        "road_scale_score",
        "road_veg_boundary_ratio",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_key, row in selected_cases.items():
            writer.writerow(
                {
                    "case_key": case_key,
                    "case_name": CASE_NAMES[case_key],
                    **row,
                }
            )


def main() -> None:
    """主函数"""
    rows = collect_valid_samples()
    if not rows:
        raise RuntimeError("没有找到验证集标签文件")

    selected_cases = select_cases(rows)
    save_selected_cases(selected_cases)
    models = load_models()

    for case_key, row in selected_cases.items():
        image_path, mask_path = get_sample_paths(row["sample_id"])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = next(iter(models.values())).resize_and_crop_input(image, mask)
        predictions = predict_one_sample(models, image)
        draw_case_figure(case_key, row["sample_id"], image, mask, predictions)
        print(f"{CASE_NAMES[case_key]}：{row['sample_id']}")

    print("典型场景三模型预测对比图生成完成")


if __name__ == "__main__":
    main()
