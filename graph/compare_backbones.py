"""生成多骨干网络对比表格和论文图片"""

import time
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


BACKBONES = ["mobilenetv3large", "resnet50", "resnet101"]
NUM_EPOCHS = 50
NUM_MASK_CHANNELS = 9
DATA_PATH = Path("data/yamaha_v0")
VALID_IMAGE_COUNT = 20
WARMUP_TIMES = 5

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import DeepLabWrapper


def load_config() -> None:
    """读取配置文件，尽量复用项目里的路径和类别数量"""
    global DATA_PATH, NUM_EPOCHS, NUM_MASK_CHANNELS
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    DATA_PATH = PROJECT_ROOT / config.get("DATA_PATH", str(DATA_PATH))
    NUM_EPOCHS = int(config.get("NUM_EPOCHS", NUM_EPOCHS))
    NUM_MASK_CHANNELS = int(config.get("NUM_MASK_CHANNELS", NUM_MASK_CHANNELS))


def get_history_path(backbone: str) -> Path:
    """获取每个 backbone 对应的 wandb 历史 CSV 路径"""
    history_path = GRAPH_DIR / f"{backbone}_history.csv"
    if backbone == "mobilenetv3large" and not history_path.exists():
        old_history_path = GRAPH_DIR / "wandb_history_raw.csv"
        if old_history_path.exists():
            return old_history_path
    return history_path


def get_model_path(backbone: str) -> Path:
    """获取每个 backbone 对应的模型文件路径"""
    return PROJECT_ROOT / "runs" / f"{backbone}_v1.{NUM_EPOCHS}.pt"


def merge_epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    """wandb 可能把 train 和 valid 写在同一个 epoch 的不同行，这里按 epoch 合并"""
    if "epoch" not in df.columns:
        return df
    return df.groupby("epoch", as_index=False).first().sort_values("epoch")


def read_history(backbone: str) -> pd.DataFrame | None:
    """读取一个 backbone 的训练历史"""
    history_path = get_history_path(backbone)
    if not history_path.exists():
        print(f"{backbone} 缺少训练历史文件：{history_path}")
        return None

    df = pd.read_csv(history_path)
    df = merge_epoch_rows(df)
    return df


def count_parameters(model_path: Path, backbone: str) -> float:
    """统计模型参数量，单位为 M"""
    try:
        loaded_model = torch.load(model_path, map_location="cpu", weights_only=False)
        param_count = sum(p.numel() for p in loaded_model.parameters())
        return param_count / 1e6
    except Exception as e:
        print(f"{backbone} 参数量统计失败：{e}")
        return 0.0


def load_valid_images() -> List[Image.Image]:
    """从验证集读取前若干张图片，用于推理速度测试"""
    valid_dir = DATA_PATH / "valid"
    image_paths = sorted(valid_dir.glob("*/rgb.jpg"))[:VALID_IMAGE_COUNT]
    images = []
    for image_path in image_paths:
        images.append(Image.open(image_path).convert("RGB"))
    return images


def synchronize_if_cuda() -> None:
    """GPU计时时同步，避免异步执行影响结果"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_inference(model_path: Path, images: List[Image.Image]) -> Dict[str, float]:
    """测试单张图片平均推理时间，不包含模型加载时间"""
    if not images:
        return {"avg_ms": 0.0, "std_ms": 0.0, "fps": 0.0}

    model = DeepLabWrapper(model_path=str(model_path))
    processed_images = [model.resize_and_crop_input(image) for image in images]

    for image in processed_images[: min(WARMUP_TIMES, len(processed_images))]:
        _ = model(image)
    synchronize_if_cuda()

    times = []
    for image in processed_images:
        synchronize_if_cuda()
        start = time.perf_counter()
        _ = model(image)
        synchronize_if_cuda()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000 / avg_ms if avg_ms > 0 else 0.0
    return {"avg_ms": avg_ms, "std_ms": std_ms, "fps": fps}


def plot_history(histories: Dict[str, pd.DataFrame], metric: str, ylabel: str, output_name: str) -> None:
    """绘制不同 backbone 的训练曲线"""
    plt.figure(figsize=(5.8, 3.8))
    for backbone, df in histories.items():
        if metric not in df.columns:
            continue
        metric_df = df[["epoch", metric]].dropna()
        plt.plot(metric_df["epoch"], metric_df[metric], marker="o", markersize=3, label=backbone)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"{output_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(GRAPH_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_bar(table: pd.DataFrame, value_col: str, ylabel: str, output_name: str) -> None:
    """绘制柱状图"""
    if table.empty or value_col not in table.columns:
        return

    plt.figure(figsize=(5.8, 3.8))
    plt.bar(table["Backbone"], table[value_col])
    plt.xlabel("Backbone")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"{output_name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(GRAPH_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_speed_accuracy(table: pd.DataFrame) -> None:
    """绘制速度-精度折中散点图"""
    needed_cols = {"Avg Inference Time/ms", "Best Valid mIoU", "Model Size/MB"}
    if table.empty or not needed_cols.issubset(table.columns):
        return

    plt.figure(figsize=(5.8, 3.8))
    size = table["Model Size/MB"].clip(lower=1) * 12
    plt.scatter(table["Avg Inference Time/ms"], table["Best Valid mIoU"], s=size, alpha=0.7)
    for _, row in table.iterrows():
        plt.text(row["Avg Inference Time/ms"], row["Best Valid mIoU"], row["Backbone"], fontsize=9)

    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Best Valid mIoU")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "backbone_speed_accuracy_tradeoff.png", dpi=600, bbox_inches="tight")
    plt.savefig(GRAPH_DIR / "backbone_speed_accuracy_tradeoff.pdf", bbox_inches="tight")
    plt.close()


def build_metric_table(histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总模型大小、参数量、最优指标和推理速度"""
    images = load_valid_images()
    rows = []

    for backbone, df in histories.items():
        model_path = get_model_path(backbone)
        if not model_path.exists():
            print(f"{backbone} 缺少模型文件：{model_path}")
            continue

        best_valid_miou = float(df["valid_mean_iou"].dropna().max()) if "valid_mean_iou" in df.columns else 0.0
        best_valid_gds = float(df["valid_gds"].dropna().max()) if "valid_gds" in df.columns else 0.0
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        params_m = count_parameters(model_path, backbone)
        speed = benchmark_inference(model_path, images)

        rows.append(
            {
                "Backbone": backbone,
                "Params/M": params_m,
                "Model Size/MB": model_size_mb,
                "Best Valid mIoU": best_valid_miou,
                "Best Valid GDS": best_valid_gds,
                "Avg Inference Time/ms": speed["avg_ms"],
                "Inference Std/ms": speed["std_ms"],
                "FPS": speed["fps"],
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        output_path = GRAPH_DIR / "backbone_metric_table.csv"
        table.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"对比表已生成：{output_path}")
        print(table)

    return table


def main() -> None:
    """脚本入口"""
    load_config()

    histories = {}
    for backbone in BACKBONES:
        df = read_history(backbone)
        if df is not None:
            histories[backbone] = df

    if not histories:
        print("没有找到任何可用的训练历史 CSV，请先运行 export_wandb_history.py。")
        return

    plot_history(histories, "valid_loss", "Valid Loss", "backbone_valid_loss_compare")
    plot_history(histories, "valid_mean_iou", "Valid mIoU", "backbone_valid_miou_compare")
    plot_history(histories, "valid_gds", "Valid GDS", "backbone_valid_gds_compare")

    table = build_metric_table(histories)
    plot_bar(table, "Model Size/MB", "Model Size (MB)", "backbone_model_size_compare")
    plot_bar(table, "Avg Inference Time/ms", "Inference Time (ms)", "backbone_inference_time_compare")
    plot_speed_accuracy(table)


if __name__ == "__main__":
    main()
