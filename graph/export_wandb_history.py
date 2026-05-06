"""从 wandb 导出多个骨干网络的训练历史，用于论文画图"""

from pathlib import Path

import pandas as pd
import wandb


ENTITY = "a3071093490-k"
PROJECT = "offroad-1"

# 训练完不同 backbone 后，把对应的 wandb run id 填到这里。
# 例如：{"mobilenetv3large": "vsyc679f", "resnet50": "xxxxxxx", "resnet101": "yyyyyyy"}
RUN_IDS = {
    "mobilenetv3large": "vsyc679f",
    "resnet50": "",
    "resnet101": "",
}

KEEP_COLS = [
    "epoch",
    "train_loss",
    "valid_loss",
    "train_mean_iou",
    "valid_mean_iou",
    "train_gds",
    "valid_gds",
]


def export_one_run(api: wandb.Api, backbone: str, run_id: str) -> None:
    """导出一个 backbone 对应的 wandb 训练历史"""
    if not run_id:
        print(f"{backbone} 没有填写 run id，跳过。")
        return

    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history())
    df = pd.DataFrame(rows)

    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    if not keep_cols:
        print(f"{backbone} 没有找到可导出的训练指标，跳过。")
        return

    df = df[keep_cols]
    output_path = Path(__file__).resolve().parent / f"{backbone}_history.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{backbone} 导出完成：{output_path}")
    print(df.head(10))


def main() -> None:
    """脚本入口"""
    api = wandb.Api()
    for backbone, run_id in RUN_IDS.items():
        export_one_run(api, backbone, run_id)


if __name__ == "__main__":
    main()
