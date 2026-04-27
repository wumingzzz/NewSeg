import wandb
import pandas as pd

ENTITY = "a3071093490-k"
PROJECT = "offroad-1"
RUN_ID = "vsyc679f"

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")


rows = list(run.scan_history())
df = pd.DataFrame(rows)

print("原始列名：", df.columns.tolist())
print(df.head())

# 只保留你论文需要的列
keep_cols = ["epoch", "train_loss", "valid_loss", "train_mean_iou", "valid_mean_iou"]
keep_cols = [c for c in keep_cols if c in df.columns]

df = df[keep_cols]
df.to_csv("wandb_history_raw.csv", index=False, encoding="utf-8-sig")

print("导出完成：wandb_history_raw.csv")
print(df.head(10))