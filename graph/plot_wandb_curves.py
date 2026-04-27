import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wandb_history_raw.csv")

train_loss_df = df[["epoch", "train_loss"]].dropna().drop_duplicates(subset=["epoch"]).sort_values("epoch")
valid_loss_df = df[["epoch", "valid_loss"]].dropna().drop_duplicates(subset=["epoch"]).sort_values("epoch")

train_miou_df = df[["epoch", "train_mean_iou"]].dropna().drop_duplicates(subset=["epoch"]).sort_values("epoch")
valid_miou_df = df[["epoch", "valid_mean_iou"]].dropna().drop_duplicates(subset=["epoch"]).sort_values("epoch")

# 统一字体大小
plt.rcParams["font.size"] = 10

# 图5：Loss 曲线
plt.figure(figsize=(5.8, 3.8))
plt.plot(train_loss_df["epoch"], train_loss_df["train_loss"], marker="o", markersize=3, label="Train Loss")
plt.plot(valid_loss_df["epoch"], valid_loss_df["valid_loss"], marker="s", markersize=3, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("loss_curves_small.png", dpi=600, bbox_inches="tight")
plt.savefig("loss_curves_small.pdf", bbox_inches="tight")
plt.close()

# 图6：mIoU 曲线
plt.figure(figsize=(5.8, 3.8))
plt.plot(train_miou_df["epoch"], train_miou_df["train_mean_iou"], marker="o", markersize=3, label="Train mIoU")
plt.plot(valid_miou_df["epoch"], valid_miou_df["valid_mean_iou"], marker="s", markersize=3, label="Valid mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("miou_curves_small.png", dpi=600, bbox_inches="tight")
plt.savefig("miou_curves_small.pdf", bbox_inches="tight")
plt.close()

print("已生成。")