"""从配置文件，训练一个DeeplabV3模型"""

import os
from typing import Any, Dict

import torch
import yaml  #导入 YAML 解析库，用来读取config.yaml文件内容，并把它转成Python字典。

import wandb  #导入 Weights & Biases，简称 wandb。这是一个实验记录工具，用来记录

from models import DeepLabWrapper
from utils import Trainer, get_dataloaders

with open("config/config.yaml", "r") as f:   #打开配置文件，以只读模式读取
    config: Dict[str, Any] = yaml.safe_load(f) #读取文件保存到config里，注解告诉python和我们这是一个字典

# 创建一个名为 runs 的文件夹，用来保存训练输出。
os.makedirs("runs", exist_ok=True) #如果 runs 文件夹不存在，就创建它，如果存在也不要报错
#初始化wandb
run = wandb.init(
    entity="a3071093490", #表示 wandb 账户或团队名。
    project="offroad-1",  #项目
    config={  #本次实验的关键信息
        "learning_rate": config.get("LEARNING_RATE", 1e-4),#从配置文件里取学习率，取不到默认1e-4 下面同样
        "batch_size": config.get("BATCH_SIZE", 16),
        "backbone": config.get("BACKBONE", "mobilenetv3large"),
        "dataset": "Yamaha",
        "epochs": config.get("NUM_EPOCHS", 25),
    },
)

dataloaders = get_dataloaders(config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
model = DeepLabWrapper(backbone=config["BACKBONE"], num_mask_channels=config["NUM_MASK_CHANNELS"])
class_weights = torch.tensor(config["CLASS_WEIGHTS"])  #损失权重
class_weights = class_weights.to("cuda")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters, lr=float(config["LEARNING_RATE"]))
trainer = Trainer(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=config["NUM_EPOCHS"],
    logger=run,
    save_model_path=config.get("SAVE_MODEL_PATH"),
)
trainer.train()
