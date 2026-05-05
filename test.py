"""对单张输入图像运行推理"""

import os.path as op

import numpy as np
import yaml
from PIL import Image

from models import DeepLabWrapper
from utils import vis_segmentation, vis_traversability  #vis_segmentation，这是可视化函数，把输入图像和预测分割结果组合显示
with open("config/config.yaml", "r",encoding="utf-8") as f:
    config = yaml.safe_load(f)

image = Image.open(op.join(config["DATA_PATH"], "valid/iid000878/rgb.jpg"))
mask = Image.open(op.join(config["DATA_PATH"], "valid/iid000878/labels.png"))

model = DeepLabWrapper(model_path=config["LOAD_MODEL_PATH"])
# 动态地把输入图像调整大小并裁剪成模型所需的尺寸
image, mask = model.resize_and_crop_input(image, mask)
predicted_mask = model(image)
vis_segmentation(image, np.array(predicted_mask)) #把输入图像和预测结果做可视化显示。
vis_traversability(image, np.array(predicted_mask)) #基于预测mask生成三类风险图和通行安全评分图。
#这个vis_segmentation通常会做：显示原图--显示预测mask--生成overlay叠加图--显示图例legend
