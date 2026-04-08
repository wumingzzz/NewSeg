"""This file is used to process and save video data.
逐帧读取视频 → 每帧做语义分割 → 叠加结果 → 写回新视频"""

import cv2
import numpy as np
import yaml
from PIL import Image

from models import DeepLabWrapper
from utils import overlay_mask_cv2


def process_video(model_path: str, data_path: str, outfile_path: str) -> None:
    """使用指定模型，对指定视频做处理，并把结果保存到指定输出路径。

    Args:
        model_path: (str)
            要使用的模型文件路径
        data_path: (str)
            要使用的模型文件路径
        outfile_path: (str)
            处理后视频的保存路径

    Returns:
        None
    """
    video = cv2.VideoCapture(data_path) #用 OpenCV 打开输入视频文件。
    #准备一个新视频文件，把后续处理好的每一帧写进去，cv2.VideoWriter_fourcc(*"mp4v")是视频编码格式，表示使用MP4v编码
    #fps为30 尺寸为（513，513）
    out = cv2.VideoWriter(outfile_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (513, 513))
    model = DeepLabWrapper(model_path=model_path)
    #不断读取视频帧，直到视频结束。
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # 对输入帧做 resize 和 crop，并返回用于构建输出视频的帧
        #把 OpenCV 读出来的一帧，从 BGR 数组转换成 RGB，并进一步转换成 PIL 图像
        pil_frame = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        resized_frame = model.resize_and_crop_input(pil_frame)#对这帧做resize和crop
        predicted_mask = model(resized_frame)#预测
        masked_image = overlay_mask_cv2(np.array(resized_frame), np.array(predicted_mask))#把当前帧和预测的 mask 叠加成一张彩色结果图。
        out.write(masked_image) #把当前处理后的叠加帧写进输出视频文件。

    video.release() #释放
    out.release()
    cv2.destroyAllWindows()#关窗口


if __name__ == "__main__":
    with open("config/video_config.yaml", "r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    process_video(config["LOAD_MODEL_PATH"], config["DATA_PATH"], config["SAVE_VIDEO"])
