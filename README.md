# Semantic segmentation of off-road images using transfer learning and DeepLabv3+


## Abstract:
许多最前沿的深度学习算法既需要大规模训练数据集，又需要强大的计算能力，而用户往往因种种原因无法同时具备这两项条件。从零开始训练大型网络对终端用户而言既繁琐又耗时。迁移学习是一种机器学习方法，通过将知识从一个领域迁移到另一个领域，最终消除了使用随机初始化网络和基准数据集从零开始训练的必要性。鉴于训练神经网络架构需要消耗海量的计算与时间资源，迁移学习已成为计算机视觉领域广受欢迎的技术。本文将探究迁移学习对应用于语义分割任务的大型编码器-解码器结构深度神经网络的影响。DeepLabv3+正是这类架构之一，在2018年发布时代表了当时最先进水平。通过将深度可分离卷积应用于空洞空间金字塔池化模块和解码器模块，DeepLabv3+成功整合了2016年Xception模型的相关技术，构建出更快、更强大且规模更大的网络。我们提出将预训练的DeepLabv3+模型拓展应用至具有挑战性的非道路场景感知任务中。借助发布的"Yamaha-CMU非道路数据集"，我们成功将迁移学习技术应用于预训练模型，实现了非道路图像的语义分割任务。  



## Installation:

### Setting up the Repository: 

按照以下步骤设置代码仓库，即可进行训练和推理。操作过程中会创建一个 `data` 目录，用于存放训练数据。第3步和第4步会将 *[Yamaha-CMU 非道路数据集](https://theairlab.org/yamaha-offroad-dataset/)* 下载并解压至该 `data` 目录。

1. `git clone https://github.com/nmhaddad/semantic-segmentation.git`
1. `mkdir data && cd data`
1. `wget https://cmu.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt`
1. `unzip yamaha_v0.zip`

### Installing Dependencies
This project was setup and tested using Python 3.12. The simpliest way to get this repo up and running is to use `uv` to create a virtual environment, then install dependencies using the `requirements.txt` file:

`uv pip install -r requirements.txt`

## Models

[Pretrained Models](https://drive.google.com/drive/folders/1Gmk8vOF9qBNMg3-TEL-st6KWieB4Af5e?usp=sharing)

## Running:

您可以通过 `config/config.yaml` 文件配置训练环境。该 YAML 文件包含用于训练和测试模型的各种超参数及路径设置，旨在简化实验操作。

代码仓库中提供了 `training_demo.ipynb` 和 `inference_demo.ipynb` 两个笔记本，可方便地运行仓库中的代码。只需确保在训练前已下载 *[Yamaha-CMU 非道路数据集](https://theairlab.org/yamaha-offroad-dataset/)* 即可。

运行 `python train.py` 可启动独立训练会话。

运行 `python test.py` 可对单张图像进行独立推理会话。

运行 `python process_video.py` 可对视频进行独立推理会话。

## References:

[1] Chen, Liang-Chieh, Zhu, Yukun, Papandreou, George, Schroff, Florian, and Adam, Hartwig. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.” Computer Vision – ECCV2018 (2018): 833-51. Web.  

[2] Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 1800-807. Web.  

[3]  Daniel Maturana and Po-Wei Chou and Masashi Uenoyama and Sebastian Scherer, “Real-time Semantic Mapping for Autonomous Off-Road Navigation” in Maturana-2017-102768, September 2017, pp. 335 - 350.  

[4]  Stevo. Bozinovski  and  Ante  Fulgosi  (1976).  "The  influence of pattern similarity and transfer learning upon the training of a base perceptronB2.” (original in  Croatian) Proceedings of Symposium Informatica 3-121-5, Bled.  

[5] Stevo Bozinovski (2020) "Reminder of the first paper on transfer learning in neural networks, 1976”. Informatica 44: 291–302.  

[6] Pan, S.J.; Yang, Q. A survey on transfer learning. IEEE Trans. Knowl. Data Eng. 2010, 22, 1345–1359  

[7] M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLabv3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.





# 仓库代码结构梳理

## **1.配置文件 `config/.yaml`**

 → 告诉程序“数据在哪、模型权重在哪、训练参数是多少

#### 1）`config/config.yaml`

这是**训练和单图推理**共用的主配置文件。当前内容里主要有：

- `DATA_PATH`：数据路径
- `LOAD_MODEL_PATH`：加载模型路径
- `CLASS_WEIGHTS`：类别权重
- `CROP_SIZE`：513
- `NUM_EPOCHS`：50
- `NUM_MASK_CHANNELS`：9
- `BATCH_SIZE`：16
- `LEARNING_RATE`：1e-5
- `BACKBONE`：`mobilenetv3large`，并注明还可选 `resnet50`、`resnet101`。

#### 2）`config/video_config.yaml`

这个是专门给视频推理用的，只有三个关键参数：

- `DATA_PATH`：输入视频路径
- `LOAD_MODEL_PATH`：要用的模型
- `SAVE_VIDEO`：输出视频路径。

所以以后跑视频时，主要改这个文件就行。





## **2.模型封装 `models`**

 → 负责创建 DeepLabV3+、切换骨干网络、替换分类头、加载/保存模型、做输入预处理和前向推理。

`models/` 目录里只有两个文件：`__init__.py` 和 `deeplab_wrapper.py`。

#### 1）`models/__init__.py`

这个文件很简单，它只是把 `DeepLabWrapper` 导出来，方便你在别处直接写：

```
from models import DeepLabWrapper
```

而不用写完整路径。

#### 2）`models/deeplab_wrapper.py`

这是整个仓库最核心的文件之一。它不是自己从头实现 DeepLab，而是**封装了 torchvision 自带的 DeepLabV3 模型**。代码里支持三种骨干网络：

- `mobilenetv3large`
- `resnet50`
- `resnet101` 

它主要做了几件事：

**第一，初始化模型。**
 如果你传入 `backbone` 和 `num_mask_channels`，它就用 torchvision 的预训练权重初始化 DeepLabV3，然后把分类头替换成适合你自己数据集类别数的 `DeepLabHead`。比如 `mobilenetv3large` 用 960 通道输出接分类头，`resnet50/101` 用 2048 通道。

**第二，加载已有模型。**
 如果传了 `model_path`，它就直接 `torch.load()` 已训练好的模型，并切到 `eval()` 模式用于推理。

**第三，做输入预处理。**
 它有两套变换：

- 一套是 `create_input_transform`，把输入按短边缩放到 513，再中心裁剪到 `513×513`；
- 另一套是 `create_transform`，把图像 resize 到 `513×513`、转 tensor、归一化。

**第四，定义前向推理。**
 `forward()` 会把输入图像预处理后送入模型，取输出 `["out"]`，再对通道做 `argmax`，最后返回一个预测类别图的 `PIL.Image`。

你可以把这个文件理解成：**“模型管理员”**。所有和模型有关的事情，几乎都在这里统一处理。

## **3.`utils/` 文件夹：数据、训练、可视化工具**



`utils/` 下有 4 个文件：`__init__.py`、`dataset.py`、`trainer.py`、`utils.py`。

#### 1）`utils/__init__.py`

它把常用工具统一导出，包括：

- `get_dataloaders`
- `Trainer`
- `vis_segmentation`
- `display_example_pair`
- `overlay_mask_cv2` 

这让其他脚本可以直接：

```
from utils import Trainer, get_dataloaders
```

#### 2）`utils/dataset.py`

这是**数据集读取文件**。它定义了一个 `YamahaCMUDataset` 类，继承自 `VisionDataset`。

它的逻辑很重要：

**数据组织假设**
 它会在 `root` 下找一堆子文件夹，然后每个子文件夹里找一个 `*.jpg` 和一个 `*.png`，分别当图像和标签。也就是说，作者默认 Yamaha-CMU 的样本组织方式是“每个样本一个文件夹，里面有 RGB 图和标签图”。

**读取方式**

- 图像：`Image.open(...).convert("RGB")`
- 掩膜：读成 numpy 数组
   如果 mask 是三维，就只取第一个通道；再转换成 `tv_tensors.Mask(dtype=torch.long)`。

这说明作者把标签当作**类别索引图**来处理，而不是彩色 RGB 掩膜。

**数据增强**
 `get_dataloaders()` 里用了这些变换：

- `ColorJitter`
- `ToImage`
- `ToDtype(torch.float32, scale=True)`
- `RandomCrop(513)`
- `RandomHorizontalFlip`
- `Normalize(mean/std)` 

然后它构建 `train` 和 `valid` 两个 dataloader，`shuffle=True`，`num_workers=4`。

这个文件你要重点理解，因为它决定了**你的数据到底怎么被喂给网络**。



#### 3）`utils/trainer.py`

这是训练循环。它定义了 `Trainer` 类，负责真正训练。

它的核心逻辑是：

- 自动判断设备：`cuda` 或 `cpu`
- 每个 epoch 分 `train` 和 `valid` 两个阶段
- 计算 loss
- 统计 `MeanIoU` 和 `GeneralizedDiceScore`
- 记录最优验证集 mIoU
- 最后把最优权重保存下来。

优化器是在外部传入的，但从 `train.py` 看，用的是 `AdamW`。

值得注意的是，这里还用了：

- `GradScaler`
- `autocast`
- `torchmetrics.segmentation.MeanIoU`
- `torchmetrics.segmentation.GeneralizedDiceScore` 

也就是说，作者做了一个相对规范的小型训练器，而不是把训练代码全塞进 `train.py`。

#### 4）`utils/utils.py`

这个文件是**可视化与辅助函数**合集。

里面最关键的是三部分：

**类别名与颜色表**
 它定义了 9 个类别名：

- unknown
- non-traversable
- rough trail
- smooth trail
- traversable grass
- low vegetation
- obstacle
- high vegetation
- sky 

同时还给每个类别配了颜色。这个信息很重要，因为它告诉你这个项目不是简单二分类“路/非路”，而是**9 类语义分割**。

**`vis_segmentation()`**
 这个函数会画四部分：

1. 输入图像
2. 预测 mask
3. overlay 叠加图
4. 图例 legend
    最后保存成 `segmentation_visualization.png`。

**`overlay_mask_cv2()`**
 这个函数把 mask 映射成彩色，再和原图做 alpha 混合，返回适合视频写出的 BGR 图像；它还会额外输出 `debug_image.png` 和 `debug_mask.png` 便于调试。

**`display_example_pair()`**
 只负责简单显示原图和标签图，主要给 notebook 演示用。







## **4.入口脚本**

#### 1）`train.py`

这是训练入口。它做的事很直白：

1. 读 `config/config.yaml`
2. 创建 `runs/` 输出目录
3. 初始化 `wandb` 实验记录
4. 用 `get_dataloaders()` 加载数据
5. 用 `DeepLabWrapper` 创建模型
6. 构造带类别权重的 `CrossEntropyLoss`
7. 用 `AdamW` 优化器
8. 构建 `Trainer` 并执行 `trainer.train()`。

所以 `train.py` 更像是一个“装配脚本”，真正的细节分别在 `config/`、`models/`、`utils/` 里。



#### 2）`test.py`

这是单图推理入口。它会：

1. 读 `config/config.yaml`
2. 默认打开一个示例图 `train/iid000008/rgb.jpg`
3. 同时读对应标签 `labels.png`
4. 用 `DeepLabWrapper(model_path=...)` 加载模型
5. 调用 `resize_and_crop_input()` 做尺寸处理
6. 预测 mask
7. 用 `vis_segmentation()` 可视化。

这说明 `test.py` 不是那种“带命令行参数的通用测试脚本”，而是一个**示例式推理脚本**。你以后很可能要把它改成“输入你自己的图片路径”。

#### 3）`process_video.py`

这是视频推理入口。它会：

1. 用 `cv2.VideoCapture` 打开视频
2. 创建一个 `513×513`、30fps 的 `mp4v` 输出视频
3. 逐帧读取
4. 把 BGR 帧转成 RGB，再转成 PIL 图
5. 调用模型推理
6. 用 `overlay_mask_cv2()` 叠加彩色 mask
7. 写回输出视频。

你可以把它理解成“把 `test.py` 的单图流程放进 while 循环里”。





## 6.`runs/`

仓库根目录里有 `runs/` 文件夹。`train.py` 会自动创建这个目录，训练器默认也会把模型保存到这里，例如 `runs/{backbone}_v1.{num_epochs}.pt`。

所以 `runs/` 的作用就是：**存放训练结果和模型权重。**
 它不是算法文件夹，而是输出文件夹。

