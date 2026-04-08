#怎么一轮一轮训练、怎么验证、怎么算指标、怎么保存最好模型。
"""这个文件是用来训练 DeepLab 模型的类"""

import copy   #深拷贝一份模型权重，用来保存最优模型参数
import time   #时间模块，用来记录训练的各种时间
from typing import Tuple #导入类型注解

import torch

from torch.amp import GradScaler, autocast #混合精度训练相关工具
#导入one_hot函数，把类别编号图转成 one-hot 格式，比如原来一个像素类别是3，one-hot后变成[0, 0, 0, 1, 0, 0, 0, 0, 0]
from torch.nn.functional import one_hot
#导入两个语义分割评价指标 MeanIoU：平均交并比 GeneralizedDiceScore：广义 Dice 分数。
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from tqdm import tqdm   #导入进度条工具，这样训练时每个 batch 的处理进度会显示得更直观。

from models import DeepLabWrapper #导入之前写的 DeepLabWrapper


class Trainer:
    """这是一个根据给定超参数配置来训练 DeepLab 模型的类
    它不负责定义模型结构，也不负责读取原始图片，而是负责：接收模型、接收数据、接收损失函数和优化器、组织整个训练流程"""

    def __init__(
        self,
        deeplab: DeepLabWrapper,
        dataloaders: torch.utils.data.DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        num_epochs: int = 25,
        logger=None,
        save_model_path: str = None,
    ):
        """Initialization method for Trainer base class

        Args:
            deeplab: (DeepLabWrapper)
                要训练的模型对象，是封装过的DeepLabWrapper
            dataloaders: (torch.utils.data.DataLoader)
                Dataloaders for training and validation
            criterion: (torch.nn.CrossEntropyLoss)
                损失函数，交叉熵损失
            optimizer: (torch.optim.Adam)
                优化器 to use for training
            num_epochs: (int, optional)
                训练轮次 默认25
            logger: (optional)
                日志记录器
            save_model_path: (str, optional)
                Path to save the trained model 保存路径
        """
        self.deeplab = deeplab
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logger = logger
        self.save_model_path = save_model_path #都是把传进来的参数保存成对象属性，这样类的函数能用使用

    def train(self) -> Tuple[DeepLabWrapper, list]:
        """ 用来训练模型，并返回训练后的模型和每一轮验证集 mIoU 组成的列表

        Returns:
            model, val_mean_iou_history
        """
        since = time.time() #记录训练开始时间
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #决定训练设备

        val_mean_iou_history = [] #创建一个空列表，后面用来保存每一轮验证集的miou
        #先把当前模型参数完整复制一份，作为“目前最优模型权重”的初始值,保存的是参数字典state_dict。
        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
        best_mean_iou = 0.0 #当前最好的验证 mIoU，初始设为 0
        self.deeplab.model.to(device) #把模型搬到刚才选定的设备上。

        scaler = GradScaler()  #创建梯度缩放器，用于混合精度训练。

        # 开始epoch循环训练
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}") #输出如：epoch 1/25
            print("-" * 10)                     #打印10个-

            for phase in ["train", "valid"]: #每个epoch都分为两个阶段，分别为train：训练，valid：验证
                if phase == "train":
                    self.deeplab.model.train()   #模型切换到训练模式
                else:
                    self.deeplab.model.eval()    #模型切换到评估模式
                #每个epoch都重新创建moiu指标对象
                mean_iou = MeanIoU(
                    num_classes=self.deeplab.num_mask_channels,#类别数等于模型输出类别数
                    include_background=False, #计算时不把背景算进去
                ).to(device)
                #创建Generalized Dice Score 指标对象，内容同上
                gds = GeneralizedDiceScore(
                    num_classes=self.deeplab.num_mask_channels,
                    include_background=False,
                ).to(device)
                running_loss = 0.0  #每轮epoch要初始化loss

                # Iterate over data.     batch级循环
                for inputs, labels in tqdm(iter(self.dataloaders[phase])): #tqdm用于显示进度条
                    inputs = inputs.to(device)  #把输入图像搬到设备上
                    labels = labels.to(device)  #把输入标签搬到设备上
                    # 清空优化器里上一批次残留的梯度  一定要做的常规操作
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    #如果当前是训练阶段，开启梯度计算
                    with torch.set_grad_enabled(phase == "train"):
                        # 进入自动混合精度环境，device用cuda，dtype用float16，得到模型outputs 和 calculate loss
                        with autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.deeplab.model(inputs)
                            # 对于 DeepLabV3，这个 outputs 通常是一个字典，里面最重要的是outputs["out"]
                            #他的形状一般是[batch_size, num_classes, H, W]，也就是每个像素对每个类别的预测分数。
                            loss = self.criterion(outputs["out"], labels)  #计算损失
                        #把模型输出的“每类分数图”变成“最终类别图”。
                        #就是[batch_size, num_classes, H, W]-->[batch_size, H, W],像素点里面是概率最大的类别
                        #dim=1 因为0维是batchsize  1维才是num_classes
                        preds = torch.argmax(outputs["out"], dim=1)
                        # 反向传播 + 优化 only if in training phase
                        if phase == "train":
                            scaler.scale(loss).backward() #先把 loss 缩放，再反向传播，这是混合精度训练的标准写法，防止梯度过小。
                            scaler.step(self.optimizer) #执行优化器更新参数，相当于普通训练里的：optimizer.step()
                            scaler.update()       #更新缩放因子，以便后续 batch 继续稳定训练。

                    # 统计loss和指标
                    #loss.item() 是当前 batch 的平均（因为是交叉熵 默认返回平均）损失，inputs.size(0) 是当前 batch 的样本数。
                    running_loss += loss.item() * inputs.size(0) #每batch加上这个batch的总损失，得到一个累计损失
                    mean_iou.update(preds, labels) #用当前 batch 的预测和标签更新 mIoU 统计器。
                    gds.update(  #更新 Dice 分数统计器，不是直接传进preds和labels，而是先做了one_hot
                        #把预测类别图转成 one-hot 格式，原本 preds 形状是：[N, H, W]
                        #one-hot 后会变成：[N,H,W,C]然后后面的.permute(0, 3, 1, 2)转换为 [N,C,H,W]
                        one_hot(
                            preds,
                            num_classes=self.deeplab.num_mask_channels,
                        ).permute(0, 3, 1, 2),
                        one_hot(
                            labels,
                            num_classes=self.deeplab.num_mask_channels,
                        ).permute(0, 3, 1, 2),
                    )
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)#计算当前阶段的平均 loss，分母是数据集总样本数
                epoch_mean_iou = mean_iou.compute().item()#计算整个阶段的 mIoU，并转成 Python 标量。
                epoch_gds = gds.compute().item() #计算整个阶段的 Generalized Dice Score。
                #记录日志
                if self.logger:
                    self.logger.log(
                        {
                            f"{phase}_loss": epoch_loss,
                            f"{phase}_mean_iou": epoch_mean_iou,
                            f"{phase}_gds": epoch_gds,
                            "epoch": epoch + 1,
                        }
                    )
                #打印当前结果，格式如：train Loss: 0.5421 mIoU: 0.4862 GDS: 0.6023
                print(f"{phase} Loss: {epoch_loss:.4f} mIoU: {epoch_mean_iou:.4f} GDS: {epoch_gds:.4f}")
                #只有验证阶段的 mIoU 才加入历史列表。因为真正用来判断泛化能力的是验证集。
                if phase == "valid":
                    val_mean_iou_history.append(epoch_mean_iou)

                    if epoch_mean_iou > best_mean_iou:  #如果当前验证集 mIoU 比历史最好值还高
                        best_mean_iou = epoch_mean_iou  #就更新“最佳 mIoU”
                        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())#把当前模型权重深拷贝保存下来
            print()

        time_elapsed = time.time() - since #计算训练总耗时
        print(f"训练完成总时间为 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"最佳验证 mIoU。: {best_mean_iou:4f}")

        # 加载成验证集表现最好那一轮的权重
        self.deeplab.model.load_state_dict(best_model_wts)

        # 保存模型
        model_path = self.save_model_path or f"runs/{self.deeplab.backbone}_v1.{self.num_epochs}.pt"
        self.deeplab.save_model(model_path)

        if self.logger:
            self.logger.finish() #结束日志

        return self.deeplab, val_mean_iou_history
