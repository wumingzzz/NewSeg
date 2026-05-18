"""Wrapper for torchvision DeepLabv3 models
把 torchvision 现成的 DeepLabV3 模型包装成一个更方便训练、加载、保存、预处理和推理的类。
"""

from typing import Optional, Tuple  #导入类型注解工具

import pytorch_lightning as pl
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,   #上面三个是模型的预训练权重
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,   #这些函数用来直接创建 DeepLabV3 模型。
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead #用来替换模型最后的分类头，让输出类别数适配自己的数据集。
from torchvision.transforms import v2 #导入新版 transforms


class DeepLabWrapper(pl.LightningModule):
    """Wrapper for torchvision DeepLabv3 models
    这个类是 torchvision DeepLabV3 模型的封装类"""

    def __init__(
        self,
        backbone: Optional[str] = None,         #主干网络名字
        num_mask_channels: Optional[int] = None, #输出类别数
        model_path: Optional[str] = None,        #模型文件路径
    ):
        """Initializes a DeepLabWrapper instance

        Args:
            backbone: (str, optional)
                选哪个 backbone If None, a model will be initialized with the default backbone.
            num_mask_channels: (int, optional)
                分割输出通道数
            model_path: (str, optional)
                如果给了路径，就从文件加载模型；如果没有，就新建模型
        """
        super().__init__() #父类继承
        self.backbone = backbone
        self.num_mask_channels = num_mask_channels
        self.model_path = model_path       #把传进来的三个参数保存成对象属性。

        self.model = None       #先站位   模型
        self.input_transform = None  #输入变换
        self.transform = None  #正式推理前的预处理变换
        if self.model_path:       #如果输入了model_path 则加载一个训练好的模型
            self.load_model()
        else:
            self.initialize_model()  #没有的话 则按backbone + num_mask_channels 新建一个模型

        self.create_transform()                     #调用函数，创建正式预处理流程，self.transform有了
        self.cuda = torch.cuda.is_available()       #检查当前环境有没有可用GPU
        self.parameters = self.model.parameters()   #取出模型的参数放到self.parameters
        if self.cuda:
            self.model.to("cuda")                  #如果有 GPU，就把模型搬到 GPU 上。

    #用于test等的图像调整函数
    def resize_and_crop_input(
        self, image: Image.Image, mask: Optional[Image.Image] = None
    ) -> Image.Image | Tuple[Image.Image, Image.Image]:
        """把输入图像调整到适合模型的尺寸，如果同时给了 mask，就连 mask 一起同步处理。

        Args:
            image: (PIL.Image)
                要处理的输入图像 进行裁剪和resize
            mask: (PIL.Image, optional)
                mask to resize and crop. If None, only the image is resized and cropped.

        Returns:
            PIL.Image: resized and cropped image (and mask if provided)
        """
        if not self.input_transform:                 #如果没有创建输入transform
            self.create_input_transform(image.size)  #就(调用函数)根据当前输入图像尺寸，动态创建一个

        if mask is not None:                          #如果有mask 返回image和mask 没有就单单返回image
            return self.input_transform(image, mask)
        return self.input_transform(image)

    def load_model(self, eval: bool = True) -> None:
        """从文件里加载模型，用在init

        Args:
            eval: (bool, optional)
                将模型设置为评估模式（evaluation mode），用于推理/预测
        """
        # 从磁盘加载模型，并且表示加载的不只是纯权重，而是完整对象。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#新增，为了能在cpu上验证
        self.model = torch.load(self.model_path,map_location=device, weights_only=False)#新增map_location，为了能在cpu上验证
        self.model.to(device)#新增，为了能在cpu上验证
        if eval:
            self.model.eval()

    def create_input_transform(self, input_shape: Tuple[int, int]) -> None:
        """创建 input transform 用于resize和crop  input images
        根据原始图片大小，生成“缩放 + 中心裁剪”的输入变换。
        Args:
            input_shape: (Tuple[int, int])
                输入图像的尺寸 (width, height)

        """
        # 先把原图缩放到“短边等于 513”，然后按原短边比例缩放长边,再从中间裁成 513×513
        w, h = input_shape
        if w < h:
            new_w = 513
            new_h = int(h * (513 / w))
        else:
            new_h = 513
            new_w = int(w * (513 / h))
        self.input_transform = v2.Compose(
            [
                v2.Resize((new_h, new_w)),
                v2.CenterCrop((513, 513)),
            ]
        )

    def create_transform(self) -> None:
        """创建正式送入网络前的预处理流程，用在init"""
        self.transform = v2.Compose(
            [
                v2.Resize((513, 513)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def save_model(self, model_path: str) -> None:
        """保存模型到所给的路径，用在trainer.py文件中

        Args:
            model_path: (str)
                Path to save the model to
        """
        torch.save(self.model, model_path)  #和前面的load_model()配套

    def initialize_model(self) -> None:
        """Initializes a DeepLabv3 model from the torchvision package 用在init
        根据 backbone 名称，创建对应的 DeepLabV3 模型，并替换最后的分类头
        """
        match self.backbone.lower():#match-case 语法，相当于“按不同 backbone 名字分支处理，lower表示字符串转为小写”
            case "resnet101":
                # 创建相应backbone的deeplabv3，并加载默认的预训练模型
                self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
                #把原模型最后的分类头换掉，换成：输入通道2048(不同backbone不一样），输出通道为输入的num_mask_channels
                self.model.classifier = DeepLabHead(2048, self.num_mask_channels)
            case "resnet50":
                self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
                self.model.classifier = DeepLabHead(2048, self.num_mask_channels)
            case "mobilenetv3large":
                self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
                self.model.classifier = DeepLabHead(960, self.num_mask_channels)
            case _:
                raise ValueError(
                    "Unknown backbone selected in configuration. Please select from RESNET50, RESNET101, or MOBILENETV3LARGE"
                )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses input into format required for processing （预处理输入，成模型所需格式）
        把输入图片变成模型真正能输入的张量
        """
        # apply the same transforms that were applied to input images when training the model (training-serving skew)
        # 推理时也要尽量使用和训练时一致的输入变换，避免训练和部署之间的数据分布不一致
        input_tensor: torch.Tensor = self.transform(image)   #对输入的图片进行create_transform() 定义的预处理
        # 因为单张图像本来是：[C,H,W] 模型需要batch格式：[N,C,H,W],执行后变成[1,3,513,513],
        # 就是把一张图伪装成 batch size=1 的一批数据。
        input_batch: torch.Tensor = input_tensor.unsqueeze(0)
        if self.cuda:
            input_batch = input_batch.to("cuda")  #如果有GPU,就把输入张量也搬到 GPU。
        return input_batch

    def forward(self, image: Image.Image) -> Image.Image:
        """Processes input through a DeepLabv3 model
        输入一张 PIL 图片，输出一张预测类别图。
        """
        input_batch = self.preprocess(image) #先对输入图片做预处理，变成模型能接受的 batch 张量。
        with torch.no_grad():   #关闭梯度，表示下面推理时不计算梯度
            # DeepLabV3 的输出通常是一个字典，里面常见有："out"：主输出,有些模型还可能有 "aux"：辅助输出
            # 这里只取out的0，如果模型输出形状是：[1, num_classes, 513, 513]，那取 [0] 之后就是：[num_classes, 513, 513]
            output: torch.Tensor = self.model(input_batch)["out"][0]
        # 得到得到最终类别预测
        # 这里在类别维度上取最大值下标，对0维做argmax，得到每个像素最可能属于哪个类别
        # [num_classes,513,513] 如果num_classes是1，就是像素点为1的概率--->[513,513]里面的值是类别值（取得该像素点概率最大的num_classes值）
        output_predictions = output.argmax(0)
        #最后把预测结果转换成 PIL 图片返回。这里分几步：.byte()：转成 8 位整数，.cpu()：从 GPU 搬回 CPU，
        #.numpy()：转成 NumPy 数组，Image.fromarray(...)：转成 PIL 图片
        return Image.fromarray(output_predictions.byte().cpu().numpy())
