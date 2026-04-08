"""包含多种辅助函数的工具文件"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches #matplotlib的两个工具 gridspec用来更灵活的子图布局 patches用来画图形块
#导入两个颜色相关工具，ListedColormap把一组固定颜色定义成一个颜色映射表。
#to_rgb 把颜色字符串转换成RGB浮点数格式
from matplotlib.colors import ListedColormap, to_rgb

#列表 定义了这个数据集里每个类别对应的名字 0-未知 1-平整道路 2--可通行草地 3-粗糙道路 4-不可通行 5-障碍物 6-低矮植被 7-高植被 8-天空
#可以知道mask的各个数字代表什么
LABEL_NAMES = [
    "unknown",
    "smooth trail",
    "traversable gras",
    "rough trail",
    "non-traversable",
    "obstacle",
    "low vegetation",
    "high vegetation",
    "sky",
]
COLORS = [
    "#000000",  # unknown - black 黑色
    "#8B4513",  # smooth trail - brown 棕色
    "#D2691E",  # traversable gras - chocolate 巧克力色
    "#F4A460",  # rough trail - sandy brown 沙棕色
    "#90EE90",  # non-traversable - light green 浅绿色
    "#228B22",  # obstacle - forest  森林绿
    "#FF0000",  # low vegetation - red 红色
    "#006400",  # high vegetation - dark green 深绿色
    "#87CEEB",  # sky - sky blue 天蓝色
]
#把colors里的十六进制颜色，转换成0-255的RGB整数元组列表
#外层 for h in COLORS 遍历每一个COLORS里的颜色字符串
#to_rgb(h) 把这个颜色字符串转换成Matplotlib的RGB浮点表示，如：红色to_rgb("#FF0000")得到(1.0, 0.0, 0.0)
#int(c * 255) for c in to_rgb(h) 把每个浮点颜色分量都乘以255再转化为整数，如(1.0, 0.0, 0.0)就会变成（255，0，0）
#tuple 把结果变成元组，最终一项颜色就会变成（255，0，0）
#最终RGB_COLORS变成列表如：[(255,0,0),()]这种
RGB_COLORS = [tuple(int(c * 255) for c in to_rgb(h)) for h in COLORS]


def vis_segmentation(image: np.ndarray, mask: np.ndarray) -> None:
    """把输入图像、分割 mask、叠加结果和图例一起可视化

    Args:
        image: (np.ndarray)
            the rgb image
        mask: (np.ndarray)
            the mask of the input image
    """
    # 创建一个离散颜色映射表，取和类别数量一样多的颜色
    #ListedColormap(...)把这些颜色包装成一个颜色映射对象，后面显示 mask 时就能自动用0映射到第1种颜色，1映射到第2种颜色，无渐变插值
    cmap = ListedColormap(COLORS[: len(LABEL_NAMES)])

    plt.figure(figsize=(20, 5)) #创建20*5的图像窗口
    #定义额1行4列的网络布局，前三个区域宽度是6，，按比例来的
    grid_spec = gridspec.GridSpec(
        1,
        4,
        width_ratios=[6, 6, 6, 4],
    )

    # input image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    # mask
    plt.subplot(grid_spec[1])
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(LABEL_NAMES) - 1)#显示 mask，并指定颜色映射。 vmin=0 max表示最小最大类别编号
    plt.axis("off")
    plt.title("Mask")
    # overlay 叠加图
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(LABEL_NAMES) - 1, alpha=0.5)#alpha为透明度 0.5
    plt.axis("off")
    plt.title("Mask Overlay")
    # legend 图例
    legend_elements = [] #创建图例列表
    #循环构造图例项，同时遍历类别名字和类别颜色，并且 enumerate 还会给出类别编号i，
    #zip(...) 同时遍历两个列表 打包成对 如：("unknown","000000"),  enumerate(...) 添加索引
    # 所以每一轮会拿到i,label,color,例如：某一轮可能是：i=6,label = "obstacle" color="#FF0000"
    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS[: len(LABEL_NAMES)])):
        #为当前类别创建一个颜色方块，并加上标签文字，然后加入图例列表。
        #patches.Rectangle((0, 0), 1, 1, ...)在（0，0）创建一个1*1的矩形块，填充颜色为该类别的颜色，label为图例文字
        legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor=color, label=f"{i}: {label}"))
    plt.subplot(grid_spec[3]) #选择第四个子图区域
    plt.legend( #handles：显示图例,   loc：图例位置,   frameon：边框,   title：标题，    后两个为字体大小
        handles=legend_elements, loc="center", frameon=False, title="Legend", title_fontsize="large", fontsize="large"
    )
    plt.axis("off")

    plt.grid("off")
    # 把整张可视化结果图保存成png文件，bbox_inches="tight"表示尽量裁掉多余空白边缘。pad_inches=0.1表示给图像边缘保留一点点留白
    plt.savefig("segmentation_visualization.png", bbox_inches="tight", pad_inches=0.1)


def overlay_mask_cv2(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """用 OpenCV 的方式，把分割 mask 叠加到图像上，返回叠加后的图像。更偏 Matplotlib 展示，这个函数主要给 process_video.py 用。

    Args:
        image: (np.ndarray)
            the rgb image
        mask: (np.ndarray)
            the mask to overlay

    Returns:
        np.ndarray: the image with the mask overlayed
    """

    # 创建彩色mask图
    color_mask = np.zeros_like(image, dtype=np.uint8) #建了一个和原图大小一样的全黑空白图像。
    for idx, rgb in enumerate(RGB_COLORS):
        # 遍历每一个类别，把 mask 中属于该类别的像素位置，填成对应颜色。
        #mask == idx 会生成布尔矩阵，即矩阵内是True和False，如果是True的话就会把对应的地方涂成相应的颜色
        color_mask[mask == idx] = rgb

    # 保证输入是  uint8
    image = image.astype(np.uint8)

    # RGB转换成BGR 为了 OpenCV 处理
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # write the image and mask to debug
    cv2.imwrite("debug_image.png", image)
    cv2.imwrite("debug_mask.png", color_mask)

    # 加权融合 a*w1+b*w2+gamma
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def display_example_pair(image: np.ndarray, mask: np.ndarray) -> None:
    """可视化输入图像和分割图，用于展示
    并排显示一张原图和它的 mask。
    Args:
        image: (np.ndarray) the rgb image
        mask: (np.ndarray) the mask of the input image
    """
    _, ax = plt.subplots(1, 2, figsize=(15, 15)) #创建一个1行2列画布，_是整个figure,ax是两个子图对象组成的数组
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Original Image")
    ax[1].imshow(mask)
    ax[1].axis("off")
    ax[1].set_title("Mask")
