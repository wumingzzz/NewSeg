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
    "#000000",  # unknown - 黑色
    "#D2B48C",  # smooth trail - 浅棕/土黄色（平整道路）
    "#7CFC00",  # traversable gras - 亮绿色（可通行草地）
    "#FF8C00",  # rough trail - 橙色（粗糙道路）
    "#800080",  # non-traversable - 紫色（不可通行）
    "#FF0000",  # obstacle - 红色（障碍物，警示性强）
    "#9ACD32",  # low vegetation - 黄绿色（低矮植被）
    "#006400",  # high vegetation - 深绿色（高植被）
    "#87CEEB",  # sky - 天蓝色
]
#把colors里的十六进制颜色，转换成0-255的RGB整数元组列表
#外层 for h in COLORS 遍历每一个COLORS里的颜色字符串
#to_rgb(h) 把这个颜色字符串转换成Matplotlib的RGB浮点表示，如：红色to_rgb("#FF0000")得到(1.0, 0.0, 0.0)
#int(c * 255) for c in to_rgb(h) 把每个浮点颜色分量都乘以255再转化为整数，如(1.0, 0.0, 0.0)就会变成（255，0，0）
#tuple 把结果变成元组，最终一项颜色就会变成（255，0，0）
#最终RGB_COLORS变成列表如：[(255,0,0),()]这种
RGB_COLORS = [tuple(int(c * 255) for c in to_rgb(h)) for h in COLORS]

# 三类通行风险等级：0-可通行 1-谨慎通行 2-不可通行
RISK_NAMES = [
    "可通行",
    "谨慎通行",
    "不可通行",
]
RISK_COLORS = [
    "#00B050",  # 可通行 - 绿色
    "#FFC000",  # 谨慎通行 - 黄色
    "#FF0000",  # 不可通行 - 红色
]
RGB_RISK_COLORS = [tuple(int(c * 255) for c in to_rgb(h)) for h in RISK_COLORS]


def convert_mask_to_risk(mask: np.ndarray) -> np.ndarray:
    """把9类语义分割mask转换成3类通行风险图

    Args:
        mask: (np.ndarray)
            模型预测得到的语义分割mask，每个像素值为0-8之间的类别编号

    Returns:
        np.ndarray: 风险等级图，0表示可通行，1表示谨慎通行，2表示不可通行
    """
    risk_map = np.full(mask.shape, 2, dtype=np.uint8)  # 默认都先按不可通行处理

    # smooth trail、rough trail 按可通行处理
    risk_map[np.isin(mask, [1, 3])] = 0
    # traversable grass、low vegetation 按谨慎通行处理
    risk_map[np.isin(mask, [2, 6])] = 1
    # unknown、non-traversable、obstacle、high vegetation、sky 保持不可通行

    return risk_map


def calculate_safety_score(mask: np.ndarray) -> dict:
    """根据语义分割结果统计越野场景通行风险占比

    Args:
        mask: (np.ndarray)
            模型预测得到的语义分割mask

    Returns:
        dict: 包含风险等级占比和有效地面像素数量
    """
    risk_map = convert_mask_to_risk(mask)
    valid_area = mask != 8  # sky不参与评分，避免天空面积影响地面通行判断
    valid_pixels = int(valid_area.sum())

    if valid_pixels == 0:
        safe_ratio = 0.0
        caution_ratio = 0.0
        danger_ratio = 0.0
    else:
        safe_ratio = float(((risk_map == 0) & valid_area).sum() / valid_pixels)
        caution_ratio = float(((risk_map == 1) & valid_area).sum() / valid_pixels)
        danger_ratio = float(((risk_map == 2) & valid_area).sum() / valid_pixels)

    return {
        "safe_ratio": safe_ratio,
        "caution_ratio": caution_ratio,
        "danger_ratio": danger_ratio,
        "valid_pixels": valid_pixels,
    }


def colorize_risk_map(risk_map: np.ndarray) -> np.ndarray:
    """把三类风险图转换成彩色图，方便显示和叠加"""
    color_risk_map = np.zeros((risk_map.shape[0], risk_map.shape[1], 3), dtype=np.uint8)
    for idx, rgb in enumerate(RGB_RISK_COLORS):
        color_risk_map[risk_map == idx] = rgb
    return color_risk_map


def vis_traversability(image: np.ndarray, mask: np.ndarray) -> None:
    """可视化越野场景通行风险等级结果，并保存到本地图片

    Args:
        image: (np.ndarray)
            输入模型的RGB图像
        mask: (np.ndarray)
            模型预测得到的语义分割mask
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    image = np.array(image).astype(np.uint8)
    risk_map = convert_mask_to_risk(mask)
    color_risk_map = colorize_risk_map(risk_map)
    score_info = calculate_safety_score(mask)

    overlay = image.astype(np.float32) * 0.55 + color_risk_map.astype(np.float32) * 0.45
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    semantic_cmap = ListedColormap(COLORS[: len(LABEL_NAMES)])
    risk_cmap = ListedColormap(RISK_COLORS)

    plt.figure(figsize=(22, 5))
    grid_spec = gridspec.GridSpec(
        1,
        5,
        width_ratios=[5, 5, 5, 5, 4],
    )

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")

    plt.subplot(grid_spec[1])
    plt.imshow(mask, cmap=semantic_cmap, vmin=0, vmax=len(LABEL_NAMES) - 1)
    plt.axis("off")
    plt.title("Semantic Mask")

    plt.subplot(grid_spec[2])
    plt.imshow(risk_map, cmap=risk_cmap, vmin=0, vmax=len(RISK_NAMES) - 1)
    plt.axis("off")
    plt.title("Risk Map")

    plt.subplot(grid_spec[3])
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Risk Overlay")

    plt.subplot(grid_spec[4])
    plt.axis("off")
    info_text = (
        f"可通行区域: {score_info['safe_ratio'] * 100:.2f}%\n"
        f"谨慎通行区域: {score_info['caution_ratio'] * 100:.2f}%\n"
        f"不可通行区域: {score_info['danger_ratio'] * 100:.2f}%\n"
        f"有效地面像素: {score_info['valid_pixels']}"
    )
    plt.text(0.0, 0.95, info_text, va="top", fontsize=13)

    legend_elements = []
    for risk_name, color in zip(RISK_NAMES, RISK_COLORS):
        legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor=color, label=risk_name))
    plt.legend(handles=legend_elements, loc="lower left", frameon=False, title="风险等级")

    plt.grid("off")
    plt.savefig("traversability_evaluation.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


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
