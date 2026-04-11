from __future__ import annotations

"""
设计目标：
1. 不改动你原有的 train.py / test.py / process_video.py 训练与推理主流程。
2. 单独新增一个 gradio_app.py，方便直接做课程设计 / 毕业答辩演示。
3. 中文注释尽量详细，代码结构尽量简单，让你后面能自己继续改。
4. 尽量兼容你仓库现在的目录结构：config / models / utils / runs。

建议把本文件放在项目根目录下运行：
    python gradio_app.py
"""


from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image


# ============================================================
# 一、导入你项目里已经写好的模块
# ============================================================
# 这里用 try / except，是为了兼容两种常见情况：
# 1）你把 gradio_app.py 放在项目根目录，且 models 是一个包
# 2）你后面调整了 models 的导出方式
try:
    from models import DeepLabWrapper
except Exception:
    from models.deeplab_wrapper import DeepLabWrapper


# 你的类别名称、颜色，最好直接复用项目里已有定义，避免界面和训练结果不一致。
# 同样做两种导入兼容。
try:
    from utils import LABEL_NAMES, RGB_COLORS
except Exception:
    from utils.utils import LABEL_NAMES, RGB_COLORS


# ============================================================
# 二、全局缓存：避免每点一次按钮都重新加载模型
# ============================================================
# 键：模型绝对路径
# 值：已经加载好的 DeepLabWrapper 对象
MODEL_CACHE: Dict[str, DeepLabWrapper] = {}


# ============================================================
# 三、工具函数
# ============================================================
def resolve_model_path(model_path: str) -> Path:
    """
    把用户在界面中输入的模型路径，转换成绝对路径并检查是否存在。

    参数：
        model_path: 用户输入的模型路径，例如 runs/mobilenetv3large_v1.50.pt

    返回：
        Path 对象（绝对路径）
    """
    if not model_path or not model_path.strip():
        raise gr.Error("请先填写模型路径，例如：runs/mobilenetv3large_v1.50.pt")

    path = Path(model_path.strip())

    # 如果用户输入的是相对路径，就默认相对于“当前运行目录”解析。
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise gr.Error(f"模型文件不存在：{path}")

    return path


def load_model_cached(model_path: str) -> DeepLabWrapper:
    """
    加载模型，并使用缓存机制。

    为什么要缓存：
    - 你的模型比较大，每次点击按钮都重新加载会很慢。
    - 第一次加载后，后面继续推理可以直接复用。
    """
    resolved_path = str(resolve_model_path(model_path))

    if resolved_path not in MODEL_CACHE:
        MODEL_CACHE[resolved_path] = DeepLabWrapper(model_path=resolved_path)

    return MODEL_CACHE[resolved_path]


def clear_model_cache() -> str:
    """
    清空模型缓存。

    使用场景：
    - 你更换了模型文件
    - 你想释放内存
    - 你担心旧模型还留在缓存里
    """
    MODEL_CACHE.clear()
    return "模型缓存已清空。下次推理时会重新加载模型。"


def convert_mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    把类别编号图（二维 mask）转换成彩色图，方便显示。

    输入：
        mask.shape = [H, W]
        mask 里的每个像素值是类别编号，如 0、1、2、...、8

    输出：
        color_mask.shape = [H, W, 3]
        每个类别会被映射成预定义颜色
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, rgb in enumerate(RGB_COLORS[: len(LABEL_NAMES)]):
        color_mask[mask == class_id] = rgb

    return color_mask


def blend_image_and_mask(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    把原图和分割彩色图按一定透明度叠加。

    参数：
        image_rgb: 原始 RGB 图像
        color_mask: 彩色分割图
        alpha: 掩膜透明度，范围 0~1

    返回：
        overlay: 叠加后的图像
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 为了保证计算稳定，这里先转成 float32 做加权，再转回 uint8。
    image_float = image_rgb.astype(np.float32)
    mask_float = color_mask.astype(np.float32)

    overlay = image_float * (1.0 - alpha) + mask_float * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_legend_html() -> str:
    """
    生成网页中展示的“类别-颜色图例”。
    """
    rows = []
    for class_id, (name, rgb) in enumerate(zip(LABEL_NAMES, RGB_COLORS)):
        color_str = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        row_html = f"""
        <div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>
            <div style='width:18px;height:18px;border-radius:4px;background:{color_str};border:1px solid #888;'></div>
            <div><b>{class_id}</b> - {name}</div>
        </div>
        """
        rows.append(row_html)

    return "\n".join(rows)


def summarize_mask(mask: np.ndarray) -> List[List[str]]:
    """
    统计当前分割结果中，各类别像素占比。

    不仅能展示“看起来分割了”，还可以展示“分割结果的定量统计”。

    返回格式：
        [
            [类别ID, 类别名称, 像素数, 占比],
            ...
        ]
    """
    total_pixels = int(mask.size)
    unique_values = np.unique(mask)
    rows: List[List[str]] = []

    for class_id in unique_values:
        class_id = int(class_id)
        class_name = LABEL_NAMES[class_id] if 0 <= class_id < len(LABEL_NAMES) else f"class_{class_id}"
        pixel_count = int((mask == class_id).sum())
        ratio = 100.0 * pixel_count / max(total_pixels, 1)
        rows.append([str(class_id), class_name, str(pixel_count), f"{ratio:.2f}%"])

    # 按占比从大到小排序，展示更直观。
    rows.sort(key=lambda x: float(x[3].replace("%", "")), reverse=True)
    return rows


# ============================================================
# 四、图片推理函数
# ============================================================
def infer_single_image(model_path: str, input_image: np.ndarray, alpha: float):
    """
    对单张图片做语义分割。

    这是“图片检测”页面点击按钮后真正执行的核心函数。

    输入：
        model_path: 模型路径
        input_image: Gradio 上传的图片（numpy 数组，RGB）
        alpha: 叠加透明度

    输出：
        1. 模型输入图（resize + crop 之后）
        2. 彩色分割图
        3. 原图与分割叠加图
        4. 统计表
        5. 状态文字
    """
    if input_image is None:
        raise gr.Error("请先上传一张图片。")

    # 1）加载模型（带缓存）
    model = load_model_cached(model_path)

    # 2）Gradio 传进来的图片一般已经是 RGB numpy 数组，这里转成 PIL 方便复用你项目里的预处理流程。
    pil_image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")

    # 3）调用你项目中的 resize_and_crop_input，让界面推理和你原 test.py 的思路保持一致。
    processed_pil_image = model.resize_and_crop_input(pil_image)

    # 4）前向推理，得到类别编号图
    predicted_mask_pil = model(processed_pil_image)

    # 5）转成 numpy，便于后续显示和统计
    processed_image = np.array(processed_pil_image)
    predicted_mask = np.array(predicted_mask_pil)

    # 6）把类别图转成彩色图，再叠加到原图上
    color_mask = convert_mask_to_color(predicted_mask)
    overlay = blend_image_and_mask(processed_image, color_mask, alpha)

    # 7）做类别统计
    summary_table = summarize_mask(predicted_mask)

    # 8）运行结果提示
    status_text = (
        f"图片分割完成。当前送入模型的尺寸为 {processed_image.shape[1]}×{processed_image.shape[0]}，"
        f"共识别出 {len(np.unique(predicted_mask))} 个类别。"
    )

    return processed_image, color_mask, overlay, summary_table, status_text


# ============================================================
# 五、视频推理函数
# ============================================================
def get_video_input_path(video_file) -> str:
    """
    从 Gradio 的视频组件中，提取真实的视频文件路径。

    因为 Gradio 不同版本下，video_file 可能是：
    - 字符串路径
    - 带 .name 属性的对象
    所以这里统一兼容处理。
    """
    if video_file is None:
        raise gr.Error("请先上传一个视频文件。")

    if isinstance(video_file, str):
        return video_file

    if hasattr(video_file, "name"):
        return str(video_file.name)

    return str(video_file)


def process_video_file(model_path: str, video_path: str, alpha: float) -> Tuple[str, str]:
    """
    对整个视频逐帧做分割，并输出一个新视频文件。

    这里的处理思路与你仓库里的 process_video.py 保持一致：
    - 逐帧读取
    - 每帧转成 PIL
    - 调用 DeepLabWrapper 预测
    - 把 mask 叠加到图像上
    - 最后写成 mp4 视频

    返回：
        output_video_path, status_text
    """
    model = load_model_cached(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("视频无法打开，请检查文件是否损坏。")

    # 读取输入视频的帧率。
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 20.0

    # 由于你的模型最终输入固定是 513×513，输出视频也统一写成 513×513。
    output_width = 513
    output_height = 513

    # 使用临时目录保存输出视频，避免覆盖原视频。
    out_dir = Path("gradio_outputs")
    out_dir.mkdir(exist_ok=True)
    output_video_path = out_dir / "segmentation_result.mp4"

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (output_width, output_height),
    )

    frame_count = 0

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            # OpenCV 读进来是 BGR，要先转 RGB。
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # 与单张图片处理保持一致
            processed_pil_frame = model.resize_and_crop_input(frame_pil)
            predicted_mask_pil = model(processed_pil_frame)

            processed_frame = np.array(processed_pil_frame)
            predicted_mask = np.array(predicted_mask_pil)

            color_mask = convert_mask_to_color(predicted_mask)
            overlay_rgb = blend_image_and_mask(processed_frame, color_mask, alpha)

            # 写视频时要转回 BGR
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            writer.write(overlay_bgr)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise gr.Error("没有成功读取到任何视频帧，请更换视频后重试。")

    status_text = f"视频处理完成，共处理 {frame_count} 帧，结果视频已生成。"
    return str(output_video_path), status_text


def infer_video(model_path: str, video_file, alpha: float):
    """
    Gradio 视频页按钮对应的函数。
    """
    video_path = get_video_input_path(video_file)
    output_video_path, status_text = process_video_file(model_path, video_path, alpha)
    return output_video_path, status_text


# 六、构建 Gradio 界面

def build_demo() -> gr.Blocks:
    """
    构建整个 Gradio 页面。

    界面设计思路：
    1. 顶部：模型路径 + 透明度
    2. 中间：两个 tab
       - 图片检测
       - 视频检测
    3. 底部：使用说明
    """
    legend_html = build_legend_html()

    with gr.Blocks(title="非结构化道路检测系统") as demo:
        gr.Markdown(
            """
            # 基于 DeepLab 的非结构化道路检测系统
            这是一个基于 **NewSeg** 项目的中文 Gradio 交互界面。

            你可以在这里完成：
            - 单张图片语义分割
            - 视频语义分割
            - 分割结果可视化与类别统计
            """
        )

        # =====================
        # 全局参数区域
        # =====================
        with gr.Row():
            model_path_box = gr.Textbox(
                label="模型路径",
                value="runs/mobilenetv3large_v1.50.pt",
                placeholder="例如：runs/mobilenetv3large_v1.50.pt",
                scale=4,
            )
            alpha_slider = gr.Slider(
                label="分割图叠加透明度 alpha",
                minimum=0.0,
                maximum=1.0,
                value=0.50,
                step=0.05,
                scale=2,
            )

        with gr.Row():
            clear_cache_button = gr.Button("清空模型缓存", variant="secondary")
            cache_status_box = gr.Textbox(label="缓存状态", interactive=False)
            clear_cache_button.click(fn=clear_model_cache, outputs=cache_status_box)

        with gr.Accordion("类别图例", open=False):
            gr.HTML(legend_html)

        # =====================
        # 图片检测标签页
        # =====================
        with gr.Tabs():
            with gr.Tab("图片检测"):
                with gr.Row():
                    input_image = gr.Image(label="上传图片", type="numpy")
                    with gr.Column():
                        image_run_button = gr.Button("开始图片分割", variant="primary")
                        image_status_box = gr.Textbox(label="运行状态", interactive=False)

                with gr.Row():
                    processed_image_output = gr.Image(label="模型输入图（已缩放/裁剪）")
                    color_mask_output = gr.Image(label="彩色分割结果")
                    overlay_output = gr.Image(label="分割叠加结果")

                summary_table_output = gr.Dataframe(
                    headers=["类别ID", "类别名称", "像素数", "占比"],
                    datatype=["str", "str", "str", "str"],
                    label="分割结果统计表",
                    wrap=True,
                )

                image_run_button.click(
                    fn=infer_single_image,
                    inputs=[model_path_box, input_image, alpha_slider],
                    outputs=[
                        processed_image_output,
                        color_mask_output,
                        overlay_output,
                        summary_table_output,
                        image_status_box,
                    ],
                )

            # =====================
            # 视频检测标签页
            # =====================
            with gr.Tab("视频检测"):
                with gr.Row():
                    input_video = gr.Video(label="上传视频")
                    with gr.Column():
                        video_run_button = gr.Button("开始视频分割", variant="primary")
                        video_status_box = gr.Textbox(label="运行状态", interactive=False)
                        gr.Markdown(
                            "建议优先使用较短视频测试。因为视频推理本质上是逐帧分割，耗时会明显高于单张图片。"
                        )

                output_video = gr.Video(label="输出视频")

                video_run_button.click(
                    fn=infer_video,
                    inputs=[model_path_box, input_video, alpha_slider],
                    outputs=[output_video, video_status_box],
                )

        # =====================
        # 底部说明
        # =====================
        gr.Markdown(
            """
            ## 使用说明
            1. 先在顶部填写模型路径。
            2. 在“图片检测”页上传图片，查看分割效果。
            3. 在“视频检测”页上传视频，生成分割视频。
            4. 如果你更换了模型，请点击“清空模型缓存”。

            """
        )

    return demo


# ============================================================
# 七、程序入口
# ============================================================
if __name__ == "__main__":
    demo = build_demo()

    # 这里默认只在本机打开，不主动暴露到公网。
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
