from __future__ import annotations


import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image


from imageio_ffmpeg import get_ffmpeg_exe



# ============================================================
# 一、导入你项目里已经写好的模块
# ============================================================
from models import DeepLabWrapper
from utils.utils import LABEL_NAMES, RGB_COLORS


# 二、全局缓存：避免每点一次按钮都重新加载模型
MODEL_CACHE: Dict[str, DeepLabWrapper] = {}
# ============================================================
# 三、工具函数
# ============================================================
def resolve_model_path(model_path: str) -> Path:
    """
    把用户在界面中输入的模型路径，转换成绝对路径并检查是否存在。
    """
    if not model_path or not model_path.strip():
        raise gr.Error("请先填写模型路径，例如：runs/mobilenetv3large_v1.50.pt")

    path = Path(model_path.strip())
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise gr.Error(f"模型文件不存在：{path}")

    return path


def load_model_cached(model_path: str) -> DeepLabWrapper:
    """
    加载模型，并使用缓存机制。
    """
    resolved_path = str(resolve_model_path(model_path))

    if resolved_path not in MODEL_CACHE:
        MODEL_CACHE[resolved_path] = DeepLabWrapper(model_path=resolved_path)

    return MODEL_CACHE[resolved_path]


def clear_model_cache() -> str:
    """
    清空模型缓存。
    """
    MODEL_CACHE.clear()
    return "模型缓存已清空。下次推理时会重新加载模型。"


def convert_mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    把类别编号图（二维 mask）转换成彩色图，方便显示。
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, rgb in enumerate(RGB_COLORS[: len(LABEL_NAMES)]):
        color_mask[mask == class_id] = rgb

    return color_mask


def blend_image_and_mask(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    把原图和分割彩色图按一定透明度叠加。
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

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

    rows.sort(key=lambda x: float(x[3].replace("%", "")), reverse=True)
    return rows


def get_ffmpeg_path() -> str:
    """
    获取 imageio-ffmpeg 自带的 ffmpeg 可执行文件路径。
    """
    return get_ffmpeg_exe()


def transcode_video_for_web(raw_video_path: Path, output_video_path: Path) -> None:
    """
    把 OpenCV 写出的原始 mp4v 视频，再转码成更适合网页播放的 H.264 mp4。
    这里不做缩放，只做 1 像素级别的补边，避免 513x513 这样的奇数尺寸导致编码失败。
    """
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(raw_video_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an",
        "-movflags", "+faststart",
        "-c:v", "h264_mf",
        "-b:v", "12M",
        "-pix_fmt", "yuv420p",
        str(output_video_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise gr.Error(
            f"ffmpeg 转码失败。\n"
            f"错误信息：{e.stderr[-1200:] if e.stderr else '无 stderr 输出'}"
        ) from e

    if not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise gr.Error("转码后的网页播放视频没有成功生成。")


# ============================================================
# 四、图片推理函数
# ============================================================
def infer_single_image(model_path: str, input_image: np.ndarray, alpha: float):
    """
    对单张图片做语义分割。
    """
    if input_image is None:
        raise gr.Error("请先上传一张图片。")

    model = load_model_cached(model_path)
    pil_image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
    processed_pil_image = model.resize_and_crop_input(pil_image)
    predicted_mask_pil = model(processed_pil_image)

    processed_image = np.array(processed_pil_image)
    predicted_mask = np.array(predicted_mask_pil)

    color_mask = convert_mask_to_color(predicted_mask)
    overlay = blend_image_and_mask(processed_image, color_mask, alpha)

    summary_table = summarize_mask(predicted_mask)

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
    对整个视频逐帧做分割，并输出一个网页更容易直接播放的视频文件。
    这版保留你最初的 OpenCV 写视频方式，只在最后多做一步转码：
    1. 先按原逻辑写出 raw mp4v 视频
    2. 再用 ffmpeg 转成 H.264 mp4
    这样分割效果和你最初版本保持一致，不会因为强行改编码流程而变模糊。
    """
    model = load_model_cached(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("视频无法打开，请检查文件是否损坏。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 20.0

    output_width = 513
    output_height = 513

    out_dir = Path("gradio_outputs")
    out_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    raw_video_path = out_dir / f"segmentation_result_raw_{timestamp}.mp4"
    output_video_path = out_dir / f"segmentation_result_web_{timestamp}.mp4"

    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (output_width, output_height),
    )

    if not writer.isOpened():
        cap.release()
        raise gr.Error("原始视频写入器打开失败。")

    frame_count = 0

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            processed_pil_frame = model.resize_and_crop_input(frame_pil)
            predicted_mask_pil = model(processed_pil_frame)

            processed_frame = np.array(processed_pil_frame)
            predicted_mask = np.array(predicted_mask_pil)

            color_mask = convert_mask_to_color(predicted_mask)
            overlay_rgb = blend_image_and_mask(processed_frame, color_mask, alpha)

            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            writer.write(overlay_bgr)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise gr.Error("没有成功读取到任何视频帧，请更换视频后重试。")

    transcode_video_for_web(raw_video_path, output_video_path)

    status_text = (
        f"视频处理完成，共处理 {frame_count} 帧。\n"
        f"网页播放视频已生成。\n"
        f"输出文件：{output_video_path}"
    )
    return str(output_video_path), status_text


def infer_video(model_path: str, video_file, alpha: float):
    """
    Gradio 视频页按钮对应的函数。
    """
    video_path = get_video_input_path(video_file)
    output_video_path, status_text = process_video_file(model_path, video_path, alpha)
    return output_video_path, status_text


# ============================================================
# 六、构建 Gradio 界面
# ============================================================
def build_demo() -> gr.Blocks:
    """
    构建整个 Gradio 页面。
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

            with gr.Tab("视频检测"):
                with gr.Row():
                    input_video = gr.Video(label="上传视频")
                    with gr.Column():
                        video_run_button = gr.Button("开始视频分割", variant="primary")
                        video_status_box = gr.Textbox(label="运行状态", interactive=False)
                        gr.Markdown(
                            "建议优先使用较短视频测试。因为视频推理本质上是逐帧分割，耗时会明显高于单张图片。"
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=420):
                        output_video = gr.Video(
                            label="输出视频",
                            format="mp4",
                            height=320
                        )
                video_run_button.click(
                    fn=infer_video,
                    inputs=[model_path_box, input_video, alpha_slider],
                    outputs=[output_video, video_status_box],
                )

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


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
