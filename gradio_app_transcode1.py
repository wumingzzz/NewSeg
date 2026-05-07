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
from models import DeepLabWrapper
from utils import LABEL_NAMES, RGB_COLORS, convert_mask_to_risk


# 全局缓存：避免每点一次按钮都重新加载模型,这行定义了一个全局字典
MODEL_CACHE: Dict[str, DeepLabWrapper] = {}
RISK_RGB_COLORS = [
    (0, 176, 80),    # 可通行 - 绿色
    (255, 192, 0),   # 谨慎通行 - 黄色
    (255, 0, 0),     # 不可通行 - 红色
]

# 一、工具函数

def resolve_model_path(model_path: str) -> Path:
    """
    把用户在界面中输入的模型路径，转换成绝对路径并检查是否存在。
    """
    if not model_path or not model_path.strip():  #如果路径为空   .strip() 表示去掉字符串前后的空格。
        raise gr.Error("请先填写模型路径，例如：runs/mobilenetv3large_v1.50.pt") #报错提示

    path = Path(model_path.strip()) #转换成path对象
    if not path.is_absolute():  #判断这个路径是不是绝对路径，不是的话转换成绝对路径
        path = (Path.cwd() / path).resolve()  #Path.cwd() 表示当前项目所在目录

    if not path.exists():
        raise gr.Error(f"模型文件不存在：{path}")
    return path


def load_model_cached(model_path: str) -> DeepLabWrapper:
    """
    加载模型，并使用缓存机制。
    """
    resolved_path = str(resolve_model_path(model_path)) #调用上面的函数然后转成字符串

    if resolved_path not in MODEL_CACHE: #判断这个模型是否已经在缓存里，不在的话就加载
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
    把类别编号图（二维 mask）转换成彩色图，方便显示，函数输入预测的mask
    """
    # 创建一个全黑的彩色图 mask是一个二维数组 shape[0]是H  shape[1]是W
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

#遍历颜色 enumerate 会同时给出编号和内容 RGB_COLORS是列表
    for class_id, rgb in enumerate(RGB_COLORS[: len(LABEL_NAMES)]):
        color_mask[mask == class_id] = rgb #当mask里面的值等于这个id的时候，就让他的颜色变成这个颜色

    return color_mask


def convert_risk_to_color(risk_map: np.ndarray) -> np.ndarray:
    """
    把三类风险图转换成彩色图，0-可通行，1-谨慎通行，2-不可通行
    """
    color_risk_map = np.zeros((risk_map.shape[0], risk_map.shape[1], 3), dtype=np.uint8)

    for risk_id, rgb in enumerate(RISK_RGB_COLORS):
        color_risk_map[risk_map == risk_id] = rgb

    return color_risk_map


def blend_image_and_mask(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    把原图和分割彩色图按一定透明度叠加，输入了三个 一个是img_rgb即原图，color_maskc彩色分割图，alpha透明度
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))#确保 alpha 在0到1之间,如果alpha小于0，就变成0,大于1就变成1,否则保持原值

    image_float = image_rgb.astype(np.float32)#转换成浮点数，因为后面要加权计算
    mask_float = color_mask.astype(np.float32)#转换成浮点数

    overlay = image_float * (1.0 - alpha) + mask_float * alpha #加权计算
    return np.clip(overlay, 0, 255).astype(np.uint8) #结果限制在0-255之间，然后转换回图像常用的uint8格式


def build_legend_html() -> str:
    """
    生成网页中展示的“类别-颜色图例”。
    """
    rows = []  #创建空列表，用来保存每行HTML
    #zip(LABEL_NAMES, RGB_COLORS) 会把名称和颜色配对，例如[("sky", [128, 128, 255]),()]
    for class_id, (name, rgb) in enumerate(zip(LABEL_NAMES, RGB_COLORS)):
        # 把颜色数组变成网页 CSS 能识别的格式。例如[255,0,0]变成 rgb(255,0,0)
        color_str = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        #HTML代码
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
    total_pixels = int(mask.size)  #统计总像素
    unique_values = np.unique(mask) #统计像素类别
    rows: List[List[str]] = []   #创造空列表  冒号后面是格式

    for class_id in unique_values: #遍历每个出现的类别
        class_id = int(class_id)  #把类别编号转换成int
        #如果编号在 0和长度之间 则标签的名字赋给class_name 不是的话就单独命名class_xxx
        class_name = LABEL_NAMES[class_id] if 0 <= class_id < len(LABEL_NAMES) else f"class_{class_id}"
        pixel_count = int((mask == class_id).sum()) #统计这个类别有多少个像素
        ratio = 100.0 * pixel_count / max(total_pixels, 1)  #计算这个像素的占比 max是防止除以0
        rows.append([str(class_id), class_name, str(pixel_count), f"{ratio:.2f}%"]) #这些东西添加到rows列表中
    #rows.sort()排序，key=lambda x:...指定排序依据，取x[3]即ratio，先去除掉%，再转化成浮点数，reverse=True降序排列
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
    #ffmpeg命令,ffmpeg_path调用ffmpeg,-y 如果输入文件存在直接替换 -i 指定输入视频
    #-vf 给视频补边，保证宽高是偶数  -an 去掉音频 -movflags +faststart 使其适合网页播放
    #-c:v h264_mf 使用H264编码器 -b：v 12M设置视频码率为12M  -pix_fmt 设置像素格式
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
#尝试执行转码命令
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



# 四、图片推理函数

def infer_single_image(model_path: str, input_image: np.ndarray, alpha: float):
    """
    对单张图片做语义分割。
    """
    if input_image is None:
        raise gr.Error("请先上传一张图片。")

    model = load_model_cached(model_path)
    pil_image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")#把Numpy转化成PIL,convert保证图片是RGB三通道
    processed_pil_image = model.resize_and_crop_input(pil_image) #调用图像调整 即中心裁剪等
    predicted_mask_pil = model(processed_pil_image) #调用模型推断

    processed_image = np.array(processed_pil_image) #图像转换回numpy
    predicted_mask = np.array(predicted_mask_pil)  #把mask转换为numpy

    color_mask = convert_mask_to_color(predicted_mask)  #mask转换成对应颜色
    overlay = blend_image_and_mask(processed_image, color_mask, alpha) #重叠图片
    risk_map = convert_mask_to_risk(predicted_mask) #把9类语义mask转换成3类风险图
    risk_color_map = convert_risk_to_color(risk_map) #三类风险图转换成颜色图
    risk_overlay = blend_image_and_mask(processed_image, risk_color_map, alpha) #三类风险叠加图

    summary_table = summarize_mask(predicted_mask) #调用统计函数 创建统计表

    status_text = (
        f"图片分割完成。当前送入模型的尺寸为 {processed_image.shape[1]}×{processed_image.shape[0]}，"
        f"共识别出 {len(np.unique(predicted_mask))} 个类别。"
    )

    return processed_image, color_mask, overlay, risk_color_map, risk_overlay, summary_table, status_text



# 五、视频推理函数
def get_video_input_path(video_file) -> str:
    """
    从 Gradio 的视频组件中，提取真实的视频文件路径。
    """
    if video_file is None:
        raise gr.Error("请先上传一个视频文件。")

    if isinstance(video_file, str): #如果给的就是字符串路径，那就直接返回。
        return video_file

    if hasattr(video_file, "name"): #判断对象是否有name属性，针对文件对象，name属性表示文件路径
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

    cap = cv2.VideoCapture(video_path) #打开视频文件
    if not cap.isOpened():#判断是否能打开
        raise gr.Error("视频无法打开，请检查文件是否损坏。")

    fps = cap.get(cv2.CAP_PROP_FPS)#读取视频帧率，如果读取不到就设置成20
    if fps is None or fps <= 1e-6:
        fps = 20.0

    output_width = 513
    output_height = 513

    out_dir = Path("gradio_outputs") #创建输出文件夹
    out_dir.mkdir(exist_ok=True) #如果已经存在也不要报错

    timestamp = int(time.time())#获取当前时间戳
    #生成两个视频路径：
    raw_video_path = out_dir / f"segmentation_result_raw_{timestamp}.mp4"
    output_video_path = out_dir / f"segmentation_result_web_{timestamp}.mp4"
#创建视频写入器，输出视频路径，视频编码格式用MP4V,输出视频帧率，输出视频尺寸
    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (output_width, output_height),
    )

    if not writer.isOpened():#判断视频写入器能不能打开
        cap.release()
        raise gr.Error("原始视频写入器打开失败。")

    frame_count = 0 #记录处理了多少帧

    try:
        while True:
            success, frame_bgr = cap.read()  #视频处理的常用 success判断是否读取成功，视频结束了就变成0了，bgr就是像素
            if not success: #读取不到就break 脱离循环
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #BGR转换成RGB，模型用RGB
            frame_pil = Image.fromarray(frame_rgb) #转换成PIL格式

            processed_pil_frame = model.resize_and_crop_input(frame_pil)#中心裁剪等预操作
            predicted_mask_pil = model(processed_pil_frame)#预测得到分割mask

            processed_frame = np.array(processed_pil_frame)#转换回numpy
            predicted_mask = np.array(predicted_mask_pil) #转换为numpy

            color_mask = convert_mask_to_color(predicted_mask) #把mask转换为彩色图像
            overlay_rgb = blend_image_and_mask(processed_frame, color_mask, alpha) #混合

            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR) #转换回BGR，因为Opencv写视频需要BGR
            writer.write(overlay_bgr)#把这一帧写入输出视频
            frame_count += 1 #处理帧数+1

    finally:
        cap.release()
        writer.release() #最后都要释放读取器和写入器

    if frame_count == 0:
        raise gr.Error("没有成功读取到任何视频帧，请更换视频后重试。")

    transcode_video_for_web(raw_video_path, output_video_path) #调用转码函数

    status_text = (
        f"视频处理完成，共处理 {frame_count} 帧。\n"
        f"网页播放视频已生成。\n"
        f"输出文件：{output_video_path}"
    )
    return str(output_video_path), status_text


def infer_video(model_path: str, video_file, alpha: float):
    """
    Gradio视频按钮对应的函数，就是把上面的整合成一个
    """
    video_path = get_video_input_path(video_file) #取视频路径
    output_video_path, status_text = process_video_file(model_path, video_path, alpha)#调用处理函数
    return output_video_path, status_text



# 六、构建 Gradio 界面

def build_demo() -> gr.Blocks:
    """
    这个函数负责构建整个Gradio页面。
    """
    legend_html = build_legend_html() #调用之前定义的函数，生成类别图例HTML
    #创建Gradio界面,title是浏览器页面的标题，并将其命名为demo，以下的所有内容都是属于这个Gradio界面的
    with gr.Blocks(title="非结构化道路检测系统") as demo:
        gr.Markdown(
            """
            # 基于DeepLab的非结构化道路检测系统
            可以在这里完成：
            - 单张图片的语义分割
            - 视频的语义分割
            - 分割结果可视化与类别统计
            """
        )
        #创建一行布局，这一行里面放模型路径输入框，以及透明度滑条，label输入框标题，value默认值，placeholder提示词，scale行所占比例
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
        #再创建一行
        with gr.Row():
            clear_cache_button = gr.Button("清空模型缓存", variant="secondary")#variant表示次要按钮样式。
            cache_status_box = gr.Textbox(label="缓存状态", interactive=False)#创建状态显示框，interactive表示不能编辑，只用来显示结果
            #绑定按钮点击事件，fn是使用的函数，outputs是输出值显示到什么地方
            clear_cache_button.click(fn=clear_model_cache, outputs=cache_status_box)
        #创建一个可折叠区域，标题是类别图例，open=False是表示默认收起来
        with gr.Accordion("类别图例", open=False):
            gr.HTML(legend_html)#显示刚才的类别颜色图例

        #创建标签页（可切换），里面有两个页面：图片检测、视频检测
        with gr.Tabs():
            #创建图片检测标签页
            with gr.Tab("图片检测"):
                #创建一行
                with gr.Row():
                    input_image = gr.Image(label="上传图片", type="numpy")#创建图片上传组件
                    #创建一列，在图片上传组件旁边
                    with gr.Column():
                        #创建开始图片分割按钮
                        image_run_button = gr.Button("开始图片分割", variant="primary")
                        #创建状态窗口用来显示运行状态
                        image_status_box = gr.Textbox(label="运行状态", interactive=False)
                #创建一行，这行用来显示结果
                with gr.Row():
                    processed_image_output = gr.Image(label="模型输入图（已缩放/裁剪）")
                    color_mask_output = gr.Image(label="彩色分割结果")
                    overlay_output = gr.Image(label="分割叠加结果")
                #创建一行，这行用来显示三类风险图
                with gr.Row():
                    risk_color_output = gr.Image(label="三类风险等级图")
                    risk_overlay_output = gr.Image(label="三类风险叠加图")
                #创建表格，用来显示类别结果
                summary_table_output = gr.Dataframe(
                    headers=["类别ID", "类别名称", "像素数", "占比"],
                    datatype=["str", "str", "str", "str"],
                    label="分割结果统计表",
                    wrap=True,
                )
                #以下为按钮绑定
                image_run_button.click(
                    fn=infer_single_image,
                    inputs=[model_path_box, input_image, alpha_slider],
                    outputs=[
                        processed_image_output,
                        color_mask_output,
                        overlay_output,
                        risk_color_output,
                        risk_overlay_output,
                        summary_table_output,
                        image_status_box,
                    ],
                )
            #创建视频检测标签页
            with gr.Tab("视频检测"):
                #创建一行
                with gr.Row():
                    input_video = gr.Video(label="上传视频")#创建视频上传组件
                    #创建一列，在视频上传组件旁边
                    with gr.Column():
                        video_run_button = gr.Button("开始视频分割", variant="primary")
                        video_status_box = gr.Textbox(label="运行状态", interactive=False)
                        gr.Markdown(
                            "建议优先使用较短视频测试。因为视频推理是逐帧分割，耗时会明显高于单张图片。"
                        )
                #创建一行用来显示输出视频
                with gr.Row():
                    #创建一列，设置最小宽度420
                    with gr.Column(scale=1, min_width=420):
                        output_video = gr.Video(
                            label="输出视频",
                            format="mp4",
                            height=420
                        )
                #按钮绑定函数，显示输出
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
