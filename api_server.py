"""MuseTalk API 服务器

基于 FastAPI 的音频驱动数字人服务。

API 端点:
    - GET /: 服务状态
    - GET /health: 健康检查
    - POST /generate: 生成说话视频（音频 + 图片 → 视频）
    - POST /shutdown: 关闭服务
"""

import os
import sys
import uuid
import shutil
import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional

# 配置日志 - 确保输出到终端
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MuseTalk")

# 设置环境变量，确保 MPS 后备
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# MuseTalk 模块（延迟加载）
musetalk_pipeline = None


def get_device():
    """获取最佳可用设备"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# 创建 FastAPI 应用
app = FastAPI(
    title="MuseTalk API",
    description="音频驱动数字人服务",
    version="1.0.0"
)

# 临时文件目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)
(TEMP_DIR / "uploads").mkdir(exist_ok=True)
(TEMP_DIR / "outputs").mkdir(exist_ok=True)


class MuseTalkPipeline:
    """MuseTalk 推理管道"""

    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.face_parser = None
        self.loaded = False

    def load(self, use_float16: bool = False):
        """加载模型"""
        if self.loaded:
            return

        import torch
        from transformers import WhisperModel
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing

        logger.info("正在加载 MuseTalk 模型...")

        self.device = get_device()
        logger.info(f"使用设备: {self.device}")

        # 模型路径
        models_dir = PROJECT_ROOT / "models"
        unet_model_path = models_dir / "musetalk" / "pytorch_model.bin"
        unet_config = models_dir / "musetalk" / "musetalk.json"
        whisper_dir = str(models_dir / "whisper")

        # 检查模型文件
        if not unet_model_path.exists():
            raise FileNotFoundError(f"MuseTalk 模型不存在: {unet_model_path}")

        # 加载模型
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=str(unet_model_path),
            vae_type="sd-vae",
            unet_config=str(unet_config),
            device=self.device
        )

        # 转换为半精度（如果启用且不是 MPS）
        if use_float16 and self.device.type != "mps":
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        # 移动到设备
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        # 初始化音频处理器
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)

        # 加载 Whisper
        weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=weight_dtype).eval()
        self.whisper.requires_grad_(False)

        # 初始化人脸解析器
        self.face_parser = FaceParsing()

        self.loaded = True
        logger.info("MuseTalk 模型加载完成")

    def _save_frame(self, frame, path):
        """保存单帧到磁盘，处理格式和尺寸"""
        import numpy as np
        import cv2

        # 确保 frame 是正确的类型
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # libx264 要求宽高都是偶数
        h, w = frame.shape[:2]
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        if new_w != w or new_h != h:
            frame = frame[:new_h, :new_w]

        cv2.imwrite(str(path), frame)

    def generate(
        self,
        source_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25,
        bbox_shift: int = 0,
        upper_boundary_ratio: float = 0.5,
        expand: float = 1.5
    ) -> str:
        """生成说话视频

        Args:
            source_path: 源图像或视频路径
            audio_path: 音频文件路径
            output_path: 输出视频路径
            fps: 帧率
            bbox_shift: 边界框偏移
            upper_boundary_ratio: 混合区域上边界比例，0.3-0.7，越小保留越多上半脸
            expand: 人脸区域扩展系数，1.0-2.0，越大混合区域越大

        Returns:
            输出视频路径
        """
        import torch
        import cv2
        import glob
        import numpy as np
        from tqdm import tqdm
        from musetalk.utils.utils import get_file_type, get_video_fps, datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
        from musetalk.utils.blending import get_image

        if not self.loaded:
            self.load()

        timesteps = torch.tensor([0], device=self.device)

        # 创建临时目录
        task_id = str(uuid.uuid4())[:8]
        temp_dir = TEMP_DIR / "processing" / task_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 提取源帧
            source_type = get_file_type(source_path)
            if source_type == "video":
                frame_dir = temp_dir / "frames"
                frame_dir.mkdir(exist_ok=True)
                cmd = f'ffmpeg -v fatal -i "{source_path}" -start_number 0 "{frame_dir}/%08d.png"'
                os.system(cmd)
                input_img_list = sorted(glob.glob(str(frame_dir / "*.[jpJP][pnPN]*[gG]")))
                source_fps = get_video_fps(source_path)
            elif source_type == "image":
                input_img_list = [source_path]
                source_fps = fps
            else:
                raise ValueError(f"不支持的源文件类型: {source_path}")

            # 提取音频特征
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
            weight_dtype = self.unet.model.dtype
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                weight_dtype,
                self.whisper,
                librosa_length,
                fps=source_fps
            )

            # 预处理输入图像 - 使用正确的 API 调用
            input_latent_list = []

            # 批量获取人脸关键点和边界框
            bbox_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            # 处理帧 - 保持 BGR 格式（VAE 和 blending 都期望 BGR）
            coord_list = []
            for idx, (bbox, frame) in enumerate(zip(bbox_list, frame_list)):
                # frame 已经是 BGR，保持 BGR 格式
                # frame_list 保持原样，不做颜色转换

                if bbox == coord_placeholder:
                    coord_list.append(coord_placeholder)
                    continue

                # 裁剪人脸区域（BGR 格式）
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                # 编码为潜在向量 (使用正确的API)
                latent = self.vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latent)
                coord_list.append((y1, y2, x1, x2))

            if len(input_latent_list) == 0:
                raise ValueError("未能检测到人脸")

            # 提前创建输出帧目录（流式写入，避免内存累积）
            frame_dir_out = temp_dir / "output_frames"
            frame_dir_out.mkdir(exist_ok=True)

            # 生成帧（使用索引循环获取，避免大量列表复制）
            num_chunks = len(whisper_chunks)
            num_inputs = len(input_latent_list)
            logger.info(f"生成 {num_chunks} 帧...")

            import gc

            for idx in tqdm(range(num_chunks)):
                audio_feature = whisper_chunks[idx]
                # 循环索引获取，避免复制整个列表
                input_latent = input_latent_list[idx % num_inputs]
                coord = coord_list[idx % num_inputs]
                frame = frame_list[idx % num_inputs].copy()

                if coord == coord_placeholder:
                    # 无人脸帧，直接写入磁盘
                    self._save_frame(frame, frame_dir_out / f"{idx:08d}.png")
                    continue

                # 使用 UNet 预测
                # audio_feature 需要添加 batch 维度并通过位置编码处理
                audio_feature = audio_feature.to(device=self.device, dtype=weight_dtype)
                if audio_feature.dim() == 2:
                    audio_feature = audio_feature.unsqueeze(0)  # 添加 batch 维度
                audio_feature = self.pe(audio_feature)  # 通过位置编码

                input_latent = input_latent.to(device=self.device, dtype=weight_dtype)

                pred_latent = self.unet.model(
                    input_latent,
                    timesteps,
                    encoder_hidden_states=audio_feature
                ).sample

                # 解码 - decode_latents 返回 (batch, h, w, c) BGR 数组，取第一个
                pred_frames = self.vae.decode_latents(pred_latent)
                pred_frame = pred_frames[0]  # 取 batch 中的第一帧，已经是 BGR

                # 混合回原图
                # coord 存储为 (y1, y2, x1, x2)，需要转换为 get_image 需要的 [x1, y1, x2, y2]
                y1, y2, x1, x2 = coord
                h, w = y2 - y1, x2 - x1
                if h <= 0 or w <= 0:
                    # 无效坐标，直接写入磁盘
                    self._save_frame(frame, frame_dir_out / f"{idx:08d}.png")
                    continue
                pred_frame_resized = cv2.resize(pred_frame.astype(np.uint8), (w, h), interpolation=cv2.INTER_LANCZOS4)

                # 使用人脸解析进行混合
                # get_image 期望: BGR 原图, BGR 人脸, [x1,y1,x2,y2]
                # 返回: RGB 格式的混合图像（内部做了两次 [::-1] 转换）
                frame = get_image(
                    frame,  # 完整原图 (BGR)
                    pred_frame_resized,  # 生成的人脸 (BGR)
                    [x1, y1, x2, y2],  # face_box 坐标
                    upper_boundary_ratio=upper_boundary_ratio,
                    expand=expand,
                    fp=self.face_parser
                )

                # 直接写入磁盘，避免内存累积
                self._save_frame(frame, frame_dir_out / f"{idx:08d}.png")

                # 清理当前迭代的中间张量
                del pred_latent, pred_frames, pred_frame, pred_frame_resized

                # 每 50 帧清理一次内存缓存
                if idx % 50 == 0:
                    gc.collect()
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            # 合成视频（帧已在循环中保存到 frame_dir_out）
            silent_video = str(temp_dir / "silent.mp4")
            frame_count = len(list(frame_dir_out.glob("*.png")))
            logger.info(f"帧目录: {frame_dir_out}, 帧数: {frame_count}")

            # 检查帧文件是否存在
            frame_files = list(frame_dir_out.glob("*.png"))
            logger.info(f"保存的帧文件数: {len(frame_files)}")

            cmd = f'ffmpeg -y -v warning -r {source_fps} -i "{frame_dir_out}/%08d.png" -c:v libx264 -pix_fmt yuv420p "{silent_video}"'
            ret = os.system(cmd)
            logger.info(f"ffmpeg 合成视频返回: {ret}")

            # 添加音频
            cmd = f'ffmpeg -y -v warning -i "{silent_video}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{output_path}"'
            ret = os.system(cmd)
            logger.info(f"ffmpeg 添加音频返回: {ret}")

            return output_path

        finally:
            # 清理临时文件
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

            # 清理内存
            import gc
            gc.collect()

            # 清理 GPU/MPS 缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except:
                pass


def load_pipeline():
    """加载或获取 MuseTalk 管道"""
    global musetalk_pipeline

    if musetalk_pipeline is None:
        musetalk_pipeline = MuseTalkPipeline()

    if not musetalk_pipeline.loaded:
        musetalk_pipeline.load()

    return musetalk_pipeline


@app.on_event("startup")
async def startup_event():
    """服务启动时检查模型"""
    logger.info("正在启动 MuseTalk 服务...")

    models_dir = PROJECT_ROOT / "models"
    required_models = [
        models_dir / "musetalk" / "pytorch_model.bin",
        models_dir / "sd-vae" / "diffusion_pytorch_model.bin",
        models_dir / "whisper" / "pytorch_model.bin",
    ]

    missing = [str(m) for m in required_models if not m.exists()]
    if missing:
        logger.warning("缺少以下模型文件:")
        for m in missing:
            logger.warning(f"  - {m}")
        logger.warning("请运行 download_weights.sh 下载模型")
    else:
        logger.info("模型文件检查通过")

    logger.info("MuseTalk 服务已启动")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    global musetalk_pipeline

    logger.info("正在关闭 MuseTalk 服务...")

    if musetalk_pipeline is not None:
        del musetalk_pipeline
        musetalk_pipeline = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass

    logger.info("MuseTalk 服务已关闭")


@app.get("/")
async def root():
    """根路径 - 服务状态"""
    return {
        "service": "MuseTalk API",
        "status": "running",
        "model_loaded": musetalk_pipeline is not None and musetalk_pipeline.loaded,
        "endpoints": ["/health", "/generate", "/shutdown"]
    }


@app.get("/health")
async def health():
    """健康检查"""
    models_dir = PROJECT_ROOT / "models"
    return {
        "status": "healthy",
        "model_weights_exist": (models_dir / "musetalk" / "pytorch_model.bin").exists(),
        "model_loaded": musetalk_pipeline is not None and musetalk_pipeline.loaded
    }


@app.post("/generate")
async def generate(
    source: UploadFile = File(..., description="源图像或视频文件"),
    audio: UploadFile = File(..., description="音频文件"),
    fps: int = Form(25, description="帧率"),
    bbox_shift: int = Form(0, description="边界框偏移，正值向下扩展包含更多下巴，负值向上"),
    upper_boundary_ratio: float = Form(0.5, description="混合区域上边界比例，0.3-0.7，越小保留越多上半脸"),
    expand: float = Form(1.5, description="人脸区域扩展系数，1.0-2.0，越大混合区域越大"),
):
    """
    生成说话视频

    Args:
        source: 源图像（人脸）或视频
        audio: 音频文件
        fps: 帧率（默认25）
        bbox_shift: 边界框偏移（默认0），正值向下扩展包含更多下巴
        upper_boundary_ratio: 混合区域上边界比例（默认0.5），越小保留越多上半脸原图
        expand: 人脸区域扩展系数（默认1.5），越大混合区域越大越自然

    Returns:
        生成的说话视频文件
    """
    task_id = str(uuid.uuid4())[:8]

    # 保存上传的文件
    source_ext = Path(source.filename).suffix or ".jpg"
    audio_ext = Path(audio.filename).suffix or ".wav"

    source_path = TEMP_DIR / "uploads" / f"{task_id}_source{source_ext}"
    audio_path = TEMP_DIR / "uploads" / f"{task_id}_audio{audio_ext}"
    output_path = TEMP_DIR / "outputs" / f"{task_id}_output.mp4"

    try:
        # 保存文件
        with open(source_path, "wb") as f:
            content = await source.read()
            f.write(content)

        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # 加载模型
        pipeline = load_pipeline()

        # 生成视频
        logger.info(f"[{task_id}] 开始生成说话视频...")
        result_path = pipeline.generate(
            source_path=str(source_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            fps=fps,
            bbox_shift=bbox_shift,
            upper_boundary_ratio=upper_boundary_ratio,
            expand=expand
        )
        logger.info(f"[{task_id}] 视频生成完成")

        if Path(result_path).exists():
            return FileResponse(
                path=result_path,
                media_type="video/mp4",
                filename=f"talking_{task_id}.mp4"
            )
        else:
            raise HTTPException(status_code=500, detail="输出文件未找到")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")

    finally:
        # 清理上传的临时文件
        try:
            if source_path.exists():
                source_path.unlink()
            if audio_path.exists():
                audio_path.unlink()
        except:
            pass


@app.post("/shutdown")
async def shutdown():
    """关闭服务"""
    async def shutdown_server():
        await asyncio.sleep(0.5)
        os._exit(0)

    asyncio.create_task(shutdown_server())
    return {"status": "shutting down"}


@app.delete("/cleanup")
async def cleanup():
    """清理临时文件"""
    try:
        for subdir in ["uploads", "outputs"]:
            dir_path = TEMP_DIR / subdir
            if dir_path.exists():
                for file in dir_path.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass

        return {"success": True, "message": "临时文件已清理"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


def signal_handler(signum, frame):
    """处理 SIGINT 和 SIGTERM 信号"""
    logger.info(f"收到信号 {signum}，正在关闭服务...")
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    # 注册信号处理器，确保 Ctrl+C 能正常关闭
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="MuseTalk API Server")
    parser.add_argument("-a", "--host", default="127.0.0.1", help="服务器地址")
    parser.add_argument("-p", "--port", type=int, default=8770, help="服务器端口")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
