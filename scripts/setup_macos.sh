#!/bin/bash
# MuseTalk macOS 安装脚本
# 适用于 Apple Silicon (M1/M2/M3)

set -e

echo "=== MuseTalk macOS 安装脚本 ==="
echo ""

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "警告: 推荐使用 Python 3.10，当前版本: $PYTHON_VERSION"
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "安装基础依赖..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy opencv-python pillow tqdm omegaconf
pip install fastapi uvicorn python-multipart
pip install transformers diffusers accelerate
pip install librosa soundfile
pip install mmengine mmdet mmpose

echo ""
echo "=== 编译安装 mmcv (这需要一些时间) ==="
echo ""

# 安装 mmcv（从源码编译）
pip install ninja

if [ ! -d "mmcv" ]; then
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    git checkout v2.1.0
else
    cd mmcv
fi

export MMCV_WITH_OPS=1
export FORCE_CUDA=0
pip install -e . -v

cd ..

echo ""
echo "=== 下载模型 ==="
echo ""

# 创建模型目录
mkdir -p models/musetalk models/sd-vae models/whisper models/face-parse-bisent models/dwpose

# 下载模型
echo "下载 MuseTalk 模型..."
huggingface-cli download TMElyralab/MuseTalk \
    --local-dir models \
    --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

echo "下载 SD-VAE 模型..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
    --local-dir models/sd-vae \
    --include "config.json" "diffusion_pytorch_model.bin"

echo "下载 Whisper 模型..."
huggingface-cli download openai/whisper-tiny \
    --local-dir models/whisper

echo "下载人脸解析模型..."
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o models/face-parse-bisent/resnet18-5c106cde.pth

# BiSeNet 模型需要从 Google Drive 下载
pip install gdown
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
    -O models/face-parse-bisent/79999_iter.pth

echo ""
echo "=== 安装完成! ==="
echo ""
echo "启动服务:"
echo "  python api_server.py --host 127.0.0.1 --port 8765"
