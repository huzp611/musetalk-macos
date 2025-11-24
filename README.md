# MuseTalk macOS 部署指南

本指南详细说明如何在 macOS (Apple Silicon M1/M2/M3) 上部署 MuseTalk。

## 环境要求

- macOS 12.0+ (Monterey 或更高版本)
- Apple Silicon (M1/M2/M3) 或 Intel Mac
- Python 3.10
- Xcode Command Line Tools
- 约 10GB 磁盘空间（模型文件）

## 一、基础环境配置

### 1.1 安装 Python 3.10

```bash
# 使用 pyenv 安装
brew install pyenv
pyenv install 3.10.13
pyenv local 3.10.13

# 或使用 homebrew
brew install python@3.10
```

### 1.2 创建虚拟环境

```bash
cd /path/to/musetalk
python3.10 -m venv venv
source venv/bin/activate
```

## 二、依赖安装

### 2.1 基础依赖

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy opencv-python pillow tqdm omegaconf
pip install fastapi uvicorn python-multipart
pip install transformers diffusers accelerate
pip install librosa soundfile
```

### 2.2 编译安装 mmcv (关键步骤)

macOS ARM 需要从源码编译 mmcv：

```bash
# 安装编译依赖
pip install ninja

# 克隆 mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0

# 设置环境变量，启用自定义 ops
export MMCV_WITH_OPS=1
export FORCE_CUDA=0

# 编译安装（约 10-20 分钟）
pip install -e . -v

cd ..
```

### 2.3 安装 mmpose 和 mmdet

```bash
pip install mmengine
pip install mmdet
pip install mmpose
```

### 2.4 安装 face_alignment

```bash
pip install face_alignment
```

## 三、模型下载

### 3.1 MuseTalk 模型

```bash
# 创建模型目录
mkdir -p models/musetalk

# 下载 MuseTalk 模型
huggingface-cli download TMElyralab/MuseTalk \
    --local-dir models \
    --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"
```

### 3.2 SD-VAE 模型

```bash
mkdir -p models/sd-vae
huggingface-cli download stabilityai/sd-vae-ft-mse \
    --local-dir models/sd-vae \
    --include "config.json" "diffusion_pytorch_model.bin"
```

### 3.3 Whisper 模型

```bash
mkdir -p models/whisper
huggingface-cli download openai/whisper-tiny \
    --local-dir models/whisper
```

### 3.4 人脸解析模型

```bash
mkdir -p models/face-parse-bisent

# ResNet18 backbone
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o models/face-parse-bisent/resnet18-5c106cde.pth

# BiSeNet 模型 (从 Google Drive 下载)
pip install gdown
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
    -O models/face-parse-bisent/79999_iter.pth
```

### 3.5 DWPose 模型

```bash
mkdir -p models/dwpose
# 下载 dw-ll_ucoco_384.pth 到 models/dwpose/
```

## 四、代码兼容性修复

### 4.1 PyTorch 2.6+ 兼容性

在使用 `torch.load` 的文件中添加 `weights_only=False`：

```python
# 在文件开头添加
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

需要修改的文件：
- `musetalk/utils/preprocessing.py`
- `musetalk/models/vae.py`
- `musetalk/models/unet.py`
- `musetalk/utils/face_parsing/__init__.py`
- `musetalk/utils/face_parsing/resnet.py`

### 4.2 face_alignment API 兼容性

新版 face_alignment 库 API 变更，需修改 `musetalk/utils/preprocessing.py`：

```python
# 旧代码
# preds = fa.get_detections_for_batch(np.asarray(fb))

# 新代码
for img in fb:
    detected = fa.face_detector.detect_from_image(img)
    if len(detected) == 0:
        coords_list += [coord_placeholder]
        continue
    f = detected[0][:4]  # [x1, y1, x2, y2]
    # ... 后续处理
```

### 4.3 LandmarksType 兼容性

```python
# 旧代码
# fa = FaceAlignment(LandmarksType._2D, ...)

# 新代码
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device)
```

## 五、API 服务器

### 5.1 启动服务

```bash
cd /path/to/musetalk
source venv/bin/activate
python api_server.py --host 127.0.0.1 --port 8765
```

### 5.2 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/generate` | POST | 生成说话视频 |
| `/shutdown` | POST | 关闭服务 |

### 5.3 生成请求示例

```bash
curl -X POST http://127.0.0.1:8765/generate \
    -F "audio=@/path/to/audio.wav" \
    -F "source=@/path/to/image.png" \
    -F "bbox_shift=0" \
    -F "fps=25"
```

## 六、常见问题

### Q1: mmcv 编译失败

确保已安装 Xcode Command Line Tools：
```bash
xcode-select --install
```

### Q2: MPS 设备不支持某些操作

设置环境变量启用 MPS 回退：
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Q3: 模型加载警告 "weights_only"

这是 PyTorch 2.6+ 的安全警告，已通过 monkey patch 解决。

### Q4: SD-VAE 找不到 safetensors 文件

这只是警告，会自动回退使用 `.bin` 文件，不影响功能。

### Q5: 生成速度慢

- 使用 `--use_float16` 启用半精度（Intel Mac 可用）
- Apple Silicon 使用 MPS 加速
- 减少输入视频分辨率

## 七、目录结构

```
musetalk/
├── api_server.py           # API 服务器
├── models/
│   ├── musetalk/
│   │   ├── musetalk.json
│   │   └── pytorch_model.bin
│   ├── sd-vae/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   ├── whisper/
│   │   └── ...
│   ├── face-parse-bisent/
│   │   ├── resnet18-5c106cde.pth
│   │   └── 79999_iter.pth
│   └── dwpose/
│       └── dw-ll_ucoco_384.pth
├── musetalk/
│   ├── models/
│   ├── utils/
│   └── whisper/
├── scripts/
├── venv/
└── temp/
```

## 八、性能参考

在 Apple M2 Pro 上的测试结果：
- 模型加载：约 30 秒
- 1612 帧视频生成：约 10-15 分钟（CPU/MPS）
- 内存占用：约 8-10 GB

## 九、更新日志

- 2024-11-24: 完成 macOS ARM 适配，修复所有兼容性问题
