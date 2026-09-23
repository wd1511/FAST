# 资源准备指南

[English](ASSETS.md) | **简体中文**

所有命令均在 `FAST/` 中执行，相对路径以该目录为基准。源码仓库包含代码、配置和两个测试数据集。模型权重、训练数据集和运行环境分别下载。

## 下载入口

| 入口 | 内容 |
| --- | --- |
| [Hugging Face：wd1511/fast-ldm](https://huggingface.co/wd1511/fast-ldm) | `checkpoint/`、`pretrained_models/` 和 `model/annotator/`，约 19.82 GiB |
| [百度网盘：model](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg) | 模型、测试与训练数据、运行环境；提取码：`hcsc` |

Hugging Face 提供可直接在本地使用的模型目录。百度网盘提供压缩包，可按需分别下载。

## 目录结构

```text
FAST/
├── checkpoint/
│   ├── image_model/          # 图像 UNet 和 HED/深度/分割 adapter
│   └── video_model/          # 视频 UNet 和 HED/深度/分割 adapter
├── pretrained_models/
│   ├── stable-diffusion-v1-5/
│   │   ├── model_index.json
│   │   ├── tokenizer/
│   │   ├── scheduler/
│   │   ├── text_encoder/
│   │   ├── vae/
│   │   ├── unet/
│   │   ├── safety_checker/
│   │   └── feature_extractor/
│   ├── dpt-hybrid-midas/
│   ├── hed-network.pth
│   └── 150_16_swin_l_oneformer_coco_100ep.pth
├── model/annotator/          # annotator 实现、配置和权重
├── test_data/                # 包含在源码仓库中
│   ├── test_image_data/
│   │   ├── content/          # 36 张内容图
│   │   └── style/            # 36 张风格图
│   └── test_video_data1/
│       ├── content/          # 12 个视频
│       └── style/            # 12 张风格图
├── dataset/                 # 图像训练数据
│   ├── coco_image/
│   └── wiki_image/
├── data/                    # 视频训练示例和清单
│   ├── videos_part/
│   ├── video/
│   └── image/
└── diffusers_env/            # 可选的运行环境解压目录
```

`checkpoint/{image_model,video_model}/` 下的 `unet/`、`hed_adapter/`、`depth_adapter/` 和 `seg_adapter/` 目录均需包含 `config.json` 与 `diffusion_pytorch_model.bin`。根据所选控制类型下载相应 adapter。图像与视频 checkpoint 应与各自的推理入口配套使用。

## 从 Hugging Face 下载模型

安装 Hugging Face CLI 后执行：

```bash
hf download wd1511/fast-ldm --include 'checkpoint/**' --local-dir .
hf download wd1511/fast-ldm --include 'pretrained_models/**' --local-dir .
hf download wd1511/fast-ldm --include 'model/annotator/**' --local-dir .
```

这三个目录包含 659 个资源文件，目录结构与上面一致。按目录筛选可避免覆盖源码仓库的 `README.md`。测试输入包含在 GitHub 仓库中，训练数据从百度网盘下载。

## 百度网盘资源包

打开 [model 分享链接](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg)，输入提取码 `hcsc`。以下示例将下载的压缩包存放在与 `FAST/` 同级的 `HICAST/` 中。

| 压缩包 | 解压位置 | 用途 |
| --- | --- | --- |
| `checkpoint.tar.gz` | `FAST/checkpoint/` | 图像与视频 checkpoint |
| `pretrained_models.tar.gz` | `FAST/pretrained_models/` | DPT、HED、OneFormer 等辅助模型 |
| `models--runwayml--stable-diffusion-v1-5.zip` | `FAST/pretrained_models/stable-diffusion-v1-5/` | Stable Diffusion 权重与配置，转换方法见下文 |
| `annotator.tar.gz` | `FAST/model/annotator/` | annotator 实现、配置与权重 |
| `test_data.tar.gz` | `FAST/test_data/` | 图像与视频测试输入 |
| `dataset.tar.gz` | `FAST/dataset/` | COCO 内容图像与 WikiArt 风格图像 |
| `data.tar.gz` | `FAST/data/` | 视频训练示例与清单 |
| `diffusers_env.tar.gz` | `FAST/diffusers_env/` | Linux/CUDA 运行环境 |

### 测试数据

源码仓库包含两个测试集。从压缩包恢复时执行：

```bash
tar -xzf ../HICAST/test_data.tar.gz \
  test_data/test_image_data test_data/test_video_data1
```

两个目录共 96 个文件，约 43.33 MiB。支持的图像格式包括 JPG/JPEG、PNG、WebP 和 BMP；视频格式包括 MP4、MOV、AVI、MKV 和 WebM。自定义内容与风格文件可放入对应的 `content/` 和 `style/` 目录，也可通过推理参数指定其他路径。

### 模型包

通过 Hugging Face 下载过模型后，无需重复执行这些解压步骤。

```bash
mkdir -p model pretrained_models
tar -xzf ../HICAST/checkpoint.tar.gz
tar -xzf ../HICAST/annotator.tar.gz -C model
tar -xzf ../HICAST/pretrained_models.tar.gz \
  pretrained_models/dpt-hybrid-midas \
  pretrained_models/hed-network.pth \
  pretrained_models/150_16_swin_l_oneformer_coco_100ep.pth
```

`pretrained_models.tar.gz` 中的 Stable Diffusion 目录包含指向原缓存位置的绝对软链接。使用百度网盘压缩包时，下面的命令从 ZIP 的 `blobs/` 目录读取真实内容，并在项目内生成普通文件：

```bash
python - <<'PY'
from pathlib import Path, PurePosixPath
import shutil
import zipfile

target = Path("pretrained_models/stable-diffusion-v1-5")
with zipfile.ZipFile("../HICAST/models--runwayml--stable-diffusion-v1-5.zip") as archive:
    cache = "models--runwayml--stable-diffusion-v1-5/"
    revision = archive.read(cache + "refs/main").decode().strip()
    prefix = cache + "snapshots/" + revision + "/"
    for entry in archive.infolist():
        if not entry.filename.startswith(prefix) or entry.is_dir():
            continue
        name = entry.filename[len(prefix):]
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError(name)
        if ".fp16." in name or not name.endswith((".json", ".txt", ".bin")):
            continue
        blob = PurePosixPath(archive.read(entry).decode()).name
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        with archive.open(cache + "blobs/" + blob) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        print(destination)
PY
```

该布局使用 Diffusers FP32 `.bin` 权重，共 15 个文件、约 5.11 GiB。FAST 图像和视频 checkpoint 包含推理入口使用的 VGG 参数。

### 环境包

`diffusers_env.tar.gz` 适用于兼容的 Linux/CUDA 平台：

```bash
mkdir -p diffusers_env
tar -xzf ../HICAST/diffusers_env.tar.gz -C diffusers_env
source diffusers_env/bin/activate
conda-unpack
```

也可通过 `environment.yml` 创建 Conda 环境，主要依赖版本见 [环境要求](../README_zh.md#环境要求)。

## 训练数据与依赖

```bash
tar -xzf ../HICAST/dataset.tar.gz
tar -xzf ../HICAST/data.tar.gz
```

图像训练使用 `dataset/coco_image/` 和 `dataset/wiki_image/`。训练配置位于 `configs/`，图像验证使用 `test_data/test_image_data/`。

`data/` 压缩包提供 43 个视频示例和原始清单，不是完整的 HD-VILA/LAION 数据集。视频训练需要根据实际视频和图像位置生成本地清单，并设置 `configs/config_video.yaml` 中对应的数据路径。

视频训练还需要 `model/loss/temporal_loss/CCPL.py` 和 `CFC_loss.py`。源码包仅提供这两个模块的旧 Python 3.8 字节码；运行完整视频训练需要另行提供源码实现。图像与视频推理不依赖这两个训练损失模块。

## 资源存储

代码、配置和图像/视频测试输入存储在 GitHub。模型权重、训练数据集、打包环境、缓存和生成结果由 `.gitignore` 排除。`model/annotator/` 从 Hugging Face 或百度网盘下载。推理需要模型、annotator 和测试输入，无需训练数据集。
