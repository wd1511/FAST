# 资源准备指南

所有命令和相对路径均以 `FAST/` 为基准。模型、数据与环境分别下载，源码仓库不包含这些大体积资源。

## 下载入口

| 入口 | 内容 |
| --- | --- |
| [Hugging Face：wd1511/fast-ldm](https://huggingface.co/wd1511/fast-ldm) | `checkpoint/`、`pretrained_models/`、`model/annotator/`，约 19.82 GiB |
| [百度网盘：model](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg) | 模型、测试与训练数据、运行环境；提取码：`hcsc` |

Hugging Face 提供可直接使用的模型目录。百度网盘提供模型、数据和环境压缩包，可按需选择下载。

## 目录结构

```text
FAST/
├── checkpoint/
│   ├── image_model/          # 图像 UNet、hed/depth/seg adapter
│   └── video_model/          # 视频 UNet、hed/depth/seg adapter
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
├── model/annotator/          # annotator 实现、配置和 ckpts
├── test_data/
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
└── diffusers_env/            # 使用环境压缩包时的解压目录
```

`checkpoint/{image_model,video_model}/` 下的 `unet/` 及所选控制类型对应的 `hed_adapter/`、`depth_adapter/`、`seg_adapter/`，均需包含 `config.json` 和 `diffusion_pytorch_model.bin`。图像与视频 checkpoint 应与各自的推理入口配套使用。

## Hugging Face 模型下载

安装 Hugging Face CLI 后执行：

```bash
hf download wd1511/fast-ldm --include 'checkpoint/**' --local-dir .
hf download wd1511/fast-ldm --include 'pretrained_models/**' --local-dir .
hf download wd1511/fast-ldm --include 'model/annotator/**' --local-dir .
```

这三个目录包含 659 个资源文件，下载后即可得到上面的模型布局。按目录筛选可避免覆盖源码目录的 `README.md`。测试数据和训练数据从百度网盘单独获取。

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

```bash
tar -xzf ../HICAST/test_data.tar.gz \
  test_data/test_image_data test_data/test_video_data1
```

两个测试目录共 96 个文件，约 43.33 MiB。图像支持 JPG/JPEG、PNG、WebP、BMP；视频支持 MP4、MOV、AVI、MKV、WebM。自己的内容和风格文件可以放入对应的 `content/`、`style/`，或通过推理参数指定其他目录。

### 模型包

通过 Hugging Face 下载过模型后，无需重复解压这些模型包。

```bash
mkdir -p model pretrained_models
tar -xzf ../HICAST/checkpoint.tar.gz
tar -xzf ../HICAST/annotator.tar.gz -C model
tar -xzf ../HICAST/pretrained_models.tar.gz \
  pretrained_models/dpt-hybrid-midas \
  pretrained_models/hed-network.pth \
  pretrained_models/150_16_swin_l_oneformer_coco_100ep.pth
```

`pretrained_models.tar.gz` 中的 Stable Diffusion 目录使用了指向原缓存位置的绝对软链接。使用百度网盘模型包时，通过下面的命令从 ZIP 的 `blobs/` 读取真实内容，生成项目内的普通文件：

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

该布局采用 Diffusers FP32 `.bin` 文件，共 15 个文件、约 5.11 GiB。图像与视频 checkpoint 自带 VGG 参数，推理入口直接加载其中的权重。

### 环境包

`diffusers_env.tar.gz` 适用于匹配的 Linux/CUDA 平台：

```bash
mkdir -p diffusers_env
tar -xzf ../HICAST/diffusers_env.tar.gz -C diffusers_env
source diffusers_env/bin/activate
conda-unpack
```

也可按项目 `environment.yml` 创建 Conda 环境，主要版本要求见 [README](../README.md#环境)。

## 训练数据与依赖

```bash
tar -xzf ../HICAST/dataset.tar.gz
tar -xzf ../HICAST/data.tar.gz
```

图像训练使用 `dataset/coco_image/` 和 `dataset/wiki_image/`，配置位于 `configs/`。图像验证使用 `test_data/test_image_data/`。

`data/` 提供 43 个视频示例及原始清单，不是完整 HD-VILA/LAION 数据集。视频训练需要根据实际视频和图像位置生成本地清单，并设置 `configs/config_video.yaml` 中的数据路径。

视频训练还依赖 `model/loss/temporal_loss/CCPL.py` 和 `CFC_loss.py`。资源包中仅有这两个模块的旧 Python 3.8 字节码，运行完整视频训练需要另行提供源码。图像与视频推理不依赖这两个训练损失模块。

## 资源存储

模型、数据、环境、缓存和推理结果由 `.gitignore` 排除，不纳入源码仓库。`model/annotator/` 从 Hugging Face 或百度网盘获取。推理只需模型、annotator 和测试输入，无需下载训练数据集。
