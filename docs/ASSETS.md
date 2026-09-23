# Asset Setup Guide

**English** | [简体中文](ASSETS_zh.md)

Run all commands from `FAST/`. Relative paths are resolved from that directory. The repository includes source code, configurations, and the two test datasets. Download model weights, training datasets, and the runtime environment separately.

## Download Sources

| Source | Contents |
| --- | --- |
| [Hugging Face: wd1511/fast-ldm](https://huggingface.co/wd1511/fast-ldm) | `checkpoint/`, `pretrained_models/`, and `model/annotator/`, approximately 19.82 GiB |
| [Baidu Netdisk: model](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg) | Models, test and training datasets, and the runtime environment; access code: `hcsc` |

Hugging Face provides model directories ready for local use. Baidu Netdisk provides archives that can be downloaded individually as needed.

## Directory Layout

```text
FAST/
├── checkpoint/
│   ├── image_model/          # Image UNet and HED/depth/segmentation adapters
│   └── video_model/          # Video UNet and HED/depth/segmentation adapters
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
├── model/annotator/          # Annotator implementations, configurations, and weights
├── test_data/                # Included in this repository
│   ├── test_image_data/
│   │   ├── content/          # 36 content images
│   │   └── style/            # 36 style images
│   └── test_video_data1/
│       ├── content/          # 12 videos
│       └── style/            # 12 style images
├── dataset/                 # Image training data
│   ├── coco_image/
│   └── wiki_image/
├── data/                    # Video training examples and manifests
│   ├── videos_part/
│   ├── video/
│   └── image/
└── diffusers_env/            # Optional extracted runtime environment
```

Each `unet/`, `hed_adapter/`, `depth_adapter/`, and `seg_adapter/` directory under `checkpoint/{image_model,video_model}/` requires `config.json` and `diffusion_pytorch_model.bin`. Download the adapters needed by the selected controls. Use the image and video checkpoints with their corresponding inference entry points.

## Download Models from Hugging Face

With the Hugging Face CLI installed, run:

```bash
hf download wd1511/fast-ldm --include 'checkpoint/**' --local-dir .
hf download wd1511/fast-ldm --include 'pretrained_models/**' --local-dir .
hf download wd1511/fast-ldm --include 'model/annotator/**' --local-dir .
```

These directories contain 659 asset files and match the layout above. Filtering by directory avoids replacing the source repository's `README.md`. Test inputs are included in GitHub; training data is available from Baidu Netdisk.

## Baidu Netdisk Archives

Open the [model share](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg) and enter access code `hcsc`. The examples below assume downloaded archives are stored in `HICAST/`, alongside `FAST/`.

| Archive | Destination | Purpose |
| --- | --- | --- |
| `checkpoint.tar.gz` | `FAST/checkpoint/` | Image and video checkpoints |
| `pretrained_models.tar.gz` | `FAST/pretrained_models/` | DPT, HED, OneFormer, and other auxiliary models |
| `models--runwayml--stable-diffusion-v1-5.zip` | `FAST/pretrained_models/stable-diffusion-v1-5/` | Stable Diffusion weights and configurations; see the conversion below |
| `annotator.tar.gz` | `FAST/model/annotator/` | Annotator implementations, configurations, and weights |
| `test_data.tar.gz` | `FAST/test_data/` | Image and video test inputs |
| `dataset.tar.gz` | `FAST/dataset/` | COCO content images and WikiArt style images |
| `data.tar.gz` | `FAST/data/` | Video training examples and manifests |
| `diffusers_env.tar.gz` | `FAST/diffusers_env/` | Linux/CUDA runtime environment |

### Test Data

The repository includes both test datasets. To restore them from the archive:

```bash
tar -xzf ../HICAST/test_data.tar.gz \
  test_data/test_image_data test_data/test_video_data1
```

The two directories contain 96 files, approximately 43.33 MiB. Supported image formats are JPG/JPEG, PNG, WebP, and BMP. Supported video formats are MP4, MOV, AVI, MKV, and WebM. Place custom content and style files in the corresponding `content/` and `style/` directories, or supply custom paths to the inference scripts.

### Model Archives

Skip these extraction steps if the models have already been downloaded from Hugging Face.

```bash
mkdir -p model pretrained_models
tar -xzf ../HICAST/checkpoint.tar.gz
tar -xzf ../HICAST/annotator.tar.gz -C model
tar -xzf ../HICAST/pretrained_models.tar.gz \
  pretrained_models/dpt-hybrid-midas \
  pretrained_models/hed-network.pth \
  pretrained_models/150_16_swin_l_oneformer_coco_100ep.pth
```

The Stable Diffusion directory in `pretrained_models.tar.gz` contains absolute symlinks to the original cache location. For the Baidu Netdisk archives, the following command reads the actual contents from the ZIP's `blobs/` directory and writes regular files inside the project:

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

This layout uses Diffusers FP32 `.bin` weights: 15 files, approximately 5.11 GiB. FAST image and video checkpoints include the VGG parameters loaded by the inference entry points.

### Packaged Environment

Use `diffusers_env.tar.gz` on a compatible Linux/CUDA platform:

```bash
mkdir -p diffusers_env
tar -xzf ../HICAST/diffusers_env.tar.gz -C diffusers_env
source diffusers_env/bin/activate
conda-unpack
```

Alternatively, create a Conda environment from `environment.yml`. See the [requirements](../README.md#requirements) for the main dependency versions.

## Training Data and Dependencies

```bash
tar -xzf ../HICAST/dataset.tar.gz
tar -xzf ../HICAST/data.tar.gz
```

Image training uses `dataset/coco_image/` and `dataset/wiki_image/`. Training configurations are in `configs/`, and image validation uses `test_data/test_image_data/`.

The `data/` archive provides 43 example videos and the original manifests, rather than the complete HD-VILA/LAION datasets. Video training requires local manifests that match the actual video and image locations, together with the corresponding paths in `configs/config_video.yaml`.

Video training also requires `model/loss/temporal_loss/CCPL.py` and `CFC_loss.py`. Only legacy Python 3.8 bytecode for these modules is provided in the source package; the source implementations must be supplied separately for full video training. Image and video inference do not depend on these training loss modules.

## Asset Storage

The source code, configurations, and image/video test inputs are stored in GitHub. Model weights, training datasets, packaged environments, caches, and generated outputs are excluded by `.gitignore`. Download `model/annotator/` from Hugging Face or Baidu Netdisk. Inference requires models, annotators, and test inputs, but not the training datasets.
