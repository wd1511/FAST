# FAST

[English](README.md) | **简体中文**

**FAST: Flexibly Controllable Arbitrary Style Transfer via Latent Diffusion models** <br>
Hanzhang Wang*, Haoran Wang*, Zhongrui Yu, Mingming Sun, Junjun Jiang, Xianming Liu, **Deming Zhai** <br>
ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2025.

[论文](https://dl.acm.org/doi/abs/10.1145/3748655) · [arXiv](https://arxiv.org/pdf/2401.05870.pdf) · [项目主页](https://fast-ldm.github.io) · [GitHub](https://github.com/wd1511/FAST) · [Hugging Face](https://huggingface.co/wd1511/fast-ldm)

## 下载

| 下载入口 | 内容 |
| --- | --- |
| [Hugging Face：wd1511/fast-ldm](https://huggingface.co/wd1511/fast-ldm) | 图像和视频 checkpoint、预训练模型、annotator 实现及权重 |
| [百度网盘：model](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg)（提取码：`hcsc`） | 模型、测试与训练数据、运行环境 |

源码仓库包含代码、配置和图像/视频测试输入。模型权重、训练数据集和打包环境可从上面的链接下载。目录结构与准备命令见 [资源准备指南](docs/ASSETS_zh.md)。

## 环境要求

运行环境为 Linux + NVIDIA CUDA GPU，主要依赖如下：

- Python 3.8.0
- PyTorch 2.0.0
- torchvision 0.15.1、torchaudio 2.0.1
- diffusers 0.14.0
- xformers 0.0.17

```bash
conda env create -f environment.yml
conda activate diffusers-torch2
```

也可以使用百度网盘中的 `diffusers_env.tar.gz`，具体方法见 [环境包](docs/ASSETS_zh.md#环境包)。推理默认设备为 `cuda:0`。

## 图像与视频推理

将模型和 annotator 下载到源码仓库目录：

```bash
hf download wd1511/fast-ldm --include 'checkpoint/**' --local-dir .
hf download wd1511/fast-ldm --include 'pretrained_models/**' --local-dir .
hf download wd1511/fast-ldm --include 'model/annotator/**' --local-dir .
```

在 `FAST/` 中执行：

```bash
# 图像风格迁移：HED、深度和分割控制。
python -m test_sh.test_image --limit 1

# 视频风格迁移：HED 和深度控制，采样开头最多 21 帧。
python -m test_sh.test_video --limit 1
```

| 类型 | 内容输入 | 风格输入 | 输出 |
| --- | --- | --- | --- |
| 图像 | `test_data/test_image_data/content/` | `test_data/test_image_data/style/` | `outputs/image/` |
| 视频 | `test_data/test_video_data1/content/` | `test_data/test_video_data1/style/` | `outputs/video/` |

测试输入包括 36 张内容图和 36 张风格图，以及 12 个视频和 12 张视频风格图。内容与风格按文件名排序后依次配对；风格数量不足时循环使用。去掉 `--limit 1` 可处理全部内容输入。自定义输入目录可通过 `--content-dir` 和 `--style-dir` 指定。

```bash
# 调整内容、风格和 adapter 强度。
python -m test_sh.test_image --content-scale 1.0 --style-scale 1.2 \
  --controls hed depth --adapter-scales 0.1 0.2 --limit 1

# 设置视频分辨率和帧采样间隔。
python -m test_sh.test_video --size 384 --frames 21 --stride 4 --limit 1

python -m test_sh.test_image --help
python -m test_sh.test_video --help
```

`--adapter-scales` 的数量和顺序须与 `--controls` 一致。将全部 adapter 强度设为 0 可取消其贡献。图像默认分辨率为 384×384，视频为 512×512。视频输出包含采样帧，不含音频；输出帧率为原帧率除以实际采样间隔。短视频会根据长度缩短采样间隔和帧数。

也支持直接运行 `python test_sh/test_image.py` 和 `python test_sh/test_video.py`。相对路径均以 `FAST/` 为基准。

## 训练

训练入口位于 `train_sh/`，配置位于 `configs/`：

```bash
python -m train_sh.train --config configs/config_base.yaml
python -m train_sh.train_one_adapter --config configs/config_adapter_256.yaml
```

训练数据准备及视频训练所需的额外源码见 [训练数据与依赖](docs/ASSETS_zh.md#训练数据与依赖)。

## 引用

```bibtex
@article{wang2025fast,
  title={FAST: Flexibly Controllable Arbitrary Style Transfer via Latent Diffusion models},
  author={Wang, Hanzhang and Wang, Haoran and Yu, Zhongrui and Sun, Mingming and Jiang, Junjun and Liu, Xianming and Zhai, Deming},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  publisher={ACM New York, NY},
  year={2025}
}
```
