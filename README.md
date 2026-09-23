# FAST

**FAST: Flexibly Controllable Arbitrary Style Transfer via Latent Diffusion models** <br>
Hanzhang Wang*, Haoran Wang*, Zhongrui Yu, Mingming Sun, Junjun Jiang, Xianming Liu, **Deming Zhai** <br>
ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2025.

[Paper](https://dl.acm.org/doi/abs/10.1145/3748655) · [arXiv](https://arxiv.org/pdf/2401.05870.pdf) · [Project Homepage](https://fast-ldm.github.io) · [GitHub](https://github.com/wd1511/FAST) · [Hugging Face](https://huggingface.co/wd1511/fast-ldm)

## Downloads

| Source | Contents |
| --- | --- |
| [Hugging Face: wd1511/fast-ldm](https://huggingface.co/wd1511/fast-ldm) | Image and video checkpoints, pretrained models, and annotator implementations and weights |
| [Baidu Netdisk: model](https://pan.baidu.com/s/1q6lgOszfOS-p0OmOzhhYWg) (access code: `hcsc`) | Models, test and training datasets, and the runtime environment |

The repository includes source code, configurations, and image/video test inputs. Model weights, training datasets, and the packaged environment are available from the links above. See the [Asset Setup Guide](docs/ASSETS.md) for directory layouts and setup commands.

## Requirements

Use Linux with an NVIDIA CUDA GPU. The main dependencies are:

- Python 3.8.0
- PyTorch 2.0.0
- torchvision 0.15.1 and torchaudio 2.0.1
- diffusers 0.14.0
- xformers 0.0.17

```bash
conda env create -f environment.yml
conda activate diffusers-torch2
```

Alternatively, use `diffusers_env.tar.gz` from Baidu Netdisk; see [Packaged Environment](docs/ASSETS.md#packaged-environment). The default inference device is `cuda:0`.

## Image and Video Inference

Download the models and annotators into the repository directory:

```bash
hf download wd1511/fast-ldm --include 'checkpoint/**' --local-dir .
hf download wd1511/fast-ldm --include 'pretrained_models/**' --local-dir .
hf download wd1511/fast-ldm --include 'model/annotator/**' --local-dir .
```

Run the following commands from `FAST/`:

```bash
# Image style transfer with HED, depth, and segmentation controls.
python -m test_sh.test_image --limit 1

# Video style transfer with HED and depth controls, sampling up to 21 frames.
python -m test_sh.test_video --limit 1
```

| Mode | Content inputs | Style inputs | Outputs |
| --- | --- | --- | --- |
| Image | `test_data/test_image_data/content/` | `test_data/test_image_data/style/` | `outputs/image/` |
| Video | `test_data/test_video_data1/content/` | `test_data/test_video_data1/style/` | `outputs/video/` |

The test inputs contain 36 content images and 36 style images, plus 12 videos and 12 video-style images. Content and style files are sorted by filename and paired in order; style files repeat when there are fewer styles than content inputs. Omit `--limit 1` to process all content inputs. Use `--content-dir` and `--style-dir` for custom inputs.

```bash
# Adjust content, style, and adapter strengths.
python -m test_sh.test_image --content-scale 1.0 --style-scale 1.2 \
  --controls hed depth --adapter-scales 0.1 0.2 --limit 1

# Set the video resolution and frame sampling interval.
python -m test_sh.test_video --size 384 --frames 21 --stride 4 --limit 1

python -m test_sh.test_image --help
python -m test_sh.test_video --help
```

Provide one `--adapter-scales` value for each entry in `--controls`, in the same order. Setting all adapter scales to zero removes their contribution. The default resolution is 384×384 for images and 512×512 for videos. Video outputs contain the sampled frames without audio. Their frame rate is the source frame rate divided by the actual sampling interval; short videos use a smaller interval and fewer frames as needed.

Direct execution with `python test_sh/test_image.py` and `python test_sh/test_video.py` is also supported. Relative paths are resolved from `FAST/`.

## Training

Training entry points are in `train_sh/`, with configurations in `configs/`:

```bash
python -m train_sh.train --config configs/config_base.yaml
python -m train_sh.train_one_adapter --config configs/config_adapter_256.yaml
```

See [Training Data and Dependencies](docs/ASSETS.md#training-data-and-dependencies) for dataset preparation and additional source files required for video training.

## Citation

```bibtex
@article{wang2025fast,
  title={FAST: Flexibly Controllable Arbitrary Style Transfer via Latent Diffusion models},
  author={Wang, Hanzhang and Wang, Haoran and Yu, Zhongrui and Sun, Mingming and Jiang, Junjun and Liu, Xianming and Zhai, Deming},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  publisher={ACM New York, NY},
  year={2025}
}
```
