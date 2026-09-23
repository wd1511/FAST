"""Image style transfer using local FAST checkpoints (adapted from HiCAST)."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def make_parser(video=False):
    parser = argparse.ArgumentParser(description=__doc__)
    data = "test_video_data1" if video else "test_image_data"
    parser.add_argument("--content-dir", type=Path, default=Path("test_data") / data / "content")
    parser.add_argument("--style-dir", type=Path, default=Path("test_data") / data / "style")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video" if video else "outputs/image"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/video_model" if video else "checkpoint/image_model"))
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained_models/stable-diffusion-v1-5"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512 if video else 384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--limit", type=int, default=0, help="Number of content inputs; 0 means all")
    parser.add_argument("--content-scale", type=float, default=0.8 if video else 1.0)
    parser.add_argument("--style-scale", type=float, default=1.0)
    parser.add_argument("--controls", nargs="+", choices=("hed", "depth", "seg"), default=["hed", "depth"] if video else ["hed", "depth", "seg"])
    parser.add_argument("--adapter-scales", nargs="+", type=float, default=None)
    return parser


def prepare(args):
    os.chdir(ROOT)
    # Keep library caches inside FAST and require the supplied local models.
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.size <= 0 or args.size % 8 or args.steps <= 0 or args.limit < 0:
        raise ValueError("size must be a positive multiple of 8; steps > 0; limit >= 0")
    if args.adapter_scales is None:
        args.adapter_scales = [0.1] * len(args.controls)
    if len(args.adapter_scales) != len(args.controls):
        raise ValueError("Provide one adapter scale per control, in the same order")
    for path in (args.model_dir / "model_index.json", args.checkpoint / "unet/config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local model: {path}; see docs/ASSETS.md")
    if "seg" in args.controls and not Path("pretrained_models/150_16_swin_l_oneformer_coco_100ep.pth").is_file():
        raise FileNotFoundError("Missing local OneFormer weights; see docs/ASSETS.md")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def input_files(folder, suffixes):
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError(f"No supported inputs in {folder}")
    return files


def image_tensor(path, size):
    from PIL import Image
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def build_pipeline(args, video=False):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, DPTForDepthEstimation
    from model.annotator.hed import HEDNetwork
    from model.diffusion.models.adapter import MultiAdapter, T2IAdapter
    if video:
        from model.diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel as UNet
        from model.diffusion.pipelines.pipeline_stable_diffusion_adapter3d import StableDiffusionAdapter3DPipeline as Pipeline
    else:
        from model.diffusion.models.unet_2d_condition import UNet2DConditionModel as UNet
        from model.diffusion.pipelines.pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline as Pipeline
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the default device")
        torch.cuda.set_device(device)
    dtype = torch.float16 if video and device.type == "cuda" else torch.float32
    local = {"local_files_only": True}
    unet = UNet.from_pretrained(str(args.checkpoint), subfolder="unet", vgg_pretrained=False, **local).to(device, dtype)
    annotators, adapters = [], []
    for mode in args.controls:
        annotator = None
        if mode == "hed":
            annotator = HEDNetwork("pretrained_models/hed-network.pth").to(device, dtype).eval()
        elif mode == "depth":
            annotator = DPTForDepthEstimation.from_pretrained("pretrained_models/dpt-hybrid-midas", **local).to(device, dtype).eval()
        annotators.append(annotator)
        adapters.append(T2IAdapter.from_pretrained(str(args.checkpoint / f"{mode}_adapter"), **local).to(device, dtype))
    pipe = Pipeline.from_pretrained(
        str(args.model_dir), unet=unet,
        vae=AutoencoderKL.from_pretrained(str(args.model_dir), subfolder="vae", **local).to(device, dtype),
        text_encoder=CLIPTextModel.from_pretrained(str(args.model_dir), subfolder="text_encoder", **local).to(device, dtype),
        tokenizer=CLIPTokenizer.from_pretrained(str(args.model_dir), subfolder="tokenizer", **local),
        scheduler=DDIMScheduler.from_pretrained(str(args.model_dir), subfolder="scheduler", **local),
        annotator_model=annotators, adapter=MultiAdapter(adapters).to(device, dtype), **local,
    ).to(device)
    return pipe, dtype


def control_maps(pipe, content, args):
    maps = []
    for index, mode in enumerate(args.controls):
        if mode == "depth":
            control = pipe.get_depth_map(content, args.size, args.size, index, return_standard_norm=False)
        elif mode == "hed":
            control = pipe.get_hed_map(content, index)
        else:
            control = pipe.get_seg_map(content)
        maps.append(control.to(content.device, content.dtype))
    return maps


def main():
    args = make_parser().parse_args()
    prepare(args)
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    contents = input_files(args.content_dir, suffixes)
    styles = input_files(args.style_dir, suffixes)
    pipe, dtype = build_pipeline(args)
    import torch
    with torch.inference_mode():
        for index, content_path in enumerate(contents[:args.limit or None]):
            style_path = styles[index % len(styles)]
            content = image_tensor(content_path, args.size).to(args.device, dtype)
            style = image_tensor(style_path, args.size).to(args.device, dtype)
            result = pipe(
                "", image=control_maps(pipe, content, args), content=content, style=style,
                height=args.size, width=args.size, num_inference_steps=args.steps,
                generator=[torch.Generator(device=args.device).manual_seed(args.seed)],
                guidance_scale=100, content_scale=args.content_scale, style_scale=args.style_scale,
                use_adapter=True, adapter_conditioning_scale=args.adapter_scales, adapter_cfg=False,
            ).images[0]
            output = args.output_dir / f"{index:03d}_{content_path.stem}_{style_path.stem}.png"
            result.save(output)
            print(output)


if __name__ == "__main__":
    main()
