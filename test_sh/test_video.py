"""Video style transfer using local FAST checkpoints (adapted from HiCAST)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_sh.test_image import build_pipeline, control_maps, image_tensor, input_files, make_parser, prepare


def read_frames(path, size, num_frames, stride):
    import decord
    import numpy as np
    import torch
    from PIL import Image
    reader = decord.VideoReader(str(path))
    if len(reader) < 2:
        raise ValueError(f"Video must contain at least two frames: {path}")
    count = min(num_frames, len(reader))
    stride = min(stride, max(1, (len(reader) - 1) // (count - 1)))
    frames = reader.get_batch([i * stride for i in range(count)]).asnumpy()
    frames = np.stack([np.asarray(Image.fromarray(frame).resize((size, size))) for frame in frames])
    content = torch.from_numpy(frames).float().div(127.5).sub(1).permute(0, 3, 1, 2).unsqueeze(0)
    fps = reader.get_avg_fps() / stride
    if fps <= 0:
        raise ValueError(f"Invalid video frame rate: {path}")
    return content, fps


def save_video(frames, path, fps, size):
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video output: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    parser = make_parser(video=True)
    parser.description = __doc__
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--residual-threshold", type=float, default=0.15)
    args = parser.parse_args()
    if args.frames < 2 or args.stride < 1:
        parser.error("frames must be >= 2 and stride must be >= 1")
    prepare(args)
    contents = input_files(args.content_dir, {".mp4", ".mov", ".avi", ".mkv", ".webm"})
    styles = input_files(args.style_dir, {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    pipe, dtype = build_pipeline(args, video=True)
    import torch
    with torch.inference_mode():
        for index, content_path in enumerate(contents[:args.limit or None]):
            style_path = styles[index % len(styles)]
            content, fps = read_frames(content_path, args.size, args.frames, args.stride)
            content = content.to(args.device, dtype)
            style = image_tensor(style_path, args.size).to(args.device, dtype)
            frames = pipe(
                "", frames=control_maps(pipe, content, args), content=content, style=style,
                height=args.size, width=args.size, clip_length=content.shape[1], num_inference_steps=args.steps,
                generator=[torch.Generator(device=args.device).manual_seed(args.seed)],
                guidance_scale=100, content_scale=args.content_scale, style_scale=args.style_scale,
                use_adapter=True, adapter_conditioning_scale=args.adapter_scales,
                init_noise_by_residual_thres=args.residual_threshold, residual_control_steps=1,
            ).images[0]
            output = args.output_dir / f"{index:03d}_{content_path.stem}_{style_path.stem}.mp4"
            save_video(frames, output, fps, args.size)
            print(output)


if __name__ == "__main__":
    main()
