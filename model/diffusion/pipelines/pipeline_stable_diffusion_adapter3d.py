# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat

from diffusers.models import AutoencoderKL
from model.diffusion.models.adapter import MultiAdapter, T2IAdapter
from model.diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""

def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image


class StableDiffusionAdapter3DPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        adapter: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        annotator_model=None,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        
        if isinstance(adapter, (list, tuple)):
            adapter = MultiAdapter(adapter)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            adapter=adapter,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.annotator_model = annotator_model
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            # import pdb; pdb.set_trace()
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @staticmethod
    def numpy_to_pil(images):
        if len(images.shape)==5:
            pil_images = []
            for sequence in images:
                pil_images.append(DiffusionPipeline.numpy_to_pil(sequence))
            return pil_images
        else:
            return DiffusionPipeline.numpy_to_pil(images)

    def decode_latents(self, latents):
        b = latents.shape[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        
        is_video = len(latents.shape) == 5
        if is_video:
            latents = rearrange(latents, "b c f h w -> (b f) c h w")

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

        image = image.cpu().float().numpy()
        if is_video:
            image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
        else:
            image = rearrange(image, "b c h w -> b h w c")
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        clip_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):  
        if clip_length>0:
            shape = (
                batch_size,
                num_channels_latents,
                clip_length,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
        else:
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)

            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(
                    device
                )
            # if len(shape) == 5 and random_each_frame == False:
            #     for i in range(clip_length):
            #         latents[:,:,i,:,:] = latents[:,:,0,:,:]
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def get_image_hed_map(self, images, num):
        images = (images + 1) / 2

        bgr_images = images.clone()
        bgr_images[:, 0, :, :] = images[:, 2, :, :]
        bgr_images[:, 2, :, :] = images[:, 0, :, :]

        this_annotator_model = self.annotator_model[num]

        edge = this_annotator_model(bgr_images)
        out_edge = images.clone()
        out_edge[:, 0, :, :] = edge[:, 0, :, :]
        out_edge[:, 1, :, :] = edge[:, 0, :, :]
        out_edge[:, 2, :, :] = edge[:, 0, :, :]
        return out_edge

    @torch.no_grad()
    def get_image_depth_map(self, images, height, width, num, return_standard_norm=False):
        h,w = height, width
        inputs = torch.nn.functional.interpolate(
            images,
            size=(384, 384),
            mode="bicubic",
            antialias=True,
        )
        this_annotator_model = self.annotator_model[num]

        inputs = inputs.to(dtype=this_annotator_model.dtype, device=this_annotator_model.device)

        outputs = this_annotator_model(inputs)
        predicted_depths = outputs.predicted_depth

        predictions = torch.nn.functional.interpolate(
            predicted_depths.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
        ).detach()

        # normalize output
        if return_standard_norm:
            depth_min = torch.amin(predictions, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(predictions, dim=[1, 2, 3], keepdim=True)
            predictions = 2.0 * (predictions - depth_min) / (depth_max - depth_min) - 1.0
        else:
            predictions -= torch.min(predictions)
            predictions /= torch.max(predictions)

        out_depth = images.clone()
        out_depth[:, 0, :, :] = predictions[:, 0, :, :]
        out_depth[:, 1, :, :] = predictions[:, 0, :, :]
        out_depth[:, 2, :, :] = predictions[:, 0, :, :]
        return out_depth

    @torch.no_grad()
    def get_seg_map(self, input_frames):
        b, f, c, h, w = input_frames.shape
        dtype = images.dtype
        device = input_frames.device
    
        frames = rearrange(input_frames, 'b f c h w -> (b f) h w c').cpu().numpy()
        frames = (frames + 1) * 255 / 2

        seg_model = OneformerCOCODetector()

        control_maps = np.stack([seg_model(np.uint8(frames[inp])) for inp in range(frames.shape[0])])
        control_maps = rearrange(control_maps, 'b h w c ->b f c h w', f=f)

        control_maps = torch.from_numpy(control_maps).div(255).to(dtype).to(device)

        return control_maps

    @torch.no_grad()
    def get_image_canny_map(self, images):
        input_images = rearrange(images, 'b c h w -> b h w c').cpu().numpy()
        input_images = (input_images + 1) * 255 / 2
        control_maps = np.stack([cv2.Canny(np.uint8(frames[inp]), 100, 200) for inp in range(frames.shape[0])])
        control_maps = repeat(control_maps, 'b h w ->b c h w', c=1)
        control_maps = torch.from_numpy(control_maps).div(255)

        out_canny = images.clone()
        out_canny[:, 0, :, :] = control_maps[:, 0, :, :]
        out_canny[:, 1, :, :] = control_maps[:, 0, :, :]
        out_canny[:, 2, :, :] = control_maps[:, 0, :, :]
        return out_canny

    @torch.no_grad()
    def get_hed_map(self, input_frames, num):
        b, f, c, h, w = input_frames.shape
        frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
        output = self.get_image_hed_map(frames, num)
        output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
        return output

    @torch.no_grad()
    def get_depth_map(self, input_frames, height, width, num, return_standard_norm=False):
        b, f, c, h, w = input_frames.shape
        frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
        output = self.get_image_depth_map(frames, height, width, num)
        output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
        return output

    @torch.no_grad()
    def get_canny_map(self, input_frames):
        b, f, c, h, w = input_frames.shape
        frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
        output = self.get_image_canny_map(frames)
        output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
        return output

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        frames: Union[torch.FloatTensor, List[torch.FloatTensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        clip_length: int = 8, # NOTE clip_length和images的帧数一致。
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        content = None,
        style = None,
        content_scale = 1.0,
        style_scale = 1.0,
        use_adapter = False,
        init_same_noise_per_frame=False,
        init_noise_by_residual_thres=0.0,
        residual_control_steps=1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # annotator b f c h w  ->  b f c h w
        # adapter (b f) c h w -> b c f h w
        # content b c f h w
        # style b c f

        if isinstance(self.adapter, MultiAdapter):
            adapter_input = []

            for one_frame in frames:
                one_frame = rearrange(one_frame, 'b f c h w -> (b f) c h w')
                one_frame = one_frame.to(device=device, dtype=self.adapter.dtype)
                adapter_input.append(one_frame)
        else:
            this_frame = rearrange(frames, 'b f c h w -> (b f) c h w')
            adapter_input = this_frame
            adapter_input = adapter_input.to(device=device, dtype=self.adapter.dtype)

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            False,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        ).to(content.dtype)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            clip_length,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        if len(latents.shape) == 5 and init_same_noise_per_frame:
            latents[:,:,1:,:,:] = latents[:,:,0:1,:,:]
            import pdb; pdb.set_trace()

        if len(latents.shape) == 5 and init_noise_by_residual_thres > 0.0 and frames is not None:
            if isinstance(self.adapter, MultiAdapter):
                one_frame = frames[0]
            else:
                one_frame = frames

            one_frame = one_frame.to(device=device, dtype=latents_dtype)  # b c f h w
            frame_residual = torch.abs(one_frame[:,1:,:,:,:] - one_frame[:,:-1,:,:,:])
            one_frame = rearrange(one_frame, "b f c h w -> (b f) c h w")

            frame_residual = frame_residual / torch.max(frame_residual)
            frame_residual = rearrange(frame_residual, "b f c h w -> (b f) c h w")

            frame_residual = torch.nn.functional.interpolate(
                        frame_residual,
                        size=(latents.shape[-2], latents.shape[-1]),
                        mode='bilinear')
            frame_residual = torch.mean(frame_residual, dim=1)

            frame_residual_mask = (frame_residual > init_noise_by_residual_thres).float()
            frame_residual_mask = repeat(frame_residual_mask, '(b f) h w -> b f h w', b=batch_size)
            frame_residual_mask = repeat(frame_residual_mask, 'b f h w -> b c f h w', c=latents.shape[1])

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        if isinstance(self.adapter, MultiAdapter) or isinstance(self.adapter, StyleAdapter):
            adapter_state = self.adapter(adapter_input, adapter_conditioning_scale)
            for k, v in enumerate(adapter_state):
                v = rearrange(v, "(b f) c h w -> b c f h w", f=content.shape[1])
                adapter_state[k] = v
        else:
            adapter_state = self.adapter(adapter_input)
            for k, v in enumerate(adapter_state):
                v = rearrange(v, "(b f) c h w -> b c f h w", f=content.shape[1])
                adapter_state[k] = v * adapter_conditioning_scale

        if num_images_per_prompt > 1:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
        
        if do_classifier_free_guidance and content != None and style != None:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = torch.cat([v, v, v],dim=0)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if content != None:
                content_feature = self.unet.get_content_feature(content, self.vae)
                null_content_feature = torch.zeros_like(content_feature)
            else:
                content_feature = None
            if style != None:
                style_feature = self.unet.get_style_feature(style)
                null_style_feature = style_feature.clone()
                null_style_feature[:] = self.unet.null_style_vector.weight[0]
            else:
                style_feature = None

            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if i<residual_control_steps and len(latents.shape) == 5 and init_noise_by_residual_thres > 0.0 and frames is not None:
                    begin_frame = 1
                    for n_frame in range(begin_frame, latents.shape[2]):
                        latents[:,:, n_frame, :, :] = \
                            (latents[:,:, n_frame, :, :] - latents[:,:, n_frame-1, :, :]) \
                            * frame_residual_mask[:,:, n_frame-1, :, :] + \
                            latents[:,:, n_frame-1, :, :]

                if do_classifier_free_guidance and content != None and style != None:
                    latent_model_input_all = torch.cat([latent_model_input, latent_model_input, latent_model_input])
                    content_feature_all = torch.cat([content_feature, content_feature, null_content_feature])
                    style_feature_all = torch.cat([style_feature, null_style_feature, style_feature])
                    prompt_embeds_all = torch.cat([prompt_embeds, prompt_embeds, prompt_embeds])       
                    if use_adapter:
                        noise_pred_all = self.unet(
                            latent_model_input_all,
                            t,
                            content=content_feature_all,
                            style=style_feature_all,
                            encoder_hidden_states=prompt_embeds_all,
                            down_intrablock_additional_residuals=[state.clone() for state in adapter_state],
                        ).sample
                    else:
                        noise_pred_all = self.unet(
                            latent_model_input_all,
                            t,
                            content=content_feature_all,
                            style=style_feature_all,
                            encoder_hidden_states=prompt_embeds_all,
                        ).sample
                    noise_pred, noise_pred_unstyle, noise_pred_uncontent = noise_pred_all.chunk(3)
                    noise_pred = noise_pred_unstyle + content_scale * (noise_pred - noise_pred_unstyle) + noise_pred_uncontent + style_scale * (noise_pred - noise_pred_uncontent) - noise_pred
                else:
                    if use_adapter:
                        noise_pred_all = self.unet(
                            latent_model_input_all,
                            t,
                            content=content_feature_all,
                            style=style_feature_all,
                            encoder_hidden_states=prompt_embeds_all,
                            down_intrablock_additional_residuals=[state.clone() for state in adapter_state],
                        ).sample
                    else:
                        noise_pred_all = self.unet(
                            latent_model_input_all,
                            t,
                            content=content_feature_all,
                            style=style_feature_all,
                            encoder_hidden_states=prompt_embeds_all,
                        ).sample
                

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            has_nsfw_concept = None

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
