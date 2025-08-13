# 使用xinzhu的CombineDataset方式

import argparse
import datetime
from faulthandler import is_enabled
import logging
import inspect
import math
import os
import random
import gc
import copy
import pathlib

from typing import Dict, Optional, Tuple, List

# from sympy import print_fcode
from omegaconf import OmegaConf
from itertools import cycle

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
import torchvision
import numpy as np
from PIL import Image

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
# from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from einops import rearrange, repeat

from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset, CombineDataset

import sys
sys.path.append("/root/paddlejob/workspace/project/add_temporal_loss_scripts/model")
from model.annotator.uniformer import UniformerDetector
from model.annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler   
)

from model.diffusion.models.adapter import T2IAdapter, MultiAdapter
from model.diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from transformers import DPTForDepthEstimation
from model.annotator.hed import HEDNetwork
from model.diffusion.pipelines.pipeline_stable_diffusion_adapter3d import StableDiffusionAdapter3DPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from model.loss.temporal_loss.CCPL import CCPL
from model.loss.temporal_loss.CFC_loss import CFC

# torch.autograd.set_detect_anomaly(True)

already_printed_unet_trainables = False
already_printed_controlnet_trainables = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator, file_name):
    logging.basicConfig(
        filename=file_name,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def get_train_dataset(dataset_types, train_data, tokenizer):
    train_datasets = []
    print("dataset_types: {}".format(dataset_types))
    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            # print(dataset)
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))
    # print(f"dataset length: {len(train_datasets)}")
    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")

def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)

class CombinedDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        #self.iterators = [iter(loader) for loader in self.dataloaders]

    def __len__(self):
        return sum([len(loader) for loader in self.dataloaders])

    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        for i in range(len(self)):
            try:
                index =  i%len(iterators)
                loader = iterators[index]
                yield next(loader)
            except StopIteration:
                iterators.pop(index)
                continue

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_frames[0].size)
    for image in video_frames:
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()

def create_output_folders(accelerator, output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    training_log = os.path.join(out_dir, "training.log")
    if accelerator.is_local_main_process:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(f"{out_dir}/samples", exist_ok=True)
        OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
        pathlib.Path(training_log).touch()

    return out_dir, training_log

class ImageTrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        image_list = os.listdir(self.image_paths)
        self.image_list = []
        for i in range(len(image_list)):
            _, file_last = os.path.splitext(image_list[i])
            if file_last == '.jpg' or file_last == '.png' or file_last == '.jpeg' or file_last == '.JPEG':
                self.image_list.append(image_list[i])
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_dir = os.path.join(self.image_paths, image_name)
        image = Image.open(image_dir).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def load_sd_models(pretrained_sd_model_path, if_load_checkpoint, checkpoint_dir, checkpoint_2d_dir):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_sd_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_sd_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_sd_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_sd_model_path, subfolder="vae")
    if if_load_checkpoint:
        print("load unet checkpoint")
        unet = UNetPseudo3DConditionModel.from_pretrained(
            checkpoint_dir,
            subfolder='unet',
        )
    else:
        print("load unet pretrained")
        unet_path = os.path.join(checkpoint_2d_dir, "unet")
        unet = UNetPseudo3DConditionModel.from_2d_model(unet_path)
    
    return noise_scheduler, tokenizer, text_encoder, vae, unet

def load_adapter_models(control_mode, adapter_checkpoint_dir):
    all_annotator_model = []
    all_adapter = []

    for i in range(len(control_mode)):
        control_this_mode = control_mode[i]
        if adapter_checkpoint_dir == False:
            adapter_this_checkpoint_dir = False
        else:
            adapter_this_checkpoint_dir = adapter_checkpoint_dir[i]

        print(i, control_this_mode, adapter_this_checkpoint_dir)

        if control_this_mode == 'depth':
            annotator_model = DPTForDepthEstimation.from_pretrained('./pretrained_models/dpt-hybrid-midas')
        elif control_this_mode == 'canny':
            annotator_model = None
        elif control_this_mode == 'style':
            annotator_model = None
        elif control_this_mode == 'seg':
            annotator_model = None
        else:
            annotator_model = HEDNetwork('./pretrained_models/hed-network.pth')

        if adapter_this_checkpoint_dir:
            adapter = T2IAdapter.from_pretrained(os.path.join(adapter_this_checkpoint_dir))
        else:
            adapter = T2IAdapter(
                in_channels=3,
                channels=(320, 640, 1280, 1280),
                num_res_blocks=2,
                downscale_factor=8,
                adapter_type="full_adapter",
            )

        all_annotator_model.append(annotator_model)
        all_adapter.append(adapter)

    all_adapter = MultiAdapter(all_adapter)
    return all_annotator_model, all_adapter

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def unet_and_controlnet_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(unet, value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None:
            model.requires_grad_(False)

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet, controlnet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')

        if enable_xformers_memory_efficient_attention and not is_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                # from xformers.ops import memory_efficient_attention
                # from xformers.ops import MemoryEfficientAttentionOp
                # unet.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionOp)
                # controlnet.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionOp)
                unet.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionFlashAttentionOp)
                controlnet.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionFlashAttentionOp)
                # unet.enable_xformers_memory_efficient_attention(memory_efficient_attention)
                # controlnet.enable_xformers_memory_efficient_attention(memory_efficient_attention)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # if enable_torch_2_attn and is_torch_2:
        #     set_torch_2_attn(unet)
        
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, negation=None):
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        "negation": negation
    }
    
def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }

    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, negation = optim.values()
        # Check if we are doing LoRA training.
        # if is_lora and condition: 
        #     params = create_optim_params(
        #         params=itertools.chain(*model), 
        #         extra_params=extra_params
        #     )
        #     optimizer_params.append(params)
        #     continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                # should_negate = 'lora' in n
                # if should_negate: continue
                if p.requires_grad == True:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)

    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def cast_temporal_params_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None:
            for name, module in model.named_modules():
                if "_temporal" in name:
                    module.to(accelerator.device, dtype=weight_dtype)

def handle_cache_latents(
        should_cache, 
        output_dir, 
        train_dataloader, 
        train_batch_size, 
        vae, 
        cached_latent_dir=None
    ):

    # Cache latents by storing them in VRAM. 
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()

    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None 
        )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path =  f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            for k, v in batch.items(): batch[k] = v[0]
        
            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir


    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir), 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=0
    )

def handle_unet_trainable_modules(model, is_enabled=True):
    global already_printed_unet_trainables
    
    all_params = 0
    unfrozen_params = 0
    for name, module in model.named_modules():
        print ("module name:", name)
        all_params += len(list(module.parameters()))
        # if "_temporal" in name or "attn1" in name or "attn2" in name:
        # if "_temporal" in name or "conv_out" in name:
        # if "temporal" in name:   
        #      # unfreeze全模块
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        if ("temporal" in name) and ("down_blocks" not in name):         
             # unfreeze全模块       
            module.requires_grad_(is_enabled)
            unfrozen_params += len(list(module.parameters()))
        #!  之前默认的unfreeze设置      
        # if ("down_blocks.3" in name) and ("conv_temporal" in name):
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        # elif ("mid_block.attentions" in name) and ("attn_temporal" in name):
        #     # unfreeze mid_block的的temporal atten层
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        # if ("up_blocks" in name) and ("attn_temporal" in name):
        #     # unfreeze mid_block的的temporal atten层
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        # elif ("mid_block" in name) and ("conv_temporal" in name):
        #     # unfreeze mid_block的的temporal atten层
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        # elif ("up_blocks" in name) and ("conv_temporal" in name):
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))
        # elif ("conv_out" in name) and ("conv_temporal" in name):
        #     module.requires_grad_(is_enabled)
        #     unfrozen_params += len(list(module.parameters()))

    # import pdb; pdb.set_trace()
    if unfrozen_params > 0 and not already_printed_unet_trainables:
        already_printed_unet_trainables = True 
        print(f"Unet3D all {all_params} params, unfreeze {unfrozen_params} params.")
    
def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        annotator_model,
        adapter,
        output_dir,
        is_checkpoint=False,
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    u_dtype, t_dtype, v_dtype, a_dtype = unet.dtype, text_encoder.dtype, vae.dtype, adapter.dtype
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet, keep_fp32_wrapper=False))
    adapter_out = copy.deepcopy(accelerator.unwrap_model(adapter, keep_fp32_wrapper=False))

    pipeline = StableDiffusionAdapter3DPipeline.from_pretrained(
        path,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet, 
        annotator_model=annotator_model,
        adapter=adapter,
    )
    pipeline.save_pretrained(save_path)
    
    if is_checkpoint:
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype), (adapter, a_dtype),]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")
    del pipeline
    del unet_out
    del adapter_out

def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 

def prepare_latents(
        batch_size,
        num_channels_latents,
        clip_length,
        height,
        width,
        vae_scale_factor,
        dtype,
        device,
        generator,
        scheduler,
        latents=None,
    ):  
        if clip_length > 0:
            shape = (
                batch_size,
                num_channels_latents,
                clip_length,
                height // vae_scale_factor,
                width // vae_scale_factor,
            )
        else:
            shape = (
                batch_size,
                num_channels_latents,
                height // vae_scale_factor,
                width // vae_scale_factor,
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
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
        return latents

def encode_prompt(
        tokenizer, text_encoder, prompt, device, num_images_per_prompt, do_uncond, negative_prompt
    ):
    r"""
    Encodes the prompt into text encoder hidden states.
    Args:
        prompt (`str` or `list(int)`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_uncond (`bool`):
            whether to set the encoded text to be null to train the unconditional gudidance
        negative_prompt (`str` or `List[str]`):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
    """
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
    if (hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = text_encoder(
        text_input_ids.to(text_encoder.device),    # FIXME 强制对齐device的位置
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    # print(f"solve text_embedding {text_embeddings.shape}")
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings
    if do_uncond:
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

        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if (hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask):
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(text_encoder.device), # 同上，强制位置对齐。
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For training, we need to set the text prompt to be null for unconditional generation
        text_embeddings = uncond_embeddings

    return text_embeddings

def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def get_image_hed_map(annotator_model, images):
    images = (images + 1) / 2

    bgr_images = images.clone()
    bgr_images[:, 0, :, :] = images[:, 2, :, :]
    bgr_images[:, 2, :, :] = images[:, 0, :, :]

    edge = annotator_model(bgr_images)
    out_edge = images.clone()
    out_edge[:, 0, :, :] = edge[:, 0, :, :]
    out_edge[:, 1, :, :] = edge[:, 0, :, :]
    out_edge[:, 2, :, :] = edge[:, 0, :, :]
    return out_edge

def get_image_depth_map(annotator_model, images, height, width, return_standard_norm=False):
    h,w = height, width
    inputs = torch.nn.functional.interpolate(
        images,
        size=(384, 384),
        mode="bicubic",
        antialias=True,
    )
    inputs = inputs.to(dtype=annotator_model.dtype, device=annotator_model.device)

    outputs = annotator_model(inputs)
    predicted_depths = outputs.predicted_depth

    # interpolate to original size
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

def get_image_canny_map(images):
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

def get_hed_map(annotator_model, input_frames):
    b, f, c, h, w = input_frames.shape
    frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
    output = get_image_hed_map(annotator_model, frames)
    output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
    return output

def get_depth_map(annotator_model, input_frames, height, width, return_standard_norm=False):
    b, f, c, h, w = input_frames.shape
    frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
    output = get_image_depth_map(annotator_model, frames, height, width)
    output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
    return output

def get_canny_map(input_frames):
    b, f, c, h, w = input_frames.shape
    frames = rearrange(input_frames, 'b f c h w -> (b f) c h w')
    output = get_image_canny_map(frames)
    output = rearrange(output, "(b f) c h w -> b f c h w", f=f)
    return output

def get_seg_map(annotator_model, input_frames):
    b, f, c, h, w = input_frames.shape
    dtype = images.dtype
    device = input_frames.device
    
    frames = rearrange(input_frames, 'b f c h w -> (b f) h w c').cpu().numpy()
    frames = (frames + 1) * 255 / 2

    control_maps = np.stack([annotator_model(np.uint8(frames[inp])) for inp in range(frames.shape[0])])
    control_maps = rearrange(control_maps, 'b h w c ->b f c h w', f=f)

    control_maps = torch.from_numpy(control_maps).div(255).to(dtype).to(device)

    return control_maps

def get_memory():
    import shlex
    import subprocess
    output = subprocess.check_output(
        shlex.split(
            'nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader'
        )
    )
    
    memory_usage = output.decode().split('\n')
    memory_usage = [int(m) for m in memory_usage if m != '']
    mem_all = 0
    for m in memory_usage:
        mem_all += m
    mem_mean = mem_all / len(memory_usage)
    return memory_usage, mem_mean

def numpy_to_pil(images):
    if len(images.shape)==5:
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().float().numpy()
        images = rearrange(images, "b f c h w -> b f h w c")
        pil_images = []
        for sequence in images:
            pil_images.append(DiffusionPipeline.numpy_to_pil(sequence))
        return pil_images
    else:
        return DiffusionPipeline.numpy_to_pil(images)

def main(
    control_mode: List,
    adapter_conditioning_scale: List,
    pretrained_sd_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    dataset_types: Tuple[str] = ('json'),
    validation_steps: int = 100,
    extra_unet_params = None,
    extra_text_encoder_params = None,
    extra_adapter_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    ##!! gradient_checkpointing: bool = False,
    gradient_checkpointing: bool = True,
    ##!! text_encoder_gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = True,
    adapter_gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    train_text_encoder: bool = False,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    prob_uncond = 0.2,
    init_noise_by_residual_thres = 0.1,
    train_num_workers = 4,
    if_load_checkpoint = False,
    checkpoint_dir = None,
    adapter_checkpoint_dir = False,
    use_adapter = False,
    checkpoint_2d_dir = None,
    weight_content=2,
    weight_cfc=0.5,
    **kwargs
):
    print("mixed_precision: {}".format(mixed_precision))
    print("train_text_encoder: {}".format(train_text_encoder))
    print("enable_xformers_memory_efficient_attention: {}".format(enable_xformers_memory_efficient_attention))
    print("width: {}".format(train_data.width))
    print("height: {}".format(train_data.height))
    *_, config = inspect.getargvalues(inspect.currentframe())
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=output_dir
    )

    # Handle the output folder creation
    output_dir, logging_file = create_output_folders(accelerator, output_dir, config)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator, logging_file)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Load scheduler, tokenizer and models.

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_sd_models(pretrained_sd_model_path, if_load_checkpoint, checkpoint_dir, checkpoint_2d_dir)
    annotator_model, adapter = load_adapter_models(control_mode, adapter_checkpoint_dir)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet, adapter])

    for anno_i in range(len(annotator_model)):
        freeze_models([annotator_model[anno_i]])

    # Enable xformers if available
    # handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet, controlnet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    
    if 'seg' in control_mode:
        seg_model = OneformerCOCODetector()

    # unfreeze unet and controlnet layers
    already_printed_trainables = False
    unet.train()
    handle_unet_trainable_modules(unet, is_enabled=True)
    # import pdb; pdb.set_trace()

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    optim_params = [
        param_optim(unet, True, extra_params=extra_unet_params),
    ]

    params = create_optimizer_params(optim_params, learning_rate)
    # print(f"params:{params}")
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    '''data prepare'''      
    # Get the training dataset based on types (json, single_video, image)
    print("dataset_types: {}".format(dataset_types))
    print("train_data: {}".format(train_data))
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)
    # print(f"dataset:{train_datasets}")
    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    train_dataset_combined = CombineDataset(train_datasets)
    # train_dataset_combined = train_dataset_combined[0]
    # train_dataset_combined = train_datasets

    train_dataloader_combined = torch.utils.data.DataLoader(
        train_dataset_combined, 
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True
    )

    # Latents caching
    if cache_latents == True:
        train_dataloader_combined = handle_cache_latents(
            cache_latents, 
            output_dir,
            train_dataloader_combined, 
            train_batch_size, 
            vae,
            cached_latent_dir
        )

    val_dataset_types = ['json']
    val_datasets = get_train_dataset(dataset_types, validation_data, tokenizer)
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(val_datasets, attrs, extend=extend_dataset)
    val_datasets = val_datasets[0]

    image_transform = T.Compose([
        T.Resize((train_data.height,train_data.width)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    style_dataset = ImageTrainDataset(image_paths=train_data.style_dir, transform=image_transform)
    style_loader = torch.utils.data.DataLoader(style_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    style_iter = cycle(iter(style_loader))

    style_val_loader = torch.utils.data.DataLoader(style_dataset)
    style_val_iter = cycle(iter(style_val_loader))

    '''model training'''


    unet, adapter, annotator_model, optimizer, lr_scheduler, text_encoder, train_dataloader_combined = accelerator.prepare(
        unet, 
        adapter, 
        annotator_model,
        optimizer,
        lr_scheduler, 
        text_encoder,
        train_dataloader_combined
    )

    # Use Gradient Checkpointing if enabled. (default true in LVDM)
    unet_and_controlnet_g_c(
        unet, 
        text_encoder,
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision       
    # as these models are only used for inference, keeping weights in full precision is not required.       
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    for anno_i in range(len(annotator_model)):
        models_to_cast.append(annotator_model[anno_i])
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_combined) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tensorboard_dir = os.path.basename(output_dir)
        accelerator.init_trackers(os.path.join(tensorboard_dir, "finetune-tensorboard"))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_combined)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0         
    first_epoch = 0         

    mem, meanm = get_memory()
    print(f"before train memory: {mem}  mean:{meanm}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_videomodel(batch, style_image, train_encoder=False, weight_content=2, weight_cfc=0.5):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        # Convert videos to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        style_image = style_image.to(weight_dtype)
        bsz = pixel_values.shape[0]
        if not cache_latents:
            latents = tensor_to_vae_latent(pixel_values, vae)
        else:
            latents = pixel_values
        
        # Get video length
        video_length = latents.shape[2]
        
        # Encode input prompt
        uncond_tokens = [""] * pixel_values.size(0)
        uncond_input = tokenizer(uncond_tokens,padding="max_length",max_length=tokenizer.model_max_length,truncation=True,return_tensors="pt",)
        if (hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask):
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(text_encoder.device), # 同上，强制位置对齐。
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, 1, 1)
        uncond_embeddings = uncond_embeddings.view(pixel_values.size(0) * 1, seq_len, -1)

        # Prepare noise and calculate added noise
        num_channels_latents = unet.in_channels
        generator = [torch.Generator(device="cuda").manual_seed(seed) for i in range(bsz)]  
        noise_latents = prepare_latents(
            batch_size=bsz,
            num_channels_latents=num_channels_latents,
            clip_length=video_length,
            height=train_data.height,
            width=train_data.width,
            vae_scale_factor=vae_scale_factor,
            dtype=uncond_embeddings.dtype,
            device=latents.device,
            generator=generator,
            scheduler=noise_scheduler
        )
        noise_latents_dtype = noise_latents.dtype

        image_residual_mask = torch.ones_like(noise_latents)
        if video_length > 1 and len(noise_latents.shape) == 5 and init_noise_by_residual_thres > 0.0 and pixel_values is not None:
            pixel_values = pixel_values.to(device=latents.device, dtype=noise_latents_dtype)  # b c f h w
            image_residual = torch.abs(pixel_values[:,1:,:,:,:] - pixel_values[:,:-1,:,:,:])        
            # pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            
            # norm residual
            image_residual = image_residual / torch.max(image_residual)
            image_residual = rearrange(image_residual, "b f c h w -> (b f) c h w")
            image_residual = torch.nn.functional.interpolate(
                        image_residual, 
                        size=(noise_latents.shape[-2], noise_latents.shape[-1]),
                        mode='bilinear')
            image_residual = torch.mean(image_residual, dim=1)
            image_residual_mask = (image_residual > init_noise_by_residual_thres).float()
            image_residual_mask = repeat(image_residual_mask, '(b f) h w -> b f h w', b=bsz)    
            image_residual_mask = repeat(image_residual_mask, 'b f h w -> b c f h w', c=latents.shape[1])   
        begin_frame = 1     
        print(f"noise:{noise_latents.shape}")
        print(f"image_residual_mask:{image_residual_mask.shape}")
        for n_frame in range(begin_frame, video_length):
            noise_latents[:,:, n_frame, :, :] = \
                (noise_latents[:,:, n_frame, :, :] - noise_latents[:,:, n_frame-1, :, :]) \
                * image_residual_mask[:,:, n_frame-1, :, :] + noise_latents[:,:, n_frame-1, :, :]

        full_control_frames = []
        if use_adapter:
            for i in range(len(control_mode)):
                control_this_mode = control_mode[i]
                this_annotator_model = annotator_model[i]
                if control_this_mode == 'hed':
                    print(i, 'hed')
                    control_image = get_hed_map(this_annotator_model, pixel_values)
                elif control_this_mode == 'depth':
                    print(i, 'depth')
                    control_image = get_depth_map(this_annotator_model, pixel_values, train_data.height, train_data.width, return_standard_norm=False)
                elif control_this_mode == 'seg':
                    print(i, 'seg')
                    control_image = get_seg_map(seg_model, pixel_values)
                elif control_this_mode == 'canny':
                    print(i, 'canny')
                    control_image = get_canny_map(pixel_values)
                elif control_this_mode == 'style':
                    print(i, 'style')
                    control_image = torch.zeros_like(pixel_values)
                    for f_num in range(control_image.shape[1]):
                        control_image[:,f_num,:,:,:] = style_image
                else:
                    print(i, 'else')
                    control_image = pixel_values
                full_control_frames.append(control_image.to(weight_dtype))
            
            adapter_input = []

            for one_frame in full_control_frames:
                one_frame = rearrange(one_frame, 'b f c h w -> (b f) c h w')
                one_frame = one_frame.to(device=latents.device, dtype=adapter.dtype)
                adapter_input.append(one_frame)

            down_block_additional_residuals = adapter(adapter_input, adapter_conditioning_scale)
            down_block_additional_residuals = [rearrange(sample, "(b f) c h w -> b c f h w", f=pixel_values.shape[1]).to(dtype=weight_dtype) for sample in down_block_additional_residuals]
            # down_block_additional_residuals = [rearrange(sample, "(b f) c h w -> b c f h w", f=pixel_values.shape[1]) for sample in down_block_additional_residuals]
        else:
            down_block_additional_residuals = None
        #! Prepare timesteps
        noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps, device=latents.device)

        # Sample a random timestep for each video
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # print(f"scheduler:{noise_scheduler.num_train_timesteps} timesteps:{timesteps}")

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise_latents, timesteps)

        # Prepare extra step kwargs.
        extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, eta=0.0)

        # Prediction noise
        latent_model_input = noise_scheduler.scale_model_input(noisy_latents, timesteps)

        content_feature = unet.get_content_feature(pixel_values, vae)
        style_feature = unet.get_style_feature(style_image)

        if noise_scheduler.prediction_type == "epsilon":
            target = noise_latents
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise_latents, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

        noise_pred = unet(
            latent_model_input,
            timesteps,
            content=content_feature,
            style=style_feature,
            encoder_hidden_states=uncond_embeddings,
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample


        use_content_consistency = True  # 使用image content consistency loss
        if use_content_consistency:
            bsz = latents.shape[0]      
            f = latents.shape[2]        
            # 逐帧预测      
            latent_model_input_single_frame = rearrange(latent_model_input, 'b c f h w -> (b f) c h w')     
            text_embeddings_single_frame = torch.cat([uncond_embeddings] * f, dim=0)

            down_block_additional_residuals_single = [
                rearrange(sample, 'b c f h w -> (b f) c h w').to(dtype=weight_dtype) for sample in down_block_additional_residuals
            ]
            content_single_feature = rearrange(content_feature, 'b c f h w -> (b f) c h w')

            noise_pred_single_frame = unet(
                latent_model_input_single_frame,
                timesteps,
                content=content_single_feature,
                style=style_feature,
                encoder_hidden_states=text_embeddings_single_frame,
                down_block_additional_residuals=down_block_additional_residuals_single,
            ).sample
            noise_pred_single_frame = rearrange(noise_pred_single_frame, '(b f) c h w -> b c f h w', f=f)   
        
        if video_length > 1:    
            # loss = F.mse_loss(noise_pred[:,:,1:,:,:].float(), target[:,:,1:,:,:].float(), reduction="mean")   
            loss_orig = F.mse_loss(noise_pred[:,:,:,:,:].float(), target[:,:,:,:,:].float(), reduction="mean") #! 改为加在全部frame上
            loss_content = F.mse_loss(noise_pred[:,:,:,:,:].float(), noise_pred_single_frame[:,:,:,:,:].float(), reduction="mean")
        else:   
            # print(f"image loss: {noise_pred.shape} target: {target.shape}")
            loss_orig = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            # loss_content = F.mse_loss(noise_pred.float(), noise_pred_single_frame.float(), reduction="mean")
            loss_content = F.mse_loss(noise_pred.float(), noise_pred_single_frame.detach().float(), reduction="mean")    # 完全freeze单帧预测结果
        
        
        do_frame_smooth = True      
        if do_frame_smooth == True:     
            # import pdb; pdb.set_trace()       
            if video_length > 1:    
                """ Args:   
                prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images): Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the denoising loop.
                pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images): The predicted denoised sample (x_{0}) based on the model output from the current timestep. `pred_original_sample` can be used to preview progress or for guidance.
                """     
                # import pdb; pdb.set_trace()
                step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents, **extra_step_kwargs)
                # step_dict = noise_scheduler.step(noise_pred, timesteps, latents)
                # latents = step_dict.prev_sample
                pred_original_sample = step_dict.pred_original_sample   # 预测的逐帧图像
                pred_original_sample = pred_original_sample[:, :, 1:, :, :] # shape: [b,c,n,h,w]                        
                
                # 加ccpl约束    
                # latents shape: [b,c,n,h,w]            
                # import pdb; pdb.set_trace()           
                ccpl = CCPL(mlp="mlp")     
                num_neg = 8     
                loss_ccpl = ccpl(latents[:,:, 1:,:,:], pred_original_sample, num_neg)
                
                # CFC loss
                cfc = CFC(mlp="mlp")
                loss_cfc = cfc(latents[:,:, 1:,:,:], pred_original_sample)

        print("ccpl loss: %2f, cfc loss: %2f"%(loss_ccpl, loss_cfc))
        
        weight_orig = 0.01
        
        # weight_content = 10
        # weight_content = 2
        
        weight_ccpl = 10  

        # weight_cfc = 5
        
        # 进行loss数值clamp
        do_loss_clamp = True            
        # max_clamp_value = 0.8         
        max_clamp_value = 0.7   
        # max_clamp_value = 0.6    
        if do_loss_clamp and (loss_cfc > max_clamp_value):           
            loss_cfc = torch.clamp(loss_cfc, 0, max_clamp_value)     
            this_weight_cfc = 0.1
        else:
            this_weight_cfc = weight_cfc

        print('weight_content',weight_content,'this_weight_cfc',this_weight_cfc)
            
        loss = weight_orig * loss_orig + weight_content * loss_content + \
                weight_ccpl * loss_ccpl + this_weight_cfc * loss_cfc
        # import pdb; pdb.set_trace()       
        loss_container = [loss_orig, loss_content, loss_ccpl, loss_cfc]       
  
        return loss, loss_container         
        
    for epoch in range(first_epoch, num_train_epochs):      
        train_loss = 0.0        
        train_orig_loss = 0.0  
        train_content_loss = 0.0  
        train_ccpl_loss = 0.0  
        train_cfc_loss = 0.0  
        
        for step, batch in enumerate(train_dataloader_combined):
            style_image = next(style_iter)
            style_image = style_image.to(batch['pixel_values'].device)

            print("load data shape:{}".format(batch['pixel_values'].shape))
            print("load style shape:{}".format(style_image.shape))
            
            with accelerator.accumulate(unet):
                text_prompt = batch['text_prompt'][0]
                
                with accelerator.autocast():
                    # loss, hint = finetune_videocontrolnet(batch, train_encoder=train_text_encoder)
                    loss, loss_container = finetune_videomodel(batch, style_image, train_encoder=train_text_encoder, weight_content=weight_content, weight_cfc=weight_cfc)
                
                # Gather the losses across all processes for logging (if we use distributed training).      
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps # total loss            

                orig_loss = loss_container[0]; content_loss = loss_container[1];
                ccpl_loss = loss_container[2]; cfc_loss = loss_container[3]; 
                avg_orig_loss = accelerator.gather(orig_loss.repeat(train_batch_size)).mean()           
                train_orig_loss += avg_orig_loss.item() / gradient_accumulation_steps # orig loss           
                avg_content_loss = accelerator.gather(content_loss.repeat(train_batch_size)).mean()         
                train_content_loss += avg_content_loss.item() / gradient_accumulation_steps # content loss      
                avg_ccpl_loss = accelerator.gather(ccpl_loss.repeat(train_batch_size)).mean()
                train_ccpl_loss += avg_ccpl_loss.item() / gradient_accumulation_steps # ccpl loss
                
                avg_cfc_loss = accelerator.gather(cfc_loss.repeat(train_batch_size)).mean()
                train_cfc_loss += avg_cfc_loss.item() / gradient_accumulation_steps # cfc loss

                if train_cfc_loss > 0.7:
                    print(step, cfc_loss, train_cfc_loss)
                    import pdb; pdb.set_trace()
                # Backpropagate         
                try:
                    # import pdb; pdb.set_trace()
                    accelerator.backward(loss)
                    params_to_clip = (list(unet.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                    optimizer.step()
                    lr_scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}") 
                    continue

            # Checks if the accelerator has performed an optimization step behind the scenes        
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                #! 通过accelerator.log可用tensorboard保存loss信息
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"train_orig_loss": train_orig_loss}, step=global_step)
                accelerator.log({"train_content_loss": train_content_loss}, step=global_step)       
                accelerator.log({"train_ccpl_loss": train_ccpl_loss}, step=global_step)
                accelerator.log({"train_cfc_loss": train_cfc_loss}, step=global_step)
                train_loss = 0.0
                train_orig_loss = 0.0
                train_content_loss = 0.0
                train_ccpl_loss = 0.0
                train_cfc_loss = 0.0
                
                if global_step % checkpointing_steps == 0:      
                    # accelerator.wait_for_everyone()           
                    # if accelerator.is_main_process:   
                    save_pipe(  
                        pretrained_sd_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        text_encoder, 
                        vae, 
                        annotator_model,
                        adapter,
                        output_dir, 
                        is_checkpoint=True
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if global_step == 1: print("Performing validation prompt.")
                    
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            unet.eval()

                            unet_and_controlnet_g_c(unet, text_encoder, False, False)
                            print("validation")
                            pipeline = StableDiffusionAdapter3DPipeline.from_pretrained(                 
                                pretrained_sd_model_path,
                                vae=vae,
                                text_encoder=text_encoder,
                                unet=unet, 
                                adapter=adapter,
                                annotator_model=annotator_model,
                            )
                            
                            diffusion_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler
                            
                            with torch.no_grad():
                                idx = random.choice(range(len(val_datasets)))
                                val_sample = val_datasets[idx]
                                while(val_sample['pixel_values'].shape[0] <= 1):
                                    idx = random.choice(range(len(val_datasets)))
                                    val_sample = val_datasets[idx]
                                input_frames = val_sample['pixel_values'].unsqueeze(0).to(weight_dtype).to(unet.device)
                                # input_frames = rearrange(val_sample['pixel_values'].unsqueeze(0), 'b f c h w -> b c f h w ')    
                                prompt = ""
                                cur_dataset_name = val_sample['dataset']
                                save_filename = f"{global_step}_dataset-{cur_dataset_name}"

                                style_val_image = next(style_val_iter)
                                style_val_image = style_val_image.to(input_frames.device).to(weight_dtype)

                                print(input_frames.device,style_val_image.device)

                                full_control_image = []
                                for i in range(len(control_mode)):
                                    control_this_mode = control_mode[i]
                                    if control_this_mode == 'hed':
                                        control_image = pipeline.get_hed_map(input_frames, i)
                                    elif control_this_mode == 'depth':
                                        control_image = pipeline.get_depth_map(input_frames, train_data.height, train_data.width, i, return_standard_norm=False)
                                    elif control_this_mode == 'seg':
                                        control_image = pipeline.get_seg_map(input_frames)
                                    elif control_this_mode == 'canny':
                                        control_image = pipeline.get_canny_map(input_frames)
                                    elif control_this_mode == 'style':
                                        control_image = torch.zeros_like(input_frames)
                                        for f_num in range(control_image.shape[1]):
                                            control_image[:,f_num,:,:,:] = style_image
                                    else:
                                        control_image = input_frames
                                    full_control_image.append(control_image.to(weight_dtype))
                                
                                print(f"frame:{input_frames.shape}")
                                video_frames1 = pipeline(
                                    prompt='',
                                    frames=full_control_image,
                                    width=validation_data.width,
                                    height=validation_data.height,
                                    clip_length=input_frames.shape[1],
                                    num_inference_steps=20,
                                    guidance_scale=100,
                                    adapter_conditioning_scale=adapter_conditioning_scale,
                                    generator=[torch.Generator(device="cuda").manual_seed(seed)],
                                    content=input_frames,
                                    style=style_val_image,
                                    content_scale=0.75,
                                    style_scale=0.75,
                                    use_adapter=True,
                                ).images[0]
                                video_frames2 = pipeline(
                                    prompt='',
                                    frames=full_control_image,
                                    width=validation_data.width,
                                    height=validation_data.height,
                                    clip_length=input_frames.shape[1],
                                    num_inference_steps=20,
                                    guidance_scale=100,
                                    adapter_conditioning_scale=adapter_conditioning_scale,
                                    generator=[torch.Generator(device="cuda").manual_seed(seed)],
                                    content=input_frames,
                                    style=style_val_image,
                                    content_scale=0.4,
                                    style_scale=1.0,
                                    use_adapter=True,
                                ).images[0]
                                print(f"frames:{len(video_frames1)}")

                            try:
                                if len(video_frames1) == 1:
                                    out_file = f"{output_dir}/samples/{save_filename}.jpg"
                                    image = video_frames1[0]
                                    image.save(out_file)
                                else:
                                    out_file = f"{output_dir}/samples/{save_filename}_output1.mp4"
                                    export_to_video(video_frames1, out_file, 4)

                                    out_file = f"{output_dir}/samples/{save_filename}_output2.mp4"
                                    export_to_video(video_frames2, out_file, 4)

                                    video_content = numpy_to_pil(input_frames)
                                    out_file = f"{output_dir}/samples/{save_filename}_content.mp4"
                                    export_to_video(video_content[0], out_file, 4)

                                    out_file = f"{output_dir}/samples/{save_filename}_style.jpg"
                                    image_tensor = style_val_image[0,:,:,:].cpu()
                                    image_array = ((image_tensor.cpu().numpy() + 1) / 2 * 255).astype('uint8')
                                    pil_image = Image.fromarray(image_array.transpose(1,2,0))
                                    pil_image.save(out_file)
                            
                            except Exception as e:
                                print("export_to_video error, skip ...")

                            del pipeline
                            torch.cuda.empty_cache()
                            unet.train()
                        logger.info(f"Saved a new sample to {out_file}")

                        unet_and_controlnet_g_c(
                            unet, 
                            text_encoder,
                            gradient_checkpointing, 
                            text_encoder_gradient_checkpointing
                        )

            # logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}           
            # accelerator.log({"training_loss": loss.detach().item()}, step=step)
            logs = {"step_total_loss": loss.detach().item(),
                    "step_orig_loss": orig_loss.detach().item(),
                    "step_content_loss": content_loss.detach().item(),
                    "step_ccpl_loss": ccpl_loss.detach().item(),
                    "step_cfc_loss": cfc_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item(),
                            "step_orig_loss": orig_loss.detach().item(),
                            "step_content_loss": content_loss.detach().item(),
                            "step_ccpl_loss": ccpl_loss.detach().item(),
                            "step_cfc_loss": cfc_loss.detach().item(),
                            }, step=step)
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            pretrained_sd_model_path, 
            global_step, 
            accelerator, 
            unet, 
            text_encoder, 
            vae, 
            annotator_model,
            adapter,
            output_dir, 
            is_checkpoint=False
        )    
    accelerator.end_training()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/control_a_video_train_with_ch_pexels.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))