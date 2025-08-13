#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from omegaconf import OmegaConf
import time
from einops import rearrange, repeat

from typing import Dict, Optional, Tuple, List

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
import torchvision
import numpy as np
from itertools import cycle
from PIL import Image

from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import BasicTransformerBlock

import accelerate
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

from model.diffusion.models.unet_2d_condition import UNet2DConditionModel
from model.diffusion.models.adapter import T2IAdapter, MultiAdapter
from model.diffusion.pipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from model.diffusion.pipelines.pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from diffusers.utils import load_image
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler   
)

from model.loss.style_loss import Gram_loss
from model.loss.gan_loss import Discriminator, CooccurDiscriminator

from transformers import DPTForDepthEstimation
from model.annotator.hed import HEDNetwork
import cv2
import sys
sys.path.append("/root/paddlejob/workspace/project/hicast/model")
from model.annotator.uniformer import UniformerDetector
from model.annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")

logger = get_logger(__name__)

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

class ImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.image_list = os.listdir(self.image_paths)
        self.image_list.sort()
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

def load_sd_models(pretrained_sd_model_path, unet_pretrained, if_load_checkpoint, checkpoint_dir):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_sd_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_sd_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_sd_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_sd_model_path, subfolder="vae")
    if if_load_checkpoint:
        print("load unet checkpoint")
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_dir,
            subfolder='unet',
        )
    else:
        print("load unet pretrained")
        unet_path = os.path.join(pretrained_sd_model_path, "unet")
        unet = UNet2DConditionModel.from_2d_model(unet_path, unet_pretrained)
    
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
        elif control_this_mode == 'seg':
            annotator_model = None
        elif control_this_mode == 'style':
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

        if condition:
            for n, p in model.named_parameters():
                if p.requires_grad == True:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)

    return optimizer_params

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

def should_sample(global_step, validation_steps):
    return (global_step % validation_steps == 0 or global_step == 1)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        annotator_model,
        adapter,
        discriminator,
        cooccur,
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

    pipeline = StableDiffusionAdapterPipeline.from_pretrained(
        path,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet, 
        annotator_model=annotator_model,
        adapter=adapter,
    )
    pipeline.save_pretrained(save_path)

    torch.save(
        {
            "d": discriminator.state_dict(),
            "cooccur": cooccur.state_dict(),
        },
        os.path.join(save_path,'discriminator.pt'),
    )
    
    if is_checkpoint:
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype), (adapter, a_dtype),]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")
    del pipeline
    del unet_out
    del adapter_out

def calc_mean_std(feat, vae, eps=1e-5):
    feat = vae.decoder.conv_in(feat)
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def color2gray(image):
    gray_image = 0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]
    gray_image = gray_image.unsqueeze(1)
    gray_image = torch.cat([gray_image] * 3, dim=1)
    return gray_image

def get_random_batch(batch, style_image, uncontent_flag, unstyle_flag, data_random_rate, color_flag=True):
    train_rand = np.random.rand(batch.size(0))
    train_content_flag = train_rand <= uncontent_flag
    train_style_flag = train_rand > unstyle_flag
    train_content_flag = torch.from_numpy(train_content_flag).to(batch.device)
    train_style_flag = torch.from_numpy(train_style_flag).to(batch.device)
    for i in range(batch.size(0)):
        if not train_content_flag[i]:
            batch[i,:,:,:] = style_image[i,:,:,:]
        elif train_content_flag[i] and train_style_flag[i]:
            data_random_flag = random.random()
            if data_random_flag < data_random_rate:
                batch[i,:,:,:] = style_image[i,:,:,:]

    if not color_flag:
        batch = color2gray(batch)
    return batch, style_image, train_content_flag, train_style_flag

def get_noise_latent(batch, weight_dtype, vae, noise_scheduler):
    latents = vae.encode(batch.to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.cat([torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device)] * bsz)
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return latents, noise, noisy_latents, timesteps

def get_uncond_embeddings(batch, tokenizer, text_encoder):
    uncond_tokens = [""] * batch.size(0)
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
    uncond_embeddings = uncond_embeddings.view(batch.size(0) * 1, seq_len, -1)
    return uncond_embeddings

def get_hed_map(annotator_model, images):
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

def get_seg_map(annotator_model, images):
    dtype = images.dtype
    input_images = rearrange(images, 'b c h w -> b h w c').cpu().numpy()
    input_images = (input_images + 1) * 255 / 2
    control_maps = np.stack([annotator_model(np.uint8(input_images[inp])) for inp in range(input_images.shape[0])])
    control_maps = rearrange(control_maps, 'b h w c ->b c h w')
    control_maps = torch.from_numpy(control_maps).div(255).to(dtype)

    # out_canny = images.clone()
    # out_canny[:, 0, :, :] = control_maps[:, 0, :, :]
    # out_canny[:, 1, :, :] = control_maps[:, 0, :, :]
    # out_canny[:, 2, :, :] = control_maps[:, 0, :, :]
    return control_maps

def get_depth_map(annotator_model, images, height, width, return_standard_norm=False):
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

def get_canny_map(images):
    input_images = rearrange(images, 'b c h w -> b h w c').cpu().numpy()
    input_images = (input_images + 1) * 255 / 2
    control_maps = np.stack([cv2.Canny(np.uint8(input_images[inp]), 100, 200) for inp in range(input_images.shape[0])])
    control_maps = repeat(control_maps, 'b h w ->b c h w', c=1)
    control_maps = torch.from_numpy(control_maps).div(255)

    out_canny = images.clone()
    out_canny[:, 0, :, :] = control_maps[:, 0, :, :]
    out_canny[:, 1, :, :] = control_maps[:, 0, :, :]
    out_canny[:, 2, :, :] = control_maps[:, 0, :, :]
    return out_canny

def rgb2gray(rgb):
    x = .299 * rgb[:, 0, :, :] + .587 * rgb[:, 1, :, :] + .114 * rgb[:, 2, :, :]
    y = .299 * rgb[:, 0, :, :] + .587 * rgb[:, 1, :, :] + .114 * rgb[:, 2, :, :]
    z = .299 * rgb[:, 0, :, :] + .587 * rgb[:, 1, :, :] + .114 * rgb[:, 2, :, :]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)
    return out

def main(
    control_mode: List,
    adapter_conditioning_scale: List,
    pretrained_sd_model_path: str,
    output_dir: str,
    train_data: Dict,
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
    gradient_checkpointing: bool = True,
    text_encoder_gradient_checkpointing: bool = True,
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
    train_num_workers = 4,
    if_load_checkpoint = False,
    checkpoint_dir = None,
    adapter_checkpoint_dir = None,
    use_adapter = False,
    freeze_unet_flag = False,
    unet_pretrained = True,
    data_random_rate = 0.5,
    uncontent_flag = 0.5,
    unstyle_flag = 0.1,
    content_loss_type = "latent",
    content_loss_weight = 1.0,
    all_loss_type = "latent",
    ccp_loss_flag = False,
    ccp_loss_weight = 1.0,
    style_loss_flag = False,
    style_loss_weight = 1.0,
    gram_loss_flag = False,
    gram_loss_weight = 1.0,
    cos_loss_flag = False,
    cos_loss_weight = 1.0,
    gan_loss_flag = False,
    gan_loss_weight = 1.0,
    patch_gan_loss_weight = 1.0,
    update_dis_num = 5,
    use_r1_flag = False,
    color_flag = True,
    **kwargs):

    print("control_mode: {}".format(control_mode))
    print("mixed_precision: {}".format(mixed_precision))
    print("train_text_encoder: {}".format(train_text_encoder))
    print("enable_xformers_memory_efficient_attention: {}".format(enable_xformers_memory_efficient_attention))
    print("width: {}".format(train_data.width))
    print("height: {}".format(train_data.height))
    print("content_loss_type: {}".format(content_loss_type))
    print("all_loss_type: {}".format(all_loss_type))
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
        set_seed(seed)
    
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_sd_models(pretrained_sd_model_path, unet_pretrained, if_load_checkpoint, checkpoint_dir)
    annotator_model, adapter = load_adapter_models(control_mode, adapter_checkpoint_dir)
    
    if all_loss_type == "latent":
        discriminator = Discriminator(4, int(train_data.width / 8))
        cooccur = CooccurDiscriminator(all_loss_type, 32, size=int(train_data.width / 8))
    else:
        discriminator = Discriminator(3, train_data.width)
        cooccur = CooccurDiscriminator(all_loss_type, 32, size=train_data.width)        

    if if_load_checkpoint:
        ckpt = torch.load(os.path.join(checkpoint_dir,'discriminator.pt'), map_location=lambda storage, loc: storage)
        discriminator.load_state_dict(ckpt["d"])
        cooccur.load_state_dict(ckpt["cooccur"])

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    already_printed_trainables = False

    if 'seg' in control_mode:
        seg_model = OneformerCOCODetector()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    for anno_i in range(len(annotator_model)):
        freeze_models([annotator_model[anno_i]])

    if use_adapter:
        if freeze_unet_flag:
            freeze_models([vae, text_encoder, unet])
            adapter.train()
        else:
            freeze_models([vae, text_encoder])
            unet.train()
            adapter.train()
        g_optim_params = [
            param_optim(unet, True, extra_params=extra_unet_params),
            param_optim(adapter, True, extra_params=extra_adapter_params)
        ]
    else:
        freeze_models([vae, text_encoder, adapter])
        unet.train()
        g_optim_params = [
            param_optim(unet, True, extra_params=extra_unet_params)
        ]

    g_params = create_optimizer_params(g_optim_params, learning_rate)
    g_optim = optimizer_class(
        g_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    g_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=g_optim,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    d_reg_every = 16
    d_reg_ratio = d_reg_every/(d_reg_every+1)
    d_optim = torch.optim.Adam(
        list(discriminator.parameters()) + list(cooccur.parameters()),
        lr=learning_rate * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    d_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=d_optim,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    image_transform = T.Compose([
        T.Resize((train_data.height,train_data.width)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    content_dataset = ImageTrainDataset(image_paths=train_data.content_dir, transform=image_transform)
    content_loader = torch.utils.data.DataLoader(content_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)

    style_dataset = ImageTrainDataset(image_paths=train_data.style_dir, transform=image_transform)
    style_loader = torch.utils.data.DataLoader(style_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    style_iter = cycle(iter(style_loader))

    content_val_dataset = ImageTestDataset(image_paths=train_data.content_test_dir, transform=image_transform)
    content_val_loader = torch.utils.data.DataLoader(content_val_dataset)
    # content_val_loader = torch.utils.data.DataLoader(content_dataset)
    content_val_iter = cycle(iter(content_val_loader))

    style_val_dataset = ImageTestDataset(image_paths=train_data.style_test_dir, transform=image_transform)
    style_val_loader = torch.utils.data.DataLoader(style_val_dataset, shuffle=True)
    # style_val_loader = torch.utils.data.DataLoader(style_dataset)
    style_val_iter = cycle(iter(style_val_loader))

    unet, adapter, annotator_model, discriminator, cooccur, g_optim, g_scheduler, d_optim, d_scheduler, text_encoder, content_loader = accelerator.prepare(
        unet, adapter, annotator_model, discriminator, cooccur, g_optim, g_scheduler, d_optim, d_scheduler, text_encoder, content_loader,
    )

    unet_and_controlnet_g_c(unet, text_encoder, gradient_checkpointing, text_encoder_gradient_checkpointing)
    vae.enable_slicing()
    weight_dtype = is_mixed_precision(accelerator)
    models_to_cast = [text_encoder, vae]
    for i in range(len(annotator_model)):
        # if control_mode[i] != 'seg':
        models_to_cast.append(annotator_model[i])
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(content_loader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tensorboard_dir = os.path.basename(output_dir)
        accelerator.init_trackers(os.path.join(tensorboard_dir, "finetune-tensorboard"))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(content_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    mem, meanm = get_memory()
    print(f"before train memory: {mem}  mean:{meanm}")

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def get_all_feature(batch, style_image):
        full_control_image = []
        if use_adapter:
            print('use adapter')
            for i in range(len(control_mode)):
                control_this_mode = control_mode[i]
                this_annotator_model = annotator_model[i]
                if control_this_mode == 'hed':
                    print(i, 'hed')
                    control_image = get_hed_map(this_annotator_model, batch).to(weight_dtype).to(batch.device)
                elif control_this_mode == 'depth':
                    print(i, 'depth')
                    control_image = get_depth_map(this_annotator_model, batch, train_data.height, train_data.width, return_standard_norm=False).to(weight_dtype).to(batch.device)
                elif control_this_mode == 'seg':
                    print(i, 'seg')
                    control_image = get_seg_map(seg_model, batch).to(weight_dtype).to(batch.device)
                elif control_this_mode == 'canny':
                    print(i, 'canny')
                    control_image = get_canny_map(batch).to(weight_dtype).to(batch.device)
                elif control_this_mode == 'style':
                    print(i, 'style')
                    control_image = style_image.to(weight_dtype).to(batch.device)
                else:
                    print(i, 'else')
                    control_image = batch.to(weight_dtype).to(batch.device)
                full_control_image.append(control_image)

            down_block_additional_residuals = adapter(full_control_image)
            down_block_additional_residuals = [
                sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
            ]
        else:
            print('not use adapter')
            down_block_additional_residuals = None

        batch, style_image, train_content_flag, train_style_flag = get_random_batch(batch, style_image, uncontent_flag, unstyle_flag, data_random_rate)   
        latents, noise, noisy_latents, timesteps = get_noise_latent(batch, weight_dtype, vae, noise_scheduler)
        uncond_embeddings = get_uncond_embeddings(batch, tokenizer, text_encoder)
        
        content_feature = unet.get_content_feature(batch, vae, flag=train_content_flag)
        style_feature = unet.get_style_feature(style_image, flag=train_style_flag)
        style_vae_feature = vae.encode(style_image.to(dtype=weight_dtype)).latent_dist.sample()
        style_vae_feature = style_vae_feature * vae.config.scaling_factor

        return batch, style_image, train_content_flag, train_style_flag, latents, noise, noisy_latents, timesteps, uncond_embeddings, content_feature, style_feature, style_vae_feature, down_block_additional_residuals

    d_count = 0
    use_d_count = 0
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_c_loss = 0.0
        train_s_loss = 0.0
        train_g_loss = 0.0
        train_co_loss = 0.0
        train_d1_loss = 0.0
        train_d2_loss = 0.0
        for step, batch in enumerate(content_loader):
            style_image = next(style_iter)
            style_image = style_image.to(batch.device)
            with accelerator.accumulate(unet), accelerator.accumulate(adapter), accelerator.accumulate(discriminator), accelerator.accumulate(cooccur):
                if gan_loss_flag:
                    if use_d_count % update_dis_num == 0:
                        with accelerator.autocast():
                            batch, style_image, train_content_flag, train_style_flag, latents, noise, noisy_latents, timesteps, uncond_embeddings, content_feature, style_feature, style_vae_feature, down_block_additional_residuals = get_all_feature(batch, style_image)
                            use_d_count = 0 
                            requires_grad(unet, False)
                            requires_grad(adapter, False)
                            requires_grad(discriminator, True)
                            requires_grad(cooccur, True)

                            noise_pred = unet(
                                noisy_latents,
                                timesteps,
                                content=content_feature,
                                style=style_feature,
                                encoder_hidden_states=uncond_embeddings,
                                down_block_additional_residuals=down_block_additional_residuals,
                            ).sample

                            noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                            step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents)
                            pred_original_sample = step_dict.pred_original_sample

                            if all_loss_type == 'latent':
                                fake_pred = discriminator(pred_original_sample)
                                real_pred = discriminator(style_vae_feature)
                            else:
                                pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
                                image_out = vae.decode(pred_original_sample).sample
                                fake_pred = discriminator(image_out)
                                real_pred = discriminator(style_image)
                            flag = train_style_flag[..., None]
                            fake_pred = torch.where(flag, fake_pred, 0)
                            real_pred = torch.where(flag, real_pred, 0)
                            d_loss = discriminator.d_logistic_loss(real_pred, fake_pred)

                            n_crop = 8
                            ref_crop = 4
                            if all_loss_type == 'latent':
                                fake_patch = cooccur.patchify_image(pred_original_sample, n_crop)
                                real_patch = cooccur.patchify_image(style_vae_feature, n_crop)
                                ref_patch = cooccur.patchify_image(style_vae_feature, ref_crop * n_crop)
                            else:
                                fake_patch = cooccur.patchify_image(image_out, n_crop)
                                real_patch = cooccur.patchify_image(style_image, n_crop)
                                ref_patch = cooccur.patchify_image(style_image, ref_crop * n_crop)
                            fake_patch_pred, ref_input = cooccur(fake_patch, ref_patch, ref_batch=ref_crop)
                            real_patch_pred, _ = cooccur(real_patch, ref_input=ref_input)
                            flag = torch.cat([train_style_flag[..., None]] * n_crop, dim=1).view(-1)
                            real_patch_pred = torch.where(flag, real_patch_pred, 0)
                            fake_patch_pred = torch.where(flag, fake_patch_pred, 0)
                            cooccur_loss = cooccur.d_logistic_loss(real_patch_pred, fake_patch_pred)

                        avg_d1_loss = accelerator.gather(d_loss.repeat(train_batch_size)).mean()
                        train_d1_loss += avg_d1_loss.item() / gradient_accumulation_steps
                        avg_d2_loss = accelerator.gather(cooccur_loss.repeat(train_batch_size)).mean()
                        train_d2_loss += avg_d2_loss.item() / gradient_accumulation_steps

                        try:
                            accelerator.backward(d_loss+cooccur_loss)
                            params_to_clip = (list(discriminator.parameters()) + list(cooccur.parameters()))
                            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                            d_optim.step()
                            d_scheduler.step()
                            d_optim.zero_grad(set_to_none=True)
                        except Exception as e:
                            print(f"An error has occured during backpropogation! {e}") 
                            continue

                        d_count = d_count + 1
                        d_regularize = d_count % d_reg_every == 0
                        if d_regularize and use_r1_flag:
                            if all_loss_type == 'latent':
                                style_vae_feature.requires_grad = True
                                real_pred = discriminator(style_vae_feature)
                                r1_loss = discriminator.d_r1_loss(real_pred, style_vae_feature)

                                real_patch.requires_grad = True
                                real_patch_pred, _ = cooccur(real_patch, ref_patch, ref_batch=ref_crop)
                                cooccur_r1_loss = cooccur.d_r1_loss(real_patch_pred, real_patch)
                            else:
                                style_image.requires_grad = True
                                real_pred = discriminator(style_image)
                                r1_loss = discriminator.d_r1_loss(real_pred, style_image)

                                real_patch.requires_grad = True
                                real_patch_pred, _ = cooccur(real_patch, ref_patch, ref_batch=ref_crop)
                                cooccur_r1_loss = cooccur.d_r1_loss(real_patch_pred, real_patch)

                            d_optim.zero_grad()

                            r1_loss_sum = 10 / 2 * r1_loss * d_reg_every
                            r1_loss_sum += 1 / 2 * cooccur_r1_loss * d_reg_every
                            r1_loss_sum += 0 * real_pred[0, 0] + 0 * real_patch_pred[0, 0]
                            accelerator.backward(r1_loss_sum)

                            d_optim.step()
                            d_scheduler.step()
                            d_count = 0

                    use_d_count = use_d_count + 1

                if use_adapter:
                    if freeze_unet_flag:
                        requires_grad(unet, False)
                        requires_grad(adapter, True)
                    else:
                        requires_grad(unet, True)
                        requires_grad(adapter, True)
                else:
                    requires_grad(unet, True)
                    requires_grad(adapter, False)
                requires_grad(discriminator, False)
                requires_grad(cooccur, False)
                with accelerator.autocast():
                    batch, style_image, train_content_flag, train_style_flag, latents, noise, noisy_latents, timesteps, uncond_embeddings, content_feature, style_feature, style_vae_feature, down_block_additional_residuals = get_all_feature(batch, style_image)

                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        content=content_feature,
                        style=style_feature,
                        encoder_hidden_states=uncond_embeddings,
                        down_block_additional_residuals=down_block_additional_residuals,
                    ).sample

                    if content_loss_type == 'latent':
                        noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                        step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents)
                        out_latents = step_dict.pred_original_sample
                        target_mse_loss = F.mse_loss(out_latents.float(),latents.float(), reduction="mean")
                    elif content_loss_type == 'image':
                        noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                        step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents)
                        out_latents = step_dict.pred_original_sample
                        out_latents = 1 / vae.config.scaling_factor * out_latents
                        image_out = vae.decode(out_latents).sample
                        
                        target_mse_loss = F.mse_loss(rgb2gray(batch).float(), rgb2gray(image_out).float(), reduction="mean")
                        # target_mse_loss = F.mse_loss(batch.float(), image_out.float(), reduction="mean")

                    else:
                        target_mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    if all_loss_type == 'latent':
                        noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                        step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents)
                        pred_original_sample = step_dict.pred_original_sample
                        if style_loss_flag:
                            style_mean, style_std = calc_mean_std(style_vae_feature, vae)
                            out_mean, out_std = calc_mean_std(1 / vae.config.scaling_factor * pred_original_sample, vae)

                            flag = train_style_flag[..., None, None, None]
                            out_mean = torch.where(flag, out_mean, 0)
                            out_std = torch.where(flag, out_std, 0)
                            style_mean = torch.where(flag, style_mean, 0)
                            style_std = torch.where(flag, style_std, 0)

                            style_loss = F.mse_loss(out_mean,style_mean) + F.mse_loss(out_std,style_std)
                        else:
                            style_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")

                        if gan_loss_flag:
                            fake_pred = discriminator(pred_original_sample)
                            flag = train_style_flag[..., None]
                            fake_pred = torch.where(flag, fake_pred, 0)
                            g_loss = discriminator.g_nonsaturating_loss(fake_pred)

                            n_crop = 8
                            ref_crop = 4
                            fake_patch = cooccur.patchify_image(pred_original_sample, n_crop)
                            ref_patch = cooccur.patchify_image(style_vae_feature, ref_crop * n_crop)
                            fake_patch_pred, _ = cooccur(fake_patch, ref_patch, ref_batch=ref_crop)
                            flag = torch.cat([train_style_flag[..., None]] * n_crop, dim=1).view(-1)
                            fake_patch_pred = torch.where(flag, fake_patch_pred, 0)
                            g_cooccur_loss = cooccur.g_nonsaturating_loss(fake_patch_pred)
                        else:
                            g_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")
                            g_cooccur_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")

                    else:
                        noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                        step_dict = noise_scheduler.step(noise_pred, timesteps[0], noisy_latents)
                        pred_original_sample = 1 / vae.config.scaling_factor * step_dict.pred_original_sample
                        image_out = vae.decode(pred_original_sample).sample

                        # x11 = torch.sum(batch.float(), dim=1)
                        # x12 = torch.sum(image_out.float(), dim=1)
                        # # import pdb; pdb.set_trace()
                        # target_mse_loss = F.mse_loss(x11, x12, reduction="mean")

                        if style_loss_flag:
                            out_style_feature = unet.get_style_feature(image_out)
                            flag = train_style_flag[..., None]
                            out_style_feature = torch.where(flag, out_style_feature, 0)
                            style_feature = torch.where(flag, style_feature, 0)

                            style_loss = F.mse_loss(out_style_feature,style_feature)
                        else:
                            style_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")

                        if gram_loss_flag:
                            flag = train_style_flag[..., None, None, None]
                            gram_loss = Gram_loss(image_out, style_image, unet.vgg, flag)
                        else:
                            gram_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")

                        if gan_loss_flag:
                            fake_pred = discriminator(image_out)
                            flag = train_style_flag[..., None]
                            fake_pred = torch.where(flag, fake_pred, 0)
                            g_loss = discriminator.g_nonsaturating_loss(fake_pred)

                            n_crop = 8
                            ref_crop = 4
                            fake_patch = cooccur.patchify_image(image_out, n_crop)
                            ref_patch = cooccur.patchify_image(style_image, ref_crop * n_crop)
                            fake_patch_pred, _ = cooccur(fake_patch, ref_patch, ref_batch=ref_crop)
                            flag = torch.cat([train_style_flag[..., None]] * n_crop, dim=1).view(-1)
                            fake_patch_pred = torch.where(flag, fake_patch_pred, 0)
                            g_cooccur_loss = cooccur.g_nonsaturating_loss(fake_patch_pred)
                        else:
                            g_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")
                            g_cooccur_loss = F.mse_loss(torch.ones(1).to(batch.device), torch.ones(1).to(batch.device), reduction="mean")
                    
                    loss = content_loss_weight * target_mse_loss + style_loss_weight * style_loss + gan_loss_weight * g_loss + patch_gan_loss_weight * g_cooccur_loss + gram_loss_weight * gram_loss

                    
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                avg_c_loss = accelerator.gather(target_mse_loss.repeat(train_batch_size)).mean()
                train_c_loss += avg_c_loss.item() / gradient_accumulation_steps
                avg_s_loss = accelerator.gather(style_loss.repeat(train_batch_size)).mean()
                train_s_loss += avg_s_loss.item() / gradient_accumulation_steps
                avg_g_loss = accelerator.gather(g_loss.repeat(train_batch_size)).mean()
                train_g_loss += avg_g_loss.item() / gradient_accumulation_steps
                avg_co_loss = accelerator.gather(g_cooccur_loss.repeat(train_batch_size)).mean()
                train_co_loss += avg_co_loss.item() / gradient_accumulation_steps

                try:
                    accelerator.backward(loss)
                    if use_adapter:
                        if freeze_unet_flag:
                            params_to_clip = (list(adapter.parameters()))
                        else:
                            params_to_clip = (list(unet.parameters()) + list(adapter.parameters()))
                    else:
                        params_to_clip = (list(unet.parameters()))
                    
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    g_optim.step()
                    g_scheduler.step()
                    g_optim.zero_grad(set_to_none=True)
                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}") 
                    continue
            

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss, "train_c_loss": train_c_loss, "train_s_loss": train_s_loss, "train_g_loss": train_g_loss, "train_co_loss": train_co_loss, "train_d1_loss": train_d1_loss, "train_d2_loss": train_d2_loss,}, 
                    step=global_step
                )
                train_loss = 0.0
                train_c_loss = 0.0
                train_s_loss = 0.0
                train_g_loss = 0.0
                train_co_loss = 0.0
                train_d1_loss = 0.0
                train_d2_loss = 0.0
            
                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_sd_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        text_encoder, 
                        vae, 
                        annotator_model,
                        adapter,
                        discriminator,
                        cooccur,
                        output_dir, 
                        is_checkpoint=True
                    )
                if should_sample(global_step, validation_steps):
                    if global_step == 1: 
                        print("Performing validation prompt.")
                    
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            unet.eval()
                            text_encoder.eval()
                            adapter.eval()

                            unet_and_controlnet_g_c(unet, text_encoder, False, False)
                            print("validation")
                            pipeline = StableDiffusionAdapterPipeline.from_pretrained(
                                pretrained_sd_model_path,
                                vae=vae,
                                text_encoder=text_encoder,
                                unet=unet, 
                                adapter=adapter,
                                annotator_model=annotator_model,
                            )
                            diffusion_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            content_val_image = next(content_val_iter)
                            content_val_image = content_val_image.to(batch.device)
                            style_val_image = next(style_val_iter)
                            style_val_image = style_val_image.to(batch.device)

                            if not color_flag:
                                content_val_image = color2gray(content_val_image)
                            
                            with torch.no_grad():
                                # prompt = "pale golden rod circle with old lace background"
                                prompt = ""
                                full_control_image = []
                                for i in range(len(control_mode)):
                                    control_this_mode = control_mode[i]
                                    if control_this_mode == 'hed':
                                        control_image = pipeline.get_hed_map(content_val_image, i)
                                    elif control_this_mode == 'depth':
                                        control_image = pipeline.get_depth_map(content_val_image, train_data.height, train_data.width, i, return_standard_norm=False)
                                    elif control_this_mode == 'seg':
                                        control_image = pipeline.get_seg_map(content_val_image)
                                    elif control_this_mode == 'canny':
                                        control_image = pipeline.get_canny_map(content_val_image)
                                    elif control_this_mode == 'style':
                                        control_image = (style_val_image + 1) / 2
                                    else:
                                        control_image = content_val_image
                                    full_control_image.append(control_image)

                                pil_image = pipeline(prompt, num_inference_steps=20, generator=[torch.Generator(device="cuda").manual_seed(seed)], image=full_control_image, content=content_val_image, style=style_val_image, use_adapter=use_adapter, adapter_conditioning_scale=adapter_conditioning_scale).images[0]
                                out_file = "{}/samples/{}_result_ori.jpg".format(output_dir,str(global_step).zfill(6))
                                pil_image.save(out_file)

                                pil_image = pipeline(prompt, num_inference_steps=20, generator=[torch.Generator(device="cuda").manual_seed(seed)], image=full_control_image, content=content_val_image, style=style_val_image, guidance_scale=100, content_scale=0.8, style_scale=0.8, use_adapter=use_adapter, adapter_conditioning_scale=adapter_conditioning_scale).images[0]
                                out_file = "{}/samples/{}_result_cfg.jpg".format(output_dir,str(global_step).zfill(6))
                                pil_image.save(out_file)
                                
                                out_file = "{}/samples/{}_style.jpg".format(output_dir,str(global_step).zfill(6))
                                image_tensor = style_val_image[0,:,:,:].cpu()
                                image_array = ((image_tensor.cpu().numpy() + 1) / 2 * 255).astype('uint8')
                                pil_image = Image.fromarray(image_array.transpose(1,2,0))
                                pil_image.save(out_file)

                                out_file = "{}/samples/{}_content.jpg".format(output_dir,str(global_step).zfill(6))
                                image_tensor = content_val_image[0,:,:,:].cpu()
                                image_array = ((image_tensor.cpu().numpy() + 1) / 2 * 255).astype('uint8')
                                pil_image = Image.fromarray(image_array.transpose(1,2,0))
                                pil_image.save(out_file)

                                for iii in range(len(control_mode)):
                                    out_file = "{}/samples/{}_{}.jpg".format(output_dir,str(global_step).zfill(6),control_mode[iii])
                                    image_tensor = full_control_image[iii]
                                    image_tensor = image_tensor[0,:,:,:].cpu()
                                    image_array = (image_tensor.cpu().numpy() * 255).astype('uint8')
                                    pil_image = Image.fromarray(image_array.transpose(1,2,0))
                                    pil_image.save(out_file)

                            del pipeline
                            torch.cuda.empty_cache()
                            unet.train()
                        logger.info(f"Saved a new sample to {out_file}")

                        unet_and_controlnet_g_c(unet, text_encoder, gradient_checkpointing, text_encoder_gradient_checkpointing)
            

            print("train_content_flag",train_content_flag,"train_style_flag",train_style_flag)
            print("timesteps", timesteps)
            print("step_loss", loss.detach().item())
            print("target_mse_loss", target_mse_loss.detach().item())
            
            logs = {"step_loss": loss.detach().item(), "lr": g_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

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
            discriminator,
            cooccur,
            output_dir, 
            is_checkpoint=False
        )                              
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_base.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))