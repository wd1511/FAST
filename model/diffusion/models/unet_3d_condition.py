# Copyright 2023 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision
from collections import namedtuple

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_3d_blocks import (
    CrossAttnDownBlockPseudo3D,
    CrossAttnUpBlockPseudo3D,
    DownBlockPseudo3D,
    UNetMidBlockPseudo3DCrossAttn,
    UpBlockPseudo3D,
    get_down_block,
    get_up_block,
)
from .resnet import PseudoConv3d
from diffusers.models.cross_attention import AttnProcessor
from typing import Dict
from collections import OrderedDict
from einops import rearrange, repeat

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

@dataclass
class UNetPseudo3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class UNetPseudo3DConditionModel(ModelMixin, ConfigMixin):
    """
    这里把原来2D Unet的 2D卷积全换成新定义的PseudoConv3d。并且定义了从2D卷积继承的模型参数。
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "DownBlockPseudo3D",
        ),
        mid_block_type: str = "UNetMidBlockPseudo3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        fps_embed_type: Optional[str] = None,
        num_fps_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        time_embedding_type: str = "positional",
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        num_class_embeds=None,
        content_channels: int = 4,
        content_refined_channels = 3,
        style_channels = 2944,
    ):
        super().__init__()

        content_refined_channels = content_refined_channels or content_channels
        self.content_channels = content_channels
        self.content_refined_channels = content_refined_channels
        self.style_channels = style_channels
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        
        self.vgg = vgg16(pretrained=True, requires_grad=False)
        self.vgg_scaling_layer = ScalingLayer()
        self.null_style_vector = torch.nn.Embedding(1, style_channels)

        self.content_in = nn.Sequential(
            PseudoConv3d(content_channels, content_refined_channels, 1),
            nn.SiLU(),
            PseudoConv3d(content_refined_channels, content_refined_channels, 1),
        )
        self.content_adaLN_modulation = nn.Sequential(
            nn.Linear(time_embed_dim, content_channels * 2),
            nn.SiLU(),
            nn.Linear(content_channels * 2, content_channels * 2),
        )
        self.initialize_content_weights()
        
        self.style_emb = nn.Sequential(
            nn.Linear(style_channels, time_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, bias=True),
        )

        # input

        self.conv_in = PseudoConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.conv_in_new = PseudoConv3d(in_channels+self.content_refined_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if fps_embed_type is None and num_fps_embeds is not None:
            self.fps_embedding = nn.Embedding(num_fps_embeds, time_embed_dim)
        elif fps_embed_type == "timestep":
            self.fps_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif fps_embed_type == "identity":
            self.fps_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.fps_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlockPseudo3DCrossAttn":
            self.mid_block = UNetMidBlockPseudo3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = PseudoConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
    
    def initialize_content_weights(self):
        if hasattr(self, 'content_in'):
            nn.init.constant_(self.content_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.content_adaLN_modulation[-1].bias, 0)
    
    @property
    def attn_processors(self) -> Dict[str, AttnProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttnProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttnProcessor, Dict[str, AttnProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttnProcessor` or `AttnProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `CrossAttention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainablae attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = (
            num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module,
            (CrossAttnDownBlockPseudo3D, DownBlockPseudo3D, CrossAttnUpBlockPseudo3D, UpBlockPseudo3D),
        ):
            module.gradient_checkpointing = value

    def get_style_feature(self, style_image, flag=None):
        vgg_features = self.vgg(self.vgg_scaling_layer(style_image))
        style_features = torch.cat([torch.cat(torch.std_mean(f, dim=[-1, -2]), dim=1) for f in vgg_features], dim=1)
        if flag is not None:
            flag = flag[..., None]
            style_features = torch.where(flag, style_features, self.null_style_vector.weight[0])  # null style
        return style_features

    def get_content_feature(self, content_video, vae, flag=None):
        video_length = content_video.shape[1]

        content_video = rearrange(content_video, "b f c h w -> (b f) c h w")
        features = vae.encode(content_video).latent_dist.sample()
        features = rearrange(features, "(b f) c h w -> b c f h w", f=video_length)
        features = features * vae.config.scaling_factor

        content_features = features[:, :vae.latent_channels]
        std, mean = torch.std_mean(content_features, dim=[-1, -2], keepdim=True)
        content_features = (content_features - mean) / std
        if flag is not None:
            flag = flag[..., None, None, None]
            content_features = torch.where(flag, content_features, 0)  # null content
        return content_features

    def style_image_process(self, style_image):
        x1 = [1.9303, 2.0749, 2.1459]
        x2 = [-1.7923, -1.7521, -1.4802]
        style_image = (style_image + 1.0) / 2.0
        for i in range(3):
            style_image[:,i,:,:] = style_image[:,i,:,:] * (x1[i] - x2[i]) + x2[i]
        return style_image

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        content: torch.FloatTensor,
        style: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        fps_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNetPseudo3DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        # sample b,c,f,h,w(1, 4, 15, 64, 64)
        # timestep b
        # encoder_hidden_states b, 77, 768

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.fps_embedding is not None:
            if fps_labels is None:
                raise ValueError("fps_labels should be provided when num_fps_embeds > 0")

            if self.config.fps_embed_type == "timestep":
                fps_labels = self.time_proj(fps_labels) # 和timesteps共用，都是sin embedding？这里的weight不更新的。

            # 这里和上面timesteps does not contain any weights and will always return f32 tensors的bug一样。需要先cast过去，不然多机多卡就有问题了。
            fps_labels = fps_labels.to(dtype=self.dtype)
            class_emb = self.fps_embedding(fps_labels)

            emb = emb + class_emb
        if style is not None:
            emb = emb + self.style_emb(style)

        # 2. pre-process

        if content != None:
            if len(content.shape) ==5:
                content_input = rearrange(content, 'b c f h w -> (b f) c h w ')
                emb_repeat = repeat(emb, 'b c -> b f c', f=content.shape[2])
                emb_repeat = rearrange(emb_repeat, 'b f c -> (b f) c ')
                shift, scale = self.content_adaLN_modulation(emb_repeat)[..., None, None].chunk(2, dim=1)
                content_input = modulate(content_input, shift, scale)
                content_input = rearrange(content_input, '(b f) c h w -> b c f h w ', f=content.shape[2])
                content_input = self.content_in(content_input)
            else:
                content_input = content
                shift, scale = self.content_adaLN_modulation(emb)[..., None, None].chunk(2, dim=1)
                content_input = modulate(content_input, shift, scale)
                content_input = self.content_in(content_input)

            sample = torch.cat((sample, content_input), dim=1)
            sample = self.conv_in_new(sample)
        else:
            sample = self.conv_in(sample)
        

        # 3. down
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples
        
        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetPseudo3DConditionOutput(sample=sample)

    @classmethod
    def from_2d_model(cls, model_path, condition_on_fps=False):
        '''
        load a 2d model and convert it to a pseudo 3d model
        '''
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(f"{config_path} does not exist")
        with open(config_path, "r") as f:
            config = json.load(f)

        config.pop("_class_name")
        config.pop("_diffusers_version")

        block_replacer = {
            "CrossAttnDownBlock2D": "CrossAttnDownBlockPseudo3D",
            "DownBlock2D": "DownBlockPseudo3D",
            "UpBlock2D": "UpBlockPseudo3D",
            "CrossAttnUpBlock2D": "CrossAttnUpBlockPseudo3D",
            "UNetMidBlock2DCrossAttn": "UNetMidBlockPseudo3DCrossAttn",
        }

        def convert_2d_to_3d_block(block):
            return block_replacer[block] if block in block_replacer else block

        config["down_block_types"] = [convert_2d_to_3d_block(block) for block in config["down_block_types"]]
        config["up_block_types"] = [convert_2d_to_3d_block(block) for block in config["up_block_types"]]
        if 'mid_block_type' in config:
            config["mid_block_type"] = convert_2d_to_3d_block(config["mid_block_type"])
        
        if condition_on_fps:
            # config["num_fps_embeds"] = 60 # 这个在 trainable embeding时候才需要～
            config["fps_embed_type"] = "timestep"     # 和timestep保持一致的type。

        model = cls(**config)   # 调用自身(init), 传入config参数全换成3d的setting

        state_dict_path_condidates = glob.glob(os.path.join(model_path, "*.bin"))
        if state_dict_path_condidates:
            state_dict = torch.load(state_dict_path_condidates[0], map_location="cpu")
            model.load_2d_state_dict(state_dict=state_dict)

        return model

    def load_2d_state_dict(self, state_dict, **kwargs):
        '''
        2D 部分的参数名完全不变。
        '''
        state_dict_3d = self.state_dict()

        load_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k not in state_dict_3d:
                raise KeyError(f"2d state_dict key {k} does not exist in 3d model")
            # if ("attn1" in k) or ("attn2" in k):
            #     continue
            load_state_dict[k] = v


        state_dict_3d.update(load_state_dict)
        self.load_state_dict(state_dict_3d, **kwargs)



