import math
import torch
from torch.nn import functional as F
from torch import nn
import random
import os

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # print("FusedLeakyReLU: ", input.abs().mean())
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        # print("FusedLeakyReLU: ", out.abs().mean())
        return out

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):

        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride,
                        padding=self.padding, )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input
        return (out + skip) / math.sqrt(2)

class Discriminator(nn.Module):
    def __init__(self, image_channel, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1)):
        super(Discriminator,self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            320: 64 * channel_multiplier,
            384: 64 * channel_multiplier,
            448: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(image_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        b, c, h, w = input.shape
        if h == 384 or h == 320 or h == 448:
            input = F.interpolate(input, size=(256, 256), mode='bilinear', align_corners=False)
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()

        return loss

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

class CooccurDiscriminator(nn.Module):
    def __init__(self, all_loss_type, channel, size=256):
        super(CooccurDiscriminator,self).__init__()

        if all_loss_type == 'latent':
            encoder = [ConvLayer(4, channel, 1)]
            ch_multiplier = (2, 4, 8, 12, 12, 24)
            downsample = (True, False, True, False, True, False)
        else:
            encoder = [ConvLayer(3, channel, 1)]
            ch_multiplier = (2, 4, 8, 12, 12, 24)
            downsample = (True, True, True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )
    
    def patchify_image(self, img, n_crop, min_size=1 / 8, max_size=1 / 4):
        crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
        batch, channel, height, width = img.shape
        target_h = int(height * max_size)
        target_w = int(width * max_size)
        crop_h = (crop_size * height).type(torch.int64).tolist()
        crop_w = (crop_size * width).type(torch.int64).tolist()

        patches = []
        for c_h, c_w in zip(crop_h, crop_w):
            c_y = random.randrange(0, height - c_h)
            c_x = random.randrange(0, width - c_w)

            cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
            cropped = F.interpolate(
                cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

            patches.append(cropped)

        patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

        return patches

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        b, c, h, w = input.shape
        if h == 80 or h == 96 or h == 112:
            input = F.interpolate(input, size=(64, 64), mode='bilinear', align_corners=False)
            if reference != None:
                reference = F.interpolate(reference, size=(64, 64), mode='bilinear', align_corners=False)
        
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()

        return loss
    
    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        
        return grad_penalty