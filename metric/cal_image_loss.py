import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision
from torchvision import transforms
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torchvision.transforms as T
import lpips


# class vgg16(nn.Module):
#     def __init__(self, requires_grad=False, pretrained=True):
#         super(vgg16, self).__init__()
#         vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         self.N_slices = 5
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(23, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X):
#         data_type = X.dtype
#         h = self.slice1(X).to(data_type)
#         h_relu1_2 = h
#         h = self.slice2(h).to(data_type)
#         h_relu2_2 = h
#         h = self.slice3(h).to(data_type)
#         h_relu3_3 = h
#         h = self.slice4(h).to(data_type)
#         h_relu4_3 = h
#         h = self.slice5(h).to(data_type)
#         h_relu5_3 = h
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
#         out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
#         return out

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
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 28):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        data_type = X.dtype
        h = self.slice1(X).to(data_type)
        h_relu1_2 = h
        h = self.slice2(h).to(data_type)
        h_relu2_2 = h
        h = self.slice3(h).to(data_type)
        h_relu3_3 = h
        h = self.slice4(h).to(data_type)
        h_relu4_3 = h
        h = self.slice5(h).to(data_type)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

def image2torch(imagedir):
    x = Image.open(imagedir)
    image_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))    
    ])
    return image_transform(x).unsqueeze(0)

def Gram_matrix(input_tensor):
    b, c, h, w = input_tensor.shape
    input_tensor = input_tensor.reshape(b, c, -1) / c
    gram_matrix = torch.einsum("bxh, bhy->bxy", [input_tensor, input_tensor.transpose(1,2)]) / h
    
    return gram_matrix

def Gram_loss(image1, image2, vgg, vgg_scaling_layer, num):
    image1_vgg = cal_vgg(image1, vgg, vgg_scaling_layer)
    image2_vgg = cal_vgg(image2, vgg, vgg_scaling_layer)
    image1_gram = Gram_matrix(image1_vgg[num])
    image2_gram = Gram_matrix(image2_vgg[num])
    gram_loss = torch.sum((image1_gram - image2_gram)**2)
    return gram_loss

device = 'cuda:0'

vgg = vgg16(pretrained=True, requires_grad=False).to(device)
vgg_scaling_layer = ScalingLayer().to(device)

def cal_vgg(style_image, vgg, vgg_scaling_layer):
    output = []
    vgg_features = vgg(vgg_scaling_layer(style_image))
    # vgg_features = vgg(style_image)
    for i in vgg_features:
        output.append(i)
    return output

in_dir1 = './test_data/9_result1/'
in_dir2 = './test_data/style'
image_list = os.listdir(in_dir1)
image_list.sort()
transform_func = transforms.Compose([transforms.ToTensor()])

d_sum1 = 0
d_sum2 = 0
d_sum3 = 0
d_sum4 = 0

for image_name in image_list:
    image_name_sup = image_name[0:3]
    
    image1 = transform_func(Image.open(os.path.join(in_dir1, image_name)).convert('RGB').resize((512,512))).to(device)
    image2 = transform_func(Image.open(os.path.join(in_dir2, image_name_sup + '.png')).convert('RGB').resize((512,512))).to(device)
    this_gram_loss1 = Gram_loss(image2, image1, vgg, vgg_scaling_layer, 1)
    d_sum1 = d_sum1 + this_gram_loss1

    this_gram_loss2 = Gram_loss(image2, image1, vgg, vgg_scaling_layer, 2)
    d_sum2 = d_sum2 + this_gram_loss2

    this_gram_loss3 = Gram_loss(image2, image1, vgg, vgg_scaling_layer, 3)
    d_sum3 = d_sum3 + this_gram_loss3

    d_sum4 = d_sum4 + (this_gram_loss1 + this_gram_loss2 + this_gram_loss3) / 3.0

print(d_sum1/len(image_list))
print(d_sum2/len(image_list))
print(d_sum3/len(image_list))
print(d_sum4/len(image_list))

