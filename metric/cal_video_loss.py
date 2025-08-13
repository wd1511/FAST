import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torchvision.transforms as T
import lpips
import cv2
import numpy as np
from einops import rearrange, repeat

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
    return image_transform(x)

def video2torch(imagedir):
    # x = Image.open(imagedir)
    video_capture = cv2.VideoCapture(imagedir)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    image_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))    
    ])
    frame_all = []
    for j in range(21):
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rgb = Image.fromarray(np.array(frame_rgb))
        frame_rgb = image_transform(frame_rgb).unsqueeze(0)
        frame_all.append(frame_rgb)
    frame_all = torch.cat(frame_all,dim=0)
    return frame_all

def Gram_matrix(input_tensor):
    b, c, h, w = input_tensor.shape
    # input_tensor = input_tensor.reshape(b, c, -1) / 64 / 64
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

image_all_dir = './test_paper_data2'
image_all_list = os.listdir(image_all_dir)
image_all_list.sort()

for i in range(len(image_all_list)-1):
    image_dataset_name = image_all_list[i]
    image_output_all_dir = os.path.join(image_all_dir,image_dataset_name)
    image_style_all_dir = os.path.join(image_all_dir,'style')
    image_output_list = os.listdir(image_output_all_dir)
    image_style_list = os.listdir(image_style_all_dir)
    image_output_list.sort()
    image_style_list.sort()
    gram_loss = []
    lpips_loss1 = []
    lpips_loss2 = []
    for j in range(len(image_output_list)):
        image_output_name = image_output_list[j]
        image_style_name = image_style_list[j]
        image_output = video2torch(os.path.join(image_output_all_dir, image_output_name)).to(device)
        image_style = image2torch(os.path.join(image_style_all_dir, image_style_name)).to(device)
        image_style = repeat(image_style, 'c h w ->b c h w', b=image_output.shape[0])

        # import pdb; pdb.set_trace()
        # this_gram_loss = Gram_loss(image_style, image_output, vgg, vgg_scaling_layer, 1) / image_output.shape[0]
        # gram_loss.append(this_gram_loss.cpu())
        this_gram_loss = Gram_loss(image_style, image_output, vgg, vgg_scaling_layer, 2) / image_output.shape[0]
        gram_loss.append(this_gram_loss.cpu())
        # this_gram_loss = Gram_loss(image_style, image_output, vgg, vgg_scaling_layer, 3) / image_output.shape[0]
        # gram_loss.append(this_gram_loss.cpu())
        print(j, this_gram_loss)
       
    print(image_dataset_name, sum(gram_loss)/len(gram_loss))
        


