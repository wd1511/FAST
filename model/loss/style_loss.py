import math
import torch
from torch.nn import functional as F
from torch import nn
import random
import os

def CosineLoss(input1, input2):
    b = input1.shape[0]
    input1 = input1.view(b, -1)
    input2 = input2.view(b, -1)

    return torch.mean(1 - F.cosine_similarity(input1, input2))

def Gram_matrix(input_tensor):
    b, c, h, w = input_tensor.shape
    input_tensor = input_tensor.reshape(b, c, -1) / h / w

    gram_matrix = torch.einsum("bxh, bhy->bxy", [input_tensor, input_tensor.transpose(1,2)])
    
    return gram_matrix

def Gram_loss(out_image, style_image, vgg, flag = None):
    # import pdb; pdb.set_trace()
    # x1 = vgg.slice3(vgg.slice2(vgg.slice1(out_image)))
    # x2 = vgg.slice3(vgg.slice2(vgg.slice1(style_image)))
    x1_features = vgg(out_image)
    x2_features = vgg(style_image)
    x1 = x1_features[0]
    x2 = x2_features[0]
    out_gram = Gram_matrix(x1)
    style_gram = Gram_matrix(x2)

    if flag is not None:
        style_gram = torch.where(flag, style_gram, 0)
        out_gram = torch.where(flag, out_gram, 0)

    gram_loss = torch.sum((out_gram - style_gram)**2)

    return gram_loss