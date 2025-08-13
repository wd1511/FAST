import lpips
import torch
import os
from torchvision import transforms
from PIL import Image
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# d = loss_fn_alex(img0, img1)

in_dir1 = './test_data/9_result_00_100_100_0_0_05/'
in_dir2 = './test_data/style'
image_list = os.listdir(in_dir1)
image_list.sort()
transform_func = transforms.Compose([transforms.ToTensor()])

d_sum = 0

for image_name in image_list:
    image_name_sup = image_name[0:3]
    
    image1 = transform_func(Image.open(os.path.join(in_dir1, image_name_sup + '.jpg')).convert('RGB').resize((512,512)))
    image2 = transform_func(Image.open(os.path.join(in_dir2, image_name_sup + '.png')).convert('RGB').resize((512,512)))
    d = loss_fn_alex(image1, image2)
    d_sum = d_sum + d

print(d_sum/len(image_list))


    
