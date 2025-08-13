import cv2
from skimage.metrics import structural_similarity as ssim
import os

image1_dir = './test_data/8_artfusion_050_100'
image2_dir = './test_data/content'

image1_list = os.listdir(image1_dir)
image1_list.sort()
score_all = 0

image_list = ['022', '078', '088', '119', '159', '168']

for i in range(len(image_list)):
    image_name = image_list[i]

    image1 = cv2.imread(os.path.join(image1_dir, image_name + '.png'))
    image2 = cv2.imread(os.path.join(image2_dir, image_name + '.png'))
    
    grayA = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    grayA = cv2.resize(grayA, (256, 256), interpolation=cv2.INTER_AREA)
    grayB = cv2.resize(grayB, (256, 256), interpolation=cv2.INTER_AREA)

    score, diff = ssim(grayA, grayB, full=True)
    score_all = score_all + score
    print(image_name, image1.shape, image2.shape, score)
print(score_all / len(image_list))
