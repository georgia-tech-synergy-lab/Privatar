import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchjpeg import dct
from PIL import Image
import seaborn as sns
import numpy as np
import torch
import math
import cv2
import os

block_size = 4
total_frequency_component = block_size * block_size

overall_img_path_list = []
path_prefix = "/home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/"
all_dir = os.listdir(path_prefix)
for sgl_dir in all_dir:
    path_average = os.path.join(path_prefix + sgl_dir, "average")
    overall_img_path_list.append(os.path.join(path_average, os.listdir(path_average)[0]))

overall_img_path_list2 = []
path_prefix2 = "/scratch1/jianming/multiface/dataset/m--20180226--0000--6674443--GHS/unwrapped_uv_1024/"
all_dir = os.listdir(path_prefix2)
for sgl_dir in all_dir:
    path_average2 = os.path.join(path_prefix2 + sgl_dir, "average")
    overall_img_path_list2.append(os.path.join(path_average2, os.listdir(path_average2)[0]))

transform = transforms.Compose([
        transforms.ToTensor()
        ])

downsample_components_list = []
for img_path in overall_img_path_list:
    image = Image.open(img_path).convert('RGB')
    original_in_img = transform(image).unsqueeze(0)
    downsample_img = transforms.Resize(size=int(original_in_img.shape[-1]/block_size))(original_in_img)
    downsample_components_list.append(downsample_img)

downsample_components_list2 = []
for img_path in overall_img_path_list2:
    image2 = Image.open(img_path).convert('RGB')
    original_in_img2 = transform(image2).unsqueeze(0)
    downsample_img2 = transforms.Resize(size=int(original_in_img2.shape[-1]/block_size))(original_in_img2)
    downsample_components_list2.append(downsample_img2)

downsample_components_overall = downsample_components_list + downsample_components_list2
# L2 norm among highest frequency components after BDCT decomposition
num_images = len(downsample_components_overall)

l2_norm_expression_list_overall = np.zeros((len(downsample_components_overall), len(downsample_components_overall)))
for i in range(num_images):
    for j in range(num_images):
        l2_norm_expression_list_overall[i][j] = np.linalg.norm(downsample_components_overall[i] - downsample_components_overall[j])
np.save(f"l2_norm_downsample_in_img_component.npy", l2_norm_expression_list_overall)
