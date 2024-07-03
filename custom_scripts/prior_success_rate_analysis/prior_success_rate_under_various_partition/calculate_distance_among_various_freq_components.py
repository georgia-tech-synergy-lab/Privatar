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
total_frequency_component = 16

L2_norm_list_outsourcing_various_freq_comp = np.array([7817.963601820967, 1927.7927544280774, 1622.764830586238, 1506.3766966215637, 1440.4200165664515, 1191.4003726142762, 1074.0282486394933, 997.1141417742151, 940.5921349708563, 814.3635555898857, 722.5463845598816, 647.1855186994561, 582.9664950139886, 474.7725566603714, 368.8741499642574, 250.10336506082618])

def d(original_in_img, block_size):
    """
        original data:
        downsampled data:
    """
    print(downsample_img.shape)
    
    return downsample_img

overall_img_path_list = []
path_prefix = "/home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/"
all_dir = os.listdir(path_prefix)
for sgl_dir in all_dir:
    path_average = os.path.join(path_prefix + sgl_dir, "average")
    overall_img_path_list.append(os.path.join(path_average, os.listdir(path_average)[0]))
calculate_l2_norm(path_prefix, block_size)
