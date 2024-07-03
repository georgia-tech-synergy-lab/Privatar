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

overall_l2_norm_list_under_various_partition = []
for min_index_outsource_comp in range(total_frequency_component):
    l2_norm_list = np.load(f"/home/jianming/work/Privatar_prj/prior_success_rate_analysis/l2_norm_outsource_component_{total_frequency_component-min_index_outsource_comp}.npy")

    overall_l2_norm_list_under_various_partition.append(np.mean(l2_norm_list))

print(f"overall_l2_norm_list_under_various_partition={overall_l2_norm_list_under_various_partition}")


