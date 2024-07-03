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


def img_reorder(x, bs, ch, h, w):
    x = (x + 1) / 2 * 255
    assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, block_size, block_size)
    return x

## Image reordering and testing
def img_inverse_reroder(coverted_img, bs, ch, h, w):
    x = coverted_img.view(bs* ch, -1, total_frequency_component)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
    x += 128
    x = x.view(bs, ch, h, w)
    x = dct.to_rgb(x)#.squeeze(0)
    x = (x / 255.0) * 2 - 1
    return x

def calculate_block_mse(downsample_in, freq_block, num_freq_component=block_size):
    downsample_img = transforms.Resize(size=int(downsample_in.shape[-1]/num_freq_component))(downsample_in)
    loss_vector = torch.zeros(freq_block.shape[2])
    for i in range(freq_block.shape[2]):
        # calculate the MSE between each frequency components and given input downsampled images
        loss_vector[i] = F.mse_loss(downsample_img, freq_block[:,:,i,:,:])
    return loss_vector

def bdct_4x4(img_path):
    # The original input image comes with it and I disable it to reduce the computation overhead.
    # x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)

    back_input = x
    bs, ch, h, w = x.shape
    block_num = h // block_size
    x = img_reorder(x, bs, ch, h, w)
    dct_block = dct.block_dct(x) # BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_component).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)

    return  dct_block_reorder


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

for outsource_freq_base in range(total_frequency_component):
    highest_frequency_components_list = []
    for img_path in overall_img_path_list:
        img_blocks = bdct_4x4(img_path)
        highest_frequency_components_list.append(img_blocks[:,:,outsource_freq_base:,:,:])

    highest_frequency_components_list2 = []
    for img_path in overall_img_path_list2:
        img_blocks = bdct_4x4(img_path)
        highest_frequency_components_list2.append(img_blocks[:,:,outsource_freq_base:,:,:])

    highest_frequency_components_overall = highest_frequency_components_list + highest_frequency_components_list2
    # L2 norm among highest frequency components after BDCT decomposition
    num_images = len(highest_frequency_components_overall)

    l2_norm_expression_list_overall = np.zeros((len(highest_frequency_components_overall), len(highest_frequency_components_overall)))
    for i in range(num_images):
        for j in range(num_images):
            l2_norm_expression_list_overall[i][j] = np.linalg.norm(highest_frequency_components_overall[i] - highest_frequency_components_overall[j])
    np.save(f"l2_norm_outsource_component_{total_frequency_component-outsource_freq_base}.npy", l2_norm_expression_list_overall)
