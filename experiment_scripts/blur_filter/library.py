import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchjpeg import dct
# from scipy.fftpack import dct, idct
# import torch_dct as dct_2d, idct_2d
from PIL import Image
import os 
import numpy as np
import torch
import torchvision.transforms as T

# Load dataset
import os
import cv2
import json 
import glob
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from dataset import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, SequentialSampler

block_size = 4
total_frequency_components = block_size * block_size
check_reconstruct_img = True
save_block_img_to_drive = False
load_attack_dataset = False
load_test_dataset = True

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension 

"""
    Image Preprocessing Before BDCT
"""

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
    x = coverted_img.view(bs* ch, -1, total_frequency_components)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
    x += 128
    x = x.view(bs, ch, h, w)
    x = dct.to_rgb(x)#.squeeze(0)
    x = (x / 255.0) * 2 - 1
    return x

def img_reorder_no_extra_standard(x, bs, ch, h, w):
    # x = (x + 1) / 2 * 255
    assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    # x -= 128
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, block_size, block_size)
    return x

## Image reordering and testing
def img_inverse_reroder_no_extra_standard(coverted_img, bs, ch, h, w):
    x = coverted_img.view(bs* ch, -1, total_frequency_components)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
    # x += 128
    x = x.view(bs, ch, h, w)
    x = dct.to_rgb(x)#.squeeze(0)
    # x = (x / 255.0) * 2 - 1
    return x

def img_reorder_pure_bdct(x, bs, ch, h, w):
    # x = (x + 1) / 2 * 255
    assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
    # x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    # x -= 128
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, block_size, block_size)
    return x

## Image reordering and testing
def img_inverse_reroder_pure_bdct(coverted_img, bs, ch, h, w):
    x = coverted_img.view(bs* ch, -1, total_frequency_components)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
    # x += 128
    x = x.view(bs, ch, h, w)
    # x = dct.to_rgb(x)#.squeeze(0)
    # x = (x / 255.0) * 2 - 1
    return x

"""
    Overall Data Transformation
"""

## Image frequency cosine transform
def dct_transform(x, bs, ch, h, w):
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    # A given frequency reference is "dct_block_reorder[freq_id, :, :, :, :]"
    return dct_block_reorder

def dct_inverse_transform(dct_block_reorder, x, bs, ch, h, w):
    block_num = h // block_size
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, x, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def dct_transform_overall(x, bs, ch, h, w):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    for i in range(total_frequency_components):
        transforms.functional.to_pil_image(dct_block_reorder[i,0,:,:,:]).save(f'post_dataloader_bdct_colorconvert_extrastandard{i}.png')
    # A given frequency reference is "dct_block_reorder[freq_id, :, :, :, :]"
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def dct_transform_overall_no_extra_standard(x, bs, ch, h, w):
    back_input = x
    rerodered_img = img_reorder_no_extra_standard(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    for i in range(total_frequency_components):
        transforms.functional.to_pil_image(dct_block_reorder[i,0,:,:,:]).save(f'post_dataloader_bdct_colorconvert_{i}.png')
    # A given frequency reference is "dct_block_reorder[freq_id, :, :, :, :]"
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_no_extra_standard(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform --> THE ONE used in actual reconstruction!!
def dct_transform_overall_pure_bdct(x, bs, ch, h, w):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    for i in range(total_frequency_components):
        transforms.functional.to_pil_image(dct_block_reorder[i,0,:,:,:]).save(f'post_dataloader_bdct_{i}.png')
    # A given frequency reference is "dct_block_reorder[freq_id, :, :, :, :]"
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform_reorder_noise(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // 4
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb):
        dct_block_reorder[i, :, :, :, :] = dct_block_reorder[freq_comp_lb, :, :,  :, :]
 
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform(x, bs, ch, h, w):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
 
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform_drop_high_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb, total_block_num, 1):
        dct_block_reorder[i, :, :, :, :] = dct_block_reorder[i, :, :, :, :].zero_()
        # dct_block_reorder[:, :, i, :, :] = torch.zeros_like(dct_block_reorder[:, :, i, :, :])
 
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform_drop_low_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(0, freq_comp_lb, 1):
        dct_block_reorder[i, :, :, :, :] = dct_block_reorder[i, :, :, :, :].zero_()
 
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform_duplicate_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb, total_block_num, 1):
        dct_block_reorder[i, :, :, :, :] = dct_block_reorder[freq_comp_lb, :, :,  :, :]

    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img


## Image frequency cosine transform
def test_img_dct_transform_reorder_noise_outsource(x, bs, ch, h, w, freq_comp_lb, path_variance_matrix_tensor, add_noise):
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // 4
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb):
        dct_block_reorder[i, :, :, :, :] = dct_block_reorder[freq_comp_lb, :, :,  :, :]
    if add_noise:
        mean = np.zeros(256)
        variance_matrix_tensor = torch.load(path_variance_matrix_tensor).cpu()
        total_sample_noises = dct_block_reorder.shape[0] * dct_block_reorder.shape[1] * dct_block_reorder.shape[2] * dct_block_reorder.shape[3]
        noise_sample = torch.from_numpy(np.random.multivariate_normal(mean, variance_matrix_tensor.detach().numpy(), total_sample_noises)).to(torch.float)
        noise_sample = noise_sample.reshape(dct_block_reorder.shape)
        dct_block_reorder = dct_block_reorder + noise_sample
    
    idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    return inverse_transformed_img


## The one used in actual implementations
## Image frequency cosine transform
def dct_transform_nn_connect_hp( x, bs, ch, h, w):
    """
        The one that directly be used in nn based decoder HP implementation.
    """
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    # dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 4, 1, 2, 3)
    return dct_block_reorder

def dct_inverse_transform_nn_connect_hp( dct_block_reorder,bs, ch, h, w):
    """
        The one that directly be used in nn based decoder HP implementation.
    """
    block_num = h // block_size
    idct_dct_block_reorder = dct_block_reorder.permute(0, 2, 3, 4, 1).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    return inverse_transformed_img
    
    ## Image frequency cosine transform
def test_img_dct_transform_nn_connect_hp(x, bs, ch, h, w):
    """
        The one that directly be used in nn based decoder HP implementation.
    """
    dct_block_reorder = dct_transform_nn_connect_hp(x, bs, ch, h, w)
    print(f"dct_block_reorder={dct_block_reorder.shape}")
    inverse_transformed_img = dct_inverse_transform_nn_connect_hp(dct_block_reorder, bs, ch, h, w)
    return inverse_transformed_img


## The one used in actual implementations
## Image frequency cosine transform
def dct_transform_nn_connect( x, bs, ch, h, w):
    """
        The one that directly be used in multiface_partition_bdct4x4_nn_decoder
    """
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    # dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 4, 1, 2, 3).reshape(bs, ch*total_frequency_components, block_num, block_num) # into (bs, ch, block_num, block_num, 16)
    return dct_block_reorder

def dct_inverse_transform_nn_connect( dct_block_reorder,bs, ch, h, w):
    """
        The one that directly be used in multiface_partition_bdct4x4_nn_decoder
    """
    block_num = h // block_size
    idct_dct_block_reorder = dct_block_reorder.view(bs, total_frequency_components, ch, block_num, block_num).permute(0, 2, 3, 4, 1).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    return inverse_transformed_img
    
    ## Image frequency cosine transform
def test_img_dct_transform_nn_connect(x, bs, ch, h, w):
    """
        The one that directly be used in multiface_partition_bdct4x4_nn_decoder
    """
    dct_block_reorder = dct_transform_nn_connect(x, bs, ch, h, w)
    print(f"dct_block_reorder={dct_block_reorder.shape}")
    inverse_transformed_img = dct_inverse_transform_nn_connect(dct_block_reorder, bs, ch, h, w)
    return inverse_transformed_img




"""
    Data Set Loading
"""

tex_size = 1024
val_batch_size = 1
n_worker = 1
path_prefix = "/home/jianming/work/multiface/"
data_dir = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS"
krt_dir = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT"
# framelist_train = f"/home/jianming/work/Privatar_prj/custom_scripts/bdct_reconstruction/single_expression_frame_list.txt"
framelist_train = "/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/selected_expression_frame_list.txt"
subject_id = data_dir.split("--")[-2]
camera_config_path = f"{path_prefix}camera_configs/camera-split-config_{subject_id}.json"
result_path = "/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/"


if os.path.exists(camera_config_path):
    print(f"camera config file for {subject_id} exists, loading...")
    f = open(camera_config_path, "r")
    camera_config = json.load(f)
    f.close()
else:
    print(f"camera config file for {subject_id} NOT exists, generating...")
    # generate camera config based on downloaded data if not existed
    segments = [os.path.basename(x) for x in glob.glob(f"{data_dir}/unwrapped_uv_1024/*")]
    assert len(segments) > 0
    # select a segment to check available camera ids
    camera_ids = [os.path.basename(x) for x in glob.glob(f"{data_dir}/unwrapped_uv_1024/{segments[0]}/*")]
    camera_ids.remove('average')
    camera_config = {
        "full": {
            "train": camera_ids,
            "test": camera_ids,
            "visual": camera_ids[:2]
        }
    }
    # save the config for future use
    os.makedirs("camera_configs", exist_ok=True)
    with open(camera_config_path, 'w') as f:
        json.dump(camera_config, f)


test_segment = ["EXP_ROM", "EXP_free_face"]

if load_test_dataset:
    ## Generate Data Pair
    dataset_test = Dataset(
        data_dir,
        krt_dir,
        framelist_train,
        tex_size,
        camset=None if camera_config is None else camera_config["full"]["test"],
        exclude_prefix=test_segment,
    )

    print(len(dataset_test))
    texstd = dataset_test.texstd
    texmean = cv2.resize(dataset_test.texmean, (tex_size, tex_size))
    texmin = cv2.resize(dataset_test.texmin, (tex_size, tex_size))
    texmax = cv2.resize(dataset_test.texmax, (tex_size, tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to("cuda:0")
    vertstd = dataset_test.vertstd
    vertmean = (
        torch.tensor(dataset_test.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to("cuda:0")
    )

    test_sampler = SequentialSampler(dataset_test)

    test_loader = DataLoader(
        dataset_test,
        val_batch_size,
        sampler=test_sampler,
        num_workers=n_worker,
    )


## Generate Data Pair -- for empirical attack setup
if load_attack_dataset:
    attack_camera_config_path = f"/home/jianming/work/Privatar_prj/experiment_scripts/empirical_attack/attack-camera-split-config_{subject_id}.json"
    result_path = "/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/"

    print(f"camera config file for {subject_id} exists, loading...")
    f = open(attack_camera_config_path, "r")
    attack_camera_config = json.load(f)
    f.close()
    test_segment = ["EXP_ROM", "EXP_free_face"]

    dataset_attack = Dataset(
        data_dir,
        krt_dir,
        framelist_train,
        tex_size,
        camset= attack_camera_config["full"]["attack"],
    )
    attack_sampler = SequentialSampler(dataset_attack)
    attack_loader = DataLoader(
        dataset_attack,
        val_batch_size,
        sampler=attack_sampler,
        num_workers=n_worker,
    )

