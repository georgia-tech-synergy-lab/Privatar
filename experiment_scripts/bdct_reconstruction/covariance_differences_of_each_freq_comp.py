# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import json
import os

import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from torchjpeg import dct

total_frequency_component = 16
block_size = 4

def downsample_tensor(input_tensor, scale_factor=1/8):
    """
    Downsample the input tensor to 1/8 of its height and width.
    
    Parameters:
    input_tensor (torch.Tensor): The input tensor to downsample. The last two dimensions should be height and width.
    scale_factor (float): The factor by which to downsample the height and width (default is 1/8).

    Returns:
    torch.Tensor: The downsampled tensor.
    """
    # Ensure the input tensor has at least 4 dimensions (e.g., batch size, channels, height, width)
    if input_tensor.dim() < 4:
        raise ValueError("Input tensor must have at least 4 dimensions (e.g., batch size, channels, height, width).")
    
    # Downsample using interpolate with 'bilinear' mode (or choose another mode as needed)
    downsampled_tensor = F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    return downsampled_tensor

def img_reorder_pure_bdct(x, bs, ch, h, w):
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, block_size, block_size)
    return x

## Image frequency cosine transform
def dct_transform(x, bs, ch, h, w):
    rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_component).permute(0, 4, 1, 2, 3).reshape(bs, ch*total_frequency_component, block_num, block_num)
    return dct_block_reorder

def main(args, camera_config):
    device = torch.device("cpu")

    print(f"camera config file for {subject_id} exists, loading...")

    dataset_test = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=camera_config["test"],
    )
    test_sampler = SequentialSampler(dataset_test)

    test_loader = DataLoader(
        dataset_test,
        args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.n_worker,
    )

    ##############################
    # Collect the various frequency components of various 
    ############################## 
    # version 1: following the reconstruction pipeline
    freq_tensor_list = []
    for i in range(total_frequency_component):
        freq_tensor_list.append(torch.zeros(len(test_loader), 32*32*3))

    overall_freq_decomposition = []
    for i, data in tqdm(enumerate(test_loader)):
        avg_tex = data["avg_tex"].to(device)
        bs, ch, h, w = avg_tex.shape
        block_num = h // block_size
        dct_block_reorder = dct_transform(avg_tex, bs, ch, h, w)
        downsampled_blocks = downsample_tensor(dct_block_reorder)
        for j in range(total_frequency_component):
            freq_tensor_list[j][i,:] = downsampled_blocks[:, j*ch:(j+1)*ch, :, :].flatten()

    for j in range(total_frequency_component):
        print(f"trace of covariance = {np.trace(np.cov(freq_tensor_list[j]))} for freq component = {j}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Validation batch size"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="base",
        help="Model architecture - base|warp|res|non|bilinear",
    )
    parser.add_argument(
        "--nlatent", type=int, default=256, help="Latent code dimension - 128|256"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--resolution",
        default=[2048, 1334],
        nargs=2,
        type=int,
        help="Rendering resolution",
    )
    parser.add_argument("--tex_size", type=int, default=1024, help="Texture resolution")
    parser.add_argument(
        "--mesh_inp_size", type=int, default=21918, help="Input mesh dimension"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/work/dataset/m--20180227--0000--6795937--GHS",
        help="Directory to dataset root",
    )
    parser.add_argument(
        "--krt_dir",
        type=str,
        default="/work/dataset/m--20180227--0000--6795937--GHS/KRT",
        help="Directory to KRT file",
    )
    parser.add_argument(
        "--loss_weight_mask",
        type=str,
        default="./loss_weight_mask.png",
        help="Mask for weighted loss of face",
    )
    parser.add_argument(
        "--framelist_test",
        type=str,
        default="/work/dataset/m--20180227--0000--6795937--GHS/frame_list.txt",
        help="Frame list for testing",
    )
    parser.add_argument(
        "--test_segment_config",
        type=str,
        default="/work/experiment_scripts/test_segment.json",
        help="Directory of expression segments for testing (exclude from training)",
    )
    parser.add_argument(
        "--lambda_verts", type=float, default=1, help="Multiplier of vertex loss"
    )
    parser.add_argument(
        "--lambda_screen", type=float, default=0, help="Multiplier of screen loss"
    )
    parser.add_argument(
        "--lambda_tex", type=float, default=1, help="Multiplier of texture loss"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=1e-2, help="Multiplier of KL divergence"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200000,
        help="Maximum number of training iterations, overrides epoch",
    )
    parser.add_argument(
        "--log_every", type=int, default=1000, help="Interval of printing training loss"
    )
    parser.add_argument(
        "--val_every", type=int, default=5000, help="Interval of validating on test set"
    )
    parser.add_argument(
        "--val_num", type=int, default=500, help="Number of iterations for validation"
    )
    parser.add_argument(
        "--n_worker", type=int, default=0, help="Number of workers loading dataset"
    )
    parser.add_argument(
        "--pass_thres",
        type=int,
        default=50,
        help="If loss is x times higher than the previous batch, discard this batch",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./runs/experiment",
        help="Directory to output files",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="PiCA Partition - Training Task",
    )
    parser.add_argument(
        "--author_name",
        type=str,
        default=None,
        help="Jianming Tong",
    )
    parser.add_argument(
        "--num_freq_comp_outsourced", type=int, default=2, help="number of outsourced component 2,4,6,8,10,12,14"
    )
    parser.add_argument(
        "--save_latent_code",
        action='store_true', 
        default=False, 
        help="save latent code to the result folder ./result_path/latent_code"
    )
    parser.add_argument(
        "--save_img", 
        action='store_true', 
        default=False, 
        help="Control knob to enable image save"
    )
    parser.add_argument(
        "--gaussian_noise_covariance_path", 
        type=str, 
        default=None, 
        help="The path of the noise covariance"
    )
    parser.add_argument(
        "--model_path",
        type=str, 
        default=None, 
        help="Model path"
    )
    parser.add_argument(
        "--camera_configs_path",
        type=str, 
        default=None, 
        help="path for camera configuration"
    )
    experiment_args = parser.parse_args()
    print(experiment_args)

    # load camera config
    subject_id = experiment_args.data_dir.split("--")[-2]
    camera_config_path = experiment_args.camera_configs_path #"camera_configs/camera-split-config_{subject_id}.json"
    if os.path.exists(camera_config_path):
        print(f"camera config file for {subject_id} exists, loading...")
        f = open(camera_config_path, "r")
        camera_config = json.load(f)
        f.close()
    else:
        print(f"camera config file for {subject_id} NOT exists, generating...")
        # generate camera config based on downloaded data if not existed
        segments = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/*")]
        assert len(segments) > 0
        # select a segment to check available camera ids
        camera_ids = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/{segments[0]}/*")]
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

    camera_set = camera_config["full"]

    attack_accuracy_mean = main(experiment_args, camera_set)
    print(
        attack_accuracy_mean,
    )
    f = open(os.path.join(experiment_args.result_path, "result.txt"), "a")
    f.write("\n")
    f.write(
        "attack_accuracy_mean %f"
        % (
            attack_accuracy_mean
        )
    )
    f.close()
