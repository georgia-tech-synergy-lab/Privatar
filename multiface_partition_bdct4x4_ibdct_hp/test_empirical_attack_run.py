# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import time
from collections import OrderedDict

import cv2
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import DeepAppearanceVAE_IBDCT
from torch.utils.data import DataLoader, SequentialSampler
from utils import Renderer, gammaCorrect
import wandb

wandb_enable = False
accumulate_channel = True
attack_from_high_frequency_channel = True

def main(args, camera_config):
    device = torch.device("cpu")
    # device = torch.device("cuda", 0)

    print(f"camera config file for {subject_id} exists, loading...")

    dataset_attack = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=camera_config["attack"],
    )
    attack_sampler = SequentialSampler(dataset_attack)

    attack_loader = DataLoader(
        dataset_attack,
        args.val_batch_size,
        sampler=attack_sampler,
        num_workers=args.n_worker,
    )

    print("#attack expression list", len(dataset_attack))
    writer = SummaryWriter(log_dir=args.result_path)

    n_cams = len(set(camera_config["train"]).union(set(camera_config["test"])))
    if args.arch == "base":
        model = DeepAppearanceVAE_IBDCT(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, num_freq_comp_outsourced=args.num_freq_comp_outsourced, result_path=args.result_path, save_latent_code=args.save_latent_code, gaussian_noise_covariance_path=args.gaussian_noise_covariance_path
        ).to(device)
    else:
        raise NotImplementedError

    # by default load the best_model.pth
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    mse = nn.MSELoss()

    texmean = cv2.resize(dataset_attack.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_attack.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_attack.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)

    os.makedirs(args.result_path, exist_ok=True)

    if wandb_enable:
        wandb_logger = wandb.init(
            config={
                "tex_size": args.tex_size,
                "mesh_inp_size": args.mesh_inp_size,
                "n_latent": args.nlatent,
                "n_cams": n_cams,
            },
            project=args.project_name,
            entity=args.author_name,
            name="attack_" + args.project_name,
            group="group0",
            dir=args.result_path,
            job_type="empirical_attack",
            reinit=True,
        )

    ##############################
    # Collect the various frequency components of various 
    ############################## 
    # version 1: following the reconstruction pipeline
    expressions_freq_comps = []
    if not accumulate_channel:
        for i, data in tqdm(enumerate(attack_loader)):
            avg_tex = data["avg_tex"].to(device)
            bs, ch, h, w = avg_tex.shape
            block_num = h // model.block_size
            dct_block_reorder = model.dct_transform(avg_tex, bs, ch, h, w)
            dct_block_reorder = dct_block_reorder.reshape(bs, ch*model.total_frequency_component, block_num, block_num)

            projected_outsourced_freq_comp = torch.zeros((bs, model.outsourced_channel_ratio*ch, block_num, block_num))
            # print(projected_outsourced_freq_comp.shape)
            overall_channel = model.outsourced_channel_ratio - 1
            # Fill in each component group value
            for index, freq_pair in enumerate(model.sorted_index_array):
                if model.normalize_list[freq_pair[0]] + model.normalize_list[freq_pair[1]] < 2:
                    # merge two into one.
                    # print(overall_channel*ch,(overall_channel+1)*ch)
                    # print(projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:].shape)
                    projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:] = dct_block_reorder[0,ch*freq_pair[0]:ch*(freq_pair[0]+1),:,:] + dct_block_reorder[0,ch*freq_pair[1]:ch*(freq_pair[1]+1),:,:]
                    overall_channel = overall_channel - 1
                else:
                    # print(overall_channel*ch,(overall_channel+1)*ch)
                    # print(projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:].shape)
                    projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:] = dct_block_reorder[0,ch*freq_pair[0]:ch*(freq_pair[0]+1),:,:]
                    overall_channel = overall_channel - 1
                    # print(overall_channel*ch,(overall_channel+1)*ch)
                    # print(projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:].shape)
                    projected_outsourced_freq_comp[0,overall_channel*ch:(overall_channel+1)*ch,:,:] = dct_block_reorder[0,ch*freq_pair[1]:ch*(freq_pair[1]+1),:,:]
                    overall_channel = overall_channel - 1
            expressions_freq_comps.append(projected_outsourced_freq_comp)
    # version 2: only reconstruct higher frequency channels as BDCT use it to reconstruct overall data anyway
    elif attack_from_high_frequency_channel:
        for i, data in tqdm(enumerate(attack_loader)):
            avg_tex = data["avg_tex"].to(device)
            bs, ch, h, w = avg_tex.shape
            block_num = h // model.block_size
            dct_block_reorder = model.dct_transform(avg_tex, bs, ch, h, w)
            dct_block_reorder = dct_block_reorder[:, model.total_frequency_component-len(model.outsourced_freq_list):model.total_frequency_component, :, :, :].reshape(bs, ch*len(model.outsourced_freq_list), block_num, block_num)

            expressions_freq_comps.append(torch.sum(dct_block_reorder, dim=1))    
        pass
    # version 3: accumulate all channels
    else:
        for i, data in tqdm(enumerate(attack_loader)):
            avg_tex = data["avg_tex"].to(device)
            bs, ch, h, w = avg_tex.shape
            block_num = h // model.block_size
            dct_block_reorder = model.dct_transform(avg_tex, bs, ch, h, w)
            dct_block_reorder = dct_block_reorder[:, model.outsourced_freq_list, :, :, :].reshape(bs, ch*len(model.outsourced_freq_list), block_num, block_num)

            expressions_freq_comps.append(torch.sum(dct_block_reorder, dim=1))    

    val_idx = 0
    model.train()

    model.eval()
    model.to(device)
    begin_time = time.time()
    
    ##############################
    # Mount Attack
    ############################## 
    attack_accuracy = []
    for i, data in tqdm(enumerate(attack_loader)):
        avg_tex = data["avg_tex"].to(device)
        view = data["view"].to(device)
        verts = data["aligned_verts"].to(device)

        pred_tex_comps = model.attack_forward(avg_tex, verts, view)
        ## calculate the loss between pred_tex_comps and the pre-calculated tex components
        tex_loss_expression_list = torch.zeros(len(expressions_freq_comps))

        for j, expression in enumerate(expressions_freq_comps):
            try: 
                tex_loss_expression_list[j] = mse(expression, torch.sum(pred_tex_comps, dim=1))
            except:
                print(f"expression.shape={expression.shape}")
                print(f"pred_tex_comps.shape={pred_tex_comps.shape}")
                raise Exception("size mismatch")
        guess_expression_id = torch.argmin(tex_loss_expression_list)

        if guess_expression_id == i:
            attack_accuracy.append(1)
        else:
            attack_accuracy.append(0)
    
    attack_accuracy_mean = np.array(attack_accuracy).mean()
    writer.add_scalar('attack/accuracy_mean', attack_accuracy_mean, val_idx)

    if wandb_enable:
        wandb_logger.log(
            {
                "attack_accuracy_mean": attack_accuracy_mean,
            }
        )

    end_time = time.time()
    print("Attack takes %f seconds" % (end_time - begin_time))
    print(
        "attack_accuracy_mean %f"
        % (attack_accuracy_mean)
    )
    return attack_accuracy_mean

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
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS",
        help="Directory to dataset root",
    )
    parser.add_argument(
        "--krt_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT",
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
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for testing",
    )
    parser.add_argument(
        "--test_segment_config",
        type=str,
        default="/mnt/captures/ecwuu/test_segment.json",
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
