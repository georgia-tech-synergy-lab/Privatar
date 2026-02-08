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
from models import DeepAppearanceVAE_Partition, NN_Attacker
from torch.utils.data import DataLoader, SequentialSampler
from utils import Renderer, gammaCorrect
import wandb

wandb_enable = False
accumulate_channel = True
attack_from_high_frequency_channel = True

output_encoding = {"E001_Neutral_Eyes_Open": 0,
"E002_Swallow": 1,
"E017_Jaw_Open_Mouth_Corners_Down_Nose_Wrinkled": 16,
"E018_Raise_Cheeks": 17,
"E019_Frown": 18,
"E020_Lower_Eyebrows": 19,
"E021_Pressed_Lips_Brows_Down": 20,
"E022_Raise_Inner_Eyebrows": 21,
"E023_Hide_Lips_Look_Up": 22,
"E024_Kiss_Lips_Look_Down": 23,
"E025_Shh": 24,
"E026_Oooo": 25,
"E027_Scrunch_Face_Squeeze_Eyes": 26,
"E003_Neutral_Eyes_Closed": 2,
"E028_Scream_Eyebrows_Up": 27,
"E029_Show_All_Teeth": 28,
"E030_Open_Mouth_Wide_Tongue_Up_And_Back": 29,
"E031_Jaw_Open_Lips_Together": 30,
"E032_Jaw_Open_Pull_Lips_In": 31,
"E033_Jaw_Clench": 32,
"E034_Jaw_Open_Lips_Pushed_Out": 33,
"E004_Relaxed_Mouth_Open": 3,
"E035_Lips_Together_Pushed_Forward": 34,
"E036_Stick_Lower_Lip_Out": 35,
"E037_Bite_Lower_Lip": 36,
"E038_Bite_Upper_Lip": 37,
"E039_Lips_Open_Right": 38,
"E040_Lips_Open_Left": 39,
"E041_Mouth_Nose_Right": 40,
"E042_Mouth_Nose_Left": 41,
"E043_Mouth_Open_Jaw_Right_Show_Teeth": 42,
"E044_Mouth_Open_Jaw_Left_Show_Teeth": 43,
"E045_Jaw_Back": 44,
"E046_Jaw_Forward": 45,
"E047_Tongue_Over_Upper_Lip": 46,
"E005_Eyes_Wide_Open": 4,
"E006_Jaw_Drop_Brows_Up": 5,
"E048_Tongue_Out_Lips_Closed": 47,
"E049_Mouth_Open_Tongue_Out": 48,
"E050_Bite_Tongue": 49,
"E051_Tongue_Out_Flat": 50,
"E052_Tongue_Out_Thick": 51,
"E053_Tongue_Out_Rolled": 52,
"E054_Tongue_Out_Right_Teeth_Showing": 53,
"E055_Tongue_Out_Left_Teeth_Showing": 54,
"E056_Suck_Cheeks_In": 55,
"E057_Cheeks_Puffed": 56,
"E059_Left_Cheek_Puffed": 58,
"E060_Blow_Cheeks_Full_Of_Air": 59,
"E061_Lips_Puffed": 60,
"E007_Neck_Stretch_Brows_Up": 6,
"E062_Nostrils_Dilated": 61,
"E063_Nostrils_Sucked_In": 62,
"E064_Raise_Right_Eyebrow": 63,
"E074_Blink": 65,
"E065_Raise_Left_Eyebrow": 64,
"E008_Smile_Mouth_Closed": 7,
"E009_Smile_Mouth_Open": 8,
"E010_Smile_Stretched": 9,
"E011_Jaw_Open_Sharp_Corner_Lip_Stretch": 10,
"E012_Jaw_Open_Huge_Smile": 11,
"E013_Open_Lips_Mouth_Stretch_Nose_Wrinkled": 12,
"E014_Open_Mouth_Stretch_Nose_Wrinkled": 13,
"E015_Jaw_Open_Upper_Lip_Raised": 14,
"E016_Raise_Upper_Lip_Scrunch_Nose": 15}

def main(args, camera_config):
    device = torch.device("cuda:0")

    ##############################
    # Set up the attack dataset
    ##############################
    print(f"camera config file for {subject_id} exists, loading...")

    dataset_attack = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=camera_config["attack"],
    )
    train_sampler = SequentialSampler(dataset_attack)

    assert(args.train_batch_size == 1)

    attack_loader = DataLoader(
        dataset_attack,
        args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.n_worker,
    )

    print("#attack expression list", len(dataset_attack))
    
    ##############################
    # Load the offloaded VAE model
    ##############################
    n_cams = len(set(camera_config["train"]).union(set(camera_config["test"])))
    if args.arch == "base":
        model = DeepAppearanceVAE_Partition(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, num_freq_comp_offloaded=args.num_freq_comp_offloaded, result_path=args.result_path, save_latent_code=args.save_latent_code, gaussian_noise_covariance_path=args.gaussian_noise_covariance_path
        ).to(device)
    else:
        raise NotImplementedError

    # by default load the best_model.pth
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # The offloaded VAE model is only used to generate the latent code for the attacker model.
    model.to(device)

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
    # Define NN Attacker
    ############################## 
    attacker_model = NN_Attacker(
            args.input_feature, args.hidden_feature, args.output_feature
        ).to(device)
    attacker_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(attacker_model.parameters(), lr=args.lr, momentum=args.momentum)

    ##############################
    # Train NN Attacker
    ############################## 
    def _first_item(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        if isinstance(x, torch.Tensor):
            return x.flatten()[0].item()
        if isinstance(x, np.ndarray):
            flat0 = x.flatten()[0]
            try:
                return flat0.item()
            except Exception:
                return flat0
        return x
    best_loss = 1e8
    begin_time = time.time()
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        running_loss = 0
        for i, data in tqdm(enumerate(attack_loader)):
            glb_expression = str(_first_item(data.get("exp")))
            glb_camera_id = str(_first_item(data.get("cam_idx", data.get("cam"))))
            glb_frame_name = str(_first_item(data.get("frame")))
            reference_output = torch.tensor([output_encoding[glb_expression]]).to(device)
            print(f"Expression: {glb_expression}, and its reference output: {reference_output}, Camera: {glb_camera_id}, Frame: {glb_frame_name}")
            avg_tex = data["avg_tex"].to(device)
            view = data["view"].to(device)
            verts = data["aligned_verts"].to(device)

            z_outsource = model.train_attack_forward(avg_tex, verts, view)
            ## calculate the loss between pred_tex_comps and the pre-calculated tex components
            
            outputs = attacker_model(z_outsource)
            loss = criterion(outputs, reference_output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if wandb_enable:
            wandb_logger.log(
                {
                    "running_loss": running_loss,
                    "epoch": epoch,
                }
            )
        if running_loss < best_loss:
          best_loss = running_loss
          torch.save(
              attacker_model.state_dict(), os.path.join(args.result_path, f"best_attacker_model.pth")
          )
        if epoch % args.log_every == 0:
          print(f"epoch = {epoch}: running_loss = {running_loss}")
          torch.save(
              attacker_model.state_dict(), os.path.join(args.result_path, f"attacker_model.pth")
          )

    end_time = time.time()
    print("Attacker Training takes %f seconds" % (end_time - begin_time))
    print(
        "running_loss %f"
        % (running_loss)
    )
    return running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Validation batch size"
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
        "--log_every", type=int, default=50, help="Interval of printing training loss"
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
        "--num_freq_comp_offloaded", type=int, default=2, help="number of outsourced component 2,4,6,8,10,12,14"
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
    parser.add_argument(
        "--epoch",
        type=int,
        default=3,
        help="If loss is x times higher than the previous batch, discard this batch",
    )
    parser.add_argument(
        "--input_feature",
        type=int,
        default=256,
        help="The input feature of the given tensor",
    )
    parser.add_argument(
        "--hidden_feature",
        type=int,
        default=128,
        help="The hidden feature of the given tensor",
    )
    parser.add_argument(
        "--output_feature",
        type=int,
        default=65,
        help="The output feature of the given tensor",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.1,
        help="The momentum",
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

    running_loss = main(experiment_args, camera_set)
    print(
        running_loss,
    )
    f = open(os.path.join(experiment_args.result_path, "result.txt"), "a")
    f.write("\n")
    f.write(
        "running_loss %f"
        % (
            running_loss
        )
    )
    f.close()