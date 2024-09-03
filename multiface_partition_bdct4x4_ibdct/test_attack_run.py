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

wandb_enable = True

def main(args, camera_config):
    device = torch.device("cuda", 0)

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

    n_cams = len(camera_config["attack"])
    if args.arch == "base":
        model = DeepAppearanceVAE_IBDCT(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, result_path=args.result_path, save_latent_code=args.save_latent_code, gaussian_noise_covariance_path=args.gaussian_noise_covariance_path
        ).to(device)
    else:
        raise NotImplementedError

    # by default load the best_model.pth
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cuda:0")
    model.load_state_dict(state_dict)
    model = model.to(device)

    renderer = Renderer()

    mse = nn.MSELoss()

    texmean = cv2.resize(dataset_attack.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_attack.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_attack.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_attack.texstd
    vertmean = (
        torch.tensor(dataset_attack.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = dataset_attack.vertstd
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

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
    expressions_freq_comps = []
    for i, data in tqdm(enumerate(attack_loader)):
        avg_tex = data["avg_tex"].cuda()
        bs, ch, h, w = avg_tex.shape
        dct_block_reorder = model.dct_transform(avg_tex, bs, ch, h, w)
        dct_block_reorder = dct_block_reorder.view(bs, model.total_frequency_component, ch, model.block_num, model.block_num)
        expressions_freq_comps.append(expressions_freq_comps)

    def run_net(data):
        M = data["M"].cuda()
        gt_tex = data["tex"].cuda()
        vert_ids = data["vert_ids"].cuda()
        uvs = data["uvs"].cuda()
        uv_ids = data["uv_ids"].cuda()
        avg_tex = data["avg_tex"].cuda()
        view = data["view"].cuda()
        transf = data["transf"].cuda()
        verts = data["aligned_verts"].cuda()
        photo = data["photo"].cuda()
        mask = data["mask"].cuda()
        cams = data["cam"].cuda()
        batch, channel, height, width = avg_tex.shape

        output = {}

        pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.0
        gt_tex = (gt_tex * texstd + texmean) / 255.0

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution
            )
            pred_screen, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution
            )
            screen_loss = (
                torch.mean((pred_screen - photo) ** 2 * screen_mask)
                * (255**2)
                / (texstd**2)
            )
        else:
            screen_loss, pred_screen = torch.zeros([]), None

        total_loss = 0
        if args.lambda_verts > 0:
            total_loss = total_loss + args.lambda_verts * vert_loss
        if args.lambda_tex > 0:
            total_loss = total_loss + args.lambda_tex * tex_loss
        if args.lambda_screen > 0:
            total_loss = total_loss + args.lambda_screen * screen_loss
        if args.lambda_kl > 0:
            total_loss = total_loss + args.lambda_kl * kl

        losses = {
            "total_loss": total_loss,
            "vert_loss": vert_loss,
            "screen_loss": screen_loss,
            "tex_loss": tex_loss,
            "denorm_tex_loss": tex_loss * (texstd**2),
            "kl": kl,
        }

        output["pred_screen"] = pred_screen
        output["pred_verts"] = pred_verts
        output["pred_tex"] = pred_tex

        return losses, output

    def save_img(data, output, tag=""):
        gt_screen = data["photo"] * 255
        gt_tex = data["tex"].cuda() * texstd + texmean
        pred_tex = torch.clamp(output["pred_tex"] * 255, 0, 255)
        if output["pred_screen"] is not None:
            pred_screen = torch.clamp(output["pred_screen"] * 255, 0, 255)
            # apply gamma correction
            save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8) 
            save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
            if len(save_pred_image.shape) == 4:
                for _batch_id in range(save_pred_image.shape[0]):
                    Image.fromarray(save_pred_image[_batch_id]).save(
                        os.path.join(args.result_path, f"pred_{tag}_{_batch_id}.png")
                    )
     
        save_gt_image = gt_screen.detach().cpu().numpy().astype(np.uint8)
        if len(save_gt_image.shape) == 4:
            for _batch_id in range(save_gt_image.shape[0]):
                save_gt_image[_batch_id] = (255 * gammaCorrect(save_gt_image[_batch_id] / 255.0)).astype(np.uint8)
                Image.fromarray(save_gt_image[_batch_id]).save(os.path.join(args.result_path, f"gt_{tag}_{_batch_id}.png"))
            
        # apply gamma correction
        save_gt_tex_image = gt_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
        if len(save_gt_tex_image.shape) == 4:
            for _batch_id in range(save_gt_tex_image.shape[0]):
                save_gt_tex_image[_batch_id] = (255 * gammaCorrect(save_gt_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                Image.fromarray(save_gt_tex_image[_batch_id]).save(os.path.join(args.result_path, f"gt_tex_{tag}_{_batch_id}.png"))
            
        
        save_pred_tex_image = pred_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
        if len(save_pred_tex_image.shape) == 4:
            for _batch_id in range(save_pred_tex_image.shape[0]):
                save_pred_tex_image[_batch_id] = (255 * gammaCorrect(save_pred_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                Image.fromarray(save_pred_tex_image[_batch_id]).save(os.path.join(args.result_path, f"pred_tex_{tag}_{_batch_id}.png"))

        if args.arch == "warp":
            warp = output["warp_field"]
            grid_img = (
                torch.tensor(
                    np.array(
                        Image.open("grid.PNG").resize((args.tex_size, args.tex_size)),
                        dtype=np.float32,
                    )[None, ...]
                )
                .permute(0, 3, 1, 2)
                .to(warp.device)
            )
            grid_img = F.grid_sample(grid_img, warp[-1:])
            Image.fromarray(
                grid_img[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
            ).save(os.path.join(args.result_path, "warp_grid_%s.png" % tag))

    val_idx = 0
    best_screen_loss = 1e8
    best_tex_loss = 1e8
    best_vert_loss = 1e8
    model.train()

    model.eval()
    begin_time = time.time()

    total, vert, tex, screen, kl = [], [], [], [], []
    for i, data in tqdm(enumerate(attack_loader)):
        losses, output = run_net(data)
        total.append(losses["total_loss"].item())
        vert.append(losses["vert_loss"].item())
        tex.append(losses["denorm_tex_loss"].item()) # denormalized 
        screen.append(losses["screen_loss"].item())
        kl.append(losses["kl"].item())

        if wandb_enable:
            wandb_logger.log(
                {
                    "total_loss": losses["total_loss"].item(),
                    "vert_loss": losses["vert_loss"].item(),
                    "tex_loss": losses["tex_loss"].item(),
                    "screen_loss": losses["screen_loss"].item(),
                    "kl": losses["kl"].item(),
                }
            )

        if args.save_img:
            save_img(data, output, "val_%s_%s" % (val_idx, i))

    total_loss = np.array(total).mean()
    tex_loss = np.array(tex).mean()
    vert_loss = np.array(vert).mean()
    screen_loss = np.array(screen).mean()
    kl = np.array(kl).mean()

    writer.add_scalar('val/loss_tex', tex_loss, val_idx)
    writer.add_scalar('val/loss_verts', vert_loss, val_idx)
    writer.add_scalar('val/loss_screen', screen_loss, val_idx)
    writer.add_scalar('val/loss_kl', kl, val_idx)

    if wandb_enable:
        wandb_logger.log(
            {
                "val_total_loss": total_loss,
                "val_vert_loss": vert_loss,
                "val_tex_loss": tex_loss,
                "val_screen_loss": screen_loss,
                "val_kl": kl,
            }
        )

    best_screen_loss = min(best_screen_loss, screen_loss)
    best_tex_loss = min(best_tex_loss, tex_loss)
    best_vert_loss = min(best_vert_loss, vert_loss)

    end_time = time.time()
    print("Testing takes %f seconds" % (end_time - begin_time))
    print(
        "best screen loss %f, best tex loss %f best vert loss %f"
        % (best_screen_loss, best_tex_loss, best_vert_loss)
    )
    return (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Validation batch size"
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
        "--n_worker", type=int, default=8, help="Number of workers loading dataset"
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
        "--save_latent_code",
        action='store_true', 
        default=False, 
        help="save latent code to the result folder ./result_path/latent_code"
    )
    parser.add_argument(
        "--save_img", action='store_true', default=False, help="Control knob to enable image save"
    )
    parser.add_argument(
        "--gaussian_noise_covariance_path", type=str, default=None, help="The path of the noise covariance"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    experiment_args = parser.parse_args()
    print(experiment_args)

    # load camera config
    subject_id = experiment_args.data_dir.split("--")[-2]
    camera_config_path = f"camera_configs/camera-split-config_{subject_id}.json"
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


    (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
    ) = main(experiment_args, camera_set)
    print(
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
    )
    f = open(os.path.join(experiment_args.result_path, "result.txt"), "a")
    f.write("\n")
    f.write(
        "Best screen loss %f, best tex loss %f,  best vert loss %f, screen loss %f, tex loss %f, vert_loss %f"
        % (
            best_screen_loss,
            best_tex_loss,
            best_vert_loss,
            screen_loss,
            tex_loss,
            vert_loss,
        )
    )
    f.close()
