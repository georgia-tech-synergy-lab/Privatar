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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import DeepAppearanceVAE_Horizontal_Partition
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils import Renderer, gammaCorrect
from datetime import datetime
import wandb

wandb_enable = True

def main(args, camera_config, test_segment):
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda:0")

    dataset_train = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=None if camera_config is None else camera_config["train"],
        exclude_prefix=test_segment,
    )

    dataset_test = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=None if camera_config is None else camera_config["test"],
        valid_prefix=test_segment,
    )

    test_sampler = RandomSampler(dataset_test)

    test_loader = DataLoader(
        dataset_test,
        args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.n_worker,
    )

    print("#test samples", len(dataset_test))
    writer = SummaryWriter(log_dir=args.result_path)

    n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
    if args.arch == "base":
        # model = DeepAppearanceVAE_Horizontal_Partition(
        #     args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, frequency_threshold=args.frequency_threshold, average_texture_path=args.average_texture_path, prefix_path_captured_latent_code=args.prefix_path_captured_latent_code
        # ).to(device)
        model = DeepAppearanceVAE_Horizontal_Partition(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, frequency_threshold=args.frequency_threshold, average_texture_path=args.average_texture_path, prefix_path_captured_latent_code=args.prefix_path_captured_latent_code, path_variance_matrix_tensor=args.path_variance_matrix_tensor, save_latent_code_to_external_device = args.save_latent_code_to_external_device,  apply_gaussian_noise = args.apply_gaussian_noise
        ).to(device)
    else:
        raise NotImplementedError

    # by default load the best_model.pth
    # state_dict = torch.load(model_dir)
    print("loading model from", args.model_path)
    map_location = {"cuda:0": "cuda:0"}
    # map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    state_dict = torch.load(args.model_path, map_location=map_location)

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove 'module.'
    #     new_state_dict[name] = v
    model.load_state_dict(state_dict)
    model = model.to(device)

    renderer = Renderer()

    optimizer_cc = optim.Adam(model.get_cc_params(), args.lr, (0.9, 0.999))
    optimizer_enc = optim.Adam(model.enc.parameters(), args.lr, (0.9, 0.999))
    mse = nn.MSELoss()

    texmean = cv2.resize(dataset_test.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_test.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_test.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_test.texstd
    vertmean = (
        torch.tensor(dataset_test.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = dataset_test.vertstd
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    os.makedirs(args.result_path, exist_ok=True)

    date_time = datetime.now()
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
            name=args.arch + "_" + "HorizontalPartition" + str(args.frequency_threshold),
            group="group0",
            dir=args.result_path
            + "_"
            + args.arch
            + "_"
            + date_time.strftime("_%m_%d_%Y"),
            job_type="testing",
            reinit=True,
        )

    def run_net(data, iter):
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
        
        height_render, width_render = args.resolution
        width_render = width_render - (width_render % 8)
        photo_short = torch.Tensor(photo)[:, :, :width_render, :]

        if args.arch == "warp":
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = model(
                avg_tex, verts, view, cams=cams
            )
            output["unwarped_tex"] = unwarped_tex
            output["warp_field"] = warp_field
        else:
            pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.0
        gt_tex = (gt_tex * texstd + texmean) / 255.0

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)
        
        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, [height_render, width_render]#args.resolution
            )
            pred_screen, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, [height_render, width_render]#args.resolution
            )
            screen_loss = (
                torch.mean((pred_screen - photo_short) ** 2 * screen_mask[:, :, :width_render, :])
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
            save_pred_image = pred_screen[-1].detach().cpu().numpy().astype(np.uint8)
            save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
            Image.fromarray(save_pred_image).save(os.path.join(args.result_path, "pred_%s.png" % tag))
        # apply gamma correction
        save_gt_image = gt_screen[-1].detach().cpu().numpy().astype(np.uint8)
        save_gt_image = (255 * gammaCorrect(save_gt_image / 255.0)).astype(np.uint8)
        Image.fromarray(save_gt_image).save(os.path.join(args.result_path, "gt_%s.png" % tag))
        # apply gamma correction
        save_gt_tex_image = gt_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        save_gt_tex_image = (255 * gammaCorrect(save_gt_tex_image / 255.0)).astype(np.uint8)
        Image.fromarray(save_gt_tex_image).save(os.path.join(args.result_path, "gt_tex_%s.png" % tag))
        # apply gamma correction
        save_pred_tex_image = pred_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        save_pred_tex_image = (255 * gammaCorrect(save_pred_tex_image / 255.0)).astype(np.uint8)
        Image.fromarray(save_pred_tex_image).save(os.path.join(args.result_path, "pred_tex_%s.png" % tag))

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
    iter = 8
    begin_time = time.time()

    for j in range(iter):
        total, vert, tex, screen, kl = [], [], [], [], []
        for i, data in enumerate(test_loader):
            losses, output = run_net(data, j)
            optimizer_cc.zero_grad()
            optimizer_enc.zero_grad()
            total.append(losses["total_loss"].item())
            vert.append(losses["vert_loss"].item())
            tex.append(losses["denorm_tex_loss"].item()) # denormalized 
            screen.append(losses["screen_loss"].item())
            kl.append(losses["kl"].item())
            losses["total_loss"].backward()
            optimizer_cc.step()
            optimizer_enc.step()
            if i == args.val_num and j != (iter - 1):
                break
            if i < args.val_num and j == (iter - 1):
                save_img(data, output, "val_%s_%s" % (val_idx, i))

    tex_loss = np.array(tex).mean()
    vert_loss = np.array(vert).mean()
    screen_loss = np.array(screen).mean()
    kl = np.array(kl).mean()

    writer.add_scalar('val/loss_tex',losses['tex_loss'].item(), val_idx)
    writer.add_scalar('val/loss_verts', losses['vert_loss'].item(), val_idx)
    writer.add_scalar('val/loss_screen', losses['screen_loss'].item(), val_idx)
    writer.add_scalar('val/loss_kl', losses['kl'].item(), val_idx)

    val_idx += 1
    print(
        "val %d vert %.3f tex %.3f screen %.5f kl %.3f"
        % (val_idx, vert_loss, tex_loss, screen_loss, kl)
    )

    if wandb_enable:
        wandb_logger.log(
            {
                "val_total_loss": losses["total_loss"].item(),
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
    # torch.distributed.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed run"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=40, help="Validation batch size"
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
        "--frequency_threshold", type=float, default=19, help="the MSE threshold to split overall input into private branch and public branch. Available values: [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19, 28]"
    )
    parser.add_argument(
        "--average_texture_path", type=str, default="/home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png", help="the MSE threshold to split overall input into private branch and public branch. Available values: [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19, 28]"
    )
    parser.add_argument(
        "--prefix_path_captured_latent_code", type=str, default="/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_", help="the MSE threshold to split overall input into private branch and public branch. Available values: [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19, 28]"
    )
    parser.add_argument(
        "--path_variance_matrix_tensor", type=str, default="/usr/scratch/jianming/Privatar/profiled_latent_code/statistics/noise_variance_matrix_horizontal_partition_6.0_mutual_bound_1.pth", help="The path to the profiled noise covariance"
    )
    parser.add_argument(
        "--save_latent_code_to_external_device", action='store_true',default=False, help="Control knob to save latent code to external devices"
    )
    parser.add_argument(
        "--apply_gaussian_noise", action='store_true', default=False, help="Control knob to enable noisy training"
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

    if experiment_args.test_segment_config is not None:
        f = open(experiment_args.test_segment_config, "r")
        test_segment_config = json.load(f)
        f.close()
        test_segment = test_segment_config["segment"]
    else:
        test_segment = None

    (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
    ) = main(experiment_args, camera_set, test_segment)
    if torch.distributed.get_rank() == 0:
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
