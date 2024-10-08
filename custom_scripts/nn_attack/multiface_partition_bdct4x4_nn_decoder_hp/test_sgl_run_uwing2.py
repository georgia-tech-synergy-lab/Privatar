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
from tqdm import tqdm
from torchjpeg import dct
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import DeepAppearanceVAEBDCT
from torch.utils.data import DataLoader, RandomSampler
from utils_uwing2 import Renderer, gammaCorrect

wandb_enable = True

block_size = 4
total_frequency_component = int(block_size*block_size)

# def img_reorder_pure_bdct(x, bs, ch, h, w):
#     x = x.view(bs * ch, 1, h, w)
#     x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
#     x = x.transpose(1, 2)
#     x = x.view(bs, ch, -1, block_size, block_size)
#     return x

# ## Image reordering and testing
# def img_inverse_reroder_pure_bdct(self, coverted_img, bs, ch, h, w):
#     x = coverted_img.view(bs* ch, -1, total_frequency_component)
#     x = x.transpose(1, 2)
#     x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
#     x = x.view(bs, ch, h, w)
#     return x

# def dct_transform(x, bs, ch, h, w):
#     rerodered_img = img_reorder_pure_bdct(x, bs, ch, h, w)
#     block_num = h // block_size
#     dct_block = dct.block_dct(rerodered_img) #BDCT
#     # dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_component).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
#     dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_component).permute(0, 4, 1, 2, 3)
#     return dct_block_reorder

# def dct_inverse_transform(dct_block_reorder,bs, ch, h, w):
#     block_num = h // block_size
#     idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, block_size, block_size)
#     inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
#     inverse_transformed_img = img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
    # return inverse_transformed_img

def main(args, camera_config, test_segment):
    device = torch.device("cpu")
    # device = torch.device("cuda", 0)

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

    n_cams = len(set(camera_config["train"]).union(set(dataset_test.cameras)))
    if args.arch == "base":
        model = DeepAppearanceVAEBDCT(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, num_freq_comp_outsourced=args.num_freq_comp_outsourced, result_path=args.result_path, save_latent_code=args.save_latent_code
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
            name=args.project_name,
            group="group0",
            dir=args.result_path,
            job_type="testing",
            reinit=True,
        )
    
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

        height_render, width_render = args.resolution
        width_render = width_render - (width_render % 8)
        photo_short = torch.Tensor(photo)[:, :, :width_render, :]

        # pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
                # Only run the outsourced version with noise
        # pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams) # break it down into steps and apply noises accordingly
        mesh = verts
        avgtex = avg_tex

        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))

        bs, ch, h, w = avgtex.shape
        dct_block_reorder = model.dct_transform(avgtex, bs, ch, h, w)
        
        dct_block_reorder_local = dct_block_reorder[:,model.local_freq_list,:,:,:]
        dct_block_reorder_outsource = dct_block_reorder[:,model.outsourced_freq_list,:,:,:]

        block_num = h // block_size
        dct_block_reorder_local = dct_block_reorder_local.reshape(bs, ch*len(model.local_freq_list), block_num, block_num) # into (bs, ch, block_num, block_num, 16)
        dct_block_reorder_outsource = dct_block_reorder_outsource.reshape(bs, ch*len(model.outsourced_freq_list), block_num, block_num) # into (bs, ch, block_num, block_num, 16)
        # print(f"dct_block_reorder_local.shape={dct_block_reorder_local.shape}")
        # print(f"dct_block_reorder_outsource.shape={dct_block_reorder_outsource.shape}")
        mean, logstd = model.enc(dct_block_reorder_local, mesh)
        mean = mean * 0.1
        logstd = logstd * 0.01
        if model.mode == "vae":
            kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
            std = torch.exp(logstd)
            eps = torch.randn_like(mean)
            z = mean + std * eps
        else:
            z = torch.cat((mean, logstd), -1)
            kl = torch.tensor(0).to(z.device)
        

        mean_outsource, logstd_outsource = model.enc_outsourced(dct_block_reorder_outsource, mesh)
        mean_outsource = mean_outsource * 0.1
        logstd_outsource = logstd_outsource * 0.01
        if model.mode == "vae":
            kl_outsource = 0.5 * torch.mean(torch.exp(2 * logstd_outsource) + mean_outsource**2 - 1.0 - 2 * logstd_outsource)
            std_outsource = torch.exp(logstd_outsource)
            eps_outsource = torch.randn_like(mean_outsource)
            z_outsource = mean_outsource + std_outsource * eps_outsource
        else:
            z_outsource = torch.cat((mean_outsource, logstd_outsource), -1)
            kl_outsource = torch.tensor(0).to(z.device)

        ## Add noise to z_outsource [ToDo]

        ## Add noise to z_outsource -- Done
        
        pred_tex, pred_mesh = model.dec(z, z_outsource, view)
        pred_mesh = pred_mesh.view((b, n, 3))
        if cams is not None:
            pred_tex = model.cc(pred_tex, cams)
        # Only run the outsourced version with noise

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
                torch.mean((pred_screen - photo_short) ** 2 * screen_mask)
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

    val_idx = 0
    model.train()

    model.eval()
    begin_time = time.time()


    total, vert, tex, screen, kl = [], [], [], [], []
    for i, data in tqdm(enumerate(test_loader)):
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


        save_img(data, output, "val_%s_%s" % (val_idx, i))

    tex_loss = np.array(tex).mean()
    vert_loss = np.array(vert).mean()
    screen_loss = np.array(screen).mean()
    kl = np.array(kl).mean()

    writer.add_scalar('val/loss_tex', losses['tex_loss'].item(), val_idx)
    writer.add_scalar('val/loss_verts', losses['vert_loss'].item(), val_idx)
    writer.add_scalar('val/loss_screen', losses['screen_loss'].item(), val_idx)
    writer.add_scalar('val/loss_kl', losses['kl'].item(), val_idx)

    if wandb_enable:
        wandb_logger.log(
            {
                "val_total_loss": losses["total_loss"].item(),
                "val_vert_loss": losses['vert_loss'].item(),
                "val_tex_loss": losses['tex_loss'].item(),
                "val_screen_loss": losses['screen_loss'].item(),
                "val_kl": losses['kl'].item(),
            }
        )

    print(
        "val %d vert %.3f tex %.3f screen %.5f kl %.3f"
        % (val_idx, vert_loss, tex_loss, screen_loss, kl)
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
        default=True, 
        help="save latent code to the result folder ./result_path/latent_code"
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
