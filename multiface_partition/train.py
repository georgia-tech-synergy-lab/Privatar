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
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import DeepAppearanceVAE_Horizontal_Partition, ConvTranspose2dWN
from torch.utils.data import DataLoader, RandomSampler
from utils import Renderer, gammaCorrect
from datetime import datetime
import wandb

wandb_enable = True
sparsity_enable = False

def weight_kernel_pruning_l1_norm(model, in_bias, prune_ratio):
    layer_shape = model.state_dict()['weight'].size()
    weight_copy = model.weight.data.abs().clone()
    
    l1_norm = torch.sum(weight_copy, dim=(0, 2, 3))
    num_channels_to_prune = int(prune_ratio * layer_shape[1])
    response_val, prune_indices = torch.topk(l1_norm, num_channels_to_prune, largest=False)
    overall_indices = set([i for i in range(layer_shape[1])])
    prune_indices = set(prune_indices.tolist())
    remaining_indices = overall_indices - prune_indices

    new_model = ConvTranspose2dWN(int(layer_shape[0]), int(len(remaining_indices)), kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False).to(in_bias.device)
    out_bias = torch.nn.Parameter(in_bias[:,list(remaining_indices),:,:]).to(in_bias.device)

    in_weights_float = torch.zeros((int(layer_shape[0]), len(remaining_indices), int(layer_shape[2]), int(layer_shape[3])), dtype=torch.float)
    in_weights_float = weight_copy[:, list(remaining_indices), :, :]
    new_model.weight = torch.nn.Parameter(in_weights_float)
    print(f"under prune_ratio={prune_ratio}, num_channels_to_prune={num_channels_to_prune}, response_val={response_val}, remaining_indices={remaining_indices}, prune_indices={prune_indices}")
    return new_model, out_bias, prune_indices

def iAct_channel_pruning_l1_norm(model, prune_indices):
    layer_shape = model.state_dict()['weight'].size()
    weight_copy = model.weight.data.abs().clone()
    
    prune_indices = set(prune_indices)
    overall_indices = set([i for i in range(layer_shape[0])])
    remaining_indices = overall_indices - prune_indices

    new_model = ConvTranspose2dWN(len(remaining_indices), int(layer_shape[1]), kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False).to(model.weight.device)
    
    in_weights_float = torch.zeros((int(len(remaining_indices)), int(layer_shape[1]), int(layer_shape[2]), int(layer_shape[3])), dtype=torch.float)
    in_weights_float = weight_copy[list(remaining_indices), :, :, :]
    new_model.weight = torch.nn.Parameter(in_weights_float)
    print(f"prune input channel indice={prune_indices}, num_channels_to_prune={len(prune_indices)}, remaining_indices={remaining_indices}, prune_indices={prune_indices}")
    return new_model

def model_decoder_pruning(model, unified_pruning_ratio):
    model.dec.texture_decoder.upsample[0].conv1.deconv, model.dec.texture_decoder.upsample[0].conv1.bias, prune_indices_1 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[0].conv1.deconv, model.dec.texture_decoder.upsample[0].conv1.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[0].conv2.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[0].conv2.deconv, prune_indices_1)

    model.dec.texture_decoder.upsample[0].conv2.deconv, model.dec.texture_decoder.upsample[0].conv2.bias, prune_indices_2 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[0].conv2.deconv, model.dec.texture_decoder.upsample[0].conv2.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[1].conv1.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[1].conv1.deconv, prune_indices_2)

    model.dec.texture_decoder.upsample[1].conv1.deconv,  model.dec.texture_decoder.upsample[1].conv1.bias, prune_indices_3 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[1].conv1.deconv, model.dec.texture_decoder.upsample[1].conv1.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[1].conv2.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[1].conv2.deconv, prune_indices_3)

    model.dec.texture_decoder.upsample[1].conv2.deconv,  model.dec.texture_decoder.upsample[1].conv2.bias, prune_indices_4 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[1].conv2.deconv, model.dec.texture_decoder.upsample[1].conv2.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[2].conv1.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[2].conv1.deconv, prune_indices_4)

    model.dec.texture_decoder.upsample[2].conv1.deconv,  model.dec.texture_decoder.upsample[2].conv1.bias, prune_indices_5 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[2].conv1.deconv, model.dec.texture_decoder.upsample[2].conv1.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[2].conv2.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[2].conv2.deconv, prune_indices_5)

    model.dec.texture_decoder.upsample[2].conv2.deconv,  model.dec.texture_decoder.upsample[2].conv2.bias, prune_indices_6 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[2].conv2.deconv, model.dec.texture_decoder.upsample[2].conv2.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[3].conv1.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[3].conv1.deconv, prune_indices_6)

    model.dec.texture_decoder.upsample[3].conv1.deconv, model.dec.texture_decoder.upsample[3].conv1.bias, prune_indices_7 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[3].conv1.deconv, model.dec.texture_decoder.upsample[3].conv1.bias, unified_pruning_ratio)
    model.dec.texture_decoder.upsample[3].conv2.deconv = iAct_channel_pruning_l1_norm(model.dec.texture_decoder.upsample[3].conv2.deconv, prune_indices_7)

    model.dec.texture_decoder.upsample[3].conv2.deconv,  model.dec.texture_decoder.upsample[3].conv2.bias, prune_indices_8 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[3].conv2.deconv, model.dec.texture_decoder.upsample[3].conv2.bias, unified_pruning_ratio)
    return model

def main(args, camera_config, test_segment):
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset_train = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_train,
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

    train_sampler = RandomSampler(dataset_train)
    test_sampler = RandomSampler(dataset_test)

    train_loader = DataLoader(
        dataset_train,
        args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.n_worker,
    )
    test_loader = DataLoader(
        dataset_test,
        args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.n_worker,
    )

    print("#train samples", len(dataset_train))
    print("#test samples", len(dataset_test))
    writer = SummaryWriter(log_dir=args.result_path)

    n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
    if args.arch == "base":
        # print(f"args.tex_size={args.tex_size}, args.mesh_inp_size={args.mesh_inp_size}, args.nlatent={args.nlatent}, n_cams={n_cams}")
        model = DeepAppearanceVAE_Horizontal_Partition(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams, frequency_threshold=args.frequency_threshold, average_texture_path=args.average_texture_path, prefix_path_captured_latent_code=args.prefix_path_captured_latent_code, path_variance_matrix_tensor=args.path_variance_matrix_tensor, save_latent_code_to_external_device = args.save_latent_code_to_external_device,  noisy_training = args.noisy_training
        ).to(device)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)
    renderer = Renderer()

    # Cannot preload pretrained weights because the Private Path tackles private input images.
    # if args.model_ckpt is not None:
    #     print("loading checkpoint from", args.model_ckpt)
    #     map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    #     model.load_state_dict(torch.load(args.model_ckpt, map_location=map_location))

    if sparsity_enable:
        model = model_decoder_pruning(model, args.unified_pruning_ratio)

    optimizer = optim.Adam(model.module.get_model_params(), args.lr, (0.9, 0.999))
    optimizer_cc = optim.Adam(model.module.get_cc_params(), args.lr, (0.9, 0.999))
    mse = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    texmean = cv2.resize(dataset_train.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_train.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_train.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_train.texstd
    vertmean = (
        torch.tensor(dataset_train.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = dataset_train.vertstd
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
            name=args.arch + "_" + "HorizontalPartition" + str(args.frequency_threshold).split('.')[-1],
            group="group0",
            dir=args.result_path
            + "_"
            + args.arch
            + "_"
            + date_time.strftime("_%m_%d_%Y"),
            job_type="training",
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
                M, pred_verts, vert_ids, uvs, uv_ids, loss_mask,  [height_render, width_render]#args.resolution
            )
            pred_screen, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, pred_tex,  [height_render, width_render]#args.resolution
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
            save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8) 
            save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
            if len(save_pred_image.shape) == 4:
                for _batch_id in range(save_pred_image.shape[0]):
                    Image.fromarray(save_pred_image[_batch_id]).save(
                        os.path.join(args.result_path, f"pred_{tag}_{_batch_id}.png")
                    )
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

    prev_loss = 1e8
    prev_vert_loss = 1e8
    prev_kl = 1e8
    batch_idx, val_idx = 0, 0
    best_screen_loss = 1e8
    best_tex_loss = 1e8
    best_vert_loss = 1e8
    model.train()
    train_screen_losses = []
    train_tex_losses = []
    train_vert_losses = []
    window = 20

    begin_time = time.time()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            losses, output = run_net(data)
            if batch_idx % args.val_every == 0:
                torch.save(
                    model.state_dict(), os.path.join(args.result_path, "model.pth")
                )
                print(
                    "model.pth saved [Epoch {} Batch Index {}]".format(
                        epoch, batch_idx
                    )
                )

            if (
                (losses["total_loss"].item() > args.pass_thres * prev_loss)
                or (losses["vert_loss"].item() > args.pass_thres * prev_vert_loss)
                or (losses["kl"].item() > args.pass_thres * prev_kl)
            ):
                print("throw away batch")
                continue

            writer.add_scalar('train/loss_tex',losses['tex_loss'].item(), batch_idx)
            writer.add_scalar('train/loss_verts', losses['vert_loss'].item(), batch_idx)
            writer.add_scalar('train/loss_screen', losses['screen_loss'].item(), batch_idx)
            writer.add_scalar('train/loss_kl', losses['kl'].item(), batch_idx)

            prev_loss = losses["total_loss"].item()
            prev_vert_loss = losses["vert_loss"].item()
            prev_kl = losses["kl"].item()

            train_screen_losses.append(losses["screen_loss"].item())
            train_tex_losses.append(losses["tex_loss"].item())
            train_vert_losses.append(losses["vert_loss"].item())
            if len(train_screen_losses) > window:
                del train_screen_losses[0]
                del train_tex_losses[0]
                del train_vert_losses[0]

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

            optimizer.zero_grad()
            optimizer_cc.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer_cc.step()

            if batch_idx % args.log_every == 0:
                print(
                    "%d loss %.3f vert %.3f tex %.3f screen %.3f kl %.3f"
                    % (
                        batch_idx,
                        losses["total_loss"].item(),
                        losses["vert_loss"].item(),
                        losses["tex_loss"].item(),
                        losses["screen_loss"].item(),
                        losses["kl"].item(),
                    )
                )
                save_img(data, output, "train_%d" % batch_idx)

            if batch_idx % args.val_every == 0:
                model.eval()
                total, vert, tex, screen, kl = [], [], [], [], []
                for i, data in enumerate(test_loader):
                    optimizer_cc.zero_grad()
                    losses, output = run_net(data)
                    losses["total_loss"].backward()
                    optimizer_cc.step()
                    if i == args.val_num:
                        break

                for i, data in enumerate(test_loader):
                    with torch.no_grad():
                        losses, output = run_net(data)
                        total.append(losses["total_loss"].item())
                        vert.append(losses["vert_loss"].item())
                        tex.append(losses["tex_loss"].item())
                        screen.append(losses["screen_loss"].item())
                        kl.append(losses["kl"].item())
                    if i == args.val_num:
                        break

                tex_loss = np.array(tex).mean()
                vert_loss = np.array(vert).mean()
                screen_loss = np.array(screen).mean()
                kl = np.array(kl).mean()
                writer.add_scalar('val/loss_tex',tex_loss, val_idx)
                writer.add_scalar('val/loss_verts', vert_loss, val_idx)
                writer.add_scalar('val/loss_screen', screen_loss, val_idx)
                writer.add_scalar('val/loss_kl', kl, val_idx)
                save_img(data, output, "val_%d" % val_idx)

                val_idx += 1
                print(
                    "val %d vert %.3f tex %.3f screen %.3f kl %.3f"
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
                if (args.lambda_screen > 0 and best_screen_loss == screen_loss) or (
                    args.lambda_screen == 0 and best_tex_loss == tex_loss
                ):
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.result_path, "best_model.pth"),
                    )
                model.train()

            if batch_idx >= args.max_iter:
                print(
                    "best screen loss %f, best tex loss %f best vert loss %f"
                    % (best_screen_loss, best_tex_loss, best_vert_loss)
                )
                torch.save(
                    model.state_dict(), os.path.join(args.result_path, "model.pth")
                )
                train_screen_loss = np.mean(np.array(train_screen_losses))
                train_tex_loss = np.mean(np.array(train_tex_losses))
                train_vert_loss = np.mean(np.array(train_vert_losses))
                end_time = time.time()
                print("Training takes %f seconds" % (end_time - begin_time))
                return (
                    best_screen_loss,
                    best_tex_loss,
                    best_vert_loss,
                    screen_loss,
                    tex_loss,
                    vert_loss,
                    train_screen_loss,
                    train_tex_loss,
                    train_vert_loss,
                )

            batch_idx += 1
        scheduler.step()

    print(
        "best screen loss %f, best tex loss %f best vert loss %f"
        % (best_screen_loss, best_tex_loss, best_vert_loss)
    )
    torch.save(model.state_dict(), os.path.join(args.result_path, "model.pth"))
    train_screen_loss = np.mean(np.array(train_screen_losses))
    train_tex_loss = np.mean(np.array(train_tex_losses))
    train_vert_loss = np.mean(np.array(train_vert_losses))
    end_time = time.time()
    print("Training takes %f seconds" % (end_time - begin_time))
    return (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
        train_screen_loss,
        train_tex_loss,
        train_vert_loss,
    )


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed run"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=20, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=20, help="Validation batch size"
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
        "--epochs", type=int, default=5, help="Number of training epochs"
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
        "--framelist_train",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for training",
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
        default=None,
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
        "--model_ckpt", type=str, default=None, help="Model checkpoint path"
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
        "--path_variance_matrix_tensor", type=str, default="/usr/scratch/jianming/Privatar/profiled_latent_code/statistics/noise_variance_matrix_horizontal_partition_6.0_mutual_bound_1.pth", help="the MSE threshold to split overall input into private branch and public branch. Available values: [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19, 28]"
    )
    parser.add_argument(
        "--save_latent_code_to_external_device", type=bool, default=False, help="Control knob to save latent code to external devices"
    )
    parser.add_argument(
        "--noisy_training", type=bool, default=False, help="Control knob to enable noisy training"
    )



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
        test_segment = ["EXP_ROM", "EXP_free_face"]

    (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
        train_screen_loss,
        train_tex_loss,
        train_vert_loss,
    ) = main(experiment_args, camera_set, test_segment)
    print(
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
        train_screen_loss,
        train_tex_loss,
        train_vert_loss,
    )
    f = open(os.path.join(experiment_args.result_path, "result.txt"), "a")
    f.write("\n")
    f.write(
        "Best screen loss %f, best tex loss %f,  best vert loss %f, screen loss %f, tex loss %f, vert_loss %f, train screen loss %f, train tex loss %f, train vert loss %f"
        % (
            best_screen_loss,
            best_tex_loss,
            best_vert_loss,
            screen_loss,
            tex_loss,
            vert_loss,
            train_screen_loss,
            train_tex_loss,
            train_vert_loss,
        )
    )
    f.close()
