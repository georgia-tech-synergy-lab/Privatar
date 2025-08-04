# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image

import argparse
import os
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Dataset
from models import DeepAppearanceVAE, ConvTranspose2dWN
from utils import gammaCorrect, Renderer

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from the keys of the state_dict.

    Parameters:
        state_dict (OrderedDict): The state dictionary of the model.

    Returns:
        OrderedDict: A new state dictionary with the 'module.' prefix removed from the keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Remove the 'module.' prefix if it exists
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


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

    # model.dec.texture_decoder.upsample[3].conv2.deconv,  model.dec.texture_decoder.upsample[3].conv2.bias, prune_indices_8 = weight_kernel_pruning_l1_norm(model.dec.texture_decoder.upsample[3].conv2.deconv, model.dec.texture_decoder.upsample[3].conv2.bias, unified_pruning_ratio)
    return model


def evaluate_many_images(args, camera_config, test_segment):
    """
    Linearly evaluate all samples in the test dataset.

    Args:
        args: Command line arguments
        camera_config: Camera configuration
        test_segment: Test segment configuration

    Returns:
        Tuple of (average_losses, output) for all samples
    """
    device = torch.device("cuda", 0)
    # Create dataset to get preprocessing parameters
    try:
        dataset_test = Dataset(
            args.data_dir,
            args.krt_dir,
            args.framelist_test,
            args.tex_size,
            camset=None if camera_config is None else camera_config["test"],
            valid_prefix=test_segment,
        )
    except Exception as e:
        raise Exception(f"Error creating dataset: {e}")
     
    print(f"Dataset size: {len(dataset_test)}")

    # Load and initialize model
    n_cams = len(set(camera_config["train"]).union(set(dataset_test.cameras))) if camera_config else len(dataset_test.cameras)
    if args.arch == "base":
        model = DeepAppearanceVAE(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams
        ).to(device)
    else:
        raise NotImplementedError

    if args.unified_pruning_ratio is not None:
        model = model_decoder_pruning(model, args.unified_pruning_ratio)

    # Load model weights
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cuda:0")
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    renderer = Renderer()
    mse = nn.MSELoss()

    # Get preprocessing parameters from dataset
    texmean = cv2.resize(dataset_test.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_test.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_test.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = torch.tensor(dataset_test.texstd, dtype=torch.float32).to(device)
    texstd_squared = texstd ** 2  # Pre-compute squared value for loss calculations
    vertmean = (
        torch.tensor(dataset_test.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = torch.tensor(dataset_test.vertstd, dtype=torch.float32).to(device)
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    loss_list = []
    print(f"Evaluating {len(dataset_test)} samples from test dataset in batches of {args.val_batch_size}")
    os.makedirs(args.result_path, exist_ok=True)

    # Create DataLoader for batch processing
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.val_batch_size,
        shuffle=False,  # Keep order for consistent evaluation
        num_workers=0,  # Use 0 for single process to avoid issues
        drop_last=False  # Don't drop the last batch if it's smaller
    )

    def run_net_batch(data):
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
          tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / texstd_squared

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
                  / texstd_squared
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
              "denorm_tex_loss": tex_loss * texstd_squared,
              "kl": kl,
          }

          output["pred_screen"] = pred_screen
          output["pred_verts"] = pred_verts
          output["pred_tex"] = pred_tex

          return losses, output

    def save_img_single(data, output, tag="single_eval"):
          # Fix tensor shapes for proper image saving
          gt_screen = data["photo"] * 255
          pred_tex = torch.clamp(output["pred_tex"] * 255, 0, 255)

          # Check if we're using template data (when exact match wasn't found)
          using_template = data.get('_using_template', False)

          if output["pred_screen"] is not None:
              pred_screen = torch.clamp(output["pred_screen"] * 255, 0, 255)
              # Convert pred_screen from (batch, width, height, channels) to (batch, height, width, channels)
              pred_screen = pred_screen.permute(0, 2, 1, 3)
              save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8)
              save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
              # Since we're now processing single samples, take the first (and only) element
              img = Image.fromarray(save_pred_image[0])
              # Apply transformations: rotate clockwise 90 degrees and flip horizontally
              img = img.rotate(-90, expand=True)  # -90 for clockwise rotation
              img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
              # Save the transformed image
              img.save(os.path.join(args.result_path, f"pred_rendered_{tag}.png"))
              print(f"  - pred_rendered_{tag}.png: Neural face rendering result")

    # Import tqdm for progress bar
    
    # Process batches with progress bar
    batch_idx = 0
    pbar = tqdm(test_loader, desc="Evaluating batches", unit="batch")
    
    for data in pbar:
        batch_size = data["avg_tex"].shape[0]
        
        # Move data to GPU
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples (like uvs, vert_ids, uv_ids)
                data[key] = value

        # Add template usage flag
        data['_using_template'] = False

        # Run inference on batch
        with torch.no_grad():
            losses, output = run_net_batch(data)
            loss_list.append(losses)
            
            # Update progress bar with current loss values
            total_loss = losses['total_loss'].item()
            tex_loss = losses['denorm_tex_loss'].item()
            screen_loss = losses['screen_loss'].item()
            
            pbar.set_postfix({
                'Batch': f"{batch_idx+1}/{len(test_loader)}",
                'Total': f"{total_loss:.4f}",
                'Tex': f"{tex_loss:.4f}",
                'Screen': f"{screen_loss:.4f}"
            })

            # Save results for each sample in the batch
            if args.save_img:
                os.makedirs(args.result_path, exist_ok=True)
                for i in range(batch_size):
                    # Extract single sample data for saving
                    sample_data = {}
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            sample_data[key] = value[i:i+1]  # Keep batch dimension
                        else:
                            sample_data[key] = value
                    
                    # Extract single sample output
                    sample_output = {}
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            sample_output[key] = value[i:i+1]  # Keep batch dimension
                        else:
                            sample_output[key] = value
                    
                    # Calculate global sample index
                    global_sample_idx = batch_idx * args.val_batch_size + i
                    if args.save_img:
                        save_img_single(sample_data, sample_output, f"sample_{global_sample_idx}")
        
        batch_idx += 1

    # Calculate average losses across all images
    avg_losses = {}
    if loss_list:
        # Initialize sums
        loss_sums = {
            'total_loss': 0.0,
            'vert_loss': 0.0, 
            'denorm_tex_loss': 0.0,
            'screen_loss': 0.0
        }
        
        # Sum up all losses
        for losses in loss_list:
            loss_sums['total_loss'] += losses['total_loss'].item()
            loss_sums['vert_loss'] += losses['vert_loss'].item()
            loss_sums['denorm_tex_loss'] += losses['denorm_tex_loss'].item() 
            loss_sums['screen_loss'] += losses['screen_loss'].item()

        # Calculate averages
        n = len(loss_list)
        avg_losses = {k: v/n for k,v in loss_sums.items()}

        print("\nAverage losses across all images:")
        print(f"Average total loss: {avg_losses['total_loss']:.6f}")
        print(f"Average vertex loss: {avg_losses['vert_loss']:.6f}")
        print(f"Average texture loss: {avg_losses['denorm_tex_loss']:.6f}")
        print(f"Average screen loss: {avg_losses['screen_loss']:.6f}")
    return avg_losses, output


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

    parser.add_argument(
        "--unified_pruning_ratio", type=float, default=None, help="The unified pruning ratio"
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
        test_segment = None

    evaluate_many_images(experiment_args, camera_set, test_segment)  
