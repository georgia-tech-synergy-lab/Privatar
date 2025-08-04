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


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    print(f"[DEBUG] Quantization range for bitwidth {bitwidth}: min={quantized_min}, max={quantized_max}")
    return quantized_min, quantized_max

def get_quantization_scale_for_weight(weight, bitwidth):
    """
    get quantization scale for single tensor of weight
    :param weight: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [integer] quantization bit width
    :return:
        [floating scalar] scale
    """
    # we just assume values in weight are symmetric
    # we also always make zero_point 0 for weight
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    scale = fp_max / quantized_max
    print(f"[DEBUG] Weight quantization scale: {scale} (fp_max={fp_max}, quantized_max={quantized_max})")
    return scale

def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    """
    linear quantization for single fp_tensor
      from
        fp_tensor = (quantized_tensor - zero_point) * scale
      we have,
        quantized_tensor = int(round(fp_tensor / scale)) + zero_point
    :param tensor: [torch.(cuda.)FloatTensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :param scale: [torch.(cuda.)FloatTensor] scaling factor
    :param zero_point: [torch.(cuda.)IntTensor] the desired centroid of tensor values
    :return:
        [torch.(cuda.)FloatTensor] quantized tensor whose values are integers
    """
    print(f"[DEBUG] Linear quantize input tensor shape: {fp_tensor.shape}, bitwidth: {bitwidth}")
    assert(fp_tensor.dtype == torch.float)
    assert(isinstance(scale, float) or
           (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()))
    assert(isinstance(zero_point, int) or
           (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()))

    # Step 1: scale the fp_tensor
    scaled_tensor = fp_tensor / scale
    print(f"[DEBUG] Scaled tensor range: min={scaled_tensor.min().item()}, max={scaled_tensor.max().item()}")
    
    # Step 2: round the floating value to integer value
    rounded_tensor = torch.round(scaled_tensor)
    print(f"[DEBUG] Rounded tensor range: min={rounded_tensor.min().item()}, max={rounded_tensor.max().item()}")

    rounded_tensor = rounded_tensor.to(dtype)

    # Step 3: shift the rounded_tensor to make zero_point 0
    shifted_tensor = rounded_tensor + zero_point
    print(f"[DEBUG] Shifted tensor range: min={shifted_tensor.min().item()}, max={shifted_tensor.max().item()}")

    # Step 4: clamp the shifted_tensor to lie in bitwidth-bit range
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    print(f"[DEBUG] Final quantized tensor range: min={quantized_tensor.min().item()}, max={quantized_tensor.max().item()}")
    return quantized_tensor

def linear_quantize_weight_per_channel(tensor, bitwidth, datatype):
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    print(f"[DEBUG] Weight quantization - tensor shape: {tensor.shape}, bitwidth: {bitwidth}, datatype: {datatype}")
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
        print(f"[DEBUG] Channel {oc} scale: {_scale}")
    
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    print(f"[DEBUG] Scale tensor shape: {scale.shape}")
    
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0, dtype=datatype)
    return quantized_tensor, scale, 0

def linear_quantize_and_replace_weight(conv_transpose_layer, bitwidth=16, datatype=torch.int16):
    print(f"[DEBUG] Quantizing conv transpose layer with bitwidth={bitwidth}")
    new_state_dict = conv_transpose_layer.state_dict()
    weights_tensor = torch.clone(new_state_dict['weight'])
    print(f"[DEBUG] Original weights tensor shape: {weights_tensor.shape}")
    
    wnorm = torch.sqrt(torch.sum(weights_tensor**2))
    print(f"[DEBUG] Weight norm: {wnorm}")
    
    g_tensor = new_state_dict['g']
    print(f"[DEBUG] g tensor shape: {g_tensor.shape}")
    
    result_tensor = weights_tensor * g_tensor[None, :, None, None]
    event_out_tensor = result_tensor / wnorm
    print(f"[DEBUG] Normalized tensor range: min={event_out_tensor.min().item()}, max={event_out_tensor.max().item()}")
    
    post_quantized_tensor, scale, zp = linear_quantize_weight_per_channel(event_out_tensor, bitwidth, datatype)
    post_quantized_tensor_fp = post_quantized_tensor.float()
    post_quantized_tensor_fp = (post_quantized_tensor_fp - zp) * scale
    post_quantized_weights = post_quantized_tensor_fp * wnorm / g_tensor[None, :, None, None]
    print(f"[DEBUG] Final quantized weights range: min={post_quantized_weights.min().item()}, max={post_quantized_weights.max().item()}")
    
    new_state_dict['weight'] = post_quantized_weights
    conv_transpose_layer.load_state_dict(new_state_dict)
    return conv_transpose_layer

def compare_weights_before_after_quantization(model, bitwidth=14, datatype=torch.int16):
    """
    Compare weights before and after quantization to analyze the impact.
    
    Args:
        model: The model to analyze
        bitwidth: Quantization bitwidth
        datatype: Quantization datatype
        
    Returns:
        Dictionary containing comparison statistics
    """
    print(f"\n[WEIGHT COMPARISON] Starting weight comparison analysis...")
    
    comparison_stats = {
        'total_layers_analyzed': 0,
        'weight_changes': [],
        'max_weight_change': 0.0,
        'avg_weight_change': 0.0,
        'total_weight_change': 0.0,
        'layers_with_significant_change': 0
    }
    
    # Store original weights before quantization
    original_weights = {}
    
    # Collect original weights from all quantizable layers
    for i in range(len(model.dec.texture_decoder.upsample)):
        layer_name = f"upsample_{i}"
        
        # conv1.deconv
        conv1_name = f"{layer_name}_conv1_deconv"
        conv1_layer = model.dec.texture_decoder.upsample[i].conv1.deconv
        original_weights[conv1_name] = {
            'weight': torch.clone(conv1_layer.state_dict()['weight']),
            'g': torch.clone(conv1_layer.state_dict()['g']),
            'layer': conv1_layer
        }
        
        # conv2.deconv
        conv2_name = f"{layer_name}_conv2_deconv"
        conv2_layer = model.dec.texture_decoder.upsample[i].conv2.deconv
        original_weights[conv2_name] = {
            'weight': torch.clone(conv2_layer.state_dict()['weight']),
            'g': torch.clone(conv2_layer.state_dict()['g']),
            'layer': conv2_layer
        }
    
    print(f"[WEIGHT COMPARISON] Collected original weights from {len(original_weights)} layers")
    
    # Perform quantization
    quantized_model = decoder_linear_quantization(model, bitwidth, datatype)
    
    # Compare weights after quantization
    for layer_name, original_data in original_weights.items():
        comparison_stats['total_layers_analyzed'] += 1
        
        # Get quantized weights
        quantized_layer = original_data['layer']
        quantized_weight = quantized_layer.state_dict()['weight']
        original_weight = original_data['weight']
        
        # Calculate weight differences
        weight_diff = torch.abs(quantized_weight - original_weight)
        max_diff = weight_diff.max().item()
        mean_diff = weight_diff.mean().item()
        total_diff = weight_diff.sum().item()
        
        # Calculate relative change (percentage)
        relative_change = (weight_diff / (torch.abs(original_weight) + 1e-8)) * 100
        max_relative_change = relative_change.max().item()
        mean_relative_change = relative_change.mean().item()
        
        # Determine if change is significant (>1% relative change)
        significant_change = max_relative_change > 1.0
        
        layer_stats = {
            'layer_name': layer_name,
            'original_weight_shape': original_weight.shape,
            'max_absolute_diff': max_diff,
            'mean_absolute_diff': mean_diff,
            'total_absolute_diff': total_diff,
            'max_relative_change_percent': max_relative_change,
            'mean_relative_change_percent': mean_relative_change,
            'significant_change': significant_change,
            'original_weight_range': (original_weight.min().item(), original_weight.max().item()),
            'quantized_weight_range': (quantized_weight.min().item(), quantized_weight.max().item())
        }
        
        comparison_stats['weight_changes'].append(layer_stats)
        comparison_stats['max_weight_change'] = max(comparison_stats['max_weight_change'], max_diff)
        comparison_stats['total_weight_change'] += total_diff
        
        if significant_change:
            comparison_stats['layers_with_significant_change'] += 1
        
        print(f"[WEIGHT COMPARISON] {layer_name}:")
        print(f"  - Max absolute diff: {max_diff:.6f}")
        print(f"  - Mean absolute diff: {mean_diff:.6f}")
        print(f"  - Max relative change: {max_relative_change:.2f}%")
        print(f"  - Significant change: {significant_change}")
    
    # Calculate overall statistics
    if comparison_stats['total_layers_analyzed'] > 0:
        comparison_stats['avg_weight_change'] = comparison_stats['total_weight_change'] / comparison_stats['total_layers_analyzed']
    
    # Print summary
    print(f"\n[WEIGHT COMPARISON] Summary:")
    print(f"  - Total layers analyzed: {comparison_stats['total_layers_analyzed']}")
    print(f"  - Layers with significant change (>1%): {comparison_stats['layers_with_significant_change']}")
    print(f"  - Max weight change across all layers: {comparison_stats['max_weight_change']:.6f}")
    print(f"  - Average weight change: {comparison_stats['avg_weight_change']:.6f}")
    
    # Detailed analysis for layers with significant changes
    if comparison_stats['layers_with_significant_change'] > 0:
        print(f"\n[WEIGHT COMPARISON] Layers with significant changes:")
        for layer_stats in comparison_stats['weight_changes']:
            if layer_stats['significant_change']:
                print(f"  - {layer_stats['layer_name']}: {layer_stats['max_relative_change_percent']:.2f}% max relative change")
    
    return comparison_stats, quantized_model

def decoder_linear_quantization(model, bitwidth = 14, datatype = torch.int16):
    print(f"[DEBUG] Starting decoder quantization with bitwidth={bitwidth}")
    for i in range(len(model.dec.texture_decoder.upsample)):
        print(f"[DEBUG] Quantizing upsample layer {i}")
        print(f"[DEBUG] Quantizing conv1.deconv")
        model.dec.texture_decoder.upsample[i].conv1.deconv = linear_quantize_and_replace_weight(model.dec.texture_decoder.upsample[i].conv1.deconv, bitwidth, datatype)
        print(f"[DEBUG] Quantizing conv2.deconv")
        model.dec.texture_decoder.upsample[i].conv2.deconv = linear_quantize_and_replace_weight(model.dec.texture_decoder.upsample[i].conv2.deconv, bitwidth, datatype)
    print("[DEBUG] Decoder quantization complete")
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

    # Load model weights
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cuda:0")
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Perform weight comparison and quantization if bitwidth is specified
    if args.bitwidth is not None:
        print(f"Quantizing model with bitwidth {args.bitwidth}")
        
        if args.compare_weights:
            # Use the comprehensive comparison function
            comparison_stats, model = compare_weights_before_after_quantization(model, bitwidth=args.bitwidth, datatype=torch.int8)
            
            # Save comparison results to file
            comparison_file = os.path.join(args.result_path, "weight_comparison_results.json")
            # Convert numpy values to Python types for JSON serialization
            serializable_stats = {}
            for key, value in comparison_stats.items():
                if key == 'weight_changes':
                    serializable_stats[key] = []
                    for layer_stats in value:
                        serializable_layer_stats = {}
                        for layer_key, layer_value in layer_stats.items():
                            if isinstance(layer_value, tuple):
                                serializable_layer_stats[layer_key] = [float(x) for x in layer_value]
                            elif isinstance(layer_value, (int, float, bool, str)):
                                serializable_layer_stats[layer_key] = layer_value
                            elif isinstance(layer_value, torch.Size):
                                serializable_layer_stats[layer_key] = list(layer_value)
                        serializable_stats[key].append(serializable_layer_stats)
                elif isinstance(value, (int, float, bool, str)):
                    serializable_stats[key] = value
            
            with open(comparison_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            print(f"Weight comparison results saved to: {comparison_file}")
        else:
            # Use the original simple quantization
            model = decoder_linear_quantization(model, bitwidth=args.bitwidth, datatype=torch.int8)

    
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
        "--bitwidth", type=int, default=16, help="bitwidth of the actual data"
    )
    parser.add_argument(
        "--compare_weights", action='store_true', default=False, 
        help="Enable weight comparison analysis before and after quantization"
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
