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

import numpy as np
from tqdm import tqdm
import torch
from models import DeepAppearanceVAE

use_traced_model = True

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


device = torch.device("cuda", 0)
batch_size = 1
model = DeepAppearanceVAE(
    1024, 21918, n_latent=256, n_cams=38
).to(device)
best_model_path = "/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
model.load_state_dict(remove_module_prefix(torch.load(best_model_path)))
model.eval()
model = decoder_linear_quantization(model, bitwidth=8, datatype=torch.int8)

z_local = torch.randn(batch_size, 256).to(device)
v = torch.randn(batch_size, 3).to(device)

# Create a traced model for faster inference
if use_traced_model:
  class TracedDecoder(torch.nn.Module):
    def __init__(self, model_dec):
        super().__init__()
        self.model_dec = model_dec
        
    def forward(self, z_local, v):
        return self.model_dec(z_local, v)
      
  traced_model = TracedDecoder(model.dec)
  traced_model.eval()

  # Trace the model
  with torch.no_grad():
    traced_model = torch.jit.trace(traced_model, (z_local, v))

  # Trace the model
  with torch.no_grad():
     for _ in range(10):
        _ = traced_model(z_local, v)

start_time = time.time()
total_inference = 10000
if use_traced_model:
  for i in tqdm(range(total_inference)):
      traced_model(z_local, v)
else:
  for i in tqdm(range(total_inference)):
      model.dec(z_local, v)

end_time = time.time()

print(f"Under Batchsize = {batch_size}, inference latency on GPU 3090 = {(end_time - start_time) / total_inference}")
