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


def evaluate_many_images(args, camera_config, test_segment, image_paths):
    """
    Evaluate a single image from the dataset.

    Args:
        args: Command line arguments
        camera_config: Camera configuration
        test_segment: Test segment configuration
        image_paths: Paths to the images to evaluate

    Returns:
        Tuple of (losses, output) for the single image
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
        print(f"Error creating dataset: {e}")
        print("Trying with default dataset configuration...")
        dataset_test = Dataset(
            args.data_dir,
            args.krt_dir,
            args.framelist_test,
            args.tex_size,
            camset=None,
            valid_prefix=None,
        )

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
    for image_path in image_paths:
      print(f"Evaluating single image: {image_path}")
      os.makedirs(args.result_path, exist_ok=True)

      # Extract dataset info from image path
      # Path format: /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png
      import re
      path_parts = image_path.split('/')

      # Find the subject directory (format: m--YYYYMMDD--HHMMSS--SUBJECT_ID--GHS)
      subject_dir = None
      for part in path_parts:
          if re.match(r'm--\d{8}--\d{4}--\d+--GHS', part):
              subject_dir = part
              break

      if subject_dir:
          # Extract the base dataset directory
          subject_idx = path_parts.index(subject_dir)
          base_data_dir = '/'.join(path_parts[:subject_idx + 1])

          # Update args to use the correct data directory for this image
          print(f"Detected subject: {subject_dir}")
          print(f"Using data directory: {base_data_dir}")

          # Override the data directory and related paths
          args.data_dir = base_data_dir
          args.krt_dir = os.path.join(base_data_dir, "KRT")
          args.framelist_test = os.path.join(base_data_dir, "frame_list.txt")

          # Extract camera ID and frame info from path
          # Path: .../images/E001_Neutral_Eyes_Open/400009/000102.png
          # where 400009 is camera ID and 000102 is frame
          if 'images' in path_parts:
              images_idx = path_parts.index('images')
              if images_idx + 3 < len(path_parts):
                  expression = path_parts[images_idx + 1]  # E001_Neutral_Eyes_Open
                  camera_id = path_parts[images_idx + 2]   # 400009
                  frame_name = path_parts[images_idx + 3].split('.')[0]  # 000102

                  print(f"Expression: {expression}, Camera: {camera_id}, Frame: {frame_name}")



      # Try to find the specific data item that corresponds to the image path
      target_data = None
      print("Searching for matching data item in dataset...")
      print(f"len(dataset_test): {len(dataset_test)}")

      # Extract sentnum, cam, and frame from the image path
      # Path format: .../images/E001_Neutral_Eyes_Open/400009/000102.png
      extracted_info = None
      if 'images' in path_parts:
          images_idx = path_parts.index('images')
          if images_idx + 3 < len(path_parts):
              sentnum = path_parts[images_idx + 1]  # E001_Neutral_Eyes_Open
              cam = path_parts[images_idx + 2]      # 400009
              frame = path_parts[images_idx + 3].split('.')[0]  # 000102
              extracted_info = (sentnum, frame, cam)
              print(f"Extracted from path - sentnum: {sentnum}, frame: {frame}, cam: {cam}")

      if extracted_info:
          # Look for exact match in dataset framelist
          print(f"Looking for {extracted_info} in dataset framelist...")
          for i in range(len(dataset_test)):
              framelist_item = dataset_test.framelist[i]
              print(f"Checking framelist[{i}]: {framelist_item}")
              if framelist_item == extracted_info:
                  target_data = dataset_test[i]
                  print(f"Found exact match at index {i}")
                  break
              # Only check first few items to avoid spam
              if i > 5:
                  print("... (showing only first 6 items)")
                  break

      if target_data is None:
          print(f"Could not find exact match in dataset, using first available item as template...")
          target_data = dataset_test[0]
          using_template_data = True  # Flag to track template usage

          # Load the specific image manually
          if os.path.exists(image_path):
              print(f"Loading image from: {image_path}")
              img = cv2.imread(image_path)
              if img is not None:
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                  img_writeout = cv2.resize(img, (args.resolution[1], args.resolution[0]))
                  img = cv2.resize(img, (args.resolution[0], args.resolution[1]))
                  target_data['photo'] = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                  print(f"Successfully loaded image: {img.shape}")
              else:
                  print(f"Failed to load image from {image_path}")
                  return None, None
          else:
              print(f"Image path does not exist: {image_path}")
              return None, None

          # Load corresponding mesh and UV data for the target image
          if extracted_info:
              sentnum, frame, cam = extracted_info
              print(f"Loading corresponding geometric data for {sentnum}/{frame}/{cam}")

              # Load facial mesh (.obj file)
              mesh_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}.obj")
              print(f"Loading mesh from: {mesh_path}")
              if os.path.exists(mesh_path):
                  from dataset import load_obj
                  obj = load_obj(mesh_path)
                  target_data['uvs'] = obj["uvs"]
                  target_data['vert_ids'] = obj["vert_ids"]
                  target_data['uv_ids'] = obj["uv_ids"]
                  print(f"Successfully loaded mesh with {len(obj['verts'])} vertices")
              else:
                  print(f"Warning: Mesh file not found: {mesh_path}")

              # Load vertex positions (.bin file)
              verts_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}.bin")
              print(f"Loading vertex positions from: {verts_path}")
              if os.path.exists(verts_path):
                  verts = np.fromfile(verts_path, dtype=np.float32)
                  verts -= dataset_test.vertmean
                  verts /= dataset_test.vertstd
                  target_data['aligned_verts'] = verts.reshape((-1, 3)).astype(np.float32)
                  print(f"Successfully loaded vertex positions: {verts.shape}")
              else:
                  print(f"Warning: Vertex file not found: {verts_path}")

              # Load unwrapped UV texture
              uv_path = os.path.join(args.data_dir, "unwrapped_uv_1024", sentnum, cam, f"{frame}.png")
              print(f"Loading UV texture from: {uv_path}")
              if os.path.exists(uv_path):
                  tex = np.asarray(Image.open(uv_path), dtype=np.float32)[::-1, ...]
                  mask = tex == 0
                  tex -= dataset_test.texmean
                  tex /= dataset_test.texstd
                  tex[mask] = 0.0
                  tex = cv2.resize(tex, (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                  mask = 1.0 - cv2.resize(mask.astype(np.float32), (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                  target_data['tex'] = tex
                  target_data['mask'] = mask
                  print(f"Successfully loaded UV texture: {tex.shape}")
              else:
                  print(f"Warning: UV texture file not found: {uv_path}")

              # Load average texture for the frame
              avg_path = os.path.join(args.data_dir, "unwrapped_uv_1024", sentnum, "average", f"{frame}.png")
              print(f"Loading average texture from: {avg_path}")
              if os.path.exists(avg_path):
                  avgtex = np.asarray(Image.open(avg_path), dtype=np.float32)[::-1, ...]
                  mask_avg = avgtex == 0
                  avgtex -= dataset_test.texmean
                  avgtex /= dataset_test.texstd
                  avgtex[mask_avg] = 0.0
                  avgtex = cv2.resize(avgtex, (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                  target_data['avg_tex'] = avgtex
                  print(f"Successfully loaded average texture: {avgtex.shape}")
              else:
                  print(f"Warning: Average texture file not found: {avg_path}")

              # Load transform data
              transform_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}_transform.txt")
              print(f"Loading transform from: {transform_path}")
              if os.path.exists(transform_path):
                  transf = np.genfromtxt(transform_path)
                  target_data['transf'] = transf.astype(np.float32)

                  # Recalculate view direction and camera matrix
                  R_f = transf[:3, :3]
                  t_f = transf[:3, 3]
                  cam_key = None
                  for key in dataset_test.krt.keys():
                      if key == cam:
                          cam_key = key
                          break

                  if cam_key is not None:
                      campos = dataset_test.campos[cam_key]
                      campos = np.dot(R_f.T, campos - t_f).astype(np.float32)
                      view = campos / np.linalg.norm(campos)
                      target_data['view'] = view

                      # Update camera matrix
                      extrin, intrin = dataset_test.krt[cam_key]["extrin"], dataset_test.krt[cam_key]["intrin"]
                      R_C = extrin[:3, :3]
                      t_C = extrin[:3, 3]
                      camrot = np.dot(R_C, R_f).astype(np.float32)
                      camt = np.dot(R_C, t_f) + t_C
                      camt = camt.astype(np.float32)
                      M = intrin @ np.hstack((camrot, camt[None].T))
                      target_data['M'] = M.astype(np.float32)

                      print(f"Successfully loaded transform and updated camera data")
                  else:
                      print(f"Warning: Camera {cam} not found in dataset KRT data")
              else:
                  print(f"Warning: Transform file not found: {transform_path}")

      else:
          using_template_data = False

      # Convert single item to batch format
      data = {}
      for key, value in target_data.items():
          if isinstance(value, torch.Tensor):
              data[key] = value.unsqueeze(0)  # Add batch dimension
          elif isinstance(value, np.ndarray):
              data[key] = torch.tensor(value).unsqueeze(0)
          elif isinstance(value, (int, float)):
              # Handle scalar values like camera IDs
              data[key] = torch.tensor([value])
          else:
              data[key] = value

      # Add template usage flag
      data['_using_template'] = using_template_data

      def run_net_single(data):
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
          tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

          if args.lambda_screen > 0:
              screen_mask, rast_out = renderer.render(
                  M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution
              )
              pred_screen, rast_out = renderer.render(
                  M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution
              )

              # Fix tensor shape mismatch: permute photo from (batch, channels, height, width) to (batch, width, height, channels)
              photo_permuted = photo.permute(0, 3, 2, 1)

              screen_loss = (
                  torch.mean((pred_screen - photo_permuted) ** 2 * screen_mask)
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
              if len(save_pred_image.shape) == 4:
                  for _batch_id in range(save_pred_image.shape[0]):
                      # Create PIL Image from array
                      img = Image.fromarray(save_pred_image[_batch_id])
                      # Apply transformations: rotate clockwise 90 degrees and flip horizontally
                      img = img.rotate(-90, expand=True)  # -90 for clockwise rotation
                      img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
                      # Save the transformed image
                      img.save(os.path.join(args.result_path, f"pred_rendered_{tag}_{_batch_id}.png"))

          print(f"Original gt_screen shape: {gt_screen.shape}")

          # Check if gt_screen needs permutation - it should be (batch, height, width, channels)
          if len(gt_screen.shape) == 4 and gt_screen.shape[-1] != 3:
              # If last dimension is not 3 (channels), data might be in (batch, channels, height, width) format
              gt_screen = gt_screen.permute(0, 2, 3, 1)  # Convert to (batch, height, width, channels)
              print(f"Applied permute (0,2,3,1) - new shape: {gt_screen.shape}")
          elif len(gt_screen.shape) == 3:
              # Single image without batch dimension
              if gt_screen.shape[-1] != 3:
                  gt_screen = gt_screen.permute(1, 2, 0)  # Convert to (height, width, channels)
                  print(f"Applied permute (1,2,0) for single image - new shape: {gt_screen.shape}")

          save_gt_image = gt_screen.detach().cpu().numpy().astype(np.uint8)
          # Save predicted texture - convert from (batch, channels, height, width) to (batch, height, width, channels)
          save_pred_tex_image = pred_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
          if len(save_pred_tex_image.shape) == 4:
              for _batch_id in range(save_pred_tex_image.shape[0]):
                  save_pred_tex_image[_batch_id] = (255 * gammaCorrect(save_pred_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                  Image.fromarray(save_pred_tex_image[_batch_id]).save(os.path.join(args.result_path, f"pred_{tag}_{_batch_id}.png"))

          # Only save ground truth texture if we have an exact match from dataset
          if not using_template and "tex" in data:
              gt_tex = data["tex"].cuda() * texstd + texmean
              save_gt_tex_image = gt_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
              if len(save_gt_tex_image.shape) == 4:
                  for _batch_id in range(save_gt_tex_image.shape[0]):
                      save_gt_tex_image[_batch_id] = (255 * gammaCorrect(save_gt_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                      Image.fromarray(save_gt_tex_image[_batch_id]).save(os.path.join(args.result_path, f"gt_{tag}_{_batch_id}.png"))
          else:
              print(f"Note: Skipping ground truth texture save since using template data (not exact match from dataset)")

          print(f"Saved images:")
          print(f"  - pred_rendered_{tag}_0.png: Neural face rendering result")
          print(f"  - pred_texture_{tag}_0.png: Generated texture map")

      # Run inference on single image
      print("Running inference...")
      with torch.no_grad():
          losses, output = run_net_single(data)
          loss_list.append(losses)
          print(f"\nSingle image evaluation results:")
          print(f"Total loss: {losses['total_loss'].item():.6f}")
          print(f"Vertex loss: {losses['vert_loss'].item():.6f}")
          print(f"Texture loss: {losses['denorm_tex_loss'].item():.6f}")
          print(f"Screen loss: {losses['screen_loss'].item():.6f}")
          print(f"KL loss: {losses['kl'].item():.6f}")

          # Save results
          if args.save_img:
              os.makedirs(args.result_path, exist_ok=True)
              save_img_single(data, output, image_path.split('/')[-1])
              print(f"Results saved to: {args.result_path}")

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
        "--image_paths", type=str, default=None,
        help="Paths to images to evaluate. If provided, only these images will be processed instead of the full dataset."
    )
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

    assert experiment_args.image_paths is not None
    # Load image paths from file
    image_paths_list = []
    with open(experiment_args.image_paths, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                image_paths_list.append(line)
    print(f"Loaded {image_paths_list} image paths from {experiment_args.image_paths}")
    evaluate_many_images(experiment_args, camera_set, test_segment, image_paths_list)  
