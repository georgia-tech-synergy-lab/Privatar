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
import torch
from models import DeepAppearanceVAE, ConvTranspose2dWN

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

sparsity_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
latency_list = []
for sparsity_ratio in sparsity_ratio_list:
  device = torch.device("cuda", 0)
  batch_size = 1
  model = DeepAppearanceVAE(
      1024, 21918, n_latent=256, n_cams=38
  ).to(device)
  
  best_model_path = "/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
  model.load_state_dict(remove_module_prefix(torch.load(best_model_path)))

  model_decoder_pruning(model, sparsity_ratio)

  z_local = torch.zeros(batch_size, 256).to("cuda:0")
  v = torch.zeros(batch_size, 3).to("cuda:0")

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

  print(f"Under sparsity_ratio = {sparsity_ratio}, batchsize = {batch_size}, inference latency on GPU 3090 = {(end_time - start_time) / total_inference}")
  latency_list.append((end_time - start_time) / total_inference)

print(latency_list)