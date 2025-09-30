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
from models import DeepAppearanceVAE_Partition

num_freq_comp_offloaded_list = [2,4,6,8,10,12,14]
latency_list = []
for num_freq_comp_offloaded in num_freq_comp_offloaded_list:
    result_path = "/tmp/tmp"
    save_latent_code = False
    batch_size = 1

    device = torch.device("cpu")
    # device = torch.device("cuda", 0)

    model = DeepAppearanceVAE_Partition(
        1024, 21918, n_latent=256, n_cams=38, num_freq_comp_offloaded=num_freq_comp_offloaded, result_path=result_path, save_latent_code=save_latent_code
    ).to(device)
    z_local = torch.zeros(batch_size, 256).to(device)
    z_offload = torch.zeros(batch_size, 256).to(device)
    v = torch.zeros(batch_size, 3).to(device)

    # Generate sample texture_offload for tracing
    with torch.no_grad():
        view_code = model.dec.relu(model.dec.view_fc(v))
        z_code_offload = model.dec.relu(model.dec.z_fc_offload(z_offload))
        feat_offload = torch.cat((view_code, z_code_offload), 1)
        texture_code_offload = model.dec.relu(model.dec.texture_fc_offload(feat_offload))
        texture_offload = model.dec.texture_decoder_offload(texture_code_offload)

    # Create a traced model for faster inference
    class TracedDecoder(torch.nn.Module):
        def __init__(self, model_dec):
            super().__init__()
            self.model_dec = model_dec

        def forward(self, v, z_offload):
            view_code = self.model_dec.relu(self.model_dec.view_fc(v))
            z_code_offload = self.model_dec.relu(self.model_dec.z_fc_offload(z_offload))
            feat_offload = torch.cat((view_code, z_code_offload), 1)
            texture_code_offload = self.model_dec.relu(self.model_dec.texture_fc_offload(feat_offload))
            texture_offload = self.model_dec.texture_decoder_offload(texture_code_offload)
            return texture_offload

    # Create and trace the model
    traced_decoder = TracedDecoder(model.dec)
    traced_decoder.eval()

    # Create sample inputs for tracing with correct dimensions
    sample_v = torch.zeros(batch_size, 3).to(device)
    sample_z_offload = torch.zeros(batch_size, 256).to(device)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(traced_decoder, (sample_v, sample_z_offload))

    # Warm up the traced model
    with torch.no_grad():
        for _ in range(10):
            _ = traced_model(sample_v, sample_z_offload)

    start_time = time.time()
    total_inference = 100

    for i in tqdm(range(total_inference)):
        # Run the traced model for faster inference
        with torch.no_grad():
            texture_offload = traced_model(v, z_offload)

    end_time = time.time()

    print(f"Under Batchsize = {batch_size}, inference latency on GPU = {(end_time - start_time) / total_inference}")
    latency_list.append((end_time - start_time) / total_inference)

print(latency_list)
