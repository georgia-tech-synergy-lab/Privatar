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
from models import DeepAppearanceVAE_IBDCT

num_freq_comp_outsourced_list = [2,4,6,8,10,12,14]
latency_list = []
for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
    result_path = "/tmp/tmp"
    save_latent_code = False
    batch_size = 1

    device = torch.device("cuda", 0)

    model = DeepAppearanceVAE_IBDCT(
        1024, 21918, n_latent=256, n_cams=38, num_freq_comp_outsourced=num_freq_comp_outsourced, result_path=result_path, save_latent_code=save_latent_code
    ).to(device)
    z_local = torch.zeros(batch_size, 256).to("cuda:0")
    z_outsource = torch.zeros(batch_size, 256).to("cuda:0")
    v = torch.zeros(batch_size, 3).to("cuda:0")

    # Generate sample texture_outsource for tracing
    with torch.no_grad():
        view_code = model.dec.relu(model.dec.view_fc(v))
        z_code_outsource = model.dec.relu(model.dec.z_fc_outsource(z_outsource))
        feat_outsource = torch.cat((view_code, z_code_outsource), 1)
        texture_code_outsource = model.dec.relu(model.dec.texture_fc_outsource(feat_outsource))
        texture_outsource = model.dec.texture_decoder_outsource(texture_code_outsource)

    # Create a traced model for faster inference
    class TracedDecoder(torch.nn.Module):
        def __init__(self, model_dec):
            super().__init__()
            self.model_dec = model_dec
            
        def forward(self, v, z_local):
            view_code = self.model_dec.relu(self.model_dec.view_fc(v))
            z_code = self.model_dec.relu(self.model_dec.z_fc(z_local))
            mesh = self.model_dec.mesh_fc(z_code)
            texture_code = self.model_dec.relu(self.model_dec.texture_fc(torch.cat((view_code, z_code), 1)))
            texture_local = self.model_dec.texture_decoder_local(texture_code)
            return texture_local
    
    # Create and trace the model
    traced_decoder = TracedDecoder(model.dec)
    traced_decoder.eval()

    # Create sample inputs for tracing with correct dimensions
    sample_v = torch.zeros(batch_size, 3).to(device)
    sample_z_local = torch.zeros(batch_size, 256).to(device)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(traced_decoder, (sample_v, sample_z_local))

    # Warm up the traced model
    with torch.no_grad():
        for _ in range(10):
            _ = traced_model(sample_v, sample_z_local)

    start_time = time.time()
    total_inference = 10000

    for i in tqdm(range(total_inference)):
        # Run the traced model for faster inference
        with torch.no_grad():
            texture_local = traced_model(v, z_local)

    end_time = time.time()

    print(f"Under Batchsize = {batch_size}, inference latency on GPU 3090 = {(end_time - start_time) / total_inference}")
    latency_list.append((end_time - start_time) / total_inference)

print(latency_list)