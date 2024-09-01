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
from models import DeepAppearanceVAEBDCT

num_freq_comp_outsourced_list = [2,4,6,8,10,12,14]
for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
    result_path = "/tmp/tmp"
    save_latent_code = False
    batch_size = 1024

    device = torch.device("cuda", 0)

    model = DeepAppearanceVAEBDCT(
        1024, 21918, n_latent=256, n_cams=38, num_freq_comp_outsourced=num_freq_comp_outsourced, result_path=result_path, save_latent_code=save_latent_code
    ).to(device)

    z_local = torch.zeros(batch_size, 256).to("cuda:0")
    z_outsource = torch.zeros(batch_size, 256).to("cuda:0")
    v = torch.zeros(batch_size, 3).to("cuda:0")

    start_time = time.time()

    for i in tqdm(range(1000)):
        model.dec(z_local, z_outsource, v)

    end_time = time.time()

    print(f"Under Batchsize = {batch_size}, inference latency on GPU 3090 = {(end_time - start_time) / 2}")
