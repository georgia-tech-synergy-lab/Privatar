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

result_path = "/tmp/tmp"
save_latent_code = False
batch_size = 16

device = torch.device("cuda", 0)

model = DeepAppearanceVAE_Partition(
    1024, 21918, n_latent=256, n_cams=38, result_path=result_path, save_latent_code=save_latent_code
).to(device)

z = torch.zeros(batch_size, 256).to("cuda:0")
v = torch.zeros(batch_size, 3).to("cuda:0")

start_time = time.time()

for i in tqdm(range(1000)):
    model.dec(z, v)

end_time = time.time()

print(f"Under Batchsize = {batch_size}, inference latency on GPU = {(end_time - start_time) / 2}")
