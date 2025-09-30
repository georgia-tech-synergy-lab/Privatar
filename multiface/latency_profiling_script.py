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
from models import DeepAppearanceVAE

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

device = torch.device("cpu")
# device = torch.device("cuda", 0)
batch_size = 1
model = DeepAppearanceVAE(
    1024, 21918, n_latent=256, n_cams=38
).to(device)

best_model_path = "/work/pretrain_model/6795937_best_model.pth"
model.load_state_dict(remove_module_prefix(torch.load(best_model_path)))

z_local = torch.zeros(batch_size, 256).to(device)
v = torch.zeros(batch_size, 3).to(device)

start_time = time.time()
total_inference = 100

for i in tqdm(range(total_inference)):
    model.dec(z_local, v)

end_time = time.time()

print(f"Under Batchsize = {batch_size}, inference latency on GPU = {(end_time - start_time) / total_inference}")
