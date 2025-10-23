
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
from models import DeepAppearanceVAE_Partition

use_traced_model = True

device = torch.device("cuda", 0)
batch_size = 1
model = DeepAppearanceVAE_Partition(
    1024, 21918, n_latent=256, n_cams=38
).to(device)
model.eval()

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

print(f"Under Batchsize = {batch_size}, inference latency on GPU = {(end_time - start_time) / total_inference}")
