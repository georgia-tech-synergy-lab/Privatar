# from scipy.fftpack import dct, idct
# import torch_dct as dct_2d, idct_2d
import os 
import torch
import numpy as np
from PIL import Image
from torchjpeg import dct
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms

block_size = 4
total_frequency_components = block_size * block_size
check_reconstruct_img = True
save_block_img_to_drive = False

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension 

def img_reorder(x, bs, ch, h, w):
    x = (x + 1) / 2 * 255
    assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(block_size, block_size), dilation=1, padding=0, stride=(block_size, block_size))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, block_size, block_size)
    return x


## Image reordering and testing
def img_inverse_reroder(coverted_img, bs, ch, h, w):
    x = coverted_img.view(bs* ch, -1, total_frequency_components)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(h, w), kernel_size=(block_size, block_size), stride=(block_size, block_size))
    x += 128
    x = x.view(bs, ch, h, w)
    x = dct.to_rgb(x)#.squeeze(0)
    x = (x / 255.0) * 2 - 1
    return x


## Image frequency cosine transform
def test_img_dct_transform_reorder_noise(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_num = h // 4
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb):
        dct_block_reorder[:, :, i, :, :] = dct_block_reorder[:, :, freq_comp_lb, :, :]
 
    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

def test_img_dct_transform_drop_low_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb):
        dct_block_reorder[:, :, i, :, :] = dct_block_reorder[:, :, i, :, :].zero_()
 
    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    # print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform_drop_high_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb, total_block_num, 1):
        dct_block_reorder[:, :, i, :, :] = dct_block_reorder[:, :, i, :, :].zero_()
        # dct_block_reorder[:, :, i, :, :] = torch.zeros_like(dct_block_reorder[:, :, i, :, :])
 
    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    # print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img


## Image frequency cosine transform
def test_img_dct_transform_duplicate_freq_reorder(x, bs, ch, h, w, freq_comp_lb):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    total_block_num = block_size * block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb, total_block_num, 1):
        dct_block_reorder[:, :, i, :, :] = dct_block_reorder[:, :, freq_comp_lb, :, :]

    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img


## Image frequency cosine transform
def test_img_dct_transform_reorder_noise_outsource(x, bs, ch, h, w, freq_comp_lb, path_variance_matrix_tensor, add_noise):
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_num = h // 4
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
    
    for i in range(freq_comp_lb):
        dct_block_reorder[:, :, i, :, :] = dct_block_reorder[:, :, freq_comp_lb, :, :]
    if add_noise:
        mean = np.zeros(256)
        variance_matrix_tensor = torch.load(path_variance_matrix_tensor).cpu()
        total_sample_noises = dct_block_reorder.shape[0] * dct_block_reorder.shape[1] * dct_block_reorder.shape[2] * dct_block_reorder.shape[3]
        noise_sample = torch.from_numpy(np.random.multivariate_normal(mean, variance_matrix_tensor.detach().numpy(), total_sample_noises)).to(torch.float)
        noise_sample = noise_sample.reshape(dct_block_reorder.shape)
        dct_block_reorder = dct_block_reorder + noise_sample
    
    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    return inverse_transformed_img

## Image frequency cosine transform
def test_img_dct_transform(x, bs, ch, h, w):
    back_input = x
    rerodered_img = img_reorder(x, bs, ch, h, w)
    block_size = 4
    block_num = h // block_size
    dct_block = dct.block_dct(rerodered_img) #BDCT
    dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, total_frequency_components).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
 
    idct_dct_block_reorder = dct_block_reorder.view(bs, ch, total_frequency_components, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, block_size, block_size)
    inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
    inverse_transformed_img = img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
    print(torch.allclose(inverse_transformed_img, back_input, atol=1e-4))
    return inverse_transformed_img

class Renderer:
    def __init__(self):
        self.glctx = dr.RasterizeGLContext()
        # self.glctx = dr.RasterizeCudaContext()

    def render(self, M, pos, pos_idx, uv, uv_idx, tex, resolution=[2048, 1334]):
        ones = torch.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = torch.cat((pos, ones), -1)
        projected = torch.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)
        proj = torch.zeros_like(projected)
        proj[..., 0] = (
            projected[..., 0] / (resolution[1] / 2) - projected[..., 2]
        ) / projected[..., 2]
        proj[..., 1] = (
            projected[..., 1] / (resolution[0] / 2) - projected[..., 2]
        ) / projected[..., 2]
        clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = torch.cat(
            (proj, torch.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1
        )
        pos_idx_flat = pos_idx.view((-1, 3)).contiguous()
        uv_idx = uv_idx.view((-1, 3)).contiguous()
        tex = tex.permute((0, 2, 3, 1)).contiguous()

        rast_out, rast_out_db = dr.rasterize(
            self.glctx, pos_view, pos_idx_flat, resolution
        )
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        color = dr.texture(tex, texc, filter_mode="linear")
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background.
        return color, rast_out

