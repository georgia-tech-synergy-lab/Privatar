# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import wait
import os
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Inserted
from torchjpeg import dct
from PIL import Image
import torchvision.transforms as transforms

save_latent_code_to_external_device = True
noisy_training = True

# prefix_path_captured_latent_code = prefix_path_captured_latent_code_god2
Add_noise = False

# Select based on the difference of downsample input --- BDCT frequency block.
all_private_selection = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 40, 41, 42, 43, 48, 49, 50, 56, 57],
[ 0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33], #19
[ 0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 24, 25, 32], #6
[ 0,  1,  2,  3,  8,  9, 10, 16, 17, 24], #5 
[ 0,  1,  2,  8,  9, 10, 16], # 4
[ 0,  1,  2,  8,  9, 16], # 3
[0, 1, 2, 8, 9], # 2.4
[0, 1, 2, 8], # 1.6
[0, 1, 8], # 1
[0, 1], # 0.8
[0]] #0.4

all_possible_idx = [i for i in range(64)]
selected_privacy_idx = -2

private_idx = all_private_selection[selected_privacy_idx]
public_idx = []

for element in all_possible_idx:
    if element not in private_idx:
        public_idx.append(element)

if noisy_training:
    variance_matrix_tensor = torch.load("/home/jianming/work/Privatar_prj/profiled_latent_code/noise_variance_matrix_horizontal_partition_4_mutual_bound_0.1_outsource_path_latent.pth")
    mean = np.zeros(256)

class DeepAppearanceVAE_Horizontal_Partition(nn.Module):
    def __init__(
        self,
        tex_size=1024,
        mesh_inp_size=21918,
        mode="vae",
        n_latent=128,
        n_cams=38,
        n_blocks=8,
        frequency_threshold=19,
        average_texture_path="/home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png",
        prefix_path_captured_latent_code="/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_",
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DeepAppearanceVAE_Horizontal_Partition, self).__init__()
        
        z_dim = n_latent if mode == "vae" else n_latent * 2
        self.mode = mode
        self.interpolation_ratio = 2
        tex_size = tex_size * self.interpolation_ratio // n_blocks 
        
        self.frequency_threshold = frequency_threshold
        self.private_idx, self.public_idx = self.private_freq_component_thres_based_selection(average_texture_path, frequency_threshold) # ToDo
        
        self.private_total_in_chnl = len(self.private_idx) * 3 # 3 indicates RGB three channels of each frequency broken down
        self.public_total_in_chnl = len(self.public_idx) * 3 # 3 indicates RGB three channels of each frequency broken down
        print(f"public_total_in_chnl={self.public_total_in_chnl}, private_total_in_chnl={self.private_total_in_chnl}")
        self.enc = DeepApperanceEncoderChnlConfig(
            tex_size, mesh_inp_size, n_latent=z_dim, n_in_chnl=self.private_total_in_chnl, res=res
        )
        self.dec = DeepAppearanceDecoderChnlConfig(
            tex_size, mesh_inp_size, z_dim=z_dim, n_in_chnl=self.private_total_in_chnl, res=res, non=non, bilinear=bilinear
        )
        self.enc_outsource = DeepApperanceEncoderNoMeshChnlConfig(
            tex_size, mesh_inp_size, n_latent=z_dim, n_in_chnl=self.public_total_in_chnl, res=res
        )
        self.dec_outsource = DeepAppearanceDecoderNoMeshChnlConfig(
            tex_size, mesh_inp_size, z_dim=z_dim, n_in_chnl=self.public_total_in_chnl, res=res, non=non, bilinear=bilinear
        )
        self.cc = ColorCorrection(n_cams)
        self.iter = 0
        self.iter_outsource = 0
        self.prefix_path_captured_latent_code = prefix_path_captured_latent_code
        if not os.path.exists(f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code"):
            os.makedirs(f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code")

    def img_reorder(self, x, bs, ch, h, w):
        x = (x + 1) / 2 * 255
        assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
        x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
        x -= 128
        x = x.view(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0, stride=(8, 8))
        x = x.transpose(1, 2)
        x = x.view(bs, ch, -1, 8, 8)
        return x
    ## Image reordering and testing
    def img_inverse_reroder(self, coverted_img, bs, ch, h, w):
        x = coverted_img.view(bs* ch, -1, 64)
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=(h, w), kernel_size=(8, 8), stride=(8, 8))
        x += 128
        x = x.view(bs, ch, h, w)
        x = dct.to_rgb(x)#.squeeze(0)
        x = (x / 255.0) * 2 - 1
        return x

    def calculate_block_mse(self, downsample_in, freq_block, num_freq_component=8):
        downsample_img = transforms.Resize(size=int(downsample_in.shape[-1]/num_freq_component))(downsample_in)
        assert(downsample_img.shape == freq_block[:,:,0,:,:].shape, "downsample input shape does not match the shape of post-BDCT component")
        loss_vector = torch.zeros(freq_block.shape[2])
        for i in range(freq_block.shape[2]):
            # calculate the MSE between each frequency components and given input downsampled images
            loss_vector[i] = F.mse_loss(downsample_img, freq_block[:,:,i,:,:])
        return loss_vector

    def private_freq_component_thres_based_selection(self, img_path, mse_threshold):
        # The original input image comes with it and I disable it to reduce the computation overhead.
        # x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        x = transform(image).unsqueeze(0)

        back_input = x
        bs, ch, h, w = x.shape
        block_num = h // 8
        x = self.img_reorder(x, bs, ch, h, w)
        dct_block = dct.block_dct(x) # BDCT
        dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
        loss_vector = self.calculate_block_mse(back_input, dct_block_reorder)
        # Split all component based on the frequency
        private_idx = torch.where(loss_vector > mse_threshold)[0]
        public_idx = []
        all_possible_idx = [i for i in range(64)]
        for element in all_possible_idx:
            if element not in private_idx:
                public_idx.append(element)

        return private_idx,  torch.Tensor(public_idx).to(torch.int64)

    def forward(self, avgtex, mesh, view, cams=None):
# The mesh also needs to be partitioned by two sets of models
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))
        # process input avgtex
        avgtex_interpolate = F.interpolate(avgtex, scale_factor=2, mode='bilinear', align_corners=True)
        x = avgtex_interpolate
        x = (x + 1) / 2 * 255
        if x.shape[1] != 3:
            print("Wrong input, Channel should equals to 3")
            return
        x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
        x -= 128
        bs, ch, h, w = x.shape
        block_num = h // 8
        x = x.view(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0, stride=(8, 8))
        x = x.transpose(1, 2)
        x = x.view(bs, ch, -1, 8, 8)
        dct_block = dct.block_dct(x) # BDCT
        dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3) # into (bs, ch, 64, block_num, block_num)
        private_dct_block = dct_block_reorder[:,:,self.private_idx,:,:].view(bs, self.private_total_in_chnl, block_num, block_num)
        public_dct_block = dct_block_reorder[:,:,self.public_idx,:,:].view(bs, self.public_total_in_chnl, block_num, block_num)
        # Inserted Horizontal Parationing logic -- Done

        # mean, logstd = self.enc(avgtex, mesh) # Comment out for enabling horizontal partitioning
        mean, logstd = self.enc(private_dct_block, mesh)
        mean = mean * 0.1
        logstd = logstd * 0.01
        if self.mode == "vae":
            kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
            std = torch.exp(logstd)
            eps = torch.randn_like(mean)
            z = mean + std * eps
        else:
            z = torch.cat((mean, logstd), -1)
            kl = torch.tensor(0).to(z.device)
        
        if save_latent_code_to_external_device:
            path_captured_latent_code = f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code"
            torch.save(logstd, f"{path_captured_latent_code}/logstd_{self.iter}.pth")
            torch.save(mean, f"{path_captured_latent_code}/mean_{self.iter}.pth")
            torch.save(z, f"{path_captured_latent_code}/z_{self.iter}.pth")
            torch.save(kl, f"{path_captured_latent_code}/kl_{self.iter}.pth")
            self.iter = self.iter + 1

        pred_tex_private, pred_mesh = self.dec(z, view)
        pred_tex_private = pred_tex_private.view(bs, ch, -1, block_num, block_num)
        
        
        mean_outsource, logstd_outsource = self.enc_outsource(public_dct_block)
        mean_outsource = mean_outsource * 0.1
        logstd_outsource = logstd_outsource * 0.01
        if self.mode == "vae":
            std_outsource = torch.exp(logstd_outsource)
            eps_outsource = torch.randn_like(mean_outsource)
            z_outsource = mean_outsource + std_outsource * eps_outsource
        else:
            z_outsource = torch.cat((mean_outsource, logstd_outsource), -1)
        
        # Adding model outsource encoder
        if noisy_training:
            samples = torch.from_numpy(np.random.multivariate_normal(mean, variance_matrix_tensor.detach().numpy(), 1))
            z_outsource = z_outsource + samples

        if save_latent_code_to_external_device:
            path_captured_latent_code = f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code"
            torch.save(logstd_outsource, f"{path_captured_latent_code}/logstd_outsource_{self.iter_outsource}.pth")
            torch.save(mean_outsource, f"{path_captured_latent_code}/mean_outsource_{self.iter_outsource}.pth")
            torch.save(z_outsource, f"{path_captured_latent_code}/z_outsource_{self.iter_outsource}.pth")
            self.iter_outsource = self.iter_outsource + 1
        
        pred_tex_outsource = self.dec_outsource(z_outsource, view)
        pred_tex_outsource = pred_tex_outsource.view(bs, ch, -1, block_num, block_num)
        # Adding model outsource encoder -- Done

        # Adding block reconstructions
        pred_tex = torch.zeros(bs, ch, 64, block_num, block_num).to(pred_tex_outsource.device)

        for i, idx in enumerate(self.private_idx):
            pred_tex[:, :, idx, :, :] = pred_tex_private[:, :, i, :, :]
        for i, idx in enumerate(self.public_idx):
            pred_tex[:, :, idx, :, :] = pred_tex_outsource[:, :, i, :, :]
        # Adding block reconstructions -- Done
        # # reorder to revert the layout
        idct_dct_block_reorder = pred_tex.view(bs, ch, 64, block_num*block_num).permute(0, 1, 3, 2).view(bs, ch, block_num*block_num, 8, 8)

        idct_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT

        ## Reconstruct the overall original input image
        pred_tex = self.img_inverse_reroder(idct_dct_block, bs, ch, h, w)
        pred_tex = transforms.Resize(size=1024)(pred_tex)
        pred_mesh = pred_mesh.view((b, n, 3))
        if cams is not None:
            pred_tex = self.cc(pred_tex, cams)

        return pred_tex, pred_mesh, kl

    def get_mesh_branch_params(self):
        p = self.enc.get_mesh_branch_params() + self.dec.get_mesh_branch_params()
        return p

    def get_tex_branch_params(self):
        p = self.enc.get_tex_branch_params() + self.dec.get_tex_branch_params()
        return p

    def get_model_params(self):
        params = []
        params += list(self.enc.parameters())
        params += list(self.dec.parameters())
        return params

    def get_cc_params(self):
        return self.cc.parameters()


class WarpFieldDecoder(nn.Module):
    def __init__(self, tex_size=1024, z_dim=128):
        super(WarpFieldDecoder, self).__init__()
        self.mlp = nn.Sequential(
            LinearWN(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            LinearWN(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvUpsample(256, 128, 128, 2),
            ConvUpsample(128, 128, 64, 2 * (2**2)),
            ConvUpsample(64, 64, 32, 2 * (2**4)),
            ConvUpsample(32, 32, 16, 2 * (2**6)),
            nn.Upsample(size=tex_size, mode='bilinear'),
            nn.Conv2d(16, 2, 3, 1, 1),
        )

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.upsample[-1], 1.0)

        self.tex_size = tex_size

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, tex_size), np.linspace(-1.0, 1.0, tex_size))
        grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("normal_grid", torch.from_numpy(grid))

    def forward(self, z, img):
        b, c, h, w = img.shape

        feat = self.mlp(z).view((-1, 256, 2, 2))
        warp = self.upsample(feat) / self.tex_size
        grid = warp + self.normal_grid
        if grid.shape[2] < img.shape[2]:
            grid = F.interpolate(grid, scale_factor=img.shape[2] / grid.shape[2])
        grid = grid.permute(0, 2, 3, 1).contiguous()

        out = F.grid_sample(img, grid)
        return out, grid


class DeepAppearanceDecoderChnlConfig(nn.Module):
    def __init__(
        self, tex_size, mesh_size, z_dim=128, n_in_chnl=3, res=False, non=False, bilinear=False
    ):
        super(DeepAppearanceDecoderChnlConfig, self).__init__()
        nhidden = z_dim * 4 * 4 # if tex_size == 1024 else z_dim * 2 * 2
        self.texture_decoder = TextureDecoder_Chnl_Config(
            tex_size, z_dim, n_in_chnl, res=res, non=non, bilinear=bilinear
        )
        self.view_fc = LinearWN(3, 8)
        self.z_fc = LinearWN(z_dim, 256)
        self.mesh_fc = LinearWN(256, mesh_size)
        self.texture_fc = LinearWN(256 + 8, nhidden)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.mesh_fc, 1.0)
        glorot(self.texture_decoder.upsample[-1].conv2, 1.0)

    def forward(self, z, v):
        view_code = self.relu(self.view_fc(v))
        z_code = self.relu(self.z_fc(z))
        feat = torch.cat((view_code, z_code), 1)
        texture_code = self.relu(self.texture_fc(feat))
        texture = self.texture_decoder(texture_code)
        mesh = self.mesh_fc(z_code)
        return texture, mesh

    def get_mesh_branch_params(self):
        return list(self.mesh_fc.parameters())

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_decoder.parameters())
        p += list(self.view_fc.parameters())
        p += list(self.z_fc.parameters())
        p += list(self.texture_fc.parameters())
        return p


class DeepApperanceEncoderNoMeshChnlConfig(nn.Module):
    def __init__(self, inp_size=1024, mesh_inp_size=21918, n_latent=128, n_in_chnl=3, res=False):
        super(DeepApperanceEncoderNoMeshChnlConfig, self).__init__()
        self.n_latent = n_latent
        ntexture_feat = 2048 #if inp_size == 1024 else 512
        self.texture_encoder = TextureEncoder_Chnl_Config(n_in_chnl=n_in_chnl, res=res)
        self.texture_fc = LinearWN(ntexture_feat, 256)
        # self.fc = LinearWN(512, n_latent * 2) # Horizontal Partitioning Modification (Remove Mesh and corresponding input channels from 512 -> 256)
        self.fc = LinearWN(256, n_latent * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.fc, 1.0)

    def forward(self, tex):
        tex_feat = self.relu(self.texture_fc(self.texture_encoder(tex)))
        latent = self.fc(tex_feat)
        return latent[:, : self.n_latent], latent[:, self.n_latent :]

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_encoder.parameters())
        p += list(self.texture_fc.parameters())
        p += list(self.fc.parameters())
        return p
    

class DeepApperanceEncoderChnlConfig(nn.Module):
    def __init__(self, inp_size=1024, mesh_inp_size=21918, n_latent=128, n_in_chnl=3, res=False):
        super(DeepApperanceEncoderChnlConfig, self).__init__()
        self.n_latent = n_latent
        # ntexture_feat = 2048 if inp_size == 1024 else 512
        ntexture_feat = 2048 #if inp_size == 1024 else 128
        self.texture_encoder = TextureEncoder_Chnl_Config(n_in_chnl=n_in_chnl, res=res)
        self.texture_fc = LinearWN(ntexture_feat, 256)
        self.mesh_fc = LinearWN(mesh_inp_size, 256)
        self.fc = LinearWN(512, n_latent * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.fc, 1.0)

    def forward(self, tex, mesh):
        tex_feat = self.relu(self.texture_fc(self.texture_encoder(tex)))
        mesh_feat = self.relu(self.mesh_fc(mesh))
        feat = torch.cat((tex_feat, mesh_feat), -1)
        latent = self.fc(feat)
        return latent[:, : self.n_latent], latent[:, self.n_latent :]

    def get_mesh_branch_params(self):
        return list(self.mesh_fc.parameters())

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_encoder.parameters())
        p += list(self.texture_fc.parameters())
        p += list(self.fc.parameters())
        return p


class DeepAppearanceDecoderNoMeshChnlConfig(nn.Module):
    def __init__(
        self, tex_size, mesh_size, z_dim=128, n_in_chnl=3, res=False, non=False, bilinear=False
    ):
        super(DeepAppearanceDecoderNoMeshChnlConfig, self).__init__()
        nhidden = z_dim * 4 * 4 #if tex_size == 1024 else z_dim * 2 * 2
        self.texture_decoder = TextureDecoder_Chnl_Config(
            tex_size, z_dim, n_in_chnl=n_in_chnl, res=res, non=non, bilinear=bilinear
        )
        self.view_fc = LinearWN(3, 8)
        self.z_fc = LinearWN(z_dim, 256)
        self.texture_fc = LinearWN(256 + 8, nhidden)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.texture_decoder.upsample[-1].conv2, 1.0)

    def forward(self, z, v):
        view_code = self.relu(self.view_fc(v))
        z_code = self.relu(self.z_fc(z))
        feat = torch.cat((view_code, z_code), 1)
        texture_code = self.relu(self.texture_fc(feat))
        texture = self.texture_decoder(texture_code)
        return texture

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_decoder.parameters())
        p += list(self.view_fc.parameters())
        p += list(self.z_fc.parameters())
        p += list(self.texture_fc.parameters())
        return p


class TextureDecoder(nn.Module):
    def __init__(self, tex_size, z_dim, res=False, non=False, bilinear=False):
        super(TextureDecoder, self).__init__()
        base = 2 if tex_size == 512 else 4
        self.z_dim = z_dim

        self.upsample = nn.Sequential(
            ConvUpsample(
                z_dim, z_dim, 64, base, res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                64, 64, 32, base * (2**2), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                32, 32, 16, base * (2**4), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                16,
                16,
                3,
                base * (2**6),
                no_activ=True,
                res=res,
                use_bilinear=bilinear,
                non=non,
            ),
        )

    def forward(self, x):
        b, n = x.shape
        h = int(np.sqrt(n / self.z_dim))
        x = x.view((-1, self.z_dim, h, h))
        out = self.upsample(x)
        return out


class TextureEncoder(nn.Module):
    def __init__(self, res=False):
        super(TextureEncoder, self).__init__()
        self.downsample = nn.Sequential(
            ConvDownsample(3, 16, 16, res=res),
            ConvDownsample(16, 32, 32, res=res),
            ConvDownsample(32, 64, 64, res=res),
            ConvDownsample(64, 128, 128, res=res),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.downsample(x)
        out = feat.view((b, -1))
        return out


class TextureEncoder_Chnl_Config(nn.Module):
    def __init__(self, n_in_chnl=3, res=False):
        super(TextureEncoder_Chnl_Config, self).__init__()
        self.downsample = nn.Sequential(
            # ConvDownsample(n_in_chnl, 16, 16, res=res),
            ConvDownsample(n_in_chnl, 32, 32, res=res),
            ConvDownsample(32, 64, 64, res=res),
            ConvDownsample(64, 128, 128, res=res),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.downsample(x)
        out = feat.view((b, -1))
        return out

class TextureDecoder_Chnl_Config(nn.Module):
    def __init__(self, tex_size, z_dim, n_in_chnl, res=False, non=False, bilinear=False):
        super(TextureDecoder_Chnl_Config, self).__init__()
        base = 2 if tex_size == 512 else 4
        self.z_dim = z_dim
        if bilinear:
            print("user bilinear")
        self.upsample = nn.Sequential(
            ConvUpsample(
                z_dim, z_dim, 64, base, res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                64, 64, 32, base * (2**2), res=res, use_bilinear=bilinear, non=non
            ),
            # ConvUpsample(
            #     32, 32, 16, base * (2**4), res=res, use_bilinear=bilinear, non=non
            # ),
            ConvUpsample(
                32,
                32,
                n_in_chnl,
                base * (2**4),
                no_activ=True,
                res=res,
                use_bilinear=bilinear,
                non=non,
            ),
        )

    def forward(self, x):
        b, n = x.shape
        h = int(np.sqrt(n / self.z_dim))
        x = x.view((-1, self.z_dim, h, h))
        out = self.upsample(x)
        return out

class MLP(nn.Module):
    def __init__(self, nin, nhidden, nout):
        self.fc1 = LinearWN(nin, nhidden)
        self.fc2 = LinearWN(nhidden, nout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out


class ConvDownsample(nn.Module):
    def __init__(self, cin, chidden, cout, res=False):
        super(ConvDownsample, self).__init__()
        self.conv1 = Conv2dWN(cin, chidden, 4, 2, padding=1)
        self.conv2 = Conv2dWN(chidden, cout, 4, 2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res = res
        if res:
            self.res1 = Conv2dWN(chidden, chidden, 3, 1, 1)
            self.res2 = Conv2dWN(cout, cout, 3, 1, 1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        if self.res:
            h = self.relu(self.res1(h) + h)
        h = self.relu(self.conv2(h))
        if self.res:
            h = self.relu(self.res2(h) + h)
        return h


class ConvUpsample(nn.Module):
    def __init__(
        self,
        cin,
        chidden,
        cout,
        feature_size,
        no_activ=False,
        res=False,
        use_bilinear=False,
        non=False,
    ):
        super(ConvUpsample, self).__init__()
        self.conv1 = DeconvTexelBias(
            cin, chidden, feature_size * 2, use_bilinear=use_bilinear, non=non
        )
        self.conv2 = DeconvTexelBias(
            chidden, cout, feature_size * 4, use_bilinear=use_bilinear, non=non
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.no_activ = no_activ
        self.res = res
        if self.res:
            self.res1 = Conv2dWN(chidden, chidden, 3, 1, 1)
            self.res2 = Conv2dWN(cout, cout, 3, 1, 1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        if self.res:
            h = self.relu(self.res1(h) + h)
        if self.no_activ:
            h = self.conv2(h)
            if self.res:
                h = self.res2(h) + h
        else:
            h = self.relu(self.conv2(h))
            if self.res:
                h = self.relu(self.res2(h) + h)
        return h


class DeconvTexelBias(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        feature_size,
        ksize=4,
        stride=2,
        padding=1,
        use_bilinear=False,
        non=False,
    ):
        super(DeconvTexelBias, self).__init__()
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.deconv = Conv2dWN(cin, cout, 3, 1, 1, bias=False)
        else:
            self.deconv = ConvTranspose2dWN(
                cin, cout, ksize, stride, padding, bias=False
            )
        if non:
            self.bias = nn.Parameter(torch.zeros(1, cout, 1, 1))
        else:
            self.bias = nn.Parameter(
                torch.zeros(1, cout, feature_size[0], feature_size[1])
            )

    def forward(self, x):
        if self.use_bilinear:
            x = F.interpolate(x, scale_factor=2)
        out = self.deconv(x) + self.bias
        return out


"""
class ColorCorrection(nn.Module):
    def __init__(self, n_cameras, nc=3):
        super(ColorCorrection, self).__init__()
        weights = torch.zeros(n_cameras, nc, nc, 1, 1)
        for i in range(nc):
            weights[:, i, i] = 1
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.zeros(n_cameras, nc))
        self.anchor = 0
    def forward(self, texture, cam):
        b, c, h, w = texture.shape
        texture = texture.view(1, b*c, h, w)
        weight = self.weights[cam]
        bias = self.bias[cam]
        if self.training:
            weight[cam == self.anchor] = torch.eye(3).view(3, 3, 1, 1).to(texture.device)
            bias[cam == self.anchor] = 0
        weight = weight.view(b*c, 3, 1, 1)
        bias = bias.view(b*c)
        out = F.conv2d(texture, weight, bias, groups=b).view(b, c, h, w)
        return out
"""


class ColorCorrection(nn.Module):
    def __init__(self, n_cameras, nc=3):
        super(ColorCorrection, self).__init__()
        # anchors the 0th camera
        self.weight_anchor = nn.Parameter(torch.ones(1, nc, 1, 1), requires_grad=False)
        self.bias_anchor = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(n_cameras - 1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_cameras - 1, 3, 1, 1))

    def forward(self, texture, cam):
        weights = torch.cat([self.weight_anchor, self.weight], dim=0)
        biases = torch.cat([self.bias_anchor, self.bias], dim=0)
        w = weights[cam]
        b = biases[cam]
        output = texture * w + b
        return output


def glorot(m, alpha):
    gain = math.sqrt(2.0 / (1.0 + alpha**2))

    if isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return

    # m.weight.data.normal_(0, std)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))
    m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    # if isinstance(m, Conv2dWNUB) or isinstance(m, ConvTranspose2dWNUB) or isinstance(m, LinearWN):
    if (
        isinstance(m, Conv2dWNUB)
        or isinstance(m, Conv2dWN)
        or isinstance(m, ConvTranspose2dWN)
        or isinstance(m, ConvTranspose2dWNUB)
        or isinstance(m, LinearWN)
    ):
        norm = np.sqrt(torch.sum(m.weight.data[:] ** 2))
        m.g.data[:] = norm


class LinearWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN, self).__init__(in_features, out_features, bias)
        self.g = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.linear(input, self.weight * self.g[:, None] / wnorm, self.bias)


class Conv2dWNUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dWNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return (
            F.conv2d(
                x,
                self.weight * self.g[:, None, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class Conv2dWN(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.conv2d(
            x,
            self.weight * self.g[:, None, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2dWNUB(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvTranspose2dWNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return (
            F.conv_transpose2d(
                x,
                self.weight * self.g[None, :, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class ConvTranspose2dWN(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ConvTranspose2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.conv_transpose2d(
            x,
            self.weight * self.g[None, :, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
