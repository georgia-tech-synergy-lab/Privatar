# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import wait

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Inserted
import os
from torchjpeg import dct
from PIL import Image
import torchvision.transforms as transforms


# prefix_path_captured_latent_code = prefix_path_captured_latent_code_god2
threshold_list = [0.1, 0.3, 0.35, 0.4, 0.42, 0.45, 0.5, 0.6, 0.7, 1.1, 1.2, 3.5, 5]

# Select based on the difference of downsample input --- BDCT frequency block.
all_private_selection = [[0],
[0, 1],
[0, 1, 2],
[0, 1, 2, 3],
[0, 1, 2, 3, 4],
[0, 1, 2, 3, 4, 5],
[0, 1, 2, 3, 4, 5, 6],
[0, 1, 2, 3, 4, 5, 6, 7],
[0, 1, 2, 3, 4, 5, 6, 7, 8],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

all_possible_idx = [i for i in range(64)]
selected_privacy_idx = -2

private_idx = all_private_selection[selected_privacy_idx]
public_idx = []

for element in all_possible_idx:
    if element not in private_idx:
        public_idx.append(element)
    

class DeepAppearanceVAEHPLayerRed(nn.Module):
    def __init__(
        self,
        tex_size=1024,
        mesh_inp_size=21918,
        mode="vae",
        n_latent=128,
        n_cams=38,
        n_blocks=4,
        frequency_threshold=19,
        average_texture_path="/home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png",
        prefix_path_captured_latent_code="/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_",
        path_variance_matrix_tensor="/usr/scratch/jianming/Privatar/profiled_latent_code/statistics/noise_variance_matrix_horizontal_partition_6.0_mutual_bound_1.pth",
        save_latent_code_to_external_device = False,
        apply_gaussian_noise = True,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DeepAppearanceVAEHPLayerRed, self).__init__()
        z_dim = n_latent if mode == "vae" else n_latent * 2
        self.mode = mode
        # self.enc = DeepApperanceEncoder(
        #     tex_size, mesh_inp_size, n_latent=z_dim, res=res
        # )

        self.n_latent = z_dim
        ntexture_feat = 16384 if tex_size == 1024 else 512
        self.texture_encoder = TextureEncoderLayerRed(res=res)
        self.texture_fc = LinearWN(ntexture_feat, 256)
        self.mesh_fc = LinearWN(mesh_inp_size, 256) # The Mesh based encoder.
        self.fc = LinearWN(512, z_dim * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.fc, 1.0)

        # self.dec = DeepAppearanceDecoder(
        #     tex_size, mesh_inp_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear
        # )

        nhidden = 16384 # if tex_size == 1024 else z_dim * 2 * 2
        self.dec_texture_decoder = TextureDecoderLayerRed(
            tex_size, 64, res=res, non=True, bilinear=bilinear
        )
        self.dec_view_fc = LinearWN(3, 8)
        self.dec_z_fc = LinearWN(z_dim, 256)
        self.dec_mesh_fc = LinearWN(256, mesh_inp_size)
        self.dec_texture_fc = LinearWN(256 + 8, nhidden)
        self.dec_relu = nn.LeakyReLU(0.2, inplace=True)

        self.cc = ColorCorrection(n_cams)

        # Added Extra Code
        self.block_size = n_blocks
        self.total_frequency_component = self.block_size * self.block_size
        
        self.frequency_threshold = frequency_threshold
        self.private_idx, self.public_idx = self.private_freq_component_thres_based_selection(average_texture_path, frequency_threshold) 
        self.prefix_path_captured_latent_code = prefix_path_captured_latent_code
        self.save_latent_code_to_external_device = save_latent_code_to_external_device
        self.apply_gaussian_noise = apply_gaussian_noise
        self.mean = np.zeros(256)
        if apply_gaussian_noise:
            self.variance_matrix_tensor = torch.load(path_variance_matrix_tensor).cpu()
        directory_being_created = f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code"
        print(f"create directory {directory_being_created}")
        if not os.path.exists(f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code"):
            os.makedirs(f"{self.prefix_path_captured_latent_code}{self.frequency_threshold}_latent_code")

    def img_reorder(self, x, bs, ch, h, w):
        x = (x + 1) / 2 * 255
        assert(x.shape[1] == 3, "Wrong input, Channel should equals to 3")
        x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
        x -= 128
        x = x.view(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(self.block_size, self.block_size), dilation=1, padding=0, stride=(self.block_size, self.block_size))
        x = x.transpose(1, 2)
        x = x.view(bs, ch, -1, self.block_size, self.block_size)
        return x

    ## Image reordering and testing
    def img_inverse_reroder(self, coverted_img, bs, ch, h, w):
        x = coverted_img.view(bs* ch, -1, self.total_frequency_component)
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=(h, w), kernel_size=(self.block_size, self.block_size), stride=(self.block_size, self.block_size))
        x += 128
        x = x.view(bs, ch, h, w)
        x = dct.to_rgb(x)#.squeeze(0)
        x = (x / 255.0) * 2 - 1
        return x
    
    def private_freq_component_thres_based_selection(self, img_path, mse_threshold):
        # The original input image comes with it and I disable it to reduce the computation overhead.
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        x = transform(image).unsqueeze(0)

        back_input = x
        bs, ch, h, w = x.shape
        block_num = h // self.block_size
        x = self.img_reorder(x, bs, ch, h, w)
        dct_block = dct.block_dct(x) # BDCT
        dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, self.total_frequency_component).permute(0, 1, 4, 2, 3) # into (bs, ch, 16, block_num, block_num)
        loss_vector = self.calculate_block_mse(back_input, dct_block_reorder)
        # Split all component based on the frequency
        private_idx = torch.where(loss_vector > mse_threshold)[0]
        public_idx = []
        all_possible_idx = [i for i in range(self.total_frequency_component)]
        for element in all_possible_idx:
            if element not in private_idx:
                public_idx.append(element)

        return private_idx,  torch.Tensor(public_idx).to(torch.int64)

    ## Image frequency cosine transform
    def dct_transform(self, x, bs, ch, h, w):
        rerodered_img = self.img_reorder(x, bs, ch, h, w)
        block_num = h // self.block_size
        dct_block = dct.block_dct(rerodered_img) #BDCT
        dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, self.total_frequency_component).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
        # A given frequency reference is "dct_block_reorder[freq_id, :, :, :, :]"
        return dct_block_reorder

    def dct_inverse_transform(self, dct_block_reorder,bs, ch, h, w):
        block_num = h // self.block_size
        idct_dct_block_reorder = dct_block_reorder.permute(1, 2, 3, 4, 0).view(bs, ch, block_num*block_num, self.block_size, self.block_size)
        inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
        inverse_transformed_img = self.img_inverse_reroder(inverse_dct_block, bs, ch, h, w)
        return inverse_transformed_img

    def calculate_block_mse(self, downsample_in, freq_block):
        downsample_img = transforms.Resize(size=int(downsample_in.shape[-1]/self.block_size))(downsample_in)
        assert(downsample_img.shape == freq_block[:,:,0,:,:].shape, "downsample input shape does not match the shape of post-BDCT component")
        loss_vector = torch.zeros(freq_block.shape[2])
        for i in range(freq_block.shape[2]):
            # calculate the MSE between each frequency components and given input downsampled images
            loss_vector[i] = F.mse_loss(downsample_img, freq_block[:,:,i,:,:])
        return loss_vector

    def forward(self, avgtex, mesh, view, cams=None):
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))

        # mean, logstd = self.enc(avgtex, mesh)
        ## Encoder
        # divide avgtex into frequency components and then feed it into the texture encoder pipeline
        bs, ch, h, w = avgtex.shape
        dct_block_reorder = self.dct_transform(avgtex, bs, ch, h, w)
        digest_tex = torch.empty(bs, self.texture_fc.in_features).to("cuda:0")
        block_features = int(self.texture_fc.in_features / self.total_frequency_component)
        for i in range(self.total_frequency_component):
            digest_tex[:, i*block_features : (i+1)*block_features] = self.texture_encoder(dct_block_reorder[i,:,:,:,:])

        # def forward(self, tex, mesh):
        # digest_tex = self.texture_encoder(avgtex)
        tex_feat = self.relu(self.texture_fc(digest_tex))
        mesh_feat = self.relu(self.mesh_fc(mesh))
        feat = torch.cat((tex_feat, mesh_feat), -1)
        latent = self.fc(feat)

        mean = latent[:, : self.n_latent]
        logstd = latent[:,  self.n_latent :]
        # return latent[:, : self.n_latent], latent[:, self.n_latent :]
        
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

        # pred_tex, pred_mesh = self.dec(z, view)
        view_code = self.dec_relu(self.dec_view_fc(view))
        z_code = self.dec_relu(self.dec_z_fc(z))
        feat = torch.cat((view_code, z_code), 1)
        texture_code = self.dec_relu(self.dec_texture_fc(feat))
        dct_block_reorder_dec = torch.empty_like(dct_block_reorder)
        for i in range(self.total_frequency_component):
            temp_tex = self.dec_texture_decoder(texture_code[:,i*block_features : (i+1)*block_features])
            dct_block_reorder_dec[i,:,:,:,:] = temp_tex
        # pred_tex = self.dec_texture_decoder(texture_code)
        pred_tex = self.dct_inverse_transform(dct_block_reorder_dec, bs, ch, h, w)
        pred_mesh = self.dec_mesh_fc(z_code)
        # return texture, mesh
    
        pred_mesh = pred_mesh.view((b, n, 3))
        if cams is not None:
            pred_tex = self.cc(pred_tex, cams)
        return pred_tex, pred_mesh, kl

    def get_mesh_branch_params(self):
        p = self.mesh_fc.parameters() + self.dec_mesh_fc.parameters()
        return p

    def get_tex_branch_params(self):
        p = self.texture_encoder.parameters() + self.texture_fc.parameters() + self.fc.parameters() + self.dec_texture_decoder.parameters() + self.dec_view_fc.parameters() + self.dec_z_fc.parameters() + self.dec_texture_fc.parameters()
        return p

    def get_model_params(self):
        params = []
        params += list(self.texture_encoder.parameters())
        params += list(self.texture_fc.parameters())
        params += list(self.fc.parameters())
        params += list(self.dec_texture_decoder.parameters())
        params += list(self.dec_view_fc.parameters())
        params += list(self.dec_z_fc.parameters())
        params += list(self.dec_texture_fc.parameters())
        params += list(self.mesh_fc.parameters())
        params += list(self.dec_mesh_fc.parameters())
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


class TextureDecoderLayerRed(nn.Module):
    def __init__(self, tex_size, z_dim, res=False, non=False, bilinear=False):
        super(TextureDecoderLayerRed, self).__init__()
        base = 2 if tex_size == 512 else 4
        self.z_dim = z_dim

        self.upsample = nn.Sequential(
            # ConvUpsample(
            #     z_dim, z_dim, 64, base, res=res, use_bilinear=bilinear, non=non
            # ),
            ConvUpsample(
                64, 64, 32, base, res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                32, 32, 16, base * (2**2), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                16,
                16,
                3,
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


class TextureEncoderLayerRed(nn.Module):
    def __init__(self, res=False):
        super(TextureEncoderLayerRed, self).__init__()
        self.downsample = nn.Sequential(
            ConvDownsample(3, 16, 16, res=res),
            ConvDownsample(16, 32, 32, res=res),
            ConvDownsample(32, 64, 64, res=res),
            # ConvDownsample(64, 128, 128, res=res),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.downsample(x)
        out = feat.view((b, -1))
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