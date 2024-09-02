# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os 
import math
from os import wait
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct

freq_per_group = 2
l2_diff_list = np.array([193010000, 41593896, 26539686, 20498900, 37043116, 25669930, 21078640, 18212020, 23593842, 20278054, 17874242, 16085497, 18370926, 16896158, 15619506, 14439489])

class DeepAppearanceVAEBDCT(nn.Module):
    def __init__(
        self,
        tex_size=1024,
        mesh_inp_size=21918,
        mode="vae",
        n_latent=128,
        n_cams=38,
        n_block=4,
        res=False,
        non=False,
        bilinear=False,
        ## Added for horizontal partitioning
        num_freq_comp_outsourced=4, # must be multiple of 2
        result_path=False,
        save_latent_code=False
    ):
        super(DeepAppearanceVAEBDCT, self).__init__()
        n_blocks = 4
        self.block_size = n_blocks
        self.total_frequency_component = self.block_size * self.block_size

        z_dim = n_latent if mode == "vae" else n_latent * 2
        self.mode = mode
        outsourced_channel_ratio, local_channel_ratio, self.local_freq_list, self.outsourced_freq_list = self.generate_freq_group_index(l2_diff_list, num_freq_comp_outsourced)
        

        self.enc = DeepApperanceEncoderLayerRed(
            tex_size, mesh_inp_size, n_latent=z_dim, res=res, channel_ratio=len(self.local_freq_list)
        )
        self.enc_outsourced = DeepApperanceEncoderLayerRed(
            tex_size, mesh_inp_size, n_latent=z_dim, res=res, channel_ratio=len(self.outsourced_freq_list)
        )

        self.dec = DeepAppearanceDecoderLayerRed(
            tex_size, mesh_inp_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear, local_channel_ratio=local_channel_ratio, outsourced_channel_ratio=outsourced_channel_ratio
        )

        self.cc = ColorCorrection(n_cams)
        
        self.save_latent_code = save_latent_code
        self.latent_code_path = f"{result_path}/latent_code"
        if self.save_latent_code:
            if not os.path.exists(self.latent_code_path):
                os.makedirs(self.latent_code_path)
        self.iter = 0

    def generate_freq_group_index(self, l2_diff_list, num_freq_comp_outsourced=4):
        min_val = np.min(l2_diff_list)
        new_val_list = l2_diff_list / min_val

        normalize_list = new_val_list / np.sum(new_val_list) * 16.5 # 16 cateogries in total, and use 16.5 to make sure the accumulation of final frequency decomposition equals 16.
        
        sorted_index_list = np.argsort(normalize_list)
        
        total_num_freq_group = int(self.total_frequency_component / freq_per_group)
        form_group = np.zeros(total_num_freq_group)
        indice_group = []
        for i in range(total_num_freq_group):
            for index in range(freq_per_group):
                freq_index = i*freq_per_group + index
                form_group[i] = form_group[i] + normalize_list[sorted_index_list[freq_index]]
        
        np.sort(form_group)
        np.sum(np.round(form_group))

        assert(np.sum(np.round(form_group)) == 16)
        index_freq_div = np.round(form_group)

        assert(np.sum(np.round(form_group)) == 16)
        sorted_index_array = np.array(sorted_index_list).reshape(total_num_freq_group, freq_per_group)
        outsourced_channel_ratio = 0
        outsourced_freq_list = []
        for index, freq_pair  in enumerate(sorted_index_array):
            outsourced_freq_list.append(freq_pair[0])
            outsourced_freq_list.append(freq_pair[1])
            outsourced_channel_ratio += index_freq_div[index] 
            if len(outsourced_freq_list) == num_freq_comp_outsourced:
                break
        local_freq_list = [i for i in range(self.total_frequency_component) if i not in outsourced_freq_list ]
        local_channel_ratio = np.sum(np.round(form_group)) - outsourced_channel_ratio
        return int(outsourced_channel_ratio), int(local_channel_ratio), local_freq_list, outsourced_freq_list # normalize_list, sorted_index_array

    def img_reorder_pure_bdct(self, x, bs, ch, h, w):
        # x = (x + 1) / 2 * 255
        # x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
        # x -= 128
        x = x.view(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(self.block_size, self.block_size), dilation=1, padding=0, stride=(self.block_size, self.block_size))
        x = x.transpose(1, 2)
        x = x.view(bs, ch, -1, self.block_size, self.block_size)
        return x

    ## Image reordering and testing
    def img_inverse_reroder_pure_bdct(self, coverted_img, bs, ch, h, w):
        x = coverted_img.view(bs* ch, -1, self.total_frequency_component)
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=(h, w), kernel_size=(self.block_size, self.block_size), stride=(self.block_size, self.block_size))
        # x += 128
        x = x.view(bs, ch, h, w)
        # x = dct.to_rgb(x)#.squeeze(0)
        # x = (x / 255.0) * 2 - 1
        return x
    
    ## Image frequency cosine transform
    def dct_transform(self, x, bs, ch, h, w):
        rerodered_img = self.img_reorder_pure_bdct(x, bs, ch, h, w)
        block_num = h // self.block_size
        dct_block = dct.block_dct(rerodered_img) #BDCT
        # dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, self.total_frequency_component).permute(4, 0, 1, 2, 3) # into (bs, ch, block_num, block_num, 16)
        dct_block_reorder = dct_block.view(bs, ch, block_num, block_num, self.total_frequency_component).permute(0, 4, 1, 2, 3)
        return dct_block_reorder

    def dct_inverse_transform(self, dct_block_reorder,bs, ch, h, w):
        block_num = h // self.block_size
        idct_dct_block_reorder = dct_block_reorder.permute(0, 2, 3, 4, 1).view(bs, ch, block_num*block_num, self.block_size, self.block_size)
        inverse_dct_block = dct.block_idct(idct_dct_block_reorder) #inverse BDCT
        inverse_transformed_img = self.img_inverse_reroder_pure_bdct(inverse_dct_block, bs, ch, h, w)
        return inverse_transformed_img
    
    def forward(self, avgtex, mesh, view, cams=None):
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))

        bs, ch, h, w = avgtex.shape
        dct_block_reorder = self.dct_transform(avgtex, bs, ch, h, w)
        
        dct_block_reorder_local = dct_block_reorder[:,self.local_freq_list,:,:,:]
        dct_block_reorder_outsource = dct_block_reorder[:,self.outsourced_freq_list,:,:,:]

        block_num = h // self.block_size
        dct_block_reorder_local = dct_block_reorder_local.reshape(bs, ch*len(self.local_freq_list), block_num, block_num) # into (bs, ch, block_num, block_num, 16)
        dct_block_reorder_outsource = dct_block_reorder_outsource.reshape(bs, ch*len(self.outsourced_freq_list), block_num, block_num) # into (bs, ch, block_num, block_num, 16)
        # print(f"dct_block_reorder_local.shape={dct_block_reorder_local.shape}")
        # print(f"dct_block_reorder_outsource.shape={dct_block_reorder_outsource.shape}")
        mean, logstd = self.enc(dct_block_reorder_local, mesh)
        mean = mean * 0.1
        logstd = logstd * 0.01
        if self.mode == "vae":
            kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
            std = torch.exp(logstd)
            eps = torch.randn_like(mean)
            z = mean + std * eps
            if self.save_latent_code:
                torch.save(z, f"{self.latent_code_path}/z_{self.iter}.pth")
        else:
            z = torch.cat((mean, logstd), -1)
            kl = torch.tensor(0).to(z.device)
        
        mean_outsource, logstd_outsource = self.enc_outsourced(dct_block_reorder_outsource, mesh)
        mean_outsource = mean_outsource * 0.1
        logstd_outsource = logstd_outsource * 0.01
        if self.mode == "vae":
            kl_outsource = 0.5 * torch.mean(torch.exp(2 * logstd_outsource) + mean_outsource**2 - 1.0 - 2 * logstd_outsource)
            std_outsource = torch.exp(logstd_outsource)
            eps_outsource = torch.randn_like(mean_outsource)
            z_outsource = mean_outsource + std_outsource * eps_outsource
            if self.save_latent_code:
                torch.save(z_outsource, f"{self.latent_code_path}/z_outsource_{self.iter}.pth")
                self.iter = self.iter + 1
        else:
            z_outsource = torch.cat((mean_outsource, logstd_outsource), -1)
            kl_outsource = torch.tensor(0).to(z.device)

        kl_merge = len(self.local_freq_list)/self.total_frequency_component*kl + len(self.outsourced_freq_list)/self.total_frequency_component*kl_outsource

        pred_tex, pred_mesh = self.dec(z, z_outsource, view)
        pred_mesh = pred_mesh.view((b, n, 3))
        if cams is not None:
            pred_tex = self.cc(pred_tex, cams)

        return pred_tex, pred_mesh, kl_merge

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


class DeepAppearanceDecoderLayerRed(nn.Module):
    def __init__(
        self, tex_size, mesh_size, z_dim=128, res=False, non=False, bilinear=False, local_channel_ratio=12, outsourced_channel_ratio=4
    ):
        super(DeepAppearanceDecoderLayerRed, self).__init__()
        nhidden = z_dim * 4 * 4 if tex_size == 1024 else z_dim * 2 * 2
        self.texture_decoder_local = TextureDecoderLayerRed(
            tex_size, z_dim, res=res, non=non, bilinear=bilinear, channel_ratio=local_channel_ratio
        )
        self.texture_decoder_outsource = TextureDecoderLayerRed(
            tex_size, z_dim, res=res, non=non, bilinear=bilinear, channel_ratio=outsourced_channel_ratio
        )
        self.z_fc_outsource = LinearWN(z_dim, 256)
        self.texture_fc_outsource = LinearWN(256 + 8, nhidden)

        base = 2 if z_dim == 512 else 4
        self.texture_decoder_merge = ConvUpsample(16,16,3,base * (2**6),no_activ=True,res=res,use_bilinear=bilinear,non=non,)

        self.view_fc = LinearWN(3, 8)
        self.z_fc = LinearWN(z_dim, 256)
        self.mesh_fc = LinearWN(256, mesh_size)
        self.texture_fc = LinearWN(256 + 8, nhidden)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.mesh_fc, 1.0)
        glorot(self.texture_decoder_local.upsample[-1].conv2, 1.0)
        glorot(self.texture_decoder_outsource.upsample[-1].conv2, 1.0)

    def forward(self, z_local, z_outsource, v):
        view_code = self.relu(self.view_fc(v))
        z_code = self.relu(self.z_fc(z_local))
        mesh = self.mesh_fc(z_code)

        feat = torch.cat((view_code, z_code), 1)
        texture_code = self.relu(self.texture_fc(feat))
        texture_local = self.texture_decoder_local(texture_code)

        z_code_outsource = self.relu(self.z_fc_outsource(z_outsource))
        feat_outsource = torch.cat((view_code, z_code_outsource), 1)
        texture_code_outsource = self.relu(self.texture_fc_outsource(feat_outsource))
        texture_outsource = self.texture_decoder_outsource(texture_code_outsource)

        texture_merge = torch.cat((texture_local, texture_outsource), dim=1)
        texture = self.texture_decoder_merge(texture_merge)

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


class DeepApperanceEncoderLayerRed(nn.Module):
    def __init__(self, inp_size=1024, mesh_inp_size=21918, n_latent=128, res=False, channel_ratio=12):
        super(DeepApperanceEncoderLayerRed, self).__init__()
        self.n_latent = n_latent
        ntexture_feat = 2048 if inp_size == 1024 else 512
        self.texture_encoder = TextureEncoderLayerRed(res=res, channel_ratio=channel_ratio)
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


class TextureDecoderLayerRed(nn.Module):
    def __init__(self, tex_size, z_dim, res=False, non=False, bilinear=False, channel_ratio=2):
        super(TextureDecoderLayerRed, self).__init__()
        base = 2 if tex_size == 512 else 4
        self.z_dim = z_dim

        self.upsample = nn.Sequential(
            ConvUpsample(
                z_dim, z_dim, int(64/16*channel_ratio), base, res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                int(64/16*channel_ratio), int(64/16*channel_ratio), int(32/16*channel_ratio), base * (2**2), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                int(32/16*channel_ratio), int(32/16*channel_ratio), int(16/16*channel_ratio), base * (2**4), res=res, use_bilinear=bilinear, non=non
            ),
            # ConvUpsample(
            #     16,
            #     16,
            #     3,
            #     base * (2**6),
            #     no_activ=True,
            #     res=res,
            #     use_bilinear=bilinear,
            #     non=non,
            # ),
        )

    def forward(self, x):
        b, n = x.shape
        h = int(np.sqrt(n / self.z_dim))
        x = x.view((-1, self.z_dim, h, h))
        out = self.upsample(x)
        return out


class TextureEncoderLayerRed(nn.Module):
    def __init__(self, res=False, channel_ratio=4):
        super(TextureEncoderLayerRed, self).__init__()
        # print(f"int(48/16*channel_ratio)={int(48/16*channel_ratio)}")
        self.downsample = nn.Sequential(
            # ConvDownsample(3, 16, 16, res=res),
            ConvDownsample(int(48/16*channel_ratio), 48, 48, res=res),
            ConvDownsample(48, 64, 64, res=res),
            ConvDownsample(64, 128, 128, res=res),
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
