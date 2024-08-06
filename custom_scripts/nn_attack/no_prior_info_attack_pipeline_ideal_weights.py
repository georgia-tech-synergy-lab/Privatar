import os
import cv2
import json 
import glob
import torch
import numpy as np
import torch.nn as nn
from dataset import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, RandomSampler
from library import *

## Input Configurations
tex_size = 1024
val_batch_size = 1
n_worker = 1
path_prefix = "/home/jianming/work/multiface/"
data_dir = f"{path_prefix}dataset/m--20180227--0000--6795937--GHS"
krt_dir = f"{path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT"
framelist_train = f"{path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt"
subject_id = data_dir.split("--")[-2]
camera_config_path = f"camera_configs/camera-split-config_{subject_id}.json"
result_path = "/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/"

if os.path.exists(camera_config_path):
    print(f"camera config file for {subject_id} exists, loading...")
    f = open(camera_config_path, "r")
    camera_config = json.load(f)
    f.close()
else:
    print(f"camera config file for {subject_id} NOT exists, generating...")
    # generate camera config based on downloaded data if not existed
    segments = [os.path.basename(x) for x in glob.glob(f"{data_dir}/unwrapped_uv_1024/*")]
    assert len(segments) > 0
    # select a segment to check available camera ids
    camera_ids = [os.path.basename(x) for x in glob.glob(f"{data_dir}/unwrapped_uv_1024/{segments[0]}/*")]
    camera_ids.remove('average')
    camera_config = {
        "full": {
            "train": camera_ids,
            "test": camera_ids,
            "visual": camera_ids[:2]
        }
    }
    # save the config for future use
    os.makedirs("camera_configs", exist_ok=True)
    with open(camera_config_path, 'w') as f:
        json.dump(camera_config, f)

test_segment = ["EXP_ROM", "EXP_free_face"]

## Generate Data Pair
dataset_test = Dataset(
    data_dir,
    krt_dir,
    framelist_train,
    tex_size,
    camset=None if camera_config is None else camera_config["full"]["test"],
    exclude_prefix=test_segment,
)

texstd = dataset_test.texstd
texmean = cv2.resize(dataset_test.texmean, (tex_size, tex_size))
texmin = cv2.resize(dataset_test.texmin, (tex_size, tex_size))
texmax = cv2.resize(dataset_test.texmax, (tex_size, tex_size))
texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to("cuda:0")
vertstd = dataset_test.vertstd
vertmean = (
    torch.tensor(dataset_test.vertmean, dtype=torch.float32)
    .view((1, -1, 3))
    .to("cuda:0")
)

test_sampler = RandomSampler(dataset_test)

test_loader = DataLoader(
    dataset_test,
    val_batch_size,
    sampler=test_sampler,
    num_workers=n_worker,
)

renderer = Renderer()

mse = nn.MSELoss()
loss_weight_mask = cv2.flip(cv2.imread("/home/jianming/work/Privatar_prj/multiface_partition_bdct4x4/loss_weight_mask.png"), 0)
loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
loss_weight_mask = (
    torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")
)

loss_mask = loss_weight_mask.repeat(1, 1, 1, 1)

def gammaCorrect(img, dim=-1):
    if dim == -1:
        dim = len(img.shape) - 1 
    assert(img.shape[dim] == 3)
    gamma, black, color_scale = 2.0,  3.0 / 255.0, [1.4, 1.1, 1.6]

    if torch.is_tensor(img):
        scale = torch.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
        img = img * scale.to(img) / 1.1
        correct_img = torch.clamp((((1.0 / (1 - black)) * 0.95 * torch.clamp(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2,)
    else:
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        correct_img = np.clip((((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2, )
    
    return correct_img


def save_img(data, output, tag=""):
    gt_screen = data["photo"].to("cuda:0") * 255
    gt_tex = data["tex"].to("cuda:0") * texstd + texmean
    pred_tex = torch.clamp(output["pred_tex"] * 255, 0, 255)
    if output["pred_screen"] is not None:
        pred_screen = torch.clamp(output["pred_screen"] * 255, 0, 255)
        # apply gamma correction
        save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8) 
        save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
        if len(save_pred_image.shape) == 4:
            for _batch_id in range(save_pred_image.shape[0]):
                Image.fromarray(save_pred_image[_batch_id]).save(
                    os.path.join(result_path, f"pred_{tag}_{_batch_id}.png")
                )
    # apply gamma correction
    save_gt_image = gt_screen[-1].detach().cpu().numpy().astype(np.uint8)
    save_gt_image = (255 * gammaCorrect(save_gt_image / 255.0)).astype(np.uint8)
    Image.fromarray(save_gt_image).save(os.path.join(result_path, "gt_%s.png" % tag))
    # apply gamma correction
    save_gt_tex_image = gt_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
    save_gt_tex_image = (255 * gammaCorrect(save_gt_tex_image / 255.0)).astype(np.uint8)
    Image.fromarray(save_gt_tex_image).save(os.path.join(result_path, "gt_tex_%s.png" % tag))
    # apply gamma correction
    save_pred_tex_image = pred_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
    save_pred_tex_image = (255 * gammaCorrect(save_pred_tex_image / 255.0)).astype(np.uint8)
    Image.fromarray(save_pred_tex_image).save(os.path.join(result_path, "pred_tex_%s.png" % tag))

overall_screen_loss = np.zeros(len(test_loader))
overall_tex_loss = np.zeros(len(test_loader))

for i, data in enumerate(test_loader):
    for freq_base_id in range(1,16):
        M = data["M"].cuda()
        gt_tex = data["tex"].cuda()
        vert_ids = data["vert_ids"].cuda()
        uvs = data["uvs"].cuda()
        uv_ids = data["uv_ids"].cuda()
        avg_tex = data["avg_tex"].cuda()
        view = data["view"].cuda()
        transf = data["transf"].cuda()
        verts = data["aligned_verts"].cuda()
        photo = data["photo"].cuda()
        mask = data["mask"].cuda()
        cams = data["cam"].cuda()
        batch, channel, height, width = avg_tex.shape
        
        height_render, width_render = [2048, 1334]
        width_render = width_render - (width_render % 8)
        photo_short = torch.Tensor(photo)[:, :, :width_render, :]

        # verts   # input vertexs
        # photo   # input screen data 
        # avg_tex # input unwrapped texture

        # Perform frequency decomposition and reconstruction
        bs, ch, h, w = avg_tex.shape
        print(f"avg_tex.shape ={avg_tex.shape}")
        transforms.functional.to_pil_image(avg_tex.squeeze(0)).save(f'avg_tex.png')
        # pred_tex = test_img_dct_transform(avg_tex, bs, ch, h, w)#, 1)
        pred_tex = test_img_dct_transform_drop_low_freq_reorder(avg_tex, bs, ch, h, w, freq_base_id)
        transforms.functional.to_pil_image(pred_tex.squeeze(0)).save(f'input_outsourced_texture_freq_{freq_base_id}.png')

        pred_verts = verts
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)
        # From vertex to photo.

        pred_verts = pred_verts.to("cuda:0") * vertstd + vertmean.to("cuda:0")
        pred_tex = (pred_tex.to("cuda:0") * texstd + texmean.to("cuda:0")) / 255.0
        gt_tex = (gt_tex.to("cuda:0") * texstd + texmean.to("cuda:0")) / 255.0


        screen_mask, rast_out = renderer.render(
            M, pred_verts, vert_ids, uvs, uv_ids, loss_mask,  [height_render, width_render] #args.resolution
        )
        pred_screen, rast_out = renderer.render(M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, [height_render, width_render])

        output = {}
        output["pred_screen"] = pred_screen
        output["pred_verts"] = verts # 
        output["pred_tex"] = pred_tex
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)
        screen_loss = (
                torch.mean((pred_screen - photo_short) ** 2 * screen_mask[:, :, :width_render, :])
                * (255**2)
                / (texstd**2)
            )
        print(f"frequency_component = {freq_base_id}, vert_loss = {tex_loss}, screen_loss = {screen_loss}")
        save_img(data, output, f"reconstruct_{freq_base_id}")
        
        overall_screen_loss[i] = screen_loss
        overall_tex_loss[i] = tex_loss

    break
