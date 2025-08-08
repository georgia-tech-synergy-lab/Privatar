# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image

import argparse
import os
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from dataset import Dataset
from models import DeepAppearanceVAEDirectSplit
from utils import gammaCorrect, Renderer

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

def evaluate_single_image(args, camera_config, test_segment, image_path):
    """
    Evaluate a single image from the dataset.

    Args:
        args: Command line arguments
        camera_config: Camera configuration
        test_segment: Test segment configuration
        image_path: Path to the specific image to evaluate

    Returns:
        Tuple of (losses, output) for the single image
    """
    os.makedirs(args.result_path, exist_ok=True)
    device = torch.device("cuda", 0)

    # Extract dataset info from image path
    # Path format: /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png
    import re
    path_parts = image_path.split('/')

    # Find the subject directory (format: m--YYYYMMDD--HHMMSS--SUBJECT_ID--GHS)
    subject_dir = None
    for part in path_parts:
        if re.match(r'm--\d{8}--\d{4}--\d+--GHS', part):
            subject_dir = part
            break

    if subject_dir:
        # Extract the base dataset directory
        subject_idx = path_parts.index(subject_dir)
        base_data_dir = '/'.join(path_parts[:subject_idx + 1])

        # Update args to use the correct data directory for this image
        print(f"Detected subject: {subject_dir}")
        print(f"Using data directory: {base_data_dir}")

        # Override the data directory and related paths
        args.data_dir = base_data_dir
        args.krt_dir = os.path.join(base_data_dir, "KRT")
        args.framelist_test = os.path.join(base_data_dir, "frame_list.txt")

        # Extract camera ID and frame info from path
        # Path: .../images/E001_Neutral_Eyes_Open/400009/000102.png
        # where 400009 is camera ID and 000102 is frame
        if 'images' in path_parts:
            images_idx = path_parts.index('images')
            if images_idx + 3 < len(path_parts):
                expression = path_parts[images_idx + 1]  # E001_Neutral_Eyes_Open
                camera_id = path_parts[images_idx + 2]   # 400009
                frame_name = path_parts[images_idx + 3].split('.')[0]  # 000102

                print(f"Expression: {expression}, Camera: {camera_id}, Frame: {frame_name}")

    # Create dataset to get preprocessing parameters
    try:
        dataset_test = Dataset(
            args.data_dir,
            args.krt_dir,
            args.framelist_test,
            args.tex_size,
            camset=None if camera_config is None else camera_config["test"],
            valid_prefix=test_segment,
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Trying with default dataset configuration...")
        dataset_test = Dataset(
            args.data_dir,
            args.krt_dir,
            args.framelist_test,
            args.tex_size,
            camset=None,
            valid_prefix=None,
        )

    print(f"Evaluating single image: {image_path}")
    print(f"Dataset size: {len(dataset_test)}")

    # Load and initialize model
    n_cams = len(set(camera_config["train"]).union(set(dataset_test.cameras))) if camera_config else len(dataset_test.cameras)
    if args.arch == "base":
        model = DeepAppearanceVAEDirectSplit(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams,
            result_path=args.result_path, save_latent_code=args.save_latent_code, gaussian_noise_covariance_path=args.gaussian_noise_covariance_path
        ).to(device)
    else:
        raise NotImplementedError

    # Load model weights
    print("loading model from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cuda:0")
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    renderer = Renderer()
    mse = nn.MSELoss()

    # Get preprocessing parameters from dataset
    texmean = cv2.resize(dataset_test.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_test.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_test.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = torch.tensor(dataset_test.texstd, dtype=torch.float32).to(device)
    vertmean = (
        torch.tensor(dataset_test.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = torch.tensor(dataset_test.vertstd, dtype=torch.float32).to(device)
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    # Try to find the specific data item that corresponds to the image path
    target_data = None
    print("Searching for matching data item in dataset...")
    print(f"len(dataset_test): {len(dataset_test)}")

    # Extract sentnum, cam, and frame from the image path
    # Path format: .../images/E001_Neutral_Eyes_Open/400009/000102.png
    extracted_info = None
    if 'images' in path_parts:
        images_idx = path_parts.index('images')
        if images_idx + 3 < len(path_parts):
            sentnum = path_parts[images_idx + 1]  # E001_Neutral_Eyes_Open
            cam = path_parts[images_idx + 2]      # 400009
            frame = path_parts[images_idx + 3].split('.')[0]  # 000102
            extracted_info = (sentnum, frame, cam)
            print(f"Extracted from path - sentnum: {sentnum}, frame: {frame}, cam: {cam}")

    if extracted_info:
        # Look for exact match in dataset framelist
        print(f"Looking for {extracted_info} in dataset framelist...")
        for i in range(len(dataset_test)):
            framelist_item = dataset_test.framelist[i]
            print(f"Checking framelist[{i}]: {framelist_item}")
            if framelist_item == extracted_info:
                target_data = dataset_test[i]
                print(f"Found exact match at index {i}")
                break
            # Only check first few items to avoid spam
            if i > 5:
                print("... (showing only first 6 items)")
                break

    if target_data is None:
        print(f"Could not find exact match in dataset, using first available item as template...")
        target_data = dataset_test[0]
        using_template_data = True  # Flag to track template usage

        # Load the specific image manually
        if os.path.exists(image_path):
            print(f"Loading image from: {image_path}")
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img_writeout = cv2.resize(img, (args.resolution[1], args.resolution[0]))
                img = cv2.resize(img, (args.resolution[0], args.resolution[1]))
                target_data['photo'] = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                print(f"Successfully loaded image: {img.shape}")
            else:
                print(f"Failed to load image from {image_path}")
                return None, None
        else:
            print(f"Image path does not exist: {image_path}")
            return None, None

        # Load corresponding mesh and UV data for the target image
        if extracted_info:
            sentnum, frame, cam = extracted_info
            print(f"Loading corresponding geometric data for {sentnum}/{frame}/{cam}")
            
            # Load facial mesh (.obj file)
            mesh_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}.obj")
            print(f"Loading mesh from: {mesh_path}")
            if os.path.exists(mesh_path):
                from dataset import load_obj
                obj = load_obj(mesh_path)
                target_data['uvs'] = obj["uvs"]
                target_data['vert_ids'] = obj["vert_ids"] 
                target_data['uv_ids'] = obj["uv_ids"]
                print(f"Successfully loaded mesh with {len(obj['verts'])} vertices")
            else:
                print(f"Warning: Mesh file not found: {mesh_path}")

            # Load vertex positions (.bin file)
            verts_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}.bin")
            print(f"Loading vertex positions from: {verts_path}")
            if os.path.exists(verts_path):
                verts = np.fromfile(verts_path, dtype=np.float32)
                verts -= dataset_test.vertmean
                verts /= dataset_test.vertstd
                target_data['aligned_verts'] = verts.reshape((-1, 3)).astype(np.float32)
                print(f"Successfully loaded vertex positions: {verts.shape}")
            else:
                print(f"Warning: Vertex file not found: {verts_path}")

            # Load unwrapped UV texture
            uv_path = os.path.join(args.data_dir, "unwrapped_uv_1024", sentnum, cam, f"{frame}.png")
            print(f"Loading UV texture from: {uv_path}")
            if os.path.exists(uv_path):
                tex = np.asarray(Image.open(uv_path), dtype=np.float32)[::-1, ...]
                mask = tex == 0
                tex -= dataset_test.texmean
                tex /= dataset_test.texstd
                tex[mask] = 0.0
                tex = cv2.resize(tex, (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                mask = 1.0 - cv2.resize(mask.astype(np.float32), (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                target_data['tex'] = tex
                target_data['mask'] = mask
                print(f"Successfully loaded UV texture: {tex.shape}")
            else:
                print(f"Warning: UV texture file not found: {uv_path}")

            # Load average texture for the frame
            avg_path = os.path.join(args.data_dir, "unwrapped_uv_1024", sentnum, "average", f"{frame}.png")
            print(f"Loading average texture from: {avg_path}")
            if os.path.exists(avg_path):
                avgtex = np.asarray(Image.open(avg_path), dtype=np.float32)[::-1, ...]
                mask_avg = avgtex == 0
                avgtex -= dataset_test.texmean
                avgtex /= dataset_test.texstd
                avgtex[mask_avg] = 0.0
                avgtex = cv2.resize(avgtex, (args.tex_size, args.tex_size)).transpose((2, 0, 1))
                target_data['avg_tex'] = avgtex
                print(f"Successfully loaded average texture: {avgtex.shape}")
            else:
                print(f"Warning: Average texture file not found: {avg_path}")

            # Load transform data
            transform_path = os.path.join(args.data_dir, "tracked_mesh", sentnum, f"{frame}_transform.txt")
            print(f"Loading transform from: {transform_path}")
            if os.path.exists(transform_path):
                transf = np.genfromtxt(transform_path)
                target_data['transf'] = transf.astype(np.float32)
                
                # Recalculate view direction and camera matrix
                R_f = transf[:3, :3]
                t_f = transf[:3, 3]
                cam_key = None
                for key in dataset_test.krt.keys():
                    if key == cam:
                        cam_key = key
                        break
                
                if cam_key is not None:
                    campos = dataset_test.campos[cam_key]
                    campos = np.dot(R_f.T, campos - t_f).astype(np.float32)
                    view = campos / np.linalg.norm(campos)
                    target_data['view'] = view
                    
                    # Update camera matrix
                    extrin, intrin = dataset_test.krt[cam_key]["extrin"], dataset_test.krt[cam_key]["intrin"]
                    R_C = extrin[:3, :3]
                    t_C = extrin[:3, 3]
                    camrot = np.dot(R_C, R_f).astype(np.float32)
                    camt = np.dot(R_C, t_f) + t_C
                    camt = camt.astype(np.float32)
                    M = intrin @ np.hstack((camrot, camt[None].T))
                    target_data['M'] = M.astype(np.float32)
                    
                    print(f"Successfully loaded transform and updated camera data")
                else:
                    print(f"Warning: Camera {cam} not found in dataset KRT data")
            else:
                print(f"Warning: Transform file not found: {transform_path}")

    else:
        using_template_data = False

    # Convert single item to batch format
    data = {}
    for key, value in target_data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.unsqueeze(0)  # Add batch dimension
        elif isinstance(value, np.ndarray):
            data[key] = torch.tensor(value).unsqueeze(0)
        elif isinstance(value, (int, float)):
            # Handle scalar values like camera IDs
            data[key] = torch.tensor([value])
        else:
            data[key] = value
    
    # Add template usage flag
    data['_using_template'] = using_template_data

    def run_net_single(data):
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

        output = {}

        if args.arch == "warp":
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = model(
                avg_tex, verts, view, cams=cams
            )
            output["unwarped_tex"] = unwarped_tex
            output["warp_field"] = warp_field
        else:
            pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.0
        gt_tex = (gt_tex * texstd + texmean) / 255.0

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution
            )
            pred_screen, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution
            )
            
            # Fix tensor shape mismatch: permute photo from (batch, channels, height, width) to (batch, width, height, channels)
            photo_permuted = photo.permute(0, 3, 2, 1)
            
            screen_loss = (
                torch.mean((pred_screen - photo_permuted) ** 2 * screen_mask)
                * (255**2)
                / (texstd**2)
            )
        else:
            screen_loss, pred_screen = torch.zeros([]), None

        total_loss = 0
        if args.lambda_verts > 0:
            total_loss = total_loss + args.lambda_verts * vert_loss
        if args.lambda_tex > 0:
            total_loss = total_loss + args.lambda_tex * tex_loss
        if args.lambda_screen > 0:
            total_loss = total_loss + args.lambda_screen * screen_loss
        if args.lambda_kl > 0:
            total_loss = total_loss + args.lambda_kl * kl

        losses = {
            "total_loss": total_loss,
            "vert_loss": vert_loss,
            "screen_loss": screen_loss,
            "tex_loss": tex_loss,
            "denorm_tex_loss": tex_loss * (texstd**2),
            "kl": kl,
        }

        output["pred_screen"] = pred_screen
        output["pred_verts"] = pred_verts
        output["pred_tex"] = pred_tex

        return losses, output

    def save_img_single(data, output, tag="single_eval"):
        # Fix tensor shapes for proper image saving
        gt_screen = data["photo"] * 255
        pred_tex = torch.clamp(output["pred_tex"] * 255, 0, 255)
        
        # Check if we're using template data (when exact match wasn't found)
        using_template = data.get('_using_template', False)
        
        if output["pred_screen"] is not None:
            pred_screen = torch.clamp(output["pred_screen"] * 255, 0, 255)
            # Convert pred_screen from (batch, width, height, channels) to (batch, height, width, channels)
            pred_screen = pred_screen.permute(0, 2, 1, 3)
            save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8)
            save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
            if len(save_pred_image.shape) == 4:
                for _batch_id in range(save_pred_image.shape[0]):
                    # Create PIL Image from array
                    img = Image.fromarray(save_pred_image[_batch_id])
                    # Apply transformations: rotate clockwise 90 degrees and flip horizontally
                    img = img.rotate(-90, expand=True)  # -90 for clockwise rotation
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
                    # Save the transformed image
                    img.save(os.path.join(args.result_path, f"pred_rendered_{tag}_{_batch_id}.png"))

        print(f"Original gt_screen shape: {gt_screen.shape}")
        
        # Check if gt_screen needs permutation - it should be (batch, height, width, channels)
        if len(gt_screen.shape) == 4 and gt_screen.shape[-1] != 3:
            # If last dimension is not 3 (channels), data might be in (batch, channels, height, width) format
            gt_screen = gt_screen.permute(0, 2, 3, 1)  # Convert to (batch, height, width, channels)
            print(f"Applied permute (0,2,3,1) - new shape: {gt_screen.shape}")
        elif len(gt_screen.shape) == 3:
            # Single image without batch dimension
            if gt_screen.shape[-1] != 3:
                gt_screen = gt_screen.permute(1, 2, 0)  # Convert to (height, width, channels)
                print(f"Applied permute (1,2,0) for single image - new shape: {gt_screen.shape}")
        
        save_gt_image = gt_screen.detach().cpu().numpy().astype(np.uint8)
        # Save predicted texture - convert from (batch, channels, height, width) to (batch, height, width, channels)
        save_pred_tex_image = pred_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
        if len(save_pred_tex_image.shape) == 4:
            for _batch_id in range(save_pred_tex_image.shape[0]):
                save_pred_tex_image[_batch_id] = (255 * gammaCorrect(save_pred_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                Image.fromarray(save_pred_tex_image[_batch_id]).save(os.path.join(args.result_path, f"pred_{tag}_{_batch_id}.png"))

        # Only save ground truth texture if we have an exact match from dataset
        if not using_template and "tex" in data:
            gt_tex = data["tex"].cuda() * texstd + texmean
            save_gt_tex_image = gt_tex.detach().permute((0,2,3,1)).cpu().numpy().astype(np.uint8)
            if len(save_gt_tex_image.shape) == 4:
                for _batch_id in range(save_gt_tex_image.shape[0]):
                    save_gt_tex_image[_batch_id] = (255 * gammaCorrect(save_gt_tex_image[_batch_id] / 255.0)).astype(np.uint8)
                    Image.fromarray(save_gt_tex_image[_batch_id]).save(os.path.join(args.result_path, f"gt_{tag}_{_batch_id}.png"))
        else:
            print(f"Note: Skipping ground truth texture save since using template data (not exact match from dataset)")

        print(f"Saved images:")
        print(f"  - pred_rendered_{tag}_0.png: Neural face rendering result")  
        print(f"  - pred_texture_{tag}_0.png: Generated texture map")

        # Handle warp visualization if using warp architecture
        if args.arch == "warp" and "warp_field" in output:
            warp = output["warp_field"]
            grid_img = (
                torch.tensor(
                    np.array(
                        Image.open("grid.PNG").resize((args.tex_size, args.tex_size)),
                        dtype=np.float32,
                    )[None, ...]
                )
                .permute(0, 3, 1, 2)
                .to(warp.device)
            )
            grid_img = F.grid_sample(grid_img, warp[-1:])
            Image.fromarray(
                grid_img[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
            ).save(os.path.join(args.result_path, "warp_grid_%s.png" % tag))
            print(f"  - warp_grid_{tag}.png: Warp field visualization")

    # Run inference on single image
    print("Running inference...")
    with torch.no_grad():
        losses, output = run_net_single(data)

        print(f"\nSingle image evaluation results:")
        print(f"Total loss: {losses['total_loss'].item():.6f}")
        print(f"Vertex loss: {losses['vert_loss'].item():.6f}")
        print(f"Texture loss: {losses['denorm_tex_loss'].item():.6f}")
        print(f"Screen loss: {losses['screen_loss'].item():.6f}")
        print(f"KL loss: {losses['kl'].item():.6f}")

        # Save results
        if args.save_img:
            os.makedirs(args.result_path, exist_ok=True)
            save_img_single(data, output, image_path.split('/')[-3])
            print(f"Results saved to: {args.result_path}")

    return losses, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Validation batch size"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="base",
        help="Model architecture - base|warp|res|non|bilinear",
    )
    parser.add_argument(
        "--nlatent", type=int, default=256, help="Latent code dimension - 128|256"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--resolution",
        default=[2048, 1334],
        nargs=2,
        type=int,
        help="Rendering resolution",
    )
    parser.add_argument("--tex_size", type=int, default=1024, help="Texture resolution")
    parser.add_argument(
        "--mesh_inp_size", type=int, default=21918, help="Input mesh dimension"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS",
        help="Directory to dataset root",
    )
    parser.add_argument(
        "--krt_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT",
        help="Directory to KRT file",
    )
    parser.add_argument(
        "--loss_weight_mask",
        type=str,
        default="./loss_weight_mask.png",
        help="Mask for weighted loss of face",
    )
    parser.add_argument(
        "--framelist_test",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for testing",
    )
    parser.add_argument(
        "--test_segment_config",
        type=str,
        default="/mnt/captures/ecwuu/test_segment.json",
        help="Directory of expression segments for testing (exclude from training)",
    )
    parser.add_argument(
        "--lambda_verts", type=float, default=1, help="Multiplier of vertex loss"
    )
    parser.add_argument(
        "--lambda_screen", type=float, default=0, help="Multiplier of screen loss"
    )
    parser.add_argument(
        "--lambda_tex", type=float, default=1, help="Multiplier of texture loss"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=1e-2, help="Multiplier of KL divergence"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200000,
        help="Maximum number of training iterations, overrides epoch",
    )
    parser.add_argument(
        "--log_every", type=int, default=1000, help="Interval of printing training loss"
    )
    parser.add_argument(
        "--val_every", type=int, default=5000, help="Interval of validating on test set"
    )
    parser.add_argument(
        "--val_num", type=int, default=500, help="Number of iterations for validation"
    )
    parser.add_argument(
        "--n_worker", type=int, default=8, help="Number of workers loading dataset"
    )
    parser.add_argument(
        "--pass_thres",
        type=int,
        default=50,
        help="If loss is x times higher than the previous batch, discard this batch",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./runs/experiment",
        help="Directory to output files",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="PiCA Partition - Training Task",
    )
    parser.add_argument(
        "--author_name",
        type=str,
        default=None,
        help="Jianming Tong",
    )
    parser.add_argument(
        "--save_latent_code",
        action='store_true',
        default=False,
        help="save latent code to the result folder ./result_path/latent_code"
    )
    parser.add_argument(
        "--save_img", action='store_true', default=False, help="Control knob to enable image save"
    )
    parser.add_argument(
        "--gaussian_noise_covariance_path", type=str, default=None, help="The path of the noise covariance"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="Path to a single image to evaluate. If provided, only this image will be processed instead of the full dataset."
    )

    experiment_args = parser.parse_args()
    print(experiment_args)

    # load camera config
    subject_id = experiment_args.data_dir.split("--")[-2]
    camera_config_path = f"camera_configs/camera-split-config_{subject_id}.json"
    if os.path.exists(camera_config_path):
        print(f"camera config file for {subject_id} exists, loading...")
        f = open(camera_config_path, "r")
        camera_config = json.load(f)
        f.close()
    else:
        print(f"camera config file for {subject_id} NOT exists, generating...")
        # generate camera config based on downloaded data if not existed
        segments = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/*")]
        assert len(segments) > 0
        # select a segment to check available camera ids
        camera_ids = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/{segments[0]}/*")]
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

    camera_set = camera_config["full"]

    if experiment_args.test_segment_config is not None:
        f = open(experiment_args.test_segment_config, "r")
        test_segment_config = json.load(f)
        f.close()
        test_segment = test_segment_config["segment"]
    else:
        test_segment = None

    assert experiment_args.image_path is not None
    evaluate_single_image(experiment_args, camera_set, test_segment, experiment_args.image_path)
