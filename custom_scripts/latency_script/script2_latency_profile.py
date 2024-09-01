import sys
import os 
import json 
import time
import torch
import numpy as np
from dataset import Dataset 
from torch.utils.data import DataLoader, SequentialSampler
import cv2

model_arch = "hp_bdct4x4"
BDCT_threshold = [5]

if model_arch == "merge_conv_16chnl":
    from script1_models.multiface_partition_bdct4x4_merge_conv_end_16chnl import DeepAppearanceVAEHPLayerRed as overallModel

if model_arch == "merge_conv":
    from script1_models.multiface_partition_bdct4x4_merge_conv_end import DeepAppearanceVAEHPLayerRed as overallModel

if model_arch == "baseline":
    from script1_models.multiface_baseline import DeepAppearanceVAE as overallModel

if model_arch == "hp_bdct4x4":
    from script1_models.multiface_partition_bdct4x4 import DeepAppearanceVAE_Horizontal_Partition as overallModel
    BDCT_threshold = [0.1, 0.3, 0.35, 0.4, 0.42, 0.45, 0.5, 0.6, 0.7, 1.1, 1.2, 3.5, 5]


"""
    Data Set Loading
"""
bs = 1
nlatent = 256
tex_size = 1024
mesh_inp_size = 21918
val_batch_size = 1
n_worker = 1
path_prefix = "/home/jianming/work/multiface/"
data_dir = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS"
krt_dir = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT"
# framelist_train = f"/home/jianming/work/Privatar_prj/custom_scripts/bdct_reconstruction/single_expression_frame_list.txt"
framelist_train = "/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/selected_expression_frame_list.txt"
subject_id = data_dir.split("--")[-2]
camera_config_path = f"{path_prefix}camera_configs/camera-split-config_{subject_id}.json"
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

camera_config = camera_config["full"]

## Generate Data Pair
dataset_test = Dataset(
    data_dir,
    krt_dir,
    framelist_train,
    tex_size,
    camset=None if camera_config is None else camera_config["test"],
    exclude_prefix=test_segment,
)

print(len(dataset_test))
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

test_sampler = SequentialSampler(dataset_test)

test_loader = DataLoader(
    dataset_test,
    val_batch_size,
    sampler=test_sampler,
    num_workers=n_worker,
)

print(len(test_loader))
data_list = []
for data in test_loader:
    data_list.append(data)
    break

gt_tex = data_list[0]["tex"].cuda()
vert_ids = data_list[0]["vert_ids"].cuda()
uvs = data_list[0]["uvs"].cuda()
uv_ids = data_list[0]["uv_ids"].cuda()
transf = data_list[0]["transf"].cuda()
photo = data_list[0]["photo"].cuda()
mask = data_list[0]["mask"].cuda()

avg_tex = data_list[0]["avg_tex"].cuda()
verts = data_list[0]["aligned_verts"].cuda()
view = data_list[0]["view"].cuda()
cams = data_list[0]["cam"].cuda()

n_cams = len(set(camera_config["train"]).union(set(camera_config["test"])))

for frequency_threshold in BDCT_threshold:
    average_texture_path = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
    prefix_path_captured_latent_code = ""
    path_variance_matrix_tensor = ""
    save_latent_code_to_external_device = ""
    apply_gaussian_noise = False

    if model_arch == "merge_conv_16chnl":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=frequency_threshold, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise).to("cuda:0")

    if model_arch == "merge_conv":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=frequency_threshold, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise).to("cuda:0")

    if model_arch == "baseline":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams).to("cuda:0")

    if model_arch == "hp_bdct4x4":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=frequency_threshold, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise).to("cuda:0")
            
    def latency_profile(model, avg_tex, verts, view, cams):
        latencies = []
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(avg_tex, verts, view, cams)
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        # Convert to milliseconds
        latencies = np.array(latencies) * 1000  # convert to milliseconds

        # Compute statistics
        latency_stats = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
        
        return latency_stats

    latency_stats = latency_profile(model, avg_tex, verts, view, cams)
    print(latency_stats)

    def dec_latency_profile(model, avg_tex, verts, view, cams):
        latencies = []
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(avg_tex, verts, view, cams)
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        # Convert to milliseconds
        latencies = np.array(latencies) * 1000  # convert to milliseconds

        # Compute statistics
        latency_stats = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
        
        return latency_stats
    # general random input
# avgtex = torch.()

# , mesh, view, cams=None

# latency_profile