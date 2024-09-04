import torch
import torch.nn as nn
from dataset import Dataset 
import json 
import sys
import os 
import glob
from torchanalyse import profiler, System, Unit

model_arch = "multiface_partition_bdct4x4_ibdct_hp" 
# model_arch = "multiface_partition_bdct4x4_nn_decoder" 
# model_arch = "multiface_partition_bdct4x4_nn_decoder_hp" 
# model_arch = "multiface_partition_bdct4x4_ibdct"
# model_arch = "multiface_partition_bdct4x4_ibdct_hp"
sweep_parameter_list = [5]

tex_size = 1024
nlatent = 256
data_dir = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/"
krt_dir = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT"
framelist_test = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt"
subject_id = 6795937

test_segment = "./test_segment.json"

camera_config_path = "/home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json"


def calculate_conv2d_flops(layer, input_shape):
    """
    Calculate FLOPs for a Conv2D layer.
    
    Parameters:
    layer (nn.Conv2d): Convolutional layer.
    input_shape (tuple): Shape of input tensor as (batch_size, in_channels, height, width).

    Returns:
    int: Total FLOPs for the convolutional layer.
    """
    batch_size, in_channels, input_h, input_w = input_shape
    out_channels, kernel_h, kernel_w = layer.out_channels, layer.kernel_size[0], layer.kernel_size[1]
    stride_h, stride_w = layer.stride
    padding_h, padding_w = layer.padding

    # Output dimensions after the convolution operation
    output_h = (input_h - kernel_h + 2 * padding_h) // stride_h + 1
    output_w = (input_w - kernel_w + 2 * padding_w) // stride_w + 1

    # FLOPs calculation: batch_size * out_channels * output_height * output_width * (in_channels * kernel_h * kernel_w + bias if exists)
    flops_per_instance = out_channels * output_h * output_w * (in_channels * kernel_h * kernel_w)
    
    if layer.bias is not None:
        flops_per_instance += out_channels * output_h * output_w
    
    return batch_size * flops_per_instance

def calculate_fc_flops(layer, input_shape):
    """
    Calculate FLOPs for a fully connected (Linear) layer.

    Parameters:
    layer (nn.Linear): Fully connected layer.
    input_shape (tuple): Shape of input tensor as (batch_size, input_dim).

    Returns:
    int: Total FLOPs for the fully connected layer.
    """
    batch_size, input_dim = input_shape
    output_dim = layer.out_features

    # FLOPs: batch_size * input_dim * output_dim (for multiplication) + batch_size * output_dim (for addition of biases)
    flops_per_instance = input_dim * output_dim

    if layer.bias is not None:
        flops_per_instance += output_dim

    return batch_size * flops_per_instance

def calculate_deconv2d_flops(layer, input_shape):
    """
    Calculate FLOPs for a Deconvolution (Transposed Conv2D) layer.

    Parameters:
    layer (nn.ConvTranspose2d): Deconvolutional (Transposed Conv2d) layer.
    input_shape (tuple): Shape of input tensor as (batch_size, in_channels, height, width).

    Returns:
    int: Total FLOPs for the deconvolutional layer.
    """
    batch_size, in_channels, input_h, input_w = input_shape
    out_channels, kernel_h, kernel_w = layer.out_channels, layer.kernel_size[0], layer.kernel_size[1]
    stride_h, stride_w = layer.stride
    padding_h, padding_w = layer.padding

    # Output dimensions after the deconvolution operation
    output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h
    output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w

    # FLOPs calculation: batch_size * out_channels * output_height * output_width * (in_channels * kernel_h * kernel_w + bias if exists)
    flops_per_instance = out_channels * output_h * output_w * (in_channels * kernel_h * kernel_w)
    
    if layer.bias is not None:
        flops_per_instance += out_channels * output_h * output_w
    
    return batch_size * flops_per_instance

def calculate_total_flops(model, input_shape):
    """
    Calculate the total number of FLOPs for a model consisting of convolution, fully connected, and deconvolution layers.

    Parameters:
    model (nn.Module): The neural network model.
    input_shape (tuple): The shape of the input tensor as (batch_size, in_channels, height, width).

    Returns:
    int: Total number of FLOPs for the model.
    """
    total_flops = 0
    current_shape = input_shape

    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            flops = calculate_conv2d_flops(layer, current_shape)
            print(f"Conv2D FLOPs: {flops}")
            total_flops += flops
            # Update current shape after convolution
            current_shape = (current_shape[0], layer.out_channels,
                             (current_shape[2] - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1,
                             (current_shape[3] - layer.kernel_size[1] + 2 * layer.padding[1]) // layer.stride[1] + 1)

        elif isinstance(layer, nn.Linear):
            flops = calculate_fc_flops(layer, (current_shape[0], current_shape[1] * current_shape[2] * current_shape[3]))
            print(f"FC FLOPs: {flops}")
            total_flops += flops
            current_shape = (current_shape[0], layer.out_features)

        elif isinstance(layer, nn.ConvTranspose2d):
            flops = calculate_deconv2d_flops(layer, current_shape)
            print(f"Deconv FLOPs: {flops}")
            total_flops += flops
            # Update current shape after deconvolution
            current_shape = (current_shape[0], layer.out_channels,
                             (current_shape[2] - 1) * layer.stride[0] - 2 * layer.padding[0] + layer.kernel_size[0],
                             (current_shape[3] - 1) * layer.stride[1] - 2 * layer.padding[1] + layer.kernel_size[1])

    return total_flops


if model_arch == "baseline":
    from script1_models.multiface_baseline import DeepAppearanceVAE as overallModel

if model_arch == "multiface_partition_bdct4x4_nn_decoder":
    from script1_models.multiface_partition_bdct4x4_nn_decoder import DeepAppearanceVAEBDCT as overallModel

if model_arch == "multiface_partition_bdct4x4_nn_decoder_hp":
    from script1_models.multiface_partition_bdct4x4_nn_decoder_hp import DeepAppearanceVAEBDCT as overallModel
    sweep_parameter_list = [2, 4, 6, 8, 10, 12, 14]

if model_arch == "multiface_partition_bdct4x4_ibdct":
    from script1_models.multiface_partition_bdct4x4_ibdct import DeepAppearanceVAE_IBDCT as overallModel

if model_arch == "multiface_partition_bdct4x4_ibdct_hp":
    from script1_models.multiface_partition_bdct4x4_ibdct_hp import DeepAppearanceVAE_IBDCT as overallModel
    sweep_parameter_list = [2, 4, 6, 8, 10, 12, 14]

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

camera_config = camera_config["full"]

dataset_train = Dataset(
    data_dir,
    krt_dir,
    framelist_test,
    tex_size,
    camset=None if camera_config is None else camera_config["train"],
    exclude_prefix=test_segment,
)

dataset_test = Dataset(
    data_dir,
    krt_dir,
    framelist_test,
    tex_size,
    camset=None if camera_config is None else camera_config["test"],
    valid_prefix=test_segment,
)

n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
mesh_inp_size = 21918
overall_list = []
for parameter_value in sweep_parameter_list:
    average_texture_path = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
    prefix_path_captured_latent_code = "/tmp/"
    path_variance_matrix_tensor = "/tmp/"
    save_latent_code_to_external_device = "/tmp/"
    apply_gaussian_noise = False

    if model_arch == "merge_conv_16chnl":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=parameter_value, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise)

    if model_arch == "merge_conv":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=parameter_value, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise)

    if model_arch == "baseline":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams)

    if model_arch == "hp_bdct4x4":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, frequency_threshold=parameter_value, average_texture_path=average_texture_path, prefix_path_captured_latent_code=prefix_path_captured_latent_code, path_variance_matrix_tensor=path_variance_matrix_tensor, save_latent_code_to_external_device = save_latent_code_to_external_device, apply_gaussian_noise = apply_gaussian_noise)

    if model_arch == "multiface_partition_bdct4x4_nn_decoder":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, result_path="/tmp", save_latent_code=False)

    if model_arch == "multiface_partition_bdct4x4_nn_decoder_hp":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, num_freq_comp_outsourced=parameter_value, result_path="/tmp", save_latent_code=False)

    if model_arch == "multiface_partition_bdct4x4_ibdct":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, result_path="/tmp", save_latent_code=False, gaussian_noise_covariance_path=None)

    if model_arch == "multiface_partition_bdct4x4_ibdct_hp":
        model = overallModel(tex_size, mesh_inp_size, n_latent=nlatent, n_cams=n_cams, num_freq_comp_outsourced=parameter_value, result_path="/tmp", save_latent_code=False, gaussian_noise_covariance_path=None)

    unit = Unit()
    system = System(
        unit,
        frequency=940,
        flops=123,
        onchip_mem_bw=900,
        pe_min_density_support=0.0001,
        accelerator_type="structured",
        model_on_chip_mem_implications=False,
        on_chip_mem_size=32,
    )

    inputs = (
        torch.randn([1, 3, 1024, 1024]),
        torch.randn([1, 7306, 3]),
        torch.randn([1, 3]),
        torch.randn([1]).to(torch.long),
    )

    # macs = profile_macs(model, inputs)
    # print('transformer: {:.4g} G'.format(macs / 1e9))
    op_df = profiler(model, inputs, system, unit)
    op_df.to_csv(f"multiface_partition_bdct4x4_ibdct_hp_{parameter_value}.csv")
    flops_local_decoder = op_df.loc[87:116, "Flops (MFLOP)"].sum() + op_df.loc[141:155, "Flops (MFLOP)"].sum()
    
    flops_outsourced_decoder = op_df.loc[117:140, "Flops (MFLOP)"].sum()
    memory_transfer_MB = op_df.loc[140, "Output (MB)"]
    print(f"FLOPS of local decoder = {flops_local_decoder}, FLOPS of outsourced decoder = {flops_outsourced_decoder}, transfer data size = {memory_transfer_MB} MB")