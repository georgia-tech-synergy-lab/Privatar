# Given an input model, counts total number of parameters in the models

## Pls copy and paste the model below 
from dataset import Dataset 
import json 
import sys
import os 
import glob
model_arch = "baseline" 
# model_arch = "multiface_partition_bdct4x4_nn_decoder" 
# model_arch = "multiface_partition_bdct4x4_nn_decoder_hp" 
# model_arch = "multiface_partition_bdct4x4_ibdct"
# model_arch = "multiface_partition_bdct4x4_ibdct_hp"
sweep_parameter_list = [5]

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

tex_size = 1024
nlatent = 256
data_dir = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/"
krt_dir = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT"
framelist_test = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt"
subject_id = 6795937

test_segment = "./test_segment.json"

camera_config_path = "/home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json"

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
print(n_cams)
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

    """
        Memory Analysis
    """
    parameter_list = []
    local_enc_parameter_list = []
    local_dec_parameter_list = []
    outsource_enc_parameter_list = []
    outsource_dec_parameter_list = []

    if model_arch == "baseline":
        for p in model.parameters():
            parameter_list.append(p.numel())

        for p in model.enc.parameters():
            local_enc_parameter_list.append(p.numel())

        for p in model.dec.parameters():
            local_dec_parameter_list.append(p.numel())

    if model_arch == "merge_conv":
        for p in model.parameters():
            parameter_list.append(p.numel())

        for p in model.texture_encoder.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.texture_fc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.mesh_fc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.fc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.tex_latent_code_merge.parameters():
            local_enc_parameter_list.append(p.numel())

        for p in model.dec_texture_decoder.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_z_fc.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_view_fc.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_texture_fc.parameters():
            local_dec_parameter_list.append(p.numel())

    if model_arch == "merge_conv_16chnl":
        for p in model.parameters():
            parameter_list.append(p.numel())

        for p in model.texture_encoder.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.texture_fc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.mesh_fc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.fc.parameters():
            local_enc_parameter_list.append(p.numel())

        for p in model.dec_texture_decoder.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_z_fc.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_view_fc.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_mesh_fc.parameters():
            local_dec_parameter_list.append(p.numel())
        for p in model.dec_texture_fc.parameters():
            local_dec_parameter_list.append(p.numel())

    if model_arch == "hp_bdct4x4":
        for p in model.parameters():
            parameter_list.append(p.numel())
        
        for p in model.enc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.dec.parameters():
            local_dec_parameter_list.append(p.numel())
        if len(model.public_idx) > 0:
            for p in model.enc_outsource.parameters():
                outsource_enc_parameter_list.append(p.numel())
            for p in model.dec_outsource.parameters():
                outsource_dec_parameter_list.append(p.numel())

    
    if model_arch == "multiface_partition_bdct4x4_nn_decoder":
        for p in model.parameters():
            parameter_list.append(p.numel())
        
        for p in model.enc.parameters():
            local_enc_parameter_list.append(p.numel())
        for p in model.dec.parameters():
            local_dec_parameter_list.append(p.numel())
        outsource_enc_parameter_list.append(0)
        outsource_dec_parameter_list.append(0)

    
    if model_arch == "multiface_partition_bdct4x4_nn_decoder_hp":
        for p in model.parameters():
            parameter_list.append(p.numel())
        
        for p in model.enc.parameters():
            local_enc_parameter_list.append(p.numel())
        texture_decoder_local_param = 0
        for p in model.dec.texture_decoder_local.parameters():
            local_dec_parameter_list.append(p.numel())
            texture_decoder_local_param = texture_decoder_local_param + p.numel()
        print(f"texture_decoder_local_param={texture_decoder_local_param}")
        
        texture_decoder_merge_param = 0
        for p in model.dec.texture_decoder_merge.parameters():
            local_dec_parameter_list.append(p.numel())
            texture_decoder_merge_param = texture_decoder_merge_param + p.numel()
        print(f"texture_decoder_merge_param={texture_decoder_merge_param}")

        view_fc_param = 0
        for p in model.dec.view_fc.parameters():
            local_dec_parameter_list.append(p.numel())
            view_fc_param = view_fc_param + p.numel()
        print(f"view_fc_param={view_fc_param}")
        
        z_fc_param = 0
        for p in model.dec.z_fc.parameters():
            local_dec_parameter_list.append(p.numel())
            z_fc_param = z_fc_param + p.numel()
        print(f"z_fc_param={z_fc_param}")
        
        mesh_fc_param = 0
        for p in model.dec.mesh_fc.parameters():
            local_dec_parameter_list.append(p.numel())
            mesh_fc_param = mesh_fc_param + p.numel()
        print(f"mesh_fc_param={mesh_fc_param}")
        
        texture_fc_param = 0
        for p in model.dec.texture_fc.parameters():
            local_dec_parameter_list.append(p.numel())
            texture_fc_param = texture_fc_param + p.numel()
        print(f"texture_fc_param={texture_fc_param}")

        print(f"")
        for p in model.enc_outsourced.parameters():
            outsource_enc_parameter_list.append(p.numel())
        texture_decoder_outsource_param = 0
        for p in model.dec.texture_decoder_outsource.parameters():
            outsource_dec_parameter_list.append(p.numel())
            texture_decoder_outsource_param = texture_decoder_outsource_param + p.numel()
        print(f"texture_decoder_local_param={texture_decoder_local_param}")
        print(f"texture_decoder_outsource_param={texture_decoder_outsource_param}")

    print(f"overall parameters = {sum(parameter_list)}")
    print(f"local encoder parameters size = {sum(local_enc_parameter_list)}")
    print(f"local decoder parameters size = {sum(local_dec_parameter_list)}")
    print(f"outsourced encoder parameters size = {sum(outsource_enc_parameter_list)}")
    print(f"outsourced decoder parameters size = {sum(outsource_dec_parameter_list)}")
    overall_list.append([sum(parameter_list), sum(local_enc_parameter_list) + sum(local_dec_parameter_list) + sum(outsource_enc_parameter_list) + sum(outsource_dec_parameter_list), sum(local_enc_parameter_list), sum(local_dec_parameter_list), sum(outsource_enc_parameter_list), sum(outsource_dec_parameter_list)])

    """
        Profile Latency
        Overall latency could be profiled from running on a single batch size data.
    """

print(f"overall_list={overall_list}")
