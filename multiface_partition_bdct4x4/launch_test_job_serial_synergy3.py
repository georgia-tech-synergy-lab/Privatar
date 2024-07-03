import os

# Please select one path out of the following three paths based on the server
path_prefix = "/usr/scratch/jianming/multiface/"
path_testing_latent_code = "/usr/scratch/jianming/Privatar/testing_results/bdct_4x4_hp_"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"

BDCT_threshold = [5] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
device_id = 0
port_id = 25689
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_testing_latent_code}{str(BDCT_thres)}"
    best_model_path = f"/usr/scratch/jianming/Privatar/training_results/bdct4x4_horizontal_partition_{str(BDCT_thres)}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'CUDA_VISIBLE_DEVICES={device_id} torchrun --nproc_per_node=1 --master_port={port_id}  test.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --save_latent_code_to_external_device')
    os.system(f'CUDA_VISIBLE_DEVICES={device_id} torchrun --nproc_per_node=1 --master_port={port_id}  test.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --save_latent_code_to_external_device')


