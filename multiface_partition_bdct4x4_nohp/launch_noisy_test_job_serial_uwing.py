import os

# Please select one path out of the following three paths based on the server
path_prefix ="/workspace/uwing2/multiface/"
path_testing_latent_code = "/workspace/uwing2/Privatar/testing_results/dummy"
average_texture_path = "/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"

BDCT_threshold = [0.4] 
# mi_list = [0.01, 0.1]
mi_list = [1]

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
port_id = 25690
for mi_val in mi_list:
    for BDCT_thres in BDCT_threshold:
        pretrain_model_path = f"/workspace/uwing2/Privatar/training_results/bdct_4x4_hp_{BDCT_thres}/best_model.pth"
        result_path = f"{path_testing_latent_code}{str(BDCT_thres)}"
        path_variance_matrix_tensor = f"/usr/scratch/jianming/Privatar/profiled_latent_code/statistics/noise_variance_matrix_horizontal_partition_{BDCT_thres}_mutual_bound_{mi_val}.pth"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'torchrun --nproc_per_node=1 --master_port={port_id}  test_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {pretrain_model_path} --frequency_threshold {BDCT_thres} --project_name noisy_hp_test_{mi_val} --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
        os.system(f'torchrun --nproc_per_node=1 --master_port={port_id}  test_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {pretrain_model_path} --frequency_threshold {BDCT_thres}  --project_name noisy_hp_test_{mi_val} --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
