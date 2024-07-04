import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

pretrain_model_path_prefix_uwing2 = "/workspace/uwing2/Privatar/training_results/horizontal_partition_"

result_path_prefix_uwing2 = "/workspace/uwing2/Privatar/training_results"

prefix_path_captured_latent_code_uwing2 = "/workspace/uwing2/Privatar/testing_results/horizontal_partition_"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_uwing2
result_path_prefix = result_path_prefix_uwing2
pretrain_model_path_prefix = pretrain_model_path_prefix_uwing2
prefix_path_captured_latent_code = prefix_path_captured_latent_code_uwing2
#######

average_texture_path = path_prefix + average_texture_path

BDCT_threshold = [0.35]
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 
# BDCT_threshold = BDCT_threshold[::-1]
# BDCT_threshold = BDCT_threshold[2:] # 5, 4, 3 ..

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{result_path_prefix}/noisy_horizontal_partition_{str(BDCT_thres)}"
    path_variance_matrix_tensor = f"/workspace/uwing2/Privatar/profiled_latent_code/statistics/noise_variance_matrix_horizontal_partition_{str(BDCT_thres)}_mutual_bound_1.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{pretrain_model_path_prefix}{str(BDCT_thres)}/best_model.pth" --arch base --frequency_threshold {BDCT_thres} --project_name noisy_hp_training_no_upsample --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
    os.system(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{pretrain_model_path_prefix}{str(BDCT_thres)}/best_model.pth" --arch base --frequency_threshold {str(BDCT_thres)} --project_name noisy_hp_training_no_upsample --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
