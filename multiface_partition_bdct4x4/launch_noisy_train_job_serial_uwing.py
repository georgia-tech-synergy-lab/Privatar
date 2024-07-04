import os

# Please select one path out of the following three paths based on the server
path_prefix = "/workspace/uwing2/multiface/"
result_path_prefix = "/workspace/uwing2/Privatar/training_results"
prefix_path_captured_latent_code = "/workspace/uwing2/Privatar/testing_results/dummy"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

average_texture_path = path_prefix + average_texture_path

BDCT_threshold = [0.6]
mi_list = [0.01, 0.1, 1]

for mi_val in mi_list:
    for BDCT_thres in BDCT_threshold:
        result_path = f"{result_path_prefix}/noisy_bdct_4x4_hp_{str(BDCT_thres)}_mi_{str(mi_val)}"
        path_variance_matrix_tensor = f"/workspace/uwing2/Privatar/profiled_latent_code/statistics/bdct_4x4_noisy_hp_{str(BDCT_thres)}_mutual_bound_{str(mi_val)}.pth"
        best_model_path = f"/workspace/uwing2/Privatar/training_results/bdct_4x4_hp_{str(BDCT_thres)}/best_model.pth"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {best_model_path} --arch base --frequency_threshold {BDCT_thres} --project_name bdct_4x4_noisy_hp_training --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor} --epoch 1')
        os.system(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {best_model_path} --arch base --frequency_threshold {str(BDCT_thres)} --project_name bdct_4x4_noisy_hp_training --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor} --epoch 1')
