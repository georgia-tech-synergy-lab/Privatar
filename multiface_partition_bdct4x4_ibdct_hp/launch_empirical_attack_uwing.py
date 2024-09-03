import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface"
result_path_prefix = "/workspace/uwing2/Privatar/attacking_results/"

val_batch_size = 1
project_name = f"attack_bdct_hp_ibdct_decoder_0"
best_model_path = "/workspace/uwing2/Privatar/training_results/bdct_hp_ibdct_decoder_0/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"
framelist_test = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
camera_configs_path = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"

mi_list = [0.01, 0.1, 1]
for v in mi_list:
    gaussian_noise_covariance_path = f"/workspace/uwing2/Privatar/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_0_{v}.npy"

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test_attack_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path} --camera_configs_path {camera_configs_path}')
    os.system(f'python test_attack_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path} --camera_configs_path {camera_configs_path}')
