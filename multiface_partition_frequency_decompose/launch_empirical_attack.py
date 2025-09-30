import os

# Please select one path out of the following three paths based on the server
using_pac_noise = True # True for using PAC noise, False for using DP noise
path_to_privatar = "/work"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS"

val_batch_size = 1 # This must be one, because attacker does not support batch input limited by the implementations.
mi_list = [4, 3, 1, 0.1, 0.01]
num_freq_comp_offloaded_list = [2, 4, 6, 8, 10, 12, 14]
framelist_test = f"{path_to_privatar}/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
camera_configs_path = f"{path_to_privatar}/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"

for num_freq_comp_offloaded in num_freq_comp_offloaded_list:
    best_model_path = f"{path_to_privatar}/training_results/partition_{num_freq_comp_offloaded}/best_model.pth"
    for v in mi_list:
        project_name = f"empirical_attack_partition_{num_freq_comp_offloaded}_{v}"
        result_path = f"{path_to_privatar}/testing_results/{project_name}"
        if using_pac_noise:
          gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/pac_analysis/noise_covariance/pac_noise_partition_local_decoder_{num_freq_comp_offloaded}_{v}.npy"
        else:
          gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_partition_local_decoder_{num_freq_comp_offloaded}_{v}.npy"

        if not os.path.exists(result_path):
          os.makedirs(result_path)
        print(f"num_freq_comp_offloaded={num_freq_comp_offloaded}  mi={v}")
        os.system(f'python test_empirical_attack_run.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size}  --num_freq_comp_offloaded {num_freq_comp_offloaded} --gaussian_noise_covariance_path {gaussian_noise_covariance_path} --camera_configs_path {camera_configs_path}')
        