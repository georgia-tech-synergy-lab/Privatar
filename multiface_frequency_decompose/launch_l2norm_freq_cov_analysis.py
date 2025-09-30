import os

# Please select one path out of the following three paths based on the server
path_to_privatar = "/work"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 

val_batch_size = 1
num_freq_comp_offloaded = 14
project_name = f"analyse_l2norm_partition_{num_freq_comp_offloaded}"
result_path = f"{path_to_privatar}/testing_results/{project_name}"
framelist_test = f"{path_to_privatar}/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
camera_configs_path = f"{path_to_privatar}/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python analyse_run.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --camera_configs_path {camera_configs_path}')
os.system(f'python analyse_run.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1   --arch base  --project_name {project_name} --author_name {wandb_author_name}  --num_freq_comp_offloaded {num_freq_comp_offloaded} --val_batch_size {val_batch_size} --camera_configs_path {camera_configs_path}')
