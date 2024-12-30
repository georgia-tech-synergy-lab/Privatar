import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface/dataset"
result_path_prefix = "/home/jianming/work/Privatar_prj/attacking_results/"

val_batch_size = 1
num_freq_comp_outsourced = 14
project_name = f"attack_bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}"
result_path = f"{result_path_prefix}{project_name}"
framelist_test = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
camera_configs_path = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"


print(f'python analyse_run.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --camera_configs_path {camera_configs_path}')
os.system(f'python analyse_run.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1   --arch base  --project_name {project_name} --author_name {wandb_author_name}  --num_freq_comp_outsourced {num_freq_comp_outsourced} --val_batch_size {val_batch_size} --camera_configs_path {camera_configs_path}')
