import os

# Please specify following configurations
#######
num_freq_comp_offloaded_list = [2, 4, 6, 8, 10, 12, 14]
path_to_privatar = "/work"

wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS"
initial_model_weights_path = f"{path_to_privatar}/pretrain_model/6795937_best_model.pth"
train_batch_size = 10
val_batch_size = 10
epochs = 2 
val_num = 500 # control total number of data samples to be computed for validation.
max_iter = 100000 # control total number of training iterations.
#######

for num_freq_comp_offloaded in num_freq_comp_offloaded_list:
    project_name = f"partition_{num_freq_comp_offloaded}"
    result_path = f"{path_to_privatar}/training_results/{project_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f'python3 train.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_path {initial_model_weights_path} --num_freq_comp_offloaded {num_freq_comp_offloaded} --epochs {epochs} --val_num {val_num} --max_iter {max_iter}')
    os.system(f'python3 train.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_path {initial_model_weights_path} --num_freq_comp_offloaded {num_freq_comp_offloaded} --epochs {epochs} --val_num {val_num} --max_iter {max_iter}')
