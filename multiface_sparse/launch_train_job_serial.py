import os
# Please specify following configurations
#######
path_to_privatar = "/work"
wandb_author_name = "jimmytong"
sparsity_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
initial_model_weights_path = f"{path_to_privatar}/pretrain_model/6795937_best_model.pth"
train_batch_size = 10
val_batch_size = 10
epochs = 2
val_num = 1 # control total number of data samples to be computed for validation.
max_iter = 1 # control total number of training iterations.
#######

for sparsity in sparsity_list:
    project_name = f"sparse_0_{str(sparsity).split(".")[-1]}" 
    result_path = f"{path_to_privatar}/training_results/{project_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f'python3 train.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --unified_pruning_ratio {sparsity} --model_path {initial_model_weights_path} --epochs {epochs} --val_num {val_num} --max_iter {max_iter}')
    os.system(f'python3 train.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --unified_pruning_ratio {sparsity} --model_path {initial_model_weights_path} --epochs {epochs} --val_num {val_num} --max_iter {max_iter}')

