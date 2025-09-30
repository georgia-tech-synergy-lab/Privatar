import os

# Please specify following configurations
#######
path_to_privatar = "/work"
wandb_author_name = "jimmytong"
sparsity_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
val_batch_size = 10
val_num = 4 # control total number of data samples to be computed for validation.
#######

for sparsity in sparsity_list:
    project_name = f'sparse_0_{str(sparsity).split(".")[-1]}'
    result_path = f"{path_to_privatar}/testing_results/{project_name}"
    post_training_model_path = f"{path_to_privatar}/training_results/{project_name}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --unified_pruning_ratio {sparsity} --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num}')
    os.system(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --unified_pruning_ratio {sparsity} --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num}')
