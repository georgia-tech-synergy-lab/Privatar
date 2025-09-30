import os

# !!! Note: this script requires training to be done (so that it could use the post-training model weights)
# Please select one path out of the following three paths based on the server

# Please specify following configurations
#######
path_to_privatar = "/work"
wandb_author_name = "jimmytong"
num_freq_comp_offloaded_list = [2, 4, 6, 8, 10, 12, 14]
val_batch_size = 10
data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
val_num = 64 # control total number of data samples to be computed for validation.
for num_freq_comp_offloaded in num_freq_comp_offloaded_list:
    post_training_model_path = f"{path_to_privatar}/training_results/partition_{num_freq_comp_offloaded}/best_model.pth" 
    project_name = f"test_partition_{num_freq_comp_offloaded}"
    result_path = f"{path_to_privatar}/testing_results/{project_name}"
    #######

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num} --num_freq_comp_offloaded {num_freq_comp_offloaded} --save_latent_code')
    os.system(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num} --num_freq_comp_offloaded {num_freq_comp_offloaded} --save_latent_code')
