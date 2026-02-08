import os

# !!! Note: this script requires training to be done (so that it could use the post-training model weights)
# Please select one path out of the following three paths based on the server

# Please specify following configurations
#######
path_to_privatar = "/work"
wandb_author_name = "jimmytong"
bitwidth_list = [8, 9, 10, 11, 12, 13, 14, 15, 16]
val_batch_size = 10
data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
val_num = 4 # control total number of data samples to be computed for validation.
for bitwidth in bitwidth_list:
    post_training_model_path = f"{path_to_privatar}/training_results/quant_{bitwidth}/best_model.pth" 
    project_name = f"test_quant_{bitwidth}"
    result_path = f"{path_to_privatar}/testing_results/{project_name}"
    #######

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num} --bitwidth {bitwidth} --use_lpips')
    os.system(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num} --bitwidth {bitwidth} --use_lpips')
