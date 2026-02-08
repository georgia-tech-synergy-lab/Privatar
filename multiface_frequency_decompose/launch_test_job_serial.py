import os

# Please specify following configurations
#######
path_to_privatar = "/work"
project_name = f"test_partition_0"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
result_path = f"{path_to_privatar}/testing_results/{project_name}"
post_training_model_path = f"{path_to_privatar}/training_results/partition_0/best_model.pth" 
val_batch_size = 10
val_num = 64 # control total number of data samples to be computed for validation.
#######

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num}  --save_latent_code --use_lpips')
os.system(f'python test.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --val_num {val_num}  --save_latent_code --use_lpips')
